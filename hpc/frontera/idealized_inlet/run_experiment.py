#!/usr/bin/env python3
"""
HPC Wrapper Script for Idealized Inlet DA Experiments on TACC Frontera.

This script provides a command-line interface for running data assimilation
experiments with the idealized inlet problem. It includes:
- Timer instrumentation with PETSc logging
- Memory usage monitoring
- Checkpoint save/restart capability
- MPI-aware output handling

Usage:
    ibrun python run_experiment.py --nx 100 --ny 100 --method 4dvar
    ibrun python run_experiment.py --help

Author: SWE4DVar HPC Infrastructure
"""

import argparse
import json
import os
import sys
import time
import resource
import signal
import pickle
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

# Add project source to path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "experiments" / "serial_da"))

# Import swe4dvar components
from swe4dvar.forward.problems import IdealizedInlet
from swe4dvar.forward.solvers import get_solver
from swe4dvar.data_assimilation import (
    FourDVarCost,
    DiagonalCovariance,
    PointObservationOperator,
)
from swe4dvar.optimization.lbfgs import LBFGSOptimizer
from swe4dvar.utils import get_default_solver_params
from swe4dvar.utils.output_paths import ensure_output_dirs
from swe4dvar.utils.parallel_ops import ParallelTimer, ParallelIO

# Import experiment utilities
from da_experiment_utils import (
    DAExperimentConfig,
    DAExperimentResults,
    ForwardModelWrapper,
    generate_observation_points,
    generate_observations,
    generate_background_state,
    compute_rms_error,
    compute_innovation_statistics,
)


#==============================================================================
# Data Classes
#==============================================================================

@dataclass
class HPCExperimentConfig:
    """Configuration for HPC experiment runs."""
    # Problem configuration
    nx: int = 100
    ny: int = 100
    dt: float = 1200.0
    final_time: float = 345600.0  # 4 days

    # Solver configuration
    solver_type: str = "DG"
    theta: float = 1.0
    friction_law: str = "quadratic"
    dramp: float = 2.0

    # DA configuration
    method: str = "4dvar"  # 4dvar or dcwme
    obs_fraction: float = 0.5
    obs_interval: int = 18  # Every N timesteps (18 * 1200s = 6 hours)
    obs_noise_level: float = 0.01
    background_error_std: float = 0.1

    # Optimization configuration
    max_iterations: int = 50
    gradient_tolerance: float = 1e-6
    cost_tolerance: float = 1e-8
    lbfgs_memory: int = 10

    # Output configuration
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 10  # Save checkpoint every N iterations

    # Profiling
    profile: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HPCTimingResults:
    """Timing results from HPC experiment."""
    total_time: float = 0.0
    setup_time: float = 0.0
    forward_model_time: float = 0.0
    observation_setup_time: float = 0.0
    optimization_time: float = 0.0
    evaluation_time: float = 0.0

    # Per-iteration breakdown
    forward_solve_times: List[float] = field(default_factory=list)
    adjoint_solve_times: List[float] = field(default_factory=list)
    gradient_times: List[float] = field(default_factory=list)

    # Parallel efficiency
    n_ranks: int = 1
    speedup: float = 1.0
    efficiency: float = 1.0

    # Memory stats (in MB)
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0


#==============================================================================
# Memory Monitoring
#==============================================================================

class MemoryMonitor:
    """Monitor memory usage during experiment."""

    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.samples = []

    def sample(self):
        """Take a memory sample."""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        mem_mb = usage.ru_maxrss / 1024  # Convert to MB (Linux)
        if sys.platform == 'darwin':
            mem_mb = usage.ru_maxrss / (1024 * 1024)  # macOS reports bytes
        self.samples.append(mem_mb)

    def get_peak(self) -> float:
        """Get peak memory usage across all ranks."""
        local_peak = max(self.samples) if self.samples else 0.0
        global_peak = self.comm.allreduce(local_peak, op=MPI.MAX)
        return global_peak

    def get_average(self) -> float:
        """Get average memory usage."""
        if not self.samples:
            return 0.0
        local_avg = sum(self.samples) / len(self.samples)
        global_avg = self.comm.allreduce(local_avg, op=MPI.SUM) / self.comm.Get_size()
        return global_avg


#==============================================================================
# Checkpoint Management
#==============================================================================

class CheckpointManager:
    """Manage checkpoints for restart capability."""

    def __init__(self, checkpoint_dir: str, comm: MPI.Comm):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.comm = comm
        self.rank = comm.Get_rank()
        self.io = ParallelIO(comm)

        if self.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

    def save_checkpoint(
        self,
        iteration: int,
        state: PETSc.Vec,
        cost_history: List[float],
        gradient_history: List[float],
        config: Dict,
    ):
        """Save checkpoint for restart."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{iteration:04d}"

        # Save state vector (all ranks)
        self.io.save_vec(state, str(checkpoint_path) + "_state.bin")

        # Save metadata (rank 0 only)
        if self.rank == 0:
            metadata = {
                "iteration": iteration,
                "cost_history": cost_history,
                "gradient_history": gradient_history,
                "config": config,
                "timestamp": datetime.now().isoformat(),
            }
            with open(str(checkpoint_path) + "_meta.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"  Checkpoint saved: iteration {iteration}")

        self.comm.Barrier()

    def load_checkpoint(self, iteration: int = None) -> Optional[Dict]:
        """Load checkpoint for restart.

        Args:
            iteration: Specific iteration to load, or None for latest.

        Returns:
            Dictionary with state vector and metadata, or None if not found.
        """
        # Find latest checkpoint if iteration not specified
        if iteration is None:
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_*_meta.json"))
            if not checkpoints:
                return None
            latest = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
            iteration = int(latest.stem.split("_")[1])

        checkpoint_path = self.checkpoint_dir / f"checkpoint_{iteration:04d}"
        state_file = str(checkpoint_path) + "_state.bin"
        meta_file = str(checkpoint_path) + "_meta.json"

        if not Path(state_file).exists() or not Path(meta_file).exists():
            return None

        # Load state vector
        state = self.io.load_vec(state_file)

        # Load metadata
        with open(meta_file, "r") as f:
            metadata = json.load(f)

        return {
            "state": state,
            "iteration": metadata["iteration"],
            "cost_history": metadata["cost_history"],
            "gradient_history": metadata["gradient_history"],
            "config": metadata["config"],
        }


#==============================================================================
# Main Experiment Runner
#==============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Idealized Inlet DA Experiment on Frontera",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Problem configuration
    parser.add_argument("--nx", type=int, default=100,
                        help="Number of elements in x direction")
    parser.add_argument("--ny", type=int, default=100,
                        help="Number of elements in y direction")
    parser.add_argument("--dt", type=float, default=1200.0,
                        help="Time step in seconds")
    parser.add_argument("--final-time", type=float, default=345600.0,
                        help="Final time in seconds (default: 4 days)")
    parser.add_argument("--solver", type=str, default="DG",
                        choices=["CG", "SUPG", "DG", "DGCG", "DGNC"],
                        help="Solver type")
    parser.add_argument("--friction", type=str, default="quadratic",
                        choices=["linear", "quadratic", "mannings"],
                        help="Friction law")

    # DA configuration
    parser.add_argument("--method", type=str, default="4dvar",
                        choices=["4dvar", "dcwme"],
                        help="Data assimilation method")
    parser.add_argument("--obs-fraction", type=float, default=0.5,
                        help="Fraction of spatial points to observe (0-1)")
    parser.add_argument("--obs-interval", type=int, default=18,
                        help="Observe every N timesteps")
    parser.add_argument("--noise-level", type=float, default=0.01,
                        help="Observation noise level (fraction of signal)")
    parser.add_argument("--background-error", type=float, default=0.1,
                        help="Background error std (fraction of signal)")

    # Optimization configuration
    parser.add_argument("--max-iter", type=int, default=50,
                        help="Maximum L-BFGS iterations")
    parser.add_argument("--gtol", type=float, default=1e-6,
                        help="Gradient tolerance")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                        help="Save checkpoint every N iterations")

    # Restart
    parser.add_argument("--restart", action="store_true",
                        help="Restart from latest checkpoint")
    parser.add_argument("--restart-iter", type=int, default=None,
                        help="Restart from specific iteration")

    # Profiling
    parser.add_argument("--profile", action="store_true",
                        help="Enable detailed timing profile")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")

    return parser.parse_args()


def run_experiment(args):
    """Run the idealized inlet DA experiment."""

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initialize timers
    timer = ParallelTimer(comm)
    timer.start("total")

    # Initialize memory monitor
    memory = MemoryMonitor(comm)
    memory.sample()

    # Create output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        ensure_output_dirs()
    comm.Barrier()

    # Print header
    if rank == 0:
        print("=" * 70)
        print("Idealized Inlet Data Assimilation Experiment")
        print("TACC Frontera HPC")
        print("=" * 70)
        print(f"Timestamp:        {datetime.now().isoformat()}")
        print(f"MPI ranks:        {size}")
        print(f"Method:           {args.method.upper()}")
        print(f"Solver:           {args.solver}")
        print(f"Grid:             {args.nx} x {args.ny}")
        print(f"Time step:        {args.dt} s")
        print(f"Final time:       {args.final_time} s ({args.final_time/86400:.1f} days)")
        print(f"Obs fraction:     {args.obs_fraction}")
        print(f"Obs interval:     {args.obs_interval} timesteps")
        print(f"Output dir:       {output_dir}")
        print("=" * 70)

    # =========================================================================
    # Step 1: Setup Problem and Solver
    # =========================================================================
    if rank == 0:
        print("\nStep 1: Setting up problem and solver...")

    timer.start("setup")

    num_time_steps = int(np.ceil(args.final_time / args.dt))

    # Get path to mesh file
    mesh_file = project_root / "data" / "Ideal_Inlet" / "Ideal_Inlet.xdmf"
    if not mesh_file.exists():
        if rank == 0:
            print(f"ERROR: Mesh file not found: {mesh_file}")
        sys.exit(1)

    problem = IdealizedInlet(
        dt=args.dt,
        nt=num_time_steps,
        xdmf_file=str(mesh_file),
        friction_law=args.friction,
        solution_var="h",
        dramp=2.0,
    )

    solver = get_solver(args.solver)(problem, theta=1.0, p_degree=[1, 1])

    solver_params = get_default_solver_params(
        rtol=1e-9,
        atol=1e-10,
        max_it=15,
        relaxation_parameter=1.0,
        comm=comm,
        error_if_not_converged=True,
    )

    if rank == 0 and args.verbose:
        solver.print_config()

    timer.stop("setup")
    memory.sample()

    if rank == 0:
        print(f"  Problem setup complete. Mesh has {problem.mesh.topology.index_map(0).size_global} vertices")
        print(f"  Number of time steps: {num_time_steps}")

    # =========================================================================
    # Step 2: Generate Truth Trajectory
    # =========================================================================
    if rank == 0:
        print("\nStep 2: Generating truth trajectory...")

    timer.start("forward_model")

    # Use consistent random seed
    np.random.seed(42)

    # Run forward model
    solver.time_loop(
        solver_parameters=solver_params,
        stations=np.array([[25000.5, 15000.5, 0.0]]),
        plot_every=9999,
        save_state=True,
        store_jacobians=True,
        enable_video=False,
        monitor_progress=(rank == 0 and args.verbose),
    )

    # Store truth trajectory
    truth_trajectory = []
    for state_array in solver.storage.saved_states:
        vec = PETSc.Vec().createWithArray(state_array.copy(), comm=comm)
        truth_trajectory.append(vec)

    truth_jacobians = solver.storage.saved_jacobians.copy()
    m_true = truth_trajectory[0].copy()

    timer.stop("forward_model")
    memory.sample()

    if rank == 0:
        print(f"  Truth trajectory: {len(truth_trajectory)} states")
        print(f"  Jacobians stored: {len(truth_jacobians)}")

    # =========================================================================
    # Step 3: Setup Observations
    # =========================================================================
    if rank == 0:
        print("\nStep 3: Setting up observations...")

    timer.start("observation_setup")

    # Generate observation points (50% of mesh points)
    obs_points = generate_observation_points(
        problem.mesh,
        fraction=args.obs_fraction,
        seed=42
    )

    if rank == 0:
        print(f"  Observation points: {len(obs_points)}")

    # Create observation operator
    obs_operator = PointObservationOperator(
        solver.V,
        obs_points,
        comm=comm
    )

    # Determine observation times (every obs_interval timesteps = 6 hours with default)
    obs_times = list(range(args.obs_interval, num_time_steps + 1, args.obs_interval))

    if rank == 0:
        obs_time_hours = args.obs_interval * args.dt / 3600
        print(f"  Observation times: {len(obs_times)} (every {obs_time_hours:.1f} hours)")

    # Generate observations with noise
    observations, obs_noise_stds = generate_observations(
        truth_trajectory,
        obs_operator,
        obs_times,
        noise_level=args.noise_level,
        seed=42
    )

    timer.stop("observation_setup")
    memory.sample()

    if rank == 0:
        print(f"  Observations generated with noise std: {obs_noise_stds.mean():.6f}")

    # =========================================================================
    # Step 4: Setup Background State
    # =========================================================================
    if rank == 0:
        print("\nStep 4: Setting up background state...")

    m_background = generate_background_state(
        m_true,
        error_std=args.background_error,
        seed=123
    )

    background_error = compute_rms_error(m_background, m_true, comm)

    if rank == 0:
        print(f"  Background RMS error: {background_error:.6f}")

    # =========================================================================
    # Step 5: Setup Covariance Matrices
    # =========================================================================
    if rank == 0:
        print("\nStep 5: Setting up covariance matrices...")

    state_size = m_true.getSize()

    # Background covariance
    truth_magnitude = np.abs(m_true.getArray()).mean()
    background_variance = (args.background_error * truth_magnitude) ** 2
    B = DiagonalCovariance(comm, state_size, variance=background_variance)

    # Observation covariance
    n_obs = obs_operator.get_num_observations()
    obs_variance = obs_noise_stds.mean() ** 2
    R = DiagonalCovariance(comm, n_obs, variance=obs_variance)

    if rank == 0:
        print(f"  Background covariance: diagonal, variance = {background_variance:.6e}")
        print(f"  Observation covariance: diagonal, variance = {obs_variance:.6e}")

    # =========================================================================
    # Step 6: Create Forward Model Wrapper and Cost Function
    # =========================================================================
    if rank == 0:
        print("\nStep 6: Creating forward model wrapper and cost function...")

    # Reset solver for optimization
    solver.storage.clear()
    problem.t = 0.0

    forward_model = ForwardModelWrapper(solver, problem, solver_params)

    cost_function = FourDVarCost(
        forward_model=forward_model,
        observation_operator=obs_operator,
        background_cov=B,
        observation_cov=R,
        m_background=m_background,
        observations=observations,
        obs_times=obs_times,
        comm=comm,
    )

    # =========================================================================
    # Step 7: Run Optimization
    # =========================================================================
    if rank == 0:
        print("\nStep 7: Running L-BFGS optimization...")
        print(f"  Max iterations: {args.max_iter}")
        print(f"  Gradient tolerance: {args.gtol}")

    timer.start("optimization")

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(args.checkpoint_dir, comm)

    # Check for restart
    initial_state = m_background.copy()
    initial_iteration = 0

    if args.restart or args.restart_iter is not None:
        checkpoint = checkpoint_mgr.load_checkpoint(args.restart_iter)
        if checkpoint is not None:
            initial_state = checkpoint["state"]
            initial_iteration = checkpoint["iteration"]
            if rank == 0:
                print(f"  Restarting from iteration {initial_iteration}")

    # Create optimizer
    optimizer = LBFGSOptimizer(
        cost_function,
        memory_size=10,
        options={
            "max_iterations": args.max_iter,
            "gradient_tolerance": args.gtol,
            "cost_tolerance": 1e-8,
            "verbose": (rank == 0),
        }
    )

    # Run optimization
    opt_start = time.time()
    m_analysis = optimizer.solve(initial_state)
    opt_time = time.time() - opt_start

    timer.stop("optimization")
    memory.sample()

    if rank == 0:
        print(f"\nOptimization completed in {opt_time:.2f} seconds")
        print(f"  Iterations: {optimizer.iteration}")
        print(f"  Converged: {optimizer.converged}")

    # Save final checkpoint
    checkpoint_mgr.save_checkpoint(
        optimizer.iteration,
        m_analysis,
        optimizer.cost_history,
        optimizer.gradient_history,
        args.__dict__,
    )

    # =========================================================================
    # Step 8: Evaluate Results
    # =========================================================================
    if rank == 0:
        print("\nStep 8: Evaluating results...")

    timer.start("evaluation")

    # Compute analysis error
    analysis_error = compute_rms_error(m_analysis, m_true, comm)
    error_reduction = (background_error - analysis_error) / background_error * 100

    if rank == 0:
        print(f"  Analysis RMS error: {analysis_error:.6f}")
        print(f"  Error reduction: {error_reduction:.1f}%")

    # Run analysis forward for innovation statistics
    solver.storage.clear()
    problem.t = 0.0

    m_analysis_array = m_analysis.getArray()
    solver.u_n.x.array[:] = m_analysis_array
    solver.u_n_old.x.array[:] = m_analysis_array
    solver.u.x.array[:] = m_analysis_array

    solver.time_loop(
        solver_parameters=solver_params,
        stations=np.array([[25000.5, 15000.5, 0.0]]),
        plot_every=9999,
        save_state=True,
        enable_video=False,
    )

    analysis_trajectory = []
    for state_array in solver.storage.saved_states:
        vec = PETSc.Vec().createWithArray(state_array.copy(), comm=comm)
        analysis_trajectory.append(vec)

    innov_mean, innov_std = compute_innovation_statistics(
        analysis_trajectory, obs_operator, observations, obs_times
    )

    timer.stop("evaluation")
    timer.stop("total")
    memory.sample()

    if rank == 0:
        print(f"  Innovation mean: {innov_mean:.6f}")
        print(f"  Innovation std: {innov_std:.6f}")

    # =========================================================================
    # Step 9: Save Results
    # =========================================================================
    if rank == 0:
        print("\nStep 9: Saving results...")

    # Gather timing information
    timing = HPCTimingResults(
        total_time=timer.timers.get("total", {}).get("total", 0.0),
        setup_time=timer.timers.get("setup", {}).get("total", 0.0),
        forward_model_time=timer.timers.get("forward_model", {}).get("total", 0.0),
        observation_setup_time=timer.timers.get("observation_setup", {}).get("total", 0.0),
        optimization_time=timer.timers.get("optimization", {}).get("total", 0.0),
        evaluation_time=timer.timers.get("evaluation", {}).get("total", 0.0),
        n_ranks=size,
        peak_memory_mb=memory.get_peak(),
        avg_memory_mb=memory.get_average(),
    )

    # Create results object
    results = DAExperimentResults(
        method=f"{args.method}_mpi",
        test_case="idealized_inlet",
        cost_history=optimizer.cost_history,
        gradient_norm_history=optimizer.gradient_history,
        background_error=background_error,
        analysis_error=analysis_error,
        error_reduction=error_reduction,
        innovation_mean=innov_mean,
        innovation_std=innov_std,
        num_iterations=optimizer.iteration,
        converged=optimizer.converged,
        wall_time=timing.total_time,
        config={
            **args.__dict__,
            "mpi_ranks": size,
            "timing": asdict(timing),
        },
    )

    if rank == 0:
        # Save results
        results_file = output_dir / f"inlet_{args.method}_results.json"
        results.save(str(results_file))
        print(f"  Results saved to: {results_file}")

    # =========================================================================
    # Summary and Timing Report
    # =========================================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("SUMMARY: Idealized Inlet DA Experiment")
        print("=" * 70)
        print(f"Method:            {args.method.upper()}")
        print(f"MPI ranks:         {size}")
        print(f"Background error:  {background_error:.6f}")
        print(f"Analysis error:    {analysis_error:.6f}")
        print(f"Error reduction:   {error_reduction:.1f}%")
        print(f"Iterations:        {optimizer.iteration}")
        print(f"Converged:         {optimizer.converged}")
        print(f"Total time:        {timing.total_time:.2f} s")
        print(f"Peak memory:       {timing.peak_memory_mb:.1f} MB")
        print("=" * 70)

    # Print detailed timing report if requested
    if args.profile:
        timer.report(root=0)

    # Cleanup
    for vec in truth_trajectory:
        vec.destroy()
    for vec in analysis_trajectory:
        vec.destroy()
    for vec in observations:
        vec.destroy()
    m_true.destroy()
    m_background.destroy()
    m_analysis.destroy()

    return 0 if optimizer.converged else 1


#==============================================================================
# Entry Point
#==============================================================================

if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_experiment(args))

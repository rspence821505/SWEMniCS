#!/usr/bin/env python3
"""
Tidal case: Parallel DC-WME-4DVar data assimilation experiment.

This script runs a twin experiment for the tidal problem using
the Data-Consistent Weighted Mean Error 4DVar cost function with MPI parallelization.

Run with:
    mpirun -n 4 python tidal_dcwme_mpi.py

DC-WME uses cumulative time-averaged innovation as QoI:
    Q_wme(m) = (1/sqrt(N)) * sum_k R_k^{-1/2} * (H_k(M_{k:0}(m)) - y_k)

Usage:
    mpirun -n 4 python tidal_dcwme_mpi.py [--nx 10] [--ny 5] [--dt 3600]
"""

import argparse
import time
import sys
import json
import numpy as np
from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.data_assimilation import (
    DCWMEFourDVarCost,
    DiagonalCovariance,
    PointObservationOperator,
)
from swe4dvar.optimization.lbfgs import LBFGSOptimizer
from swe4dvar.utils import get_default_solver_params
from swe4dvar.utils.output_paths import FIGURES_DIR, DATA_DIR, ensure_output_dirs
from swe4dvar.utils.parallel_ops import ParallelTimer

# Import utilities from serial experiments
sys.path.insert(0, str(Path(__file__).parent.parent / "serial_da"))
from da_experiment_utils import (
    DAExperimentConfig,
    DAExperimentResults,
    ForwardModelWrapper,
    generate_observation_points,
    generate_observations,
    generate_background_state,
    compute_rms_error,
    compute_innovation_statistics,
    save_experiment_results,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Parallel Tidal DC-WME-4DVar data assimilation experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nx", type=int, default=10, help="Elements in x direction")
    parser.add_argument("--ny", type=int, default=5, help="Elements in y direction")
    parser.add_argument("--dt", type=float, default=3600.0, help="Time step (seconds)")
    parser.add_argument("--final-time", type=float, default=24*3600.0, help="Final time (seconds)")
    parser.add_argument("--obs-fraction", type=float, default=0.5, help="Fraction of points to observe")
    parser.add_argument("--obs-frequency", type=int, default=1, help="Observe every N timesteps")
    parser.add_argument("--noise-level", type=float, default=0.01, help="Observation noise level")
    parser.add_argument("--background-error", type=float, default=0.1, help="Background error std")
    parser.add_argument("--max-iter", type=int, default=50, help="Max L-BFGS iterations")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--profile", action="store_true", help="Enable detailed timing")
    return parser.parse_args()


def main():
    """Run parallel tidal DC-WME-4DVar experiment."""
    args = parse_args()

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initialize timer
    timer = ParallelTimer(comm)
    timer.start("total")

    # Ensure output directories exist (rank 0 only)
    if rank == 0:
        ensure_output_dirs()
    comm.Barrier()

    # Configuration
    config = DAExperimentConfig(
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        final_time=args.final_time,
        solver_type="CG",
        obs_fraction=args.obs_fraction,
        obs_frequency=args.obs_frequency,
        obs_noise_level=args.noise_level,
        background_error_std=args.background_error,
        max_iterations=args.max_iter,
    )

    if rank == 0:
        print("=" * 70)
        print("Parallel Tidal DC-WME-4DVar Data Assimilation Experiment")
        print("=" * 70)
        print(f"MPI ranks: {size}")
        print(f"Grid: {config.nx} x {config.ny}")
        print(f"Time step: {config.dt} s, Final time: {config.final_time} s")
        print(f"Observation fraction: {config.obs_fraction}")
        print(f"Noise level: {config.obs_noise_level}")
        print(f"Background error: {config.background_error_std}")
        print("=" * 70)

    # =========================================================================
    # Step 1: Setup problem and solver
    # =========================================================================
    timer.start("setup")

    num_time_steps = int(np.ceil(config.final_time / config.dt))

    problem = TidalProblem(
        nx=config.nx,
        ny=config.ny,
        dt=config.dt,
        nt=num_time_steps,
        friction_law="mannings",
        solution_var="h",
    )

    solver = get_solver(config.solver_type)(problem, theta=0.5, p_degree=[1, 1])

    # Solver parameters
    solver_params = get_default_solver_params(
        rtol=1e-5,
        atol=1e-6,
        max_it=10,
        relaxation_parameter=1.0,
        comm=comm,
        error_if_not_converged=True,
    )

    if rank == 0 and args.verbose:
        solver.print_config()

    timer.stop("setup")

    # =========================================================================
    # Step 2: Generate truth trajectory
    # =========================================================================
    if rank == 0:
        print("\nStep 1: Generating truth trajectory...")

    timer.start("forward_model")
    start_time = time.time()

    # Use consistent random seed across all ranks
    np.random.seed(42)

    # Run forward model to generate truth
    solver.time_loop(
        solver_parameters=solver_params,
        stations=np.array([[800.5, 1000.5, 0.0]]),
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

    # True initial condition
    m_true = truth_trajectory[0].copy()

    timer.stop("forward_model")

    if rank == 0:
        print(f"  Truth trajectory generated: {len(truth_trajectory)} states")
        print(f"  Jacobians stored: {len(truth_jacobians)}")

    # =========================================================================
    # Step 3: Setup observations
    # =========================================================================
    if rank == 0:
        print("\nStep 2: Setting up observations...")

    timer.start("observation_setup")

    # Generate observation points (use consistent seed)
    obs_points = generate_observation_points(
        problem.mesh,
        fraction=config.obs_fraction,
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

    # Determine observation times
    obs_times = list(range(config.obs_frequency, num_time_steps + 1, config.obs_frequency))
    if rank == 0:
        print(f"  Observation times: {len(obs_times)} (every {config.obs_frequency} timesteps)")

    # Generate observations with noise (use consistent seed)
    observations, obs_noise_stds = generate_observations(
        truth_trajectory,
        obs_operator,
        obs_times,
        noise_level=config.obs_noise_level,
        seed=42
    )

    timer.stop("observation_setup")

    if rank == 0:
        print(f"  Observations generated with noise std: {obs_noise_stds.mean():.6f}")

    # =========================================================================
    # Step 4: Setup background state
    # =========================================================================
    if rank == 0:
        print("\nStep 3: Setting up background state...")

    # Use consistent seed for background perturbation
    m_background = generate_background_state(
        m_true,
        error_std=config.background_error_std,
        seed=123
    )

    background_error = compute_rms_error(m_background, m_true, comm)
    if rank == 0:
        print(f"  Background RMS error: {background_error:.6f}")

    # =========================================================================
    # Step 5: Setup covariance matrices
    # =========================================================================
    if rank == 0:
        print("\nStep 4: Setting up covariance matrices...")

    state_size = m_true.getSize()

    # Background covariance
    truth_magnitude = np.abs(m_true.getArray()).mean()
    background_variance = (config.background_error_std * truth_magnitude) ** 2
    B = DiagonalCovariance(comm, state_size, variance=background_variance)

    if rank == 0:
        print(f"  Background covariance: diagonal, variance = {background_variance:.6e}")

    # Observation covariance
    n_obs = obs_operator.get_num_observations()
    obs_variance = obs_noise_stds.mean() ** 2
    R = DiagonalCovariance(comm, n_obs, variance=obs_variance)

    if rank == 0:
        print(f"  Observation covariance: diagonal, variance = {obs_variance:.6e}")

    # =========================================================================
    # Step 6: Create forward model wrapper
    # =========================================================================
    if rank == 0:
        print("\nStep 5: Creating forward model wrapper...")

    # Reset solver for optimization
    solver.storage.clear()
    problem.t = 0.0

    forward_model = ForwardModelWrapper(solver, problem, solver_params)

    # =========================================================================
    # Step 7: Setup DC-WME-4DVar cost function
    # =========================================================================
    if rank == 0:
        print("\nStep 6: Setting up DC-WME-4DVar cost function...")
        print("  DC-WME uses weighted mean error QoI for improved stability")

    cost_function = DCWMEFourDVarCost(
        forward_model=forward_model,
        observation_operator=obs_operator,
        background_cov=B,
        observation_cov=R,
        m_background=m_background,
        observations=observations,
        obs_times=obs_times,
        predicted_cov_wme=None,  # Will be estimated automatically
        comm=comm,
    )

    # =========================================================================
    # Step 8: Run optimization
    # =========================================================================
    if rank == 0:
        print("\nStep 7: Running L-BFGS optimization...")

    timer.start("optimization")

    optimizer = LBFGSOptimizer(
        cost_function,
        memory_size=config.lbfgs_memory,
        options={
            "max_iterations": config.max_iterations,
            "gradient_tolerance": config.gradient_tolerance,
            "cost_tolerance": config.cost_tolerance,
            "verbose": (rank == 0),
        }
    )

    opt_start = time.time()
    m_analysis = optimizer.solve(m_background.copy())
    opt_time = time.time() - opt_start

    timer.stop("optimization")

    if rank == 0:
        print(f"\nOptimization completed in {opt_time:.2f} seconds")
        print(f"  Iterations: {optimizer.iteration}")
        print(f"  Converged: {optimizer.converged}")

    # =========================================================================
    # Step 9: Evaluate results
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

    # Run analysis forward and compute innovation statistics
    solver.storage.clear()
    problem.t = 0.0

    m_analysis_array = m_analysis.getArray()
    solver.u_n.x.array[:] = m_analysis_array
    solver.u_n_old.x.array[:] = m_analysis_array
    solver.u.x.array[:] = m_analysis_array

    solver.time_loop(
        solver_parameters=solver_params,
        stations=np.array([[800.5, 1000.5, 0.0]]),
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

    if rank == 0:
        print(f"  Innovation mean: {innov_mean:.6f}")
        print(f"  Innovation std: {innov_std:.6f}")

    # =========================================================================
    # Step 10: Save results
    # =========================================================================
    timer.stop("total")

    if rank == 0:
        print("\nStep 9: Saving results...")

    total_time = time.time() - start_time

    results = DAExperimentResults(
        method="dcwme_mpi",
        test_case="tidal",
        cost_history=optimizer.cost_history,
        gradient_norm_history=optimizer.gradient_history,
        background_error=background_error,
        analysis_error=analysis_error,
        error_reduction=error_reduction,
        innovation_mean=innov_mean,
        innovation_std=innov_std,
        num_iterations=optimizer.iteration,
        converged=optimizer.converged,
        wall_time=total_time,
        config=config.to_dict(),
    )

    # Add MPI-specific info
    results.config["mpi_ranks"] = size

    if rank == 0:
        filepath = save_experiment_results(results, str(DATA_DIR))
        print(f"  Results saved to: {filepath}")

    # =========================================================================
    # Summary and Timing Report
    # =========================================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("SUMMARY: Parallel Tidal DC-WME-4DVar Experiment")
        print("=" * 70)
        print(f"MPI ranks:         {size}")
        print(f"Background error:  {background_error:.6f}")
        print(f"Analysis error:    {analysis_error:.6f}")
        print(f"Error reduction:   {error_reduction:.1f}%")
        print(f"Iterations:        {optimizer.iteration}")
        print(f"Converged:         {optimizer.converged}")
        print(f"Total time:        {total_time:.2f} s")
        print("=" * 70)

    # Print detailed timing report
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


if __name__ == "__main__":
    main()

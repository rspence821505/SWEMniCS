#!/usr/bin/env python3
"""
Dam break case: DC-WME-4DVar data assimilation experiment.

This script runs a twin experiment for the dam break problem using
the Data-Consistent Weighted Mean Error 4DVar cost function (DCWMEFourDVarCost).

The dam break problem features:
- Discontinuous initial water height
- Rapid flow dynamics with shock formation
- No friction for analytical comparison

DC-WME is particularly suited for:
- Problems with sparse observations
- Situations where predictability is limited
- Cases requiring robust error estimation

Twin Experiment Setup:
1. Run forward model with true initial condition to generate truth trajectory
2. Sample observations at 50% of spatial points with regular frequency
3. Add 1% Gaussian noise to observations
4. Define background state with 10% error from truth
5. Minimize DC-WME-4DVar cost function using L-BFGS

Usage:
    python dam_break_dcwme.py [--nx 30] [--ny 30] [--dt 0.5] [--final-time 20]
"""

import argparse
import time
import sys
import numpy as np
from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.forward.problems import DamProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar import FrictionLaw
from swe4dvar.data_assimilation import (
    DCWMEFourDVarCost,
    DiagonalCovariance,
    PointObservationOperator,
)
from swe4dvar.optimization.lbfgs import LBFGSOptimizer
from swe4dvar.utils import get_default_solver_params
from swe4dvar.utils.output_paths import FIGURES_DIR, DATA_DIR, ensure_output_dirs

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
        description="Dam break DC-WME-4DVar data assimilation experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nx", type=int, default=30, help="Elements in x direction")
    parser.add_argument("--ny", type=int, default=30, help="Elements in y direction")
    parser.add_argument("--dt", type=float, default=0.5, help="Time step (seconds)")
    parser.add_argument("--final-time", type=float, default=20.0, help="Final time (seconds)")
    parser.add_argument("--obs-fraction", type=float, default=0.5, help="Fraction of points to observe")
    parser.add_argument("--obs-frequency", type=int, default=4, help="Observe every N timesteps")
    parser.add_argument("--noise-level", type=float, default=0.01, help="Observation noise level")
    parser.add_argument("--background-error", type=float, default=0.1, help="Background error std")
    parser.add_argument("--max-iter", type=int, default=50, help="Max L-BFGS iterations")
    parser.add_argument("--solver", type=str, default="DG", choices=["CG", "DG", "SUPG"],
                        help="Solver type")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main():
    """Run dam break DC-WME-4DVar experiment."""
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Ensure output directories exist
    ensure_output_dirs()

    # Configuration
    config = DAExperimentConfig(
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        final_time=args.final_time,
        solver_type=args.solver,
        obs_fraction=args.obs_fraction,
        obs_frequency=args.obs_frequency,
        obs_noise_level=args.noise_level,
        background_error_std=args.background_error,
        max_iterations=args.max_iter,
    )

    if rank == 0:
        print("=" * 70)
        print("Dam Break DC-WME-4DVar Data Assimilation Experiment")
        print("=" * 70)
        print(f"Grid: {config.nx} x {config.ny}")
        print(f"Solver: {config.solver_type}")
        print(f"Time step: {config.dt} s, Final time: {config.final_time} s")
        print(f"Observation fraction: {config.obs_fraction}")
        print(f"Noise level: {config.obs_noise_level}")
        print(f"Background error: {config.background_error_std}")
        print("=" * 70)

    # =========================================================================
    # Step 1: Setup problem and solver
    # =========================================================================
    num_time_steps = int(np.ceil(config.final_time / config.dt))

    problem = DamProblem(
        dt=config.dt,
        nt=num_time_steps,
        nx=config.nx,
        ny=config.ny,
        friction_law=FrictionLaw.none,
        solution_var="h",
        spherical=False,
    )

    solver = get_solver(config.solver_type)(problem, theta=0.5, p_degree=[1, 1])

    # Solver parameters
    solver_params = get_default_solver_params(
        rtol=1e-5,
        atol=1e-6,
        max_it=10,
        relaxation_parameter=1.0,
        ksp_type="gmres",
        pc_type="ilu",
        comm=comm,
        error_if_not_converged=True,
    )

    if rank == 0 and args.verbose:
        solver.print_config()

    # =========================================================================
    # Step 2: Generate truth trajectory
    # =========================================================================
    if rank == 0:
        print("\nStep 1: Generating truth trajectory...")

    start_time = time.time()

    # Time series output stations for dam break
    nx_stations = 10
    stations = np.zeros((nx_stations, 3))
    stations[:, 0] = np.linspace(0, 1000, nx_stations)
    stations[:, 1] = 450

    # Run forward model to generate truth
    solver.time_loop(
        solver_parameters=solver_params,
        stations=stations,
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

    if rank == 0:
        print(f"  Truth trajectory generated: {len(truth_trajectory)} states")
        print(f"  Jacobians stored: {len(truth_jacobians)}")

    # =========================================================================
    # Step 3: Setup observations
    # =========================================================================
    if rank == 0:
        print("\nStep 2: Setting up observations...")

    # Generate observation points
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

    # Generate observations with noise
    observations, obs_noise_stds = generate_observations(
        truth_trajectory,
        obs_operator,
        obs_times,
        noise_level=config.obs_noise_level,
        seed=42
    )

    if rank == 0:
        print(f"  Observations generated with noise std: {obs_noise_stds.mean():.6f}")

    # =========================================================================
    # Step 4: Setup background state
    # =========================================================================
    if rank == 0:
        print("\nStep 3: Setting up background state...")

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

    # Background covariance: diagonal with variance = (background_error_std * magnitude)^2
    truth_magnitude = np.abs(m_true.getArray()).mean()
    background_variance = (config.background_error_std * truth_magnitude) ** 2
    B = DiagonalCovariance(comm, state_size, variance=background_variance)

    if rank == 0:
        print(f"  Background covariance: diagonal, variance = {background_variance:.6e}")

    # Observation covariance: diagonal based on noise level
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

    if rank == 0:
        print(f"\nOptimization completed in {opt_time:.2f} seconds")
        print(f"  Iterations: {optimizer.iteration}")
        print(f"  Converged: {optimizer.converged}")

    # =========================================================================
    # Step 9: Evaluate results
    # =========================================================================
    if rank == 0:
        print("\nStep 8: Evaluating results...")

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
        stations=stations,
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

    if rank == 0:
        print(f"  Innovation mean: {innov_mean:.6f}")
        print(f"  Innovation std: {innov_std:.6f}")

    # =========================================================================
    # Step 10: Save results
    # =========================================================================
    if rank == 0:
        print("\nStep 9: Saving results...")

    total_time = time.time() - start_time

    results = DAExperimentResults(
        method="dcwme",
        test_case="dam_break",
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

    if rank == 0:
        filepath = save_experiment_results(results, str(DATA_DIR))
        print(f"  Results saved to: {filepath}")

    # =========================================================================
    # Summary
    # =========================================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("SUMMARY: Dam Break DC-WME-4DVar Experiment")
        print("=" * 70)
        print(f"Background error:  {background_error:.6f}")
        print(f"Analysis error:    {analysis_error:.6f}")
        print(f"Error reduction:   {error_reduction:.1f}%")
        print(f"Iterations:        {optimizer.iteration}")
        print(f"Converged:         {optimizer.converged}")
        print(f"Total time:        {total_time:.2f} s")
        print("=" * 70)

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

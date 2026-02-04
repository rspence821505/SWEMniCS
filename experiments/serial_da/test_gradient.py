#!/usr/bin/env python3
"""Test gradient computation with finite difference verification."""

import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils import get_default_solver_params

from da_experiment_utils import (
    ForwardModelWrapper,
    generate_observation_points,
    generate_observations,
)
from swe4dvar.data_assimilation import (
    FourDVarCost,
    DiagonalCovariance,
    PointObservationOperator,
)


def test_gradient():
    """Test gradient using finite differences."""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Problem parameters
    nx, ny = 4, 2
    dt = 3600.0
    num_time_steps = 2  # Keep small
    obs_fraction = 0.5
    obs_noise_level = 0.01
    background_error_std = 0.1

    if rank == 0:
        print("=" * 60)
        print("Gradient Verification Test")
        print("=" * 60)
        print(f"Grid: {nx} x {ny}")
        print(f"Time steps: {num_time_steps}")
        print()

    # Create problem
    problem = TidalProblem(
        nx=nx, ny=ny,
        dt=dt,
        nt=num_time_steps,
    )

    solver = get_solver("CG")(problem, theta=0.5, p_degree=[1, 1])
    solver_params = get_default_solver_params()

    if rank == 0:
        print("Step 1: Generating truth trajectory...")

    # Generate truth
    solver.time_loop(
        solver_parameters=solver_params,
        store_jacobians=True,
        save_state=True,
        enable_video=False,
    )

    truth_trajectory = [
        PETSc.Vec().createWithArray(s.copy(), comm=comm)
        for s in solver.storage.saved_states
    ]
    truth_jacobians = solver.storage.saved_jacobians.copy()

    if rank == 0:
        print(f"  Trajectory length: {len(truth_trajectory)}")
        print(f"  Jacobians: {len(truth_jacobians)}")

    # Observation setup
    obs_points = generate_observation_points(problem.mesh, fraction=obs_fraction)
    obs_op = PointObservationOperator(solver.V, obs_points, comm=comm)

    obs_times = list(range(1, len(truth_trajectory)))
    observations, _ = generate_observations(
        truth_trajectory, obs_op, obs_times,
        noise_level=obs_noise_level, seed=42
    )

    if rank == 0:
        print(f"\nStep 2: Created {len(observations)} observations at times {obs_times}")

    # Perturbed background
    m_truth = truth_trajectory[0]
    m_background = m_truth.copy()
    np.random.seed(42)
    perturbation = background_error_std * np.random.randn(m_truth.getSize())
    m_background.setArray(m_background.getArray() + perturbation)

    # Covariances
    state_size = m_truth.getSize()
    n_obs = obs_op.get_num_observations()
    B = DiagonalCovariance(comm, state_size, variance=background_error_std**2)
    R = DiagonalCovariance(comm, n_obs, variance=obs_noise_level**2)

    # Forward model wrapper
    forward_wrapper = ForwardModelWrapper(
        solver, problem, solver_params
    )

    # Cost function
    cost = FourDVarCost(
        forward_model=forward_wrapper,
        observation_operator=obs_op,
        background_cov=B,
        observation_cov=R,
        m_background=m_background,
        observations=observations,
        obs_times=obs_times,
        comm=comm,
    )

    # Test point: use background as test point
    m_test = m_background.copy()

    # Compute analytic gradient
    if rank == 0:
        print("\nStep 3: Computing analytic gradient...")

    J0, grad = cost.value_gradient(m_test)
    grad_norm = grad.norm()

    if rank == 0:
        print(f"  Cost J(m) = {J0:.6e}")
        print(f"  ||grad|| = {grad_norm:.6e}")

    # Finite difference verification
    # Pick a random direction
    np.random.seed(123)
    direction = np.random.randn(m_test.getSize())
    direction /= np.linalg.norm(direction)
    d = PETSc.Vec().createWithArray(direction, comm=comm)

    # Directional derivative: grad^T * d
    analytic_dd = grad.dot(d)

    if rank == 0:
        print(f"\nStep 4: Finite difference test")
        print(f"  Analytic directional derivative: {analytic_dd:.6e}")

    # Finite difference with several step sizes
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    if rank == 0:
        print(f"\n  {'eps':>10}  {'FD deriv':>14}  {'Error':>14}  {'Rel Error':>12}")
        print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*12}")

    best_error = float('inf')
    for eps in epsilons:
        m_plus = m_test.copy()
        m_plus.axpy(eps, d)
        m_minus = m_test.copy()
        m_minus.axpy(-eps, d)

        # Clear cache and evaluate
        cost.clear_cache()
        J_plus = cost.value(m_plus)
        cost.clear_cache()
        J_minus = cost.value(m_minus)

        # Central difference
        fd_dd = (J_plus - J_minus) / (2 * eps)
        error = abs(fd_dd - analytic_dd)
        rel_error = error / (abs(analytic_dd) + 1e-14)

        if rel_error < best_error:
            best_error = rel_error

        if rank == 0:
            print(f"  {eps:10.1e}  {fd_dd:14.6e}  {error:14.6e}  {rel_error:12.2%}")

    if rank == 0:
        print(f"\n  Best relative error: {best_error:.2%}")

        if best_error < 0.01:
            print(f"\n  ✓ GRADIENT PASSED (rel error < 1%)")
        else:
            print(f"\n  ✗ GRADIENT FAILED (rel error > 1%)")

    return best_error < 0.01


if __name__ == "__main__":
    success = test_gradient()
    exit(0 if success else 1)

#!/usr/bin/env python3
"""Check gradient component by component."""

import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

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


def component_gradient_check():
    """Check gradient component by component."""
    comm = MPI.COMM_WORLD

    # Small problem
    nx, ny = 4, 2
    dt = 3600.0
    num_time_steps = 2
    obs_fraction = 0.5
    obs_noise_level = 0.01
    background_error_std = 0.1

    print("=" * 70)
    print("COMPONENT-WISE GRADIENT CHECK")
    print("=" * 70)

    # Create problem and solver
    problem = TidalProblem(nx=nx, ny=ny, dt=dt, nt=num_time_steps)
    # theta=1.0 gives pure BDF2 (theta=0.5 is blended scheme)
    solver = get_solver("CG")(problem, theta=1.0, p_degree=[1, 1])
    solver_params = get_default_solver_params()

    # Generate truth trajectory
    print("\nGenerating truth...")
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

    # Setup observations
    obs_points = generate_observation_points(problem.mesh, fraction=obs_fraction)
    obs_op = PointObservationOperator(solver.V, obs_points, comm=comm)
    obs_times = list(range(1, len(truth_trajectory)))
    observations, _ = generate_observations(
        truth_trajectory, obs_op, obs_times,
        noise_level=obs_noise_level, seed=42
    )

    # Create perturbed background
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
    forward_wrapper = ForwardModelWrapper(solver, problem, solver_params)

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

    # Test point
    m_test = m_background.copy()

    # Compute analytic gradient
    print("\nComputing analytic gradient...")
    J0, grad = cost.value_gradient(m_test)
    grad_array = grad.getArray()
    print(f"J(m) = {J0:.6e}")
    print(f"||grad|| = {np.linalg.norm(grad_array):.6e}")

    # Check gradient components
    n_check = min(45, state_size)  # All 45 DOFs
    eps = 1e-5

    print(f"\nComponent-wise check (eps={eps}):")
    print(f"{'i':>4}  {'analytic':>14}  {'FD':>14}  {'ratio':>10}")
    print(f"{'-'*4}  {'-'*14}  {'-'*14}  {'-'*10}")

    ratios = []
    for i in range(n_check):
        # Perturb component i
        m_plus = m_test.copy()
        m_minus = m_test.copy()

        arr_plus = m_plus.getArray().copy()
        arr_minus = m_minus.getArray().copy()
        arr_plus[i] += eps
        arr_minus[i] -= eps
        m_plus.setArray(arr_plus)
        m_minus.setArray(arr_minus)

        cost.clear_cache()
        J_plus = cost.value(m_plus)
        cost.clear_cache()
        J_minus = cost.value(m_minus)

        fd_grad_i = (J_plus - J_minus) / (2 * eps)
        analytic_i = grad_array[i]

        if abs(analytic_i) > 1e-10:
            ratio = fd_grad_i / analytic_i
        else:
            ratio = float('nan')

        ratios.append(ratio)
        print(f"{i:4d}  {analytic_i:14.6e}  {fd_grad_i:14.6e}  {ratio:10.4f}")

    # Summary
    valid_ratios = [r for r in ratios if not np.isnan(r) and abs(r) < 1000]
    if valid_ratios:
        mean_ratio = np.mean(valid_ratios)
        std_ratio = np.std(valid_ratios)
        print(f"\nMean ratio: {mean_ratio:.4f} (std: {std_ratio:.4f})")
        print(f"Expected ratio: 1.0000")
        if abs(mean_ratio - 1.0) < 0.05:
            print("✓ Gradient looks correct!")
        else:
            print(f"✗ Gradient is off by factor of ~{mean_ratio:.2f}")
    else:
        print("\nNo valid ratios to compute mean")


if __name__ == "__main__":
    component_gradient_check()

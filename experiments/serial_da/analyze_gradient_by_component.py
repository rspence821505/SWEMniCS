#!/usr/bin/env python3
"""Analyze gradient by physical component (h vs momentum)."""

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


def analyze_gradient():
    """Analyze gradient by physical component."""
    comm = MPI.COMM_WORLD

    nx, ny = 4, 2
    dt = 3600.0
    num_time_steps = 2

    problem = TidalProblem(nx=nx, ny=ny, dt=dt, nt=num_time_steps)
    solver = get_solver("CG")(problem, theta=1.0, p_degree=[1, 1])
    solver_params = get_default_solver_params()

    # Get DOF sets
    V = solver.V
    sub0 = V.sub(0)
    _, h_to_parent = sub0.collapse()
    h_dofs = set(h_to_parent)
    mom_dofs = set(range(45)) - h_dofs

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

    # Setup
    obs_points = generate_observation_points(problem.mesh, fraction=0.5)
    obs_op = PointObservationOperator(solver.V, obs_points, comm=comm)
    obs_times = list(range(1, len(truth_trajectory)))
    observations, _ = generate_observations(
        truth_trajectory, obs_op, obs_times,
        noise_level=0.01, seed=42
    )

    m_background = truth_trajectory[0].copy()
    np.random.seed(42)
    perturbation = 0.1 * np.random.randn(truth_trajectory[0].getSize())
    m_background.setArray(m_background.getArray() + perturbation)

    B = DiagonalCovariance(comm, truth_trajectory[0].getSize(), variance=0.1**2)
    R = DiagonalCovariance(comm, obs_op.get_num_observations(), variance=0.01**2)

    forward_wrapper = ForwardModelWrapper(solver, problem, solver_params)
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
    J0, grad = cost.value_gradient(m_test)
    grad_array = grad.getArray()

    # Compute FD gradient for all DOFs
    eps = 1e-5
    fd_grad = np.zeros(45)

    for i in range(45):
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

        fd_grad[i] = (J_plus - J_minus) / (2 * eps)

    # Analyze by component
    print("=" * 70)
    print("GRADIENT ANALYSIS BY PHYSICAL COMPONENT")
    print("=" * 70)

    h_list = sorted(h_dofs)
    mom_list = sorted(mom_dofs)

    # h DOFs
    h_ratios = []
    print("\nh DOFs (water depth):")
    print(f"{'DOF':>4}  {'Analytic':>12}  {'FD':>12}  {'Ratio':>8}")
    for i in h_list:
        ratio = fd_grad[i] / grad_array[i] if abs(grad_array[i]) > 1e-10 else np.nan
        h_ratios.append(ratio)
        print(f"{i:4d}  {grad_array[i]:12.4e}  {fd_grad[i]:12.4e}  {ratio:8.4f}")

    valid_h = [r for r in h_ratios if not np.isnan(r) and abs(r) < 100]
    if valid_h:
        print(f"\n  Mean ratio: {np.mean(valid_h):.4f} (std: {np.std(valid_h):.4f})")

    # Momentum DOFs
    mom_ratios = []
    print("\nMomentum DOFs (uh, vh):")
    print(f"{'DOF':>4}  {'Analytic':>12}  {'FD':>12}  {'Ratio':>8}")
    for i in mom_list[:15]:  # First 15 only
        ratio = fd_grad[i] / grad_array[i] if abs(grad_array[i]) > 1e-10 else np.nan
        mom_ratios.append(ratio)
        print(f"{i:4d}  {grad_array[i]:12.4e}  {fd_grad[i]:12.4e}  {ratio:8.4f}")
    print("  ...")

    for i in mom_list[15:]:
        ratio = fd_grad[i] / grad_array[i] if abs(grad_array[i]) > 1e-10 else np.nan
        mom_ratios.append(ratio)

    valid_mom = [r for r in mom_ratios if not np.isnan(r) and abs(r) < 100]
    if valid_mom:
        print(f"\n  Mean ratio: {np.mean(valid_mom):.4f} (std: {np.std(valid_mom):.4f})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"h DOFs: mean ratio = {np.mean(valid_h):.4f} (expected: 1.0000)")
    print(f"Momentum DOFs: mean ratio = {np.mean(valid_mom):.4f}")
    print(f"Momentum/h scaling factor: {np.mean(valid_mom)/np.mean(valid_h):.2f}x")


if __name__ == "__main__":
    analyze_gradient()

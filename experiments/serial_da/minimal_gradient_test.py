#!/usr/bin/env python3
"""Minimal gradient test to isolate the momentum gradient error."""

import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils import get_default_solver_params
from swe4dvar.adjoint.implicit_adjoint import ImplicitAdjointSolver
from swe4dvar.forward.variational_forms import SWEVariationalForm

from da_experiment_utils import ForwardModelWrapper


def compute_fd_gradient_single_dof(forward_wrapper, m0, cost_fn, dof_idx, eps=1e-5):
    """Compute FD gradient for single DOF."""
    m_plus = m0.copy()
    m_minus = m0.copy()

    arr_plus = m_plus.getArray().copy()
    arr_minus = m_minus.getArray().copy()
    arr_plus[dof_idx] += eps
    arr_minus[dof_idx] -= eps
    m_plus.setArray(arr_plus)
    m_minus.setArray(arr_minus)

    J_plus = cost_fn(m_plus, forward_wrapper)
    J_minus = cost_fn(m_minus, forward_wrapper)

    m_plus.destroy()
    m_minus.destroy()

    return (J_plus - J_minus) / (2 * eps)


def simple_cost(m, forward_wrapper):
    """Simple cost: sum of squared h values at final time."""
    trajectory, _ = forward_wrapper.solve(m, store_jacobians=False)

    h_dofs = set([0, 1, 2, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42])

    final_state = trajectory[-1].getArray()
    h_values = [final_state[i] for i in sorted(h_dofs)]

    # Cleanup
    for v in trajectory:
        v.destroy()

    return 0.5 * np.sum(np.array(h_values)**2)


def minimal_gradient_test():
    """Minimal test for gradient accuracy."""
    comm = MPI.COMM_WORLD

    nx, ny = 4, 2
    dt = 3600.0
    nt = 2

    problem = TidalProblem(nx=nx, ny=ny, dt=dt, nt=nt)
    solver = get_solver("CG")(problem, theta=1.0, p_degree=[1, 1], verbose=False)
    solver_params = get_default_solver_params()

    V = solver.V
    n_dofs = 45

    # Get DOF sets for h, ux, uy
    sub0 = V.sub(0)
    _, h_to_parent = sub0.collapse()
    h_dofs = set(h_to_parent)
    mom_dofs = set(range(n_dofs)) - h_dofs

    # Get ux and uy DOF indices separately (for full flux transform)
    try:
        _, ux_to_parent = V.sub(1).sub(0).collapse()
        ux_dofs = set(ux_to_parent)
        _, uy_to_parent = V.sub(1).sub(1).collapse()
        uy_dofs = set(uy_to_parent)
    except Exception as e:
        print(f"Warning: Could not get ux/uy DOF indices: {e}")
        ux_dofs = None
        uy_dofs = None

    forward_wrapper = ForwardModelWrapper(solver, problem, solver_params)

    # Initial condition
    m0 = PETSc.Vec().createWithArray(solver.u.x.array.copy(), comm=comm)

    print("=" * 70)
    print("MINIMAL GRADIENT TEST")
    print("=" * 70)
    print(f"h DOFs: {sorted(h_dofs)[:5]}...")
    print(f"Mom DOFs: {sorted(mom_dofs)[:5]}...")
    if ux_dofs is not None:
        print(f"ux DOFs: {sorted(ux_dofs)[:5]}...")
        print(f"uy DOFs: {sorted(uy_dofs)[:5]}...")

    # Test FD gradient for a few h DOFs and mom DOFs
    eps = 1e-5

    print(f"\nFD Gradient (eps={eps}):")
    print("-" * 50)

    # Pick specific DOFs to test
    test_h_dofs = sorted(h_dofs)[:3]
    test_mom_dofs = sorted(mom_dofs)[:3]

    for i in test_h_dofs:
        fd_grad = compute_fd_gradient_single_dof(forward_wrapper, m0, simple_cost, i, eps)
        print(f"  h DOF {i:2d}: grad = {fd_grad:12.6e}")

    for i in test_mom_dofs:
        fd_grad = compute_fd_gradient_single_dof(forward_wrapper, m0, simple_cost, i, eps)
        print(f"  Mom DOF {i:2d}: grad = {fd_grad:12.6e}")

    # Now compute analytic gradient via adjoint
    print("\n" + "=" * 70)
    print("ADJOINT-BASED GRADIENT")
    print("=" * 70)

    # Run forward with Jacobians
    trajectory, jacobians = forward_wrapper.solve(m0, store_jacobians=True)

    # Terminal condition for adjoint: J^T λ_N = -∂J/∂u_N
    # Since J = 0.5 * ||h_T||^2, we have ∂J/∂u_N = h_T for h DOFs, 0 for mom DOFs
    # The adjoint RHS is -∂J/∂u_N, so terminal_forcing = -h_T
    terminal_forcing = trajectory[-1].duplicate()
    terminal_forcing.zeroEntries()
    final_arr = trajectory[-1].getArray()
    tf_arr = np.zeros(n_dofs)
    for i in h_dofs:
        tf_arr[i] = -final_arr[i]  # -d(0.5*h^2)/dh = -h (sign flip for adjoint RHS)
    terminal_forcing.setArray(tf_arr)

    print(f"\nTerminal forcing (=-∂J/∂u_N):")
    print(f"  ||terminal_h|| = {np.linalg.norm([tf_arr[i] for i in h_dofs]):.6e}")
    print(f"  ||terminal_mom|| = {np.linalg.norm([tf_arr[i] for i in mom_dofs]):.6e}")

    # No observation forcing at interior times for this simple cost
    obs_forcings = [None] * (nt + 1)

    # Run adjoint - pass variational form for proper mass matrix
    var_form = SWEVariationalForm(V, dt)
    adjoint_solver = ImplicitAdjointSolver(
        forward_model=forward_wrapper,
        trajectory=trajectory,
        jacobians=jacobians,
        dt=dt,
        variational_form=var_form,
        bdf2_start_step=3,  # Match forward solver: Euler for steps 1-2, BDF2 for 3+
        flux_formulation=True,  # Account for Q=[h, h*ux, h*uy] in time coupling
        h_dof_indices=h_dofs,  # Provide h DOF indices for flux scaling
        ux_dof_indices=ux_dofs,  # ux DOF indices for full (∂Q/∂u)^T
        uy_dof_indices=uy_dofs,  # uy DOF indices for full (∂Q/∂u)^T
    )

    gradient = adjoint_solver.solve(
        terminal_forcing=terminal_forcing,
        observation_forcings=obs_forcings,
    )

    grad_arr = gradient.getArray()

    print(f"\nAdjoint gradient:")
    print(f"  ||grad_h|| = {np.linalg.norm([grad_arr[i] for i in h_dofs]):.6e}")
    print(f"  ||grad_mom|| = {np.linalg.norm([grad_arr[i] for i in mom_dofs]):.6e}")

    # Compare individual DOFs
    print("\n" + "=" * 70)
    print("COMPARISON (Analytic vs FD)")
    print("=" * 70)
    print(f"{'DOF':>4}  {'Type':>5}  {'Analytic':>12}  {'FD':>12}  {'Ratio':>8}")
    print("-" * 55)

    for i in test_h_dofs:
        fd_grad = compute_fd_gradient_single_dof(forward_wrapper, m0, simple_cost, i, eps)
        ratio = fd_grad / grad_arr[i] if abs(grad_arr[i]) > 1e-14 else np.nan
        print(f"{i:4d}  {'h':>5}  {grad_arr[i]:12.6e}  {fd_grad:12.6e}  {ratio:8.4f}")

    for i in test_mom_dofs:
        fd_grad = compute_fd_gradient_single_dof(forward_wrapper, m0, simple_cost, i, eps)
        ratio = fd_grad / grad_arr[i] if abs(grad_arr[i]) > 1e-14 else np.nan
        print(f"{i:4d}  {'mom':>5}  {grad_arr[i]:12.6e}  {fd_grad:12.6e}  {ratio:8.4f}")

    # Clean up
    for v in trajectory:
        v.destroy()
    for J in jacobians:
        J.destroy()
    terminal_forcing.destroy()
    gradient.destroy()
    m0.destroy()


if __name__ == "__main__":
    minimal_gradient_test()

#!/usr/bin/env python3
"""Verify gradient via forward sensitivity and compare with adjoint."""

import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils import get_default_solver_params
from swe4dvar.forward.variational_forms import SWEVariationalForm

from da_experiment_utils import ForwardModelWrapper


def verify_sensitivity():
    """Verify gradient via multiple methods."""
    comm = MPI.COMM_WORLD

    nx, ny = 4, 2
    dt = 3600.0
    nt = 2

    problem = TidalProblem(nx=nx, ny=ny, dt=dt, nt=nt)
    solver = get_solver("CG")(problem, theta=1.0, p_degree=[1, 1], verbose=False)
    solver_params = get_default_solver_params()

    V = solver.V
    n_dofs = 45

    # Get DOF sets
    sub0 = V.sub(0)
    _, h_to_parent = sub0.collapse()
    h_dofs = set(h_to_parent)
    mom_dofs = set(range(n_dofs)) - h_dofs

    forward_wrapper = ForwardModelWrapper(solver, problem, solver_params)
    m0 = PETSc.Vec().createWithArray(solver.u.x.array.copy(), comm=comm)
    trajectory, jacobians = forward_wrapper.solve(m0, store_jacobians=True)

    # Get mass matrix
    var_form = SWEVariationalForm(V, dt)
    M = var_form.assemble_mass_matrix()

    J1, J2 = jacobians[0], jacobians[1]

    print("=" * 70)
    print("VERIFY GRADIENT VIA FORWARD SENSITIVITY")
    print("=" * 70)

    # For J = 0.5 * ||h_T||^2, the gradient w.r.t. initial condition is:
    # grad = (du_T/du_0)^T * ∂J/∂u_T
    #
    # For backward Euler:
    # du_1/du_0 = -J_1^{-1} * ∂R_1/∂u_0 = (1/dt) * J_1^{-1} * M
    # du_2/du_1 = -J_2^{-1} * ∂R_2/∂u_1 = (1/dt) * J_2^{-1} * M
    # du_2/du_0 = du_2/du_1 * du_1/du_0 = (1/dt)^2 * J_2^{-1} * M * J_1^{-1} * M
    #
    # Gradient (forward sensitivity):
    # grad = (1/dt)^2 * M^T * J_1^{-T} * M^T * J_2^{-T} * ∂J/∂u_T
    #
    # Gradient (adjoint):
    # λ_2 = J_2^{-T} * (-∂J/∂u_T)
    # λ_1 = J_1^{-T} * (M/dt * λ_2)
    # grad = -M/dt * λ_1

    # ∂J/∂u_T
    final_arr = trajectory[-1].getArray()
    dJ_duT = PETSc.Vec().createWithArray(np.zeros(n_dofs), comm=comm)
    dJ_arr = np.zeros(n_dofs)
    for i in h_dofs:
        dJ_arr[i] = final_arr[i]  # d(0.5*h^2)/dh = h
    dJ_duT.setArray(dJ_arr)
    print(f"||∂J/∂u_T|| = {dJ_duT.norm():.6e}")

    # Method 1: Full forward sensitivity (explicit)
    print("\n--- Method 1: Forward Sensitivity ---")

    # Step 1: J_2^{-T} * ∂J/∂u_T
    ksp = PETSc.KSP().create(J2.getComm())
    ksp.setOperators(J2)
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.getPC().setType(PETSc.PC.Type.NONE)
    ksp.setTolerances(rtol=1e-10, atol=1e-12)

    v1 = dJ_duT.duplicate()
    ksp.solveTranspose(dJ_duT, v1)
    print(f"  J_2^{{-T}} * ∂J/∂u_T: ||v1|| = {v1.norm():.6e}")

    # Step 2: M^T * v1
    v2 = v1.duplicate()
    M.multTranspose(v1, v2)  # M is symmetric, so M^T = M
    print(f"  M^T * v1: ||v2|| = {v2.norm():.6e}")

    # Step 3: J_1^{-T} * v2
    ksp.setOperators(J1)
    v3 = v2.duplicate()
    ksp.solveTranspose(v2, v3)
    print(f"  J_1^{{-T}} * M^T * v1: ||v3|| = {v3.norm():.6e}")

    # Step 4: M^T * v3
    v4 = v3.duplicate()
    M.multTranspose(v3, v4)
    print(f"  M^T * J_1^{{-T}} * M^T * v1: ||v4|| = {v4.norm():.6e}")

    # Step 5: scale by (1/dt)^2
    grad_fwd = v4.copy()
    grad_fwd.scale(1.0 / (dt * dt))
    print(f"  (1/dt)^2 * v4: ||grad_fwd|| = {grad_fwd.norm():.6e}")

    grad_fwd_arr = grad_fwd.getArray()

    # Method 2: Adjoint computation
    print("\n--- Method 2: Adjoint ---")

    # λ_2 = J_2^{-T} * (-∂J/∂u_T)
    neg_dJ = dJ_duT.copy()
    neg_dJ.scale(-1.0)
    lambda_2 = neg_dJ.duplicate()
    ksp.setOperators(J2)
    ksp.solveTranspose(neg_dJ, lambda_2)
    print(f"  λ_2 = J_2^{{-T}} * (-∂J/∂u_T): ||λ_2|| = {lambda_2.norm():.6e}")

    # RHS_1 = (M/dt) * λ_2
    rhs1 = lambda_2.duplicate()
    M.mult(lambda_2, rhs1)
    rhs1.scale(1.0 / dt)
    print(f"  RHS_1 = (M/dt) * λ_2: ||RHS_1|| = {rhs1.norm():.6e}")

    # λ_1 = J_1^{-T} * RHS_1
    lambda_1 = rhs1.duplicate()
    ksp.setOperators(J1)
    ksp.solveTranspose(rhs1, lambda_1)
    print(f"  λ_1 = J_1^{{-T}} * RHS_1: ||λ_1|| = {lambda_1.norm():.6e}")

    # grad = (-M/dt) * λ_1
    grad_adj = lambda_1.duplicate()
    M.mult(lambda_1, grad_adj)
    grad_adj.scale(-1.0 / dt)
    print(f"  grad = (-M/dt) * λ_1: ||grad_adj|| = {grad_adj.norm():.6e}")

    grad_adj_arr = grad_adj.getArray()

    # Method 3: FD
    print("\n--- Method 3: Finite Difference ---")

    def simple_cost(m):
        traj, _ = forward_wrapper.solve(m, store_jacobians=False)
        final = traj[-1].getArray()
        h_vals = [final[i] for i in h_dofs]
        for v in traj:
            v.destroy()
        return 0.5 * np.sum(np.array(h_vals)**2)

    eps = 1e-5
    fd_grad = np.zeros(n_dofs)
    for i in range(n_dofs):
        m_plus = m0.copy()
        m_minus = m0.copy()
        arr_plus = m_plus.getArray().copy()
        arr_minus = m_minus.getArray().copy()
        arr_plus[i] += eps
        arr_minus[i] -= eps
        m_plus.setArray(arr_plus)
        m_minus.setArray(arr_minus)

        fd_grad[i] = (simple_cost(m_plus) - simple_cost(m_minus)) / (2 * eps)
        m_plus.destroy()
        m_minus.destroy()

    print(f"  ||FD grad|| = {np.linalg.norm(fd_grad):.6e}")

    # Compare all three
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'DOF':>4}  {'Type':>5}  {'Fwd Sens':>12}  {'Adjoint':>12}  {'FD':>12}  {'Adj/FD':>8}")
    print("-" * 70)

    for i in sorted(h_dofs)[:5]:
        ratio = fd_grad[i] / grad_adj_arr[i] if abs(grad_adj_arr[i]) > 1e-14 else np.nan
        print(f"{i:4d}  {'h':>5}  {grad_fwd_arr[i]:12.6e}  {grad_adj_arr[i]:12.6e}  {fd_grad[i]:12.6e}  {ratio:8.4f}")

    for i in sorted(mom_dofs)[:5]:
        ratio = fd_grad[i] / grad_adj_arr[i] if abs(grad_adj_arr[i]) > 1e-14 else np.nan
        print(f"{i:4d}  {'mom':>5}  {grad_fwd_arr[i]:12.6e}  {grad_adj_arr[i]:12.6e}  {fd_grad[i]:12.6e}  {ratio:8.4f}")

    # Check if forward sensitivity matches adjoint
    diff_fwd_adj = grad_fwd.copy()
    diff_fwd_adj.axpy(-1.0, grad_adj)
    print(f"\n||Fwd - Adj|| = {diff_fwd_adj.norm():.6e}")
    print(f"||Fwd - Adj|| / ||FD|| = {diff_fwd_adj.norm() / np.linalg.norm(fd_grad):.6e}")

    # Cleanup
    ksp.destroy()
    for v in trajectory:
        v.destroy()
    for J in jacobians:
        J.destroy()
    m0.destroy()


if __name__ == "__main__":
    verify_sensitivity()

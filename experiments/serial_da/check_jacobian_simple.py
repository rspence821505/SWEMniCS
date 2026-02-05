#!/usr/bin/env python3
"""Simple check of sensitivity: du_1/du_0."""

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


def check_sensitivity():
    """Check sensitivity du_1/du_0 via FD vs chain rule."""
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

    forward_wrapper = ForwardModelWrapper(solver, problem, solver_params)
    m0 = PETSc.Vec().createWithArray(solver.u.x.array.copy(), comm=comm)
    trajectory, jacobians = forward_wrapper.solve(m0, store_jacobians=True)

    # Get mass matrix
    var_form = SWEVariationalForm(V, dt)
    M = var_form.assemble_mass_matrix()

    J1 = jacobians[0]

    print("=" * 70)
    print("CHECK SENSITIVITY du_1/du_0")
    print("=" * 70)

    eps = 1e-6
    i = 0  # Test DOF 0 (h DOF)

    # Perturb initial condition
    m_pert = m0.copy()
    arr = m_pert.getArray().copy()
    arr[i] += eps
    m_pert.setArray(arr)

    # Run forward with perturbed IC
    traj_pert, _ = forward_wrapper.solve(m_pert, store_jacobians=False)

    # FD: (u_1(m0+ε) - u_1(m0)) / ε
    sens_FD = traj_pert[1].copy()
    sens_FD.axpy(-1.0, trajectory[1])
    sens_FD.scale(1.0 / eps)

    # Chain rule: du_1/du_0 * e_i = (1/dt) * J_1^{-1} * M * e_i
    # Create e_i
    e_i = PETSc.Vec().createWithArray(np.zeros(n_dofs), comm=comm)
    e_i_arr = np.zeros(n_dofs)
    e_i_arr[i] = 1.0
    e_i.setArray(e_i_arr)

    # M * e_i
    M_ei = e_i.duplicate()
    M.mult(e_i, M_ei)

    # J_1^{-1} * M * e_i
    ksp = PETSc.KSP().create(J1.getComm())
    ksp.setOperators(J1)
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.getPC().setType(PETSc.PC.Type.NONE)
    ksp.setTolerances(rtol=1e-10, atol=1e-12)

    J_inv_M_ei = M_ei.duplicate()
    ksp.solve(M_ei, J_inv_M_ei)

    # Scale by 1/dt
    sens_chain = J_inv_M_ei.copy()
    sens_chain.scale(1.0 / dt)

    print(f"\nDOF {i} (h DOF):")
    print(f"  ||du_1/du_0 * e_i|| (FD)    = {sens_FD.norm():.6e}")
    print(f"  ||du_1/du_0 * e_i|| (Chain) = {sens_chain.norm():.6e}")
    print(f"  Ratio FD/Chain              = {sens_FD.norm() / sens_chain.norm():.6f}")

    sens_FD_arr = sens_FD.getArray()
    sens_chain_arr = sens_chain.getArray()

    print(f"\n  Component comparison (first 10):")
    print(f"  {'Comp':>4}  {'Type':>4}  {'FD':>12}  {'Chain':>12}  {'Ratio':>8}")
    for j in range(10):
        dof_type = 'h' if j in h_dofs else 'mom'
        ratio = sens_FD_arr[j] / sens_chain_arr[j] if abs(sens_chain_arr[j]) > 1e-14 else np.nan
        print(f"  {j:4d}  {dof_type:>4}  {sens_FD_arr[j]:12.6e}  {sens_chain_arr[j]:12.6e}  {ratio:8.4f}")

    # Check correlation
    corr = np.corrcoef(sens_FD_arr, sens_chain_arr)[0, 1]
    print(f"\n  Overall correlation = {corr:.6f}")

    # Cleanup
    ksp.destroy()
    for v in trajectory:
        v.destroy()
    for v in traj_pert:
        v.destroy()
    for J in jacobians:
        J.destroy()
    m0.destroy()
    m_pert.destroy()


if __name__ == "__main__":
    check_sensitivity()

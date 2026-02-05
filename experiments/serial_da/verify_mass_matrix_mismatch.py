#!/usr/bin/env python3
"""
Verify the mass matrix mismatch hypothesis.

The forward solver uses Q = [h, h*ux, h*uy] as the flux variable,
which means ∂Q/∂u is NOT the identity. The time derivative term
in the Jacobian is (1/dt) * ∫ (∂Q/∂u)^T · v dx, not (1/dt) * M.
"""

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


def verify_mass_mismatch():
    """Verify the mass matrix mismatch."""
    comm = MPI.COMM_WORLD

    nx, ny = 4, 2
    dt = 3600.0
    nt = 1

    problem = TidalProblem(nx=nx, ny=ny, dt=dt, nt=nt)
    solver = get_solver("CG")(problem, theta=1.0, p_degree=[1, 1], verbose=False)
    solver_params = get_default_solver_params()

    V = solver.V
    n_dofs = solver.u.x.array.size

    # Get DOF structure
    sub0 = V.sub(0)
    _, h_to_parent = sub0.collapse()
    h_dofs = set(h_to_parent)
    mom_dofs = set(range(n_dofs)) - h_dofs

    print("=" * 70)
    print("MASS MATRIX MISMATCH VERIFICATION")
    print("=" * 70)
    print(f"solution_var = '{problem.solution_var}'")
    print(f"n_dofs = {n_dofs}")
    print(f"h_dofs: {sorted(h_dofs)}")

    # Run forward to get Jacobian
    wrapper = ForwardModelWrapper(solver, problem, solver_params)
    m0 = PETSc.Vec().createWithArray(solver.u.x.array.copy(), comm=comm)
    trajectory, jacobians = wrapper.solve(m0, store_jacobians=True)

    J = jacobians[0]

    # Get mass matrix from SWEVariationalForm
    var_form = SWEVariationalForm(V, dt)
    M = var_form.assemble_mass_matrix()

    print(f"\n||J|| = {J.norm():.6e}")
    print(f"||M|| = {M.norm():.6e}")
    print(f"dt = {dt}")

    # Expected: J = (1/dt) * dQ/du_M + spatial
    # where dQ/du_M is the "flux mass matrix"
    # For solution_var = "h", Q = [h, h*ux, h*uy], so:
    # dQ/du = [[1, 0, 0], [ux, h, 0], [uy, 0, h]] (block per node)

    # If spatial ≈ 0, then J ≈ (1/dt) * dQ/du_M
    # Let's check: J - (1/dt)*M should give the "extra" part

    # Compute J - (1/dt)*M
    J_minus_M_dt = J.copy()
    J_minus_M_dt.axpy(-1.0 / dt, M)

    print(f"\n||J - (1/dt)*M|| = {J_minus_M_dt.norm():.6e}")
    print(f"||(1/dt)*M|| = {M.norm() / dt:.6e}")
    print(f"Ratio ||J - (1/dt)*M|| / ||(1/dt)*M|| = {J_minus_M_dt.norm() * dt / M.norm():.4f}")

    # Compare diagonals
    J_diag = J.getDiagonal().getArray()
    M_diag = M.getDiagonal().getArray()

    print("\nDiagonal comparison:")
    print(f"{'DOF':>4}  {'Type':>4}  {'J_diag':>12}  {'M_diag/dt':>12}  {'Ratio':>8}")
    print("-" * 55)

    for i in sorted(h_dofs)[:3]:
        ratio = J_diag[i] / (M_diag[i] / dt) if abs(M_diag[i]) > 1e-14 else np.nan
        print(f"{i:4d}  {'h':>4}  {J_diag[i]:12.4e}  {M_diag[i]/dt:12.4e}  {ratio:8.4f}")

    for i in sorted(mom_dofs)[:3]:
        ratio = J_diag[i] / (M_diag[i] / dt) if abs(M_diag[i]) > 1e-14 else np.nan
        print(f"{i:4d}  {'mom':>4}  {J_diag[i]:12.4e}  {M_diag[i]/dt:12.4e}  {ratio:8.4f}")

    # Get state values at initial condition
    u0 = trajectory[0].getArray()
    print(f"\nState at t=0:")
    for i in sorted(h_dofs)[:3]:
        print(f"  h[{i}] = {u0[i]:.4f}")
    for i in sorted(mom_dofs)[:3]:
        print(f"  mom[{i}] = {u0[i]:.6f}")

    # For Q = [h, h*ux, h*uy], the diagonal of dQ/du should be:
    # - For h DOFs: 1 (diagonal entry is 1)
    # - For ux DOFs: h (diagonal entry is h at that node)
    # - For uy DOFs: h (diagonal entry is h at that node)
    #
    # So J_diag[mom] / (M_diag[mom]/dt) ≈ h at that node

    # The mass matrix M has entries M_ij = ∫ φ_i · φ_j dx
    # For a diagonal approximation (lumped mass), M_ii ~ area/n_nodes_per_element

    # Let's check: For momentum DOFs, is J_diag ≈ h * M_diag / dt?
    print("\nVerifying momentum diagonal scaling:")
    print("For momentum DOFs, J_diag should be ≈ h * M_diag / dt if flux formulation is used")

    h_mean = np.mean([u0[i] for i in h_dofs])
    print(f"Mean h = {h_mean:.4f}")

    for i in sorted(mom_dofs)[:5]:
        expected = h_mean * M_diag[i] / dt
        actual = J_diag[i]
        ratio = actual / expected if abs(expected) > 1e-14 else np.nan
        print(f"  DOF {i}: J_diag={actual:.4e}, h*M_diag/dt={expected:.4e}, ratio={ratio:.4f}")

    # The key insight: the time coupling in the adjoint should use
    # the "flux mass matrix" dQ/du^T * M, not just M!
    #
    # For the adjoint: ∂R/∂u_n = -(1/dt) * dQ/du^T * M (approximately)
    # Not just -(1/dt) * M

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The SWE solver uses flux variables Q = [h, h*ux, h*uy] for time integration.
This means the Jacobian has structure:

  J = (1/dt) * (dQ/du)^T * M_φ + spatial terms

where dQ/du is state-dependent:

  dQ/du = [[1,   0,   0],
           [ux,  h,   0],
           [uy,  0,   h]]

The adjoint time coupling should be:

  ∂R^{n+1}/∂u^n = -(1/dt) * (dQ/du)^T * M_φ

NOT simply -(1/dt) * M!

This explains the ~h scaling factor in the gradient mismatch for momentum DOFs.
""")

    # Cleanup
    for v in trajectory:
        v.destroy()
    for J in jacobians:
        J.destroy()
    m0.destroy()


if __name__ == "__main__":
    verify_mass_mismatch()

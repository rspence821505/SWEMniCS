#!/usr/bin/env python3
"""Check Jacobian structure and scaling."""

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


def check_jacobian():
    """Check Jacobian structure."""
    comm = MPI.COMM_WORLD

    nx, ny = 4, 2
    dt = 3600.0
    nt = 2

    problem = TidalProblem(nx=nx, ny=ny, dt=dt, nt=nt)
    solver = get_solver("CG")(problem, theta=1.0, p_degree=[1, 1])
    solver_params = get_default_solver_params()

    V = solver.V

    # Get DOF sets
    sub0 = V.sub(0)
    _, h_to_parent = sub0.collapse()
    h_dofs = set(h_to_parent)
    mom_dofs = set(range(45)) - h_dofs

    print("=" * 60)
    print("JACOBIAN STRUCTURE")
    print("=" * 60)

    # Run forward to get Jacobians
    forward_wrapper = ForwardModelWrapper(solver, problem, solver_params)
    m0 = PETSc.Vec().createWithArray(
        solver.u.x.array.copy(), comm=comm
    )
    _, jacobians = forward_wrapper.solve(m0, store_jacobians=True)

    J = jacobians[0]  # First Jacobian
    print(f"Jacobian size: {J.getSize()}")
    print(f"Jacobian norm: {J.norm():.6e}")

    # Also get mass matrix for comparison
    var_form = SWEVariationalForm(V, dt)
    M = var_form.assemble_mass_matrix()

    # Compare diagonal scaling
    J_diag = J.getDiagonal().getArray()
    M_diag = M.getDiagonal().getArray()

    print("\nDiagonal comparison (J vs M):")
    print("For h DOFs:")
    h_list = sorted(h_dofs)[:5]
    for i in h_list:
        ratio = J_diag[i] / M_diag[i] if M_diag[i] != 0 else float('inf')
        print(f"  DOF {i}: J_diag={J_diag[i]:.6e}, M_diag={M_diag[i]:.6e}, ratio={ratio:.4f}")

    print("\nFor momentum DOFs:")
    mom_list = sorted(mom_dofs)[:5]
    for i in mom_list:
        ratio = J_diag[i] / M_diag[i] if M_diag[i] != 0 else float('inf')
        print(f"  DOF {i}: J_diag={J_diag[i]:.6e}, M_diag={M_diag[i]:.6e}, ratio={ratio:.4f}")

    # Expected ratio for BDF2: 3/(2*dt) = 3/(2*3600) = 0.000417
    expected_ratio = 3.0 / (2.0 * dt)
    print(f"\nExpected ratio for BDF2 time derivative: {expected_ratio:.6f}")

    # Check h-mom coupling in Jacobian
    print("\nJacobian h-mom coupling:")
    for i in h_list[:3]:
        cols, vals = J.getRow(i)
        h_vals = sum(abs(v) for c, v in zip(cols, vals) if c in h_dofs)
        mom_vals = sum(abs(v) for c, v in zip(cols, vals) if c in mom_dofs)
        print(f"  Row {i} (h): ||h-h||={h_vals:.6e}, ||h-mom||={mom_vals:.6e}")

    print("\nJacobian mom-h coupling:")
    for i in mom_list[:3]:
        cols, vals = J.getRow(i)
        h_vals = sum(abs(v) for c, v in zip(cols, vals) if c in h_dofs)
        mom_vals = sum(abs(v) for c, v in zip(cols, vals) if c in mom_dofs)
        print(f"  Row {i} (mom): ||mom-h||={h_vals:.6e}, ||mom-mom||={mom_vals:.6e}")


if __name__ == "__main__":
    check_jacobian()

#!/usr/bin/env python3
"""Check mass matrix structure for mixed space."""

import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.forward.variational_forms import SWEVariationalForm


def check_mass_matrix():
    """Check mass matrix structure."""
    comm = MPI.COMM_WORLD

    nx, ny = 4, 2
    dt = 3600.0
    nt = 2

    problem = TidalProblem(nx=nx, ny=ny, dt=dt, nt=nt)
    solver = get_solver("CG")(problem, theta=1.0, p_degree=[1, 1])

    V = solver.V

    print("=" * 60)
    print("MASS MATRIX STRUCTURE")
    print("=" * 60)

    # Get h DOF indices
    sub0 = V.sub(0)
    _, h_to_parent = sub0.collapse()
    h_dofs = set(h_to_parent)
    mom_dofs = set(range(45)) - h_dofs

    print(f"h DOFs: {sorted(h_dofs)}")
    print(f"Momentum DOFs: {sorted(mom_dofs)}")

    # Create variational form and assemble mass matrix
    var_form = SWEVariationalForm(V, dt)
    M = var_form.assemble_mass_matrix()

    print(f"\nMass matrix size: {M.getSize()}")
    print(f"Mass matrix norm: {M.norm():.6e}")

    # Check diagonal entries
    diag = M.getDiagonal()
    diag_arr = diag.getArray()
    print(f"\nDiagonal entries (first 20): {diag_arr[:20]}")

    # Check block structure
    print("\nChecking h-h block:")
    h_list = sorted(h_dofs)[:5]
    for i in h_list:
        row_start, row_end = M.getOwnershipRange()
        if row_start <= i < row_end:
            cols, vals = M.getRow(i)
            h_vals = [(c, v) for c, v in zip(cols, vals) if c in h_dofs and abs(v) > 1e-14]
            mom_vals = [(c, v) for c, v in zip(cols, vals) if c in mom_dofs and abs(v) > 1e-14]
            print(f"  Row {i} (h): {len(h_vals)} h entries, {len(mom_vals)} mom entries")
            if mom_vals:
                print(f"    h-mom coupling: {mom_vals[:3]}...")

    print("\nChecking mom-mom block:")
    mom_list = sorted(mom_dofs)[:5]
    for i in mom_list:
        row_start, row_end = M.getOwnershipRange()
        if row_start <= i < row_end:
            cols, vals = M.getRow(i)
            h_vals = [(c, v) for c, v in zip(cols, vals) if c in h_dofs and abs(v) > 1e-14]
            mom_vals = [(c, v) for c, v in zip(cols, vals) if c in mom_dofs and abs(v) > 1e-14]
            print(f"  Row {i} (mom): {len(h_vals)} h entries, {len(mom_vals)} mom entries")
            if h_vals:
                print(f"    mom-h coupling: {h_vals[:3]}...")

    # Compare to identity scaling
    # For a "correct" mass matrix, we expect M to be block diagonal
    # M = [M_h   0  ]
    #     [0   M_mom]
    print("\nChecking if mass matrix is block diagonal:")
    is_block_diagonal = True
    for i in range(M.getSize()[0]):
        row_start, row_end = M.getOwnershipRange()
        if row_start <= i < row_end:
            cols, vals = M.getRow(i)
            for c, v in zip(cols, vals):
                if abs(v) > 1e-14:
                    # Check if off-diagonal block
                    if (i in h_dofs) != (c in h_dofs):
                        print(f"  Off-diagonal entry: M[{i},{c}] = {v:.6e}")
                        is_block_diagonal = False
                        break
        if not is_block_diagonal:
            break

    if is_block_diagonal:
        print("  Mass matrix is block diagonal (h-mom decoupled)")
    else:
        print("  Mass matrix has h-mom coupling!")


if __name__ == "__main__":
    check_mass_matrix()

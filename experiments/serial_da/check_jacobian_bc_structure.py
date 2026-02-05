#!/usr/bin/env python3
"""Check Jacobian structure to identify boundary condition effects."""

import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils import get_default_solver_params

from da_experiment_utils import ForwardModelWrapper


def check_jacobian_bc_structure():
    """Check Jacobian structure for BC effects."""
    comm = MPI.COMM_WORLD

    nx, ny = 4, 2
    dt = 3600.0
    nt = 2

    problem = TidalProblem(nx=nx, ny=ny, dt=dt, nt=nt)
    solver = get_solver("CG")(problem, theta=1.0, p_degree=[1, 1])
    solver_params = get_default_solver_params()

    V = solver.V
    n_dofs = 45

    # Get DOF sets
    sub0 = V.sub(0)
    _, h_to_parent = sub0.collapse()
    h_dofs = set(h_to_parent)
    mom_dofs = set(range(n_dofs)) - h_dofs

    print("=" * 70)
    print("JACOBIAN BOUNDARY CONDITION STRUCTURE")
    print("=" * 70)

    # Run forward to get Jacobians
    forward_wrapper = ForwardModelWrapper(solver, problem, solver_params)
    m0 = PETSc.Vec().createWithArray(solver.u.x.array.copy(), comm=comm)
    _, jacobians = forward_wrapper.solve(m0, store_jacobians=True)

    J = jacobians[1]

    # Check for identity/zero rows (indicating BC DOFs)
    print("\nChecking for BC-modified rows (identity or near-zero rows):")
    print("-" * 70)

    bc_dofs = []
    identity_dofs = []

    for i in range(n_dofs):
        cols, vals = J.getRow(i)

        # Check if this is an identity row (1 on diagonal, 0 elsewhere)
        is_identity = False
        is_zero = np.sum(np.abs(vals)) < 1e-10

        if not is_zero:
            diag_idx = np.where(cols == i)[0]
            if len(diag_idx) > 0:
                diag_val = vals[diag_idx[0]]
                off_diag_sum = np.sum(np.abs(vals)) - np.abs(diag_val)
                if np.abs(diag_val - 1.0) < 1e-10 and off_diag_sum < 1e-10:
                    is_identity = True
                    identity_dofs.append(i)
                    bc_dofs.append(i)

        if is_identity:
            dof_type = "h" if i in h_dofs else "mom"
            print(f"  Row {i:2d} ({dof_type}): IDENTITY (BC row)")

    h_bc_dofs = [d for d in identity_dofs if d in h_dofs]
    mom_bc_dofs = [d for d in identity_dofs if d in mom_dofs]

    print(f"\nSummary:")
    print(f"  Total DOFs: {n_dofs}")
    print(f"  h DOFs: {len(h_dofs)} (BC: {len(h_bc_dofs)})")
    print(f"  Mom DOFs: {len(mom_dofs)} (BC: {len(mom_bc_dofs)})")
    print(f"  h BC DOFs: {sorted(h_bc_dofs)}")
    print(f"  Mom BC DOFs: {sorted(mom_bc_dofs)}")

    # Now analyze J^T h-to-mom coupling more carefully
    print("\n" + "=" * 70)
    print("J^T ROW STRUCTURE ANALYSIS (for h-to-mom coupling)")
    print("=" * 70)

    # J^T[mom, h] = J[h, mom]^T
    # We want to see how h forcing propagates to momentum λ

    # For the transpose solve J^T λ = rhs:
    # The h-forcing is in rhs_h, rhs_mom = 0
    # The momentum equations are: J^T[mom, :] λ = 0
    # => J[h, mom]^T λ_h + J[mom, mom]^T λ_mom = 0
    # => λ_mom = -(J[mom,mom]^T)^{-1} (J[h,mom]^T λ_h)

    # The coupling from h to mom in J^T comes from J[h, mom] (transposed)
    # Let's examine J[h, :] rows for the h DOFs (excluding BC h DOFs)

    interior_h_dofs = [d for d in sorted(h_dofs) if d not in h_bc_dofs]
    print(f"\nInterior h DOFs (non-BC): {interior_h_dofs}")

    print("\nJ rows for interior h DOFs (h-to-mom coupling in J -> mom-to-h coupling in J^T):")
    for i in interior_h_dofs[:5]:  # First 5
        cols, vals = J.getRow(i)
        h_entries = [(c, v) for c, v in zip(cols, vals) if c in h_dofs and abs(v) > 1e-10]
        mom_entries = [(c, v) for c, v in zip(cols, vals) if c in mom_dofs and abs(v) > 1e-10]
        print(f"  Row {i} (h): {len(h_entries)} h entries, {len(mom_entries)} mom entries")
        if mom_entries:
            print(f"         h-to-mom values: {[f'{c}:{v:.2e}' for c, v in mom_entries[:5]]}")

    # Check mom rows for interior momentum DOFs
    interior_mom_dofs = [d for d in sorted(mom_dofs) if d not in mom_bc_dofs]
    print(f"\nInterior mom DOFs (non-BC): {interior_mom_dofs[:10]}...")

    print("\nJ rows for interior mom DOFs (mom-to-h coupling in J):")
    for i in interior_mom_dofs[:5]:  # First 5
        cols, vals = J.getRow(i)
        h_entries = [(c, v) for c, v in zip(cols, vals) if c in h_dofs and abs(v) > 1e-10]
        mom_entries = [(c, v) for c, v in zip(cols, vals) if c in mom_dofs and abs(v) > 1e-10]
        print(f"  Row {i} (mom): {len(h_entries)} h entries, {len(mom_entries)} mom entries")
        if h_entries:
            print(f"          mom-to-h values: {[f'{c}:{v:.2e}' for c, v in h_entries[:5]]}")

    # Now let's manually compute what the solution should be
    print("\n" + "=" * 70)
    print("MANUAL BLOCK SOLUTION CHECK")
    print("=" * 70)

    # Extract J^T blocks as dense matrices for analysis
    h_list = sorted(h_dofs)
    mom_list = sorted(mom_dofs)
    n_h = len(h_list)
    n_mom = len(mom_list)

    # J^T_{mm} = J_{mm}^T
    JT_mm = np.zeros((n_mom, n_mom))
    for i_local, i_global in enumerate(mom_list):
        cols, vals = J.getRow(i_global)
        for c, v in zip(cols, vals):
            if c in mom_dofs:
                j_local = mom_list.index(c)
                # J^T[i,j] = J[j,i], so JT_mm[i_local, j_local] = J[j_global, i_global]
                JT_mm[i_local, j_local] = v

    # Actually, J^T is computed by KSP automatically. Let me just extract J blocks
    # J_{hm} block (h rows, mom cols)
    J_hm = np.zeros((n_h, n_mom))
    for i_local, i_global in enumerate(h_list):
        cols, vals = J.getRow(i_global)
        for c, v in zip(cols, vals):
            if c in mom_dofs:
                j_local = mom_list.index(c)
                J_hm[i_local, j_local] = v

    # J_{mm} block (mom rows, mom cols)
    J_mm = np.zeros((n_mom, n_mom))
    for i_local, i_global in enumerate(mom_list):
        cols, vals = J.getRow(i_global)
        for c, v in zip(cols, vals):
            if c in mom_dofs:
                j_local = mom_list.index(c)
                J_mm[i_local, j_local] = v

    print(f"\nJ_{'{hm}'} block: {J_hm.shape}, rank={np.linalg.matrix_rank(J_hm)}")
    print(f"J_{'{mm}'} block: {J_mm.shape}, rank={np.linalg.matrix_rank(J_mm)}")

    # Condition number of J_mm
    cond_mm = np.linalg.cond(J_mm + 1e-12 * np.eye(n_mom))  # Regularize
    print(f"Condition number of J_{'{mm}'}: {cond_mm:.2e}")

    # For J^T λ = [rhs_h; 0]:
    # Second block equation: J_{hm}^T λ_h + J_{mm}^T λ_mom = 0
    # => λ_mom = -J_{mm}^{-T} J_{hm}^T λ_h

    # Create test λ_h (unit values at all interior h DOFs)
    lambda_h_test = np.ones(n_h)
    # Zero out BC h DOFs
    for i_local, i_global in enumerate(h_list):
        if i_global in h_bc_dofs:
            lambda_h_test[i_local] = 0.0

    print(f"\nTest λ_h: {lambda_h_test}")

    # Compute expected λ_mom from block formula
    try:
        # J_{mm}^T λ_mom = -J_{hm}^T λ_h
        rhs_block = -J_hm.T @ lambda_h_test
        # Solve J_mm^T λ_mom = rhs_block
        lambda_mom_expected = np.linalg.lstsq(J_mm.T, rhs_block, rcond=None)[0]

        print(f"\nExpected λ_mom from block formula:")
        print(f"  ||λ_mom|| = {np.linalg.norm(lambda_mom_expected):.6e}")
        print(f"  ||λ_h|| = {np.linalg.norm(lambda_h_test):.6e}")
        print(f"  Ratio |λ_mom|/|λ_h|: {np.linalg.norm(lambda_mom_expected)/np.linalg.norm(lambda_h_test):.4f}")
    except Exception as e:
        print(f"Block formula failed: {e}")

    # Now check what the full system gives
    print("\n" + "=" * 70)
    print("FULL SYSTEM SOLVE")
    print("=" * 70)

    # Construct full RHS: [ones at interior h DOFs, zeros at mom DOFs]
    rhs_full = np.zeros(n_dofs)
    for i in interior_h_dofs:
        rhs_full[i] = 1.0

    rhs_vec = PETSc.Vec().createWithArray(rhs_full.copy(), comm=comm)
    lambda_vec = rhs_vec.duplicate()

    ksp = PETSc.KSP().create(J.getComm())
    ksp.setOperators(J)
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.getPC().setType(PETSc.PC.Type.NONE)
    ksp.setTolerances(rtol=1e-10, atol=1e-12)

    ksp.solveTranspose(rhs_vec, lambda_vec)
    lambda_arr = lambda_vec.getArray()

    lambda_h_actual = np.array([lambda_arr[i] for i in h_list])
    lambda_mom_actual = np.array([lambda_arr[i] for i in mom_list])

    print(f"\nActual solution from full transpose solve:")
    print(f"  ||λ_h|| = {np.linalg.norm(lambda_h_actual):.6e}")
    print(f"  ||λ_mom|| = {np.linalg.norm(lambda_mom_actual):.6e}")
    print(f"  Ratio |λ_mom|/|λ_h|: {np.linalg.norm(lambda_mom_actual)/np.linalg.norm(lambda_h_actual):.4f}")

    # Check residual
    residual = rhs_vec.duplicate()
    J.multTranspose(lambda_vec, residual)
    residual.axpy(-1.0, rhs_vec)
    print(f"  ||J^T λ - rhs|| = {residual.norm():.6e}")


if __name__ == "__main__":
    check_jacobian_bc_structure()

"""parity_test_petsc.py — solve a tiny deterministic linear system.

Uses PETSc's preonly + LU (mumps if available else petsc built-in) so the
answer is determined solely by the matrix factorization (no iterative
solver randomness). Single-rank by default; MPI-safe for N>1 but we use
N=1 for the tight MATCH tolerance.

Usage:
    python parity_test_petsc.py > petsc.json
"""
import json
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.rank

# A well-conditioned SPD matrix (fixed integer Laplacian-like band):
#   A = tridiag(-1, 2, -1), 16x16
# with known exact solution u = 1..16 for RHS b = A @ u.
N = 16
A = PETSc.Mat().create(comm=comm)
A.setSizes([N, N])
A.setType("aij")
A.setUp()
rstart, rend = A.getOwnershipRange()
for i in range(rstart, rend):
    A.setValue(i, i, 2.0)
    if i > 0:     A.setValue(i, i - 1, -1.0)
    if i < N - 1: A.setValue(i, i + 1, -1.0)
A.assemblyBegin(); A.assemblyEnd()

u_true = PETSc.Vec().create(comm)
u_true.setSizes(N); u_true.setUp()
for i in range(rstart, rend):
    u_true.setValue(i, float(i + 1))
u_true.assemble()

b = u_true.duplicate()
A.mult(u_true, b)
b_norm = b.norm(PETSc.NormType.NORM_2)

# Solve A x = b with a direct factorization
u = u_true.duplicate()
ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
ksp.setFromOptions()
ksp.setUp()
ksp.solve(b, u)

residual = b.duplicate()
A.mult(u, residual)
residual.axpy(-1.0, b)          # r = A u - b
res_norm = residual.norm(PETSc.NormType.NORM_2)
rel_res = res_norm / b_norm if b_norm > 0 else res_norm

# error = u - u_true. Note: PETSc Vec.copy(dst) copies SELF -> DST,
# so correct pattern is u.copy(err), not err.copy(u).
err = u.duplicate()
u.copy(err)                       # err <- u
err.axpy(-1.0, u_true)            # err <- err - u_true = u - u_true
err_norm = err.norm(PETSc.NormType.NORM_2)
rel_err = err_norm / u_true.norm(PETSc.NormType.NORM_2)

if rank == 0:
    out = {
        "N": N,
        "b_norm": float(b_norm),
        "residual_abs": float(res_norm),
        "residual_rel": float(rel_res),
        "error_abs": float(err_norm),
        "error_rel": float(rel_err),
        "ksp_converged_reason": int(ksp.getConvergedReason()),
        "ksp_iterations": int(ksp.getIterationNumber()),
        "petsc_version": list(PETSc.Sys.getVersion()),
    }
    print(json.dumps(out, indent=2, sort_keys=True))

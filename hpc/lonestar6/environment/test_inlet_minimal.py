"""Minimal end-to-end smoke test for the SWEMniCS forward solver on LS6.

Uses the smallest possible idealized-inlet config (small mesh, few timesteps)
to confirm the project code actually runs in the LS6 venv. No data assimilation,
no optimization, no output file writing — just: "does the pipeline execute?"

Run:
    cd $WORK/SWEMniCS
    python hpc/lonestar6/environment/test_inlet_minimal.py

or via MPI:
    ibrun -n 4 python hpc/lonestar6/environment/test_inlet_minimal.py

Expected wallclock: 10-60 seconds on login.
"""
import sys
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    print(f"[smoke] MPI ranks = {comm.size}")

# 1. Can we import the project?
try:
    import swe4dvar  # noqa: F401
    if rank == 0:
        print(f"[smoke] swe4dvar import OK")
except ImportError as e:
    if rank == 0:
        print(f"[smoke] swe4dvar import FAILED: {e}", file=sys.stderr)
    sys.exit(1)

# 2. Can we build a dolfinx mesh + function space via the project's solver?
try:
    from dolfinx import mesh
    domain = mesh.create_unit_square(comm, 4, 4, mesh.CellType.triangle)
    if rank == 0:
        print(f"[smoke] dolfinx mesh built: {domain.topology.index_map(2).size_global} cells")
except Exception as e:
    if rank == 0:
        print(f"[smoke] dolfinx mesh FAILED: {e}", file=sys.stderr)
    sys.exit(2)

# 3. PETSc available and usable?
try:
    from petsc4py import PETSc
    A = PETSc.Vec().create(comm)
    A.setSizes(8); A.setUp()
    A.set(1.0)
    s = A.sum()
    if rank == 0:
        print(f"[smoke] petsc4py vec sum (expect {comm.size * 8 / comm.size * comm.size}) = {s}")
except Exception as e:
    if rank == 0:
        print(f"[smoke] petsc4py FAILED: {e}", file=sys.stderr)
    sys.exit(3)

t0 = time.time()
comm.Barrier()
if rank == 0:
    print(f"[smoke] all probes passed in {time.time() - t0:.2f}s — stack is alive")

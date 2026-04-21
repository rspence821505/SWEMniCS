"""Minimal MPI sanity test — verifies mpi4py links to the expected MPI library and
that all ranks see each other through the launcher (ibrun on LS6).

Run single-process:
    python test_mpi.py

Run multi-rank on login (light — OK for 2-4 ranks, a few seconds):
    ibrun -n 4 python test_mpi.py

Run inside sbatch:
    ibrun -n <N> python test_mpi.py
"""
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# One-line banner from every rank (easy to spot silent-single-rank failures).
host = MPI.Get_processor_name()
print(f"[rank {rank:3d}/{size:3d}] host={host}  py={sys.executable}")

# Simple collective to confirm rails actually talk.
local = rank + 1
total = comm.allreduce(local, op=MPI.SUM)
expected = size * (size + 1) // 2
if rank == 0:
    print(f"allreduce sum of ranks+1 = {total}  (expected {expected})  "
          f"{'OK' if total == expected else 'FAIL'}")
    # Also dump MPI library identity — critical for diagnosing ABI mismatches.
    print(f"MPI version: {MPI.Get_version()}")
    print(f"MPI library: {MPI.Get_library_version().strip()}")

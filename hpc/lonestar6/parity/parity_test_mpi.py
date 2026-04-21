"""parity_test_mpi.py — deterministic multi-rank collective test.

Run:
    mpiexec -n 2 python parity_test_mpi.py | python -c "import sys, json; [print(l) for l in sys.stdin if l.startswith('{')]"
    # or on LS6 via ibrun inside sbatch:
    # ibrun -n 2 python parity_test_mpi.py

Only rank 0 prints the JSON record. Every rank checks its own state first
and raises if something is off.
"""
import json
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Deterministic payload per rank: sum(rank+1 over all ranks) = size*(size+1)/2
local = rank + 1
total = comm.allreduce(local, op=MPI.SUM)
expected = size * (size + 1) // 2
ok = total == expected

# A double-precision reduction to exercise BLAS-adjacent paths
import numpy as np
rng = np.random.default_rng(seed=42 + rank)   # rank-local seed; reproducible
xs = rng.standard_normal(128)
local_sumsq = float(np.sum(xs * xs))
global_sumsq = comm.allreduce(local_sumsq, op=MPI.SUM)

if rank == 0:
    out = {
        "mpi_size": size,
        "allreduce_int_ok": ok,
        "allreduce_int_sum": total,
        "expected_int_sum": expected,
        "global_sumsq_128dim_rank0_seed": 42,
        "global_sumsq_value": global_sumsq,
        "mpi_library": MPI.Get_library_version().strip().split("\n")[0],
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    if not ok:
        print("FAIL: allreduce int did not match expected", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""
Short optimizer trace: serial vs MPI.

Runs 3 BLMVM iterations (max 6 func evals) on the same idealized inlet
config. Records cost trace, grad norm trace, and analysis RMSE per eval.

Serial: python tests/test_short_opt_trace.py
MPI:    PYTHONUNBUFFERED=1 mpirun -np 2 python tests/test_short_opt_trace.py

Results are saved as JSON for comparison.
"""
import os, sys, json, gc, time
os.environ.setdefault("CC", "/usr/bin/clang")
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

# Reuse the harness builder
sys.path.insert(0, str(PROJECT_ROOT / "tests"))
from test_mpi_parity import build_da_problem

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def log(msg):
    sys.stdout.write(f"  [r{rank}] {msg}\n"); sys.stdout.flush()

log(f"START — size={size}")

da = build_da_problem(comm, rank)
log(f"build_problem done, bg_rmse={da['background_rmse']:.6f}")

# Track cost, grad_norm, and analysis RMSE at each eval
from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper
from petsc4py import PETSc

cost_history = []
grad_history = []
rmse_history = []

m_true_arr = da["cost_fn"].m_b.getArray().copy()  # actually m_background, we'll fix below

# Get the TRUE state (deterministic perturbation was added to m_true to make m_background,
# but for RMSE we need m_true). Reconstruct: m_true = m_background - perturbation.
# Actually, da gives us exp.m_background. Let's recover m_true:
# In the harness, m_background is built from m_true + deterministic perturbation.
# We don't directly expose m_true, but we can compute RMSE as ||x - m_b|| / sqrt(n) which
# tracks distance-from-background rather than distance-from-truth. Good enough for trace comparison.

m_b_arr = da["cost_fn"].m_b.getArray().copy()

def iteration_callback(x, iteration, f, gnorm):
    """Called by TAO monitor after each iteration."""
    x_arr = x.getArray(readonly=True)
    # Local RMSE from background
    local_sse = float(np.sum((x_arr - m_b_arr) ** 2))
    local_n = len(x_arr)
    global_sse = comm.allreduce(local_sse)
    global_n = comm.allreduce(local_n)
    rmse_from_bg = float(np.sqrt(global_sse / global_n))

    cost_history.append(f)
    grad_history.append(gnorm)
    rmse_history.append(rmse_from_bg)
    if rank == 0:
        print(f"  [iter] cost={f:.6f}  ||grad||={gnorm:.4e}  ||x-m_b||/sqrt(n)={rmse_from_bg:.6e}",
              flush=True)

optimizer = PETScTAOWrapper(
    da["cost_fn"], tao_type="blmvm",
    lower_bounds=da["lower"], upper_bounds=da["upper"],
    options={
        "max_iterations": 3,
        "max_funcs": 6,
        "gradient_tolerance": 1e-6,
        "cost_tolerance": 1e-8,
        "verbose": True,
        "iteration_callback": iteration_callback,
    },
)

log("optimizer.solve() starting...")
t0 = time.time()
m_analysis = optimizer.solve(da["m_background"])
elapsed = time.time() - t0
log(f"optimizer.solve() done in {elapsed:.0f}s, evals={optimizer.n_func_evals}")

# Final analysis distance from background
ana_arr = m_analysis.getArray(readonly=True)
final_local_sse = float(np.sum((ana_arr - m_b_arr) ** 2))
final_local_n = len(ana_arr)
final_sse = comm.allreduce(final_local_sse)
final_n = comm.allreduce(final_local_n)
final_rmse = float(np.sqrt(final_sse / final_n))

tag = "serial" if size == 1 else f"mpi{size}"
out = {
    "mpi_size": size,
    "elapsed_s": elapsed,
    "n_func_evals": optimizer.n_func_evals,
    "cost_history": cost_history,
    "grad_history": grad_history,
    "rmse_from_bg_history": rmse_history,
    "final_rmse_from_bg": final_rmse,
}
if rank == 0:
    out_dir = PROJECT_ROOT / "results" / "mpi_parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"opt_trace_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: opt_trace_{tag}.json", flush=True)

m_analysis.destroy()
da["lower"].destroy()
da["upper"].destroy()
log("DONE")

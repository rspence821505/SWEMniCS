#!/usr/bin/env python3
"""
Per-evaluation instrumented short-trace harness for serial vs MPI BLMVM.

Wraps cost_fn.value_gradient to log each line-search probe (not just accepted
iterations). Saves x_k and grad_k as globally-coord-ordered (h, u, v) arrays so
that per-eval cosine similarity / componentwise movement can be computed
offline across runs.

Usage:
  # Serial baseline:
  python tests/test_short_trace_per_eval.py

  # 2-rank MPI:
  PYTHONUNBUFFERED=1 mpirun -np 2 python tests/test_short_trace_per_eval.py

Environment:
  SWE4DVAR_SHORT_TRACE_TAG       output tag (default "default")
  SWE4DVAR_SHORT_TRACE_MAX_FUNCS max function evaluations (default 6)
  SWE4DVAR_FORCE_MUMPS=1         force MUMPS on both serial and MPI
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CC", "/usr/bin/clang")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

import numpy as np
from mpi4py import MPI

# Reuse the DA problem builder + global-vector saver
from test_mpi_parity import build_da_problem, _save_global_ordered_vector  # noqa: E402


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    tag = os.environ.get("SWE4DVAR_SHORT_TRACE_TAG", "default")
    max_funcs = int(os.environ.get("SWE4DVAR_SHORT_TRACE_MAX_FUNCS", "6"))

    def log(msg):
        sys.stdout.write(f"  [r{rank}] {msg}\n")
        sys.stdout.flush()

    size_tag = "serial" if size == 1 else f"mpi{size}"
    run_tag = f"{tag}_{size_tag}"
    out_dir = PROJECT_ROOT / "results" / "short_trace"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    log(f"START — size={size}, tag={run_tag}, max_funcs={max_funcs}")

    # ------------------------------------------------------------------
    # Build problem
    # ------------------------------------------------------------------
    da = build_da_problem(comm, rank)
    smoother = da["gradient_smoother"]
    cost_fn = da["cost_fn"]
    m_background = da["m_background"]
    lower = da["lower"]
    upper = da["upper"]
    log(f"build_problem done, bg_rmse={da['background_rmse']:.6e}")

    # Snapshot m_background globally (for step-norm / componentwise movement)
    prefix = str(out_dir / f"trace_{run_tag}")
    _save_global_ordered_vector(
        m_background, smoother, f"{prefix}_m_background",
        _LoggerShim(rank),
    )

    # Owned indices for componentwise local RMSE and bound-activation counts
    h_idx = da["h_indices"]
    u_idx = da["u_indices"]
    v_idx = da["v_indices"]
    m_b_arr_full = m_background.getArray().copy()
    lb_arr = lower.getArray().copy()
    ub_arr = upper.getArray().copy()

    def count_active_bounds(x_arr):
        n_low = int(np.sum(x_arr <= lb_arr + 1e-12))
        n_up = int(np.sum(x_arr >= ub_arr - 1e-12))
        return n_low, n_up

    def rmse_from_bg(x_arr, idx):
        diff = x_arr[idx] - m_b_arr_full[idx]
        local_sse = float(np.sum(diff ** 2))
        local_n = len(idx)
        global_sse = comm.allreduce(local_sse)
        global_n = comm.allreduce(local_n)
        return float(np.sqrt(global_sse / max(global_n, 1)))

    # ------------------------------------------------------------------
    # Per-eval instrumentation: wrap value_gradient to capture every probe
    # ------------------------------------------------------------------
    eval_records = []
    original_vg = cost_fn.value_gradient

    def wrapped_value_gradient(m):
        eval_id = len(eval_records) + 1
        if rank == 0:
            print(f"  [WRAP] entry eval {eval_id}", flush=True)
        t0 = time.time()

        # Save x_k globally-ordered BEFORE the solve so we can compute step-norm
        # against x_{k-1} offline.
        _save_global_ordered_vector(
            m, smoother, f"{prefix}_eval{eval_id:02d}_x",
            _LoggerShim(rank),
        )

        f, grad = original_vg(m)
        elapsed = time.time() - t0

        # Save grad_k (post-smoother) globally-ordered
        _save_global_ordered_vector(
            grad, smoother, f"{prefix}_eval{eval_id:02d}_grad",
            _LoggerShim(rank),
        )

        # Local stats — global reduction happens via allreduce
        x_arr = m.getArray(readonly=True).copy()
        g_arr = grad.getArray(readonly=True).copy()

        # Componentwise grad norms (local contribution, then allreduce)
        def comp_stats(arr, idx):
            local_sumsq = float(np.sum(arr[idx] ** 2))
            global_sumsq = comm.allreduce(local_sumsq)
            return float(np.sqrt(global_sumsq))

        grad_h_norm = comp_stats(g_arr, h_idx)
        grad_u_norm = comp_stats(g_arr, u_idx)
        grad_v_norm = comp_stats(g_arr, v_idx)

        rmse_total = rmse_from_bg(x_arr, np.arange(len(x_arr)))
        rmse_h = rmse_from_bg(x_arr, h_idx)
        rmse_u = rmse_from_bg(x_arr, u_idx)
        rmse_v = rmse_from_bg(x_arr, v_idx)

        n_low, n_up = count_active_bounds(x_arr)
        n_low = comm.allreduce(n_low)
        n_up = comm.allreduce(n_up)

        x_local_mm = (float(x_arr.min()), float(x_arr.max()))
        x_min = comm.allreduce(x_local_mm[0], op=MPI.MIN)
        x_max = comm.allreduce(x_local_mm[1], op=MPI.MAX)

        rec = {
            "eval_id": eval_id,
            "cost": float(f),
            "grad_norm_total": float(grad.norm()),
            "grad_norm_h": grad_h_norm,
            "grad_norm_u": grad_u_norm,
            "grad_norm_v": grad_v_norm,
            "rmse_from_bg_total": rmse_total,
            "rmse_from_bg_h": rmse_h,
            "rmse_from_bg_u": rmse_u,
            "rmse_from_bg_v": rmse_v,
            "n_active_lower": n_low,
            "n_active_upper": n_up,
            "x_min": x_min,
            "x_max": x_max,
            "elapsed_s": elapsed,
        }
        eval_records.append(rec)
        if rank == 0:
            print(
                f"  [eval {eval_id:2d}] cost={f:.6e} ||g||={rec['grad_norm_total']:.4e} "
                f"(h={grad_h_norm:.3e}, u={grad_u_norm:.3e}, v={grad_v_norm:.3e}) "
                f"rmse={rmse_total:.4e} active(lo={n_low},up={n_up}) dt={elapsed:.1f}s",
                flush=True,
            )
        return f, grad

    cost_fn.value_gradient = wrapped_value_gradient

    # ------------------------------------------------------------------
    # Iter-level tracking via iter_records (populated offline from
    # per-eval cost trace since TAO's monitor at iter 0 sees an
    # unset gradient in PETSc 3.22+ and triggers a spurious
    # GATOL convergence if any monitor is attached).
    # ------------------------------------------------------------------
    iter_records = []  # kept for schema compatibility; left empty

    # ------------------------------------------------------------------
    # Run optimizer — no monitor attached; rely on value_gradient wrap
    # ------------------------------------------------------------------
    from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper

    optimizer = PETScTAOWrapper(
        cost_fn, tao_type="blmvm",
        lower_bounds=lower, upper_bounds=upper,
        options={
            "max_iterations": max_funcs,
            "max_funcs": max_funcs,
            "gradient_tolerance": 1e-6,
            "cost_tolerance": 1e-8,
            "verbose": False,
            "use_tao_monitor": False,
            # NO iteration_callback: attaching the monitor triggers a
            # spurious iter-0 GATOL convergence in PETSc 3.22 BLMVM.
        },
    )

    # Sanity check: can we call value_gradient directly?
    log("sanity: calling value_gradient directly on m_background...")
    t_sanity = time.time()
    try:
        f_direct, g_direct = cost_fn.value_gradient(m_background.copy())
        log(f"sanity: direct call OK  f={float(f_direct):.4e}  ||g||={float(g_direct.norm()):.4e}  "
            f"dt={time.time()-t_sanity:.1f}s  wrap_fired_evals={len(eval_records)}")
        g_direct.destroy()
    except Exception as e:
        log(f"sanity: DIRECT CALL FAILED: {e}")
        import traceback
        traceback.print_exc()

    log("optimizer.solve() starting...")
    t0 = time.time()
    try:
        m_analysis = optimizer.solve(m_background)
        solve_err = None
    except Exception as e:
        import traceback
        solve_err = str(e) + "\n" + traceback.format_exc()
        m_analysis = m_background.copy()
        if rank == 0:
            print(f"  [WARN] optimizer.solve raised: {e}", flush=True)
    solve_elapsed = time.time() - t0
    log(f"optimizer.solve() done in {solve_elapsed:.0f}s, evals={optimizer.n_func_evals}")

    # Fill in post-hoc ordering info for iter events
    # Each iter fires after BLMVM accepts an iterate. The iter_record[i] was
    # the accepted state whose cost matches eval_records[j].cost for some j.
    # We leave matching to offline analysis since we saved both traces.

    # ------------------------------------------------------------------
    # Save m_analysis and trace JSON
    # ------------------------------------------------------------------
    _save_global_ordered_vector(
        m_analysis, smoother, f"{prefix}_m_analysis",
        _LoggerShim(rank),
    )

    if rank == 0:
        reason = None
        try:
            reason = int(optimizer.converged)
        except Exception:
            pass
        out = {
            "mpi_size": size,
            "tag": run_tag,
            "max_funcs": max_funcs,
            "elapsed_s": solve_elapsed,
            "n_func_evals": optimizer.n_func_evals,
            "n_accepted_iterations": len(iter_records),
            "solve_error": solve_err,
            "evals": eval_records,
            "iterations": iter_records,
            "background_rmse": da["background_rmse"],
            "n_obs": da["n_obs"],
            "converged": optimizer.converged,
            "total_iterations": optimizer.iteration,
        }
        out_file = out_dir / f"trace_{run_tag}.json"
        with open(out_file, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved: {out_file}", flush=True)

    # Cleanup
    m_analysis.destroy()
    lower.destroy()
    upper.destroy()
    comm.Barrier()
    log("DONE")


class _LoggerShim:
    """Minimal logger interface for _save_global_ordered_vector."""

    def __init__(self, rank):
        self.rank = rank

    def log(self, msg):
        if self.rank == 0:
            sys.stdout.write(f"    [save] {msg}\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

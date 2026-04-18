#!/usr/bin/env python3
"""
Manual line-search probe harness — serial vs MPI equivalent of BLMVM iter-0.

Used when PETSc TAO BLMVM refuses to call the objective in this PETSc 3.22
environment. Emulates what TAO's Armijo line search does at iter 0 with the
identity initial inverse Hessian:

  g_0 = ∇J(m_background)
  m_k = project_bounds(m_background - alpha_k * g_0)
  α_0 = 1.0, α_{k+1} = 0.5 α_k   (geometric backtrack)

For each k we record cost, grad norm, per-component grad norm, x snapshot,
grad snapshot, step norm, RMSE — everything the per-eval comparator needs.

Usage:
  SWE4DVAR_FORCE_MUMPS=1 python tests/test_short_trace_manual_probe.py              # serial
  SWE4DVAR_FORCE_MUMPS=1 mpirun -np 2 python tests/test_short_trace_manual_probe.py # MPI
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

from test_mpi_parity import build_da_problem, _save_global_ordered_vector  # noqa: E402


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    tag = os.environ.get("SWE4DVAR_SHORT_TRACE_TAG", "manual")
    max_funcs = int(os.environ.get("SWE4DVAR_SHORT_TRACE_MAX_FUNCS", "6"))
    # alpha0 of 1.0 with raw g0 of magnitude ~850 drives h negative in our
    # test problem (forward Newton fails). Default to step magnitude that
    # scales with ||g0|| so that initial trial step magnitude ≈ alpha0 in L2.
    # Set SWE4DVAR_MANUAL_NORMALIZE=0 to keep raw TAO-style α*g.
    alpha0 = float(os.environ.get("SWE4DVAR_MANUAL_ALPHA0", "0.1"))
    backtrack = float(os.environ.get("SWE4DVAR_MANUAL_BACKTRACK", "0.5"))
    normalize = os.environ.get("SWE4DVAR_MANUAL_NORMALIZE", "1") != "0"
    # Armijo parameter — typical value
    c1 = float(os.environ.get("SWE4DVAR_MANUAL_ARMIJO_C1", "1e-4"))

    size_tag = "serial" if size == 1 else f"mpi{size}"
    run_tag = f"{tag}_{size_tag}"
    out_dir = PROJECT_ROOT / "results" / "short_trace"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    def log(msg):
        sys.stdout.write(f"  [r{rank}] {msg}\n")
        sys.stdout.flush()

    log(f"START manual probe  size={size}  tag={run_tag}  max_funcs={max_funcs}  α0={alpha0}  β={backtrack}")

    # Build problem (same as TAO harness)
    da = build_da_problem(comm, rank)
    smoother = da["gradient_smoother"]
    cost_fn = da["cost_fn"]
    m_background = da["m_background"]
    lower = da["lower"]
    upper = da["upper"]
    log(f"build_problem done  bg_rmse={da['background_rmse']:.6e}")

    prefix = str(out_dir / f"trace_{run_tag}")
    _save_global_ordered_vector(m_background, smoother, f"{prefix}_m_background", _LoggerShim(rank))

    h_idx = da["h_indices"]
    u_idx = da["u_indices"]
    v_idx = da["v_indices"]
    m_b_arr = m_background.getArray().copy()
    lb_arr = lower.getArray().copy()
    ub_arr = upper.getArray().copy()

    def comp_norm(arr, idx):
        local = float(np.sum(arr[idx] ** 2))
        return float(np.sqrt(comm.allreduce(local)))

    def rmse_from_bg(arr, idx):
        diff = arr[idx] - m_b_arr[idx]
        local_sse = float(np.sum(diff ** 2))
        local_n = len(idx)
        gsse = comm.allreduce(local_sse)
        gn = comm.allreduce(local_n)
        return float(np.sqrt(gsse / max(gn, 1)))

    def count_active(arr):
        nlo = int(np.sum(arr <= lb_arr + 1e-12))
        nhi = int(np.sum(arr >= ub_arr - 1e-12))
        return comm.allreduce(nlo), comm.allreduce(nhi)

    # ---- Eval 1: at m_background, compute g_0 ----
    eval_records = []
    t_all = time.time()

    def save_and_record(m_vec, grad_vec, cost_val, eval_id, duration):
        _save_global_ordered_vector(m_vec, smoother, f"{prefix}_eval{eval_id:02d}_x", _LoggerShim(rank))
        _save_global_ordered_vector(grad_vec, smoother, f"{prefix}_eval{eval_id:02d}_grad", _LoggerShim(rank))
        x_arr = m_vec.getArray(readonly=True).copy()
        g_arr = grad_vec.getArray(readonly=True).copy()
        nlo, nhi = count_active(x_arr)
        xmin = comm.allreduce(float(x_arr.min()) if len(x_arr) > 0 else 0.0, op=MPI.MIN)
        xmax = comm.allreduce(float(x_arr.max()) if len(x_arr) > 0 else 0.0, op=MPI.MAX)
        rec = {
            "eval_id": eval_id,
            "cost": float(cost_val),
            "grad_norm_total": float(grad_vec.norm()),
            "grad_norm_h": comp_norm(g_arr, h_idx),
            "grad_norm_u": comp_norm(g_arr, u_idx),
            "grad_norm_v": comp_norm(g_arr, v_idx),
            "rmse_from_bg_total": rmse_from_bg(x_arr, np.arange(len(x_arr))),
            "rmse_from_bg_h": rmse_from_bg(x_arr, h_idx),
            "rmse_from_bg_u": rmse_from_bg(x_arr, u_idx),
            "rmse_from_bg_v": rmse_from_bg(x_arr, v_idx),
            "n_active_lower": nlo,
            "n_active_upper": nhi,
            "x_min": xmin,
            "x_max": xmax,
            "elapsed_s": duration,
        }
        eval_records.append(rec)
        if rank == 0:
            print(
                f"  [eval {eval_id:2d}] cost={cost_val:.6e} ||g||={rec['grad_norm_total']:.4e} "
                f"(h={rec['grad_norm_h']:.3e}, u={rec['grad_norm_u']:.3e}, v={rec['grad_norm_v']:.3e}) "
                f"rmse={rec['rmse_from_bg_total']:.4e} active(lo={nlo},up={nhi}) dt={duration:.1f}s",
                flush=True,
            )
        return rec

    log("eval 1: baseline ∇J at m_background...")
    t0 = time.time()
    f0, g0 = cost_fn.value_gradient(m_background)
    rec1 = save_and_record(m_background, g0, f0, 1, time.time() - t0)

    # Armijo backtracking from eval 2 onward.
    # Armijo: accept m_k = m0 - α*d  if  J(m_k) ≤ J(m0) + c1*α*⟨g0, -d⟩
    # Use d = g0 / ||g0|| (normalized) when SWE4DVAR_MANUAL_NORMALIZE=1.
    # Then α is the L2 step-length directly and is comparable across runs.
    g0_norm = float(g0.norm())
    g0_dot_g0 = float(g0.dot(g0))  # ||g0||²
    if normalize:
        d = g0.copy()
        d.scale(1.0 / max(g0_norm, 1e-30))   # unit direction
        g0_dot_d = float(g0.dot(d))          # = ||g0||
    else:
        d = g0.copy()
        g0_dot_d = g0_dot_g0
    if rank == 0:
        print(f"  [probe] ||g0|| = {g0_norm:.4e}  ||g0||² = {g0_dot_g0:.4e}  "
              f"normalize_d={normalize}  g0·d = {g0_dot_d:.4e}  "
              f"c1·g0·d = {c1*g0_dot_d:.4e}", flush=True)

    accepted = {1}
    backtracks = []  # (eval_id, alpha, cost, armijo_rhs, accepted)
    alpha = alpha0

    for k in range(2, max_funcs + 1):
        # Build trial: m_k = clip(m_background - alpha * d)
        m_k = m_background.duplicate()
        m_k.waxpy(-alpha, d, m_background)   # m_k = -alpha*d + m_background
        m_k_arr = m_k.getArray()
        m_k_arr = np.minimum(ub_arr, np.maximum(lb_arr, m_k_arr))
        m_k.setArray(m_k_arr)
        m_k.assemble()

        log(f"eval {k}: α={alpha:.4e}  probe m_k = clip(m₀ - α d)")
        t0 = time.time()
        try:
            f_k, g_k = cost_fn.value_gradient(m_k)
        except Exception as e:
            log(f"eval {k}: value_gradient failed: {e}")
            # Record a failure marker but continue — compare script handles it
            break
        duration = time.time() - t0
        if not np.isfinite(f_k):
            log(f"eval {k}: non-finite cost (forward model failed); keep probing at smaller α")
            # Record the failure, skip vector save to avoid polluting comparison
            if rank == 0:
                eval_records.append({
                    "eval_id": k,
                    "cost": float("inf"),
                    "grad_norm_total": float("nan"),
                    "grad_norm_h": float("nan"),
                    "grad_norm_u": float("nan"),
                    "grad_norm_v": float("nan"),
                    "rmse_from_bg_total": float("nan"),
                    "rmse_from_bg_h": float("nan"),
                    "rmse_from_bg_u": float("nan"),
                    "rmse_from_bg_v": float("nan"),
                    "n_active_lower": 0,
                    "n_active_upper": 0,
                    "x_min": float("nan"),
                    "x_max": float("nan"),
                    "elapsed_s": duration,
                    "status": "forward_failed",
                    "alpha": alpha,
                })
            else:
                eval_records.append({"eval_id": k, "status": "forward_failed"})
            alpha *= backtrack
            try:
                g_k.destroy()
            except Exception:
                pass
            m_k.destroy()
            continue
        save_and_record(m_k, g_k, f_k, k, duration)
        eval_records[-1]["alpha"] = alpha
        armijo_rhs = f0 - c1 * alpha * g0_dot_d
        accepted_k = bool(f_k <= armijo_rhs)
        backtracks.append({
            "eval_id": k,
            "alpha": alpha,
            "cost": float(f_k),
            "armijo_rhs": float(armijo_rhs),
            "accepted": accepted_k,
        })
        if accepted_k:
            accepted.add(k)
            log(f"eval {k}: ACCEPTED  (cost={f_k:.4e} ≤ rhs={armijo_rhs:.4e})")
            break
        else:
            if rank == 0:
                print(f"    Armijo reject: cost={f_k:.4e} > rhs={armijo_rhs:.4e}", flush=True)
        alpha *= backtrack
        g_k.destroy()
        m_k.destroy()

    total_elapsed = time.time() - t_all
    log(f"manual probe done  n_evals={len(eval_records)}  elapsed={total_elapsed:.0f}s")

    # Save JSON
    if rank == 0:
        out = {
            "mpi_size": size,
            "tag": run_tag,
            "mode": "manual_armijo_probe",
            "alpha0": alpha0,
            "backtrack": backtrack,
            "armijo_c1": c1,
            "max_funcs": max_funcs,
            "n_func_evals": len(eval_records),
            "n_accepted_iterations": sum(1 for b in backtracks if b["accepted"]),
            "evals": eval_records,
            "iterations": [],
            "backtracks": backtracks,
            "background_rmse": da["background_rmse"],
            "n_obs": da["n_obs"],
            "converged": False,
            "elapsed_s": total_elapsed,
            "accepted_eval_ids": sorted(accepted),
        }
        out_file = out_dir / f"trace_{run_tag}.json"
        with open(out_file, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved: {out_file}", flush=True)

    lower.destroy()
    upper.destroy()
    comm.Barrier()
    log("DONE")


class _LoggerShim:
    def __init__(self, rank):
        self.rank = rank

    def log(self, msg):
        if self.rank == 0:
            sys.stdout.write(f"    [save] {msg}\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

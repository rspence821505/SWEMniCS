#!/usr/bin/env python3
"""
Compare serial vs MPI short-trace outputs (per-eval optimization probes).

Reads results/short_trace/trace_{tag}_serial.json and trace_{tag}_mpi{N}.json,
plus per-eval x and grad vectors saved as globally-ordered (h, u, v) npz
files, and emits a per-eval comparison table (cost, grad cosine similarity,
step norm, RMSE, componentwise movement).

Usage:
  python tests/compare_short_trace.py [--tag v1] [--mpi-size 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


RESULTS = Path(__file__).resolve().parents[1] / "results" / "short_trace"


def load_vec(path):
    if not path.exists():
        return None
    d = np.load(path)
    return {"coords": d["coords"], "h": d["h"], "u": d["u"], "v": d["v"]}


def concat(v):
    return np.concatenate([v["h"], v["u"], v["v"]])


def cos_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + 1e-30))


def rel_diff(a, b):
    return float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-30))


def metrics_pair(a, b):
    """Compare two (h,u,v) vectors; return total + per-component metrics."""
    if a is None or b is None:
        return None
    tot_a = concat(a)
    tot_b = concat(b)
    out = {
        "total_norm_s": float(np.linalg.norm(tot_a)),
        "total_norm_m": float(np.linalg.norm(tot_b)),
        "total_cos": cos_sim(tot_a, tot_b),
        "total_rel_diff": rel_diff(tot_a, tot_b),
    }
    for c in ("h", "u", "v"):
        out[f"{c}_norm_s"] = float(np.linalg.norm(a[c]))
        out[f"{c}_norm_m"] = float(np.linalg.norm(b[c]))
        out[f"{c}_cos"] = cos_sim(a[c], b[c])
        out[f"{c}_rel_diff"] = rel_diff(a[c], b[c])
    return out


def step_metrics(x_prev, x_curr):
    """Componentwise step vector metrics."""
    if x_prev is None or x_curr is None:
        return None
    out = {}
    for c in ("h", "u", "v"):
        dx = x_curr[c] - x_prev[c]
        out[f"{c}_step_norm"] = float(np.linalg.norm(dx))
        out[f"{c}_step_max"] = float(np.max(np.abs(dx))) if len(dx) > 0 else 0.0
    dx_tot = concat(x_curr) - concat(x_prev)
    out["total_step_norm"] = float(np.linalg.norm(dx_tot))
    return out


def _manual_accepted(trace):
    """For manual_armijo_probe traces, use `accepted_eval_ids` directly."""
    return set(trace.get("accepted_eval_ids", []))


def infer_accepted(eval_recs, iter_recs):
    """Match each iter monitor to the eval whose cost matches it.

    BLMVM fires the monitor AFTER accepting a step. The accepted state's
    cost equals the eval_record whose value_gradient call produced the
    accepted iterate.

    Returns: list of (iter_index, accepted_eval_id) pairs.
    """
    matches = []
    used = set()
    for ir in iter_recs:
        target = ir["cost"]
        best_j = None
        best_err = float("inf")
        for e in eval_recs:
            if e["eval_id"] in used:
                continue
            err = abs(e["cost"] - target) / max(abs(target), 1e-30)
            if err < best_err:
                best_err = err
                best_j = e["eval_id"]
        if best_j is not None and best_err < 1e-6:
            matches.append((ir["iteration"], best_j))
            used.add(best_j)
        else:
            matches.append((ir["iteration"], None))
    return matches


def load_trace(tag, size_tag):
    path = RESULTS / f"trace_{tag}_{size_tag}.json"
    if not path.exists():
        print(f"  [WARN] missing {path}", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--mpi-size", type=int, default=2)
    ap.add_argument("--json-out", default=None, help="optional JSON summary output path")
    args = ap.parse_args()

    tag = args.tag
    serial = load_trace(tag, "serial")
    mpi = load_trace(tag, f"mpi{args.mpi_size}")
    if serial is None or mpi is None:
        print("Missing trace JSON(s). Run both serial and MPI harnesses first.")
        return 1

    print(f"=" * 80)
    print(f"Short-trace comparison  tag={tag}")
    print(f"=" * 80)
    print(f"  Serial: n_func_evals={serial['n_func_evals']}, "
          f"n_accepted_iters={serial['n_accepted_iterations']}, "
          f"elapsed={serial['elapsed_s']:.0f}s, converged={serial.get('converged', 'n/a')}")
    print(f"  MPI{args.mpi_size}: n_func_evals={mpi['n_func_evals']}, "
          f"n_accepted_iters={mpi['n_accepted_iterations']}, "
          f"elapsed={mpi['elapsed_s']:.0f}s, converged={mpi.get('converged', 'n/a')}")
    print(f"  bg RMSE: serial={serial['background_rmse']:.4e}, mpi={mpi['background_rmse']:.4e}")
    print()

    # Determine accepted evals. Manual-probe traces carry the info directly.
    if serial.get("mode") == "manual_armijo_probe":
        s_accepted_eval_ids = _manual_accepted(serial)
    else:
        s_matches = infer_accepted(serial["evals"], serial["iterations"])
        s_accepted_eval_ids = {e for (_, e) in s_matches if e is not None}
    if mpi.get("mode") == "manual_armijo_probe":
        m_accepted_eval_ids = _manual_accepted(mpi)
    else:
        m_matches = infer_accepted(mpi["evals"], mpi["iterations"])
        m_accepted_eval_ids = {e for (_, e) in m_matches if e is not None}
    print(f"  Accepted evals (serial): {sorted(s_accepted_eval_ids)}")
    print(f"  Accepted evals (MPI):    {sorted(m_accepted_eval_ids)}")
    print()

    # Per-eval comparison table
    header = (
        f"  {'eval':>4} {'serial_cost':>13} {'mpi_cost':>13} {'ratio':>6} "
        f"{'grad_s':>10} {'grad_m':>10} {'||g||r':>7} "
        f"{'cos_g':>6} {'cos_h':>6} {'cos_u':>6} {'cos_v':>6} "
        f"{'step_s':>10} {'step_m':>10} {'accept':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    pair_prefix_s = str(RESULTS / f"trace_{tag}_serial")
    pair_prefix_m = str(RESULTS / f"trace_{tag}_mpi{args.mpi_size}")

    per_eval = []
    x_prev_s = load_vec(Path(f"{pair_prefix_s}_m_background.npz"))
    x_prev_m = load_vec(Path(f"{pair_prefix_m}_m_background.npz"))

    max_k = min(len(serial["evals"]), len(mpi["evals"]))
    for k in range(1, max_k + 1):
        s_rec = serial["evals"][k - 1]
        m_rec = mpi["evals"][k - 1]

        # Load per-eval vectors
        x_s = load_vec(Path(f"{pair_prefix_s}_eval{k:02d}_x.npz"))
        g_s = load_vec(Path(f"{pair_prefix_s}_eval{k:02d}_grad.npz"))
        x_m = load_vec(Path(f"{pair_prefix_m}_eval{k:02d}_x.npz"))
        g_m = load_vec(Path(f"{pair_prefix_m}_eval{k:02d}_grad.npz"))

        # Grad cosine
        if g_s is not None and g_m is not None:
            gm = metrics_pair(g_s, g_m)
            cos_total = gm["total_cos"]
            cos_h = gm["h_cos"]; cos_u = gm["u_cos"]; cos_v = gm["v_cos"]
        else:
            gm = None
            cos_total = cos_h = cos_u = cos_v = float("nan")

        # Step norms (against prev accepted or prev eval?)
        # For line-search probes, step from x_{k-1} in the SAME run is what TAO actually tried.
        if x_s is not None and x_prev_s is not None:
            step_s = step_metrics(x_prev_s, x_s)["total_step_norm"]
        else:
            step_s = float("nan")
        if x_m is not None and x_prev_m is not None:
            step_m = step_metrics(x_prev_m, x_m)["total_step_norm"]
        else:
            step_m = float("nan")

        # Rolling previous x: update to CURRENT x (so step at eval k+1 is relative to x_k)
        x_prev_s = x_s if x_s is not None else x_prev_s
        x_prev_m = x_m if x_m is not None else x_prev_m

        cost_ratio = m_rec["cost"] / max(abs(s_rec["cost"]), 1e-30)
        grad_ratio = m_rec["grad_norm_total"] / max(s_rec["grad_norm_total"], 1e-30)
        s_acc = "S" if k in s_accepted_eval_ids else ""
        m_acc = "M" if k in m_accepted_eval_ids else ""
        acc = s_acc + m_acc or "-"

        print(
            f"  {k:>4d} {s_rec['cost']:>13.4e} {m_rec['cost']:>13.4e} {cost_ratio:>6.2f} "
            f"{s_rec['grad_norm_total']:>10.3e} {m_rec['grad_norm_total']:>10.3e} "
            f"{grad_ratio:>7.3f} "
            f"{cos_total:>6.3f} {cos_h:>6.3f} {cos_u:>6.3f} {cos_v:>6.3f} "
            f"{step_s:>10.3e} {step_m:>10.3e} {acc:>7}"
        )

        per_eval.append({
            "eval": k,
            "serial_cost": s_rec["cost"],
            "mpi_cost": m_rec["cost"],
            "cost_ratio_mpi_over_serial": cost_ratio,
            "serial_grad_norm": s_rec["grad_norm_total"],
            "mpi_grad_norm": m_rec["grad_norm_total"],
            "grad_norm_ratio": grad_ratio,
            "serial_grad_h": s_rec["grad_norm_h"],
            "serial_grad_u": s_rec["grad_norm_u"],
            "serial_grad_v": s_rec["grad_norm_v"],
            "mpi_grad_h": m_rec["grad_norm_h"],
            "mpi_grad_u": m_rec["grad_norm_u"],
            "mpi_grad_v": m_rec["grad_norm_v"],
            "cos_grad_total": cos_total,
            "cos_grad_h": cos_h,
            "cos_grad_u": cos_u,
            "cos_grad_v": cos_v,
            "grad_pair_metrics": gm,
            "serial_step_norm": step_s,
            "mpi_step_norm": step_m,
            "serial_rmse_total": s_rec["rmse_from_bg_total"],
            "mpi_rmse_total": m_rec["rmse_from_bg_total"],
            "serial_rmse_h": s_rec["rmse_from_bg_h"],
            "mpi_rmse_h": m_rec["rmse_from_bg_h"],
            "serial_rmse_u": s_rec["rmse_from_bg_u"],
            "mpi_rmse_u": m_rec["rmse_from_bg_u"],
            "serial_rmse_v": s_rec["rmse_from_bg_v"],
            "mpi_rmse_v": m_rec["rmse_from_bg_v"],
            "serial_n_active_lower": s_rec["n_active_lower"],
            "serial_n_active_upper": s_rec["n_active_upper"],
            "mpi_n_active_lower": m_rec["n_active_lower"],
            "mpi_n_active_upper": m_rec["n_active_upper"],
            "serial_accepted": k in s_accepted_eval_ids,
            "mpi_accepted": k in m_accepted_eval_ids,
        })

    print()
    # Divergence diagnosis
    cost_ratios = [r["cost_ratio_mpi_over_serial"] for r in per_eval]
    grad_ratios = [r["grad_norm_ratio"] for r in per_eval]
    cos_grads = [r["cos_grad_total"] for r in per_eval if not np.isnan(r["cos_grad_total"])]
    print("  DIAGNOSTICS:")
    print(f"    cost ratio (MPI/serial) min={min(cost_ratios):.3f}, max={max(cost_ratios):.3f}")
    print(f"    grad norm ratio (MPI/serial) min={min(grad_ratios):.3f}, max={max(grad_ratios):.3f}")
    if cos_grads:
        print(f"    grad cosine (serial, MPI) min={min(cos_grads):.3f}, "
              f"max={max(cos_grads):.3f}, mean={np.mean(cos_grads):.3f}")
    # Find first material divergence (>5% in cost)
    first_div = next((r["eval"] for r in per_eval
                     if abs(r["cost_ratio_mpi_over_serial"] - 1.0) > 0.05), None)
    if first_div is not None:
        print(f"    first eval with >5% cost deviation: eval {first_div}")
    else:
        print(f"    no eval with >5% cost deviation")

    # Bounds interaction
    s_act = set()
    m_act = set()
    for r in per_eval:
        if r["serial_n_active_lower"] > 0 or r["serial_n_active_upper"] > 0:
            s_act.add(r["eval"])
        if r["mpi_n_active_lower"] > 0 or r["mpi_n_active_upper"] > 0:
            m_act.add(r["eval"])
    print(f"    serial evals with active bounds: {sorted(s_act) or 'none'}")
    print(f"    MPI evals with active bounds: {sorted(m_act) or 'none'}")

    # Write JSON summary
    if args.json_out is None:
        args.json_out = str(RESULTS / f"comparison_{tag}.json")
    summary = {
        "tag": tag,
        "mpi_size": args.mpi_size,
        "serial_meta": {k: serial[k] for k in serial if k not in ("evals", "iterations")},
        "mpi_meta": {k: mpi[k] for k in mpi if k not in ("evals", "iterations")},
        "per_eval": per_eval,
        "accepted_evals_serial": sorted(s_accepted_eval_ids),
        "accepted_evals_mpi": sorted(m_accepted_eval_ids),
        "first_eval_5pct_diverge": first_div,
    }
    with open(args.json_out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved comparison: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

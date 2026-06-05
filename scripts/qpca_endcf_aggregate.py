#!/usr/bin/env python3
"""Aggregate and compare QPCA-EnDCF / 4D-EnKF result JSONs.

Walks one or more roots looking for ``result_qpca_endcf.json`` files,
extracts the per-window metrics produced by
``experiments/idealized_inlet_qpca_endcf.py``, and prints two tables:

  1. *summary* — one row per run with config (method, κ, λ_infl, L_loc,
     N, #windows) plus aggregates (mean state Δ%, mean obs Δ%, last-
     window RMSE, last-window γ̄).
  2. *per-window* — only with ``--per-window``: one block per run with
     window-by-window state Δ%, obs Δ%, analysis spread, γ̄, top-1/trace.

Usage
-----
    python scripts/qpca_endcf_aggregate.py results/
    python scripts/qpca_endcf_aggregate.py --per-window results/
    python scripts/qpca_endcf_aggregate.py path/to/dir1 path/to/dir2 ...
    python scripts/qpca_endcf_aggregate.py --csv summary.csv results/

Stdlib-only.
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Discovery & loading
# ---------------------------------------------------------------------------


def _find_result_jsons(roots: Iterable[Path]) -> List[Path]:
    """Recursively locate every ``result_qpca_endcf.json`` under each root."""
    found: List[Path] = []
    for root in roots:
        if not root.exists():
            print(f"[warn] root does not exist: {root}", file=sys.stderr)
            continue
        if root.is_file() and root.name == "result_qpca_endcf.json":
            found.append(root)
            continue
        for path in root.rglob("result_qpca_endcf.json"):
            found.append(path)
    # Deduplicate while preserving order.
    seen = set()
    out: List[Path] = []
    for p in found:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] could not read {path}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _summarize(path: Path, doc: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a result document to a flat row of summary stats."""
    cfg = doc.get("config", {}) or {}
    windows = doc.get("windows", []) or []
    state_deltas = [float(w.get("state_improvement_pct", 0.0)) for w in windows]
    obs_deltas = [float(w.get("obs_improvement_pct", 0.0)) for w in windows]

    last = windows[-1] if windows else {}
    last_rmse = float(last.get("analysis_rmse_truth_end", float("nan")))
    last_gamma = float(last.get("analysis_spread_skill_truth", float("nan")))
    last_top1_over_tr = float(last.get("spectrum_top1_over_trace", float("nan")))

    return {
        "run": path.parent.name,
        "method": doc.get("method", "?"),
        "kappa": int(doc.get("k_modes", 0) or 0),
        "infl": float(cfg.get("inflation", 1.0)),
        "loc_km": float(cfg.get("loc_radius_m", 0.0)) / 1000.0,
        "N": int(doc.get("ensemble_size", 0) or 0),
        "n_w": int(doc.get("n_windows", 0) or 0),
        "vmax": float(cfg.get("vmax", float("nan"))),
        "bg_std": float(cfg.get("background_error_std", float("nan"))),
        "mean_state_d": _mean(state_deltas),
        "mean_obs_d": _mean(obs_deltas),
        "last_rmse": last_rmse,
        "last_gamma": last_gamma,
        "last_top1_tr": last_top1_over_tr,
        "path": str(path),
    }


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------


_SUMMARY_COLS = [
    ("run",          "run",                 26, "s"),
    ("method",       "method",               8, "s"),
    ("kappa",        "κ",                    3, "d"),
    ("infl",         "λ_infl",               6, ".2f"),
    ("loc_km",       "L_km",                 5, ".1f"),
    ("N",            "N",                    3, "d"),
    ("n_w",          "n_w",                  4, "d"),
    ("vmax",         "vmax",                 5, ".1f"),
    ("bg_std",       "bg_σ",                 5, ".2f"),
    ("mean_state_d", "⟨ΔRMSE⟩%",            10, "+.2f"),
    ("mean_obs_d",   "⟨Δobs⟩%",              9, "+.2f"),
    ("last_rmse",    "rmse_T",               9, ".4f"),
    ("last_gamma",   "γ̄_T",                  6, ".2f"),
    ("last_top1_tr", "λ1/tr_T",              8, ".2f"),
]


def _fmt_cell(val: Any, width: int, spec: str) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or
                                                   math.isinf(val))):
        return f"{'—':>{width}s}"
    if spec == "s":
        s = str(val)
        if len(s) > width:
            s = s[: width - 1] + "…"
        return f"{s:<{width}s}"
    fmt = "{:" + spec + "}"
    return f"{fmt.format(val):>{width}s}"


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("[no rows]")
        return
    header = " ".join(
        f"{label:>{w}s}" if spec != "s" else f"{label:<{w}s}"
        for _, label, w, spec in _SUMMARY_COLS
    )
    rule = "-" * len(header)
    print(rule)
    print(header)
    print(rule)
    for row in rows:
        print(" ".join(
            _fmt_cell(row.get(k), w, spec)
            for k, _, w, spec in _SUMMARY_COLS
        ))
    print(rule)


def _print_per_window(rows: List[Dict[str, Any]], docs: Dict[str, Dict]) -> None:
    cols = [
        ("window",                     "w",          3,  "d"),
        ("forecast_rmse_truth_end",    "fc_rmse",    9,  ".4f"),
        ("analysis_rmse_truth_end",    "an_rmse",    9,  ".4f"),
        ("state_improvement_pct",      "ΔRMSE%",     8,  "+.2f"),
        ("prior_obs_misfit_end",       "obs_pri",    9,  ".4f"),
        ("analysis_obs_misfit_end",    "obs_ana",    9,  ".4f"),
        ("obs_improvement_pct",        "Δobs%",      7,  "+.2f"),
        ("forecast_spread",            "fc_sprd",    9,  ".4f"),
        ("analysis_spread",            "an_sprd",    9,  ".4f"),
        ("analysis_spread_skill_truth","γ̄",          5,  ".2f"),
        ("spectrum_top1_over_trace",   "λ1/tr",      6,  ".2f"),
        ("n_failed_members",           "fail",       4,  "d"),
    ]
    for row in rows:
        run = row["run"]
        doc = docs.get(row["path"])
        if doc is None:
            continue
        print()
        method = doc.get("method", "?")
        n_w = doc.get("n_windows", 0)
        infl = float((doc.get("config") or {}).get("inflation", 1.0))
        loc_km = float((doc.get("config") or {}).get("loc_radius_m", 0.0)) / 1000
        kappa = doc.get("k_modes", 0)
        N = doc.get("ensemble_size", 0)
        print(f"=== {run}  ({method}, κ={kappa}, λ_infl={infl:.2f}, "
              f"L={loc_km:.1f} km, N={N}, n_w={n_w}) ===")
        header = " ".join(
            f"{label:>{w}s}" if spec != "s" else f"{label:<{w}s}"
            for _, label, w, spec in cols
        )
        rule = "-" * len(header)
        print(rule)
        print(header)
        print(rule)
        for wdoc in doc.get("windows", []):
            print(" ".join(
                _fmt_cell(wdoc.get(k), w, spec)
                for k, _, w, spec in cols
            ))
        print(rule)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Aggregate QPCA-EnDCF / 4D-EnKF result JSONs",
    )
    p.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("results")],
        help="Directories to scan (recursive). Default: results/",
    )
    p.add_argument(
        "--per-window",
        action="store_true",
        help="Also print per-window tables for each run.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write summary rows to CSV at this path.",
    )
    p.add_argument(
        "--sort",
        choices=("name", "mean_state", "mean_obs", "last_rmse"),
        default="name",
        help="Sort key for the summary table (default: name).",
    )
    args = p.parse_args()

    paths = _find_result_jsons(args.roots)
    if not paths:
        print("[no result_qpca_endcf.json found under: "
              f"{', '.join(str(r) for r in args.roots)}]",
              file=sys.stderr)
        return 1

    docs: Dict[str, Dict] = {}
    rows: List[Dict[str, Any]] = []
    for path in paths:
        doc = _load(path)
        if doc is None:
            continue
        docs[str(path)] = doc
        rows.append(_summarize(path, doc))

    key_map = {
        "name":        lambda r: r["run"],
        "mean_state":  lambda r: -r["mean_state_d"],
        "mean_obs":    lambda r: -r["mean_obs_d"],
        "last_rmse":   lambda r: r["last_rmse"],
    }
    rows.sort(key=key_map[args.sort])

    _print_summary(rows)

    if args.per_window:
        _print_per_window(rows, docs)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = _csv.DictWriter(
                f, fieldnames=[c[0] for c in _SUMMARY_COLS] + ["path"]
            )
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k) for k in writer.fieldnames})
        print(f"[wrote] {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

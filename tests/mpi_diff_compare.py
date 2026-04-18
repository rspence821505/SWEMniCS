#!/usr/bin/env python3
"""
Compare two object-diff exports written by tests/mpi_diff_export.py.

For each artifact present in BOTH runs, computes:
  * relative L2 error per component (h, u, v) and total
  * cosine similarity per component
  * max absolute error and its (x, y) location
  * top-K worst DOFs with coordinates and per-h-DOF interface distance
  * coordinate-binned mean abs error (NX × NY tiles over the bounding box)
  * interface-distance summary (rank0's interface_distance from artifact A
    is binned into 0–125 / 125–500 / 500–1500 / 1500–5000 / 5000+ m bands;
    mean abs h-error is reported per band — this answers "is the difference
    localized near partition interfaces?")

Outputs:
  * Human-readable summary printed to stdout
  * results/mpi_object_diff/_compare/{a}_vs_{b}/comparison.json

Matrix-row artifacts are compared row-by-row: matched by row_coord (with
small tolerance), then column entries matched by (component, x, y) so that
permutation in PETSc.Mat.getRow ordering is irrelevant.

Usage:
    python tests/mpi_diff_compare.py --a serial --b mpi2
    python tests/mpi_diff_compare.py --a serial --b mpi2 --top-k 25 --verbose

Environment knob:
    --root  results/mpi_object_diff (default)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------
# loaders + matchers
# ----------------------------------------------------------------------
def list_artifacts(run_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for f in sorted(run_dir.glob("*.npz")):
        m = re.match(r"^(.+)__(?P<tag>[^_]+(?:[^_]*))$", f.stem)
        if m:
            base = m.group(1)
        else:
            base = f.stem
        out[base] = f
    return out


def load_dof_metadata(run_dir: Path, tag: str) -> Optional[dict]:
    p = run_dir / f"dof_metadata__{tag}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {
        "coords": d["coords"],
        "owner": d["owner"],
        "interface_distance": d["interface_distance"],
        "mpi_size": int(d["mpi_size"]),
        "global_n_h": int(d["global_n_h"]),
        "bs": int(d["bs"]),
    }


# ----------------------------------------------------------------------
# vector-comparison primitives
# ----------------------------------------------------------------------
@dataclass
class VectorComponentDiff:
    norm_a: float
    norm_b: float
    diff_norm: float
    rel_diff: float
    cos_sim: float
    max_abs_diff: float
    max_abs_diff_at_xy: Tuple[float, float]


def _component_metrics(va: np.ndarray, vb: np.ndarray, coords: np.ndarray) -> VectorComponentDiff:
    diff = vb - va
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    diff_norm = float(np.linalg.norm(diff))
    denom = norm_a * norm_b
    cos_sim = float(np.dot(va, vb) / denom) if denom > 0 else 0.0
    if len(diff) > 0:
        i_max = int(np.argmax(np.abs(diff)))
        max_abs = float(abs(diff[i_max]))
        loc = (float(coords[i_max, 0]), float(coords[i_max, 1]))
    else:
        max_abs = 0.0
        loc = (float("nan"), float("nan"))
    return VectorComponentDiff(
        norm_a=norm_a,
        norm_b=norm_b,
        diff_norm=diff_norm,
        rel_diff=diff_norm / max(norm_a, 1e-30),
        cos_sim=cos_sim,
        max_abs_diff=max_abs,
        max_abs_diff_at_xy=loc,
    )


def _spatial_bin_summary(va: np.ndarray, vb: np.ndarray, coords: np.ndarray,
                         nx: int = 8, ny: int = 8) -> dict:
    if len(coords) == 0:
        return {"nx": nx, "ny": ny, "bin_mean_abs_diff": [], "bin_centers": [], "bin_counts": []}
    diff = np.abs(vb - va)
    xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
    ymin, ymax = float(coords[:, 1].min()), float(coords[:, 1].max())
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        ymax = ymin + 1.0
    ix = np.clip(((coords[:, 0] - xmin) / (xmax - xmin) * nx).astype(int), 0, nx - 1)
    iy = np.clip(((coords[:, 1] - ymin) / (ymax - ymin) * ny).astype(int), 0, ny - 1)
    bin_id = ix * ny + iy
    n_bins = nx * ny
    counts = np.bincount(bin_id, minlength=n_bins)
    sums = np.bincount(bin_id, weights=diff, minlength=n_bins)
    mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    cx = (np.arange(nx) + 0.5) / nx * (xmax - xmin) + xmin
    cy = (np.arange(ny) + 0.5) / ny * (ymax - ymin) + ymin
    centers = np.array([(cx[i // ny], cy[i % ny]) for i in range(n_bins)])
    return {
        "nx": nx, "ny": ny,
        "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
        "bin_centers": centers.tolist(),
        "bin_counts": counts.astype(int).tolist(),
        "bin_mean_abs_diff": [None if np.isnan(v) else float(v) for v in mean.tolist()],
    }


def _interface_distance_bands(va: np.ndarray, vb: np.ndarray,
                              interface_dist: np.ndarray) -> dict:
    """Bin |b - a| by interface distance bands. Answers: is the difference
    concentrated near partition boundaries?
    """
    bands = [(0.0, 125.0), (125.0, 500.0), (500.0, 1500.0),
             (1500.0, 5000.0), (5000.0, np.inf)]
    diff = np.abs(vb - va)
    out = []
    for lo, hi in bands:
        mask = (interface_dist >= lo) & (interface_dist < hi)
        n = int(np.sum(mask))
        if n == 0:
            out.append({"lo": lo, "hi": hi if np.isfinite(hi) else None, "n": 0,
                        "mean_abs_diff": None, "max_abs_diff": None})
        else:
            out.append({
                "lo": lo, "hi": hi if np.isfinite(hi) else None, "n": n,
                "mean_abs_diff": float(diff[mask].mean()),
                "max_abs_diff": float(diff[mask].max()),
            })
    return {"bands": out}


def _top_k_worst_dofs(va: np.ndarray, vb: np.ndarray, coords: np.ndarray,
                      interface_dist: Optional[np.ndarray], k: int) -> List[dict]:
    diff = vb - va
    if len(diff) == 0:
        return []
    idx = np.argsort(-np.abs(diff))[:k]
    out = []
    for i in idx:
        entry = {
            "dof_canonical_idx": int(i),
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "value_a": float(va[i]),
            "value_b": float(vb[i]),
            "abs_diff": float(abs(diff[i])),
        }
        if interface_dist is not None:
            entry["interface_distance"] = (
                float(interface_dist[i]) if np.isfinite(interface_dist[i]) else None
            )
        out.append(entry)
    return out


# ----------------------------------------------------------------------
# vector & matrix-action artifact comparisons
# ----------------------------------------------------------------------
def compare_vector_artifact(path_a: Path, path_b: Path, top_k: int = 10,
                            iface_for_h: Optional[np.ndarray] = None) -> dict:
    A = np.load(path_a)
    B = np.load(path_b)
    cA = A["coords"]
    cB = B["coords"]
    coord_diff = float(np.max(np.abs(cA - cB))) if cA.shape == cB.shape else float("inf")

    out = {
        "artifact_a": str(path_a),
        "artifact_b": str(path_b),
        "n_h": int(len(cA)),
        "coord_max_abs_diff": coord_diff,
        "coord_aligned": coord_diff < 1e-6,
        "components": {},
        "interface_bands": {},
        "spatial_bins": {},
        "top_k_h": [],
    }
    if not out["coord_aligned"]:
        out["error"] = "coords mismatched between artifacts; cannot compare"
        return out

    iface = iface_for_h
    if iface is None and "interface_distance" in A.files:
        iface = A["interface_distance"]

    for comp in ("h", "u", "v"):
        if comp not in A.files or comp not in B.files:
            continue
        m = _component_metrics(A[comp], B[comp], cA)
        out["components"][comp] = asdict(m)
        out["spatial_bins"][comp] = _spatial_bin_summary(A[comp], B[comp], cA)
        if iface is not None:
            out["interface_bands"][comp] = _interface_distance_bands(A[comp], B[comp], iface)

    if "h" in A.files and "h" in B.files:
        out["top_k_h"] = _top_k_worst_dofs(A["h"], B["h"], cA, iface, top_k)

    if all(c in A.files for c in ("h", "u", "v")):
        ta = np.concatenate([A["h"], A["u"], A["v"]])
        tb = np.concatenate([B["h"], B["u"], B["v"]])
        denom = float(np.linalg.norm(ta)) * float(np.linalg.norm(tb))
        out["total"] = {
            "norm_a": float(np.linalg.norm(ta)),
            "norm_b": float(np.linalg.norm(tb)),
            "diff_norm": float(np.linalg.norm(tb - ta)),
            "rel_diff": float(np.linalg.norm(tb - ta)) / max(float(np.linalg.norm(ta)), 1e-30),
            "cos_sim": float(np.dot(ta, tb) / denom) if denom > 0 else 0.0,
        }
    return out


def compare_matrix_action_artifact(path_a: Path, path_b: Path, top_k: int = 10,
                                   iface_for_h: Optional[np.ndarray] = None) -> dict:
    """Matrix-action artifacts contain both input and output vectors. The
    input MUST agree exactly between runs (it's coord-deterministic); the
    output is the assembled-then-applied matrix-vector product whose
    discrepancy we want to characterize.
    """
    A = np.load(path_a)
    B = np.load(path_b)
    out = {
        "artifact_a": str(path_a),
        "artifact_b": str(path_b),
        "transpose": int(A["transpose"]) if "transpose" in A.files else None,
        "input_components": {},
        "output_components": {},
        "spatial_bins_output": {},
        "interface_bands_output": {},
        "top_k_h_output": [],
    }
    cA = A["coords"]
    cB = B["coords"]
    if cA.shape != cB.shape or float(np.max(np.abs(cA - cB))) > 1e-6:
        out["error"] = "coords mismatched between artifacts; cannot compare"
        return out

    iface = iface_for_h if iface_for_h is not None else (
        A["interface_distance"] if "interface_distance" in A.files else None)

    for comp in ("h", "u", "v"):
        if f"input_{comp}" in A.files and f"input_{comp}" in B.files:
            m = _component_metrics(A[f"input_{comp}"], B[f"input_{comp}"], cA)
            out["input_components"][comp] = asdict(m)

    for comp in ("h", "u", "v"):
        if f"output_{comp}" not in A.files or f"output_{comp}" not in B.files:
            continue
        m = _component_metrics(A[f"output_{comp}"], B[f"output_{comp}"], cA)
        out["output_components"][comp] = asdict(m)
        out["spatial_bins_output"][comp] = _spatial_bin_summary(
            A[f"output_{comp}"], B[f"output_{comp}"], cA)
        if iface is not None:
            out["interface_bands_output"][comp] = _interface_distance_bands(
                A[f"output_{comp}"], B[f"output_{comp}"], iface)

    if "output_h" in A.files and "output_h" in B.files:
        out["top_k_h_output"] = _top_k_worst_dofs(
            A["output_h"], B["output_h"], cA, iface, top_k)

    return out


# ----------------------------------------------------------------------
# matrix-row artifact comparisons
# ----------------------------------------------------------------------
def _load_matrix_rows(path: Path) -> List[dict]:
    d = np.load(path, allow_pickle=False)
    n = int(d["n_rows"]) if "n_rows" in d.files else 0
    rows = []
    for i in range(n):
        rows.append({
            "row_coord": tuple(map(float, d[f"row_{i}__row_coord"])),
            "row_component": str(d[f"row_{i}__row_component"]),
            "row_owner": int(d[f"row_{i}__row_owner"]),
            "row_global_idx": int(d[f"row_{i}__row_global_idx"]),
            "row_distance_to_query": float(d[f"row_{i}__row_distance_to_query"]),
            "col_global_idx": d[f"row_{i}__col_global_idx"],
            "col_component": d[f"row_{i}__col_component"],
            "col_x": d[f"row_{i}__col_x"],
            "col_y": d[f"row_{i}__col_y"],
            "col_value": d[f"row_{i}__col_value"],
        })
    return rows


def compare_matrix_rows_artifact(path_a: Path, path_b: Path,
                                 coord_tol: float = 1e-6, value_tol: float = 1e-12) -> dict:
    rows_a = _load_matrix_rows(path_a)
    rows_b = _load_matrix_rows(path_b)

    def _row_key(r):
        return (round(r["row_coord"][0], 4), round(r["row_coord"][1], 4), r["row_component"])
    map_a = {_row_key(r): r for r in rows_a}
    map_b = {_row_key(r): r for r in rows_b}

    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    only_a = sorted(set(map_a.keys()) - set(map_b.keys()))
    only_b = sorted(set(map_b.keys()) - set(map_a.keys()))

    row_diffs = []
    for k in common:
        ra, rb = map_a[k], map_b[k]
        ka_keys = [(int(ra["col_component"][i]),
                    round(float(ra["col_x"][i]), 4),
                    round(float(ra["col_y"][i]), 4))
                   for i in range(len(ra["col_value"]))]
        kb_keys = [(int(rb["col_component"][i]),
                    round(float(rb["col_x"][i]), 4),
                    round(float(rb["col_y"][i]), 4))
                   for i in range(len(rb["col_value"]))]
        ka_map = {key: float(ra["col_value"][i]) for i, key in enumerate(ka_keys)}
        kb_map = {key: float(rb["col_value"][i]) for i, key in enumerate(kb_keys)}

        all_keys = set(ka_map.keys()) | set(kb_map.keys())
        per_col_abs_diffs = []
        only_in_a, only_in_b = 0, 0
        max_diff_key = None
        max_diff_val = 0.0
        for key in all_keys:
            va = ka_map.get(key, 0.0)
            vb = kb_map.get(key, 0.0)
            if key not in ka_map and abs(vb) > value_tol:
                only_in_b += 1
            if key not in kb_map and abs(va) > value_tol:
                only_in_a += 1
            d = abs(vb - va)
            per_col_abs_diffs.append(d)
            if d > max_diff_val:
                max_diff_val = d
                max_diff_key = key
        diffs = np.array(per_col_abs_diffs)
        row_diffs.append({
            "row_coord": list(ra["row_coord"]),
            "row_component": ra["row_component"],
            "row_owner_a": ra["row_owner"],
            "row_owner_b": rb["row_owner"],
            "row_global_idx_a": ra["row_global_idx"],
            "row_global_idx_b": rb["row_global_idx"],
            "n_cols_a": int(len(ra["col_value"])),
            "n_cols_b": int(len(rb["col_value"])),
            "n_cols_only_in_a": int(only_in_a),
            "n_cols_only_in_b": int(only_in_b),
            "max_abs_value_diff": float(max_diff_val),
            "max_abs_value_diff_at": list(max_diff_key) if max_diff_key else None,
            "mean_abs_value_diff": float(diffs.mean()) if len(diffs) > 0 else 0.0,
            "frob_value_diff": float(np.sqrt(np.sum(diffs ** 2))),
        })
    return {
        "artifact_a": str(path_a),
        "artifact_b": str(path_b),
        "n_rows_a": len(rows_a),
        "n_rows_b": len(rows_b),
        "n_common": len(common),
        "n_only_in_a": len(only_a),
        "n_only_in_b": len(only_b),
        "row_diffs": row_diffs,
    }


# ----------------------------------------------------------------------
# CLI driver + summary
# ----------------------------------------------------------------------
def _classify(name: str) -> str:
    if name.startswith("dof_metadata"):
        return "metadata"
    if name.startswith("matrix_rows__"):
        return "matrix_rows"
    if name.startswith("jac_action_"):
        return "matrix_action"
    return "vector"


def _format_vector_summary(name: str, r: dict) -> str:
    if "error" in r:
        return f"  [{name}] ERROR: {r['error']}"
    lines = [f"  [{name}]  n={r['n_h']}  coord_aligned={r['coord_aligned']}"]
    if "total" in r:
        t = r["total"]
        lines.append(f"    TOTAL: ||a||={t['norm_a']:.4e} ||b||={t['norm_b']:.4e} "
                     f"rel_diff={t['rel_diff']:.4e} cos_sim={t['cos_sim']:.6f}")
    for comp, m in r["components"].items():
        lines.append(
            f"    {comp}: ||a||={m['norm_a']:.4e} rel={m['rel_diff']:.4e} "
            f"cos={m['cos_sim']:.6f} max_abs={m['max_abs_diff']:.4e} "
            f"@({m['max_abs_diff_at_xy'][0]:.0f},{m['max_abs_diff_at_xy'][1]:.0f})"
        )
    if r.get("interface_bands"):
        bands = r["interface_bands"].get("h")
        if bands:
            lines.append("    h interface-distance bands (mean abs diff):")
            for b in bands["bands"]:
                hi = "inf" if b["hi"] is None else f"{b['hi']:.0f}"
                if b["n"] == 0:
                    s = "(empty)"
                else:
                    s = f"mean={b['mean_abs_diff']:.4e} max={b['max_abs_diff']:.4e}"
                lines.append(f"      [{b['lo']:>5.0f}, {hi:>5}) m  n={b['n']:>5} {s}")
    return "\n".join(lines)


def _format_matrix_action_summary(name: str, r: dict) -> str:
    if "error" in r:
        return f"  [{name}] ERROR: {r['error']}"
    lines = [f"  [{name}]  transpose={r.get('transpose')}"]
    if r["input_components"]:
        in_h = r["input_components"].get("h")
        if in_h:
            lines.append(f"    input  h: rel={in_h['rel_diff']:.4e} cos={in_h['cos_sim']:.6f}  "
                         "(should be ~0 since input is coord-deterministic)")
    for comp, m in r["output_components"].items():
        lines.append(
            f"    output {comp}: ||a||={m['norm_a']:.4e} rel={m['rel_diff']:.4e} "
            f"cos={m['cos_sim']:.6f} max_abs={m['max_abs_diff']:.4e} "
            f"@({m['max_abs_diff_at_xy'][0]:.0f},{m['max_abs_diff_at_xy'][1]:.0f})"
        )
    bands = r.get("interface_bands_output", {}).get("h")
    if bands:
        lines.append("    output h interface-distance bands (mean abs diff):")
        for b in bands["bands"]:
            hi = "inf" if b["hi"] is None else f"{b['hi']:.0f}"
            if b["n"] == 0:
                s = "(empty)"
            else:
                s = f"mean={b['mean_abs_diff']:.4e} max={b['max_abs_diff']:.4e}"
            lines.append(f"      [{b['lo']:>5.0f}, {hi:>5}) m  n={b['n']:>5} {s}")
    return "\n".join(lines)


def _format_matrix_rows_summary(name: str, r: dict) -> str:
    lines = [f"  [{name}]  n_rows: a={r['n_rows_a']} b={r['n_rows_b']} "
             f"common={r['n_common']} only_a={r['n_only_in_a']} only_b={r['n_only_in_b']}"]
    for rd in r["row_diffs"][:8]:
        lines.append(
            f"    @({rd['row_coord'][0]:.0f},{rd['row_coord'][1]:.0f}) {rd['row_component']}  "
            f"|cols|: a={rd['n_cols_a']} b={rd['n_cols_b']}  "
            f"only_a={rd['n_cols_only_in_a']} only_b={rd['n_cols_only_in_b']}  "
            f"max_diff={rd['max_abs_value_diff']:.4e} "
            f"frob_diff={rd['frob_value_diff']:.4e}"
        )
    if len(r["row_diffs"]) > 8:
        lines.append(f"    ... ({len(r['row_diffs']) - 8} more rows)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare two MPI object-diff exports")
    parser.add_argument("--root", default=str(PROJECT_ROOT / "results" / "mpi_object_diff"),
                        help="Root directory containing per-run subdirs")
    parser.add_argument("--a", required=True, help="Run A tag (e.g. serial)")
    parser.add_argument("--b", required=True, help="Run B tag (e.g. mpi2)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K worst h-DOFs to report")
    parser.add_argument("--verbose", action="store_true", help="Print per-artifact details")
    args = parser.parse_args()

    root = Path(args.root)
    dir_a = root / args.a
    dir_b = root / args.b
    if not dir_a.exists():
        sys.exit(f"missing run dir: {dir_a}")
    if not dir_b.exists():
        sys.exit(f"missing run dir: {dir_b}")

    out_dir = root / "_compare" / f"{args.a}_vs_{args.b}"
    out_dir.mkdir(parents=True, exist_ok=True)

    art_a_files = sorted(dir_a.glob("*.npz"))
    art_b_files = sorted(dir_b.glob("*.npz"))

    def base_name(p: Path, tag: str) -> str:
        s = p.stem
        suffix = f"__{tag}"
        return s[:-len(suffix)] if s.endswith(suffix) else s

    map_a = {base_name(p, args.a): p for p in art_a_files}
    map_b = {base_name(p, args.b): p for p in art_b_files}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))

    iface_h = None
    meta_a = load_dof_metadata(dir_a, args.a)
    meta_b = load_dof_metadata(dir_b, args.b)
    if meta_b is not None:
        iface_h = meta_b["interface_distance"]
        if meta_a is not None and meta_a["coords"].shape == meta_b["coords"].shape:
            cd = float(np.max(np.abs(meta_a["coords"] - meta_b["coords"])))
            if cd > 1e-6:
                print(f"WARNING: dof metadata coord mismatch max={cd:.2e} between runs")

    print("=" * 78)
    print(f"OBJECT DIFF: {args.a}  vs  {args.b}")
    print(f"  artifacts in A: {len(map_a)}, in B: {len(map_b)}, common: {len(common)}")
    only_a = sorted(set(map_a.keys()) - set(map_b.keys()))
    only_b = sorted(set(map_b.keys()) - set(map_a.keys()))
    if only_a:
        print(f"  only in A: {only_a}")
    if only_b:
        print(f"  only in B: {only_b}")
    print("=" * 78)

    summary: Dict[str, dict] = {}
    for name in common:
        kind = _classify(name)
        if kind == "metadata":
            continue
        a_path, b_path = map_a[name], map_b[name]
        if kind == "vector":
            r = compare_vector_artifact(a_path, b_path, top_k=args.top_k, iface_for_h=iface_h)
            summary[name] = r
            print(_format_vector_summary(name, r))
        elif kind == "matrix_action":
            r = compare_matrix_action_artifact(a_path, b_path, top_k=args.top_k, iface_for_h=iface_h)
            summary[name] = r
            print(_format_matrix_action_summary(name, r))
        elif kind == "matrix_rows":
            r = compare_matrix_rows_artifact(a_path, b_path)
            summary[name] = r
            print(_format_matrix_rows_summary(name, r))

    out_json = out_dir / "comparison.json"
    with open(out_json, "w") as f:
        json.dump({
            "a_tag": args.a,
            "b_tag": args.b,
            "n_artifacts_common": len(common),
            "only_in_a": only_a,
            "only_in_b": only_b,
            "meta_a": {"mpi_size": meta_a["mpi_size"], "global_n_h": meta_a["global_n_h"]} if meta_a else None,
            "meta_b": {"mpi_size": meta_b["mpi_size"], "global_n_h": meta_b["global_n_h"]} if meta_b else None,
            "results": summary,
        }, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o))
    print(f"\nWrote: {out_json}")
    print(_first_difference_hint(summary))


def _first_difference_hint(summary: dict) -> str:
    """Quick guidance: which artifact, in canonical-step order, first
    breaches a small relative-diff threshold?"""
    order_priority = [
        "state__m_background",
        "state__trajectory_00",
        "state__trajectory_01",
        "state__trajectory_02",
        "jac_action_00__J_at_ones",
        "jac_action_00__J_at_sin2D",
        "jac_action_00__J_at_localized",
        "jac_action_00__JT_at_ones",
        "jac_action_00__JT_at_sin2D",
        "jac_action_00__JT_at_localized",
        "adjoint_rhs__obs_forcing_00",
        "adjoint_rhs__obs_forcing_01",
        "adjoint_rhs__obs_forcing_02",
        "adjoint__lambda0",
    ]
    REL_TOL = 1e-6
    lines = ["", "FIRST-DIFFERENCE HINT (rel_diff threshold = 1e-6):"]
    found_any = False
    for name in order_priority:
        if name not in summary:
            continue
        r = summary[name]
        rel = None
        comps = r.get("components") or r.get("output_components") or {}
        if comps:
            rels = [c["rel_diff"] for c in comps.values() if isinstance(c, dict) and "rel_diff" in c]
            if rels:
                rel = max(rels)
        if rel is None:
            continue
        ok = rel < REL_TOL
        marker = "OK  " if ok else "DIFF"
        lines.append(f"  {marker}  {name:50s}  max comp rel_diff = {rel:.3e}")
        if not ok and not found_any:
            lines.append(f"  ===> first object exceeding threshold: {name}")
            found_any = True
    if not found_any:
        lines.append("  All probed objects within threshold.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare adjoint vectors from different solver configurations.

Loads globally-coord-ordered vectors saved by test_mpi_parity.py
and computes vector-level similarity metrics.

Usage:
  python tests/compare_adjoint_vectors.py
"""
import json
import sys
import numpy as np
from pathlib import Path
from glob import glob

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "mpi_parity"


def load_vector(tag, size, kind="lambda0"):
    """Load a globally coord-ordered (h, u, v) vector."""
    path = RESULTS_DIR / f"vec_{tag}_size{size}_{kind}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {"coords": d["coords"], "h": d["h"], "u": d["u"], "v": d["v"], "path": str(path)}


def compare(a, b, label):
    """Compare two coord-ordered vectors. Returns dict of metrics."""
    if a is None or b is None:
        return None

    # Verify coord ordering matches
    coord_diff = np.max(np.abs(a["coords"] - b["coords"]))
    if coord_diff > 1e-6:
        print(f"  WARNING: coord mismatch max={coord_diff:.2e} in {label}")

    out = {"label": label}
    for comp in ("h", "u", "v"):
        va, vb = a[comp], b[comp]
        diff = vb - va
        norm_a = float(np.linalg.norm(va))
        norm_b = float(np.linalg.norm(vb))
        norm_diff = float(np.linalg.norm(diff))
        # Cosine similarity
        denom = norm_a * norm_b
        cos_sim = float(np.dot(va, vb) / denom) if denom > 0 else 0.0
        max_abs = float(np.max(np.abs(diff))) if len(diff) > 0 else 0.0

        out[f"{comp}_norm_a"] = norm_a
        out[f"{comp}_norm_b"] = norm_b
        out[f"{comp}_diff_norm"] = norm_diff
        out[f"{comp}_rel_diff"] = norm_diff / max(norm_a, 1e-30)
        out[f"{comp}_cos_sim"] = cos_sim
        out[f"{comp}_max_abs_diff"] = max_abs
    # Total
    va = np.concatenate([a["h"], a["u"], a["v"]])
    vb = np.concatenate([b["h"], b["u"], b["v"]])
    out["total_norm_a"] = float(np.linalg.norm(va))
    out["total_norm_b"] = float(np.linalg.norm(vb))
    out["total_diff_norm"] = float(np.linalg.norm(vb - va))
    out["total_rel_diff"] = out["total_diff_norm"] / max(out["total_norm_a"], 1e-30)
    denom = out["total_norm_a"] * out["total_norm_b"]
    out["total_cos_sim"] = float(np.dot(va, vb) / denom) if denom > 0 else 0.0
    return out


def format_row(d):
    return (
        f"  label={d['label']:40s}\n"
        f"    TOTAL: ||a||={d['total_norm_a']:10.4f}  ||b||={d['total_norm_b']:10.4f}  "
        f"rel_diff={d['total_rel_diff']:.4e}  cos_sim={d['total_cos_sim']:.6f}\n"
        f"    h:     ||a||={d['h_norm_a']:10.4f}  rel_diff={d['h_rel_diff']:.4e}  "
        f"cos={d['h_cos_sim']:.6f}  max_diff={d['h_max_abs_diff']:.4e}\n"
        f"    u:     ||a||={d['u_norm_a']:10.4f}  rel_diff={d['u_rel_diff']:.4e}  "
        f"cos={d['u_cos_sim']:.6f}  max_diff={d['u_max_abs_diff']:.4e}\n"
        f"    v:     ||a||={d['v_norm_a']:10.4f}  rel_diff={d['v_rel_diff']:.4e}  "
        f"cos={d['v_cos_sim']:.6f}  max_diff={d['v_max_abs_diff']:.4e}"
    )


def main():
    # Discover what's been saved.
    # Filename format: vec_{tag}_size{N}_{kind}.npz  where kind is lambda0 or grad_smoothed
    import re
    available = {}
    pattern = re.compile(r"vec_(.+)_size(\d+)_(lambda0|grad_smoothed)$")
    for f in glob(str(RESULTS_DIR / "vec_*.npz")):
        name = Path(f).stem
        m = pattern.match(name)
        if not m:
            continue
        tag, size, kind = m.group(1), int(m.group(2)), m.group(3)
        available.setdefault((tag, size), set()).add(kind)

    print("Available runs:")
    for (tag, size), kinds in sorted(available.items()):
        print(f"  tag={tag} size={size} kinds={sorted(kinds)}")

    print()
    print("=" * 80)
    print("ADJOINT λ₀ PAIRWISE COMPARISON")
    print("=" * 80)
    # Load all λ₀ vectors
    vecs = {}
    for (tag, size), kinds in available.items():
        if "lambda0" in kinds:
            v = load_vector(tag, size, "lambda0")
            if v is not None:
                vecs[(tag, size)] = v

    # Pairwise compare
    keys = sorted(vecs.keys())
    for i, ka in enumerate(keys):
        for kb in keys[i+1:]:
            r = compare(vecs[ka], vecs[kb], f"{ka[0]}(size={ka[1]}) vs {kb[0]}(size={kb[1]})")
            if r:
                print(format_row(r))
                print()

    # Same for post-smoother gradient
    print("=" * 80)
    print("POST-SMOOTHER GRADIENT PAIRWISE COMPARISON")
    print("=" * 80)
    vecs = {}
    for (tag, size), kinds in available.items():
        if "grad_smoothed" in kinds:
            v = load_vector(tag, size, "grad_smoothed")
            if v is not None:
                vecs[(tag, size)] = v

    keys = sorted(vecs.keys())
    for i, ka in enumerate(keys):
        for kb in keys[i+1:]:
            r = compare(vecs[ka], vecs[kb], f"{ka[0]}(size={ka[1]}) vs {kb[0]}(size={kb[1]})")
            if r:
                print(format_row(r))
                print()


if __name__ == "__main__":
    main()

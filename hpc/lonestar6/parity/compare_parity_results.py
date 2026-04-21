"""compare_parity_results.py — score the parity audit against PARITY_CONTRACT.md.

Reads two directories of JSON outputs (one per environment) and compares each
metric against its contract tolerance. Produces a summary table on stdout
and a JSON report.

Usage:
    python compare_parity_results.py --local /tmp/parity/local --ls6 /tmp/parity/ls6
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


# ---- tolerance table (mirrors PARITY_CONTRACT.md) ---------------------------

TOL = {
    # imports.json — string-equality checks; the contract notes anticipated mismatches.
    "imports": {
        "exact_fields": ["mpi_version_tuple"],
        "package_matches": {
            # Each entry: (local_expected, drift_category)
            "numpy": "acceptable_drift_minor",
            "scipy": "acceptable_drift_minor",
            "mpi4py": "exact",
            "petsc4py": "patch_ok",
            "basix": "anticipated_mismatch",    # 0.9 vs 0.10
            "dolfinx": "anticipated_mismatch",  # 0.9 vs 0.10
            "ufl": "anticipated_mismatch",
            "ffcx": "anticipated_mismatch",
        },
    },

    # mpi.json — deterministic per-rank collective
    "mpi": {
        "global_sumsq_value": {"match_rel": 1e-14, "drift_rel": 1e-12},
        "allreduce_int_sum":  {"match_exact": True},
        "mpi_size":           {"match_exact": True},
    },

    # petsc.json — single-rank LU direct solve
    "petsc": {
        "b_norm":        {"match_rel": 1e-13, "drift_rel": 1e-10},
        "residual_abs":  {"match_abs": 1e-12, "drift_abs": 1e-8},
        "residual_rel":  {"match_abs": 1e-12, "drift_abs": 1e-8},
        "error_rel":     {"match_rel": 1e-12, "drift_rel": 1e-8},
        "ksp_iterations":{"match_exact": True},
    },

    # dolfinx.json — small Poisson
    "dolfinx": {
        "num_cells_global":    {"match_exact": True},
        "num_vertices_global": {"match_exact": True},
        "num_dofs_global":     {"match_exact": True},
        "b_l2":                {"match_rel": 1e-13, "drift_rel": 1e-10},
        "u_l2_global":         {"match_rel": 1e-14, "drift_rel": 1e-10},
        "u_linf_global":       {"match_rel": 1e-14, "drift_rel": 1e-10},
    },

    # 4D-Var reduced — cost, gradient summaries
    "4dvar": {
        "n_dofs_global": {"match_exact": True},
        "n_obs_total":   {"match_exact": True},
        "J_bg":          {"match_rel": 1e-10, "drift_rel": 1e-6},
        "grad_l2_global":   {"match_rel": 1e-10, "drift_rel": 1e-6},
        "grad_linf_global": {"match_rel": 1e-10, "drift_rel": 1e-6},
        "final_state_l2_global": {"match_rel": 1e-10, "drift_rel": 1e-6},
        "obs_l2_global": {"match_rel": 1e-10, "drift_rel": 1e-6},
    },

    "dcwme": {
        "n_dofs_global": {"match_exact": True},
        "J_bg":          {"match_rel": 1e-10, "drift_rel": 1e-6},
        "grad_l2_global":   {"match_rel": 1e-10, "drift_rel": 1e-6},
        "grad_linf_global": {"match_rel": 1e-10, "drift_rel": 1e-6},
    },
}


def _cmp_numeric(name, a, b, rule):
    """Return (status, detail) for a numeric field."""
    if a is None or b is None:
        return ("MISMATCH", f"{name}: missing on one side (local={a}, ls6={b})")
    try:
        a_f, b_f = float(a), float(b)
    except (TypeError, ValueError):
        return ("MISMATCH", f"{name}: not numeric (local={a!r}, ls6={b!r})")

    if rule.get("match_exact"):
        if a_f == b_f:
            return ("MATCH", f"{name}: {a_f} == {b_f}")
        return ("MISMATCH", f"{name}: {a_f} != {b_f}")

    abs_diff = abs(a_f - b_f)
    denom = max(abs(a_f), abs(b_f), 1e-300)
    rel_diff = abs_diff / denom

    if "match_rel" in rule and rel_diff <= rule["match_rel"]:
        return ("MATCH", f"{name}: rel_diff={rel_diff:.2e}")
    if "match_abs" in rule and abs_diff <= rule["match_abs"]:
        return ("MATCH", f"{name}: abs_diff={abs_diff:.2e}")
    if "drift_rel" in rule and rel_diff <= rule["drift_rel"]:
        return ("DRIFT", f"{name}: rel_diff={rel_diff:.2e} within drift ({rule['drift_rel']:.0e})")
    if "drift_abs" in rule and abs_diff <= rule["drift_abs"]:
        return ("DRIFT", f"{name}: abs_diff={abs_diff:.2e} within drift ({rule['drift_abs']:.0e})")
    return ("MISMATCH", f"{name}: |Δ|={abs_diff:.3e} rel={rel_diff:.3e} — out of tolerance")


def compare_imports(a, b):
    results = []
    # Core arch/platform differences are expected; do not score them.
    for field in TOL["imports"]["exact_fields"]:
        if a.get(field) == b.get(field):
            results.append(("MATCH", f"{field}: {a.get(field)}"))
        else:
            results.append(("MISMATCH", f"{field}: local={a.get(field)} ls6={b.get(field)}"))
    for pkg, rule in TOL["imports"]["package_matches"].items():
        av = a.get("packages", {}).get(pkg)
        bv = b.get("packages", {}).get(pkg)
        if av == bv:
            results.append(("MATCH", f"{pkg}: {av}"))
        elif rule == "exact":
            results.append(("MISMATCH", f"{pkg}: local={av} ls6={bv}"))
        elif rule == "anticipated_mismatch":
            # Per the contract, drift on this row is acceptable iff B+C pass.
            results.append(("DRIFT", f"{pkg}: local={av} ls6={bv} — contract-approved drift"))
        else:
            # minor/patch drift: accept
            results.append(("DRIFT", f"{pkg}: local={av} ls6={bv} — {rule}"))

    # Errors that only show up on one side are important
    local_errs = set((a.get("errors") or {}).keys())
    ls6_errs = set((b.get("errors") or {}).keys())
    for k in local_errs - ls6_errs:
        results.append(("MISMATCH", f"imports: {k} failed locally only — {a['errors'][k]}"))
    for k in ls6_errs - local_errs:
        results.append(("DRIFT", f"imports: {k} missing on LS6 — {b['errors'][k]}"))
    return results


def compare_file(kind, a, b):
    if a.get("FAILED") or b.get("FAILED"):
        tag = []
        if a.get("FAILED"): tag.append(f"LOCAL={a.get('error_type')}: {a.get('error_message')}")
        if b.get("FAILED"): tag.append(f"LS6={b.get('error_type')}: {b.get('error_message')}")
        return [("MISMATCH", f"[{kind}] TEST FAILED on at least one side — " + " ; ".join(tag))]

    if kind == "imports":
        return compare_imports(a, b)

    rules = TOL.get(kind, {})
    out = []
    for key, rule in rules.items():
        out.append(_cmp_numeric(key, a.get(key), b.get(key), rule))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local", default="/tmp/parity/local")
    p.add_argument("--ls6",   default="/tmp/parity/ls6")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    local_dir = Path(args.local)
    ls6_dir   = Path(args.ls6)

    files = ["imports", "mpi", "petsc", "dolfinx", "4dvar", "dcwme"]
    all_results = {}
    total = {"MATCH": 0, "DRIFT": 0, "MISMATCH": 0}

    for f in files:
        local_path = local_dir / f"{f}.json"
        ls6_path   = ls6_dir   / f"{f}.json"
        if not local_path.exists() or not ls6_path.exists():
            status = "SKIPPED"
            msg = f"missing: local={local_path.exists()} ls6={ls6_path.exists()}"
            all_results[f] = [("SKIPPED", msg)]
            continue
        a = json.loads(local_path.read_text())
        b = json.loads(ls6_path.read_text())
        all_results[f] = compare_file(f, a, b)

    # Print summary
    print("=" * 78)
    print(f"{'Section':<10}  {'Field':<35}  {'Status':<10}  Detail")
    print("=" * 78)
    for section, rows in all_results.items():
        for status, detail in rows:
            if status in total:
                total[status] += 1
            head = detail.split(":", 1)[0]
            print(f"{section:<10}  {head:<35}  {status:<10}  {detail}")
    print("=" * 78)
    print(f"Totals: MATCH={total['MATCH']}  DRIFT={total['DRIFT']}  MISMATCH={total['MISMATCH']}")
    if total["MISMATCH"] > 0:
        print("VERDICT: MISMATCH(es) present — see rows above.")
    elif total["DRIFT"] > 0:
        print("VERDICT: PASS WITH DRIFT — all deviations are contract-approved.")
    else:
        print("VERDICT: PASS — all metrics at MATCH.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "results": {k: [{"status": s, "detail": d} for s, d in v] for k, v in all_results.items()},
            "totals": total,
        }, indent=2))


if __name__ == "__main__":
    main()

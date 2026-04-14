#!/usr/bin/env python3
"""
Hostile Validation Audit of Manning's-n Identifiability Study
=============================================================

Attempts to break the conclusions of identifiability_audit.py:
  - σ = [0.579, 0.031, 0.005], effective rank 1
  - all modes >95% collinear in observation space
  - no sign ambiguity (all linear)

Attack surfaces:
  1. FD epsilon sensitivity (does the spectrum change with ε?)
  2. Clipping contamination check
  3. Whitening correctness (unwhitened vs whitened)
  4. Column scaling effects
  5. Sign-ambiguity via cost comparison (not just trajectory)
  6. Reproducibility check (different state reset)

Usage:
  python experiments/identifiability_hostile_audit.py --kl-modes 3
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.identifiability_audit import (
    AuditConfig,
    _to_ladder_config,
    build_parameter_sensitivity_matrix,
    evaluate_observation_map,
    whiten_sensitivity,
)
from experiments.validation_ladder import (
    build_problem_and_solver,
    create_observations,
    generate_truth_trajectory,
)


def _svd_summary(J, label=""):
    """Compute and print SVD summary."""
    U, sigma, Vt = np.linalg.svd(J, full_matrices=False)
    cond = sigma[0] / max(sigma[-1], 1e-30)
    energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    rank_5 = int(np.sum(sigma > 0.05 * sigma[0]))
    rank_10 = int(np.sum(sigma > 0.10 * sigma[0]))
    print(f"  [{label}] σ = {sigma}")
    print(f"  [{label}] cond = {cond:.2e}, rank@5% = {rank_5}, rank@10% = {rank_10}")
    print(f"  [{label}] energy = {energy}")
    return sigma, Vt.T


# =========================================================================
# ATTACK 1: FD epsilon sensitivity
# =========================================================================

def attack_fd_epsilon(*, solver, problem, controller, solver_params,
                      truth_state, obs_bundle, cfg):
    """Vary ε over 2 orders of magnitude and check spectral stability."""
    print("\n" + "="*70)
    print("ATTACK 1: FD EPSILON SENSITIVITY")
    print("="*70)

    epsilons = [0.001, 0.005, 0.01, 0.02, 0.05]
    all_sigma = {}

    for eps in epsilons:
        print(f"\n  --- ε = {eps} ---")
        cfg_eps = AuditConfig(
            kl_n_modes=cfg.kl_n_modes,
            kl_correlation_length=cfg.kl_correlation_length,
            fd_epsilon=eps,
        )
        J_theta, _ = build_parameter_sensitivity_matrix(
            solver=solver, problem=problem, controller=controller,
            solver_params=solver_params, truth_state=truth_state,
            obs_operator=obs_bundle["obs_operator"],
            obs_times=obs_bundle["obs_times"], cfg=cfg_eps,
        )
        J_w = whiten_sensitivity(
            J_theta, obs_bundle["noise_stds"],
            n_obs_points=obs_bundle["obs_points"].shape[0],
        )
        sigma, _ = _svd_summary(J_w, f"ε={eps}")
        all_sigma[eps] = sigma.tolist()

    # Check stability: max relative change in leading singular value
    ref_sigma = np.array(all_sigma[0.01])
    print(f"\n  --- Stability summary (reference: ε=0.01) ---")
    for eps, sig in all_sigma.items():
        sig = np.array(sig)
        rel_change = np.abs(sig - ref_sigma) / np.maximum(ref_sigma, 1e-30)
        print(f"  ε={eps}: rel_change = {rel_change}, max = {np.max(rel_change):.4e}")

    max_change = max(
        np.max(np.abs(np.array(s) - ref_sigma) / np.maximum(ref_sigma, 1e-30))
        for eps, s in all_sigma.items() if eps != 0.01
    )
    stable = max_change < 0.10
    print(f"\n  VERDICT: {'STABLE' if stable else 'UNSTABLE'} (max relative change = {max_change:.4e})")
    return {"all_sigma": all_sigma, "stable": stable, "max_relative_change": float(max_change)}


# =========================================================================
# ATTACK 2: Clipping contamination
# =========================================================================

def attack_clipping(*, controller, cfg):
    """Check whether any FD perturbation enters clipping."""
    print("\n" + "="*70)
    print("ATTACK 2: CLIPPING CONTAMINATION CHECK")
    print("="*70)

    n_params = cfg.n_params
    eps = cfg.fd_epsilon
    n_min, n_max = cfg.n_bounds

    any_clipped = False
    for p in range(n_params):
        for sign, label in [(+1, "+ε"), (-1, "-ε")]:
            theta = np.zeros(n_params)
            theta[p] = sign * eps
            field = controller.evaluate_field(theta)
            n_clipped = np.sum((field <= n_min + 1e-10) | (field >= n_max - 1e-10))
            pct = 100 * n_clipped / field.size
            if n_clipped > 0:
                any_clipped = True
            print(f"  Mode {p} {label}: field range [{field.min():.6f}, {field.max():.6f}], "
                  f"clipped={n_clipped}/{field.size} ({pct:.1f}%)")

    # Also check at larger perturbations
    for eps_large in [0.05, 0.10, 0.20]:
        for p in range(n_params):
            theta = np.zeros(n_params)
            theta[p] = eps_large
            field = controller.evaluate_field(theta)
            n_clipped = np.sum((field <= n_min + 1e-10) | (field >= n_max - 1e-10))
            if n_clipped > 0:
                print(f"  Mode {p} ε={eps_large}: CLIPPING ACTIVE — {n_clipped} DOFs")

    print(f"\n  VERDICT: {'CLIPPING ACTIVE — results may be contaminated' if any_clipped else 'NO CLIPPING at ε=' + str(eps) + ' — results clean'}")
    return {"clipping_at_fd_epsilon": any_clipped}


# =========================================================================
# ATTACK 3: Whitening correctness
# =========================================================================

def attack_whitening(*, J_theta, obs_bundle, cfg):
    """Compare unwhitened vs whitened spectra."""
    print("\n" + "="*70)
    print("ATTACK 3: WHITENING CORRECTNESS")
    print("="*70)

    n_obs_points = obs_bundle["obs_points"].shape[0]

    # Unwhitened
    sigma_raw, V_raw = _svd_summary(J_theta, "unwhitened")

    # Whitened (as in original audit)
    J_w = whiten_sensitivity(J_theta, obs_bundle["noise_stds"], n_obs_points)
    sigma_w, V_w = _svd_summary(J_w, "whitened")

    # Check: does whitening change the rank conclusion?
    rank_raw = int(np.sum(sigma_raw > 0.05 * sigma_raw[0]))
    rank_w = int(np.sum(sigma_w > 0.05 * sigma_w[0]))

    print(f"\n  Unwhitened rank@5%: {rank_raw}")
    print(f"  Whitened rank@5%:   {rank_w}")
    print(f"  Rank changed: {'YES — whitening matters!' if rank_raw != rank_w else 'NO — same rank'}")

    # Check if leading right singular vectors agree
    for i in range(min(len(sigma_raw), len(sigma_w))):
        cos = abs(np.dot(V_raw[:, i], V_w[:, i]))
        print(f"  v_{i} alignment (raw vs whitened): cos = {cos:.6f}")

    return {
        "sigma_raw": sigma_raw.tolist(),
        "sigma_whitened": sigma_w.tolist(),
        "rank_raw": rank_raw,
        "rank_whitened": rank_w,
    }


# =========================================================================
# ATTACK 4: Column scaling effects
# =========================================================================

def attack_column_scaling(*, J_theta, obs_bundle, cfg):
    """Rescale J_θ columns by parameter effect magnitude and re-SVD."""
    print("\n" + "="*70)
    print("ATTACK 4: COLUMN SCALING EFFECTS")
    print("="*70)

    n_obs_points = obs_bundle["obs_points"].shape[0]
    J_w = whiten_sensitivity(J_theta, obs_bundle["noise_stds"], n_obs_points)

    # Original
    sigma_orig, _ = _svd_summary(J_w, "original")

    # Scale columns to unit norm
    col_norms = np.linalg.norm(J_w, axis=0)
    J_normalized = J_w / col_norms[np.newaxis, :]
    sigma_norm, _ = _svd_summary(J_normalized, "col-normalized")

    print(f"\n  Column norms before normalization: {col_norms}")
    print(f"  Column norm ratio (max/min): {col_norms.max() / max(col_norms.min(), 1e-30):.2f}")

    # If normalization dramatically changes the condition number, the original
    # conclusion depends on the arbitrary amplitude of KL modes
    cond_orig = sigma_orig[0] / max(sigma_orig[-1], 1e-30)
    cond_norm = sigma_norm[0] / max(sigma_norm[-1], 1e-30)
    print(f"\n  Condition number (original): {cond_orig:.2e}")
    print(f"  Condition number (normalized): {cond_norm:.2e}")

    scaling_matters = abs(cond_norm - cond_orig) / cond_orig > 0.5
    print(f"  VERDICT: {'SCALING MATTERS — conclusions depend on basis amplitude' if scaling_matters else 'SCALING DOES NOT MATERIALLY CHANGE rank conclusions'}")

    return {
        "col_norms": col_norms.tolist(),
        "cond_original": float(cond_orig),
        "cond_normalized": float(cond_norm),
        "sigma_normalized": sigma_norm.tolist(),
    }


# =========================================================================
# ATTACK 5: Sign ambiguity via cost comparison
# =========================================================================

def attack_sign_via_cost(*, solver, problem, controller, solver_params,
                         truth_state, obs_bundle, cfg):
    """Compare J(+θ_truth) vs J(-θ_truth) to check if costs are distinguishable."""
    print("\n" + "="*70)
    print("ATTACK 5: SIGN AMBIGUITY VIA COST COMPARISON")
    print("="*70)

    n_params = cfg.n_params
    obs_times = obs_bundle["obs_times"]
    n_obs_points = obs_bundle["obs_points"].shape[0]

    fwd_kwargs = dict(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params, truth_state=truth_state,
        obs_operator=obs_bundle["obs_operator"],
        obs_times=obs_times, cfg=cfg,
    )

    # Reference (θ=0)
    G_ref = evaluate_observation_map(np.zeros(n_params), **fwd_kwargs)

    # Truth observations
    truth_obs_flat = obs_bundle["truth_obs"].ravel()
    noisy_obs_flat = obs_bundle["noisy_obs"].ravel()

    # Build R^{-1} diagonal for cost computation
    R_inv = np.zeros(len(obs_times) * n_obs_points)
    for k, sigma in enumerate(obs_bundle["noise_stds"]):
        start = k * n_obs_points
        end = (k + 1) * n_obs_points
        R_inv[start:end] = 1.0 / max(sigma**2, 1e-30)

    # Compute observation cost for various θ
    test_thetas = {}
    truth_coeffs = np.array([0.30] + [-0.25] * (n_params - 1))
    test_thetas["truth"] = truth_coeffs
    test_thetas["neg_truth"] = -truth_coeffs
    test_thetas["zero"] = np.zeros(n_params)

    # Also test individual modes
    for p in range(n_params):
        e_p = np.zeros(n_params)
        e_p[p] = 0.10
        test_thetas[f"+0.10*e_{p}"] = e_p.copy()
        test_thetas[f"-0.10*e_{p}"] = -e_p.copy()

    results = {}
    for label, theta in test_thetas.items():
        G = evaluate_observation_map(theta, **fwd_kwargs)
        residual = G - noisy_obs_flat
        cost_obs = 0.5 * np.sum(R_inv * residual**2)
        print(f"  θ={label:20s}: J_obs = {cost_obs:.6f}")
        results[label] = float(cost_obs)

    # Key comparison: J(+truth) vs J(-truth)
    if "truth" in results and "neg_truth" in results:
        j_plus = results["truth"]
        j_minus = results["neg_truth"]
        j_zero = results["zero"]
        print(f"\n  J(+θ_truth) = {j_plus:.6f}")
        print(f"  J(-θ_truth) = {j_minus:.6f}")
        print(f"  J(0)        = {j_zero:.6f}")
        print(f"  |J(+) - J(-)| / J(0) = {abs(j_plus - j_minus) / max(j_zero, 1e-30):.6e}")
        distinguishable = abs(j_plus - j_minus) / max(j_zero, 1e-30) > 0.01
        print(f"  VERDICT: {'COSTS DISTINGUISH ±θ — sign ambiguity is NOT the issue' if distinguishable else 'COSTS NEARLY IDENTICAL — genuine sign ambiguity'}")

    # Per-mode sign comparison
    print(f"\n  --- Per-mode cost asymmetry ---")
    for p in range(n_params):
        jp = results.get(f"+0.10*e_{p}", 0)
        jm = results.get(f"-0.10*e_{p}", 0)
        j0 = results["zero"]
        asym = abs(jp - jm) / max(abs(jp - j0) + abs(jm - j0), 1e-30)
        print(f"  Mode {p}: J(+) = {jp:.6f}, J(-) = {jm:.6f}, "
              f"asymmetry = {asym:.4f} (0=symmetric, 1=fully asymmetric)")

    return results


# =========================================================================
# ATTACK 6: Reproducibility
# =========================================================================

def attack_reproducibility(*, solver, problem, controller, solver_params,
                           truth_state, obs_bundle, cfg):
    """Run the same forward evaluation twice and check bitwise agreement."""
    print("\n" + "="*70)
    print("ATTACK 6: REPRODUCIBILITY CHECK")
    print("="*70)

    n_params = cfg.n_params
    theta_test = np.array([0.01, -0.005, 0.003])[:n_params]

    fwd_kwargs = dict(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params, truth_state=truth_state,
        obs_operator=obs_bundle["obs_operator"],
        obs_times=obs_bundle["obs_times"], cfg=cfg,
    )

    G1 = evaluate_observation_map(theta_test, **fwd_kwargs)
    G2 = evaluate_observation_map(theta_test, **fwd_kwargs)

    max_diff = np.max(np.abs(G1 - G2))
    rel_diff = max_diff / max(np.max(np.abs(G1)), 1e-30)
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative difference: {rel_diff:.2e}")
    print(f"  VERDICT: {'REPRODUCIBLE' if rel_diff < 1e-10 else 'NOT REPRODUCIBLE — stale state?'}")

    return {"max_abs_diff": float(max_diff), "max_rel_diff": float(rel_diff)}


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Hostile audit of identifiability study")
    parser.add_argument("--kl-modes", type=int, default=3)
    parser.add_argument("--kl-length", type=float, default=None)
    parser.add_argument("--skip-epsilon", action="store_true", help="Skip expensive epsilon sweep")
    args = parser.parse_args()

    cfg = AuditConfig(
        kl_n_modes=args.kl_modes,
        kl_correlation_length=args.kl_length,
        fd_epsilon=0.01,
    )
    output_dir = PROJECT_ROOT / "results" / "identifiability_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("HOSTILE VALIDATION AUDIT — IDENTIFIABILITY STUDY")
    print("="*70)

    t0 = time.time()
    ladder_cfg = _to_ladder_config(cfg)
    problem, solver, controller, solver_params, forcing = build_problem_and_solver(ladder_cfg)
    truth_state, theta_truth = generate_truth_trajectory(
        ladder_cfg, problem, solver, controller, solver_params,
    )
    obs_bundle = create_observations(
        ladder_cfg, solver, truth_state, controller, solver_params, problem,
    )
    print(f"  Setup time: {time.time() - t0:.1f}s")

    all_results = {}

    # Attack 6 first (cheapest, validates infrastructure)
    all_results["reproducibility"] = attack_reproducibility(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params, truth_state=truth_state,
        obs_bundle=obs_bundle, cfg=cfg,
    )

    # Attack 2: Clipping (no forward solves needed)
    all_results["clipping"] = attack_clipping(controller=controller, cfg=cfg)

    # Build baseline J_θ for attacks 3 and 4
    print("\n  Building baseline J_θ (ε=0.01)...")
    J_theta_base, G_ref = build_parameter_sensitivity_matrix(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params, truth_state=truth_state,
        obs_operator=obs_bundle["obs_operator"],
        obs_times=obs_bundle["obs_times"], cfg=cfg,
    )

    # Attack 3: Whitening
    all_results["whitening"] = attack_whitening(
        J_theta=J_theta_base, obs_bundle=obs_bundle, cfg=cfg,
    )

    # Attack 4: Column scaling
    all_results["column_scaling"] = attack_column_scaling(
        J_theta=J_theta_base, obs_bundle=obs_bundle, cfg=cfg,
    )

    # Attack 5: Sign ambiguity via cost
    all_results["sign_via_cost"] = attack_sign_via_cost(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params, truth_state=truth_state,
        obs_bundle=obs_bundle, cfg=cfg,
    )

    # Attack 1: Epsilon sensitivity (most expensive — many forward solves)
    if not args.skip_epsilon:
        all_results["fd_epsilon"] = attack_fd_epsilon(
            solver=solver, problem=problem, controller=controller,
            solver_params=solver_params, truth_state=truth_state,
            obs_bundle=obs_bundle, cfg=cfg,
        )

    # Final summary
    print("\n" + "="*70)
    print("HOSTILE AUDIT SUMMARY")
    print("="*70)

    attacks_that_broke = []
    attacks_that_held = []

    # Reproducibility
    if all_results["reproducibility"]["max_rel_diff"] < 1e-10:
        attacks_that_held.append("Reproducibility: forward map is deterministic")
    else:
        attacks_that_broke.append("Reproducibility: stale state detected")

    # Clipping
    if not all_results["clipping"]["clipping_at_fd_epsilon"]:
        attacks_that_held.append("Clipping: no contamination at FD epsilon")
    else:
        attacks_that_broke.append("Clipping: active at FD epsilon, may invalidate linearization")

    # Whitening
    wr = all_results["whitening"]
    if wr["rank_raw"] == wr["rank_whitened"]:
        attacks_that_held.append(f"Whitening: rank unchanged (raw={wr['rank_raw']}, whitened={wr['rank_whitened']})")
    else:
        attacks_that_broke.append(f"Whitening: rank changed! raw={wr['rank_raw']} vs whitened={wr['rank_whitened']}")

    # Column scaling
    cs = all_results["column_scaling"]
    if abs(cs["cond_normalized"] - cs["cond_original"]) / cs["cond_original"] < 0.5:
        attacks_that_held.append(f"Column scaling: cond number stable ({cs['cond_original']:.0f} → {cs['cond_normalized']:.0f})")
    else:
        attacks_that_broke.append(f"Column scaling: cond number changed dramatically ({cs['cond_original']:.0f} → {cs['cond_normalized']:.0f})")

    # Epsilon sensitivity
    if "fd_epsilon" in all_results:
        if all_results["fd_epsilon"]["stable"]:
            attacks_that_held.append(f"FD epsilon: spectrum stable across ε range (max change={all_results['fd_epsilon']['max_relative_change']:.2e})")
        else:
            attacks_that_broke.append(f"FD epsilon: spectrum UNSTABLE (max change={all_results['fd_epsilon']['max_relative_change']:.2e})")

    print("\n  ATTACKS THAT HELD (original conclusion survived):")
    for a in attacks_that_held:
        print(f"    ✓ {a}")

    print("\n  ATTACKS THAT BROKE (original conclusion weakened/refuted):")
    if attacks_that_broke:
        for a in attacks_that_broke:
            print(f"    ✗ {a}")
    else:
        print("    (none)")

    # Verdict
    if len(attacks_that_broke) == 0:
        verdict = "CONFIRMED"
        print(f"\n  VERDICT: {verdict} — original non-identifiability conclusion is robust")
    elif any("rank changed" in a or "UNSTABLE" in a for a in attacks_that_broke):
        verdict = "WEAKENED"
        print(f"\n  VERDICT: {verdict} — conclusion holds but has caveats")
    else:
        verdict = "CONFIRMED WITH MINOR CAVEATS"
        print(f"\n  VERDICT: {verdict}")

    all_results["verdict"] = verdict
    all_results["attacks_held"] = attacks_that_held
    all_results["attacks_broke"] = attacks_that_broke
    all_results["total_time_s"] = time.time() - t0

    results_file = output_dir / "hostile_audit_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    raise SystemExit(main())

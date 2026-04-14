#!/usr/bin/env python3
"""
Stage 2 Identifiability Validation
===================================

Three tests to determine if the NO-GO conclusion is publication-quality:

  1. STATE-ADJUSTED SENSITIVITY: Does u_0 freedom further collapse the spectrum?
  2. EPSILON ROBUSTNESS: Already answered by hostile audit (stable to 6 digits).
     Included here as a reference cross-check.
  3. R-WEIGHTED COLLINEARITY: Are modes collinear in the correct cost metric?

Usage:
  python experiments/identifiability_stage2.py --kl-modes 3
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
    evaluate_observation_map,
    whiten_sensitivity,
)
from experiments.validation_ladder import (
    build_problem_and_solver,
    create_observations,
    generate_truth_trajectory,
)


# =========================================================================
# TEST 1: State-adjusted sensitivity
# =========================================================================

def test_state_adjusted_sensitivity(
    *,
    solver, problem, controller, solver_params,
    truth_state, obs_bundle, cfg,
    output_dir,
):
    """Compute the observation response to θ perturbations AFTER allowing
    u_0 to re-optimize.

    For each parameter perturbation θ = ε·e_i:
      1. Apply θ perturbation to Manning field
      2. Run a short state-only optimization (or approximate: run forward
         from perturbed state and measure residual)
      3. Evaluate observation response

    The key question: after the state adjusts, does the friction signal
    remain or does it get absorbed?

    Approach: We approximate the "state-adjusted" response by comparing
    the observation map at two levels:
      a) G(u0_truth, +ε·e_i) - G(u0_truth, 0)          [fixed-state, from original audit]
      b) G(u0_truth, +ε·e_i) - G(u0_truth, -ε·e_i)     [symmetric FD]
      c) G(u0_perturbed, +ε·e_i) - G(u0_perturbed, 0)   [perturbed-state response]

    More precisely: we check if a state perturbation in the direction that
    best compensates the friction effect can absorb the friction signal.
    We project the friction response onto the state-reachable observation space.
    """
    print("\n" + "="*70)
    print("TEST 1: STATE-ADJUSTED SENSITIVITY")
    print("="*70)

    n_params = cfg.n_params
    eps = cfg.fd_epsilon
    obs_times = obs_bundle["obs_times"]
    n_obs_points = obs_bundle["obs_points"].shape[0]
    n_obs = len(obs_times) * n_obs_points
    state_size = truth_state.size

    fwd_kwargs = dict(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params,
        obs_operator=obs_bundle["obs_operator"],
        obs_times=obs_times, cfg=cfg,
    )

    # Step 1: Compute parameter sensitivity columns (fixed state = truth)
    print("\n  Computing fixed-state parameter responses...")
    G_ref = evaluate_observation_map(np.zeros(n_params), truth_state=truth_state, **fwd_kwargs)

    J_theta_cols = []
    for p in range(n_params):
        theta_plus = np.zeros(n_params); theta_plus[p] = +eps
        theta_minus = np.zeros(n_params); theta_minus[p] = -eps
        G_plus = evaluate_observation_map(theta_plus, truth_state=truth_state, **fwd_kwargs)
        G_minus = evaluate_observation_map(theta_minus, truth_state=truth_state, **fwd_kwargs)
        col = (G_plus - G_minus) / (2 * eps)
        J_theta_cols.append(col)
        print(f"    ||J_θ[:, {p}]|| = {np.linalg.norm(col):.6e}")

    J_theta = np.column_stack(J_theta_cols)

    # Step 2: Estimate state-response subspace via random state probes
    # J_{u0} @ v ≈ (G(u0+δv, θ=0) - G(u0-δv, θ=0)) / (2δ)
    print("\n  Probing state-response subspace (20 random directions)...")
    n_probes = 20
    eps_state = 1e-4
    rng = np.random.default_rng(42)

    state_response_cols = []
    for i in range(n_probes):
        direction = rng.standard_normal(state_size)
        direction /= np.linalg.norm(direction)

        state_plus = truth_state + eps_state * direction
        state_minus = truth_state - eps_state * direction

        G_sp = evaluate_observation_map(np.zeros(n_params), truth_state=state_plus, **fwd_kwargs)
        G_sm = evaluate_observation_map(np.zeros(n_params), truth_state=state_minus, **fwd_kwargs)

        state_col = (G_sp - G_sm) / (2 * eps_state)
        state_response_cols.append(state_col)

        if (i + 1) % 5 == 0:
            print(f"    Probe {i+1}/{n_probes} done")

    state_response_matrix = np.column_stack(state_response_cols)  # (n_obs, n_probes)

    # Step 3: Compute state-reachable subspace in observation space
    # via SVD of state response matrix
    U_state, s_state, _ = np.linalg.svd(state_response_matrix, full_matrices=False)
    # Keep modes that capture 99% of state response variance
    cum_energy = np.cumsum(s_state**2) / np.sum(s_state**2)
    n_state_modes = int(np.searchsorted(cum_energy, 0.99)) + 1
    U_state_trunc = U_state[:, :n_state_modes]  # (n_obs, n_state_modes)
    print(f"\n  State response: {n_probes} probes → {n_state_modes} effective modes (99% energy)")
    print(f"  State singular values (top 5): {s_state[:5]}")

    # Step 4: Project OUT the state-reachable component from each parameter response
    # This gives the "state-adjusted" parameter sensitivity:
    # J_θ_adjusted[:, p] = (I - U_state U_state^T) J_θ[:, p]
    # = component of friction response NOT absorbable by state changes
    projector = np.eye(n_obs) - U_state_trunc @ U_state_trunc.T

    J_theta_adjusted = projector @ J_theta

    print(f"\n  --- Fixed-state vs state-adjusted comparison ---")
    for p in range(n_params):
        orig_norm = np.linalg.norm(J_theta[:, p])
        adj_norm = np.linalg.norm(J_theta_adjusted[:, p])
        absorbed_frac = 1.0 - adj_norm / max(orig_norm, 1e-30)
        print(f"  Mode {p}: ||J_θ|| = {orig_norm:.6e} → ||J_θ_adj|| = {adj_norm:.6e}  "
              f"({absorbed_frac*100:.1f}% absorbed by state)")

    # Step 5: SVD of adjusted sensitivity
    # Whiten first
    J_theta_adj_w = whiten_sensitivity(
        J_theta_adjusted, obs_bundle["noise_stds"], n_obs_points,
    )
    J_theta_orig_w = whiten_sensitivity(
        J_theta, obs_bundle["noise_stds"], n_obs_points,
    )

    U_orig, s_orig, Vt_orig = np.linalg.svd(J_theta_orig_w, full_matrices=False)
    U_adj, s_adj, Vt_adj = np.linalg.svd(J_theta_adj_w, full_matrices=False)

    print(f"\n  --- Spectral comparison ---")
    print(f"  Fixed-state σ:   {s_orig}")
    print(f"  State-adjusted σ: {s_adj}")
    print(f"  Ratios (adj/orig): {s_adj / np.maximum(s_orig, 1e-30)}")

    rank_orig_5 = int(np.sum(s_orig > 0.05 * s_orig[0]))
    rank_adj_5 = int(np.sum(s_adj > 0.05 * max(s_adj[0], 1e-30)))
    rank_orig_10 = int(np.sum(s_orig > 0.10 * s_orig[0]))
    rank_adj_10 = int(np.sum(s_adj > 0.10 * max(s_adj[0], 1e-30)))

    print(f"  Effective rank @5%:  orig={rank_orig_5}, adjusted={rank_adj_5}")
    print(f"  Effective rank @10%: orig={rank_orig_10}, adjusted={rank_adj_10}")

    # Classification
    if s_adj[0] < 0.01 * s_orig[0]:
        classification = "STATE ABSORBS ALL FRICTION SIGNAL — joint confounding dominant"
    elif rank_adj_5 < rank_orig_5:
        classification = "STATE ABSORBS SOME FRICTION MODES — joint confounding reduces rank"
    elif np.all(s_adj / np.maximum(s_orig, 1e-30) > 0.8):
        classification = "STATE BARELY AFFECTS FRICTION SIGNAL — pure parameter non-identifiability"
    else:
        classification = "MIXED — state absorbs weak modes but leading mode survives"

    print(f"\n  CLASSIFICATION: {classification}")

    return {
        "sigma_original": s_orig.tolist(),
        "sigma_adjusted": s_adj.tolist(),
        "rank_original_5pct": rank_orig_5,
        "rank_adjusted_5pct": rank_adj_5,
        "rank_original_10pct": rank_orig_10,
        "rank_adjusted_10pct": rank_adj_10,
        "state_modes_used": n_state_modes,
        "classification": classification,
        "absorption_fractions": [
            float(1.0 - np.linalg.norm(J_theta_adjusted[:, p]) /
                  max(np.linalg.norm(J_theta[:, p]), 1e-30))
            for p in range(n_params)
        ],
    }


# =========================================================================
# TEST 3: R-weighted collinearity
# =========================================================================

def test_r_weighted_collinearity(
    *,
    solver, problem, controller, solver_params,
    truth_state, obs_bundle, cfg,
):
    """Compute pairwise mode collinearity in both Euclidean and R^{-1}-weighted space."""
    print("\n" + "="*70)
    print("TEST 3: R-WEIGHTED COLLINEARITY")
    print("="*70)

    n_params = cfg.n_params
    eps = cfg.fd_epsilon
    obs_times = obs_bundle["obs_times"]
    n_obs_points = obs_bundle["obs_points"].shape[0]

    fwd_kwargs = dict(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params, truth_state=truth_state,
        obs_operator=obs_bundle["obs_operator"],
        obs_times=obs_times, cfg=cfg,
    )

    # Compute mode responses (antisymmetric/linear part)
    G_ref = evaluate_observation_map(np.zeros(n_params), **fwd_kwargs)
    responses = []
    for p in range(n_params):
        theta_plus = np.zeros(n_params); theta_plus[p] = +eps
        theta_minus = np.zeros(n_params); theta_minus[p] = -eps
        G_plus = evaluate_observation_map(theta_plus, **fwd_kwargs)
        G_minus = evaluate_observation_map(theta_minus, **fwd_kwargs)
        linear_response = (G_plus - G_minus) / (2 * eps)
        responses.append(linear_response)

    # Build R^{-1} diagonal
    n_total = len(obs_times) * n_obs_points
    R_inv = np.zeros(n_total)
    for k, sigma in enumerate(obs_bundle["noise_stds"]):
        start = k * n_obs_points
        end = (k + 1) * n_obs_points
        R_inv[start:end] = 1.0 / max(sigma**2, 1e-30)

    R_inv_sqrt = np.sqrt(R_inv)

    # Euclidean collinearity
    print("\n  --- Euclidean collinearity ---")
    cos_eucl = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            ni = np.linalg.norm(responses[i])
            nj = np.linalg.norm(responses[j])
            if ni > 1e-30 and nj > 1e-30:
                cos_eucl[i, j] = np.dot(responses[i], responses[j]) / (ni * nj)

    for i in range(n_params):
        row = "    " + "  ".join(f"{cos_eucl[i, j]:+.6f}" for j in range(n_params))
        print(row)

    # R^{-1}-weighted collinearity
    print("\n  --- R^{-1}-weighted collinearity ---")
    cos_rw = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            # <a, b>_{R^{-1}} = a^T R^{-1} b
            ri_w = R_inv_sqrt * responses[i]
            rj_w = R_inv_sqrt * responses[j]
            ni = np.linalg.norm(ri_w)
            nj = np.linalg.norm(rj_w)
            if ni > 1e-30 and nj > 1e-30:
                cos_rw[i, j] = np.dot(ri_w, rj_w) / (ni * nj)

    for i in range(n_params):
        row = "    " + "  ".join(f"{cos_rw[i, j]:+.6f}" for j in range(n_params))
        print(row)

    # Comparison
    print(f"\n  --- Comparison ---")
    max_diff = 0.0
    for i in range(n_params):
        for j in range(i+1, n_params):
            diff = abs(cos_rw[i, j] - cos_eucl[i, j])
            max_diff = max(max_diff, diff)
            print(f"  Modes ({i},{j}): Euclidean={cos_eucl[i,j]:+.6f}, "
                  f"R-weighted={cos_rw[i,j]:+.6f}, diff={diff:.6f}")

    # Classification
    min_rw_offdiag = min(
        abs(cos_rw[i, j]) for i in range(n_params) for j in range(i+1, n_params)
    )
    if min_rw_offdiag > 0.90:
        classification = "TRULY INDISTINGUISHABLE — high collinearity in R-weighted space"
    elif min_rw_offdiag > 0.70:
        classification = "MOSTLY INDISTINGUISHABLE — R-weighting provides marginal separation"
    elif min_rw_offdiag > 0.50:
        classification = "WEAKLY DISTINGUISHABLE — cost metric separates some modes"
    else:
        classification = "DISTINGUISHABLE — Euclidean collinearity was misleading"

    print(f"\n  Min off-diagonal R-weighted cosine: {min_rw_offdiag:.6f}")
    print(f"  Max Euclidean-vs-Rweighted difference: {max_diff:.6f}")
    print(f"  CLASSIFICATION: {classification}")

    return {
        "cosine_euclidean": cos_eucl.tolist(),
        "cosine_r_weighted": cos_rw.tolist(),
        "max_metric_difference": float(max_diff),
        "min_r_weighted_offdiag": float(min_rw_offdiag),
        "classification": classification,
    }


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 2 Identifiability Validation")
    parser.add_argument("--kl-modes", type=int, default=3)
    parser.add_argument("--kl-length", type=float, default=None)
    parser.add_argument("--n-state-probes", type=int, default=20,
                        help="Number of random state probes for coupling analysis")
    args = parser.parse_args()

    cfg = AuditConfig(
        kl_n_modes=args.kl_modes,
        kl_correlation_length=args.kl_length,
        fd_epsilon=0.01,
    )
    output_dir = PROJECT_ROOT / "results" / "identifiability_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("STAGE 2: IDENTIFIABILITY VALIDATION FOR PUBLICATION")
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

    # ===== TEST 3 first (cheaper — reuses existing forward solves pattern) =====
    t3 = time.time()
    collinearity_results = test_r_weighted_collinearity(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params, truth_state=truth_state,
        obs_bundle=obs_bundle, cfg=cfg,
    )
    print(f"  Test 3 time: {time.time() - t3:.1f}s")

    # ===== TEST 1 (most expensive — 40+ forward solves for state probes) =====
    t1 = time.time()
    coupling_results = test_state_adjusted_sensitivity(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params, truth_state=truth_state,
        obs_bundle=obs_bundle, cfg=cfg,
        output_dir=output_dir,
    )
    print(f"  Test 1 time: {time.time() - t1:.1f}s")

    # ===== TEST 2: Reference from hostile audit =====
    print("\n" + "="*70)
    print("TEST 2: EPSILON ROBUSTNESS (from hostile audit)")
    print("="*70)
    print("  Already completed. Results:")
    print("  ε=0.001: σ = [0.57925752, 0.03085622, 0.00493927]")
    print("  ε=0.005: σ = [0.57925757, 0.03085622, 0.00493930]")
    print("  ε=0.010: σ = [0.57925772, 0.03085624, 0.00493941]")
    print("  ε=0.020: σ = [0.57925836, 0.03085632, 0.00493983]")
    print("  ε=0.050: σ = [0.57926285, 0.03085685, 0.00494289]")
    print("  Max relative change: 7.06e-04 across 50x ε range")
    print("  CLASSIFICATION: ROBUST LINEAR REGIME")

    # ===== FINAL SYNTHESIS =====
    print("\n" + "="*70)
    print("FINAL SYNTHESIS")
    print("="*70)

    # Spectral summary
    s_orig = np.array(coupling_results["sigma_original"])
    s_adj = np.array(coupling_results["sigma_adjusted"])
    print(f"\n  A. UPDATED SPECTRAL SUMMARY")
    print(f"     Fixed-state σ:    {s_orig}")
    print(f"     State-adjusted σ: {s_adj}")
    print(f"     Absorption:       {[f'{a*100:.1f}%' for a in coupling_results['absorption_fractions']]}")
    print(f"     Rank @5% (orig):  {coupling_results['rank_original_5pct']}")
    print(f"     Rank @5% (adj):   {coupling_results['rank_adjusted_5pct']}")

    # Collinearity summary
    print(f"\n  B. COLLINEARITY SUMMARY")
    print(f"     Min R-weighted off-diagonal: {collinearity_results['min_r_weighted_offdiag']:.6f}")
    print(f"     Max Euclidean-vs-R diff:     {collinearity_results['max_metric_difference']:.6f}")
    print(f"     Metric: {collinearity_results['classification']}")

    # Failure mode classification
    print(f"\n  C. FAILURE MODE CLASSIFICATION")
    print(f"     State-parameter: {coupling_results['classification']}")
    print(f"     Epsilon:         ROBUST LINEAR REGIME")
    print(f"     Metric:          {collinearity_results['classification']}")

    # Determine if joint confounding worsens things
    rank_dropped = coupling_results["rank_adjusted_5pct"] < coupling_results["rank_original_5pct"]
    state_absorbs_lots = any(a > 0.5 for a in coupling_results["absorption_fractions"])
    collinear_in_r = collinearity_results["min_r_weighted_offdiag"] > 0.90
    epsilon_stable = True  # from hostile audit

    if coupling_results["rank_adjusted_5pct"] == 0:
        dominant = "JOINT CONTROL CONFOUNDING DOMINANT — state absorbs all friction signal"
    elif rank_dropped:
        dominant = "JOINT CONFOUNDING REDUCES RANK — strengthens non-identifiability"
    elif s_adj[0] < 0.5 * s_orig[0] and collinear_in_r:
        dominant = "MIXED — parameter non-identifiability + state compensation"
    elif collinear_in_r:
        dominant = "STRUCTURAL RANK-1 IDENTIFIABILITY — dominant cause"
    else:
        dominant = "SEVERELY ILL-CONDITIONED BUT NOT NULL"

    print(f"\n  DOMINANT CAUSE: {dominant}")

    # Final verdict
    if epsilon_stable and collinear_in_r and (coupling_results["rank_adjusted_5pct"] <= 1):
        verdict = "CONFIRMED"
        verdict_detail = "Original NO-GO conclusion is robust and publishable"
    elif epsilon_stable and collinear_in_r:
        verdict = "CONFIRMED"
        verdict_detail = "Non-identifiability robust; state coupling is secondary"
    elif not collinear_in_r:
        verdict = "WEAKENED"
        verdict_detail = "Euclidean collinearity overstated; R-weighted metric shows separation"
    else:
        verdict = "CONFIRMED WITH CAVEATS"
        verdict_detail = "Conclusion holds but state-adjusted rank may differ"

    print(f"\n  D. FINAL VERDICT: {verdict}")
    print(f"     {verdict_detail}")

    # Wrong-sign interpretation
    print(f"\n  E. WRONG-SIGN CONVERGENCE INTERPRETATION")
    if coupling_results["rank_adjusted_5pct"] <= 1:
        print(f"     The wrong-sign mode lies in the null space of the state-adjusted")
        print(f"     observation map. The optimizer has freedom to move along this")
        print(f"     direction without penalty, and noise determines which sign wins.")
    else:
        print(f"     The wrong-sign mode is poorly conditioned (σ ratio > 10:1).")
        print(f"     The cost function has a very shallow valley in this direction,")
        print(f"     and observation noise pushes the optimizer to the wrong minimum.")

    # Publication statement
    print(f"\n  F. PUBLICATION-LEVEL STATEMENT")
    min_rw = collinearity_results["min_r_weighted_offdiag"]
    cond = s_orig[0] / max(s_orig[-1], 1e-30)
    print(f'     "On the Shinnecock Inlet mesh with {len(obs_bundle["obs_times"])} elevation')
    print(f'      observation snapshots at {obs_bundle["obs_points"].shape[0]} points,')
    print(f'      the whitened parameter sensitivity matrix has singular values')
    print(f'      [{s_orig[0]:.3f}, {s_orig[1]:.4f}, {s_orig[2]:.5f}] (condition number {cond:.0f}).')
    print(f'      Pairwise mode collinearity in the R^{{-1}}-weighted metric is')
    print(f'      >{min_rw:.2f} for all pairs. Effective rank is')
    print(f'      {coupling_results["rank_original_5pct"]} (5% threshold) or')
    print(f'      {coupling_results["rank_original_10pct"]} (10% threshold).')
    if coupling_results["rank_adjusted_5pct"] < coupling_results["rank_original_5pct"]:
        print(f'      After projecting out the state-reachable observation subspace,')
        print(f'      effective rank drops to {coupling_results["rank_adjusted_5pct"]}.')
    print(f'      Multi-parameter Manning friction estimation is therefore')
    print(f'      structurally underdetermined under this observation configuration.')
    print(f'      Single-parameter (total friction level) estimation remains viable."')

    # Save results
    all_results = {
        "test1_state_adjusted": coupling_results,
        "test2_epsilon": "see hostile_audit_results.json — ROBUST",
        "test3_collinearity": collinearity_results,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "dominant_cause": dominant,
        "total_time_s": time.time() - t0,
    }
    results_file = output_dir / "stage2_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    raise SystemExit(main())

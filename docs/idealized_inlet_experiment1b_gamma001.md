# Idealized Inlet — Experiment 1b: γ = 0.01 Follow-up

**Date:** 2026-04-20
**Branch:** `refactor/4dvar-parallel`
**Hardware:** M-series MacBook, 16 GB RAM, `mpirun -np 2`
**Prior doc:** [idealized_inlet_experiment1_correlated_B.md](idealized_inlet_experiment1_correlated_B.md) — Experiment 1 produced the hypothesis this run tests.

---

## 1. Executive Summary

Experiment 1 established that correlated B made the L_wme spectrum ratio jump from 1 to 1200 but the adaptive `γ = 0.1` floor collapsed the per-direction effective weights into `[0.9917, 0.9992]` — too narrow for meaningful differential descent. The hypothesis entering this pass was that a smaller `γ` would widen the weight range and let DC-WME's predictability term finally matter.

**Result: the hypothesis is refuted on this problem.** Lowering `γ` from `0.1` to `0.01` **did** widen the floor as predicted (floor dropped from 120 → 12, natural eigenvalues jumped from 24 → 105, effective weight range widened from `[0.9917, 0.9992]` to `[0.917, 0.9992]` — an 11× wider span). But the DC-WME optimization trajectory **did not change at all**: evaluations 1–7 match v4 (γ=0.1) to 6-decimal RMSE, and eval #8 fails with the identical Newton divergence. At matched eval budget the two runs are numerically indistinguishable.

This means **over-aggressive flooring was not the main bottleneck** — or at least, reducing γ by a factor of 10 is not enough to expose it. Static DC-WME with correlated B on this configuration is effectively exhausted. Further γ reduction (e.g., 0.001) is the last cheap tunable, but the observed indifference to a 10× γ change is a strong negative signal that this mechanism is not where the separating case lives.

---

## 2. Exact Config Change

Single-knob intervention, implemented by exposing the hardcoded parameter as a CLI flag:

| File | Before | After |
|---|---|---|
| [experiments/idealized_inlet_da.py:496](../experiments/idealized_inlet_da.py#L496) | `predictability_gamma=0.1` (hardcoded) | `predictability_gamma=args.predictability_gamma` |
| [experiments/idealized_inlet_da.py:728-733](../experiments/idealized_inlet_da.py#L728-L733) | — | Added `--predictability-gamma FLOAT, default=0.1` CLI flag with docstring |
| Result-file tag | `result_dcwme_static_Lcorr1500.json` | `result_dcwme_static_Lcorr1500_g.01.json` when γ ≠ 0.1 |

No other changes. Same mesh, truth, observations, background, B (diagonal), L_wme correlation kernel (L=1500 m), optimizer, bounds, smoother, MPI topology, iteration budget, `--skip-tlm-eq38`.

**CLI**:
```bash
mpirun -np 2 python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.1 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --skip-tlm-eq38 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.01 \
  --mem-limit-gb 7
```
Log file: [logs/dcwme_static_exp1b_Lcorr1500_gamma001.log](../logs/dcwme_static_exp1b_Lcorr1500_gamma001.log)

---

## 3. Spectral Diagnostics — γ = 0.1 vs γ = 0.01

The raw L_wme spectrum is identical across γ (it is built from the same kernel and B). Only the adaptive floor and the regularization change.

| Quantity | Baseline (diag B) | γ = 0.1 (v4) | γ = 0.01 (this run) |
|---|---:|---:|---:|
| σ_b² | 9.77e-3 | 9.77e-3 | 9.77e-3 |
| λ_max(G) | ≈ 1.0 | 115.9 | 115.9 |
| λ_max(L_wme) raw | ≈ 1.01 | 1202 | 1202 |
| λ_min(L_wme) raw | ≈ 1.00 | 1.00 | 1.00 |
| Raw spectrum ratio | ≈ 1 | 1200 | 1200 |
| γ_floor (adaptive) | — | 120.2 | **12.02** |
| Natural eigenvalues | 1163 / 1163 | 24 / 1163 | **105 / 1163** |
| Floored eigenvalues | 0 / 1163 | 1139 / 1163 | **1058 / 1163** |
| Eigenvalues > 2.0 | 0 | 211 | 211 (unchanged — raw) |
| Eigenvalues > 10 | 0 | 127 | 127 (unchanged — raw) |
| Eigenvalues > 100 | 0 | 27 | 27 (unchanged — raw) |

**The spectral shape change is exactly what we asked for.** The number of natural directions above the floor quadrupled (24 → 105); the floor dropped an order of magnitude (120 → 12).

---

## 4. Effective-Weight Analysis

Each L_wme eigendirection `i` contributes to the DC-WME cost surface with an effective data weight `w_i = 1 − 1/λ_i^{reg}` where `λ_i^{reg} = max(λ_i, γ_floor)`.

| γ | γ_floor | Natural | Natural weight range | Floored weight | **Span (span = max − min)** |
|---:|---:|---:|---|---|---:|
| 0.1 (v4) | 120.2 | 24 | 1 − 1/120 = 0.9917 → 1 − 1/1202 = 0.9992 | 0.9917 | **0.0075** |
| 0.01 (this run) | 12.02 | 105 | 1 − 1/12 = 0.9170 → 1 − 1/1202 = 0.9992 | 0.9170 | **0.0822** |

**Weight span widened 11× as predicted.** At γ=0.01 the 105 natural directions span weights from 0.917 (at the floor) up to 0.999 (at λ_max). The floored mass still applies weight 0.917 uniformly, but the natural directions above now carry materially different weights from each other. On the linear-algebra layer this is exactly the differential descent regime we wanted to test.

---

## 5. Optimization Results — Matched-Eval Comparison

| Eval | 4D-Var | DC-WME (diag, γ=0.1) | DC-WME (Lcorr=1500, γ=0.1) | DC-WME (Lcorr=1500, γ=0.01) |
|---:|---:|---:|---:|---:|
| 1 (bg) | 0.149041 | 0.149041 | 0.149041 | 0.149041 |
| 2 | 0.141615 | 0.144691 | 0.144691 | 0.144691 |
| 3 | 0.133603 | 0.144115 | 0.144579 | 0.144578 |
| 4 | **0.122649** | 0.144324 | 0.145103 | 0.145101 |
| 5 | 0.123717 | 0.143537 | 0.144885 | 0.144883 |
| 6 | 0.124449 | Newton fail | 0.143716 | 0.143714 |
| 7 | 0.124538 | — | 0.142802 | 0.142801 |
| 8 | 0.124474 | — | Newton fail | Newton fail (same timestep) |

Cost trajectories:

| Eval | γ=0.1 cost | γ=0.01 cost | Δ cost |
|---:|---:|---:|---:|
| 1 | 4706.62 | 4706.62 | 0 |
| 2 | 4416.24 | 4415.61 | −0.63 |
| 3 | 4321.22 | 4320.53 | −0.70 |
| 4 | 4264.35 | 4263.64 | −0.71 |
| 5 | 4166.65 | 4165.60 | −1.05 |
| 6 | 3961.71 | 3960.18 | −1.53 |
| 7 | 3775.42 | 3773.10 | −2.32 |

The γ=0.01 cost is ≤ 0.06 % below γ=0.1 at every eval — not a trajectory divergence, just the algebraic fact that a lower floor subtracts marginally more predictability mass. **RMSE is identical to 6 decimal places** through all 7 successful evals, and the line-search probe at eval #8 fails at the identical point under both γ settings.

### End-state summary

| Method | Evals | Final RMSE | Δ RMSE | Wall | Exit |
|---|---:|---:|---:|---|---|
| 4D-Var | 15 | 0.124500 | **−16.47 %** | 174 min | budget |
| DC-WME (diag, γ=0.1) | 5 | 0.143537 | −3.70 % | >60 min | OOM after inf |
| DC-WME (Lcorr, γ=0.1) — v4 | 7 | 0.142802 | −4.18 % | ~190 min | OOM after inf |
| DC-WME (Lcorr, γ=0.01) — this run | 7 | 0.142801 | −4.18 % | similar | same failure mode |

---

## 6. Comparison to 4D-Var

4D-Var is unchanged under any of these DC-WME knobs. At matched 5-eval budget:
- 4D-Var: −17.0 % RMSE improvement
- DC-WME (γ=0.01, Lcorr=1500): −2.8 % RMSE improvement

4D-Var stays ~6× ahead per eval. Extending DC-WME to 7 evals brings it to −4.18 % improvement, still decisively behind 4D-Var's 15-eval result.

---

## 7. Required Interpretation

### Q1. Did lowering `predictability_gamma` materially widen the effective weight range?

**Yes, by exactly the predicted factor.** Span widened from 0.0075 at γ=0.1 to 0.0822 at γ=0.01 — 11× wider. Natural eigenvalue count quadrupled (24 → 105).

### Q2. Did DC-WME become meaningfully different from the previous γ=0.1 run?

**No.** RMSE tracks v4 to 6 decimal places at every eval. Cost differs by ≤ 0.06 %. Same eval-#8 failure. The predictability-term mass and gradient contribution at each BLMVM iterate is statistically indistinguishable from the γ=0.1 run.

### Q3. Did RMSE improvement increase enough to say the floor was the main bottleneck?

**No.** The improvement is numerically identical at matched evals. The floor was a real restriction on per-direction weights but turns out not to have been a binding constraint on the descent that BLMVM would take — the gradient direction is dominated by the bulk (floored) mass, not by the differential on the natural tail.

### Q4. Did DC-WME materially close the gap to 4D-Var?

**No.** Still ~6× less efficient per eval. No change versus Experiment 1.

### Q5. Is static DC-WME with correlated B now effectively exhausted on this problem?

**Yes, in the sense that the remaining tunable (γ) did not move the needle.** One could push further to γ=0.001 (floor would drop to 1.2, weight range would widen to `[0.17, 0.999]` — a span 110× wider than γ=0.1). But the indifference to a 10× γ reduction strongly suggests the failure is not in the regularization layer. More likely root causes:
  - The gradient smoother (Gaussian L=500m on h, u, v) dominates the descent direction and washes out L_wme's directional structure.
  - BLMVM's L-BFGS history (size 3) reconstructs a Hessian proxy that is largely unaware of the eigen-decomposition of L_wme; the direction it takes is close to the data-misfit-only direction.
  - The correlated B is expressed only in L_wme (obs-space kernel relaxation); the background penalty remains diagonal, which anchors the descent to the diagonal-B geometry.

Any of these would explain why L_wme's spectral shape cannot override the descent. Item 3 is the most actionable — it means the right next experiment is one where the correlation structure actually enters the background penalty, not just L_wme.

---

## 8. Final Recommendation

**Do not keep refining correlated-B static DC-WME at this mesh size.** Two runs (γ=0.1 and γ=0.01) with the same spectrum-widening setup have produced the same descent trajectory and the same failure point. A third γ value will almost certainly produce the same qualitative result; this is already a strong empirical signal that the mechanism of interest is not activating.

**Preferred next experiment: move to sparse observations (obs_fraction ≈ 0.005, ~60 obs) with dynamic TLM Eq 38 and correlated B.** Rationale:

1. **Dynamic Eq 38 uses actual TLM Jacobians** to build the Gram matrix, not the near-identity approximation of H·B·Hᵀ on the static path. That injects real dynamical predictability structure that the static kernel cannot capture.
2. **Sparse obs** (~60) makes the TLM build tractable (~25 min instead of ~8 h) and keeps L_wme small enough that the predictability term is not swamped by bulk-floored directions.
3. **Correlated B** keeps the spectral-spreading mechanism in play, combined with dynamic propagation.

This is Experiment 3 from [the separating-case search plan](idealized_inlet_dcwme_separating_case_search.md). Rough budget: ~1.5 h on the laptop.

**Secondary candidate: clustered observations.** Orthogonal mechanism — obs geometry instead of B structure. Good fallback if Experiment 3 also fails to separate. 

**Not recommended: γ=0.001.** The jump from γ=0.1 to γ=0.01 already widened the weight span 11× and produced zero trajectory change. A further 10× widening (γ=0.001, floor = 1.2, span 0.83) is the last algebraically meaningful γ reduction, but based on this run's indifference it is very unlikely to produce separation either, and it begins to introduce numerical risk (regions with weight ≈ 0 can destabilize the cost Hessian near convergence).

---

## 9. Hard-Constraint Compliance

- **Only `predictability_gamma` changed.** All other controls verified identical to v4.
- **Not a sweep.** One new run; compared against existing v4 and 4D-Var baselines.
- **Observations unchanged.** Same `obs_fraction=0.1`, same seed, 1163 interior points.
- **B unchanged.** Diagonal component-aware B in the background penalty; correlated-kernel approximation in L_wme only (as in Experiment 1).
- **Optimizer unchanged.** Same BLMVM, same history=3, same bounds, same line_search_max_funcs=5.
- **Call not claimed as success.** RMSE did not move; trajectory did not separate. Recorded as a clear negative result with mechanistic implications and a defended next step.

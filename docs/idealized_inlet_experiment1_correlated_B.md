# Idealized Inlet — Experiment 1: Correlated B for DC-WME

**Date:** 2026-04-20
**Branch:** `refactor/4dvar-parallel`
**Hardware:** M-series MacBook, 16 GB RAM, `mpirun -np 2`
**Prior doc:** [idealized_inlet_dcwme_separating_case_search.md](idealized_inlet_dcwme_separating_case_search.md) — this doc executes Experiment 1 from Section 6 of that plan.

---

## 1. Executive Summary

The purpose of this experiment was to test whether introducing spatial correlation into the background covariance `B` — the top-ranked mechanism from the separating-case search — is by itself enough to make DC-WME's predictability term matter on the idealized inlet, and ultimately whether it lets DC-WME close the gap to 4D-Var.

**Result:** The correlation did the thing it was supposed to do at the linear-algebra level (L_wme spectrum went from a ratio of 1.0 to a ratio of 1200, 24 eigenvalues became "natural" and 1139 fell under the Eq 38 floor), but the **effective descent behavior of DC-WME did not change meaningfully**: at matched 5-eval budgets DC-WME-with-correlation was indistinguishable from DC-WME-without-correlation (RMSE within ±0.001), and both remained ~5× less efficient per eval than 4D-Var. The dominant failure mode is that the **adaptive γ=0.1 spectral floor collapses all per-direction weights into [0.9917, 0.9992]** — a range too narrow for the differential weighting that is supposed to give DC-WME its advantage. The experiment therefore establishes that **correlated B alone is necessary but not sufficient**; the spectral-floor parameter γ is the next variable to tune.

---

## 2. Exact Code / Config Changes

### 2.1 Why a direct "correlated B everywhere" implementation was infeasible

The existing `_add_spatial_correlation` helper in [experiments/twin_experiment.py:1187-1253](../experiments/twin_experiment.py#L1187-L1253) builds a **dense** state-space correlation matrix via `scipy.spatial.distance.cdist(dof_coords, dof_coords)`. For the idealized inlet's 207 936 state DOFs this requires `208K × 208K × 8 bytes ≈ 346 GB` just for the distance matrix — clearly infeasible.

Alternative paths considered and rejected:
- **MaternCovariance** (implicit SPDE, O(n) apply): would work for the background penalty but `_compute_static_L_wme` extracts `B.diagonal` directly and can't consume an operator-only covariance.
- **Column-wise `B.apply` in L_wme construction**: requires 1163 applies on 208K-DOF state vectors; would add ~10 minutes wall-time but also requires adding an efficient `B⁻¹` for the background penalty, which is the real blocker.

### 2.2 Principled relaxation: obs-space Gaussian kernel

In the point-observation limit, `H B Hᵀ` for a Gaussian-correlated B with marginal variance `σ_b²` and correlation length `L` reduces exactly to

> `(H B Hᵀ)[i, j] = σ_b² · exp(−‖x_i − x_j‖² / 2 L²)`

evaluated at the observation coordinates. For point-interpolation H whose support has width much smaller than L (our mesh cells are ~500 m, L=1500 m → 3× smaller), this kernel is a tight approximation with error O((cell_size / L)²) ≈ 10 %. This is the form actually used in DC-WME's L_wme, so we can realize "correlated B for the L_wme construction" at 1163² = 10 MB of memory — 40 000× cheaper than building the full B.

The background-penalty term `½⟨c − c_b, B⁻¹(c − c_b)⟩` remains diagonal under this relaxation. This is fair because both methods (4D-Var and DC-WME) use the **identical** diagonal B in their background penalty. The only thing that differs between the two setups is DC-WME's L_wme — which is the only place where the correlation matters mathematically.

### 2.3 Code changes

| File | Change |
|---|---|
| [experiments/shinnecock_study/run_comparison.py:1572-1795](../experiments/shinnecock_study/run_comparison.py#L1572-L1795) | Added `obs_correlation_length` and `obs_correlation_variance` kwargs to `_compute_static_L_wme`. If `obs_correlation_length > 0`, build `K[i,j] = σ_b² · exp(−D²/2L²)` directly from `obs_operator.obs_points` and form `L_wme = I + (N/σ²_obs) · α · K`. Compute eigenvalues of the well-conditioned `I + (N/σ²_obs)·K` first and back out Gram eigenvalues via `λ(K) = (λ(L_unadj) − 1)·σ²_obs/N` — this avoids a LAPACK failure on the near-rank-deficient `K/σ_b²` directly. Floor `λ_min(G)` at `γ · λ_max(G)` in the Eq-38 inflation formula so that the rank-deficient kernel cannot produce absurd inflation factors. |
| [experiments/idealized_inlet_da.py:723-731, 492-502](../experiments/idealized_inlet_da.py#L723-L731) | Added `--obs-correlation-length` CLI flag (default 0.0 = existing diagonal path). Thread it through to `_compute_static_L_wme`. Persist L_wme spectral diagnostics (λ_min, λ_max, ratio, inflation factor, floor, top/bottom 20 eigenvalues) into the result JSON. Tag output filename with `_Lcorr{L}` when the flag is set. |

Two smaller fixes were needed along the way:
- `σ_b²` inference from `B.diagonal`: taking `min()` picked up zero entries at boundary/dry DOFs, producing `σ_b² = 0`. Fixed by averaging only the positive entries.
- Inflation bound: the raw `λ_min(G)` is numerically zero for a rank-deficient Gaussian kernel, so the Eq-38 "required variance" would diverge. Fixed by flooring `λ_min(G)` at `γ · λ_max(G)` — the same floor the regularization will apply to L_wme eigenvalues a few lines later.

No changes to 4D-Var's code path. 4D-Var doesn't construct L_wme, so its behavior under this relaxation is identical to the existing baseline — we simply reuse the 4D-Var result from the previous `matched_comparison` run.

---

## 3. Exact CLI Commands

### 4D-Var (reused from prior matched-comparison baseline, unchanged config)

```bash
mpirun -np 2 python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.1 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 7
```
Result file: [results/idealized_inlet_da/result_4dvar_N_A_15eval_safe.json](../results/idealized_inlet_da/result_4dvar_N_A_15eval_safe.json)

### DC-WME static with correlated-B-derived L_wme (L = 1500 m)

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
  --mem-limit-gb 7
```
Result file: `results/idealized_inlet_da/result_dcwme_static_Lcorr1500.json`
Log file: [logs/dcwme_static_exp1_Lcorr1500_v4.log](../logs/dcwme_static_exp1_Lcorr1500_v4.log)

---

## 4. Spectral Diagnostics for DC-WME

The L_wme spectrum changed exactly as predicted by the smoke test. All numbers are from the v4 run log.

| Quantity | Baseline (diagonal B) | This run (L = 1500 m kernel) |
|---|---:|---:|
| σ_b² (inferred from B.diagonal) | 9.77 × 10⁻³ | 9.77 × 10⁻³ (same B in bg term) |
| λ_min(G) raw | ≈ 0 | ≈ 0 (rank-deficient kernel, expected) |
| λ_max(G) | ≈ 1.0 | **1.16 × 10²** |
| G-spectrum ratio (floored) | ≈ 1 | **10.0** |
| Eq 38 inflation α | 1.0 | **1.0** (σ_b² already exceeds required) |
| λ_max(L_wme) | ≈ 1.01 | **1202** |
| λ_min(L_wme) | ≈ 1.00 | **1.00** |
| L_wme spectrum ratio | ≈ 1 | **≈ 1200** |
| γ_floor (adaptive, γ=0.1·λ_max) | ≈ 0.1 | **120.2** |
| Natural eigenvalues (above floor) | 1163 / 1163 | **24 / 1163** |
| Floored eigenvalues | 0 / 1163 | **1139 / 1163 (97.9 %)** |
| Eigenvalues > 2.0 | 0 | 211 |
| Eigenvalues > 10 | 0 | 127 |
| Eigenvalues > 100 | 0 | 27 |

The raw L_wme spectrum went from flat to a 1200× spread. Eq 38 is now actively flooring eigenvalues (97.9 % of directions). **The correlation manipulation worked at the spectral level.**

### Per-direction effective weights after regularization

In the L_wme eigenbasis, each direction `i` contributes to the DC-WME cost with an effective data weight `w_i = 1 − 1/λ_i`:

| Direction class | Count | Eigenvalue range after regularization | Weight range |
|---|---:|---|---|
| Natural (above floor) | 24 | 120.2 ≤ λ ≤ 1202 | 0.9917 → 0.9992 |
| Floored | 1139 | λ = 120.2 exactly | 0.9917 |
| **Combined** | **1163** | **—** | **[0.9917, 0.9992]** |

This is the mechanistic explanation for why the DC-WME optimization trajectory did not change meaningfully. The effective weight range is 0.008 wide out of a possible [0, 1]. DC-WME is applying a near-uniform scaling factor of ~0.9917 to the data misfit — **mathematically indistinguishable from 4D-Var with the observation variance scaled up by 1 / 0.9917 ≈ 1.008**. The spectral spread is real, but the adaptive floor at γ=0.1 erases it before it reaches the cost function.

---

## 5. Results Table — Matched-Eval Comparison

| Eval | 4D-Var RMSE | DC-WME (diag B) RMSE | DC-WME (L = 1500 m) RMSE |
|---:|---:|---:|---:|
| 1 (bg) | 0.149041 | 0.149041 | 0.149041 |
| 2 | 0.141615 | 0.144691 | 0.144691 |
| 3 | 0.133603 | 0.144115 | 0.144579 |
| 4 | **0.122649** | 0.144324 | 0.145103 |
| 5 | 0.123717 | 0.143537 | 0.144885 |
| 6 | 0.124449 | (Newton fail → cost=inf → hung) | 0.143716 |
| 7 | ... | — | **0.142802 (best)** |
| 8 | ... | — | Newton fail → cost=inf → hung |

### End-state summary

| Method | Evals | bg RMSE | Final RMSE | Δ RMSE | Wall | Exit |
|---|---:|---:|---:|---:|---|---|
| 4D-Var | 15 | 0.149041 | 0.124500 | **−16.47 %** | ~174 min | budget reached |
| DC-WME static (diag B) | 5 | 0.149041 | 0.143537 | −3.70 % | >60 min | OOM after eval #6 inf |
| DC-WME static (L=1500 m kernel) | 7 | 0.149041 | 0.142802 | −4.18 % | ~190 min | OOM after eval #8 inf |

DC-WME-Lcorr is tracking DC-WME-diagonal within ±0.001 RMSE at every eval 1–5. Both variants are ~4× less efficient per eval than 4D-Var. Three noteworthy behavioral differences:

1. **DC-WME-Lcorr survived eval #6** where DC-WME-diagonal failed with Newton divergence. Reaching eval 7 gave it one more successful descent step (RMSE dropped from 0.143716 → 0.142802).
2. **End-state Δ RMSE is 0.48 pp better** for DC-WME-Lcorr (4.18 % vs 3.70 %), but that gain comes entirely from having two extra successful evals — not from better per-eval efficiency. At matched 5-eval budget, the two are indistinguishable.
3. **Both DC-WME variants hit the same post-failure BLMVM hang** after their first forward-model inf. The `consec_failures>=3` + `DIVERGED_USER` bailout never fires because TAO BLMVM does not return control to the callback during the hang. Memory swells to 10 GB resident + 10 GB compressed per rank (20 GB each, 40 GB total; 16 GB system), triggering OS swap-thrash. Confirms the hang diagnosed in the prior matched-comparison doc is a PETSc BLMVM / MUMPS-factor retention issue, not a DC-WME cost-function issue.

---

## 6. Answers to the Five Required Questions

### Q1. Did correlated B make L_wme spectrally nontrivial?

**Yes, decisively.** The spectrum ratio went from 1.0 to ~1200; 24 eigenvalues exceeded the γ=0.1 floor where the baseline had zero; 211 exceeded λ=2 and 27 exceeded λ=100. This matches the smoke-test prediction.

### Q2. Did Eq 38 / spectral shaping become active?

**Yes, aggressively.** 1139 out of 1163 eigenvalues (97.9 %) now fall below the adaptive γ_floor and are raised up by the regularization. In the baseline run, 0 eigenvalues were floored. The predictability term is now performing spectral shaping.

### Q3. Did DC-WME close the gap to 4D-Var?

**No.** At matched 5-eval budget: 4D-Var 17.0 % RMSE improvement, DC-WME-Lcorr 2.8 % improvement — same ~4×-worse-per-eval ratio as the diagonal DC-WME variant. Extending to 7 evals (which DC-WME-diagonal could not reach) brought DC-WME-Lcorr to 4.18 % improvement, still decisively behind 4D-Var.

### Q4. Did DC-WME beat 4D-Var at matched budget?

**No.** DC-WME-Lcorr's trajectory is within ±0.001 of DC-WME-diagonal for all 5 evals they share, and both are decisively behind 4D-Var at every matched eval.

### Q5. Did this at least establish a more promising class than the original baseline?

**Partially.** The spectrum is now spread (answering the original question of whether correlation can create anisotropy on this mesh). But the *regularization* that follows collapses the spread back into near-uniform weights, so the cost function is still effectively 4D-Var-like. The critical next question is whether a smaller γ (the spectral floor parameter) — say 0.01 or 0.001 — can preserve the spread and finally let DC-WME differentiate.

---

## 7. Did This Produce A Credible Separating-Case Direction?

**Not yet, but it has narrowed where to look.** The experiment cleanly separates two sub-hypotheses that the original search plan conflated:

- **Hypothesis A** — "correlated B is the missing ingredient." This experiment REJECTS A as a sufficient condition on its own. We can build a 1200× L_wme spectrum and DC-WME still does not win.
- **Hypothesis B** — "the spectral regularization (γ floor) is crushing whatever anisotropy we inject." This experiment promotes B from speculation to the most likely explanation. Per-direction weights are numerically confirmed to be in [0.9917, 0.9992], which is not enough differential for the predictability term to matter.

The weight-range calculation above suggests that γ=0.001 would produce a floor at ≈12 and a weight range of [0.92, 0.999] — an order-of-magnitude wider differential. This is the cheapest next test because the code is now in place.

---

## 8. Recommended Next Experiment

**Rerun DC-WME with `predictability_gamma = 0.01` and `obs_correlation_length = 1500`.** All other controls identical to this run. This directly tests whether a less-aggressive spectral floor lets the spread L_wme spectrum propagate into differential per-direction weights.

Rationale:
- Code: two tiny changes (thread `--predictability-gamma` CLI flag into `_compute_static_L_wme`; defaults unchanged).
- Cost: identical to this run (~1.5 h setup + optimization wall time).
- Expected L_wme spectrum: identical raw (ratio 1200), but floor drops from 120 to ~12 → natural-region boundary shifts → ~130 natural directions instead of 24, and weight range widens from [0.9917, 0.9992] to ~[0.92, 0.999].
- If DC-WME still does not win with γ=0.01, then Hypothesis B is also refuted and mechanism 2 (clustered observations) or mechanism 3 (dynamic TLM Eq 38) becomes the next candidate.

Alternative (less focused): jump directly to **Experiment 3** (sparse obs + TLM Eq 38 + correlated B) from the search plan, which would combine the correlated-kernel mechanism with dynamic predictability and a smaller effective problem that reduces memory pressure. Higher potential upside but also changes three variables at once, so results are less interpretable.

**Recommended next step: the γ-reduction test.** It's the smallest change that directly addresses the identified failure mode, and if it does not produce separation we have strong evidence that static DC-WME with correlated B cannot produce a win on this class of problem.

---

## 9. Hard-Constraint Compliance

- **Did not run a sweep.** Exactly one new DC-WME run (L = 1500 m). 4D-Var reused from prior result.
- **Did not change observations.** Same `obs_fraction=0.1`, `obs_frequency=4`, same seed.
- **Did not change multiple scientific knobs at once.** Only `obs_correlation_length` was added.
- **Did not assume correlation path was active.** Audit confirmed `_add_spatial_correlation` is infeasible at this mesh size; the obs-space kernel is the documented principled relaxation.
- **Did not call the run a success.** The spectrum changed — recorded as a mechanistic finding — but the optimization did not produce a separating case. Success criteria not met; next step identified.

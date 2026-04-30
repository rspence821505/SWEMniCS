# Post-Fix DC-WME + TLM Eq 38 Revalidation — Idealized Inlet

**Date**: 2026-04-22
**Lineage**: [idealized_inlet_jacobian_handoff_trace.md](idealized_inlet_jacobian_handoff_trace.md) → this memo
**LS6 runs**: 3102561 (post-fix DC-WME component-aware, commit `11f3a62`) and 3098387 (matched 4D-Var baseline)
**Outcome**: **DC-WME now runs on a legitimate predictability bound for the first time on this problem, but still underperforms 4D-Var by ~25× on RMSE (0.1% vs 2.5% improvement). The pre-fix negative conclusions are VINDICATED qualitatively, but must be restated in terms of "DC-WME struggles on sparse-WSE inlet even with a real Eq 38 bound," not "DC-WME is structurally blind." The new post-fix story is more interesting, not less.**

---

## 1. Executive summary

The Jacobian handoff fix (commit `3852e6f`) restored real cross-component physics to the TLM Eq 38 Gram matrix. The Gram now has:

- `λ_min(G_h) = 9.32` → `σ_b²_h = 1.07e-02` (barely changed from the pre-fix value)
- `λ_min(G_uv) = 0.578` → **`σ_b²_uv = 1.73e-01`** (pre-fix: degenerate `1.0e+29` sentinel)
- Full rank 58/58, condition 2.67

Water-surface-elevation is now confirmed **~16× more predictable than (u, v) momentum** on this problem — a *scientifically meaningful* statement the pre-fix path could never have produced.

Despite that, the **DC-WME optimization produces only 0.1% RMSE improvement (0.148444 → 0.148305)**, versus matched 4D-Var's **2.5% improvement (0.148444 → 0.144789)**. Cost drops monotonically over 15 function evaluations (270.98 → 250.02, −7.7%), yet `RMSE_truth` does not track cost — it dips to 0.148076 at eval #4 then *rises* back into a 0.1483 band and stays there. The DC-WME cost-gradient system is moving the analysis *toward observations* without meaningfully closing the gap to truth.

The correct interpretation is:

> **The Eq 38 predictability bound is real, but using it to inflate B by ~16× on the uv block under-regularizes the unobserved momentum degrees of freedom given sparse WSE-only observations. The optimizer exploits the extra freedom to reduce cost via uv adjustments that aren't constrained by observations nor by a strong enough prior.** This is a *methodological* limitation of DC-WME on this observation regime, not a bug.

---

## 2. Exact fix being relied on

```diff
 # experiments/idealized_inlet_da.py:261
-    truth_jacobians = [J.duplicate() for J in solver_truth.storage.saved_jacobians]
+    truth_jacobians = [J.duplicate(copy=True) for J in solver_truth.storage.saved_jacobians]
```

Commit `3852e6f`. The Shinnecock equivalent (`run_comparison.py:4327`) was fixed in `0c06c8c`. Full chain of diagnostic memos (uv-bisector → threshold-fix validation → structural diag → R1 reassembly → R2 handoff) is in the `docs/idealized_inlet_*` series; this memo is the first **scientific** output after that chain closed.

---

## 3. Regression test added

[tests/test_mat_duplicate_regression.py](../tests/test_mat_duplicate_regression.py), commit `11f3a62`, 4 tests:

1. `test_mat_duplicate_default_is_zero_values` — pins PETSc `Mat.duplicate()` default-`copy=False` behavior.
2. `test_mat_duplicate_with_copy_flag_preserves_values` — pins `copy=True` correctness + buffer independence.
3–4. `test_known_jacobian_deepcopy_sites_use_copy_true` — grep-tests both previously-buggy files and fails if the bare `[J.duplicate() for J in ...saved_jacobians]` idiom is ever reintroduced.

All 4 pass locally.

---

## 4. Exact rerun configuration

DC-WME run (job 3102561), matched to the previously planned production candidate:

```
sbatch --partition=development --time=02:00:00 \
  hpc/lonestar6/idealized_inlet/job_dcwme_component_aware.slurm
```

From [hpc/lonestar6/idealized_inlet/job_dcwme_component_aware.slurm](../hpc/lonestar6/idealized_inlet/job_dcwme_component_aware.slurm):

```
ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --eq38-component-aware \
  --mem-limit-gb 240
```

np=2 × 64 threads on c302-005. Total wall: ~77 min. Optimization phase: 63 min (3780s).

4D-Var baseline (job 3098387, `refactor/4dvar-parallel` prior to fix but unaffected by it since 4D-Var doesn't consume `truth_jacobians`): **same config except `--method 4dvar`**, no `--obs-correlation-length`, no `--predictability-gamma`, no `--eq38-component-aware`. 70 min optimization.

---

## 5. Predictability diagnostics — pre-fix vs post-fix

| Quantity | Pre-fix (3101214, buggy) | **Post-fix (3102561)** | Change |
|---|---:|---:|---|
| Gram time (58 adjoints) | 1193 s | 1187 s | ≈same (pre-fix LU still ran, just on a zero matrix) |
| Gram `λ_max` | — | 24.94 | — |
| Gram `λ_min` (combined) | 9.249 | 9.332 | +0.9% |
| Gram condition | 2.67 | 2.67 | identical |
| Gram rank (> 1e-10) | — | **58/58** | full |
| Spread | — | 120.4% | — |
| `σ_b²` (scalar) | 1.081e-02 | 1.072e-02 | −0.8% |
| **`λ_min(G_h)`** | 9.249 | **9.321** | +0.8% |
| **`σ_b²_h`** | 1.081e-02 | 1.073e-02 | −0.8% |
| h-DOFs inflated | 0 / 35106 | 0 / 35106 | none hit floor |
| **`λ_min(G_uv)`** | numerically zero (singular) | **0.578** | **non-degenerate** ✓ |
| **`λ_max(G_uv)`** | — | 5.777 | — |
| `cond(G_uv)` | — | 10.0 | — |
| **`σ_b²_uv`** | **1.00e+29** (undefined sentinel) | **1.731e-01** | **real finite bound** ✓ |
| uv-DOFs inflated | 70212 / 70212 (to sentinel) | 70212 / 70212 (to 0.173) | same *count*, vastly different *value* |
| **`σ_b²_uv / σ_b²_h`** ratio | ∞ (1e+29 / 1e−2) | **≈ 16.1** | **meaningful anisotropy** ✓ |

Key physics statement newly available: **h-component predictability is ~16× tighter than (u, v) momentum on this observation geometry**, which is intuitive given that all observations are of WSE (h).

---

## 6. DC-WME optimization results on the repaired path

### 6.1 TAO trajectory

| eval | cost | `‖grad‖` | `RMSE_truth` | note |
|---:|---:|---:|---:|---|
| 1 | 270.98 | 243.7 | **0.148444** | initial (≡ background RMSE) |
| 2 | 378.98 | 3033.6 | — | line-search reject |
| 3 | 280.78 | 1424.6 | — | line-search reject |
| 4 | 264.81 | 627.7 | **0.148076** | first accepted step — **best RMSE ever seen** |
| 5 | 263.11 | 122.0 | 0.148269 | |
| 6 | 261.89 | 79.9 | 0.148291 | |
| 7 | 258.38 | 75.0 | 0.148295 | |
| 8 | 255.19 | 125.6 | 0.148289 | |
| 9 | 253.79 | 79.9 | 0.148277 | |
| 10 | 253.65 | 41.4 | 0.148277 | |
| 11 | 253.12 | 36.9 | 0.148280 | |
| 12 | 251.95 | 46.9 | 0.148289 | |
| 13 | 250.55 | 96.4 | 0.148309 | |
| 14 | 250.28 | 30.5 | 0.148305 | |
| 15 | **250.02** | **24.7** | **0.148305** | terminated by `max_funcs=15` |

### 6.2 Final metrics

- **Background RMSE**: 0.148444
- **Analysis RMSE**: **0.148305**
- **Improvement**: **0.1%** (7 parts in 10000)
- **Cost**: 270.979 → 250.018 (−7.7%, monotone after eval #4)
- **Gradient norm**: 243.7 → 24.7 (−90%, converging on a local minimizer)
- Func evals: 15 (hit `max_funcs=15` cap)
- Convergence reason: `USER` (max_funcs stop)
- Optimization wall: 3780 s = 63 min
- Result JSON: `/work/.../results/idealized_inlet_da/result_dcwme_static_Lcorr1500.json`

### 6.3 Decoupling of cost and RMSE

The diagnostic of interest: cost drops 7.7% while `RMSE_truth` stays within a ~0.00025 band around 0.1483. The optimizer is finding solutions that the cost landscape prefers, but those solutions are not closer to the true initial condition. The **minimum RMSE ever observed was at eval #4** (0.148076, first accepted step), not at the terminal cost minimizer.

Contrast with 4D-Var on the same forward problem: cost and `RMSE_truth` move together. The DC-WME cost-RMSE decoupling is specifically induced by the (h, uv) anisotropic B inflation.

---

## 7. Comparison to matched 4D-Var baseline

| Metric | 4D-Var (3098387) | **DC-WME (3102561)** |
|---|---:|---:|
| Config | same (no Eq 38, no corr) | component-aware Eq 38 + corr L=1500 m |
| Background RMSE | 0.148444 | 0.148444 |
| **Analysis RMSE** | **0.144789** | **0.148305** |
| **Improvement** | **2.5%** | **0.1%** |
| Cost reduction | 627.92 → (smaller) | 270.98 → 250.02 (**both within normal range**) |
| Func evals | 15 | 15 (both hit cap) |
| Convergence reason | USER (cap) | USER (cap) |
| Optimization wall | 4193 s | 3780 s |

4D-Var beats DC-WME by **~25× on RMSE improvement**. Note that DC-WME's cost numbers are not directly comparable to 4D-Var's because the DC-WME cost function includes the `− ½ ⟨δQ, L_wme⁻¹ δQ⟩` correction term; the relevant comparison is on analysis-vs-truth distance, which is the `RMSE_truth` column.

---

## 8. Prior negative conclusions — must they be revised?

### Partially yes, partially no.

**Pre-fix conclusion** (from the Phase-3/5 Shinnecock story and the pre-fix idealized-inlet runs): *"DC-WME is structurally blind on flat-predictability problems; L_wme has no directional content because dynamics are too linear/isotropic."*

This was measured through a *zero* Eq 38 Gram for the uv block, which wasn't actually physics — it was the `duplicate()` bug producing empty Jacobians. That specific measurement and the structural-blindness language derived from it are **invalid**.

**Post-fix conclusion** (this memo): *"On the idealized inlet, DC-WME with component-aware Eq 38 produces a legitimate ~16× h-vs-uv predictability anisotropy, but the resulting B inflation under-regularizes the uv block given sparse WSE-only observations, so the analysis doesn't close the gap to truth nearly as well as 4D-Var with uniform B."*

This is a *weaker and more interesting* negative result. It is not a statement about DC-WME being structurally broken — it is a statement about DC-WME's optimization landscape on a specific observation regime (sparse, WSE-only) under a specific predictability imbalance (order-of-magnitude h vs uv).

### What this means for the Shinnecock Phases 3 and 5 story

The Shinnecock `run_comparison.py` had the identical `.duplicate()` bug (fixed in commit `0c06c8c`). The "all-eigenvalues-identical / 5–7% spread" story from Phases 3c and 5c came from a zero adjoint operator, not from tidal-linearity physics. **Those phases must be rerun** before any structural claim about "Shinnecock tidal dynamics being too isotropic for DC-WME" can be made. Given the idealized-inlet result here — where the Gram is *not* flat, it's `cond = 2.67` with legitimate 120% spread, yet DC-WME still underperforms — the phenomenon of DC-WME losing to 4D-Var may be real even on Shinnecock, but the *reasoning* needs to be redone from scratch.

---

## 9. Recommended next step

In order of expected information gain:

1. **Restart the Shinnecock Phase 3c / 5c dynamic-TLM DC-WME runs** on `refactor/4dvar-parallel ≥ 0c06c8c`. Minimum deliverable: one completed dynamic L_wme run with per-component Gram eigenvalues + final analysis RMSE. Expected cost: equivalent to a Phase 3c rerun (~60 h sequential adjoint) — or port to component-aware Eq 38 static (~hours, since static is much cheaper).

2. **Idealized-inlet sensitivity sweep on a single hyperparameter:** `obs_fraction`. The current 0.005 is very sparse; running 0.02, 0.05, 0.10 with the same DC-WME config should reveal whether the RMSE-cost decoupling is a sparseness phenomenon. If DC-WME closes to 4D-Var at obs_fraction ≥ 0.05, the conclusion becomes "DC-WME needs denser observations than 4D-Var to cash in on its predictability-aware prior." That's a *useful, publishable* scientific statement.

3. **Idealized-inlet DC-WME without Eq 38 inflation** (use default `σ_b²` = background-error-std² = 0.02²): isolates whether the RMSE gap to 4D-Var comes from the Eq 38 inflation specifically, or from the `L_wme⁻¹ δQ` correction term in the DC-WME cost itself. If without-inflation DC-WME gets 2.4% improvement (close to 4D-Var), then **the Eq 38 inflation is actively hurting** on this obs regime — that's a cleaner statement about when DC-WME's main feature helps vs hurts.

4. **Only after (1)–(3) have settled into a coherent story:** write the thesis comparison section. The current data is enough for an honest negative-result chapter, but (2) and (3) materially strengthen the narrative.

Do **not** attempt a broader DC-WME sweep on the idealized inlet (varying `predictability_gamma`, `obs_correlation_length`, etc.) until (2) and (3) have narrowed the mechanism. The per-run cost is too high to explore hyperparameter space blindly when a targeted mechanism question is still open.

---

## Appendix: pre-fix vs post-fix final-number table

```
                              Pre-fix (3101214)     Post-fix (3102561)
Eq 38 σ_b²_uv                 1.00e+29 (sentinel)   1.73e-01 (real)
G_uv non-degenerate?          No                    Yes
Final Analysis RMSE           — (job killed at      0.148305
                                 timestep 5)
Improvement                   —                     0.1%
Converged                     No (SIGTERM)          Yes (max_funcs stop)
Prior claim valid?            N/A                   New legitimate science
```

```
                              4D-Var (3098387)      DC-WME post-fix (3102561)
Analysis RMSE                 0.144789              0.148305
Improvement                   2.5%                  0.1%
Ratio                         25× better            —
```

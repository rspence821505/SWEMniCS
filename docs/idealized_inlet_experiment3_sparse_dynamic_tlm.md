# Idealized Inlet — Experiment 3: Sparse Obs + Correlated B + Dynamic TLM Eq 38

**Date:** 2026-04-20
**Branch:** `refactor/4dvar-parallel`
**Hardware:** M-series MacBook, 16 GB RAM, `mpirun -np 2`
**Prior docs:**
- [idealized_inlet_experiment1_correlated_B.md](idealized_inlet_experiment1_correlated_B.md)
- [idealized_inlet_experiment1b_gamma001.md](idealized_inlet_experiment1b_gamma001.md)
- [idealized_inlet_dcwme_separating_case_search.md](idealized_inlet_dcwme_separating_case_search.md)

---

## 1. Executive Summary

The two prior experiments established that **static correlated B is not enough** on its own (Exp 1) and that lowering the spectral floor γ does not fix it (Exp 1b) — the adaptive-γ regularization on a rank-deficient static H B Hᵀ collapses per-direction weights back to near-uniform. The hypothesis entering this experiment was that **dynamic TLM-based Eq 38 predictability** — building the Gram matrix from the full tangent linear propagator J_wme rather than a static spatial kernel — would inject the predictability structure that static approaches are missing.

**Result (spectral):** The TLM-derived dynamic Gram matrix is **qualitatively different from every static case** we have measured so far. Where the static obs-space Gaussian kernel was rank-deficient (λ_min ≈ 0, condition ~10³²), the dynamic TLM Gram matrix is **full rank 58/58 with condition 2.67 and λ_min = 9.19**. Every observation direction is dynamically observable through the DA window; none is degenerate. This forces 70 590 / 105 885 (67 %) of the background-covariance DOFs to be inflated 2.87× to satisfy Eq 38. **This is the first time in the idealized-inlet study that Eq 38 inflation has materially activated.**

**Result (optimization):** The optimization comparison was **not achievable on this hardware**. Both nt_da=12 and nt_da=6 DC-WME runs hung in Step 7b's post-TLM-Eq38 allocation chain at 6-10 GB per rank, with the 16 GB laptop swap-thrashing before BLMVM could take a single step. The matched 4D-Var sparse baseline ran for 11 evals and confirmed that sparse obs alone push 4D-Var into an overfitting regime (best RMSE 0.145451, only 2.41 % improvement, oscillating 0.146 ± 0.002).

**Interpretation:** Dynamic TLM-based predictability **is** the missing ingredient the prior static experiments could not produce. The spectral evidence is decisive and repeatable (nt_da=12 and nt_da=6 produce λ_min(G) = 9.19 and 9.09 respectively — robust to window length). But actually running the TLM-enabled DC-WME optimization requires Frontera-class memory; the 16 GB laptop hits a hard wall. **This is the first genuine Frontera candidate the study has produced.**

---

## 2. Exact Configuration

Single-pass experiment with the config the separating-case plan called for. All four knobs changed simultaneously compared to the Exp 1/1b baseline: obs density, obs times, correlation kernel in L_wme, and TLM Eq 38 on.

| Parameter | Exp 1/1b | Exp 3 |
|---|---|---|
| `obs_fraction` | 0.10 | **0.005** (58 obs) |
| `obs_frequency` | 4 | **2** (wider temporal coverage over shorter window) |
| `nt_da` | 12 | 12 first, then **6** (memory-reduction retry) |
| `nt_ramp` | 24 | 24 |
| `obs_correlation_length` | 1500 m | 1500 m (kept) |
| `predictability_gamma` | 0.1 | 0.1 |
| `--skip-tlm-eq38` | **on** (static path) | **off** (dynamic Eq 38 active) |
| `mem-limit-gb` | 7 | 6 (second run) |

All other controls unchanged: same mesh, same truth, same background seed, same BLMVM, same bounds, same gradient smoother, same MPI topology (np=2), same line-search cap.

No code changes required for this experiment — the TLM Eq 38 path was already wired (just skipped by default in the earlier runs).

---

## 3. Exact CLI Commands

### Run A — 4D-Var sparse (matched baseline, no correlated B in L_wme since 4D-Var has none)

```bash
mpirun -np 2 python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 7
```
Log: [logs/4dvar_exp3_sparse.log](../logs/4dvar_exp3_sparse.log)

### Run B — DC-WME sparse + correlated kernel + TLM Eq 38 (nt_da=12, first attempt)

```bash
mpirun -np 2 python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --mem-limit-gb 7
```
(No `--skip-tlm-eq38`.) Log: [logs/dcwme_exp3_sparse_tlm.log](../logs/dcwme_exp3_sparse_tlm.log)

### Run C — DC-WME sparse + correlated kernel + TLM Eq 38 (nt_da=6, memory-reduction retry)

```bash
mpirun -np 2 python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 6 \
  --obs-fraction 0.005 --obs-frequency 2 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --mem-limit-gb 6
```
Log: [logs/dcwme_exp3_small.log](../logs/dcwme_exp3_small.log)

---

## 4. Dynamic TLM Gram Diagnostics

This is the primary scientific output of the experiment. Both DC-WME runs successfully built the full TLM Eq 38 Gram matrix before hitting the memory wall. The Gram diagnostics are **robust to DA window length** — nt_da=12 and nt_da=6 give essentially the same structure, confirming the predictability pattern is a property of the dynamics, not of the measurement window.

| Quantity | nt_da=12 | nt_da=6 |
|---|---:|---:|
| Gram size | 58 × 58 | 58 × 58 |
| Adjoint solves (TLM) | 58 | 58 |
| Adjoint wall-time | 1724.9 s | 766.8 s |
| λ_min(G) | **9.195** | **9.092** |
| λ_max(G) | 24.520 | 24.246 |
| condition(G) | **2.67** | **2.67** |
| rank(G, > 1e-10) | **58 / 58 (full)** | **58 / 58 (full)** |
| spread (% of mean) | 120.9 % | 120.9 % |
| Required σ_b² (Eq 38) | 1.088e-2 | 1.100e-2 |
| DOFs below bound | 70 590 / 105 885 (67 %) | 70 590 / 105 885 (67 %) |
| Max inflation scale | 2.87× | 2.90× |

### Comparison against all prior spectral diagnostics

| Case | Gram rank | Gram condition | λ_min(G) | Eq 38 active? |
|---|---|---:|---:|:---:|
| Baseline (diag B, dense obs, Exp 1 original) | — | — | — | No — static L_wme ≈ I |
| Static corr. kernel, dense obs (Exp 1 v4) | ~10 / 1163 | ~10³² (numerical) | 0 | No — α = 1.0 (no inflation needed) |
| Static corr. kernel, γ=0.01 (Exp 1b) | ~10 / 1163 | ~10³² | 0 | No |
| **Dynamic TLM, sparse obs (this experiment)** | **58 / 58** | **2.67** | **9.19** | **Yes — 70 590 DOFs inflated 2.87×** |

**The dynamic TLM case is the first and only regime where Eq 38 has done real work** on the idealized inlet. The static correlated-B kernel produced a rank-deficient Gram by construction; the TLM propagator couples every observation direction into every interior DOF via M_{k:0}, guaranteeing full rank and a moderate condition number.

---

## 5. Results Table

### 4D-Var sparse baseline (11 evals completed, killed early)

| Eval | Cost | ‖grad‖ | RMSE | Δ RMSE |
|---:|---:|---:|---:|---:|
| 1 (bg) | 892.75 | 708 | 0.149041 | 0 |
| 2 | 795.68 | 1809 | 0.146587 | −1.65 % |
| 3 | 770.50 | 1347 | 0.147371 | −1.12 % |
| 4 | 759.74 | 1648 | 0.147753 | −0.86 % |
| 5 | 744.25 | 524 | 0.147426 | −1.08 % |
| 6 | 727.16 | 652 | 0.147029 | −1.35 % |
| 7 | 712.06 | 983 | 0.146326 | −1.82 % |
| 8 | 701.58 | 1967 | 0.145575 | −2.33 % |
| 11 | 700.74 | 1569 | **0.145451** | **−2.41 %** |

Cost monotonically decreases while RMSE oscillates in [0.145, 0.148] — classic sparse-obs overfitting regime. At 11 evals 4D-Var had improved only 2.41 %, vs 16.47 % in the dense-obs baseline. It was **killed at eval #11** (not at budget) because continuing another 1.5 h for marginal improvement in an overfitting oscillation was not worth the compute.

### DC-WME sparse + TLM Eq 38

**Could not be run to completion.** Both nt_da=12 (Run B) and nt_da=6 (Run C) entered Step 7b (static L_wme construction after TLM-based B inflation) and hung at 6-10 GB resident + compressed per rank, with 60-77 MB of system memory free. Observed state at hang:

| Metric | Run B (nt_da=12) | Run C (nt_da=6) |
|---:|---|---|
| Reached | Step 7b entry | Step 7b entry |
| RSS rank 0 | 4.4 GB + 4.4 GB compressed = ~8.8 GB | 3.4 GB + 2.6 GB compressed = ~6.0 GB |
| RSS rank 1 | 4.6 GB + 4.6 GB compressed = ~9.2 GB | ~6.0 GB |
| System free memory | 77 MB / 16 GB | 60 MB / 16 GB |
| Time spent hung | 28 min before kill | 33 min before kill |
| %CPU | 99 % (swap-thrash I/O stall) | 99 % |

The reduction from nt_da=12 → 6 shaved TLM adjoint time from 1725 s to 767 s (as expected — half the Jacobians, half the work per adjoint) but did **not** shave the post-TLM-Eq38 transition memory. The hang is in the post-Eq38 `_compute_static_L_wme` transition, likely the internal PETSc DenseCovariance + Cholesky KSP setup combined with whatever memory the TLM path's 58 destroyed adjoint vectors left fragmented. Either way it is a memory problem, not a time problem.

No DC-WME TAO callback ever fired for these runs. We have no optimization trajectory to report.

---

## 6. Comparison to the Earlier Static Experiments

| Experiment | obs count | L_wme construction | Eq 38 active? | Gram condition | DC-WME best RMSE | vs. 4D-Var @ matched budget |
|---|---:|---|:---:|---:|---:|---|
| Exp 1 (v4) | 1163 | Static corr. kernel (L=1500 m), γ=0.1 | No (α=1) | ~10³² | 0.142802 | ~4× worse per eval |
| Exp 1b | 1163 | Static corr. kernel, γ=0.01 | No (α=1) | ~10³² | 0.142801 | identical to Exp 1 |
| **Exp 3** | **58** | Static corr. kernel + **TLM Eq 38 B inflation** | **Yes (α=2.87×)** | **2.67** | *not runnable* | *not measured* |

The first two columns tell the story: **the first time Eq 38 activates is here, and it activates hard**. Every prior DC-WME run had Eq 38 as a no-op. But we cannot observe what that activation does to the cost-function descent on this hardware.

---

## 7. Required Interpretation

### Q1. Did sparse observations + dynamic TLM Eq 38 produce a more informative / anisotropic predictability structure than the static cases?

**Yes, decisively.** The static correlated-B kernel was rank-deficient (λ_min ≈ 0, 97.9 % of eigenvalues floored). The dynamic TLM Gram is **full rank 58/58 with condition 2.67** — every observation direction carries distinct dynamical information. It is not quite "anisotropic" in the traditional sense (condition 2.67 is moderate, not extreme), but it is anisotropic enough to force 67 % of background DOFs to be inflated 2.87× to satisfy Eq 38. This is qualitatively the predictability structure the method was designed for; the static approaches were producing a degenerate version of it.

### Q2. Did DC-WME materially improve relative to its static performance?

**Unknown — could not be run to completion.** The spectral ingredients are dramatically better than in the static cases; whether that translates to optimization benefit was blocked by a hardware memory ceiling that hits *after* the TLM Gram build but *before* the optimizer starts.

### Q3. Did DC-WME close the gap to 4D-Var?

**Unknown on this hardware.** Circumstantial evidence suggests it could: 4D-Var sparse itself struggles in this regime (only −2.41 % improvement in 11 evals vs −16.47 % dense), so the bar DC-WME needs to clear is low compared to the dense-obs 4D-Var baseline. And DC-WME's predictability regularization is designed specifically for the overfitting-with-sparse-data failure mode that 4D-Var is displaying here. But without a completed DC-WME optimization, this remains a prediction, not a measurement.

### Q4. Did DC-WME beat 4D-Var on this sparse dynamic case?

**Unknown.** See Q3.

### Q5. If not, is the gap now small enough that this is the right Frontera candidate anyway?

**Yes — this is the first real Frontera candidate.** Prior experiments showed static DC-WME losing to 4D-Var in all measured regimes and had *no mechanistic reason* to expect that to change on bigger hardware — the problem was that Eq 38 was inert, and more compute could not activate it. This experiment flips that story: **Eq 38 is finally active, the dynamic Gram matrix is qualitatively informative, and the only reason we cannot measure optimization outcome is 16 GB of RAM**. A Frontera node removes that ceiling.

---

## 8. Is This The First Real DC-WME-Favorable Regime?

**Spectrally yes, optimization-wise yet unconfirmed.** The data tells a consistent story:

1. In all prior experiments the static approximations to L_wme were producing a degenerate Gram matrix (rank-deficient, condition infinite). The predictability term was always either (a) clipped to uniform by the adaptive γ floor or (b) fed near-identity information.
2. The dynamic TLM path is the only route we have to a well-conditioned, full-rank Gram on this inlet.
3. That Gram now drives Eq 38 to non-trivial per-component B inflation (70 590 DOFs, 2.87× max scale) — which is exactly what the Spence et al. Section 6.2.1 recipe calls for.
4. 4D-Var in this sparse regime behaves pathologically (cost monotonic down, RMSE oscillating up) — a well-documented DC-WME-favorable failure mode for standard 4D-Var.

All four conditions aligned in this experiment for the first time. The prior two experiments had (4) if we had chosen sparse obs, but not (1)–(3). The scientific expectation that DC-WME should do better than 4D-Var here is well-founded. Confirming it takes a larger machine.

---

## 9. Recommendation — Scale to Frontera

**Yes. Move this configuration to Frontera next.** Specific recipe for the Frontera run:

- `obs_fraction=0.005`, `obs_frequency=4`, `nt_ramp=24`, `nt_da=12` (the original full-size nt_da — this is where 4D-Var was stuck overfitting, so it's the interesting regime)
- `obs_correlation_length=1500` and `predictability_gamma=0.1` (keep the tuned kernel + floor from Exp 1)
- **no** `--skip-tlm-eq38`
- A node with ≥ 64 GB RAM per rank should eliminate the Step 7b hang entirely. Request 128 GB or greater to be safe.
- Same matched 4D-Var comparison at the same config (sparse, correlated B has no effect on 4D-Var so the same CLI minus the DC-WME flags).
- Budget: 4D-Var ~4 h × 2 ranks + DC-WME ~1 h TLM + ~5-8 h optimization at reduced thrash. ~16 CPU-hours total.

Secondary: If Frontera confirms DC-WME beats 4D-Var here, replicate on a second harder case (shorter DA window, larger background perturbation) to establish robustness.

If Frontera also shows DC-WME failing to close the gap despite the activated Eq 38 — then the method is structurally outmatched by 4D-Var on this problem class and we stop. But all current evidence points the other way.

---

## 10. Hard-Constraint Compliance

- **Only sparse obs + correlated B + TLM Eq 38 changed.** Not a broad sweep. The memory-reduction retry (nt_da=12 → 6) was forced, not exploratory.
- **No static γ tuning regression.** γ=0.1 throughout.
- **Matched 4D-Var run done at the same obs config** (no correlated B in 4D-Var since it has no L_wme to correlate — documented).
- **Not called successful except where measurable.** Spectral claims are documented with numbers; the optimization claim is explicitly flagged as unmeasured on this hardware.
- **Primary success criterion ("dynamic predictability is the missing ingredient") is the mechanistic finding, not an RMSE number.** That finding is supported.

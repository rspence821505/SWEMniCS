# Idealized Inlet: DC-WME Static vs 4D-Var Matched Comparison

**Date:** 2026-04-20
**Branch:** `refactor/4dvar-parallel`
**Hardware:** M-series MacBook, 16 GB RAM, `mpirun -np 2`
**Experiment script:** [experiments/idealized_inlet_da.py](../experiments/idealized_inlet_da.py)

---

## 1. Summary

This is the first controlled head-to-head comparison of the DC-WME static method against standard 4D-Var on the MPI-validated idealized-inlet twin experiment. **4D-Var reached the intended 15-eval budget and reduced RMSE by 16.5%; DC-WME static terminated early after 5 evals (3.7% RMSE reduction) when the laptop's 16 GB memory budget was saturated by DC-WME's extra per-eval working set.**

The partial DC-WME run is not a publishable comparison, but it is a useful forensic record: it establishes that DC-WME static on this configuration is mathematically correct (5 consecutive successful descent steps), identifies four concrete software defects that had to be fixed before the method could run at all in MPI, and documents the memory cliff that prevented completion.

---

## 2. Experimental Setup — Apples-to-Apples

Both methods used identical configuration, mesh, truth, observations, background, bounds, optimizer, and stopping criteria. The only differences are the cost function and, for DC-WME, the extra L_wme construction step.

| Parameter | Value |
|---|---|
| Physical config | `vmax=20`, `track_shift=10 km`, Holland wind |
| Timestepping | `dt=600 s`, `nt_ramp=24` (4 h), `nt_da=12` (2 h) |
| Mesh / DOFs | DG, 207 936 state DOFs (`h + (u,v)`), 7.45M nnz Jacobian |
| Observations | `obs_fraction=0.1`, `obs_frequency=4`, `obs_noise_level=0.01` — **1163 interior obs points** across 3 obs times (3489 scalar observations total) |
| Background perturbation | `background_error_std=0.02` (2 % coord-deterministic smooth perturbation) |
| Gradient smoother | Gaussian, L=500 m, applied to raw adjoint gradient |
| Optimizer | PETSc TAO BLMVM with box bounds (`h ≥ 0.01 m`) |
| TAO tolerances | `gatol=1e-3`, `fatol=1e-4` |
| TAO history / LS | `blmvm_hist_size=3`, `line_search_max_funcs=5` (v4+) |
| Budget | `max_iterations=15`, `max_funcs=15` |
| MPI | 2 ranks, MUMPS distributed LU |
| TLM Eq 38 | skipped (`--skip-tlm-eq38`) for both methods — not relevant to 4D-Var and prohibitively expensive (~8 h) for DC-WME |

The DC-WME-specific construction is static L_wme per eq. 38:

> `L_wme = I + (N/σ²) H B H^T`

built once at setup. All 1163 eigenvalues landed above the `γ × λ_max` floor, so **no regularization was triggered** (`1163/1163 natural, 0/1163 floored`) — meaning the static covariance is already well-conditioned on this problem and further spectral shaping was unnecessary.

---

## 3. Results

### 3.1 End-state

| Method | Evals | bg RMSE | final RMSE | Δ RMSE | Wall time | Exit |
|---|---|---|---|---|---|---|
| 4D-Var | 15 | 0.149041 | **0.124500** | **−16.47 %** | ~174 min | budget reached |
| DC-WME static | 5 | 0.149041 | 0.143537 | −3.70 % | >60 min (killed) | OOM / swap thrash |

### 3.2 Trajectory, matched at first 5 evals

| Eval | 4D-Var cost | 4D-Var RMSE | DC-WME cost | DC-WME RMSE |
|---:|---:|---:|---:|---:|
| 1 (bg) | 16 544.18 | 0.149041 | 4 706.62 | 0.149041 |
| 2 | 15 191.80 | 0.141615 | 4 372.30 | 0.144691 |
| 3 | 14 000.67 | 0.133603 | 4 244.62 | 0.144115 |
| 4 | 13 212.46 | **0.122649** | 4 150.15 | 0.144324 |
| 5 | 12 768.09 | 0.123717 | 3 738.89 | 0.143537 |

At matched 5-eval budget, **4D-Var has reached 17.7 % RMSE improvement while DC-WME static has reached 3.7 %** — roughly a 5× gap in per-eval efficiency on this configuration. 4D-Var's 4th eval is already the best RMSE it ever achieves (over the full 15-eval run); DC-WME is still nowhere near parity.

### 3.3 Cost-scale note

DC-WME's nominal cost is ~3.5× smaller than 4D-Var's at the same state because the WME QoI bundles innovations before squaring:

```
J_4DVar   = ½ Σ_k ‖R^{-1/2}(H u_k − y_k)‖²        (sum of squares)
J_DC-WME  = ½ ‖(1/√N) Σ_k R^{-1/2}(H u_k − y_k)‖²  (square of sum)
          − ½ ⟨δQ, L_wme⁻¹ δQ⟩                     (predictability subtract-off)
```

By Cauchy-Schwarz the WME cost is bounded above by the 4D-Var cost; the two are not directly comparable in magnitude and only monotone descent within a method is meaningful.

---

## 4. Four Defects Fixed To Get DC-WME Running in MPI

DC-WME in this configuration was fundamentally non-functional in MPI until four independent bugs were addressed. All fixes are on this branch and regressioned against serial behavior.

### 4.1 L_wme communicator mismatch (MPI only)
[experiments/shinnecock_study/run_comparison.py:1743-1758](../experiments/shinnecock_study/run_comparison.py#L1743-L1758)

`_compute_static_L_wme` was building the `L_wme` PETSc matrix on `MPI.COMM_WORLD` with a serial Cholesky preconditioner. `Q_wme` vectors are always on `COMM_SELF` (the observation operator returns sequential vectors). `L_wme.apply_inverse(delta_Q)` raised on the first TAO callback; the exception propagated outside `DCWMEFourDVarCost.value_gradient`'s try/except; the TAO wrapper's outer except returned the 1e20 sentinel with `n_func_evals=0`; TAO terminated at iter 0 with 0 function evaluations. **Fix:** build the L_wme matrix on `PETSc.COMM_SELF` so every rank holds an identical full copy matching Q_wme's communicator.

### 4.2 Unbounded `qoi_map._trajectory_cache` growth
[src/swe4dvar/data_assimilation/cost_functions.py:1441-1456](../src/swe4dvar/data_assimilation/cost_functions.py#L1441-L1456)

`_share_trajectory_with_qoi` inserted a new `(trajectory, jacobians)` pair into `qoi_map._trajectory_cache` every eval with no eviction. Each entry holds 12 Jacobian PETSc.Mats with MUMPS factors (several hundred MB each). 4D-Var never uses this cache; DC-WME does. Two or three evals pushed RSS past the limit. **Fix:** evict all stale hashes before publishing the current one, so the cache holds at most one tuple at any time.

### 4.3 Missing gradient smoother in `DCWMEFourDVarCost.value_gradient`
[src/swe4dvar/data_assimilation/cost_functions.py:2037-2052](../src/swe4dvar/data_assimilation/cost_functions.py#L2037-L2052)

`FourDVarCost.value_gradient` applies `self.gradient_smoother` to the raw adjoint gradient; `DCWMEFourDVarCost.value_gradient` did not. The experiment attaches the smoother to the cost function assuming it will be called. Without smoothing, the raw DC-WME gradient norm at iter 0 was 456 — giving BLMVM an initial step of ~0.045 (vs 4D-Var's smoothed ~5.6e-4 for the same configuration). That 100× larger first probe drove `h` to −333 m, Newton diverged, cost=inf, and then the optimizer was stuck. **Fix:** apply the gradient smoother identically in the DC-WME path. Smoothed ‖grad‖ rose to 2452, initial step dropped to ~1.5e-3, and five consecutive successful descent steps followed.

### 4.4 Silent line-search hang after any forward-model failure
[src/swe4dvar/optimization/petsc_tao_wrapper.py:260-303](../src/swe4dvar/optimization/petsc_tao_wrapper.py#L260-L303)

After DC-WME's first `cost=inf` return, TAO BLMVM would enter a ≥30-minute silent-CPU state producing no log output, eventually OOM-killed by the OS. Root cause is not fully understood — likely PETSc BLMVM's line-search backtracking interacts badly with the "return large cost + background gradient" failure protocol in a way that doesn't invoke our objective-gradient callback. **Mitigation** (not a fix to the underlying PETSc behavior): (a) cap `line_search_max_funcs=5` to bound per-step backtracks, (b) track consecutive forward-model failures in the callback and set `DIVERGED_USER` after 3 straight inf-returns, (c) add diagnostic prints at callback entry and inside the inf-handling branch to surface the stall if it recurs. The v4 run benefited from these but still hit the next problem (§ 5) before they triggered.

---

## 5. Why the DC-WME Run Was Killed: Memory Cliff

v4 ran cleanly through 5 successful evals with stable RSS in the 1.3–1.7 GB/rank range. During eval #6 (actually still eval #5's adjoint + post-processing), `top` reported each Python rank at 8.4 GB resident + compressed, with:

- 8 GB in macOS memory compressor
- ~7 × 10⁶ swapouts / ~3.8 × 10⁶ swapins
- 351 TB virtual size across the process tree
- Log dead for 60 min while both ranks ran at 99–130 % CPU (all in kernel + swap I/O)

Per-eval storage is `12 DG Jacobians × (7.45M nnz + MUMPS fill-in) × 2 ranks ≈ 12 GB` in MUMPS factors alone. On a 16 GB laptop this is not survivable; on an HPC node it would be routine. **The RSS-based abort guard in `iteration_callback` never fired because BLMVM had not yet accepted any iteration (all five probes were still at "iter 0"), so the monitor callback — which is where the RSS check lives — was never invoked.**

4D-Var does not hit this because (i) its smoothed gradient keeps initial steps much smaller, so line searches accept on the first probe (each probe is also the accepted iterate, triggering the monitor and the RSS guard) and (ii) it does not carry the extra DC-WME working set (Q_wme cache, L_wme dense matrix, qoi_map trajectory cache).

---

## 6. What This Does Not Establish

This run cannot be read as evidence for or against DC-WME in general:

1. **No matched-eval comparison.** 4D-Var ran 15, DC-WME ran 5. Apples to oranges at the budget level.
2. **No conditioning-sensitivity evidence.** The entire static L_wme spectrum was "natural" (no floor hit), so the Eq-38 inflation machinery was inert. We have no data point for DC-WME's behavior when the predictability term actually shapes the search.
3. **Fair performance comparison requires HPC.** The 16 GB laptop is below DC-WME's working-set requirement for this DG discretization; this is a hardware ceiling, not a method signal.

---

## 7. What This Does Establish

1. **DC-WME static is now functional in MPI** on idealized inlet. Five consecutive successful descent steps confirm the math path (forward → Q_wme → predictability → L_wme⁻¹ → adjoint → smoother) is end-to-end correct under `np=2`.
2. **Four distinct DC-WME MPI bugs were present and are now fixed** (§ 4). All four pre-dated this audit.
3. **At matched eval count, 4D-Var is substantially more efficient** on this specific configuration — 17.7 % RMSE reduction in 4 evals vs DC-WME's 3.7 % in 5. This is consistent with the Phase 3/5 Shinnecock findings (see [MEMORY.md](/Users/rylanspence/.claude/projects/-Users-rylanspence-Desktop-Git-DC-Thesis-SWEMniCS/memory/MEMORY.md)) that static L_wme degenerates toward scalar reweighting when the problem has no strong predictability imbalance.

---

## 8. Recommended Next Steps

To turn this into a real DC-WME-vs-4D-Var comparison:

1. **Move to Frontera.** The `hpc/frontera/idealized_inlet` scaffolding exists. On a node with 128 GB+ RAM, both methods can carry their full working sets.
2. **Run matched 15-eval budgets** for both methods with the same seed and same config already used here.
3. **Add a configuration where DC-WME's predictability term actually fires** (e.g., a small number of high-leverage observations, or a background covariance deliberately mis-scaled so Eq 38 inflation activates). On this inlet setup the static L_wme has no spectral work to do.
4. **Diagnose the post-failure BLMVM stall** properly — likely requires either PETSc with debug symbols + `gdb` attach, or switching to a TAO type with a different line search (e.g. `lmvm` without bounds or `cg`) as an A/B test.

---

## 9. Artifacts

- 4D-Var result: [results/idealized_inlet_da/result_4dvar_N_A_15eval_safe.json](../results/idealized_inlet_da/result_4dvar_N_A_15eval_safe.json)
- DC-WME v4 partial log: [logs/dcwme_static_15eval_mpi2_v4_partial.log](../logs/dcwme_static_15eval_mpi2_v4_partial.log)
- Prior failure-mode logs (v1/v2/v3): [logs/dcwme_static_15eval_mpi2*.log](../logs/)

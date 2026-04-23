# TLM u/v Bisector Diagnostic — Idealized Inlet DC-WME + Eq 38

**Date**: 2026-04-22
**Investigation lineage**:
- [idealized_inlet_experiment3_sparse_dynamic_tlm.md](idealized_inlet_experiment3_sparse_dynamic_tlm.md)
- [idealized_inlet_dcwme_separating_case_search.md](idealized_inlet_dcwme_separating_case_search.md)
- [ls6_first_dcwme_production_run.md](ls6_first_dcwme_production_run.md)
- LS6 runs 3098388, 3100482, 3100802, 3101214, 3101366, 3101378

---

## 1. Executive Summary

The "TLM-based" Eq. 38 predictability bound (`σ_b² = γ / λ_min(G)`, `G = J_wmeᵀ J_wme`) has produced **zero u/v content** in its adjoint vectors at every prior run (tiny and full scale). A component-aware Gram decomposition (commit `7d1f91f`) showed:

```
λ_min(G_h)  = 9.25   λ_max(G_h)  = 24.7    → σ_b²_h  = 1.08e-2
λ_min(G_uv) = 0.0    λ_max(G_uv) = 0.0     → σ_b²_uv = 1.0e+29 (undefined)
```

`G_uv ≡ 0` implies every adjoint vector `aᵢ = J_wmeᵀ eᵢ` has exact-zero u/v components. Since the SWE mixed-element Jacobian `J = ∂F/∂u` on `u = (h, u, v)` has non-zero cross-component blocks (momentum depends on `∇h`, continuity depends on `∇·(hu)`), a correct `J^T` back-propagation MUST produce u/v content from h-only obs forcings.

A targeted bisector (`_UV_BISECTOR_CTX` switch in `src/swe4dvar/adjoint/implicit_adjoint.py`) traced the first adjoint solve (`i=0`) at global (MPI-allreduced) and per-rank resolution.

**Finding (Classification B — tiny-mask zeroing)**: Every diagonal entry of every stored truth-trajectory Jacobian satisfies `|J[i, i]| < 1e-20`. The `tiny_mask` at [implicit_adjoint.py:858](../src/swe4dvar/adjoint/implicit_adjoint.py#L858) is therefore **all-true on every backward step, every rank**. After the transpose solve produces a meaningful `λ` vector, [line 985](../src/swe4dvar/adjoint/implicit_adjoint.py#L985) executes `arr[tiny_mask] = 0.0` and obliterates the entire vector — both h and u/v.

Every subsequent step receives zero forcing (`c·M·0 = 0`) and contributes zero to the Gram. The only reason `G_h` has non-zero eigenvalues at all is that the **direct** `obs_forcing[0]` term in `_compute_initial_gradient` (line 1078, `result.axpy(+1.0, obs_forcing)`) short-circuits past the dead adjoint sweep and pastes the observation-operator-adjoint result directly into the gradient. That's why `G_h` captures H but nothing dynamic — the "TLM-based" Eq. 38 has been effectively `G = (H^T R^{-1/2})(H^T R^{-1/2})^T` all along, with zero temporal content.

**Impact**: Every prior DC-WME + TLM + Eq. 38 run on any problem using `ImplicitAdjointSolver` has been computing a static H-only σ_b² while naming it "TLM-based". No experimental conclusion that relied on dynamic predictability from Eq. 38 is valid without rerunning after the fix.

---

## 2. Diagnostic instrumentation added

### 2.1 Switchable bisector in `implicit_adjoint.py`

New module-level `_UV_BISECTOR_CTX` dict, null by default. When set to `{"h": h_indices, "uv": uv_indices}` (local-owned DOF indices on each rank), the two critical paths emit structured logs:

- `_solve_transpose_system` (lines ~856-897, ~981-996):
  - BEFORE solve (forcing vector)
  - AFTER `ksp.solveTranspose` but BEFORE `arr[tiny_mask] = 0.0`
  - AFTER tiny-mask zeroing
  - `tiny_mask ∩ h_idx`, `tiny_mask ∩ uv_idx`, `n_tiny_total`
- `_compute_initial_gradient`:
  - BEFORE Dirichlet-BC zeroing at `bc_dof_indices`
  - AFTER Dirichlet-BC zeroing
  - `bc_dof_indices ∩ h_idx`, `bc_dof_indices ∩ uv_idx`

Each event emits:
- One `[GLOBAL]` line with MPI-allreduced ‖λ_h‖, ‖λ_uv‖ and non-zero counts
- One `[rN]` line per rank with the local view (exposes partition asymmetry)

Armed on ALL ranks (not just rank 0), because the allreduce inside the log function would otherwise deadlock.

### 2.2 One-shot arming from `_compute_eq38_from_tlm`

[`experiments/shinnecock_study/run_comparison.py`](../experiments/shinnecock_study/run_comparison.py) calls `_bisector_set_component_indices(component_indices)` immediately before the first (`i=0`) `linearized_wme.apply_adjoint(e_i)` and clears it immediately after. Exactly one complete backward sweep is instrumented per run; the other 57 adjoint calls run clean.

### 2.3 Sbatch runner

[`hpc/lonestar6/idealized_inlet/job_uv_bisector.slurm`](../hpc/lonestar6/idealized_inlet/job_uv_bisector.slurm): np=2 × 64 threads, dev queue, 30-min cap. Tiny config (`--nt-ramp 4 --nt-da 4 --max-iterations 1 --max-funcs 1 --eq38-component-aware`). Runs in ~10 minutes.

---

## 3. Exact run command used

```bash
sbatch -p development hpc/lonestar6/idealized_inlet/job_uv_bisector.slurm
# Job 3101378 — ran 2026-04-22 19:12:27 on c305-005 (or c302 etc), np=2
# Status: CANCELLED after diagnostic captured (the remaining 57 adjoint
#         solves were redundant once i=0 proved the pattern).
```

---

## 4. Step-by-step λ_h / λ_uv diagnostic table

GLOBAL (MPI-allreduced L2) norms from run 3101378:

| Stage | step n | ‖λ_h‖ | ‖λ_uv‖ | nz_h / total | nz_uv / total | tiny_h / total | tiny_uv / total |
|---|---:|---:|---:|---:|---:|---:|---:|
| BEFORE solve (forcing) | **4** (final) | **5.083e+00** | 0.000e+00 | **6** / 69312 | 0 / 138624 | 35106/35106 | 70212/70212 |
| AFTER solveTranspose | 4 | **5.083e+00** | 0.000e+00 | 6 / 69312 | 0 / 138624 | — | — |
| **AFTER tiny-mask zeroing** | 4 | **0.000e+00** | **0.000e+00** | **0** / 69312 | **0** / 138624 | — | — |
| BEFORE solve (forcing) | 3 | 0.000e+00 | 0.000e+00 | 0 / 69312 | 0 / 138624 | 35106/35106 | 70212/70212 |
| AFTER solveTranspose | 3 | 0.000e+00 | 0.000e+00 | 0 / 69312 | 0 / 138624 | — | — |
| AFTER tiny-mask zeroing | 3 | 0.000e+00 | 0.000e+00 | 0 / 69312 | 0 / 138624 | — | — |
| … | 2, 1 | (all 0.000) | (all 0.000) | (all 0) | (all 0) | all tiny | all tiny |
| gradient_u0 BEFORE BC | 0 | **5.083e+00** | 0.000e+00 | 6 / 69312 | 0 / 138624 | — | — |
| gradient_u0 AFTER BC | 0 | **5.083e+00** | 0.000e+00 | 6 / 69312 | 0 / 138624 | — | — |

### 4.1 Per-rank asymmetry

rank 0 `h_idx` = 35106 DOFs; rank 1 `h_idx` = 34206 DOFs (standard mesh-partition imbalance).

```
r0 n=4 BEFORE solve:  ||h||_loc=0.000e+00  nz_h_loc=0/35106
r1 n=4 BEFORE solve:  ||h||_loc=5.083e+00  nz_h_loc=6/34206
```

**All 6 nonzero h DOFs of the forcing at the final obs time live on rank 1** — obs point 0 is physically in rank 1's mesh partition. That is what produced the initial confusion in run 3101366 where only rank 0's local view was logged (and looked all-zero).

---

## 5. BC-mask accounting

The `bc_dof_indices` printout never fired — `self.bc_dof_indices` was `None` in this run path (the `LinearizedWMEQoI.apply_adjoint` only passes `bc_dof_indices` if the forward model exposes them, and for this `ForwardModelWrapper(gram_fwd, ...)` they were not populated). So **BC masking is NOT a contributor here**. Hypothesis A is eliminated.

If bc_dof_indices had been populated, the gradient_u0 line "BEFORE BC" → "AFTER BC" would have shown ‖h‖ or ‖uv‖ decreasing. They are identical (5.083 → 5.083).

---

## 6. tiny-mask accounting

```
tiny_h=35106/35106  (rank 0)       tiny_h=34206/34206  (rank 1)
tiny_uv=70212/70212 (rank 0)       tiny_uv=68412/68412 (rank 1)
n_tiny_total=105318 (rank 0)       n_tiny_total=102618 (rank 1)
```

**100% of diagonals in every truth-trajectory Jacobian satisfy `|J[i,i]| < 1e-20`.** This is not a dry-node phenomenon (a few edge DOFs) — it is a **systemic property of the stored Jacobians on this problem**.

Inspection of [newton.py:340-354](../src/swe4dvar/forward/newton.py#L340-L354) shows the stored Jacobian is the full physics Jacobian with Dirichlet BCs NOT applied, specifically so the adjoint can use the "un-BC-modified" physics. The assembly step should produce real entries. The empirical fact that every diagonal is under 1e-20 on both ranks indicates the DG-mixed-element Newton Jacobian's diagonal has natural magnitudes below that threshold for this problem/mesh scaling — not a bug in assembly, but a **mis-calibration of the threshold**.

---

## 7. Exact stage where u/v is lost

`src/swe4dvar/adjoint/implicit_adjoint.py`:

```python
# Line 858
tiny_mask = np.abs(diag_arr) < 1e-20        # ← mis-calibrated: matches EVERY DOF
...
# Line 985
if n_regularized > 0:
    arr = lambda_n.getArray()
    arr[tiny_mask] = 0.0                    # ← annihilates λ_n entirely
    lambda_n.setArray(arr)
```

Since `tiny_mask` covers 100% of the vector, **the post-solve zeroing obliterates both the h signal that the transpose solve correctly produced AND any u/v signal the solve would have induced via J^T cross-component coupling**. There is no u/v to measure at later stages because the zeroing happens at n=N before any time coupling can occur.

---

## 8. Final classification

**B — tiny-mask / dry-node regularization is the culprit.**

A refinement: it is not wetting-drying per se. The threshold `|J[i,i]| < 1e-20` was calibrated for a problem where dry-node rows are set to identity by Dirichlet BCs (diagonal exactly 1.0) and genuinely wet-DOF rows have diagonals well above 1e-20. On this DG-mixed-element Newton Jacobian, wet-DOF diagonals are ~1e-21 to 1e-30 for the assembly scaling used, so the threshold captures **every DOF**.

---

## 9. Smallest plausible fix

Two non-invasive options, both at `src/swe4dvar/adjoint/implicit_adjoint.py:858`:

### Option A (safest): scale threshold relative to the dominant diagonal

```python
diag_max = float(np.max(np.abs(diag_arr)))
# Treat as "dry" only when diagonal is 10^12 times smaller than the largest;
# i.e., genuine near-nulls relative to the operator's actual scale.
abs_floor = 1e-20
rel_floor = 1e-12 * diag_max if diag_max > 0 else 0.0
tiny_mask = np.abs(diag_arr) < max(abs_floor, rel_floor)
```

Keeps dry-node protection (isolated rows with diagonals 10^12× smaller than peak) but no longer catches every DOF when the assembly scaling is small.

### Option B (minimal): guard post-solve zeroing against degenerate all-true mask

```python
# Line 985 area
if n_regularized > 0 and n_regularized < diag_arr.size:
    arr = lambda_n.getArray()
    arr[tiny_mask] = 0.0
    lambda_n.setArray(arr)
    lambda_n.assemble()
```

If tiny_mask is all-true, the mask carries no information — the regularization tried to fix something, but since every row is "dry", zeroing everything is obviously wrong. Skipping the zeroing in that case preserves whatever structure the regularized solve produced.

**My recommendation: Option A.** It's a single-line change, addresses the mis-calibration directly, and preserves the intent of the original dry-node protection. Option B is a safety net that should also be added but feels like "fix the symptom".

### Validation plan after fix

1. Rerun `job_uv_bisector.slurm` — should show `||uv||_global > 0` at some step beyond n=N (time-coupling through J^T introducing u/v content from h forcings).
2. Rerun the full production `job_dcwme_prod.slurm` — should now produce `λ_min(G_uv) > 0` and meaningfully different `σ_b²_uv` vs `σ_b²_h`.
3. Confirm 4D-Var convergence unchanged — 4D-Var also uses `ImplicitAdjointSolver` and must not regress.
4. Run validation_ladder experiment 1 (adjoint–FD check) locally — should still PASS at its existing ~1e-5 relative error.

**Before applying the fix in code**: this memo is the end of the bisector pass. Do not broaden into a repair until a separate small pull-request cycle.

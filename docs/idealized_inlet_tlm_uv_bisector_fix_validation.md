# TLM u/v Bisector Fix Validation — Threshold Hypothesis DISPROVEN

**Date**: 2026-04-22
**Related**: [idealized_inlet_tlm_uv_bisector.md](idealized_inlet_tlm_uv_bisector.md)
**LS6 run**: 3101386 (commit `1bb99d8`, cancelled after i=0 sweep captured)
**Outcome**: **3 — No real change. Revised hypothesis required.**

---

## 1. Executive summary

The narrow threshold fix at [src/swe4dvar/adjoint/implicit_adjoint.py:917-934](../src/swe4dvar/adjoint/implicit_adjoint.py#L917-L934) replaced the absolute tiny-mask threshold `|diag| < 1e-20` with a MPI-collective relative threshold `|diag| < max(1e-20, 1e-12·max_global|diag|)`. The re-bisector reports:

```
diag_max    = 0.000e+00       (12/12 log events, both ranks, every backward step)
tiny_thresh = 1.000e-20       (relative floor falls back to absolute floor)
tiny_h  = 35106/35106 (r0),  34206/34206 (r1)       — unchanged
tiny_uv = 70212/70212 (r0),  68412/68412 (r1)       — unchanged
‖λ_h‖ AFTER tiny-mask zeroing = 0.000e+00           — unchanged
‖λ_uv‖ AFTER tiny-mask zeroing = 0.000e+00          — unchanged
```

The diagonal of the stored Jacobian is **exactly zero** at every row on every stored timestep on every rank — not small, not sub-threshold. The relative threshold falls back to the absolute floor (`rel_floor = 1e-12·0 = 0`), so the mask continues to catch every DOF, and the post-solve zeroing continues to obliterate the entire adjoint vector.

**Classification B (tiny-mask threshold mis-calibration) is disproven.** The tiny-mask safeguard is a *proximate symptom*, not the root cause. The actual structural problem is upstream: `J.getDiagonal()` returns all zeros on the Jacobians `self.jacobians[k]` fed to the adjoint. This is either (i) a legitimate property of the DG-mixed-element Newton Jacobian storage, or (ii) a bug where the intended physics+mass diagonal is absent from the stored matrix.

---

## 2. Exact code patch made

```diff
--- a/src/swe4dvar/adjoint/implicit_adjoint.py
+++ b/src/swe4dvar/adjoint/implicit_adjoint.py
@@ -914,9 +914,26 @@ class ImplicitAdjointSolver:
         # Dry nodes produce zero rows in J, making J^T singular.
         # We regularize by using a shifted operator J + εI for the solve,
         # then zero out the adjoint at dry-node DOFs afterward.
+        #
+        # Threshold is RELATIVE to the largest diagonal magnitude of J, with
+        # an absolute floor. ...
         diag = J.getDiagonal()
         diag_arr = diag.getArray()
-        tiny_mask = np.abs(diag_arr) < 1e-20
+
+        # Collective MAX of |diag| across ranks so all ranks agree on scale.
+        from mpi4py import MPI as _MPI
+        local_max = float(np.max(np.abs(diag_arr))) if diag_arr.size > 0 else 0.0
+        comm_j = J.getComm().tompi4py()
+        diag_max = comm_j.allreduce(local_max, op=_MPI.MAX)
+
+        abs_floor = 1e-20
+        rel_floor = 1e-12 * diag_max if diag_max > 0.0 else 0.0
+        tiny_thresh = max(abs_floor, rel_floor)
+        tiny_mask = np.abs(diag_arr) < tiny_thresh
         n_regularized = int(np.sum(tiny_mask))
         diag.destroy()
```

Plus `diag_max` and `tiny_thresh` added to the bisector log extras so both are visible at every step.

Commit: `1bb99d8 fix(adjoint): scale tiny-mask threshold to diag_max`.

---

## 3. Old vs new tiny-mask behavior

| Aspect | Baseline (3101378, commit `421076c`) | After fix (3101386, commit `1bb99d8`) |
|---|---|---|
| Threshold formula | `\|diag\| < 1e-20` | `\|diag\| < max(1e-20, 1e-12·diag_max)` |
| `diag_max` (observed) | n/a (not logged) | **0.000e+00** (12/12 events) |
| `tiny_thresh` (effective) | 1.000e-20 | 1.000e-20 (rel-floor=0 → abs-floor wins) |
| `n_tiny_total` (r0) | 105318/105318 | **105318/105318** (identical) |
| `n_tiny_total` (r1) | 102618/102618 | **102618/102618** (identical) |
| `tiny_h` fraction | 100% | **100%** (no change) |
| `tiny_uv` fraction | 100% | **100%** (no change) |

The fix is inert on this problem because `diag_max ≡ 0`.

---

## 4. Step-by-step λ_h / λ_uv table

GLOBAL (MPI-allreduced L2) norms, run 3101386:

| Stage | step n | ‖λ_h‖ | ‖λ_uv‖ | nz_h | nz_uv | tiny_h/tot | tiny_uv/tot | diag_max | tiny_thresh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BEFORE solve (forcing) | **4** | **5.083e+00** | 0.000e+00 | 6/69312 | 0/138624 | 35106/35106 | 70212/70212 | **0.0** | 1.0e-20 |
| AFTER solveTranspose | 4 | **5.083e+00** | 0.000e+00 | 6/69312 | 0/138624 | — | — | — | — |
| **AFTER tiny-mask zeroing** | 4 | **0.000e+00** | **0.000e+00** | 0/69312 | 0/138624 | — | — | — | — |
| BEFORE / AFTER solve | 3 | 0.000e+00 | 0.000e+00 | 0 | 0 | 35106/35106 | 70212/70212 | 0.0 | 1.0e-20 |
| AFTER tiny-mask zeroing | 3 | 0.000e+00 | 0.000e+00 | 0 | 0 | — | — | — | — |
| BEFORE / AFTER / zeroing | 2 | 0.000e+00 | 0.000e+00 | 0 | 0 | 35106/35106 | 70212/70212 | 0.0 | 1.0e-20 |
| BEFORE / AFTER / zeroing | 1 | 0.000e+00 | 0.000e+00 | 0 | 0 | 35106/35106 | 70212/70212 | 0.0 | 1.0e-20 |
| gradient_u0 BEFORE BC | 0 | **5.083e+00** | 0.000e+00 | 6/69312 | 0/138624 | — | — | — | — |
| gradient_u0 AFTER BC | 0 | **5.083e+00** | 0.000e+00 | 6/69312 | 0/138624 | — | — | — | — |

The raw per-rank split matches the baseline exactly: the 6 nonzero h-forcing DOFs all live on rank 1 (the rank whose partition contains obs point 0). r0 sees zero locally at n=4 but the GLOBAL allreduce confirms the content is present.

---

## 5. Four required checks

### A. Does `n_tiny_h` and `n_tiny_uv` drop from "all DOFs" to something plausible?
**NO.** Identical to baseline: `tiny_h = 35106/35106 (r0), 34206/34206 (r1)`; `tiny_uv = 70212/70212 (r0), 68412/68412 (r1)`. Because `diag_max = 0`, the relative term `1e-12·diag_max = 0`, so the effective threshold remains `1e-20` and every diagonal (all of which are `< 1e-20`, in fact exactly zero) is still caught.

### B. Does `‖λ_uv‖` survive after the transpose solve?
**NO.** `‖λ_uv‖ = 0` at AFTER solveTranspose on every step. This is not caused by tiny-mask zeroing — it is already zero *before* the mask is applied.

### C. Does `‖λ_uv‖` still survive after the post-solve masking stage?
**NO.** Still zero (trivially, since it was zero before).

### D. Does `‖λ_h‖` remain numerically sensible and not become pathological?
Partially. The raw `solveTranspose` output at `n=4` preserves `‖λ_h‖ = 5.083` (same as forcing — so the transpose "solve" is effectively an identity pass-through through a regularized J_reg whose only diagonal populated is the 1.0 dry-node identity block). The subsequent `tiny_mask` zeroing then wipes it to 0. Same pathology as baseline.

---

## 6. u/v survival after transpose solve
`‖λ_uv‖ = 0.000e+00` at n=4 AFTER solveTranspose. The transpose solve is **not generating any u/v content from the h-forcing.** This is the key structural finding of this re-bisector: even if the subsequent tiny-mask zeroing were disabled, u/v would still be zero because the solve itself yields no cross-component coupling. This rules out tiny-mask as the mechanism by which u/v content is destroyed — **there is no u/v content to destroy.**

## 7. u/v survival after post-solve masking
N/A — u/v is already zero before the mask.

---

## 8. Final outcome classification

### **Outcome 3 — No real change. The hypothesis must be revised.**

Evidence:
1. `diag_max = 0` exactly on 12/12 log events → the relative-threshold fix degenerates to the original absolute-threshold behavior.
2. `tiny_h`, `tiny_uv`, `n_tiny_total` are byte-identical to the baseline.
3. `‖λ_uv‖ = 0` at AFTER solveTranspose (pre tiny-mask) → the solve itself does not produce u/v content, meaning tiny-mask was not the mechanism destroying it.
4. `‖λ_h‖` at AFTER solveTranspose equals the forcing norm exactly (5.083 → 5.083) → the solve is pass-through, consistent with a J_reg whose only non-zero structure is the identity mask from dry-node regularization.

The conclusion of [idealized_inlet_tlm_uv_bisector.md §6](idealized_inlet_tlm_uv_bisector.md) was that the *threshold* was mis-calibrated. That is now disproven. The *threshold* is fine; the **input** — the stored Jacobian's diagonal — is exactly zero.

### Revised hypothesis (not yet validated): **C — Stored Jacobian structural zero-diagonal.**

`J.getDiagonal()` returns a zero vector on every `self.jacobians[k]`. Candidate root causes, ranked:

1. **DG-mixed-element AIJ layout quirk (most likely)**: PETSc's `MatGetDiagonal` on a block-structured AIJ matrix assembled from a DOLFINx DG mixed-element `(h, (u, v))` form may not align the "matrix diagonal" with what we intuitively think of as per-DOF self-coupling. The assembled `A` contains physics (diagonals of `α₀·M` plus `∂F/∂u` blocks) but its AIJ diagonal entries may be zero because the assembled sparsity pattern places the natural diagonal contributions at off-diagonal column indices due to DOF interleaving within each cell block.
2. **Assembly path mismatch**: `return_jacobian` path at [newton.py:346-353](../src/swe4dvar/forward/newton.py#L346-L353) reassembles `A` without BCs and `.copy()`s it, but the BC-modified intermediate solve may leave `A` in a state where reassembly produces a diagonally-empty matrix (e.g., if the form is reduced to flux-only terms in the un-BC version).
3. **Pre-existing bug elsewhere**: something between newton.py's `A.copy()` and the adjoint consumption mutates the matrix or substitutes a different one.

None of these is confirmed. All require a separate short diagnostic.

---

## 9. Recommended next step

**One more focused diagnostic, narrower than this pass. Then decide.**

Propose a ~15-line patch that, on the first stored Jacobian only (a single timestep, i=0 adjoint sweep), prints:

- `J.getSize()`, `J.getLocalSize()`, `J.getBlockSize()`
- number of nonzeros: `J.getInfo()['nz_used']`
- histogram of `|entry|` by magnitude bucket (count of entries in `[0, 1e-20)`, `[1e-20, 1e-10)`, `[1e-10, 1e-5)`, `[1e-5, ∞)`)
- max/min/mean of |diag|
- max/min/mean of |off-diagonal|
- a 3×3 dense snapshot from the upper-left corner (MPI-safe)

Goal: in **one** diagnostic we learn whether the Jacobian has any content at all and if so where it lives. Three outcomes:

- **C1 — J has nonzero off-diag content, zero diag**: DG AIJ layout quirk. Fix at the adjoint path (e.g., use `J.norm()` / per-row checks, not `getDiagonal()`, for the dry-node criterion). Plus need to understand how PETSc LU factorizes a zero-diagonal matrix (pivoting required).
- **C2 — J is structurally empty (all zeros)**: the Jacobian being fed to the adjoint isn't the physics Jacobian at all. Fix is upstream in `newton.py` storage path.
- **C3 — J has nonzero diag but `getDiagonal()` returns zero**: PETSc API quirk, probably MPI-related. Fix is to use a different API.

Keep the patch as narrow as the tiny-mask patch was — single site, single diagnostic, no broader refactor. Write a `docs/idealized_inlet_stored_jacobian_diagnostic.md` with the result and pick the fix direction.

**Do not** yet:
- apply any Jacobian-storage fix in newton.py
- reinstate the old absolute threshold (the relative threshold is correct in principle; if diag_max ever becomes nonzero for this problem, it will behave properly)
- rerun production DC-WME or 4D-Var — they have the same blocker upstream
- launch comparison passes

The threshold fix itself is harmless and correct in principle, so it can stay in. The blocker is now clearly upstream of the tiny-mask.

---

## Appendix: links

- Baseline bisector memo: [idealized_inlet_tlm_uv_bisector.md](idealized_inlet_tlm_uv_bisector.md)
- Re-bisector stdout: LS6 `$WORK/SWEMniCS/inlet_uvbis.3101386.out` (cancelled after i=0, job state `CG`)
- Patched implicit_adjoint.py: [src/swe4dvar/adjoint/implicit_adjoint.py:917-934](../src/swe4dvar/adjoint/implicit_adjoint.py#L917-L934)
- Newton Jacobian return path: [src/swe4dvar/forward/newton.py:332-358](../src/swe4dvar/forward/newton.py#L332-L358)

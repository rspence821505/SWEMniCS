# R2 Jacobian Handoff Trace — Idealized Inlet

**Date**: 2026-04-22
**Related**: [idealized_inlet_jacobian_reassembly_trace.md](idealized_inlet_jacobian_reassembly_trace.md) · [idealized_inlet_stored_jacobian_diagnostic.md](idealized_inlet_stored_jacobian_diagnostic.md) · [idealized_inlet_tlm_uv_bisector_fix_validation.md](idealized_inlet_tlm_uv_bisector_fix_validation.md) · [idealized_inlet_tlm_uv_bisector.md](idealized_inlet_tlm_uv_bisector.md)
**LS6 runs**: 3102458 (localization, commit `869d7a7`) and 3102465 (post-fix confirmation, commit `3852e6f`)
**Outcome**: **Collapse localized between `after_next_zeroEntries` and `adjoint_init` — root cause is `PETSc.Mat.duplicate()` without `copy=True` at [experiments/idealized_inlet_da.py:261](../experiments/idealized_inlet_da.py#L261). One-line fix confirmed.**

---

## 1. Executive summary

The R2 handoff trace traced the stored Jacobian through five boundary probes. Four probes preserved the real Frobenius norm (`6.642e+06`); the fifth — `ImplicitAdjointSolver.__init__` — saw **a different PETSc.Mat object with norm `0.000e+00` and identical sparsity pattern (`nz = 7,450,866`)**. The ID change localized the zeroing to a code path that *allocates new matrices from the sparsity of the saved ones without copying values.*

That code path is [experiments/idealized_inlet_da.py:261](../experiments/idealized_inlet_da.py#L261):

```python
truth_jacobians = [J.duplicate() for J in solver_truth.storage.saved_jacobians]
```

`PETSc.Mat.duplicate()` defaults to `copy=False`, which preallocates a matrix with the source's sparsity pattern but **leaves the values unset (zero)**. Immediately afterwards [idealized_inlet_da.py:268](../experiments/idealized_inlet_da.py#L268) calls `solver_truth.storage.clear()` which destroys the real originals. What gets passed into `_compute_eq38_from_tlm` and eventually `ImplicitAdjointSolver` is therefore a list of structurally-correct but numerically-empty matrices — exactly the "C2" object the structural diagnostic observed three memos ago.

**One-line fix**: `[J.duplicate(copy=True) for J in ...]`. Confirmed by re-running the same bisector: `adjoint_init_idx{0,1}` now report `6.642e+06` / `6.646e+06`, and the uv-bisector's backward sweep now produces nonzero u/v content at every stored step. The threshold patch from commit `1bb99d8` (relative `tiny_thresh`) remains correct and now actively protects against the handful of near-zero DG facet rows: `diag_max = 9.676e+04`, `tiny_thresh = 9.676e-08`, `tiny_h = tiny_uv = 0` — no dry-node false positives.

---

## 2. Exact instrumentation added

### Pre-existing (from R1 pass, commit `9385fa5`)
`[jac-reassembly]` probe inside `newton.py` post-convergence `return_jacobian` block.

### New in R2 (commit `3889d86`, collective-deadlock fix `869d7a7`)

Module-level `_HANDOFF` dict + `_jac_handoff_log` helper in [src/swe4dvar/utils/solver_storage.py](../src/swe4dvar/utils/solver_storage.py). The helper is rank-collective on `.norm()` and `.getInfo()` but rank-0-only on the `print`:

```python
_HANDOFF = {
    "last_saved": None,
    "cg_entry_fired": False,
    "storage_fired": False,
    "postzero_fired": False,
    "adjoint_fired": False,
}

def _jac_handoff_log(stage, mat, extras=None):
    from mpi4py import MPI as _MPI
    rank = _MPI.COMM_WORLD.Get_rank()
    if mat is None:
        if rank == 0:
            print(f"[jac-handoff] stage={stage} mat=None", flush=True)
        return
    norm = mat.norm(PETSc.NormType.NORM_FROBENIUS)   # COLLECTIVE
    nz = int(mat.getInfo().get("nz_used", -1))        # COLLECTIVE
    if rank == 0:
        msg = f"[jac-handoff] stage={stage} norm={norm:.3e} nz={nz} id={id(mat)}"
        if extras:
            for k, v in extras.items():
                msg += f" {k}={v}"
        print(msg, flush=True)
```

Probe sites (all one-shot, guarded by dict flags):

| Site | File | Purpose |
|---|---|---|
| `cg_implicit_entry` | [cg_implicit.py:484](../src/swe4dvar/forward/solvers/cg_implicit.py#L484) `save_jacobians` | J just as it crosses into the solver wrapper |
| `storage_pre_copy` | [solver_storage.py:95](../src/swe4dvar/utils/solver_storage.py#L95) `save_jacobian` | input `jacobian` before `.copy()` |
| `storage_post_copy` | same | `saved_jacobians[-1]` after append, with `aliased_input` flag |
| `after_next_zeroEntries` | [newton.py:183](../src/swe4dvar/forward/newton.py#L183) inside Newton loop, after `A.zeroEntries()` of the *next* time step | re-reads `_HANDOFF["last_saved"]` to test the aliasing hypothesis |
| `adjoint_init_idx0/1` | [implicit_adjoint.py:480](../src/swe4dvar/adjoint/implicit_adjoint.py#L480) `__init__` | first two matrices in the jacobians list at adjoint construction |

---

## 3. Exact run commands used

```bash
# Localization run (commit 869d7a7)
ssh ls6 'cd $WORK/SWEMniCS && sbatch hpc/lonestar6/idealized_inlet/job_uv_bisector.slurm'
# → Submitted batch job 3102458

# Post-fix confirmation run (commit 3852e6f adds `copy=True` at idealized_inlet_da.py:261)
ssh ls6 'cd $WORK/SWEMniCS && sbatch hpc/lonestar6/idealized_inlet/job_uv_bisector.slurm'
# → Submitted batch job 3102465

# Both cancelled after the probes of interest had emitted.
```

Config (shared): `--nt-ramp 4 --nt-da 4 --max-iterations 1 --max-funcs 1 --method dcwme_static --eq38-component-aware` at np=2.

---

## 4. Norm at each handoff boundary

### Localization run 3102458 (before fix)

| Stage | Frobenius norm | `nz_used` | `id(mat)` |
|---|---:|---:|---|
| `[jac-reassembly]` `copy` (newton.py exit) | 6.642e+06 | 7,450,866 | — |
| `cg_implicit_entry` | 6.642e+06 | 7,450,866 | `…331856` |
| `storage_pre_copy` | 6.642e+06 | 7,450,866 | `…331856` (same as entry) |
| `storage_post_copy` | 6.642e+06 | 7,450,866 | `…942032` (`aliased_input=False`) |
| `after_next_zeroEntries` | **6.642e+06** | 7,450,866 | `…942032` (same as storage) |
| `adjoint_init_idx0` | **0.000e+00** | 7,450,866 | `…575536` (**different object**) |
| `adjoint_init_idx1` | **0.000e+00** | 7,450,866 | `…575296` (different object) |

### Confirmation run 3102465 (after fix)

| Stage | Frobenius norm | `nz_used` | `id(mat)` |
|---|---:|---:|---|
| `[jac-reassembly]` `copy` | 6.642e+06 | 7,450,866 | — |
| `cg_implicit_entry` | 6.642e+06 | 7,450,866 | `…827536` |
| `storage_pre_copy` | 6.642e+06 | 7,450,866 | `…827536` |
| `storage_post_copy` | 6.642e+06 | 7,450,866 | `…823456` |
| `after_next_zeroEntries` | 6.642e+06 | 7,450,866 | `…823456` |
| `adjoint_init_idx0` | **6.642e+06** | 7,450,866 | `…051472` |
| `adjoint_init_idx1` | **6.646e+06** | 7,450,866 | `…051232` |

The tiny difference between `idx0` and `idx1` (`6.642` vs `6.646`) reflects the real physics — each stored Jacobian corresponds to a different time step with slightly different linearization state.

---

## 5. Exact stage where the norm collapses

### Before fix

`after_next_zeroEntries` has `norm=6.642e+06, id=…942032` — the **saved storage matrix** is intact even after the *next* Newton iteration zeroes the live `self.A`. This rules out the original aliasing hypothesis.

`adjoint_init_idx0` has `norm=0.000e+00, id=…575536` — a **different** PETSc.Mat object, with the same `nz_used=7,450,866` but all-zero values.

**The zeroing is not a mutation of the saved matrix.** It is the creation of new, unpopulated matrices from the sparsity pattern of the saved ones, on a code path that runs between `TimeStepDataManager` storing the Jacobian and `LinearizedWMEQoI` invoking the adjoint.

Tracing the jacobian argument up the call stack:

```
ImplicitAdjointSolver(... jacobians=self._jacobians ...)        ← qoi_maps.py:1065
          ↑
LinearizedWMEQoI(... jacobians=jacobians ...)                    ← qoi_maps.py:562
          ↑
wme_qoi.linearize(... jacobians=truth_jacobians ...)             ← run_comparison.py:310
          ↑
_compute_eq38_from_tlm(... truth_jacobians=truth_jacobians ...)  ← idealized_inlet_da.py:494
          ↑
truth_jacobians = [J.duplicate() for J in storage.saved_jacobians]   ← idealized_inlet_da.py:261  ★
```

The `★` line produces the new, zero-valued objects. `.duplicate()` **without `copy=True`** returns a new PETSc.Mat with the same layout/sparsity but values unset. Line 268 then destroys the source matrices via `solver_truth.storage.clear()`, so there's no way to recover values after the fact.

Classification (from the task's four options): **"survives storage, survives past the next live-matrix zeroing, and collapses during experiment-level handoff to the adjoint."** The fix is not in library code — it is in the idealized-inlet experiment driver.

---

## 6. `duplicate(copy=True)` confirmation

**Attempted and successful.** The one-line patch at [idealized_inlet_da.py:261](../experiments/idealized_inlet_da.py#L261):

```diff
-        truth_jacobians = [J.duplicate() for J in solver_truth.storage.saved_jacobians]
+        truth_jacobians = [J.duplicate(copy=True) for J in solver_truth.storage.saved_jacobians]
```

**Probe-level confirmation** (run 3102465, same config, same bisector, np=2):

- `adjoint_init_idx0`: `0.000e+00 → 6.642e+06` ✓
- `adjoint_init_idx1`: `0.000e+00 → 6.646e+06` ✓

**Bonus — physics-level confirmation from the uv-bisector backward sweep** (this is the whole point of the investigation series):

| Stage | Pre-fix (3101378) | Post-fix (3102465) |
|---|---|---|
| `n=4` BEFORE solve | `‖h‖=5.083e+00, ‖uv‖=0` | `‖h‖=5.083e+00, ‖uv‖=0` (same forcing) |
| `n=4` BEFORE solve `tiny_h / tiny_uv` | **35106 / 35106** (100% tiny) | **0 / 35106** — **0 / 70212** (no false tinies) |
| `n=4` BEFORE solve `diag_max / tiny_thresh` | `0 / 1.0e-20` (threshold inert) | `9.676e+04 / 9.676e-08` (threshold active and correct) |
| `n=4` AFTER solveTranspose | `‖h‖=5.083, ‖uv‖=0` | **`‖h‖=1.702e-02, ‖uv‖=6.802e-03`** ← real adjoint solve |
| `n=4` AFTER tiny-mask zeroing | `‖h‖=0, ‖uv‖=0` (wiped) | `‖h‖=1.702e-02, ‖uv‖=6.802e-03` (nothing to zero) |
| `n=3` BEFORE solve (time-coupled) | `‖h‖=0, ‖uv‖=0` | `‖h‖=1.297e-01, ‖uv‖=1.804e-01` ← **u/v time-couples from h** |
| `n=3` AFTER solveTranspose | `‖h‖=0, ‖uv‖=0` | `‖h‖=7.790e-03, ‖uv‖=2.994e-03` |
| `n=2` BEFORE / AFTER | all 0 | `‖uv‖=1.030e-01 → 3.211e-03` |
| `n=1` BEFORE / AFTER | all 0 | `‖uv‖=5.636e-02 → 1.948e-03` |
| `n=0` gradient_u0 | `‖h‖=5.083, ‖uv‖=0` | **`‖h‖=5.085, ‖uv‖=6.346e-02`** ← **real u/v gradient** |

The adjoint is now back-propagating u/v content from h forcings through the full DA window — exactly what the SWE mixed-element cross-component Jacobian is supposed to do. **The TLM Eq 38 path will now produce a non-degenerate `G_uv`.** The predictability-assumption result that DC-WME rests on can finally be computed legitimately on this problem.

---

## 7. Recommended next fix location

**The fix at [experiments/idealized_inlet_da.py:261](../experiments/idealized_inlet_da.py#L261) is already landed and confirmed** (commit `3852e6f`).

Suggested clean-up, in order of priority:

1. **Audit other `.duplicate()` uses in the repo.** Anywhere `PETSc.Mat.duplicate()` is used to deep-copy a matrix (as opposed to allocate a zero workspace with matching layout), the `copy=True` flag is required. A safe next pass: grep the repo for `\.duplicate\(\)` on matrix objects and inspect each call site. Likely hits to check:
   - other experiment drivers that hold truth trajectories across solver deletion boundaries
   - any place that prepares Jacobians for parallel adjoint handoff
   - ch-sim / pyadcirc shared patterns if similar code was copied over

2. **Land the structural diagnostic commits** (`486471e`, `4b12767`, `9385fa5`, `3889d86`, `869d7a7`) or decide whether to revert them. They produce no behavior change on normal runs because `_UV_BISECTOR_CTX["comp_idx"]` is `None` by default and all probes are one-shot-gated. If you want them around as permanent safety nets, they can stay; otherwise clean them up once the downstream experiments that rely on this code path have been re-validated.

3. **Re-run the Step-7b DC-WME production** on the idealized inlet (the cost comparison that originally motivated the entire investigation). `σ_b²_uv` should now be finite and `G_uv` non-trivial. The "all eigenvalues identical" degeneracy described in Phase 3/5 of the Shinnecock study may have been the *same* bug in a different experiment driver — worth checking whether Shinnecock's comparison also uses a bare `.duplicate()` somewhere in its truth-Jacobian handoff.

4. **Regression test**: add a tiny unit test that asserts `J.duplicate()` yields a zero matrix by default, and that the codebase's Jacobian-deep-copy helpers use `copy=True`. Prevents recurrence.

---

## Hard constraints respected

- **No repair pass performed beyond the confirmed one-line fix.** The fix is a single boolean flag change, fully reversible, and the probe chain proves it resolves the C2 observation.
- **No production comparison rerun.** The two sbatch invocations were the same tiny bisector job, cancelled after the probes of interest had emitted.
- **Single-purpose localization maintained.** No changes to Newton, storage, adjoint solver, cost function, or QoI map logic — the actual library code was instrumented, not modified.
- **No cross-hypothesis dilution.** The trace decides one question (*where does the norm collapse?*) and then answers the follow-up (*does the fix actually work?*) with a single targeted rerun.

---

## Appendix: complete investigation chain

| Step | Memo | Key result |
|---|---|---|
| 1 | [idealized_inlet_tlm_uv_bisector.md](idealized_inlet_tlm_uv_bisector.md) | Hypothesis B: tiny-mask threshold mis-calibrated |
| 2 | [idealized_inlet_tlm_uv_bisector_fix_validation.md](idealized_inlet_tlm_uv_bisector_fix_validation.md) | Outcome 3: threshold fix inert because `diag_max = 0` |
| 3 | [idealized_inlet_stored_jacobian_diagnostic.md](idealized_inlet_stored_jacobian_diagnostic.md) | C2: matrix allocated with sparsity but values all zero |
| 4 | [idealized_inlet_jacobian_reassembly_trace.md](idealized_inlet_jacobian_reassembly_trace.md) | R1a: newton.py's reassembly is correct; collapse is downstream |
| **5** | **this memo** | **R2: collapse at `.duplicate()` without `copy=True` in experiment driver; one-line fix confirmed; u/v adjoint content now correct** |

Investigation closed.

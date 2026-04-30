# R1 Jacobian Reassembly Trace — Idealized Inlet

**Date**: 2026-04-22
**Related**: [idealized_inlet_stored_jacobian_diagnostic.md](idealized_inlet_stored_jacobian_diagnostic.md) · [idealized_inlet_tlm_uv_bisector_fix_validation.md](idealized_inlet_tlm_uv_bisector_fix_validation.md) · [idealized_inlet_tlm_uv_bisector.md](idealized_inlet_tlm_uv_bisector.md)
**LS6 run**: 3102449 (commit `9385fa5`, cancelled after R1 line captured)
**Outcome**: **R1a — matrix is real at reassembly and after the in-place `A.copy()`; zeroing happens downstream of `newton.py`.**

---

## 1. Executive summary

The post-convergence no-BC reassembly at [src/swe4dvar/forward/newton.py:346-358](../src/swe4dvar/forward/newton.py#L346-L358) is **not** the culprit. Frobenius-norm and `nz_used` at all four instrumented stages are intact:

```
[jac-reassembly] converged=True
  pre_zero = 6.642e+06    (A at entry — still holds last Newton iter's BC-modified J)
  mid      = 0.000e+00    (after A.zeroEntries() — as expected)
  post     = 6.642e+06    (after petsc.assemble_matrix(A, self.jacobian) + A.assemble())
  copy     = 6.642e+06    (after J_final = A.copy())
  nz(pre/post/copy) = 7,450,866 / 7,450,866 / 7,450,866
```

The Jacobian that leaves `newton.py` via `return None, J_final` has Frobenius norm `6.642e+06` — a realistic magnitude for a DG physics Jacobian on a mesh with ~105K DOFs. The previous structural-diagnostic run showed that the same matrix arrives at `ImplicitAdjointSolver.jacobians[k]` with Frobenius norm `0`. Therefore the zeroing must happen between:

- **Upstream boundary** (confirmed intact): `newton.py` `J_final = A.copy(); return None, J_final`
- **Downstream boundary** (confirmed zero): `ImplicitAdjointSolver._solve_transpose_system` reading `self.jacobians[n-1]`

The code path between these two boundaries is:

```
newton.py time_loop() returns (_, J_final)
  ↓
cg_implicit.py:484  save_jacobians(J)  → self.storage.save_jacobian(J)
  ↓
solver_storage.py:95-105  save_jacobian()  → self.saved_jacobians.append(jacobian.copy())
  ↓
augmented_control.py:658-659  jacobians = self.solver.storage.saved_jacobians.copy()  (list copy)
  ↓  (… control-vector wrapper passes this to the adjoint …)
ImplicitAdjointSolver.__init__(jacobians=…)  →  self.jacobians[k]
```

The **most likely culprit is `solver_storage.save_jacobian`** at [src/swe4dvar/utils/solver_storage.py:95-105](../src/swe4dvar/utils/solver_storage.py#L95-L105):

```python
def save_jacobian(self, jacobian: PETSc.Mat):
    if hasattr(jacobian, 'copy'):
        self.saved_jacobians.append(jacobian.copy())    # second copy of the matrix
    else:
        self.saved_jacobians.append(jacobian)
```

This does a **second** `.copy()` on top of the `A.copy()` that `newton.py` already did. A `PETSc.Mat.copy()` without arguments copies structure and values; it should preserve the norm. But if anywhere in this chain the matrix's underlying AIJ value buffer is being released, reset, or aliased to a reused buffer that subsequently gets zeroed (because `newton.py`'s `self.A` is reused across timesteps and is zeroed by `A.zeroEntries()` at the start of every new Newton iteration at [newton.py:183](../src/swe4dvar/forward/newton.py#L183)), the stored "copy" may actually still reference the shared buffer.

That's the specific testable hypothesis for R2: **the stored Jacobian aliases `self.A`'s value buffer instead of owning an independent copy.** When the next Newton solve calls `A.zeroEntries()`, all previously saved Jacobians observe the zeroing because they share the buffer.

---

## 2. Exact instrumentation added

[src/swe4dvar/forward/newton.py](../src/swe4dvar/forward/newton.py) one-shot via class attribute `_R1_FIRED`:

```python
_R1 = not getattr(CustomNewtonProblem, "_R1_FIRED", False)
if _R1:
    CustomNewtonProblem._R1_FIRED = True

if converged:
    if _R1:
        pre_zero_norm = A.norm(PETSc.NormType.NORM_FROBENIUS)
        pre_nz = int(A.getInfo().get("nz_used", -1))
    A.zeroEntries()
    if _R1:
        mid_norm = A.norm(PETSc.NormType.NORM_FROBENIUS)
    petsc.assemble_matrix(A, self.jacobian)
    A.assemble()
    if _R1:
        post_assembly_norm = A.norm(PETSc.NormType.NORM_FROBENIUS)
        post_nz = int(A.getInfo().get("nz_used", -1))

J_final = A.copy()
if _R1:
    copy_norm = J_final.norm(PETSc.NormType.NORM_FROBENIUS)
    copy_nz = int(J_final.getInfo().get("nz_used", -1))
    if self.comm.rank == 0:
        print(f"[jac-reassembly] converged={converged} "
              f"pre_zero={pre_zero_norm:.3e} "
              f"mid={mid_norm:.3e} "
              f"post={post_assembly_norm:.3e} "
              f"copy={copy_norm:.3e} "
              f"nz(pre/post/copy)={pre_nz}/{post_nz}/{copy_nz}",
              flush=True)
```

Commit: `9385fa5 diag(newton-R1): one-shot reassembly trace in return_jacobian path`.

---

## 3. Exact run command used

```bash
# LS6 job 3102449 — commit 9385fa5
sbatch -p development hpc/lonestar6/idealized_inlet/job_uv_bisector.slurm
```

Config (from [hpc/lonestar6/idealized_inlet/job_uv_bisector.slurm](../hpc/lonestar6/idealized_inlet/job_uv_bisector.slurm)):

```
np=2, nt-ramp=4, nt-da=4, max-iterations=1 max-funcs=1
--method dcwme_static --eq38-component-aware
```

The R1 line appeared at line 298 of `inlet_uvbis.3102449.out`, approximately 2:30 into the run (first `return_jacobian=True` call during the truth trajectory). Job cancelled immediately after capture.

---

## 4. Norm + nz results

| Stage | Norm (Frobenius) | nz_used |
|---|---:|---:|
| `pre_zero` (A on entry to the block) | **6.642e+06** | 7,450,866 |
| `mid` (after `A.zeroEntries()`) | 0.000e+00 | — |
| `post` (after `assemble_matrix` + `A.assemble()`) | **6.642e+06** | 7,450,866 |
| `copy` (after `J_final = A.copy()`) | **6.642e+06** | 7,450,866 |

**All pre-return values are intact.** The reassembly produces a correct physics Jacobian, and the `A.copy()` inside `newton.py` preserves it.

Cross-reference: the previous structural-diagnostic run (job 3101406) showed Frobenius `0.0`, `nz_used = 7,450,866` on the matrix that `ImplicitAdjointSolver` actually consumed. The **`nz_used` is identical in both runs** — same sparsity pattern, same AIJ allocator state, but different numerical values. This is consistent with a buffer-aliasing corruption rather than an assembly-failure.

---

## 5. R1 classification

### **R1a — matrix real at reassembly, zeroed downstream.**

`pre_zero > 0`, `post > 0`, `copy > 0`, `nz` preserved at all four stages. The `newton.py` `return_jacobian` block is behaving exactly as intended. The fix is **not** in `newton.py`.

R1b and R1c are both explicitly falsified by the trace.

---

## 6. `scatter_forward()` follow-up — NOT attempted

The task description stated:

> Do **not** patch `u.x.scatter_forward()` yet in this pass unless the trace clearly points to R1b and you absolutely need a second quick confirmation.

The trace cleanly points to **R1a**, not R1b. A `scatter_forward()` test would address a coefficient-packing staleness theory that the trace has already falsified (since `post_assembly_norm = 6.642e+06`, coefficient packing at reassembly time works correctly). No follow-up run executed.

---

## 7. Recommended next fix location

**Most likely: [src/swe4dvar/utils/solver_storage.py:95-105](../src/swe4dvar/utils/solver_storage.py#L95-L105) — `save_jacobian`.**

Hypothesis: `jacobian.copy()` in petsc4py returns a `PETSc.Mat` whose value buffer is an independent allocation — but *if `jacobian` was itself the return of `A.copy()` from `newton.py`*, and if there's any path in the overall call chain that later invalidates that second-level copy (e.g., via `A.zeroEntries()` on the *original* `self.A` from `newton.py` if the copy is somehow shallow or sharing internal state through a PETSc mat-type issue), the saved Jacobian observes zeros.

### Next narrow diagnostic (R2, ≤10 lines)

Add one-shot Frobenius-norm traces at each boundary crossing:

1. **Entry to `cg_implicit.save_jacobians(J)`** ([cg_implicit.py:484](../src/swe4dvar/forward/solvers/cg_implicit.py#L484)): print `J.norm(NORM_FROBENIUS)` on first call — expected `6.642e+06` if the J returned by `newton.py` survives the interfile pass.

2. **Inside `solver_storage.save_jacobian`** ([solver_storage.py:95-105](../src/swe4dvar/utils/solver_storage.py#L95-L105)): print `jacobian.norm(...)` before `.copy()` and `saved_jacobians[-1].norm(...)` after the append.

3. **Also log `jacobian.norm(...)` after the *next* Newton solve fires `A.zeroEntries()`** — i.e., between `save_jacobian` calls 1 and 2. If norm drops to 0 here, the saved-copy-aliases-A hypothesis is confirmed.

4. **At `ImplicitAdjointSolver.__init__`** (or equivalent entry): iterate over `jacobians` and print each `.norm(...)`. If these are already 0, the corruption has happened by the time the adjoint is constructed.

That R2 trace isolates the exact hand-off at which the norm crashes.

### Candidate fix, once R2 confirms

If the saved Jacobian is observed to lose its values exactly when `newton.py` runs the *next* timestep's `A.zeroEntries()`, the fix is to replace the inline `jacobian.copy()` at `solver_storage.py:102` with an explicit duplicate-and-copy sequence that guarantees an independent buffer:

```python
def save_jacobian(self, jacobian: PETSc.Mat):
    if hasattr(jacobian, 'copy'):
        J_new = jacobian.duplicate(copy=True)  # explicit independent alloc
        self.saved_jacobians.append(J_new)
    else:
        self.saved_jacobians.append(jacobian)
```

Or alternatively, convert to a serialized form that cannot alias the live PETSc matrix. But the fix decision should wait on the R2 localization — do not patch blindly.

---

## Hard constraints respected

- **No repair pass performed.** Only the R1 trace lines were added.
- **No production comparisons rerun.** Single bisector invocation, cancelled immediately after the one R1 line was captured.
- **No multiple simultaneous changes.** Just the trace.
- **No `scatter_forward()` speculative patch** — the trace falsified R1b and did not justify it.

---

## Appendix: links

- LS6 raw output: `$WORK/SWEMniCS/inlet_uvbis.3102449.out` (line 298 contains the R1 print; the remainder is the normal bisector+structural traces for context)
- Instrumentation commit: `9385fa5` (will be retained through the next pass to keep the pre/post anchor intact)
- Previous memos in this investigation series:
  - [idealized_inlet_tlm_uv_bisector.md](idealized_inlet_tlm_uv_bisector.md) — original bisector (Hypothesis B: tiny-mask threshold)
  - [idealized_inlet_tlm_uv_bisector_fix_validation.md](idealized_inlet_tlm_uv_bisector_fix_validation.md) — Outcome 3: threshold hypothesis disproven
  - [idealized_inlet_stored_jacobian_diagnostic.md](idealized_inlet_stored_jacobian_diagnostic.md) — C2: matrix allocated but numerically empty

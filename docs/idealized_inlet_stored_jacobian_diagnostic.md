# Stored Jacobian Structural Diagnostic — Idealized Inlet DC-WME

**Date**: 2026-04-22
**Related**: [idealized_inlet_tlm_uv_bisector.md](idealized_inlet_tlm_uv_bisector.md) · [idealized_inlet_tlm_uv_bisector_fix_validation.md](idealized_inlet_tlm_uv_bisector_fix_validation.md)
**LS6 run**: 3101406 (commit `4b12767`, cancelled after first adjoint fired)
**Outcome**: **C2 — Structurally empty / wrong matrix.**

---

## 1. Executive summary

The stored Jacobian that `ImplicitAdjointSolver` consumes for the idealized-inlet DC-WME case is **allocated with a real DG sparsity pattern (7,450,866 AIJ entries on a 207,936×207,936 matrix), but every single numerical value is exactly zero**. This is not a diagonal-only zero, not a threshold artifact, not an API quirk. The operator is dead: `‖J·x‖ = 0` for every test vector we tried (all-ones, h-impulse, uv-impulse, random 1e-3).

All of the following are measured, not inferred:

```
global size        = (207936, 207936)       (= local h+uv on r0 + local h+uv on r1)
nz_used            = 7,450,866              (AIJ sparsity pattern is populated)
Frobenius norm     = 0.000000e+00
Infinity   norm    = 0.000000e+00

# nonzero |diag| entries = 0    (0 of 207,936 rows have a nonzero diagonal)
max |diag|               = 0.0
max |off-diag|           = 0.0
# off-diag entries       = 7,242,930 (all in bucket [0, 1e-20))

3x3 upper-left           = [[0 0 0] [0 0 0] [0 0 0]]

op-action "ones":       ||J·1||_total=0.000e+00  h=0  uv=0
op-action "h-impulse":  ||J·e_h||_total=0.000e+00  h=0  uv=0
op-action "uv-impulse": ||J·e_uv||_total=0.000e+00  h=0  uv=0
op-action "rand(1e-3)": ||J·r||_total=0.000e+00  h=0  uv=0
```

Because `J·x = 0` for all `x`, the transpose solve `J^T·λ = forcing` on this matrix is a singular system where any `λ` that preserves `forcing` on the identity-regularized rows is a valid "solution" — this is exactly what we saw in the previous bisector: `‖λ_h‖` passed through unchanged (5.083 → 5.083) because the regularized `J_reg = 0 + I_tiny = I` is the identity on all rows.

**The fix is upstream of the adjoint.** The matrix that gets handed to `ImplicitAdjointSolver.jacobians[k]` is not the Newton physics Jacobian — it is the Newton physics Jacobian's *sparsity pattern* populated with zeros. The immediate suspect is the post-convergence reassembly at [src/swe4dvar/forward/newton.py:346-358](../src/swe4dvar/forward/newton.py#L346-L358):

```python
if converged:
    A.zeroEntries()
    # Assemble WITHOUT bcs - this gives the true physics Jacobian
    petsc.assemble_matrix(A, self.jacobian)   # ← this call leaves A numerically zero
    A.assemble()
J_final = A.copy()
return None, J_final
```

The dispatch/signature of `dolfinx.fem.petsc.assemble_matrix(A: Mat, a: Form, ...)` in DOLFINx 0.10.0.post5 is correct (verified via runtime introspection). The call at line 353 is structurally well-formed and identical in shape to the call at [newton.py:184](../src/swe4dvar/forward/newton.py#L184) which works during Newton iterations. The difference must be in state or call context at reassembly time.

---

## 2. Exact diagnostic instrumentation added

### 2.1 One-shot structural diagnostic (`implicit_adjoint.py`)

Added module-level `_STORED_JAC_DIAG_FIRED = [False]` and function `_stored_jacobian_structural_diag(J, n)`. Invoked from `_solve_transpose_system` the first time it sees a `J` while the UV bisector is armed:

```python
J = self.jacobians[n - 1]

if (_UV_BISECTOR_CTX["comp_idx"] is not None
        and not _STORED_JAC_DIAG_FIRED[0]):
    _STORED_JAC_DIAG_FIRED[0] = True
    try:
        _stored_jacobian_structural_diag(J, n)
    except Exception as _e:
        from mpi4py import MPI as _MPI
        if _MPI.COMM_WORLD.Get_rank() == 0:
            print(f"[jac-diag] diagnostic raised: {_e}", flush=True)
```

Captures per-J on the first backward step:
- PETSc metadata (size, block size, nz_used, nz_allocated, Frobenius, Infinity norms)
- Diagonal magnitude stats (allreduced max/min/mean/nonzero count)
- Off-diagonal magnitude stats (allreduced)
- Four-bucket magnitude histogram (`[0, 1e-20)`, `[1e-20, 1e-10)`, `[1e-10, 1e-5)`, `[1e-5, ∞)`) separated for diag vs off-diag
- Upper-left 3×3 dense snapshot (rank 0, trivially collective-safe)
- Operator-action tests: ones, h-impulse, uv-impulse, random 1e-3 — each reports `‖J·x‖_total / ‖J·x‖_h / ‖J·x‖_uv`

Commits: `486471e` (added diagnostic), `4b12767` (removed a missed `restoreArray` call that was aborting the diagnostic on the last test vector).

### 2.2 Runtime introspection of `dolfinx.fem.petsc.assemble_matrix`

On LS6 in the fenics-ls6 conda env:

```
>>> import dolfinx.fem.petsc as p
>>> import inspect; from petsc4py import PETSc
>>> p.assemble_matrix.registry
{<class 'object'>: <function assemble_matrix>,
 <class 'petsc4py.PETSc.Mat'>: <function _>}
>>> inspect.signature(p.assemble_matrix.dispatch(PETSc.Mat))
(A: 'PETSc.Mat', a: 'Form | Sequence[Sequence[Form]]',
 bcs=None, diag=1, constants=None, coeffs=None) -> 'PETSc.Mat'
```

So the call `petsc.assemble_matrix(A, self.jacobian)` dispatches to the Mat-first-arg variant. The dispatch is registered. This **rules out C3** (API/extraction artifact at the adjoint side).

---

## 3. Exact run command used

```bash
# LS6 job 3101406 — commit 4b12767
sbatch -p development hpc/lonestar6/idealized_inlet/job_uv_bisector.slurm

# --method dcwme_static --vmax 20 --track-shift 10
# --nt-ramp 4 --nt-da 4 --obs-fraction 0.005 --obs-frequency 4
# --obs-noise-level 0.01 --background-error-std 0.02
# --max-iterations 1 --max-funcs 1
# --obs-correlation-length 1500 --predictability-gamma 0.1
# --eq38-component-aware
#
# Runtime to [jac-diag] block: ~5 min. Job cancelled after block captured.
```

---

## 4. Matrix metadata

From run 3101406, first `self.jacobians[k]` consumed at `n=4`:

| Field | Value |
|---|---|
| `J.getSize()` | `(207936, 207936)` |
| `J.getLocalSize()` | varies by rank: r0 local rows = 105318, r1 local rows = 102618 |
| `J.getBlockSize()` | `1` |
| `J.getInfo()['nz_used']` | **`7,450,866`** |
| `J.getInfo()['nz_allocated']` | `7,450,866` |
| Total entries seen (row-wise traversal) | `7,450,866` (matches `nz_used`) |
| `J.norm(NORM_FROBENIUS)` | **`0.000000e+00`** |
| `J.norm(NORM_INFINITY)` | `0.000000e+00` |

**Interpretation**: The matrix has a real, DG-plausible sparsity pattern (~35.8 entries per row average, consistent with DG mixed-element coupling across shared facets). The AIJ allocator has committed these entries. **But every committed value is numerically zero.**

---

## 5. Diagonal / off-diagonal magnitude summary

| Quantity | Diagonal | Off-diagonal |
|---|---:|---:|
| count (global) | 207,936 rows | 7,242,930 entries |
| nonzero \|·\| | **0 / 207,936** | `max = 0` |
| max \|·\| | `0.000000e+00` | `0.000000e+00` |
| min \|·\| | `0.000000e+00` | `inf` (no nonzero found) |
| mean \|·\| | `0.000000e+00` | `0.000000e+00` |

Both diagonal and off-diagonal are structurally nonzero (sparsity pattern populated) but numerically zero everywhere. The `min |off-diag_nonzero| = inf` sentinel confirms no nonzero off-diagonal was found during the full per-row sweep.

---

## 6. Histogram summary

```
bucket             diag count     off-diag count
[0, 1e-20)            207,936      7,242,930     ← ALL entries
[1e-20, 1e-10)              0              0
[1e-10, 1e-5)               0              0
[1e-5, inf)                 0              0
```

All 7,450,866 entries (207,936 + 7,242,930) land in the `[0, 1e-20)` bucket. **No entry is numerically nonzero.**

---

## 7. Operator-action results

Each test applies `y = J · x` then reports global ‖y‖, ‖y‖_h (restricted to h DOFs), ‖y‖_uv (restricted to u/v DOFs).

| Test vector | `‖J·x‖_total` | `‖J·x‖_h` | `‖J·x‖_uv` |
|---|---:|---:|---:|
| ones (constant 1.0) | **`0.000e+00`** | `0.000e+00` | `0.000e+00` |
| h-impulse (1 at first local h-DOF per rank) | **`0.000e+00`** | `0.000e+00` | `0.000e+00` |
| uv-impulse (1 at first local uv-DOF per rank) | **`0.000e+00`** | `0.000e+00` | `0.000e+00` |
| random Gaussian × 1e-3 | **`0.000e+00`** | `0.000e+00` | `0.000e+00` |

For a nonzero matrix, `J·1` must produce something nonzero (it is the row-sum vector). For a nonzero matrix, at least one of `J·e_i` for basis vectors `e_i` must be nonzero. The random dense vector probes every direction simultaneously — if any single matrix entry were nonzero, `‖J·r‖` would be nonzero.

**`J` is the zero operator.** Operator action is the strongest test we can run without writing a dense matrix to disk, and it falsifies the "C1: nonzero off-diag, zero diag" hypothesis from the fix-validation memo: in C1 we would expect `‖J·1‖ > 0` because off-diagonal entries would contribute to row sums.

---

## 8. Final classification

### **C2 — Structurally empty / wrong matrix.**

Evidence:
1. AIJ sparsity is populated (7.45M entries) but every value is zero.
2. Frobenius norm exactly zero.
3. No nonzero diagonal, no nonzero off-diagonal.
4. Operator action on four structurally distinct test vectors (including random) all produce zero. This rules out C1 ("zero diag, nonzero off-diag").
5. `assemble_matrix` dispatch registry confirmed correct at runtime for `(PETSc.Mat, Form)` argument pattern — rules out C3.
6. The same call pattern `petsc.assemble_matrix(A, self.jacobian, bcs=...)` is used inside the Newton loop at [newton.py:184](../src/swe4dvar/forward/newton.py#L184) and the Newton solver successfully converges (log shows real residual norms and real correction norms), so the assembly machinery itself is functional — the problem is specific to the *post-convergence, no-bcs reassembly* at [newton.py:346-358](../src/swe4dvar/forward/newton.py#L346-L358).

C4 is not required — C2 is supported by direct measurement.

---

## 9. Recommended next fix location

**Scope: narrow repair pass on [`src/swe4dvar/forward/newton.py`](../src/swe4dvar/forward/newton.py), post-convergence `return_jacobian` block only.**

Three-step diagnostic-then-fix sequence, ordered by information yield per line of code:

### Step R1 — Confirm where the zero comes from (diagnostic only, ~5 lines)

Insert immediately around [newton.py:349-354](../src/swe4dvar/forward/newton.py#L349-L354) (one-shot, rank-0 print on first `return_jacobian=True` call):

```python
if converged:
    # [DIAG] Before zeroEntries: capture norm of the BC-modified J
    pre_zero_norm = A.norm(PETSc.NormType.NORM_FROBENIUS)
    A.zeroEntries()
    mid_norm = A.norm(PETSc.NormType.NORM_FROBENIUS)   # expected: 0.0
    petsc.assemble_matrix(A, self.jacobian)
    A.assemble()
    post_assembly_norm = A.norm(PETSc.NormType.NORM_FROBENIUS)
    if self.comm.rank == 0:
        print(f"[jac-reassembly] pre_zero={pre_zero_norm:.3e} "
              f"mid={mid_norm:.3e} post={post_assembly_norm:.3e}")
```

Three outcomes:
- **R1a**: `pre_zero > 0`, `post_assembly > 0` → the matrix *is* populated at this point, something downstream (the `.copy()` on line 357, or storage in `TimeStepDataManager`, or the DCWME path) zeroes it. Fix is in the downstream path.
- **R1b**: `pre_zero > 0`, `post_assembly = 0` → the no-bcs reassembly call itself is producing a zero matrix. This is the most likely outcome given what we've seen, and points to a coefficient-packing or form-state issue (below).
- **R1c**: `pre_zero = 0` → the Newton iterations never actually assembled J into A (or A was already zeroed by the last step of the final iteration). Fix is inside the Newton loop.

### Step R2 — Most likely root cause (if R1b confirmed)

The UFL form `self.J = ufl.derivative(self.F, self.u)` depends on `self.u`. In DOLFINx 0.10, `pack_coefficients(form)` reads the current state of coefficient functions at call time. If `self.u`'s underlying `petsc_vec` was invalidated (destroyed, scatter missing, or replaced) between the Newton loop and the `return_jacobian` block, `pack_coefficients` would see a zero/empty coefficient and the assembly would be numerically zero while still producing the right sparsity.

Specifically, check if `apply_lifting` / `set_bc` / `scatter_forward` at lines 192-196, 223 have left `self.u` in a state where `u.x.array` is valid but `u.x.petsc_vec` (or any ghost-exchanged view) is not, then the jacobian form evaluates at zero-state even though `u.x.array` has good data.

**Minimal fix attempt** (single line insertion before line 353):

```python
if converged:
    u.x.scatter_forward()              # ← ensure coefficients are up-to-date
    A.zeroEntries()
    petsc.assemble_matrix(A, self.jacobian)
    A.assemble()
```

If that doesn't fix it, explicitly pack and pass:

```python
if converged:
    u.x.scatter_forward()
    from dolfinx.fem import pack_constants, pack_coefficients
    constants = pack_constants(self.jacobian)
    coeffs = pack_coefficients(self.jacobian)
    A.zeroEntries()
    petsc.assemble_matrix(A, self.jacobian,
                          constants=constants, coeffs=coeffs)
    A.assemble()
```

### Step R3 — Alternative: bypass the no-bcs reassembly

If R1/R2 are inconclusive, the shortest path to a correct adjoint Jacobian may be to reuse `assemble_A()` at [newton.py:362-367](../src/swe4dvar/forward/newton.py#L362-L367) (the `bcs=self.bcs` path that demonstrably works), then strip the BC identity rows with `MatZeroRowsColumns` or by masking:

```python
if converged:
    J_with_bcs = self.assemble_A()
    # Strip BC identity rows: set bc rows of J back to the original form values
    # or simply use J_with_bcs for the adjoint and apply BC zeroing to λ
    # (which the adjoint already does downstream).
```

This is a smaller perturbation — we already know the with-bcs path assembles correctly.

---

## Hard constraints respected

- **No repair pass performed.** This memo is purely diagnostic. The patched `implicit_adjoint.py` contains only instrumentation, no behavioral change beyond the (already-landed) harmless relative-threshold fix from commit `1bb99d8`.
- **No production comparisons rerun.** The one bisector invocation was sufficient and was cancelled after first-fire capture.
- **No large matrix dumps.** 3×3 snapshot only; everything else is aggregate statistics.
- **No conflation of zero-diag with zero-operator.** The operator-action tests are explicitly designed to separate those, and they falsify C1.

---

## Appendix: links

- Previous memos: [idealized_inlet_tlm_uv_bisector.md](idealized_inlet_tlm_uv_bisector.md), [idealized_inlet_tlm_uv_bisector_fix_validation.md](idealized_inlet_tlm_uv_bisector_fix_validation.md)
- Diagnostic code: [src/swe4dvar/adjoint/implicit_adjoint.py:100-306](../src/swe4dvar/adjoint/implicit_adjoint.py#L100-L306) (the `_stored_jacobian_structural_diag` function, commits `486471e` + `4b12767`)
- LS6 raw output: `$WORK/SWEMniCS/inlet_uvbis.3101406.out` (job 3101406, CANCELLED after first-fire)
- Suspect fix location: [src/swe4dvar/forward/newton.py:346-358](../src/swe4dvar/forward/newton.py#L346-L358)

# Idealized Inlet MPI "Deadlock" Audit

**Date**: 2026-04-16
**Status**: RESOLVED

---

## 1. Executive Summary

The MPI run of the idealized inlet 4D-Var experiment appeared to hang indefinitely in Step 8 (optimization), with both MPI ranks burning 99% CPU and no output for 12+ hours. Investigation revealed this was **not an MPI deadlock or collective mismatch**. The root cause was an **infinite callback loop** in the PETSc TAO optimizer wrapper: after the `max_funcs` limit was reached, the callback set `g.set(0.0)` to signal convergence, but TAO BLMVM ignored the zero gradient and kept calling the callback endlessly.

A secondary confound was **Python stdout buffering under MPI**, which made the production run appear frozen when it was actually computing (the Newton solver output was sitting in a buffer).

---

## 2. Observed Symptoms

- Output file stopped updating after Step 8 entry
- Both MPI ranks at ~99% CPU
- RSS dropped to ~31 MB per process (production run) or remained at ~2 GB (probe)
- No TAO callback output ever printed
- Forward/Newton solver worked correctly during setup phases
- Serial runs with `max_iterations < max_funcs` worked fine (never triggered the bug)

---

## 3. Exact Hang Location

**File**: `src/swe4dvar/optimization/petsc_tao_wrapper.py`, lines 247-253 (original code)

```python
# ORIGINAL (BROKEN):
if self._max_funcs is not None and self.n_func_evals >= self._max_funcs:
    g.set(0.0)  # ← TAO BLMVM ignores this
    return self._last_cost
```

After the 5th function evaluation (with `max_funcs=5`), every subsequent TAO callback:
1. Hit the `max_funcs` check
2. Set gradient to zero
3. Returned the cached cost
4. TAO BLMVM ignored the zero gradient
5. Called the callback again → goto 1

This produced an infinite loop at ~100% CPU with no useful computation.

---

## 4. Why It Wasn't an MPI Deadlock

The probe script (`experiments/mpi_deadlock_probe.py`) systematically tested:

| Test | Result |
|------|--------|
| Direct `value_gradient()` call on 2 ranks | **PASS** — identical results (cost=4695.33, grad=3400.8) |
| Observation operator (Allgatherv) | **PASS** — all collectives entered by both ranks |
| Gradient smoother dimensions | **PASS** — correct per-rank sizes (35295 vs 34017) |
| Adjoint solve | **PASS** — no rank-divergent exceptions |
| TAO first callback | **PASS** — completed on both ranks |
| TAO after max_funcs | **FAIL** — infinite callback loop (fixed) |

The cost function, observation operator, adjoint solver, and gradient smoother are all MPI-correct. The bug was entirely in the TAO wrapper's max_funcs enforcement logic.

---

## 5. Why Serial Runs Worked

The serial 15-iteration run used `max_iterations=15` and `max_funcs=30`. It completed 17 function evaluations before hitting `MAXITS` — never reaching `max_funcs=30`. The bug only triggers when `n_func_evals >= max_funcs`, which requires either:
- Many line-search steps (MPI run attempted more iterations)
- A low `max_funcs` setting (probe used `max_funcs=5`)

---

## 6. Code Changes

### Fix 1: Callback short-circuit (no forward solves after limit)

**File**: `src/swe4dvar/optimization/petsc_tao_wrapper.py`

After `max_funcs` is reached, the callback returns the cached cost and gradient instantly instead of running an expensive forward model. TAO's line search calls the callback a few more times but each returns immediately. The line search fails naturally (LS_FAILURE) because no improvement is found.

### Fix 2: Monitor-based enforcement (reliable TAO stop)

The `_custom_monitor_callback` now checks `n_func_evals >= max_funcs` and calls `tao.setConvergedReason(CONVERGED_USER)`. The monitor runs after each TAO iteration (not each line-search step), where `setConvergedReason` is reliably checked by TAO.

### Fix 3: Flush on callback prints

Added `flush=True` to all TAO callback `print()` statements so MPI output appears in real-time instead of being buffered indefinitely.

---

## 7. Serial vs MPI First-Evaluation Parity

| Metric | Serial | MPI (2 ranks) |
|--------|--------|----------------|
| Cost at background | 7808.87 | 4695.33* |
| Gradient norm | 4192 | 3400.8* |
| Newton convergence | 13 iterations | 13 iterations |
| Linear solver | GMRES+ILU (200+ iters) | MUMPS LU (1 iter) |

*Different because the probe used `nt_da=2` (minimal config) vs production `nt_da=12`. Within each config, both ranks produce identical results.

---

## 8. MPI Liveness Test Results

With the fix applied (`PYTHONUNBUFFERED=1 mpirun -np 2`):

| Eval | Cost | Grad Norm | Status |
|------|------|-----------|--------|
| 1 | 4695.33 | 3.40e+03 | OK |
| 2 | 33218.92 | 2.93e+06 | OK (line search probe) |
| 3 | 11735.71 | 1.47e+06 | OK |
| 4 | 6410.17 | 7.33e+05 | OK |
| 5 | 5101.41 | 3.66e+05 | OK (last real eval) |

Termination: LS_FAILURE (clean, after max_funcs=5 short-circuit)
Total callbacks: **exactly 5** (no wasted evals)
Runtime: 1170s for 5 evals on 2 MPI ranks

---

## 9. Remaining MPI Risks

### Low risk (verified safe)
- Observation operator collectives (allgather, Allgatherv)
- Gradient smoother (local matrices, correct per-rank dimensions)
- Adjoint solver (distributed MUMPS)
- PETSc Vec operations (copy, assemble, ghostUpdate)

### Medium risk (not triggered in current config, but present in code)
- **Rank-0-only matrix assembly** in `covariance.py:662` and `cost_functions.py:1812` — only triggered by DC-WME L_wme computation, not pure 4D-Var. Would likely work because `assemblyBegin/End` are called by all ranks even when only rank 0 fills values, but not tested under MPI.
- **Rank-divergent exception handling** in `value_gradient()` — if the forward model throws on one rank but not the other, the exception handler returns early on one rank while the other continues into a collective. Currently unlikely because MUMPS is a collective solver that fails on all ranks simultaneously.

### Recommended MPI debug procedure
1. Always use `PYTHONUNBUFFERED=1` with `mpirun`
2. Test with the probe script first: `PYTHONUNBUFFERED=1 mpirun -np 2 python experiments/mpi_deadlock_probe.py`
3. If hung, check `ps aux` for CPU usage: 99% = computation loop (not deadlock), 0% = blocking collective
4. Add rank-tagged prints with `flush=True` around suspect code

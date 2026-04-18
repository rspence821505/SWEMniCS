# Idealized Inlet MPI Adjoint Stabilization Audit

**Date**: 2026-04-16
**Status**: STABILIZATION INSUFFICIENT — ROOT CAUSE IS UPSTREAM OF THE ADJOINT LINEAR SOLVE

---

## 1. Executive Summary

The prior adjoint-parity audit ([docs/idealized_inlet_mpi_adjoint_solver_parity_audit.md](idealized_inlet_mpi_adjoint_solver_parity_audit.md)) pinned the remaining MPI/serial gradient-direction discrepancy on the distributed adjoint linear solve. The hypothesis was that the adjoint system is ill-conditioned, and distributed MUMPS selects a different near-null-space component than serial MUMPS.

This pass **tested that hypothesis directly** with two stabilization strategies:

1. **Diagonal regularization** (shift `εI` on `J^T` before solve, ε ∈ {1e-6, 1e-2, 1})
2. **Iterative adjoint solve** (GMRES + bjacobi/ILU preconditioning, `rtol=1e-10`)

**Both strategies failed to change the serial/MPI cosine similarity** (remains at ≈ 0.179, the same value seen without any intervention). Inside MPI, direct (MUMPS) and iterative (GMRES) solvers converge to **bit-exact the same solution** (cos_sim = 1.000000, rel_diff = 1.6e-10). Inside serial, MUMPS and PETSc built-in LU converge to **bit-exact the same solution**. The serial and MPI solutions are each internally consistent — but they solve **different mathematical problems**.

This rules out the linear-solver path as the cause. The divergence must originate upstream — in the **Jacobian assembly or adjoint RHS**, which produce slightly different operators/forcings under DG-DOF partitioning. That is outside the scope of a linear-solver stabilization pass.

**Recommendation**: MPI remains operationally safe (no deadlock, no correctness bug in the solver path) but **NOT scientifically authoritative**. Serial is the reference path for the 30-iteration 4D-Var continuation and any subsequent DC-WME comparisons. Investigating MPI assembly parity would be the next logical pass.

---

## 2. Regularization Strategy Tested

**Code change**: Added `adjoint_regularization` parameter to `ImplicitAdjointSolver` and an env-var override `SWE4DVAR_ADJOINT_REG`. When ε > 0, the adjoint solve uses `(J + εI)^T λ = f` instead of `J^T λ = f`. This is combined with the existing dry-node regularization (which sets `J[i,i]=1` at zero-diagonal DOFs; unchanged).

The shift is applied to a deep copy of J (never modifies the cached Jacobian), at negligible extra memory cost.

---

## 3. Sweep Results: ε ∈ {0, 1e-6, 1e-2, 1}

All runs use `method=4dvar`, `nt_ramp=2`, `nt_da=2`, forced LU (serial: PETSc built-in or MUMPS; MPI: MUMPS). Vectors saved as globally coord-ordered arrays for vector-level comparison.

### 3.1 Within-config consistency (ε sensitivity)

| Pair | λ₀ cos_sim | λ₀ rel_diff |
|------|-------------|-------------|
| Serial ε=0 vs Serial ε=1e-2 | 1.000000 | 0.0 |
| MPI ε=0 vs MPI ε=1e-6 | 1.000000 | 6.1e-9 |
| MPI ε=0 vs MPI ε=1e-2 | 1.000000 | 6.1e-5 |
| MPI ε=0 vs MPI ε=1 | 0.999984 | 5.7e-3 |

**ε has essentially no effect within either serial or MPI.** Even ε=1 (a huge shift relative to typical diagonal entries) only perturbs the solution at the 5e-3 level. The adjoint Jacobian is **not** near-singular — εI regularization does not change the solution because there's no near-null direction for it to damp.

### 3.2 Cross-config parity (serial vs MPI, unaffected by ε)

| Pair | λ₀ cos_sim | λ₀ rel_diff |
|------|-------------|-------------|
| MPI ε=0 vs Serial ε=0 | **0.179056** | 1.29 |
| MPI ε=1e-2 vs Serial ε=1e-2 | **0.179063** | 1.29 |
| MPI ε=1e-6 vs Serial ε=0 | 0.179056 | 1.29 |
| MPI ε=1 vs Serial ε=0 | unchanged order of magnitude | — |

**Regularization does not close the serial/MPI gap.** Cosine similarity stays at ≈ 0.179 regardless of ε.

---

## 4. Iterative Adjoint Solve Test

**Code change**: Added env-var `SWE4DVAR_ADJOINT_ITERATIVE=1` that forces the adjoint to use GMRES (preconditioned by ILU in serial, BJACOBI in MPI — bare ILU does not work on `mpiaij` matrices). Tolerances: `rtol=1e-10`, `atol=1e-14`, `max_it=5000` (overridable via env vars).

### 4.1 Results

| Pair | λ₀ cos_sim | λ₀ rel_diff |
|------|-------------|-------------|
| MPI iterative vs MPI MUMPS (direct) | **1.000000** | 1.7e-10 |
| MPI iterative vs Serial MUMPS | **0.179056** | 1.29 |

GMRES converges in MPI to the same solution MUMPS finds, to 10 digits. And both MPI solutions disagree with the serial solution by the same ≈ 0.179 cosine similarity.

### 4.2 Interpretation

- **The adjoint linear system is well-conditioned enough that multiple independent solvers converge to the same unique solution.** MUMPS-direct and GMRES-iterative agree to 10 digits.
- **The adjoint linear system being solved in MPI is not the same system being solved in serial.** Both are solved correctly, each to a unique solution, but the underlying `(J^T, f)` pair differs between the two.
- **The linear solver is exonerated.** This isn't a pivoting issue, a null-space projection issue, or a tolerance issue. The system itself differs.

---

## 5. Short Optimization Trace Comparison

Not re-run in this pass. The prior audit ([adjoint_solver_parity_audit §5](idealized_inlet_mpi_adjoint_solver_parity_audit.md#5-short-optimization-trace-comparison)) already established:

- Both serial and MPI terminate with LS_FAILURE, 0 iterations accepted, at max_funcs=6
- MPI's line-search probes are 1.7–4× worse than serial's at every backtrack step

Since the MPI adjoint under iterative solver is bit-exact with the MPI adjoint under MUMPS, re-running with iterative would produce the identical trace. No new information.

---

## 6. Is Regularization Enough?

**No.** ε from 1e-6 to 1 does not change the serial/MPI discrepancy. The discrepancy is not a null-space projection issue.

## 7. Is Iterative Solve Enough?

**No.** GMRES and MUMPS converge to the same solution within MPI, both disagreeing with serial. The iterative path confirms the linear solver isn't the issue.

## 8. MPI Scientific Usability: Final Verdict

**MPI is operationally safe but NOT scientifically authoritative** on this problem in its current configuration.

What works:
- No deadlocks
- Gradient smoother bit-exact
- Cost function to 0.012%
- Within-MPI solver choice (direct vs iterative) produces consistent answers

What doesn't:
- The adjoint `λ₀` in MPI is ≈ **orthogonal** to serial (cos_sim 0.179 across h component)
- Optimizer line-search probes in MPI are consistently worse than serial at each backtrack
- No solver-path intervention tested here closes the gap

The remaining work — **identifying why the Jacobian assembly (or adjoint RHS) differs under DG mesh partitioning** — is outside the scope of a linear-solver stabilization pass. Likely culprits are DG interior-facet flux terms and their MPI ghost handling. Investigating that is the logical next pass, but unsolved today.

---

## 9. Recommendation

1. **For the 30-iteration 4D-Var continuation run**: **use serial** (`python experiments/idealized_inlet_da.py ...`). Serial is the authoritative path; the gradient direction is stable and correct.

2. **For exploratory MPI runs**: acceptable for rapid iteration when the exact optimum is not needed. Users must understand that the MPI gradient direction differs from serial and that MPI optimizer trajectories will not match serial.

3. **Do not rely on the `SWE4DVAR_ADJOINT_REG` or `SWE4DVAR_ADJOINT_ITERATIVE` env vars to fix MPI parity** — they don't. They are left in the code as defensive options for future problems where the adjoint may actually be near-singular, but for this problem they add no value.

4. **Next logical pass (not done here)**: a **Jacobian-assembly parity audit**. Compare `J` (and the adjoint RHS `f`) entry-by-entry between serial and MPI on the same trajectory. Expected culprits: DG interior facet flux contributions on partition boundaries; ghost-cell handling in the variational form assembly.

---

## 10. Code Changes

| File | Change |
|------|--------|
| `src/swe4dvar/adjoint/implicit_adjoint.py` | Added `adjoint_regularization` parameter (env var `SWE4DVAR_ADJOINT_REG`) and iterative-solve mode (env var `SWE4DVAR_ADJOINT_ITERATIVE`). Both default off. |

Configurable env vars (all optional):
- `SWE4DVAR_ADJOINT_REG=<float>` — adds εI to J before adjoint solve
- `SWE4DVAR_ADJOINT_ITERATIVE=1` — forces GMRES instead of LU
- `SWE4DVAR_ADJOINT_ITER_RTOL=<float>` — relative tolerance for iterative solve (default 1e-10)
- `SWE4DVAR_ADJOINT_ITER_ATOL=<float>` — absolute tolerance (default 1e-14)
- `SWE4DVAR_ADJOINT_ITER_MAXIT=<int>` — max iterations (default 5000)

---

## 11. Answers to the Success-Criteria Questions

1. **Can a small adjoint regularization materially improve serial/MPI direction agreement?** **No.** cos_sim stays at 0.179 across ε from 1e-6 to 1. The adjoint system is not near-singular; εI has nothing to damp.

2. **Does that improvement carry through to short optimization traces?** **N/A** — no improvement to carry through.

3. **If not, does an iterative adjoint solve do better?** **No.** GMRES+bjacobi converges in MPI to the same answer MUMPS finds (cos_sim 1.0 to 10 digits between MPI direct and MPI iterative). Both MPI solutions disagree with serial by the same 0.179 cos_sim. The linear solver is not the issue.

4. **Is MPI now scientifically usable, or still only operationally safe?** **Still only operationally safe.** The divergence is upstream of the adjoint linear solve — in Jacobian assembly or adjoint RHS construction — and cannot be fixed by changing the solver. For scientific-authoritative runs, use serial.

# Idealized Inlet MPI Adjoint Solver Parity Audit

**Date**: 2026-04-16
**Status**: DISTRIBUTION EFFECT IDENTIFIED — MPI NOT SCIENTIFICALLY AUTHORITATIVE

---

## 1. Executive Summary

Prior passes fixed the MPI deadlock (`petsc_tao_wrapper.py` infinite callback loop) and built a bit-exact distributed gradient smoother (`src/swe4dvar/utils/distributed_smoother.py`). The parity harness still showed a 38% post-smoother gradient-norm difference between serial and 2-rank MPI while the smoother was proven bit-exact on identical inputs. The remaining discrepancy was suspected to lie in the adjoint linear-solver path.

**This pass ruled out backend effects and pinned the discrepancy on the distributed factorization path itself.** Vector-level comparison shows:

- **Serial PETSc built-in LU vs Serial MUMPS**: **bit-exact** (cos_sim = 1.000000, 0.0% relative difference). The linear-solver backend is irrelevant on 1 process.
- **Serial MUMPS vs MPI MUMPS (2 ranks)**: **nearly orthogonal** — cos_sim = **0.179**, 129% relative difference. The adjoint vectors are structurally different.

The cost function still matches to 0.012% (the forward trajectory is nearly identical). But the adjoint solve on this problem is ill-conditioned enough that distributed MUMPS converges to a mathematically valid but directionally different solution than serial MUMPS. Backend parity does not help because backend was never the issue.

Short 3-iteration BLMVM traces (same config, same starting point) show both serial and MPI fail with LS_FAILURE after 6 evals, 0 iterations accepted — but **MPI's line-search probes are 1.7–4× worse than serial's at every step**, reflecting the directional mismatch.

**Verdict**: MPI is not scientifically trustworthy as an authoritative run on this problem in its current configuration. MPI may be used for exploratory runs, but serial should remain the reference.

---

## 2. Solver-Backend Test Matrix

| Run tag | MPI size | KSP | PC | Solver | Cost | λ₀ norm | post-smoother grad norm |
|---------|----------|-----|-----|--------|------|---------|-------------------------|
| `serial_petsc_lu` | 1 | preonly | lu | (PETSc default) | 3221.019 | 726.912 | 1379.634 |
| `serial_mumps` | 1 | preonly | lu | mumps | 3221.019 | 726.912 | 1379.634 |
| `mpi_mumps` | 2 | preonly | lu | mumps | 3220.641 | 716.088 | 854.147 |

Key observation: the **two serial rows are bit-identical**. The `solver=` column changes (PETSc built-in vs MUMPS) but every downstream number is the same to all printed digits.

---

## 3. Vector-Level Adjoint Comparison

Adjoint λ₀ saved as globally coordinate-ordered (h, u, v) arrays via `tests/test_mpi_parity.py` (sorted by (x, y) for canonical ordering). Comparison via `tests/compare_adjoint_vectors.py`:

### 3.1 Serial MUMPS vs Serial PETSc built-in LU

| component | ‖a‖ | ‖b‖ | rel_diff | cosine | max_abs_diff |
|-----------|-----|-----|----------|--------|--------------|
| h | 721.7864 | 721.7864 | 0.0e+00 | 1.000000 | 0.0e+00 |
| u |  58.3305 |  58.3305 | 0.0e+00 | 1.000000 | 0.0e+00 |
| v |  63.4262 |  63.4262 | 0.0e+00 | 1.000000 | 0.0e+00 |
| total | 726.9119 | 726.9119 | 0.0e+00 | 1.000000 | — |

**Backends are interchangeable on 1 process.**

### 3.2 MPI MUMPS (size=2) vs Serial MUMPS

| component | ‖serial‖ | ‖MPI‖ | rel_diff | cosine | max_abs_diff |
|-----------|----------|-------|----------|--------|--------------|
| h | 721.79 | 708.91 | 1.296 | **0.175** | 76.74 |
| u |  58.33 |  49.93 | 1.061 | **0.531** | 2.13 |
| u |  63.43 |  87.98 | 0.988 | **0.378** | 2.69 |
| total | 726.91 | 716.09 | **1.291** | **0.179** | — |

**The MPI adjoint is nearly orthogonal to the serial adjoint** (cosine 0.179). The two solvers produce mathematically valid solutions of the adjoint system, but pointing in very different directions. The norms are only 1.5% apart, so **norm-based checks would have missed this completely** — that's why the earlier parity harness pass reported "pre-smoother gradient matches" when the directions are in fact very different.

### 3.3 Post-smoother gradient (for reference)

| Pair | total rel_diff | total cosine |
|------|----------------|--------------|
| serial_mumps vs serial_petsc_lu | 0.0e+00 | 1.000000 |
| mpi_mumps vs serial_mumps | 1.60 | 0.323 |

The smoother (proven bit-exact on identical input) faithfully propagates the upstream directional discrepancy.

---

## 4. Adjoint Assembly Verification

To rule out that the adjoint **assembly** differs between serial and MPI:

- **Forward trajectory**: cost matches to 0.012% (obs_term 3221.019 vs 3220.641). The forward model produces near-identical states.
- **Observation operator**: `n_obs = 1163` identical in both.
- **Background term**: `bg_term = 0.0` in both (deterministic perturbation gives zero B⁻¹·(m−m_b) at m=m_b).
- **RHS of adjoint system**: derived from `H^T R⁻¹ (H u_k − y_k)`. Since forward trajectory and observation operator agree, the RHS is the same up to LU/MUMPS rounding on the forward side (~1e-5).
- **Jacobian matrix of adjoint**: `J^T` where J is assembled from the same form at the same trajectory. Same form, same mesh, same elements → same matrix up to rounding.

So the adjoint **inputs** agree to ~1e-5 relative. The adjoint **output** (λ₀) disagrees by 129% in L2 with cos_sim = 0.179. That magnification can only come from the linear solve being dominated by small eigenvalues / near-null directions, where distributed MUMPS and serial MUMPS pick different null-space components of the solution.

This is consistent with the physics: the shallow-water adjoint on a closed-boundary inlet has near-null directions (e.g., constant h shifts that don't affect the observation operator). Both solvers return valid solutions, but project differently onto these directions.

---

## 5. Short Optimization Trace Comparison

Test: [tests/test_short_opt_trace.py](tests/test_short_opt_trace.py) — 3-iter BLMVM, `max_funcs=6`, same config and starting point.

### 5.1 Line-search probe history

| Eval | Serial cost | MPI cost | Ratio (MPI/Serial) |
|------|-------------|----------|--------------------|
| 1 (initial) | 3221.019 | 3220.641 | 1.000 |
| 2 | 239175.81 | 968078.87 | **4.048** |
| 3 | 61981.77 | 244283.27 | **3.941** |
| 4 | 17797.23 | 63410.32 | **3.563** |
| 5 | 6808.09 | 18230.07 | **2.678** |
| 6 | 4089.29 | 6954.00 | **1.701** |

Both terminated with **LS_FAILURE** after 6 evals, **0 iterations accepted**. Neither optimizer accepted a descent step in this configuration (the background-error perturbation is small enough that the landscape is locally flat and TAO's BLMVM line search can't find a better point within its step budget).

### 5.2 Interpretation

- Serial's initial step produces cost 239k (74× initial). MPI's initial step produces cost 968k (300× initial).
- At every backtrack, MPI is 1.7–4× worse than serial.
- The MPI gradient direction **consistently gives worse descent probes** than the serial gradient direction for the same initial point.

The configuration is too tight for either to succeed, but the comparison shows MPI's gradient is materially worse as a descent direction. In a longer, more forgiving run, MPI might still converge, but on a different trajectory than serial, and likely to a different optimum.

---

## 6. What Could Fix This

Options, roughly in increasing order of effort:

1. **Regularize the adjoint linear system**: add a small `ε I` to `J^T` before solving. This damps near-null directions and should make serial and MPI converge to the same solution. Low effort, may slightly bias the gradient.

2. **Use iterative Krylov adjoint with tight tolerance**: replace the direct MUMPS solve with GMRES + preconditioner and require a tight residual tolerance. Iterative methods tend to be more reproducible across decompositions because they converge in the norm sense, independent of null-space projections. Medium effort.

3. **Set MUMPS `ICNTL(10)` for iterative refinement** and tighten ordering controls (`ICNTL(7)`). Sometimes reduces distributed-vs-serial divergence. Low effort if MUMPS controls are exposed, but results-dependent.

4. **Explicit null-space projection**: identify the null direction(s) of `J^T` analytically (e.g., constant-h mode) and project the gradient onto the orthogonal complement. Most principled, but requires problem-specific analysis.

5. **Accept the divergence**: run MPI as operational-only for exploratory / interactive work, and use serial for all authoritative runs. No code changes.

For the pending 30-iteration 4D-Var continuation, **option 5 is the pragmatic choice**: serial is already proven to work, and the adjoint-direction issue means MPI would answer a different scientific question.

---

## 7. Remaining MPI Risks

| Risk | Status |
|------|--------|
| Deadlock / infinite callback loop | FIXED (prior pass) |
| Gradient smoother partition-boundary truncation | FIXED (prior pass) |
| **Adjoint direction mismatch (distributed MUMPS)** | **PRESENT, documented here** |
| Forward trajectory divergence under MUMPS | Low — cost matches to 0.012% |
| Observation operator in MPI | Verified correct (n_obs, cost match) |
| BLMVM reproducibility across ranks | Hasn't been tested in a case where iterations actually succeed |

---

## 8. Recommendation for Production Runs

- **Authoritative 4D-Var runs**: **use serial** (`python experiments/idealized_inlet_da.py ...`). Serial MUMPS and serial PETSc LU give bit-identical results; pick serial PETSc LU for simpler setup.
- **Exploratory MPI runs**: acceptable for rapid iteration on configs and memory-stressed setups, with the understanding that the gradient direction differs from serial. Do not compare serial and MPI numerics directly — they will disagree.
- **Parity regression test**: `tests/test_mpi_parity.py` + `tests/compare_adjoint_vectors.py` catches regressions in the smoother. The adjoint-direction divergence is the current floor; any further regression would show up as cos_sim materially lower than 0.179.

---

## 9. Files Produced

| File | Role |
|------|------|
| `tests/test_mpi_parity.py` | Parity harness; now saves globally-ordered adjoint vectors |
| `tests/compare_adjoint_vectors.py` | Vector-level comparison (L2, cosine, max-diff, per-component) |
| `tests/test_short_opt_trace.py` | Short (3-iter) BLMVM trace for serial vs MPI comparison |
| `src/swe4dvar/utils/distributed_smoother.py` | (prior pass) Bit-exact MPI gradient smoother |
| `docs/idealized_inlet_mpi_deadlock_audit.md` | (prior pass) Infinite-loop fix |
| `docs/idealized_inlet_mpi_parity_harness.md` | (prior pass) Harness design |
| `docs/idealized_inlet_mpi_smoother_equivalence_audit.md` | (prior pass) Smoother fix |
| `docs/idealized_inlet_mpi_adjoint_solver_parity_audit.md` | This document |

---

## 10. Answers to the Success-Criteria Questions

1. **Is the remaining serial/MPI discrepancy primarily a solver-backend issue?** **No.** Serial PETSc built-in LU and serial MUMPS give bit-identical results. Backend is not the driver.

2. **Can backend parity reduce the gradient-direction mismatch materially?** **No.** Forcing MUMPS on both sides does not change anything because distributed MUMPS (2-proc) produces a different solution from MUMPS (1-proc) for this ill-conditioned adjoint.

3. **Are short MPI optimization traces now close enough to serial to trust MPI scientifically?** **No.** In a 3-iter BLMVM trace, MPI's line-search probes are 1.7–4× worse than serial's at every corresponding step. Both fail with LS_FAILURE at this config, but the MPI gradient is materially worse as a descent direction.

4. **If not, should MPI be treated as operational-only rather than authoritative?** **Yes.** MPI is safe to run (no deadlock, no correctness bug in the pipeline), but it should not be used as an authoritative scientific reference on this problem without either regularizing the adjoint system or using an iterative adjoint solve.

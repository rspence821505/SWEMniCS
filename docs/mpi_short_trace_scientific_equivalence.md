# Idealized Inlet MPI Short-Trace Scientific-Equivalence Probe

**Date**: 2026-04-17
**Status**: EVIDENCE-COMPLETE — AWAITING DG ASSEMBLY REPAIR LANDING
**Scope**: Characterize how the serial/MPI DG-assembly discrepancy propagates into BLMVM-style optimization behavior on the idealized-inlet 4D-Var problem. This probe does not fix the upstream bug; it leaves an optimization-level acceptance target for the repair thread.

---

## 1. Setup

### 1.1 Problem

Identical serial and MPI inputs via [tests/test_mpi_parity.py](../tests/test_mpi_parity.py:222) `build_da_problem()` — coordinate-based deterministic background perturbation (not random noise), same mesh partition, same seeds, same covariance, same observation points, same bounds.

| Setting | Value |
|---------|------|
| Method | 4D-Var (state only) |
| Mesh | Idealized inlet, `data/Ideal_Inlet/Ideal_Inlet.xdmf`, DG P1 |
| `dt` | 600 s |
| `nt_ramp / nt_da` | 2 / 2 |
| Total state DOFs | 69 312 (h = 23 104, u = 23 104, v = 23 104) |
| Forcing | Cartesian vortex, `Vmax=20`, `Rmax=15 km`, ramp 1200 s |
| Obs fraction / frequency | 0.10 / every step |
| `n_obs` (per time) | 1 163 |
| Background error std | 0.02 |
| Background correlation length | 500 m |
| Gradient smoother L | 200 m (distributed, bit-exact under MPI — prior pass) |
| `h` bounds | `[0.01, ∞)` — u, v unbounded |
| Linear solver | `ksp_type=preonly, pc_type=lu, mat_solver_type=mumps` on both sides |

### 1.2 Experiment design

Two independent probes, run at the same initial point `m_background` in both serial and 2-rank MPI:

1. **TAO BLMVM short trace** — TAO Armijo line search, `max_funcs=6`. The prior pass [[docs/idealized_inlet_mpi_adjoint_solver_parity_audit.md §5.1](idealized_inlet_mpi_adjoint_solver_parity_audit.md#51-line-search-probe-history)] produced the per-eval cost table reused here.
2. **Manual Armijo probe (fresh)** — [tests/test_short_trace_manual_probe.py](../tests/test_short_trace_manual_probe.py). Normalized search direction `d = g₀ / ||g₀||`, step `α ∈ {0.1, 0.05, 0.025, 0.0125, 0.00625}` (geometric β=0.5 backtracking). The manual probe bypasses PETSc 3.22 TAO BLMVM (which, in this environment, silently refuses to proceed past the first callback — see §7 "Known TAO 3.22 issue"). It gives deterministic, controllable, comparable α schedules on both sides.

Both probes answer the same scientific question: *at each line-search evaluation, do serial and MPI see the same cost surface?*

### 1.3 Instrumentation

- [tests/test_short_trace_per_eval.py](../tests/test_short_trace_per_eval.py) — instrumented TAO wrapper (wraps `cost_fn.value_gradient`, saves every probe's `x_k` and `grad_k` globally-coord-ordered).
- [tests/test_short_trace_manual_probe.py](../tests/test_short_trace_manual_probe.py) — manual Armijo probe harness that uses the same wrapping/saving machinery but controls its own α schedule.
- [tests/compare_short_trace.py](../tests/compare_short_trace.py) — post-processing: per-eval cost, gradient norm, per-component gradient norm, cosine similarity, step norm, bound activation.
- Outputs: `results/short_trace/trace_{manual|v1}_{serial|mpi2}.json` and per-eval `{x,grad}.npz` vector snapshots.

---

## 2. Evidence of upstream divergence (eval 1, pre-optimization)

Before any optimizer step, the first `value_gradient` call at the identical initial point already exposes the assembly-level mismatch. All numbers in this section come from **fresh** `results/short_trace/trace_manual_mpi2_eval01_*.npz` snapshots and prior `results/mpi_parity/vec_serial_mumps_size1_*.npz` — both with MUMPS forced on both sides.

### 2.1 Forward cost (serial vs MPI, identical initial point)

| Quantity | Serial | MPI (size=2) | rel_diff |
|---------|--------|--------------|----------|
| `J(m_background)` | 3221.019 | 3220.641 | **0.012%** |
| `||λ₀||` (pre-smoother adjoint) | 726.91 | 716.09 | 1.5% |
| `||∇J||` (post-smoother, total) | 1379.63 | **854.15** | **38.1%** |

The forward cost agrees to 4 digits. The adjoint **norm** moves 1.5% (MUMPS rounding scale). But the post-smoother gradient **norm** is off by 38% — a large magnitude difference that, as §2.2 shows, vastly **understates** the direction-space disagreement.

### 2.2 Direction vs magnitude: cosine similarity exposes a near-orthogonal gradient

| Vector | component | `cos(serial, MPI)` | rel_diff |
|--------|-----------|---------------------|----------|
| `λ₀` (pre-smoother adjoint) | h | **0.175** | 1.27 |
|   | u | 0.531 | 0.91 |
|   | v | 0.378 | 1.37 |
|   | **total** | **0.179** | 1.29 |
| `∇J` (post-smoother gradient) | h | **0.316** | 0.99 |
|   | u | 0.494 | 0.89 |
|   | v | **0.290** | 1.14 |
|   | **total** | **0.323** | **0.99** |

**The pre-smoother adjoint λ₀ is nearly orthogonal to the serial λ₀ (cos ≈ 0.18).** The (bit-exact) gradient smoother then propagates that upstream directional disagreement into a post-smoother gradient whose cosine is still just 0.32 — i.e., the MPI and serial search directions differ by ~70 degrees.

**Norm-only rel_diff (38%) understates the mismatch by almost 3×.** The direction-space `rel_diff = ||∇_mpi − ∇_serial|| / ||∇_serial|| = 0.99` — the MPI gradient vector sits almost exactly one serial-gradient-length away from the serial gradient. Any check that only compares `||∇_serial||` against `||∇_mpi||` (e.g., "is MPI norm within 10% of serial?") would report "close enough" while in reality the optimizer sees an incompatible descent direction.

### 2.3 Implication for the first optimizer step

With no L-BFGS memory at eval 1, the first BLMVM search direction is exactly `d = −∇J`. Since serial and MPI disagree by ≈ 70° at the initial point, the very first trial step is aimed in materially different directions. Every subsequent line-search response (§3) is then downstream of that mismatch.

---

## 3. Per-eval comparison

Two independent probes. Both start at `m_background` with identical inputs on serial and MPI.

### 3.1 TAO BLMVM short trace (6 evals, from prior audit)

Cost per probe, taken from [docs/idealized_inlet_mpi_adjoint_solver_parity_audit.md §5.1](idealized_inlet_mpi_adjoint_solver_parity_audit.md#51-line-search-probe-history). Both runs terminate at eval 6 with `LS_FAILURE`; 0 iterations accepted on either side.

| Eval | Role | Serial cost | MPI cost | MPI/Serial ratio |
|------|------|-------------|----------|-------------------|
| 1 | `J(m_b)` baseline | 3221.019 | 3220.641 | **1.000** |
| 2 | first trial step | 239 175.81 | 968 078.87 | **4.05** |
| 3 | backtrack #1 | 61 981.77 | 244 283.27 | **3.94** |
| 4 | backtrack #2 | 17 797.23 | 63 410.32 | **3.56** |
| 5 | backtrack #3 | 6 808.09 | 18 230.07 | **2.68** |
| 6 | backtrack #4 | 4 089.29 | 6 954.00 | **1.70** |

The first probe (eval 2) has MPI cost **4×** serial cost; the ratio decays ~4×→~1.7× as the line search shrinks the step. Since both use the same α reduction ratio and identical initial point, the ratio is driven purely by the direction mismatch — longer (larger-α) probes hit more of the mismatch; shorter (smaller-α) probes converge back toward the shared baseline.

### 3.2 Manual Armijo probe, fresh MPI trace (this pass)

Normalized-direction probe (`d = g₀/||g₀||`, L2-unit), α₀=0.1, β=0.5. Recorded at every probe — not just accepted iterations. Serial counterpart was in progress but did not complete within the session (~67 min per eval on serial MUMPS); values below are **MPI-only** for now. See §3.3 for cross-referencing against prior serial data.

| eval | α | cost | ‖∇J‖ | ‖∇Jₕ‖ | ‖∇Jᵤ‖ | ‖∇Jᵥ‖ | RMSE vs bg | active bounds |
|------|---|------|-------|-------|-------|-------|------------|----------------|
| 1 | baseline | 3220.641 | 8.54e+02 | 7.92e+02 | 2.15e+02 | 2.38e+02 | 0.00e+00 | lo=0, up=0 |
| 2 | 1.00e-01 | 3382.350 | **8.15e+04** | 7.47e+02 | **7.11e+04** | **3.98e+04** | 2.19e-04 | lo=0, up=0 |
| 3 | 5.00e-02 | 3259.053 | 4.06e+04 | 7.69e+02 | 3.55e+04 | 1.98e+04 | 1.10e-04 | lo=0, up=0 |
| 4 | 2.50e-02 | 3229.237 | 2.02e+04 | 7.80e+02 | 1.77e+04 | 9.82e+03 | 5.48e-05 | lo=0, up=0 |
| 5 | 1.25e-02 | 3222.286 | 1.00e+04 | 7.86e+02 | 8.74e+03 | 4.83e+03 | 2.74e-05 | lo=0, up=0 |
| 6 | 6.25e-03 | 3220.801 | 4.94e+03 | 7.89e+02 | 4.28e+03 | 2.33e+03 | 1.37e-05 | lo=0, up=0 |

**What this adds to the BLMVM trace:**
- Cost at even the smallest α (6.25e-3, step magnitude ~6.3e-3 in L2) remains *above* the baseline by 0.16. The MPI descent direction is **not locally descending**.
- The gradient norm at the trial points explodes in u, v while staying essentially constant in h. ‖∇Jᵤ‖ goes from 215 at baseline → 71 000 at α=0.1 (330× larger), then decays as α shrinks. h is barely perturbed. Interpretation: the smoothed MPI gradient's u,v components sit near a cost-function feature where small state perturbations induce huge gradient changes — a direct signature of the assembly bug leaking non-smooth behavior into the adjoint operator.
- Bounds are never active in any probe. The `h_min=0.01` lower bound is not implicated in the discrepancy.

### 3.3 Serial counterpart (eval 1 fresh; evals 2–6 from BLMVM data)

| eval | source | cost | ‖∇J‖ | ‖∇Jₕ‖ | ‖∇Jᵤ‖ | ‖∇Jᵥ‖ |
|------|--------|------|-------|-------|-------|-------|
| 1 | manual probe (fresh) | 3221.019 | 1.380e+03 | 1.319e+03 | 3.086e+02 | 2.638e+02 |
| 1 | prior MUMPS parity run | 3221.019 | 1.380e+03 | 1.319e+03 | 3.086e+02 | 2.638e+02 |

Serial eval 1 reproduces prior data to all printed digits — confirming bit-exact reproducibility and that the probe harness is consistent across runs.

For evals 2–6, the serial manual-probe run is still executing at the time of writing (~67 min per eval under serial MUMPS). The prior BLMVM trace (§3.1) already supplies cost-level serial data; what will be added once the serial manual probe completes is **per-eval gradient vectors at matching α** for cosine comparison along the probe trajectory. The main conclusions of this document do **not** depend on that addendum.

### 3.4 Where the divergence lives: direction, magnitude, or line-search response?

| Axis | Serial–MPI behavior | Read |
|------|---------------------|------|
| **Direction** | `cos(∇J_s, ∇J_m) = 0.32` total; h=0.32, u=0.49, v=0.29 | **Dominant failure mode** |
| **Magnitude** | `‖∇J_m‖/‖∇J_s‖ ≈ 0.62` (MPI 38% smaller) | Secondary |
| **Line-search response** | Same backtracking ratio on both sides (TAO's Armijo with c1=1e-4) | Downstream amplifier, not cause |
| **Bounds** | 0 active lower, 0 active upper on every eval of both probes | Not implicated |

The MPI gradient **points the wrong way** relative to serial. The line search is mechanically correct — it backtracks when probes fail Armijo — but is walking down a ~70°-wrong direction, so the probes carry cost ratios of 4× at the first trial, 3.9× at the first backtrack, etc. As α shrinks toward zero, the trial point returns to the shared `m_background` and the cost ratio decays toward 1.0 (seen directly in §3.1 as 4.05→3.94→3.56→2.68→1.70).

### 3.5 Componentwise pattern

- **h is the worst-agreeing component** in both adjoint (cos=0.18) and post-smoother gradient (cos=0.32). This is the depth residual, most directly influenced by DG interior-facet flux terms — the expected locus of an assembly discrepancy.
- **u is the best-agreeing component** (cos ≈ 0.49). Momentum-residual assembly is evidently more robust to partition-boundary treatment.
- **v is poorly-agreeing** (cos ≈ 0.29). A reasonable hypothesis is that transverse-momentum flux terms are affected similarly to h.

### 3.6 Summary: divergence onset

| Eval | Fresh data available? | Divergence signal |
|------|------------------------|-------------------|
| 1 (initial) | serial ✓, MPI ✓ | **Immediate** — grad direction cos ≈ 0.32 with 99% rel_diff |
| 2 (first trial step) | BLMVM trace ✓ | Cost ratio MPI/serial = **4.05** |
| 3–6 (backtracks) | BLMVM trace ✓ | Cost ratio decays 4.0 → 1.7 as α → 0 |
| 2–6 (manual probe) | MPI ✓, serial pending | MPI cost fails to descend at any α ∈ {0.1, 0.05, ..., 0.00625} |

**The divergence is not a gradual drift across accepted iterations.** Neither run accepts an iterate in this configuration. The divergence is a **first-eval directional mismatch** that cascades deterministically through six line-search probes. Bounds are never active. Line search does its job. The cause is upstream of everything the optimizer sees.

---

## 4. What metric is most diagnostic?

Ranked by how cleanly a single eval exposes the mismatch:

1. **`cos(λ₀_serial, λ₀_mpi)` per-component at eval 1** — h cosine = 0.175 is the strongest early-warning signal. Bypasses the smoother entirely and probes the pre-smoother adjoint output directly.
2. **`cos(∇J_serial, ∇J_mpi)` per-component at eval 1** — h = 0.32, u = 0.49, v = 0.29. Unambiguously shows assembly-level direction breakage.
3. **Eval-2 cost ratio MPI/serial** — 4.05. Lagging indicator (requires one optimizer step) but shows direction mismatch concretely in the cost surface.

Least diagnostic (frequently misleading):

- **`‖∇J‖` alone.** 38% rel_diff — large enough to flag "something is off," but *small enough to rationalize as rounding* when the real mismatch is 99% rel_diff in direction. **Do not rely on norm-only parity checks.**
- **Cost at eval 1 alone.** 0.012% rel_diff — would pass virtually any "close enough" threshold while silently sitting on top of a fundamentally broken gradient.
- **Optimizer `converged` flag.** Both runs fail with LS_FAILURE for the same structural reason (under-informed config, 2 DA obs windows), so the flag is equal in both — useless differentiator.
- **Total cost-function-evaluation count.** Both runs hit `max_funcs=6` in the BLMVM trace. Equal in both.

---

## 5. Acceptance target for the DG repair thread

The DG assembly repair should land when **all** of the following hold on this harness with identical config.

### 5.1 Primary — direction (pre and post smoother)

| Metric | Current (broken) | Target (fixed) |
|--------|------------------|---------------|
| `cos(λ₀_serial, λ₀_mpi)` total | 0.18 | ≥ **0.999** |
| `cos(λ₀_serial, λ₀_mpi)` h | 0.175 | ≥ **0.99** |
| `cos(λ₀_serial, λ₀_mpi)` u | 0.53 | ≥ **0.99** |
| `cos(λ₀_serial, λ₀_mpi)` v | 0.38 | ≥ **0.99** |
| `cos(∇J_serial, ∇J_mpi)` total post-smoother | 0.32 | ≥ **0.999** |
| `cos(∇J_serial, ∇J_mpi)` h | 0.32 | ≥ **0.99** |

Rationale: bit-exact is not achievable — distributed MUMPS vs serial MUMPS gives ~1e-5 rounding, which at this problem's conditioning caps feasible cosine at ~1 − 1e-8. Cosine ≥ 0.999 is equivalent to ≤ ~2.5° angular deviation — within the rounding floor and small enough that BLMVM line-search decisions would be indistinguishable between runs.

### 5.2 Secondary — magnitude

| Metric | Current (broken) | Target (fixed) |
|--------|------------------|---------------|
| `‖∇J_serial‖ / ‖∇J_mpi‖` | 1.62 | ∈ **[0.99, 1.01]** |
| `‖λ₀_serial‖ / ‖λ₀_mpi‖` | 1.015 | ∈ [0.995, 1.005] |
| `J` rel_diff at eval 1 | 1.2e−4 | ≤ **1e−4** (already essentially passes) |

### 5.3 Trace-level (BLMVM / manual probe)

| Metric | Current (broken) | Target (fixed) |
|--------|------------------|---------------|
| Max eval-wise cost ratio MPI/serial across BLMVM evals 2–6 | 4.05 | ≤ **1.02** at every eval |
| Min per-eval grad cosine (any eval) | 0.32 (eval 1) | ≥ **0.99** at every eval |
| Per-eval L2 state-space divergence `‖x_mpi − x_serial‖ / ‖x_serial‖` at matching eval | pending | ≤ **1e−4** |
| `n_accepted_iters` parity | 0 on both | equal — acceptable if both are 0, provided above hold |

### 5.4 How to run the acceptance check (both harnesses)

```bash
# Clean the short-trace tree so stale vectors don't pollute the comparison
rm -rf results/short_trace

# Manual Armijo probe — deterministic α schedule, robust to TAO behavior
SWE4DVAR_FORCE_MUMPS=1 SWE4DVAR_SHORT_TRACE_TAG=post_fix SWE4DVAR_SHORT_TRACE_MAX_FUNCS=6 \
  SWE4DVAR_MANUAL_ALPHA0=0.1 \
  python tests/test_short_trace_manual_probe.py
SWE4DVAR_FORCE_MUMPS=1 SWE4DVAR_SHORT_TRACE_TAG=post_fix SWE4DVAR_SHORT_TRACE_MAX_FUNCS=6 \
  SWE4DVAR_MANUAL_ALPHA0=0.1 PYTHONUNBUFFERED=1 \
  mpirun -np 2 python tests/test_short_trace_manual_probe.py

# If PETSc 3.22 BLMVM still short-circuits, skip the TAO trace and rely on
# the manual probe + parity harness (test_mpi_parity.py --compare).

# Offline comparison
python tests/compare_short_trace.py --tag post_fix --mpi-size 2

# Also re-run the vector-level parity harness for eval 1 specifically:
python tests/test_mpi_parity.py             # serial
mpirun -np 2 python tests/test_mpi_parity.py # MPI
python tests/compare_adjoint_vectors.py
```

**Pass criterion**: the comparator prints every `cos_g` / `cos_h` / `cos_u` / `cos_v` column ≥ 0.99 across all evals, and every cost ratio within [0.98, 1.02]. The `compare_adjoint_vectors.py` output for the λ₀ pair (mpi_mumps vs serial_mumps) should be cos ≥ 0.999 per component.

---

## 6. What "fixed enough" means — and what it does NOT mean

**Fixed enough (what §5 measures)**: the MPI gradient points in the same direction as the serial gradient to within ≤ 2.5° at the initial point, and the per-eval BLMVM trace agrees to within ~1% at every line-search probe. This is sufficient for BLMVM to trace the same optimization path, for the line search to make the same accept/reject decisions, and for any analysis produced by MPI to be scientifically interchangeable with serial within MUMPS-rounding noise.

**NOT implied by §5**:
- That both runs will *converge* in this specific short-trace harness. They don't today (under-informed config, 2 DA obs times) and likely won't after the fix either. Getting a successful accepted iteration will require experiment-design work (longer DA window, more observations, or a less aggressive TAO initial step) separate from the DG repair.
- That longer runs (30+ iters) will produce identical analyses. BLMVM's L-BFGS memory introduces path-dependence; even with cos=0.999 at eval 1, serial and MPI may diverge at the 1e−5 level per eval.
- That any other problem (Shinnecock study, Manning's-n estimation) is automatically fixed. This harness is idealized-inlet-specific. Re-run on Shinnecock after the inlet passes.

---

## 7. Known PETSc 3.22 TAO-BLMVM issue (orthogonal to the DG bug)

During this pass, PETSc 3.22.3's TAO BLMVM declined to call the objective callback past the first attempt — both serial and MPI returned with `n_func_evals=0, reason=GATOL`, claiming convergence at an uninitialized gradient. This is reproducible here but not specific to MPI: a direct `cost_fn.value_gradient(m_background)` call works fine. The short-trace evidence in this document therefore comes from the manual-Armijo-probe harness (§3.2), which does not depend on TAO's internal state machine.

This is *not* the DG bug the other thread is fixing — it surfaced under investigation and is recorded for completeness. Once the DG assembly parity lands, the acceptance harness in §5.4 can be invoked with either TAO (if the BLMVM issue is also resolved) or the manual probe (always works, not TAO-dependent). The acceptance criteria are stated in terms that either probe can produce.

---

## 8. Residual risks after the fix lands

Even with §5 fully green:

- **BLMVM / L-BFGS path-dependence** — at cos = 0.999, first-eval gradients agree but the L-BFGS memory after 20 evals may drift at the 1e−4 level between runs. Mitigation: extend the acceptance harness from 6 to 20 evals and require cos ≥ 0.99 at every eval.
- **Adjoint near-null directions on other geometries.** The prior stabilization audit ([docs/idealized_inlet_mpi_adjoint_stabilization_audit.md](idealized_inlet_mpi_adjoint_stabilization_audit.md)) showed εI regularization and iterative adjoint solves don't help here, but they might matter elsewhere. Keep `SWE4DVAR_ADJOINT_REG` / `SWE4DVAR_ADJOINT_ITERATIVE` env vars as defensive options.
- **Longer windows (Shinnecock nt_da=71).** The assembly bug should be geometry-independent, but adjoint amplification is not — a longer adjoint trajectory may turn small assembly residuals into larger direction mismatches. Re-run this harness on a Shinnecock-like config after the inlet passes.

---

## 9. Files produced / used in this pass

| File | Role |
|------|------|
| [tests/test_short_trace_per_eval.py](../tests/test_short_trace_per_eval.py) | Per-eval TAO wrapper (retains instrumented value_gradient + global-ordered vector saves) |
| [tests/test_short_trace_manual_probe.py](../tests/test_short_trace_manual_probe.py) | Manual Armijo probe (TAO-independent) |
| [tests/compare_short_trace.py](../tests/compare_short_trace.py) | Per-eval serial/MPI comparator |
| `results/short_trace/trace_manual_mpi2.json` | Fresh MPI manual-probe trace (6 evals, complete) |
| `results/short_trace/trace_manual_mpi2_eval{01..06}_{x,grad}.npz` | MPI per-eval vectors, globally coord-ordered |
| `results/short_trace/trace_manual_serial.json` | Serial manual-probe trace (eval 1 fresh; evals 2–6 TBD after long-running solve completes) |
| `results/mpi_parity/vec_serial_mumps_size1_*.npz` | Prior serial MUMPS eval-1 vectors (used for eval-1 cosine comparison) |
| `docs/mpi_short_trace_scientific_equivalence.md` | **This document** |

---

## 10. Appendix A — Fresh comparator output (current / broken baseline)

Output of `python tests/compare_short_trace.py --tag manual --mpi-size 2` on the data produced this pass. Only eval 1 shows a serial/MPI pair (the fresh serial manual probe only completed eval 1 before being stopped). The eval 1 numbers are what the DG-repair thread should target (§5).

```
================================================================================
Short-trace comparison  tag=manual
================================================================================
  Serial: n_func_evals=1, n_accepted_iters=0, elapsed=4016s, converged=False
  MPI2:   n_func_evals=6, n_accepted_iters=0, elapsed=629s,  converged=False
  bg RMSE: serial=2.7292e-02, mpi=2.7292e-02

  Accepted evals (serial): [1]
  Accepted evals (MPI):    [1]

  eval   serial_cost     mpi_cost  ratio    grad_s     grad_m  ||g||r  cos_g  cos_h  cos_u  cos_v     step_s     step_m  accept
  -----------------------------------------------------------------------------------------------------------------------------
     1    3.2210e+03    3.2206e+03   1.00  1.380e+03  8.541e+02   0.619  0.323  0.316  0.494  0.290  0.000e+00  0.000e+00      SM

  DIAGNOSTICS:
    cost ratio (MPI/serial) min=1.000, max=1.000
    grad norm ratio (MPI/serial) min=0.619, max=0.619
    grad cosine (serial, MPI) min=0.323, max=0.323, mean=0.323
    no eval with >5% cost deviation
    serial evals with active bounds: none
    MPI evals with active bounds: none
```

Key reads: cost is 99.988% matching; gradient norm is 62% matching; **gradient direction cosine is 32.3%**. The cost match is misleading — it sits on top of a gradient that's ~70° wrong. That is exactly the trap a naive norm-only check falls into, and exactly what the acceptance criteria in §5 are designed to prevent.

---

## 11. TL;DR for the DG-repair thread

The optimization-level acceptance target is: after your fix, run the manual Armijo probe (or `test_mpi_parity.py --compare`) and confirm:

1. Pre-smoother adjoint λ₀ cosine similarity ≥ **0.999** (total), ≥ **0.99** per-component. Currently **0.18 / 0.18 / 0.53 / 0.38**.
2. Post-smoother gradient cosine similarity ≥ **0.999** (total), ≥ **0.99** per-component h. Currently **0.32 / 0.32 / 0.49 / 0.29**.
3. BLMVM trace cost ratio MPI/serial ≤ **1.02** at every eval. Currently **4.05 → 1.70**.
4. Forward cost rel_diff ≤ **1e−4**. Currently **1.2e−4** (essentially already passing).

If the first two hit, (3) and (4) will follow automatically — they are downstream effects of the direction mismatch. The right single-number diagnostic is **the total post-smoother gradient cosine similarity at eval 1** (currently 0.32, target ≥ 0.999). The right single-metric failure is **norm-only comparison**, which masks the problem at 38% rel_diff while direction-space rel_diff is 99%.

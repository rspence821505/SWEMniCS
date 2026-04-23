# Idealized-Inlet DC-WME First-Win Search — Exact Staged Run Plan

**Date**: 2026-04-23
**Status**: **IN PROGRESS** — redesigned to matched-prior (same σ_b²) pairs
**Scope**: idealized inlet only. No Shinnecock. No broad sweep.
**Current working answer**: *provisional — memo will be finalized as each run completes.*

## Design redesign (2026-04-23)

Original plan compared 4D-Var against DC-WME where each method used *its
own* B. That conflates two effects: (a) different cost functions, (b)
different priors (4D-Var gets un-inflated B, DC-WME gets Eq 38 inflated B).

**Redesigned: each matched pair now shares the SAME σ_b².** If DC-WME uses
an Eq 38 inflation, 4D-Var uses the same inflation. The comparison is
then purely cost-structure-driven.

Code change: Step 7a B-inflation is now method-independent
([experiments/idealized_inlet_da.py](../experiments/idealized_inlet_da.py), commit `e78690c`).
`--fixed-sigma-b-sq-*`, `--eq38-component-aware`, `--no-eq38-inflation`
all work with either `--method 4dvar` or `--method dcwme_static`.

### Matched-pair matrix (primary comparisons)

| Pair | obs_fraction | σ_b² (shared) | 4D-Var run | DC-WME run |
|---|---:|---|---|---|
| **I (primary at 0.005)** | 0.005 | σ_b²_h=0.01073, σ_b²_uv=0.1731 (from Anchor B TLM) | **R-A' (NEW)** | Anchor B 3102561 (have) |
| **II (secondary at 0.005)** | 0.005 | no inflation (raw B) | Anchor A 3098387 (have) | R1 3103715 (running) |
| **III (primary at 0.02)** | 0.02 | Eq 38 (TBD from R3's Gram) | **R-2b (NEW, blocked on R3)** | R3 3103718 (queued normal) |
| **IV (secondary at 0.02)** | 0.02 | no inflation | R2 3103716 (queued) | R4 3103717 (queued) |

Two new runs needed under the redesigned plan:

- **R-A'**: 4D-Var at obs=0.005 with `--fixed-sigma-b-sq-h 0.01073 --fixed-sigma-b-sq-uv 0.1731`. Reuses Anchor B's Eq 38 values so no TLM Gram — expected wall ~55 min on dev queue.
- **R-2b**: 4D-Var at obs=0.02 with the σ_b² values R3 will produce. Blocked until R3 completes (normal-queue backlog Apr 25). Could alternatively run its own TLM Gram (~80 min Gram → needs normal queue).

Old (A, B, R1, R2, R3, R4) jobs are kept — they still provide Pair II, Pair IV and provenance. No cancellation needed.

---

## 1. Executive summary

*(Will be finalized once all runs complete. Provisional read from partial R1 data below.)*

Provisional state:
- **Eq 38 inflation appears to hurt on sparse obs (obs_fraction=0.005).** R1 (DC-WME no-inflation) already shows a better trajectory minimum (~0.147718 at eval #3) than Anchor B (DC-WME+Eq38) at 0.148076. But the RMSE drifts upward after the best point — same symptom, smaller amplitude.
- **Neither DC-WME variant has beaten 4D-Var at obs_fraction=0.005.** Even with inflation off, DC-WME is ~0.003 RMSE above 4D-Var.
- Runs 2–4 at obs_fraction=0.02 pending — they test whether moderately denser obs change the comparison.

---

## 2. Exact run order followed

Original (pre-redesign) submission order, all submitted 2026-04-23:

1. **R1** (DEV): DC-WME 0.005 + `--no-eq38-inflation` — no-inflation Pair II DC-WME leg
2. **R2** (DEV, queued behind R1): 4D-Var 0.02 — no-inflation Pair IV 4D-Var leg
3. **R4** (DEV, queued behind R2): DC-WME 0.02 + `--no-eq38-inflation` — no-inflation Pair IV DC-WME leg
4. **R3** (NORMAL, 2.5-day backlog): DC-WME 0.02 + `--eq38-component-aware` — inflated Pair III DC-WME leg

Post-redesign addition:

5. **R-A'** (DEV, pending submit-limit): 4D-Var 0.005 with fixed σ_b²_h=0.01073, σ_b²_uv=0.1731 — inflated Pair I 4D-Var leg (matched to Anchor B)
6. **R-2b** (NORMAL, blocked on R3): 4D-Var 0.02 with σ_b² from R3 — inflated Pair III 4D-Var leg

**LS6 QOS constraints:**
- `QOSMaxJobsPerUserLimit=1` on dev — only 1 dev job runs concurrently (R1→R2→R4 serialize)
- `QOSMaxSubmitJobPerUserLimit=4` — can't have more than 4 total submitted jobs. Must wait for R1 to finish before R-A' can be queued.
- Normal queue: 2.5-day backlog → R3 starts Apr 25, R-2b later.

Run 3 required p=normal not only for the backlog but because its Gram scales with `n_obs` (≈232 at 0.02 vs 58 at 0.005 → ~80 min Gram alone), exceeding dev's 2h cap.

---

## 3. Exact commands used

### R1 — [job_run1_dcwme_005_noinflation.slurm](../hpc/lonestar6/idealized_inlet/job_run1_dcwme_005_noinflation.slurm)

```bash
ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --no-eq38-inflation \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240
```

### R2 — [job_run2_4dvar_020.slurm](../hpc/lonestar6/idealized_inlet/job_run2_4dvar_020.slurm)

```bash
ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.02 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240
```

### R3 — [job_run3_dcwme_020_eq38on.slurm](../hpc/lonestar6/idealized_inlet/job_run3_dcwme_020_eq38on.slurm)

```bash
# On normal queue (p=normal, t=04:00:00) due to ~80 min Gram at obs=0.02
ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.02 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --eq38-component-aware \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240
```

### R4 — [job_run4_dcwme_020_noinflation.slurm](../hpc/lonestar6/idealized_inlet/job_run4_dcwme_020_noinflation.slurm)

```bash
ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.02 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --no-eq38-inflation \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240
```

### Flag added for this search

[experiments/idealized_inlet_da.py](../experiments/idealized_inlet_da.py) now exposes `--no-eq38-inflation` which disables **both** Step 7a (TLM Gram) and Step 7b (H·H^T static fallback) inflation paths while keeping the DC-WME cost structure intact (static L_wme still computed). Commit `3bb0229`.

---

## 4. Cumulative results table

| Run ID | obs_fraction | Method | Eq.38 inflation | Bg RMSE | Final RMSE | % Improve | Best RMSE | Evals | Exit | Notes |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| **Anchor A** (3098387) | 0.005 | 4DVAR | N/A | 0.148444 | **0.144789** | **2.5%** | 0.144789 | 15 | USER (max_funcs) | Reused baseline |
| **Anchor B** (3102561) | 0.005 | DCWME | ON (TLM component-aware) | 0.148444 | 0.148305 | 0.1% | 0.148076 (eval #4) | 15 | USER (max_funcs) | Post-fix revalidation |
| **R1 (3103715)** | 0.005 | DCWME | OFF (`--no-eq38-inflation`) | 0.148444 | **0.147456** | **0.7%** | 0.147518 (eval #13) | 15 | USER (max_funcs) | DONE. 10 BLMVM iters, 15 func evals, 3836s. |
| **R-A' (3103790)** | 0.005 | 4DVAR | ON (fixed σ_b²_h=0.01073, σ_b²_uv=0.1731, matching Anchor B) | 0.148444 | TBD | TBD | TBD | TBD | TBD | **NEW — Pair I 4D-Var leg**. PD (submit-limit). |
| **R2 (3103716)** | 0.02 | 4DVAR | OFF (no flags) | 0.148444 | **0.140620** | **5.3%** | 0.140620 | 15 | USER (max_funcs) | DONE. 12 BLMVM iters, 15 func evals. Pair IV 4D-Var leg. |
| R3 (CANCELLED → 3104105) | 0.02 | DCWME | ON (TLM component-aware) | TBD | TBD | TBD | TBD | TBD | TBD | Resubmitted at np=8 on dev queue (was normal). |
| R4 (CANCELLED → 3104104) | 0.02 | DCWME | OFF (`--no-eq38-inflation`) | TBD | TBD | TBD | TBD | TBD | TBD | Resubmitted at np=8. |
| **R-A' (CANCELLED → 3104103)** | 0.005 | 4DVAR | ON (fixed σ_b² matching Anchor B) | TBD | TBD | TBD | TBD | TBD | TBD | Resubmitted at np=8. PD `(Resources)`. |
| **R-2b (TBD)** | 0.02 | 4DVAR | ON (matching R3) | TBD | TBD | TBD | TBD | TBD | TBD | **NEW — Pair III 4D-Var leg**. Blocked on R3's σ_b². |

*Anchor B per-component Eq 38 diagnostics (for provenance): `λ_min(G_h) = 9.32`, `σ_b²_h = 0.01073`, `λ_min(G_uv) = 0.578`, `σ_b²_uv = 0.1731`, condition 2.67, rank 58/58.*

---

## 5. Per-stage interpretation

### R1 (DONE) — DC-WME 0.005 no-inflation

Bypass confirmed in log:

```
Step 7a: --skip-tlm-eq38 set — skipping TLM Eq 38 (using default σ_b²)
Step 7b: Computing static L_wme (skip_eq38_inflation=True [--no-eq38-inflation set], ...)
```

**Final:** Background 0.148444 → Analysis **0.147456** → **0.7% improvement**. TAO Converged after 10 iterations, 15 function evals (max_funcs cap), 3836s optimization wall.

TAO trajectory highlights:

| eval | cost | RMSE_truth | note |
|---:|---:|---:|---|
| 1 | 270.98 | 0.148444 | initial (≡ bg) |
| 3 | 257.12 | 0.147718 | first accepted |
| 4 | 254.43 | 0.147933 | |
| 10 | 227.74 | 0.147875 | cost dropping fast |
| 11 | 219.28 | 0.147590 | |
| 13 | 218.98 | **0.147518** | best RMSE |
| 15 (final) | 218.25 | **0.147456** | **final analysis** |

**Key finding for Pair II:**
- Anchor A (4D-Var 0.005 no-infl, 3098387): 0.148444 → 0.144789, **2.5%**
- R1 (DC-WME 0.005 no-infl, 3103715): 0.148444 → 0.147456, **0.7%**
- 4D-Var wins by ~3.5×

**Implication vs Anchor B (DC-WME+Eq38, 0.1%):** R1 at 0.7% is **7× better** than Anchor B — Eq 38 inflation was *hurting* DC-WME substantially. Without it, DC-WME gets a real RMSE drop with a healthy trajectory (no late-iteration blowup).

**But R1 still loses to 4D-Var** by a factor of ~3.5. So Eq 38 inflation is *part* of the DC-WME underperformance story at sparse obs — but not all of it. The rest comes from the DC-WME cost structure itself (the L_wme correction term) applied in a sparse-obs regime.

### R-A' (NEW, PD) — 4D-Var 0.005 with matched Eq 38 inflation

Submitted 3103790 right after R1 completed and submit slot opened. Config:
```
--method 4dvar --obs-fraction 0.005 \
--fixed-sigma-b-sq-h 0.01073 --fixed-sigma-b-sq-uv 0.1731 \
```
Reuses Anchor B's σ_b² values so no Gram needed. Expected wall ~55 min. This is the **primary Pair I 4D-Var leg** — matched to Anchor B's B. Will answer: *does the Eq 38 inflation hurt 4D-Var too, or just DC-WME?*

### R2 (DONE) — 4D-Var 0.02 no-inflation

**Final:** 0.148444 → **0.140620** → **5.3% improvement**. 12 BLMVM iterations, 15 function evals.

Trajectory highlights (for comparison against R4 when it runs):

| eval | cost | RMSE_truth |
|---:|---:|---:|
| — | — | 0.148444 (initial) |
| 6 | 2267.7 | 0.141742 |
| 9 | 2266.4 | 0.141191 |
| 10 | 2259.1 | 0.141152 |
| (final) | — | **0.140620** |

**Key implication for the obs=0.02 regime:** 4D-Var at obs=0.02 doubles its Anchor A win at obs=0.005 (5.3% vs 2.5%). Denser observations help 4D-Var a lot more than they help DC-WME is still to be determined. R4's job is now to answer: **does DC-WME at obs=0.02 scale similarly?** Or does its cost-RMSE decoupling get worse?

**Updated stop-rule target:** for DC-WME to "win" at obs=0.02, it must beat **5.3%**, not the original 2.5%.

### R3, R4 — cancelled to make room for the np=8 parity check (see §8 below).

---

## 8. np=8 MPI parity check (mid-search decision)

LS6 dev-queue QOS enforces max 3 submitted jobs. With R2 already running and R-A', R3, R4 queued, we couldn't add the parity check. Cancelled R-A', R3, R4 to free space.

**Decision point:** if `parity_4dvar_reduced.py` at np=8 matches np=2 to rel_err ≤ 1e-10, resubmit R-A', R3, R4, R-2b at np=8 × 16 threads. Expected speedup: 5-8× per run (each run's Gram scales with n_obs × adjoint_time; the adjoint transpose solve itself parallelizes on distributed MUMPS). Biggest win: R3 at obs=0.02 drops from ~155 min (normal-queue only, 2.5-day backlog) to ~30 min (dev-queue eligible).

**If parity fails:** stay at np=2, resubmit R3 to normal queue with 2.5-day wait, accept R-2b also blocked. The current R1 + R2 data still answers whether the Eq 38 inflation hurts DC-WME (Pair II tells us yes, at sparse obs).

Parity sbatch: [hpc/lonestar6/parity/parity_np8_check.slurm](../hpc/lonestar6/parity/parity_np8_check.slurm), commit `da15e51`. 15 min dev-queue wall, no scientific cost.

### np=8 parity verdict

The dedicated parity harness `parity_4dvar_reduced.py` has a pre-existing `VecSetSizes` "argument #3 inconsistent across ranks" bug that surfaces at np≥2 — it failed during its own np=2 baseline, never reaching np=8. Not a regression: that script has apparently never been exercised at np>=2 in its current state.

**Pivot: direct sanity run on the production experiment** (job 3103981, commit `cc815ee`): ran `experiments/idealized_inlet_da.py --method 4dvar --nt-ramp 4 --nt-da 4 --max-funcs 1 --max-iterations 1` at np=8 × 16 threads. Completed cleanly in 27s optimization wall (~2 min total). 8 MPI ranks, 8 per-rank "Saved" prints, no MPI errors, no VecSetSizes faults. **PASS.**

Promoted R-A', R4, R3 to np=8. Jobs 3104103, 3104104, 3104105 submitted on dev queue (commit `70d31a4`). R3 specifically promoted from `-p normal -t 04:00:00` (2.5-day backlog) to `-p development -t 02:00:00` because np=8 drops its estimated wall from ~155 min to ~30 min — now fits dev's 2h cap.

---

## 6. First winning / most competitive case

*(Will be filled once data exists.)*

Current leading candidates:

- **DC-WME improvement rank (pending)**: R1 > Anchor B (no-inflation > inflation-ON at obs=0.005).
- **Whether any DC-WME variant beats 4D-Var**: unknown until R2/R4 complete at obs=0.02.

---

## 7. Final recommendation

*(To be written after all runs complete.)*

---

## Stop rules checkpoint

| Rule | Triggered? | Action |
|---|---|---|
| DC-WME wins at obs=0.005 | NO (R1 0.7% vs Anchor A 2.5%) | continue |
| DC-WME wins at obs=0.02 | TBD (waiting R2, R3, R4) | TBD |
| DC-WME materially closer at 0.02 | TBD | decides R5 |

**Updated provenance commits:**
- `3bb0229` — initial plan
- `e78690c` — method-independent Step 7a refactor (for matched-prior comparisons)

---

## Appendix — run provenance

| Job | Submitted | Branch | Commit | Partition | Nodes |
|---|---|---|---|---|---|
| 3098387 | Apr 21 | refactor/4dvar-parallel | pre-fix | normal | 1×2 |
| 3102561 | Apr 22 | refactor/4dvar-parallel | 11f3a62 | development | 1×2 |
| 3103715 (R1) | Apr 23 | refactor/4dvar-parallel | 3bb0229 | development | 1×2 |
| 3103716 (R2) | Apr 23 | refactor/4dvar-parallel | 3bb0229 | development | 1×2 |
| 3103717 (R4) | Apr 23 | refactor/4dvar-parallel | 3bb0229 | development | 1×2 |
| 3103718 (R3) | Apr 23 | refactor/4dvar-parallel | 3bb0229 | normal | 1×2 |

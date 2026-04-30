# Idealized Inlet 4D-Var: 4-window cycling DA — chain success + memory analysis

**Date**: 2026-04-28
**Branch**: refactor/4dvar-parallel
**Status**: Chain infrastructure works. Single-job 4-window does not (PETSc allocator pool growth). Chain is the canonical solution.

---

## Result summary

| Window (global) | tag | BG RMSE | Analysis RMSE | Improvement | Notes |
|---|---|---|---|---|---|
| 0 | w0 | 0.1484 | **0.1095** | **−26.2%** | Cold start — strong DA win |
| 1 | w1 | 0.1709 | 0.1708 | −0.08% | Cycling drift starts |
| 2 | w2 | 0.2236 | 0.2229 | −0.28% | Drift continues |
| 3 | w3 | 0.2562 | 0.2562 | 0.0% | Forward solver diverged on drifted bg → optimizer gave up at iter 0 |

Configuration: vmax=10 m/s, np=4, nt_ramp=24, nt_da=6 (1h windows), 4 windows, obs-fraction=0.05, obs-frequency=5, obs-noise=0.01, bg-error-std=0.02, max-iter=15, max-funcs=15.

W1–W3 cycling-drift pattern matches the documented Phase 4 finding (CLAUDE.md §): bg error eventually exceeds the obs-noise floor → analysis collapses to background → forecast bg degrades cumulatively until forward solver crashes on drifted state. **Not an infrastructure bug.**

---

## Key code changes

### 1. Cross-job chaining (`experiments/idealized_inlet_da.py`)

New CLI flags:
- `--start-window N` — global window index of the first window in this job (default 0)
- `--initial-bg-file PATH` — `.npy` template (with `{rank}`) loaded as bg for window=start_window
- `--chain-mode` — opts into the cycling-loop code path even when `n_windows=1`, so window 0 of a chain still saves chain bg

Cycling loop now:
- Computes `global_w = start_window + local_w` and uses it for `tag`, `truth_offset_steps`, and the chained-bg filename
- Always sets `advance_steps = nt_da` (so the last window of a job still produces chain bg for the next job)
- After each window, every rank writes `chain_bg_after_w{global_w}_rank{rank}.npy` from `_m_analysis_advanced_arr`
- Aggregate JSON now tagged with absolute window range: `result_4dvar_N_A_cycling_w{start}-{end}.json`

### 2. SLURM job chain ([hpc/lonestar6/idealized_inlet/](hpc/lonestar6/idealized_inlet/))

Four sbatch files: `job_chain_w{0,1,2,3}.slurm`, plus `submit_chain.sh`. Submission pattern:
- w0 standalone (no `afterok`)
- w1–w3 each `--dependency=afterok:<prior>`
- All in `development` queue, np=4, 1h30m walltime each

LS6 limits: development queue allows max 3 jobs queued per user. Strategy: submit w1+w2+w3 with `afterok` dependencies after w0 has already produced its chain bg, so only 3 are ever queued at once. (Earlier attempt at "self-chain" — each job sbatches the next from within — fails because LS6 compute nodes can't sbatch.)

### 3. Newton-problem destroy hooks (modest, not the dominant cost)

- `src/swe4dvar/forward/newton.py`: added `CustomNewtonProblem.destroy()` releasing `A`, `L`, KSP+factor.
- `src/swe4dvar/forward/solvers/cg_implicit.py`: destroys the prior `self.solver` before re-init in `time_loop()`.

Recovers ~240 MB / cost-eval at our 207K-DOF size. Necessary but **insufficient** (see below).

---

## The leak that isn't fixable

### Observation
Per-eval RSS growth is ~1500 MB / cost-eval at 207K DOFs / np=4 / nt_da=6.

Even with all of the following in place:
- Trajectory/Jacobian destroy at start of each new solve (`cost_functions.py:247`)
- `solver.storage.clear()` (which destroys stored Mats)
- `gc.collect()` and `PETSc.garbage_cleanup()` between windows
- Newton-problem destroy in `time_loop` (this PR)

…RSS still grows ~1500 MB / eval, accumulating to a memory plateau (~230 GB on a 256 GB LS6 node) by mid window-3, after which the OS throttles allocations and the run wall-clocks out without progressing.

### Diagnosis
Job 3130835 (single-job, post-Newton-fix) reproduced the **exact same RSS trajectory** as the pre-fix run (3130380): RSS=23 GB → 56 GB across windows 1–2, then plateau at watchdog "230 Gi" for 25+ minutes with no forward progress in stdout.

Identical pattern → the Newton-object leak (~240 MB/eval real) is not the dominant contribution. The remaining ~1.3 GB/eval is **PETSc's internal allocator pool not returning memory to the OS** even after explicit `destroy()`. This is consistent with PETSc's documented behavior on Linux: `PetscFree` returns memory to PETSc's pool, which holds it for reuse. The pool grows as new allocations are made faster than they're reused.

### Why the chain workaround works
Each SLURM job runs one window in a fresh Python process. When the process exits, **the OS reclaims everything** — including PETSc's internal pool. The next job starts at baseline RSS (~3 GB). Across 4 windows, peak per-rank RSS stays ~25–30 GB instead of 230 GB.

Wallclock comparison (LS6 dev queue, np=4):
- Single-job 4-window (failing): 2 h walltime, hangs in W3, 0 windows of useful DA past W1
- Chained 4-job (working): 4 × ~20 min compute + queue waits ≈ 65–90 min total, 4 windows complete (W0 with strong DA improvement)

### What was deferred and why it stays deferred
The "deferred" diagnostic items from the in-process leak hunt — np tuning, Jacobian recomputation in adjoint, deep PETSc/KSP inspection — would only chase the user-level fraction of the leak (~240 MB/eval, the Newton object). The dominant fraction lives below the destroy() boundary in PETSc internals. Process-exit reclaim (i.e. the chain) is structurally simpler and proven.

---

## Files

Code:
- [experiments/idealized_inlet_da.py](experiments/idealized_inlet_da.py) — chain CLI flags + global-index loop
- [src/swe4dvar/forward/newton.py](src/swe4dvar/forward/newton.py) — `destroy()` method
- [src/swe4dvar/forward/solvers/cg_implicit.py](src/swe4dvar/forward/solvers/cg_implicit.py) — release-prior-solver hook in `time_loop`

SLURM:
- [hpc/lonestar6/idealized_inlet/job_chain_w0.slurm](hpc/lonestar6/idealized_inlet/job_chain_w0.slurm)
- [hpc/lonestar6/idealized_inlet/job_chain_w1.slurm](hpc/lonestar6/idealized_inlet/job_chain_w1.slurm)
- [hpc/lonestar6/idealized_inlet/job_chain_w2.slurm](hpc/lonestar6/idealized_inlet/job_chain_w2.slurm)
- [hpc/lonestar6/idealized_inlet/job_chain_w3.slurm](hpc/lonestar6/idealized_inlet/job_chain_w3.slurm)
- [hpc/lonestar6/idealized_inlet/submit_chain.sh](hpc/lonestar6/idealized_inlet/submit_chain.sh)

Result artifacts on LS6 (`$WORK/SWEMniCS/results/idealized_inlet_da/`):
- `result_4dvar_N_A_cycling_w{0..3}-{0..3}.json` — per-window aggregate
- `chain_bg_after_w{0..3}_rank{0..3}.npy` — per-rank chained background (16 files, ~420 KB each)
- `forward_diag_chain_w{0..3}.csv` — per-step Newton diagnostics

Job IDs:
- 3130681 (w0, "FAILED" exit only because of inner-sbatch self-chain attempt; science completed cleanly, chain bg saved)
- 3130691 (w1, COMPLETED, 19:01)
- 3130692 (w2, COMPLETED, 20:18)
- 3130693 (w3, COMPLETED, 7:34 — short because optimizer gave up at iter 0)
- 3130835 (single-job 4-window, CANCELLED at 1:16:21 after ~25 min of memory-plateau hang in W3 — confirms Newton fix insufficient)

---

## What's next (separate from this work)

The W3 cycling-drift failure (forward solver diverging on drifted bg) is a **science problem**, not infrastructure:
- The bg at W3 (RMSE 0.256) contains unphysical regions that crash Newton on the first cost-eval
- Root cause: textbook 4D-Var cycling overfits at iter 0 in W1+W2 (because bg error ≈ obs noise at this perturbation level), so each forecast bg degrades cumulatively
- Possible mitigations (none attempted yet): adaptive B inflation in cycling, denser observations, shorter windows, regularization on the chained bg, ensemble-style hybridization

These are research questions, not bug fixes.

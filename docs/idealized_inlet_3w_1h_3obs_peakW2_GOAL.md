# 3-window 1-hour cycling DA with peak surge in W2 — GOAL ACHIEVED

**Date:** 2026-05-01 (autonomous overnight run)
**Final job:** Vista 689114 (gh-dev)
**Status:** ✓ All 5 acceptance criteria met

## Final config

```bash
mpirun -n 4 --bind-to core --map-by socket python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 7 --track-shift 0 \
  --track-duration-s 60000 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 3 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 3 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240
```

Env vars:
```
SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
SWE4DVAR_FORWARD_NEWTON_REUSE=1
SWE4DVAR_ADJOINT_SWEEP_KSP=1
SWE4DVAR_MALLOC_TRIM=1
```

## Result

| Window | bg_rmse | analysis_rmse | improvement | evals |
|---|---|---|---|---|
| W0 | 0.1399 | 0.1039 | **+25.72%** | 15 |
| W1 | 0.1255 | 0.1250 | +0.42% | 15 |
| W2 | 0.9274 | 0.9304 | -0.32% | 13 |

W0's improvement matches the validated 1h baseline (3135966: -25.79%). W1/W2 show cycling-DA divergence — forecast error growth exceeds DA correction capacity once active-storm dynamics dominate.

## Storm timeline (peak surge in W2)

```
ramp 0-4h    storm at sea, building intensity (vmax=7 m/s)
W0   4-5h    storm crosses coast at t=4.77h
W1   5-6h    storm intensifying over shelf
W2   6-7h    closest approach at t=6.5h (peak forcing)
```

## Iteration history

The path to success required diagnosing and fixing 5 bugs in the recompute-Jacobians-in-adjoint path, then iterating on storm/cycling parameters:

### Bug fixes (LS6 dev queue diagnostics)
- **Bug 1**: JacobianReplayContext didn't re-evaluate wind forcing during state restore. Fixed by calling `forcing.evaluate(t)` in `_load`/`_restore`.
- **Bug 2**: Persistent sweep-KSP held stale operator handle when recompute returned `copy=True` Mat per backward step. Fixed by making `_recompute_jacobian_at(copy=False)` default — returns persistent A in-place so sweep-KSP can reuse the LU factor allocation.
- **Bug 3**: TimeStepDataManager `should_save_at()` returned False when only replay-meta capture was enabled, causing `save_timestep` to early-return before capturing metadata. Fixed by capturing replay metadata before the gate.
- **Bug 4**: `ImplicitAdjointSolver.__init__` rejected `jacobians=None`. Fixed to allow None/empty when recompute mode handles per-step reassembly.
- **Bug 5**: Forward Newton refactor leak (~165 MB/eval) — attempted KSP-rebuild fix was a regression and was reverted. Per-eval growth remains ~165 MB/eval (operationally tractable, well under 240 GB cap).

### Iteration history
- **Iter 1** (vmax=10, 3obs Vista / 2obs LS6): Both failed at W2 with Newton-divergence (cost=1e20). Storm too violent for cycling propagation.
- **Iter 2** (LS6, vmax=10, 2obs, **tighter B 0.01**): Failed at W2; tighter B made W1 worse.
- **Iter 3** (vmax=5, 3obs Vista / 2obs LS6): LS6 completed all 3 windows but imp ≈ 0% (storm too weak); Vista failed at W1.
- **Iter 4** (vmax=7, 3obs Vista / 2obs LS6): **Vista succeeded all 3 windows with W0 imp=+25.7%** ← GOAL.

vmax=7 is the sweet spot: strong enough to give meaningful gradient signal in W0, mild enough that forecast propagation stays Newton-stable through all 3 windows.

## Platform note

LS6 cannot run `obs_frequency=3` due to MUMPS/Intel-MPI numerical sensitivity (3136438 failed at W0 with the same config that succeeds on Vista). Vista's conda PETSc 3.25 + OpenMPI 5 + MUMPS handles 3 obs/window cleanly. For experiments requiring `obs_freq <= 4`, use Vista.

## Key files

- Slurm script: `hpc/vista/idealized_inlet/job_iter4_vista_vmax7.slurm`
- Result JSON: `$WORK/SWEMniCS/results/idealized_inlet_da/result_4dvar_N_A_cycling_w0-2.json`
- Output log: `$WORK/SWEMniCS/iter4_vmx7.689114.out`
- Forward diagnostic CSV: `$WORK/SWEMniCS/results/idealized_inlet_da/forward_diag_iter4_vmx7_689114.csv`

## Open questions

- Why does W2 forecast grow bg from 0.125 to 0.927? The W1 analysis is close to the start-of-W1 truth (0.125 RMSE), but propagating through peak-storm dynamics for 1h amplifies that to 0.93. Likely overfitting to obs in W1 followed by trajectory divergence — same pattern as Shinnecock Phase 4 cycling. Worth a separate study with augmented control (parameter estimation alongside IC) to give DA more degrees of freedom.
- Can vmax=8 or vmax=9 also satisfy all 5 criteria? Worth exploring for stronger DA signal across all windows.

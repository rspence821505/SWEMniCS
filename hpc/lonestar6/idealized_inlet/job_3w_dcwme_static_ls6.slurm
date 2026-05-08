#!/bin/bash
# 3w static DC-WME 4D-Var on idealized inlet (debug for 10w followup).
#
# Same physics as 3w_slow (3139864) but method=dcwme_static. Compare:
#   - 4D-Var slow-ramp 3w (3139864): W0 imp +9.87%, W1 bg=0.100,
#     W2 bg=0.120 (slow ramp delays cycling ceiling)
#   - This run: does L_wme produce meaningful eigenvalue structure
#     on idealized inlet? On Shinnecock all eigs were identical
#     (Phase 3b/3c) -> DC-WME degenerated to uniform reweighting.
#
# Goal: characterize per-window DC-WME wall (so we can size 10w
# walltime) and confirm L_wme behavior on idealized-inlet dynamics
# before committing to longer run.
#
# 4h walltime on normal queue: 4D-Var 3w_slow took ~2h for 7
# windows -> ~30 min/window. Static DC-WME adds L_wme assembly
# overhead (~1.5x baseline). 3 * 45 + 30 setup = ~3h estimated;
# 4h gives 33% margin.
# --------------------------------------------------------
#SBATCH -J 3w_dcwme
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 06:00:00
#SBATCH -o 3w_dcwme_ls6.%j.out
#SBATCH -e 3w_dcwme_ls6.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

export SWE4DVAR_NEWTON_RELAX=0.3
# Recompute-Jacobians-in-adjoint disabled for DC-WME (validated by
# 3150948 smoke test): Eq 38 TLM adjoint can't honor the replay-meta
# capture path on the truth-solver code path. Using stored Jacobians
# instead — ~1-2 GB more memory per window, but pipeline works
# end-to-end (Step 7a Gram cond 4-5.6 vs Shinnecock 1.0).
unset SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS
unset SWE4DVAR_CAPTURE_REPLAY_META
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_3w_dcwme_ls6_${SLURM_JOB_ID}.csv"
rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 7 --track-shift 0 \
  --track-duration-s 157000 \
  --dt 300 \
  --nt-ramp 48 --nt-da 12 \
  --n-windows 3 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 6 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --smooth-length 1500 \
  --max-iterations 3 --max-funcs 3 \
  --mem-limit-gb 240
RET=$?

echo
echo "=== Per-window results ==="
for j in $RES_DIR/result_dcwme_*cycling_w*-*.json; do
  [[ -f "$j" ]] || continue
  echo "  $(basename "$j")"
  python3 -c "import json; d=json.load(open('$j'));
[print(f\"    W{w['window']-1}: bg={w['bg_rmse']:.4f}  an={w['analysis_rmse']:.4f}  imp={w['improvement_pct']:+.2f}%  evals={w['n_func_evals']}\") for w in d['windows']]" 2>/dev/null
done
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

#!/bin/bash
# vmax=30 + dt=150 probe: does halving dt unlock the v30 cycling that
# failed at dt=300? Storm geometry identical to 3164268 (track-dur=43200,
# peak at t=6h in W2). Only dt is changed.
#
# At dt=150: nt-ramp=96 (4h ramp), nt-da=24 (1h/window). Storm peak in W2.
#   W0: t=4-5h  W1: t=5-6h  W2: t=6-7h  (peak at 6h)
#
# Normal-queue rerun of dev probe 3176985 (which walltime-capped at W1
# timestep 14/24). W0 confirmed dt=150 cleared v30 Newton fragility
# (+5.92% imp, no failures). This run needs to reach W2 (storm peak).
#
# Expected pace: ~50 min/window at np=8 dt=150 max-funcs=3 → 2.5h for 3w.
# 4h walltime gives ~60% margin for peak-window slowdown.
# --------------------------------------------------------
#SBATCH -J 3w_v30_dt150_n
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 04:00:00
#SBATCH -o 3w_w2peak_v30_dt150_h0p01_uv0p15_4dvar_normal.%j.out
#SBATCH -e 3w_w2peak_v30_dt150_h0p01_uv0p15_4dvar_normal.%j.err

set -uxo pipefail
shopt -s nullglob
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

export SWE4DVAR_NEWTON_RELAX=0.3
unset SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS
unset SWE4DVAR_CAPTURE_REPLAY_META
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
TAG="3w_w2peak_4dvar_v30_dt150_h0p01_uv0p15_normal"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_4dvar_N_A_cycling_w*-*.json 2>/dev/null

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 30 --track-shift 0 \
  --track-duration-s 43200 \
  --dt 150 \
  --nt-ramp 96 --nt-da 24 \
  --n-windows 3 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 6 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --smooth-length 1500 \
  --max-iterations 3 --max-funcs 3 \
  --fixed-sigma-b-sq-h 0.01 --fixed-sigma-b-sq-uv 0.15 \
  --mem-limit-gb 240
RET=$?

for j in $RES_DIR/result_4dvar_N_A_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== vmax=30 dt=150 3w peak-W2 results ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  echo "  $(basename "$j")"
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}') for w in d['windows']]" 2>/dev/null
done

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

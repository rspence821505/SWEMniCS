#!/bin/bash
# DC-WME static v=20 + WD α=1.5, 6-window cycling with peak in W3.
# Matched companion to the 4D-Var 6w v=20 run. Includes the W2-hang
# fix (SWE4DVAR_FORWARD_NEWTON_REUSE=0) — without it this hangs at W2.
#
# track-duration-s=54000 → peak at t=7.5h sim = mid-W3 (0-indexed).
# Per finding_dcwme_reuse0_w2hang_fix.md: REUSE=0 confirmed at v=20.
# --------------------------------------------------------
#SBATCH -J 6w_dcw_v20
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH -o 6w_w3peak_v20_wd_dcwme.%j.out
#SBATCH -e 6w_w3peak_v20_wd_dcwme.%j.err

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
# W2-hang fix: fresh MUMPS factor per Newton call
export SWE4DVAR_FORWARD_NEWTON_REUSE=0
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

export SWE4DVAR_OUTPUT_SUBDIR=6w_v20_dcwme
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="6w_w3peak_v20_wd_dcwme"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 0 \
  --track-duration-s 54000 \
  --dt 150 \
  --nt-ramp 96 --nt-da 24 \
  --n-windows 6 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 6 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --smooth-length 1500 \
  --max-iterations 3 --max-funcs 3 \
  --fixed-sigma-b-sq-h 0.01 --fixed-sigma-b-sq-uv 0.15 \
  --wd --wd-alpha 1.5 \
  --mem-limit-gb 240
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== v=20 + WD DC-WME 6w peak-W3 (REUSE=0 fix) ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}') for w in d['windows']]" 2>/dev/null
done
echo
echo "=== Hang check (should be 0) ==="
echo -n 'Error allocating: '; grep -c 'Error allocating int buffer' /work/08398/tg876971/ls6/SWEMniCS/6w_w3peak_v20_wd_dcwme.${SLURM_JOB_ID}.out 2>/dev/null
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

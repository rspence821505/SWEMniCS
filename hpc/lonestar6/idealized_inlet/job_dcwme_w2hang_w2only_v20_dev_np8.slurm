#!/bin/bash
# DC-WME W2-hang fix test at v=20: chain from W1 of the hung 14w run
# (3184909, track-dur=118800), run JUST W2 with NEWTON_REUSE=0.
# Confirms whether the v=10-validated fix transfers to v=20's stronger
# forcing. ~1-1.5h, fits dev cap.
# --------------------------------------------------------
#SBATCH -J dcw_w2only_v20
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o dcw_w2only_v20.%j.out
#SBATCH -e dcw_w2only_v20.%j.err

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
# The fix
export SWE4DVAR_FORWARD_NEWTON_REUSE=0
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

# Reuse the 14w_v20_dcwme subdir where 3184909 saved chain_bg_after_w1
export SWE4DVAR_OUTPUT_SUBDIR=14w_v20_dcwme
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
TAG="dcw_w2only_v20_REUSE0"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

# Don't wipe chain_bg — need chain_bg_after_w1
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 0 \
  --track-duration-s 118800 \
  --dt 150 \
  --nt-ramp 96 --nt-da 24 \
  --n-windows 1 --start-window 2 --chain-mode \
  --initial-bg-file "$RES_DIR/chain_bg_after_w1_rank{rank}.npy" \
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
echo "=== DC-WME W2-only v=20 with REUSE=0 ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}') for w in d['windows']]" 2>/dev/null
done
echo
echo "=== Hang check (should be 0) ==="
echo -n 'Error allocating: '; grep -c 'Error allocating int buffer' /work/08398/tg876971/ls6/SWEMniCS/dcw_w2only_v20.${SLURM_JOB_ID}.out 2>/dev/null
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

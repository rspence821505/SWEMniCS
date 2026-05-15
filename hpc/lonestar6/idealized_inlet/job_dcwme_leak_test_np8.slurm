#!/bin/bash
# Leak test: 4-window DC-WME at np=8 with the cb769b0 cleanup patch.
# Verifies the fix BEFORE the long np=4 normal-queue run completes.
# Same config as 3162647 OOM (h=0.01, uv=0.15, track-dur=144000)
# but truncated to 4 windows to fit dev queue 2h.
# Pass: completes all 4 windows without OOM.
# Fail: OOM at W3-W4 like 3162647 → fix didn't catch the leak.
# --------------------------------------------------------
#SBATCH -J dcw_leak
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o dcwme_leak_test_np8.%j.out
#SBATCH -e dcwme_leak_test_np8.%j.err

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

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 7 --track-shift 0 \
  --track-duration-s 144000 \
  --dt 300 \
  --nt-ramp 48 --nt-da 12 \
  --n-windows 4 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 6 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --smooth-length 1500 \
  --max-iterations 3 --max-funcs 3 \
  --fixed-sigma-b-sq-h 0.01 --fixed-sigma-b-sq-uv 0.15 \
  --mem-limit-gb 240

echo
echo "=== leak-test result (4 windows) ==="
for j in $RES_DIR/result_dcwme_static_cycling_w0-3.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j')); print(f'partial: {d.get(\"partial\",\"?\")}, windows: {len(d[\"windows\"])}'); [print(f'  W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f} an={w[\"analysis_rmse\"]:.4f} imp={w[\"improvement_pct\"]:+.2f}%') for w in d['windows']]" 2>/dev/null
done
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

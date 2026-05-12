#!/bin/bash
# Asymmetric: sigma_b^2_h=0.075, sigma_b^2_uv=0.15
#SBATCH -J h0p075_uv15
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -o h0p075_uv0p15.%j.out
#SBATCH -e h0p075_uv0p15.%j.err

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

run_one () {
  local method="$1"; local tag="$2"
  rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
  rm -f $RES_DIR/result_4dvar_N_A_cycling_w*-*.json 2>/dev/null
  rm -f $RES_DIR/result_dcwme_*cycling_w*-*.json 2>/dev/null
  export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${tag}_${SLURM_JOB_ID}.csv"
  ibrun python -u experiments/idealized_inlet_da.py \
    --method "$method" --vmax 7 --track-shift 0 \
    --track-duration-s 157000 --dt 300 --nt-ramp 48 --nt-da 12 \
    --n-windows 1 --start-window 0 --chain-mode \
    --obs-fraction 0.05 --obs-frequency 6 \
    --obs-noise-level 0.01 --background-error-std 0.02 \
    --smooth-length 1500 --max-iterations 3 --max-funcs 3 \
    --fixed-sigma-b-sq-h 0.075 --fixed-sigma-b-sq-uv 0.15 \
    --mem-limit-gb 240 || echo "  $tag IBRUN FAILED"
  for j in $RES_DIR/result_*cycling_w*-*.json; do
    cp "$j" "$RES_DIR/asym_${tag}_$(basename $j)" 2>/dev/null || true
  done
}

run_one 4dvar        "4dvar_h0p075_uv0p15"
run_one dcwme_static "dcwme_h0p075_uv0p15"

echo
echo "=== h=0.075 uv=0.15 results ==="
python3 <<PYEOF
import json, glob, os
res_dir = "$RES_DIR"
for tag in ("4dvar_h0p075_uv0p15","dcwme_h0p075_uv0p15"):
    files = sorted(glob.glob(os.path.join(res_dir, f"asym_{tag}_result_*.json")))
    if not files: print(f"{tag}: <missing>"); continue
    d = json.load(open(files[-1]))
    for w in d.get('windows',[]):
        if w['window']-1 != 0: continue
        print(f"  {tag}: bg={w['bg_rmse']:.4f} an={w['analysis_rmse']:.4f} imp={w['improvement_pct']:+.2f}%")
PYEOF
sacct -j $SLURM_JOB_ID --format=JobID,State,Elapsed 2>&1 || true

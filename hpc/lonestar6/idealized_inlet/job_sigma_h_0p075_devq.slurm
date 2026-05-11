#!/bin/bash
# sigma_b^2_h = sigma_b^2_uv = 0.075 (1 of 3 dev queue jobs filling
# the 0.01-0.075 gap in the sigma_b^2_h sweep).
#
# Companion to 3156451 (0.1) and prior data at 4e-4, 0.0209, 1630.
# Maps the non-monotonic dip in 4D-Var/DCWME W0 imp between 4e-4 (high
# imp) and 0.0209 (deep dip) and back up at 0.1.
#
# 2 runs (4D-Var + DC-WME) × ~45 min each ≈ 1:30 total, fits dev queue 2h.
# --------------------------------------------------------
#SBATCH -J h_0p075
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -o sigma_h_0p075.%j.out
#SBATCH -e sigma_h_0p075.%j.err

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
  local method="$1"
  local tag="$2"
  echo
  echo "=== $tag : method=$method sigma_b^2=0.075 ==="
  rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
  rm -f $RES_DIR/result_4dvar_N_A_cycling_w*-*.json 2>/dev/null
  rm -f $RES_DIR/result_dcwme_*cycling_w*-*.json 2>/dev/null
  export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${tag}_${SLURM_JOB_ID}.csv"

  ibrun python -u experiments/idealized_inlet_da.py \
    --method "$method" \
    --vmax 7 --track-shift 0 \
    --track-duration-s 157000 \
    --dt 300 \
    --nt-ramp 48 --nt-da 12 \
    --n-windows 1 --start-window 0 --chain-mode \
    --obs-fraction 0.05 --obs-frequency 6 \
    --obs-noise-level 0.01 --background-error-std 0.02 \
    --smooth-length 1500 \
    --max-iterations 3 --max-funcs 3 \
    --fixed-sigma-b-sq-h 0.075 \
    --fixed-sigma-b-sq-uv 0.075 \
    --mem-limit-gb 240 \
    || echo "  $tag IBRUN FAILED (continuing)"

  for j in $RES_DIR/result_*cycling_w*-*.json; do
    cp "$j" "$RES_DIR/sigh_${tag}_$(basename $j)" 2>/dev/null || true
  done
}

run_one 4dvar         "4dvar_h0p075"
run_one dcwme_static  "dcwme_h0p075"

echo
echo "=== SUMMARY sigma=0.075 (job_id=$SLURM_JOB_ID) ==="
python3 <<PYEOF
import json, glob, os
res_dir = "$RES_DIR"
fmt = "%-22s %12s %12s %12s %12s"
print()
print(fmt % ("variant","bg_rmse","an_rmse","imp%","evals"))
print("-"*72)
for tag in ("4dvar_h0p075","dcwme_h0p075"):
    files = sorted(glob.glob(os.path.join(res_dir, f"sigh_{tag}_result_*.json")))
    if not files:
        print(fmt % (tag, "<missing>", "", "", ""))
        continue
    d = json.load(open(files[-1]))
    for w in d.get('windows',[]):
        if w['window']-1 != 0: continue
        print(fmt % (tag, f"{w['bg_rmse']:.4f}", f"{w['analysis_rmse']:.4f}",
                     f"{w['improvement_pct']:+.2f}", str(w['n_func_evals'])))
PYEOF

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

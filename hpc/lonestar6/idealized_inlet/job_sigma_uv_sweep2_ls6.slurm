#!/bin/bash
# σ_b²_uv sweep v2: 100, 1000, 10000 (missing values from 3155858).
#
# Prior sweep 3155858 failed at variant #5 due to slurm wrapper bug
# (cp on non-existent JSON tripped set -e). Got clean data at uv=0.1,
# 10, but missed uv=100, 1000 and never tested 10000. This rerun
# fills the gap and extends to 10000 as a strong-inflation point.
#
# Bug fix: shopt -s nullglob + defensive cp || true. Robust to any
# silent ibrun failure leaving missing JSON.
#
# σ_b²_h fixed = 0.0209 throughout (consistent with prior runs).
# 3 uv values × 2 methods = 6 runs × ~45 min ≈ 4.5h. 5h walltime.
# --------------------------------------------------------
#SBATCH -J sigma_uv2
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 05:00:00
#SBATCH -o sigma_uv_sweep2.%j.out
#SBATCH -e sigma_uv_sweep2.%j.err

set -uxo pipefail   # NOTE: dropped -e so missing-JSON cleanup can't kill batch
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
  local sigma_uv="$2"
  local tag="$3"
  echo
  echo "=========================================================="
  echo "=== $tag : method=$method σ_b²_uv=$sigma_uv ==="
  echo "=========================================================="
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
    --fixed-sigma-b-sq-h 0.0209 \
    --fixed-sigma-b-sq-uv "$sigma_uv" \
    --mem-limit-gb 240 \
    || echo "  $tag IBRUN FAILED (continuing sweep)"

  # Defensive: nullglob handles empty match, || true handles cp on partial files
  for j in $RES_DIR/result_*cycling_w*-*.json; do
    cp "$j" "$RES_DIR/swp2_${tag}_$(basename $j)" 2>/dev/null || true
  done
}

# 3 σ_b²_uv values × 2 methods = 6 runs
run_one 4dvar         100      "4dvar_uv100"
run_one dcwme_static  100      "dcwme_uv100"
run_one 4dvar         1000     "4dvar_uv1000"
run_one dcwme_static  1000     "dcwme_uv1000"
run_one 4dvar         10000    "4dvar_uv10k"
run_one dcwme_static  10000    "dcwme_uv10k"

echo
echo "=========================================================="
echo "=== SWEEP v2 SUMMARY (job_id=$SLURM_JOB_ID) ==="
echo "=========================================================="
python3 <<PYEOF
import json, glob, os
res_dir = "$RES_DIR"
fmt = "%-22s %12s %12s %12s %12s"
print()
print(fmt % ("variant","bg_rmse","an_rmse","imp%","evals"))
print("-"*72)
for tag in ("4dvar_uv100","dcwme_uv100","4dvar_uv1000","dcwme_uv1000",
            "4dvar_uv10k","dcwme_uv10k"):
    files = sorted(glob.glob(os.path.join(res_dir, f"swp2_{tag}_result_*.json")))
    if not files:
        print(fmt % (tag, "<missing>", "", "", ""))
        continue
    d = json.load(open(files[-1]))
    for w in d.get('windows',[]):
        if w['window']-1 != 0:
            continue
        print(fmt % (tag, f"{w['bg_rmse']:.4f}", f"{w['analysis_rmse']:.4f}",
                     f"{w['improvement_pct']:+.2f}", str(w['n_func_evals'])))
PYEOF

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

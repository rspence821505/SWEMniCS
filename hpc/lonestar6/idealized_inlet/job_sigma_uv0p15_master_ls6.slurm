#!/bin/bash
# uv=0.15 family master sweep (smaller scope than 3157411 sigma_asym).
# Drops the uv=0.1 pairs (those now covered by dev jobs 3157836/3157837).
#
# 4 sigma_b^2_h values × 2 methods = 8 runs × ~45 min = ~6h baseline.
# 7h walltime, normal queue. Shorter walltime than 3157411's 12h to
# get better backfill priority.
# --------------------------------------------------------
#SBATCH -J sigma_uv15
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 07:00:00
#SBATCH -o sigma_uv0p15.%j.out
#SBATCH -e sigma_uv0p15.%j.err

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
  local sigma_h="$2"
  local sigma_uv="$3"
  local tag="$4"
  echo
  echo "=========================================================="
  echo "=== $tag : method=$method h=$sigma_h uv=$sigma_uv ==="
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
    --fixed-sigma-b-sq-h "$sigma_h" \
    --fixed-sigma-b-sq-uv "$sigma_uv" \
    --mem-limit-gb 240 \
    || echo "  $tag IBRUN FAILED (continuing)"

  for j in $RES_DIR/result_*cycling_w*-*.json; do
    cp "$j" "$RES_DIR/asymm_${tag}_$(basename $j)" 2>/dev/null || true
  done
}

# uv=0.15 family only (4 sigma_h x 2 methods = 8 runs)
run_one 4dvar         0.01   0.15  "4dvar_h0p01_uv0p15"
run_one dcwme_static  0.01   0.15  "dcwme_h0p01_uv0p15"
run_one 4dvar         0.02   0.15  "4dvar_h0p02_uv0p15"
run_one dcwme_static  0.02   0.15  "dcwme_h0p02_uv0p15"
run_one 4dvar         0.05   0.15  "4dvar_h0p05_uv0p15"
run_one dcwme_static  0.05   0.15  "dcwme_h0p05_uv0p15"
run_one 4dvar         0.075  0.15  "4dvar_h0p075_uv0p15"
run_one dcwme_static  0.075  0.15  "dcwme_h0p075_uv0p15"

echo
echo "=========================================================="
echo "=== uv=0.15 FAMILY SUMMARY (job_id=$SLURM_JOB_ID) ==="
echo "=========================================================="
python3 <<PYEOF
import json, glob, os
res_dir = "$RES_DIR"
fmt = "%-32s %12s %12s %12s %12s"
print()
print(fmt % ("variant","bg_rmse","an_rmse","imp%","evals"))
print("-"*84)
for tag in ("4dvar_h0p01_uv0p15","dcwme_h0p01_uv0p15",
            "4dvar_h0p02_uv0p15","dcwme_h0p02_uv0p15",
            "4dvar_h0p05_uv0p15","dcwme_h0p05_uv0p15",
            "4dvar_h0p075_uv0p15","dcwme_h0p075_uv0p15"):
    files = sorted(glob.glob(os.path.join(res_dir, f"asymm_{tag}_result_*.json")))
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

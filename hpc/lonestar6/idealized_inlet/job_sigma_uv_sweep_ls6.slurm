#!/bin/bash
# σ_b²_uv sweep — find inflation level where DC-WME outperforms 4D-Var.
#
# Prior results at fixed σ_b²_h = 0.0209:
#   σ_b²_uv=0.0004 (default):  4D-Var W0=+11.76%, DC-WME ~+3.7%
#   σ_b²_uv=1630   (cmpawr):   4D-Var W0=+3.81%,  DC-WME W0=+1.86%
#
# Trend: as σ_b²_uv increases, both methods degrade but 4D-Var more.
# Gap narrowed 8pp → 2pp. This sweep extrapolates further to test
# whether at extreme uv inflation, DC-WME finally catches up or wins.
#
# 4 σ_b²_uv values × 2 methods = 8 runs, n-windows=1 each (~25 min/run).
# σ_b²_h fixed = 0.0209 throughout to isolate the uv knob.
# 5h walltime gives ~25% margin on 4h baseline estimate.
# --------------------------------------------------------
#SBATCH -J sigma_uv
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 05:00:00
#SBATCH -o sigma_uv_sweep.%j.out
#SBATCH -e sigma_uv_sweep.%j.err

set -euxo pipefail
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
  rm -f $RES_DIR/result_4dvar_N_A_cycling_w*-*.json
  rm -f $RES_DIR/result_dcwme_*cycling_w*-*.json
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
    || echo "  $tag FAILED (continuing sweep)"

  for j in $RES_DIR/result_*cycling_w*-*.json; do
    [[ -f "$j" ]] || continue
    cp "$j" "$RES_DIR/swp_${tag}_$(basename $j)"
  done
}

# Sweep — 4 σ_b²_uv values × 2 methods = 8 runs
run_one 4dvar         0.1     "4dvar_uv0p1"
run_one dcwme_static  0.1     "dcwme_uv0p1"
run_one 4dvar         10      "4dvar_uv10"
run_one dcwme_static  10      "dcwme_uv10"
run_one 4dvar         1000    "4dvar_uv1000"
run_one dcwme_static  1000    "dcwme_uv1000"
run_one 4dvar         100000  "4dvar_uv100k"
run_one dcwme_static  100000  "dcwme_uv100k"

echo
echo "=========================================================="
echo "=== SWEEP SUMMARY (job_id=$SLURM_JOB_ID) ==="
echo "=========================================================="
python3 <<PYEOF
import json, glob, os
res_dir = "$RES_DIR"
fmt = "%-22s %12s %12s %12s %12s"
print()
print(fmt % ("variant","bg_rmse","an_rmse","imp%","evals"))
print("-"*72)
for tag in ("4dvar_uv0p1","dcwme_uv0p1","4dvar_uv10","dcwme_uv10",
            "4dvar_uv1000","dcwme_uv1000","4dvar_uv100k","dcwme_uv100k"):
    files = sorted(glob.glob(os.path.join(res_dir, f"swp_{tag}_result_*.json")))
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

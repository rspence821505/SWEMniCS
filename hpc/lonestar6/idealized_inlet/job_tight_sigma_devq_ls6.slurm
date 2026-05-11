#!/bin/bash
# Tight σ_b² test (dev queue, fast turnaround).
#
# Direction 1 from the inflation strategy: DC-WME's predictability
# term ½(δQ, L_wme⁻¹ δQ) only differentiates from 4D-Var when the
# background prior is strong relative to observations. Tested here
# with σ_b²_h = σ_b²_uv = 1e-6 (~400x tighter than default 0.0004).
#
# Hypothesis: 4D-Var with very tight bg is forced near the bg
# (background term overwhelmingly dominates), barely moves from
# initial state -> small RMSE improvement. DC-WME has the same bg
# term but its L_wme adds directional structure — could let it
# move smarter in the constrained space.
#
# 2 runs in one slurm, single-window each. Pace ~43 min/run = 1:26
# total, fits dev queue 2h walltime.
# --------------------------------------------------------
#SBATCH -J tight_sb
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -o tight_sigma_devq.%j.out
#SBATCH -e tight_sigma_devq.%j.err

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
  local tag="$2"
  echo
  echo "=========================================================="
  echo "=== $tag : method=$method σ_b²_h=σ_b²_uv=1e-6 ==="
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
    --fixed-sigma-b-sq-h 1e-6 \
    --fixed-sigma-b-sq-uv 1e-6 \
    --mem-limit-gb 240 \
    || echo "  $tag FAILED (continuing)"

  for j in $RES_DIR/result_*cycling_w*-*.json; do
    [[ -f "$j" ]] || continue
    cp "$j" "$RES_DIR/tight_${tag}_$(basename $j)"
  done
}

run_one 4dvar         "4dvar_tight"
run_one dcwme_static  "dcwme_tight"

echo
echo "=========================================================="
echo "=== SUMMARY (job_id=$SLURM_JOB_ID, σ_b²_h=σ_b²_uv=1e-6) ==="
echo "=========================================================="
python3 <<PYEOF
import json, glob, os
res_dir = "$RES_DIR"
fmt = "%-18s %12s %12s %12s %12s"
print()
print(fmt % ("variant","bg_rmse","an_rmse","imp%","evals"))
print("-"*68)
for tag in ("4dvar_tight","dcwme_tight"):
    files = sorted(glob.glob(os.path.join(res_dir, f"tight_{tag}_result_*.json")))
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

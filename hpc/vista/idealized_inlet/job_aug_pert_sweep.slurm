#!/bin/bash
# Aug perturbation sweep on Vista np=1 (coarse mesh).
#
# Prior aug runs gave null results: 691222 (shift=0, bg_std=0.02)
# and 691308 (shift=5km, bg_std=0.02) both had TAO terminate at
# iter=1 with analysis ~= bg in both modes -- the gradient was
# below line-search tolerance.
#
# This sweep tests progressively stronger perturbations to find
# the threshold where the optimizer actually activates. Each
# variant runs --mode both (state_only vs augmented) on the
# coarse 50x30 mesh (~27K state DOFs). Per variant ~2-5 min.
#
# Variants:
#   v1: shift=5km,  bg_std=0.10  (5x larger IC perturbation)
#   v2: shift=15km, bg_std=0.02  (3x larger track shift)
#   v3: shift=15km, bg_std=0.10  (both)
#   v4: shift=30km, bg_std=0.10  (extreme — large bg + large shift)
#
# Pass criteria for a variant: state_only iterations > 1 AND
# augmented theta_norm > 0.01 (machinery activated). Compare
# forecast_end_rmse across modes only if both activated.
# --------------------------------------------------------
#SBATCH -J aug_sweep
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o aug_sweep.%j.out
#SBATCH -e aug_sweep.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

BASE_OUT="$WORK/SWEMniCS/outputs/aug_pert_sweep_${SLURM_JOB_ID}"
mkdir -p "$BASE_OUT"

run_variant () {
  local tag="$1"
  local shift="$2"
  local bgstd="$3"
  local out="$BASE_OUT/$tag"
  mkdir -p "$out"
  echo
  echo "=========================================================="
  echo "=== Variant $tag : shift=${shift}m bg_std=${bgstd} ==="
  echo "=========================================================="
  mpirun -n 1 --bind-to core --timeout 1500 \
    python -u experiments/idealized_inlet_augmented_serial.py \
    --mode both \
    --coarse-mesh --nx 50 --ny 30 \
    --output-dir "$out" \
    --dt 600 --nt-ramp 24 --nt-da 6 --nt-forecast 6 \
    --vmax 10 --track-shift "$shift" --track-duration-s 28800 \
    --obs-fraction 0.10 --obs-frequency 3 --obs-noise-level 0.01 \
    --background-error-std "$bgstd" \
    --basis-shape 3 2 --basis-width-fraction 0.35 \
    --reference-n 0.025 --n-bounds 0.01 0.08 \
    --max-iterations 15 --max-funcs 15 \
    || echo "  variant $tag FAILED (continuing sweep)"
}

run_variant v1_shift5km_bg10  5000  0.10
run_variant v2_shift15km_bg02 15000 0.02
run_variant v3_shift15km_bg10 15000 0.10
run_variant v4_shift30km_bg10 30000 0.10

echo
echo "=========================================================="
echo "=== SWEEP SUMMARY (aug_pert_sweep_${SLURM_JOB_ID}) ==="
echo "=========================================================="
python3 <<PYEOF
import json, os, glob
base = "$BASE_OUT"
fmt = "%-26s %10s %10s %10s %10s %10s %10s"
print()
print(fmt % ("variant","mode","bg_rmse","an_rmse","fcst_end","iters","theta_n"))
print("-"*92)
for variant_dir in sorted(glob.glob(os.path.join(base, "v*"))):
    s = os.path.join(variant_dir, "summary.json")
    if not os.path.exists(s):
        print(fmt % (os.path.basename(variant_dir),"<missing>","","","","",""))
        continue
    d = json.load(open(s))
    r = d.get("results",{})
    name = os.path.basename(variant_dir)
    for mode in ("state_only","augmented"):
        b = r.get(mode,{})
        if not b: continue
        print(fmt % (
            name, mode,
            f"{b.get('bg_state_rmse',0):.4f}",
            f"{b.get('analysis_state_rmse',0):.4f}",
            f"{b.get('forecast_end_rmse',0):.4f}",
            f"{b.get('iterations',0):.0f}",
            f"{b.get('theta_norm',0):.4f}",
        ))
print()
print("Pass = iters > 1 AND (mode==state_only OR theta_norm > 0.01)")
PYEOF

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

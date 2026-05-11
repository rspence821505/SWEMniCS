#!/bin/bash
# Augmented vs state-only: truth-theta magnitude sweep on Vista.
#
# 697943 established that augmented beats state-only by 51% on
# forecast RMSE when truth_theta = [0.10, -0.05, 0.08, 0.0, -0.03,
# 0.06] (norm 0.153). But that's one point. For the thesis we want
# to show the win is robust across model-error magnitudes.
#
# This sweep scales the truth-theta vector by 0.5x, 1.0x, 2.0x, 3.0x.
# For each magnitude, runs --mode both. Tracks how:
#   - augmented forecast_end_rmse vs state-only forecast_end_rmse
#   - recovered theta_norm tracks truth_theta_norm
#   - augmented win ratio (state-only RMSE / augmented RMSE)
#
# 4 variants × ~5 min each = ~25 min wall, fits Vista gh-dev 2h.
# --------------------------------------------------------
#SBATCH -J aug_mag
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o aug_mag.%j.out
#SBATCH -e aug_mag.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

BASE_OUT="$WORK/SWEMniCS/outputs/aug_truth_magnitude_${SLURM_JOB_ID}"
mkdir -p "$BASE_OUT"

run_variant () {
  local tag="$1"
  shift
  local theta_coeffs="$@"
  local out="$BASE_OUT/$tag"
  mkdir -p "$out"
  echo
  echo "=========================================================="
  echo "=== Variant $tag : truth_theta = $theta_coeffs ==="
  echo "=========================================================="
  mpirun -n 1 --bind-to core --timeout 1500 \
    python -u experiments/idealized_inlet_augmented_serial.py \
    --mode both \
    --coarse-mesh --nx 50 --ny 30 \
    --output-dir "$out" \
    --dt 600 --nt-ramp 24 --nt-da 6 --nt-forecast 6 \
    --vmax 10 --track-shift 0 --track-duration-s 28800 \
    --obs-fraction 0.10 --obs-frequency 3 --obs-noise-level 0.01 \
    --background-error-std 0.10 \
    --basis-shape 3 2 --basis-width-fraction 0.35 \
    --reference-n 0.025 --n-bounds 0.01 0.08 \
    --truth-theta-coefficients $theta_coeffs \
    --max-iterations 15 --max-funcs 15 \
    || echo "  $tag FAILED (continuing)"
}

# Truth-theta magnitudes (base 697943: 0.10 -0.05 0.08 0.0 -0.03 0.06, norm 0.153)
run_variant mag0p5x  0.05  -0.025 0.04  0.0  -0.015 0.03
run_variant mag1p0x  0.10  -0.05  0.08  0.0  -0.03  0.06
run_variant mag2p0x  0.20  -0.10  0.16  0.0  -0.06  0.12
run_variant mag3p0x  0.30  -0.15  0.24  0.0  -0.09  0.18

echo
echo "=========================================================="
echo "=== SWEEP SUMMARY (aug_truth_magnitude_${SLURM_JOB_ID}) ==="
echo "=========================================================="
python3 <<PYEOF
import json, glob, os
import numpy as np
base = "$BASE_OUT"
fmt = "%-12s %-10s %12s %12s %12s %12s %10s"
print()
print(fmt % ("mag","mode","bg_rmse","an_rmse","fcst_end","iters","theta_n"))
print("-"*92)
for tag in ("mag0p5x","mag1p0x","mag2p0x","mag3p0x"):
    s = os.path.join(base, tag, "summary.json")
    if not os.path.exists(s):
        print(fmt % (tag,"<missing>","","","","",""))
        continue
    d = json.load(open(s))
    r = d.get("results",{})
    truth = np.asarray(d['args']['truth_theta_coefficients'], dtype=float)
    truth_norm = np.linalg.norm(truth)
    for mode in ("state_only","augmented"):
        b = r.get(mode,{})
        if not b: continue
        print(fmt % (
            tag, mode,
            f"{b.get('bg_state_rmse',0):.4f}",
            f"{b.get('analysis_state_rmse',0):.4f}",
            f"{b.get('forecast_end_rmse',0):.4f}",
            str(b.get('iterations',0)),
            f"{b.get('theta_norm',0):.4f}",
        ))
    so_f = r.get('state_only',{}).get('forecast_end_rmse',0)
    au_f = r.get('augmented',{}).get('forecast_end_rmse',0)
    if so_f > 0:
        winratio = so_f / au_f if au_f > 0 else float('inf')
        print(f"  truth_norm={truth_norm:.3f}  win ratio (state_only/aug) = {winratio:.2f}x")
PYEOF

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

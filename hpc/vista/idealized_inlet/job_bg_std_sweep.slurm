#!/bin/bash
# bg_std sweep: characterize transition between Manning recovery and
# IC-error-absorption regimes for augmented DA.
#
# Existing data points:
#   - CLAUDE.md §9.7: bg_std → 0 (state forced at truth), 1-param Manning
#     recovery = 94.2%. Augmented recovers truth.
#   - 697943: bg_std=0.10, 6-param Manning truth, recovered theta does NOT
#     match truth pattern but augmented still wins 2x in forecast.
#   - 696832 v1: bg_std=0.10, NO truth-theta, state-only wins.
#
# Hypothesis: state-only DA misattributes unmodeled Manning errors to IC
# corrections, corrupting IC inference. Augmented's theta acts as a
# "parameter sink" that absorbs Manning model error and prevents IC
# corruption. The theta doesn't recover truth, but it keeps IC inference
# clean. That's why augmented wins ~2x even with wrong theta pattern.
#
# This sweep maps the regime transition by varying bg_std at fixed truth.
# Predictions:
#   - At bg_std→0: theta_norm → truth_theta_norm (recovery)
#   - At bg_std=0.10 (current): theta_norm stays at "damping" value ~1.0
#   - At bg_std large: state-only collapses (too much IC error to fix);
#     augmented win ratio explodes
#
# 5 bg_std values × ~5 min each = ~30 min, fits gh-dev 2h.
# --------------------------------------------------------
#SBATCH -J bg_sweep
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o bg_std_sweep.%j.out
#SBATCH -e bg_std_sweep.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

BASE_OUT="$WORK/SWEMniCS/outputs/bg_std_sweep_${SLURM_JOB_ID}"
mkdir -p "$BASE_OUT"

TRUTH_THETA="0.10 -0.05 0.08 0.0 -0.03 0.06"

run_variant () {
  local tag="$1"
  local bg_std="$2"
  local out="$BASE_OUT/$tag"
  mkdir -p "$out"
  echo
  echo "=========================================================="
  echo "=== $tag : bg_std=$bg_std (truth-theta fixed) ==="
  echo "=========================================================="
  mpirun -n 1 --bind-to core --timeout 1500 \
    python -u experiments/idealized_inlet_augmented_serial.py \
    --mode both \
    --coarse-mesh --nx 50 --ny 30 \
    --output-dir "$out" \
    --dt 600 --nt-ramp 24 --nt-da 6 --nt-forecast 6 \
    --vmax 10 --track-shift 0 --track-duration-s 28800 \
    --obs-fraction 0.10 --obs-frequency 3 --obs-noise-level 0.01 \
    --background-error-std "$bg_std" \
    --basis-type kl --kl-n-modes 6 --truth-kl-n-modes 6 \
    --kl-prior-std 0.5 \
    --reference-n 0.025 --n-bounds 0.01 0.08 \
    --truth-theta-coefficients $TRUTH_THETA \
    --max-iterations 15 --max-funcs 15 \
    || echo "  $tag FAILED (continuing)"
}

# Log scan from near-zero IC error to large IC error
run_variant bg0p001  0.001
run_variant bg0p01   0.01
run_variant bg0p03   0.03
run_variant bg0p10   0.10
run_variant bg0p30   0.30

echo
echo "=========================================================="
echo "=== BG_STD SWEEP SUMMARY (bg_std_sweep_${SLURM_JOB_ID}) ==="
echo "=========================================================="
python3 <<PYEOF
import json, os, numpy as np
base = "$BASE_OUT"
truth = np.asarray([0.10, -0.05, 0.08, 0.0, -0.03, 0.06])
truth_norm = float(np.linalg.norm(truth))
fmt = "%-10s %-10s %12s %12s %12s %12s %10s %10s"
print()
print(fmt % ("bg_std","mode","bg_rmse","an_rmse","fcst_end","iters","theta_n","cos(θ,T)"))
print("-"*102)
results = {}
for tag, bg in [("bg0p001",0.001),("bg0p01",0.01),("bg0p03",0.03),
                ("bg0p10",0.10),("bg0p30",0.30)]:
    p = os.path.join(base, tag, "summary.json")
    if not os.path.exists(p):
        print(fmt % (f"{bg:.3f}","<missing>","","","","","",""))
        continue
    d = json.load(open(p))
    r = d["results"]
    for mode_key in ("state_only","augmented"):
        b = r.get(mode_key,{})
        if not b: continue
        if mode_key == "augmented" and b.get("analysis_theta"):
            theta = np.asarray(b["analysis_theta"])
            cos_sim = float(theta.dot(truth) / (np.linalg.norm(theta)*truth_norm + 1e-30))
            cos_str = f"{cos_sim:.3f}"
        else:
            cos_str = "-"
        print(fmt % (
            f"{bg:.3f}", mode_key,
            f"{b.get('bg_state_rmse',0):.4f}",
            f"{b.get('analysis_state_rmse',0):.4f}",
            f"{b.get('forecast_end_rmse',0):.4f}",
            str(b.get('iterations',0)),
            f"{b.get('theta_norm',0):.4f}",
            cos_str,
        ))
    so_f = r.get('state_only',{}).get('forecast_end_rmse',0)
    au_f = r.get('augmented',{}).get('forecast_end_rmse',0)
    results[bg] = (so_f, au_f)

print()
print("=== REGIME TRANSITION ===")
print(f"truth theta_norm = {truth_norm:.4f}")
print(f"{'bg_std':<10} {'state_only':<12} {'augmented':<12} {'win_ratio':<10}")
for bg in sorted(results):
    so, au = results[bg]
    wr = so / au if au > 0 else float('inf')
    print(f"{bg:<10.3f} {so:<12.4f} {au:<12.4f} {wr:<10.2f}x")
PYEOF

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

#!/bin/bash
# KL-basis DOF sweep: switch from Gaussian RBF to Karhunen-Loève
# expansion (canonical Gaussian random field parameterization).
#
# Background: prior DOF ablation (703170) used Gaussian RBF basis,
# showed 6-DOF beats 1-DOF by 1.8x. This sweep redoes the test
# with a KL basis (squared-exponential covariance kernel), the
# more theoretically-defensible parameterization for spatial
# parameter inversion in geosciences.
#
# Truth: fixed at kl_n_modes=6 with the same theta vector as 697943
# (different basis -> different Manning field shape, similar variance).
# DA: sweeps kl_n_modes ∈ {1, 2, 4, 6, 9, 12}.
#
# Pass criteria: replicate the 6-DOF >> 1-DOF result under KL basis,
# confirming spatial information extraction is real and basis-agnostic.
# Identify the modal sweet spot (where additional KL modes stop helping).
#
# 6 runs × ~6 min = ~40 min wall, fits gh-dev 2h easily.
# --------------------------------------------------------
#SBATCH -J kl_dof
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o kl_dof.%j.out
#SBATCH -e kl_dof.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

BASE_OUT="$WORK/SWEMniCS/outputs/kl_dof_sweep_${SLURM_JOB_ID}"
mkdir -p "$BASE_OUT"

# Truth-theta: 6 values matching 697943 magnitudes (different basis,
# similar variance — KL modes are normalized so theta_norm has similar meaning)
TRUTH_THETA="0.10 -0.05 0.08 0.0 -0.03 0.06"
TRUTH_KL_MODES=6

run_variant () {
  local tag="$1"
  local kl_modes="$2"
  local out="$BASE_OUT/$tag"
  mkdir -p "$out"
  echo
  echo "=========================================================="
  echo "=== $tag : DA kl_n_modes=$kl_modes (truth fixed at $TRUTH_KL_MODES) ==="
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
    --basis-type kl \
    --kl-n-modes "$kl_modes" \
    --truth-kl-n-modes "$TRUTH_KL_MODES" \
    --kl-prior-std 0.5 \
    --reference-n 0.025 --n-bounds 0.01 0.08 \
    --truth-theta-coefficients $TRUTH_THETA \
    --max-iterations 15 --max-funcs 15 \
    || echo "  $tag FAILED (continuing)"
}

# DA-side KL mode counts (truth fixed at 6)
run_variant kl1   1
run_variant kl2   2
run_variant kl4   4
run_variant kl6   6
run_variant kl9   9
run_variant kl12  12

echo
echo "=========================================================="
echo "=== KL DOF SWEEP SUMMARY (kl_dof_sweep_${SLURM_JOB_ID}) ==="
echo "=========================================================="
python3 <<PYEOF
import json, os
base = "$BASE_OUT"
fmt = "%-8s %-10s %12s %12s %12s %12s %10s"
print()
print(fmt % ("DA modes","mode","bg_rmse","an_rmse","fcst_end","iters","theta_n"))
print("-"*88)
results = {}
for tag, modes in [("kl1",1),("kl2",2),("kl4",4),("kl6",6),("kl9",9),("kl12",12)]:
    p = os.path.join(base, tag, "summary.json")
    if not os.path.exists(p):
        print(fmt % (str(modes),"<missing>","","","","",""))
        continue
    d = json.load(open(p))
    r = d["results"]
    for mode_key in ("state_only","augmented"):
        b = r.get(mode_key,{})
        if not b: continue
        print(fmt % (
            str(modes), mode_key,
            f"{b.get('bg_state_rmse',0):.4f}",
            f"{b.get('analysis_state_rmse',0):.4f}",
            f"{b.get('forecast_end_rmse',0):.4f}",
            str(b.get('iterations',0)),
            f"{b.get('theta_norm',0):.4f}",
        ))
    so_f = r.get('state_only',{}).get('forecast_end_rmse',0)
    au_f = r.get('augmented',{}).get('forecast_end_rmse',0)
    results[modes] = (so_f, au_f)

print()
print("=== DIMENSIONAL CURVE ===")
print("DA_modes  state-only  augmented  win_ratio")
for m in sorted(results):
    so, au = results[m]
    wr = (so / au) if au > 0 else float('inf')
    print(f"  {m:>2d}      {so:.4f}     {au:.4f}     {wr:.2f}x")
PYEOF

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

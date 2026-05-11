#!/bin/bash
# DOF ablation: is the augmented win from spatial Manning recovery or
# from a global friction-damping knob?
#
# 703135 showed augmented beats state-only by ~2x on forecast RMSE
# across 6x variation in truth-theta magnitude, but recovered theta
# clusters at norm ~1.1 regardless of truth (which varied 0.076-0.459).
# This suggests the optimizer is using theta as a generic friction
# knob, not recovering spatial Manning structure.
#
# This run holds truth FIXED at the 3x2 spatial pattern (truth-theta
# = the 697943 vector) and varies the DA-side basis-shape across:
#   - state_only:   no theta knob (baseline)
#   - 1-DOF basis:  basis-shape (1,1) = single uniform multiplier
#   - 6-DOF basis:  basis-shape (3,2) = full spatial recovery space
#
# If 1-DOF matches 6-DOF: forecast win is global friction damping.
# If 6-DOF substantially beats 1-DOF: real spatial recovery occurs.
#
# All three use the SAME truth Manning field via --truth-basis-shape 3 2.
# 3 runs × ~6 min each = ~20 min wall, fits Vista gh-dev 2h.
# --------------------------------------------------------
#SBATCH -J aug_dof
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o aug_dof.%j.out
#SBATCH -e aug_dof.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

BASE_OUT="$WORK/SWEMniCS/outputs/aug_dof_ablation_${SLURM_JOB_ID}"
mkdir -p "$BASE_OUT"

TRUTH_THETA="0.10 -0.05 0.08 0.0 -0.03 0.06"

run_variant () {
  local tag="$1"
  local mode="$2"
  local da_basis_nx="$3"
  local da_basis_ny="$4"
  local out="$BASE_OUT/$tag"
  mkdir -p "$out"
  echo
  echo "=========================================================="
  echo "=== $tag : mode=$mode DA basis=($da_basis_nx,$da_basis_ny) ==="
  echo "=========================================================="
  mpirun -n 1 --bind-to core --timeout 1500 \
    python -u experiments/idealized_inlet_augmented_serial.py \
    --mode "$mode" \
    --coarse-mesh --nx 50 --ny 30 \
    --output-dir "$out" \
    --dt 600 --nt-ramp 24 --nt-da 6 --nt-forecast 6 \
    --vmax 10 --track-shift 0 --track-duration-s 28800 \
    --obs-fraction 0.10 --obs-frequency 3 --obs-noise-level 0.01 \
    --background-error-std 0.10 \
    --basis-shape "$da_basis_nx" "$da_basis_ny" \
    --basis-width-fraction 0.35 \
    --reference-n 0.025 --n-bounds 0.01 0.08 \
    --truth-basis-shape 3 2 \
    --truth-theta-coefficients $TRUTH_THETA \
    --max-iterations 15 --max-funcs 15 \
    || echo "  $tag FAILED (continuing)"
}

# Three DOF levels against the SAME 6-DOF spatial truth
run_variant state_only  state_only  1  1
run_variant dof1        augmented   1  1
run_variant dof6        augmented   3  2

echo
echo "=========================================================="
echo "=== DOF ABLATION SUMMARY (aug_dof_ablation_${SLURM_JOB_ID}) ==="
echo "=========================================================="
python3 <<PYEOF
import json, os
base = "$BASE_OUT"
fmt = "%-12s %-10s %12s %12s %12s %12s %10s"
print()
print(fmt % ("variant","mode","bg_rmse","an_rmse","fcst_end","iters","theta_n"))
print("-"*92)
def load(tag):
    p = os.path.join(base, tag, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None
for tag in ("state_only","dof1","dof6"):
    d = load(tag)
    if d is None:
        print(fmt % (tag,"<missing>","","","","",""))
        continue
    r = d["results"]
    # state_only variant only writes state_only block; others write both
    for mode_key in ("state_only","augmented"):
        b = r.get(mode_key,{})
        if not b: continue
        print(fmt % (
            tag, mode_key,
            f"{b.get('bg_state_rmse',0):.4f}",
            f"{b.get('analysis_state_rmse',0):.4f}",
            f"{b.get('forecast_end_rmse',0):.4f}",
            str(b.get('iterations',0)),
            f"{b.get('theta_norm',0):.4f}",
        ))

print()
print("=== ABLATION INTERPRETATION ===")
so_d = load("state_only");  so_f = so_d["results"]["state_only"]["forecast_end_rmse"] if so_d else None
d1_d = load("dof1");        d1_f = d1_d["results"]["augmented"]["forecast_end_rmse"] if d1_d else None
d6_d = load("dof6");        d6_f = d6_d["results"]["augmented"]["forecast_end_rmse"] if d6_d else None
if all(x is not None for x in (so_f, d1_f, d6_f)):
    print(f"  state_only forecast_end_rmse: {so_f:.4f}")
    print(f"  1-DOF      forecast_end_rmse: {d1_f:.4f}  (vs state_only: {so_f/d1_f:.2f}x)")
    print(f"  6-DOF      forecast_end_rmse: {d6_f:.4f}  (vs state_only: {so_f/d6_f:.2f}x)")
    print(f"  6-DOF improvement over 1-DOF: {d1_f/d6_f:.3f}x ({100*(d1_f-d6_f)/d1_f:+.1f}%)")
    if d6_f < 0.9 * d1_f:
        print("  VERDICT: 6-DOF substantially beats 1-DOF -> spatial recovery contributes")
    elif d1_f < 1.1 * d6_f:
        print("  VERDICT: 1-DOF matches 6-DOF within 10% -> global friction-damping is the mechanism")
    else:
        print("  VERDICT: partial spatial contribution (between 10-90% of total gain)")
PYEOF

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

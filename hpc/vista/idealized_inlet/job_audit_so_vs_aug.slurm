#!/bin/bash
# AUDIT: are state-only and augmented modes actually comparable?
#
# Vista 703504 (bg_std sweep) showed at bg_std=0.001: iters=0 in both
# modes (no DA happened), yet state_only fcst=0.143 vs augmented
# fcst=0.060 — a 2.4x discrepancy from configurations that should be
# identical when no optimization occurs. This calls into question every
# "augmented beats state-only" claim we've reported.
#
# Test design: with bg_std=0.001 (tiny IC error) and truth_theta=0
# (no Manning model error), state-only and augmented should produce
# identical forecasts. If they don't, there's a structural difference
# between the two modes (likely Manning field construction) that
# confounds the 2x win interpretation.
#
# Variants:
#   A: bg_std=0.001, NO truth-theta (default behavior — bg Manning =
#      truth Manning = whatever IdealizedInlet uses)
#   B: bg_std=0.001, truth-theta=[0,0,0,0,0,0] (explicit zero via KL
#      controller — exercises the controller code path with zero theta)
#
# Pass criteria: state_only fcst_end_rmse ≈ augmented fcst_end_rmse
# (within 1%). Fail: any noticeable difference indicates a bug.
# --------------------------------------------------------
#SBATCH -J audit_so
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o audit_so_vs_aug.%j.out
#SBATCH -e audit_so_vs_aug.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

BASE_OUT="$WORK/SWEMniCS/outputs/audit_so_vs_aug_${SLURM_JOB_ID}"
mkdir -p "$BASE_OUT"

# Variant A: bg_std=0.001, no truth-theta flag (truth Manning = default)
echo
echo "=========================================================="
echo "=== Variant A: bg_std=0.001, NO truth-theta ==="
echo "=========================================================="
mkdir -p "$BASE_OUT/A_no_truth_theta"
mpirun -n 1 --bind-to core --timeout 1500 \
  python -u experiments/idealized_inlet_augmented_serial.py \
  --mode both \
  --coarse-mesh --nx 50 --ny 30 \
  --output-dir "$BASE_OUT/A_no_truth_theta" \
  --dt 600 --nt-ramp 24 --nt-da 6 --nt-forecast 6 \
  --vmax 10 --track-shift 0 --track-duration-s 28800 \
  --obs-fraction 0.10 --obs-frequency 3 --obs-noise-level 0.01 \
  --background-error-std 0.001 \
  --basis-type kl --kl-n-modes 6 \
  --kl-prior-std 0.5 \
  --reference-n 0.025 --n-bounds 0.01 0.08 \
  --max-iterations 15 --max-funcs 15 \
  || echo "  Variant A FAILED"

# Variant B: bg_std=0.001, truth-theta=[0,0,0,0,0,0]
echo
echo "=========================================================="
echo "=== Variant B: bg_std=0.001, truth-theta=zero ==="
echo "=========================================================="
mkdir -p "$BASE_OUT/B_zero_truth_theta"
mpirun -n 1 --bind-to core --timeout 1500 \
  python -u experiments/idealized_inlet_augmented_serial.py \
  --mode both \
  --coarse-mesh --nx 50 --ny 30 \
  --output-dir "$BASE_OUT/B_zero_truth_theta" \
  --dt 600 --nt-ramp 24 --nt-da 6 --nt-forecast 6 \
  --vmax 10 --track-shift 0 --track-duration-s 28800 \
  --obs-fraction 0.10 --obs-frequency 3 --obs-noise-level 0.01 \
  --background-error-std 0.001 \
  --basis-type kl --kl-n-modes 6 --truth-kl-n-modes 6 \
  --kl-prior-std 0.5 \
  --reference-n 0.025 --n-bounds 0.01 0.08 \
  --truth-theta-coefficients 0 0 0 0 0 0 \
  --max-iterations 15 --max-funcs 15 \
  || echo "  Variant B FAILED"

echo
echo "=========================================================="
echo "=== AUDIT VERDICT ==="
echo "=========================================================="
python3 <<PYEOF
import json, os
base = "$BASE_OUT"
fmt = "%-22s %-10s %12s %12s %12s"
print()
print(fmt % ("variant","mode","bg_rmse","fcst_end","iters"))
print("-"*72)
for tag, label in [("A_no_truth_theta","A:no_truth_theta"),
                   ("B_zero_truth_theta","B:zero_truth_theta")]:
    p = os.path.join(base, tag, "summary.json")
    if not os.path.exists(p):
        print(fmt % (label,"<missing>","","",""))
        continue
    d = json.load(open(p))
    r = d["results"]
    so = r.get("state_only",{})
    au = r.get("augmented",{})
    for mk, b in (("state_only",so),("augmented",au)):
        if not b: continue
        print(fmt % (
            label, mk,
            f"{b.get('bg_state_rmse',0):.4f}",
            f"{b.get('forecast_end_rmse',0):.4f}",
            str(b.get('iterations',0)),
        ))
    if so and au:
        so_f = so.get('forecast_end_rmse',0)
        au_f = au.get('forecast_end_rmse',0)
        if so_f > 0:
            diff_pct = 100 * abs(so_f - au_f) / so_f
            print(f"  -> forecast difference: {diff_pct:.1f}% (state_only-augmented)")
            if diff_pct < 1.0:
                print(f"     PASS: modes are equivalent in no-DA limit")
            else:
                print(f"     FAIL: structural difference between modes detected")
    print()
PYEOF

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

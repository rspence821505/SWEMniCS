#!/bin/bash
# KL inverse-crime check: place truth energy in different mode indices.
#
# 703184 found "win ratio ~2.3x across all DA n_modes from 1 to 12".
# Hypothesis: this is an inverse-crime artifact because (a) truth and
# DA share the same KL basis and (b) the truth_theta values lead to
# energy concentrated in mode 1 by KL eigenvalue weighting.
#
# Decisive test: 2x2 grid varying which mode truth lives in vs which
# modes DA can resolve.
#
#   truth_mode1 + DA n_modes=1: WIN (basis aligned)            [control]
#   truth_mode1 + DA n_modes=6: WIN (basis contains truth)     [control]
#   truth_mode6 + DA n_modes=1: ??                             [CRITICAL]
#   truth_mode6 + DA n_modes=6: WIN (basis contains truth)     [control]
#
# CRITICAL CELL: truth_mode6 + DA n_modes=1.
#   - If still ~2.4x win: the augmented mechanism is rank-1 damping,
#     not mode-resolved spatial recovery. The 703184 flat curve was
#     genuine inverse crime.
#   - If win drops to ~1x: spatial mode resolution matters, the
#     1-mode DA cannot reach truth living in higher modes.
#
# Truth theta magnitude held = 0.5 in the active mode (zero elsewhere).
# 4 runs × ~6 min = ~25 min wall, fits gh-dev easily.
# --------------------------------------------------------
#SBATCH -J kl_xcheck
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o kl_xcheck.%j.out
#SBATCH -e kl_xcheck.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

BASE_OUT="$WORK/SWEMniCS/outputs/kl_truth_mode_xcheck_${SLURM_JOB_ID}"
mkdir -p "$BASE_OUT"

run_variant () {
  local tag="$1"
  local truth_theta="$2"
  local da_modes="$3"
  local out="$BASE_OUT/$tag"
  mkdir -p "$out"
  echo
  echo "=========================================================="
  echo "=== $tag : truth=[$truth_theta] DA n_modes=$da_modes ==="
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
    --basis-type kl --kl-n-modes "$da_modes" \
    --truth-kl-n-modes 6 \
    --kl-prior-std 0.5 \
    --reference-n 0.025 --n-bounds 0.01 0.08 \
    --truth-theta-coefficients $truth_theta \
    --max-iterations 15 --max-funcs 15 \
    || echo "  $tag FAILED (continuing)"
}

# 2x2 grid: truth in mode 1 (controls) vs mode 6 (critical), DA at 1 vs 6
run_variant tm1_dm1  "0.5 0 0 0 0 0"   1
run_variant tm1_dm6  "0.5 0 0 0 0 0"   6
run_variant tm6_dm1  "0 0 0 0 0 0.5"   1
run_variant tm6_dm6  "0 0 0 0 0 0.5"   6

echo
echo "=========================================================="
echo "=== KL TRUTH-MODE x DA-MODE CROSS-CHECK SUMMARY ==="
echo "=========================================================="
python3 <<PYEOF
import json, os
base = "$BASE_OUT"
fmt = "%-12s %-10s %12s %12s %12s %12s %10s"
print()
print(fmt % ("variant","mode","bg_rmse","an_rmse","fcst_end","iters","theta_n"))
print("-"*92)
results = {}
for tag in ("tm1_dm1","tm1_dm6","tm6_dm1","tm6_dm6"):
    p = os.path.join(base, tag, "summary.json")
    if not os.path.exists(p):
        print(fmt % (tag,"<missing>","","","","",""))
        continue
    d = json.load(open(p))
    r = d["results"]
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
    so_f = r.get('state_only',{}).get('forecast_end_rmse',0)
    au_f = r.get('augmented',{}).get('forecast_end_rmse',0)
    results[tag] = (so_f, au_f, (so_f/au_f) if au_f > 0 else float('inf'))

print()
print("=== INVERSE CRIME VERDICT ===")
print(f"{'cell':<10} {'truth':<10} {'DA modes':<10} {'win ratio':<12}")
print("-"*48)
for tag, label, modes in [
    ("tm1_dm1","mode1",1),
    ("tm1_dm6","mode1",6),
    ("tm6_dm1","mode6",1),
    ("tm6_dm6","mode6",6)]:
    if tag in results:
        _, _, wr = results[tag]
        print(f"{tag:<10} {label:<10} {modes:<10} {wr:.2f}x")
if all(k in results for k in ("tm6_dm1","tm6_dm6")):
    _,_,critical = results["tm6_dm1"]
    _,_,control = results["tm6_dm6"]
    print()
    print(f"  CRITICAL CELL (tm6_dm1): {critical:.2f}x")
    print(f"  CONTROL (tm6_dm6):       {control:.2f}x")
    if critical >= 0.85 * control:
        print(f"  -> rank-1 damping hypothesis SUPPORTED")
        print(f"     (1-mode DA matches 6-mode within 15% even when truth lives in mode 6)")
    elif critical <= 1.3:
        print(f"  -> spatial mode recovery REAL")
        print(f"     (1-mode DA fails when truth lives in unreachable higher mode)")
    else:
        print(f"  -> partial spatial info (intermediate regime)")
PYEOF

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

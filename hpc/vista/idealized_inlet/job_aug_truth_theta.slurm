#!/bin/bash
# Aug prototype with TRUTH Manning model error (the clean test).
#
# Prior aug runs gave null or non-decisive results because truth
# and bg used identical Manning's-n. With --truth-theta-coefficients
# (added in 88e4aba), the truth's Manning field is set to
# n(theta_truth) via the same Gaussian basis used for DA. The bg
# uses n(theta=0). State-only DA cannot represent this mismatch;
# augmented can recover via theta.
#
# Config:
#   - bg_std=0.10 (proven to activate optimizer in 696832 sweep;
#     bg=0.02 leaves TAO at iter=1 null)
#   - track-shift=0 (sweep showed shift didn't affect outcome)
#   - truth_theta = [0.10, -0.05, 0.08, 0.00, -0.03, 0.06]
#     -> moderate spatial Manning structure (theta_norm ~0.16)
#   - coarse 50x30 mesh, np=1, gh-dev (~5 min)
#
# Pass criteria:
#   - state_only: optimizer activates, but forecast_end_rmse stays
#     elevated (cannot reach truth dynamics)
#   - augmented: theta moves toward truth (theta_norm > 0.10),
#     forecast_end_rmse < state_only forecast_end_rmse
# --------------------------------------------------------
#SBATCH -J aug_truth
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o aug_truth.%j.out
#SBATCH -e aug_truth.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_DIR="$WORK/SWEMniCS/outputs/aug_truth_theta_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

mpirun -n 1 --bind-to core --timeout 2400 \
  python -u experiments/idealized_inlet_augmented_serial.py \
  --mode both \
  --coarse-mesh --nx 50 --ny 30 \
  --output-dir "$OUT_DIR" \
  --dt 600 --nt-ramp 24 --nt-da 6 --nt-forecast 6 \
  --vmax 10 --track-shift 0 --track-duration-s 28800 \
  --obs-fraction 0.10 --obs-frequency 3 --obs-noise-level 0.01 \
  --background-error-std 0.10 \
  --basis-shape 3 2 --basis-width-fraction 0.35 \
  --reference-n 0.025 --n-bounds 0.01 0.08 \
  --truth-theta-coefficients 0.10 -0.05 0.08 0.00 -0.03 0.06 \
  --max-iterations 15 --max-funcs 15
RET=$?

echo
echo "=== Summary JSON ==="
[[ -f "$OUT_DIR/summary.json" ]] && python3 -c "
import json
d = json.load(open('$OUT_DIR/summary.json'))
r = d['results']
print()
print('%-28s %12s %12s' % ('metric','state_only','augmented'))
for k in ('bg_state_rmse','analysis_state_rmse','forecast_end_rmse','elapsed_s','iterations','theta_norm'):
    if k in r.get('state_only',{}) and k in r.get('augmented',{}):
        so = r['state_only'][k]; au = r['augmented'][k]
        print('%-28s %12.4f %12.4f' % (k,so,au))
print()
print('Truth theta:', d['args'].get('truth_theta_coefficients','none'))
if 'augmented' in r:
    af = r['augmented'].get('mannings_field')
    if af:
        print('  Mannings field: min=%.4f mean=%.4f max=%.4f std=%.4e' % (af['min'],af['mean'],af['max'],af.get('std',0)))
    print('  recovered theta =', r['augmented'].get('analysis_theta'))
"

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

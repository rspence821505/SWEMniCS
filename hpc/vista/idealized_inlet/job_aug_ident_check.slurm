#!/bin/bash
# Identifiability check: tighter obs + smaller basis to disambiguate
# the aug_truth_theta result.
#
# 697943 showed augmented beats state-only by 51% on forecast RMSE
# but recovered theta = [0.97, 0.45, 0.11, 0.43, 0.15, -0.005] vs
# truth [0.10, -0.05, 0.08, 0.00, -0.03, 0.06]. The recovered theta
# was 7.5x larger than truth in norm with several wrong signs --
# suggested optimizer using theta as damping knob rather than pattern
# recovery. With 6 params and 10% obs density (~16 sites x 3 obs
# times = 48 scalar measurements for 6 unknowns), the system was
# under-constrained.
#
# This run tightens both:
#   - basis-shape 2 1 -> 2 params (3x fewer DOFs)
#   - obs-fraction 0.30 obs-frequency 1 -> ~50 sites every step
#     (~300 scalar measurements for 2 unknowns -- 150x over-
#     determined)
#   - truth_theta = [0.10, -0.05] (matches first two of prior run)
#
# Pass criteria:
#   - PATTERN-RECOVERY: |recovered - truth| / |truth| < 0.5 elementwise
#     -> confirms 697943 was just under-constrained
#   - DAMPING-ONLY: |recovered| >> |truth| in norm OR signs wrong
#     -> confirms damping interpretation is structural
# --------------------------------------------------------
#SBATCH -J aug_ident
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o aug_ident.%j.out
#SBATCH -e aug_ident.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_DIR="$WORK/SWEMniCS/outputs/aug_ident_check_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

mpirun -n 1 --bind-to core --timeout 2400 \
  python -u experiments/idealized_inlet_augmented_serial.py \
  --mode both \
  --coarse-mesh --nx 50 --ny 30 \
  --output-dir "$OUT_DIR" \
  --dt 600 --nt-ramp 24 --nt-da 6 --nt-forecast 6 \
  --vmax 10 --track-shift 0 --track-duration-s 28800 \
  --obs-fraction 0.30 --obs-frequency 1 --obs-noise-level 0.01 \
  --background-error-std 0.10 \
  --basis-shape 2 1 --basis-width-fraction 0.35 \
  --reference-n 0.025 --n-bounds 0.01 0.08 \
  --truth-theta-coefficients 0.10 -0.05 \
  --max-iterations 15 --max-funcs 15
RET=$?

echo
echo "=== Summary JSON ==="
[[ -f "$OUT_DIR/summary.json" ]] && python3 -c "
import json, numpy as np
d = json.load(open('$OUT_DIR/summary.json'))
r = d['results']
print()
print('%-28s %12s %12s' % ('metric','state_only','augmented'))
for k in ('bg_state_rmse','analysis_state_rmse','forecast_end_rmse','elapsed_s','iterations','theta_norm'):
    if k in r.get('state_only',{}) and k in r.get('augmented',{}):
        so = r['state_only'][k]; au = r['augmented'][k]
        print('%-28s %12.4f %12.4f' % (k,so,au))
print()
truth = np.asarray(d['args']['truth_theta_coefficients'], dtype=float)
recovered = np.asarray(r['augmented'].get('analysis_theta',[]), dtype=float)
print('Truth theta:    ', truth)
print('Recovered theta:', recovered)
if len(truth) == len(recovered) and len(truth) > 0:
    err = recovered - truth
    rel = np.abs(err) / np.maximum(np.abs(truth), 1e-12)
    print('Elementwise abs err:', err)
    print('Elementwise rel err:', rel)
    print('truth norm    : %.4f' % np.linalg.norm(truth))
    print('recovered norm: %.4f' % np.linalg.norm(recovered))
    print('rel norm err  : %.4f' % (np.linalg.norm(err)/max(np.linalg.norm(truth),1e-12)))
    if np.all(rel < 0.5):
        print('VERDICT: PATTERN-RECOVERY (rel err < 0.5 elementwise)')
    else:
        print('VERDICT: DAMPING-ONLY or partial (some rel err >= 0.5)')
"

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

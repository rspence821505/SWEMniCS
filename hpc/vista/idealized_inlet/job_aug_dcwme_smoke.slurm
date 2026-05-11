#!/bin/bash
# Augmented DC-WME 4D-Var smoke test (FIRST empirical validation).
#
# Per CLAUDE.md: AugmentedDCWMEFourDVarCost is a `pass` subclass that
# delegates everything to DCWMEFourDVarCost. The assumption is that
# the WME QoI linearization naturally handles packed controls. Never
# tested.
#
# This single-run smoke test wires --cost-method dc_wme through the
# augmented prototype. Expected outcomes:
#   (a) Runs cleanly -> first augmented DC-WME data point. Compare
#       forecast_end_rmse against augmented 4D-Var (703184 kl_n_modes=6
#       baseline).
#   (b) Crashes during Step 7a Eq 38 setup -> probably needs replay-
#       meta env var or similar plumbing. Document and fix.
#   (c) Crashes during cost evaluation -> the WME QoI doesn't actually
#       handle packed controls. Need code work to fix.
#
# Single window, coarse 50x30 mesh, np=1. KL basis matches 703184
# (kl_n_modes=6 both truth and DA) for direct comparison.
# --------------------------------------------------------
#SBATCH -J augdcsmk
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o aug_dcwme_smoke.%j.out
#SBATCH -e aug_dcwme_smoke.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_DIR="$WORK/SWEMniCS/outputs/aug_dcwme_smoke_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

mpirun -n 1 --bind-to core --timeout 6000 \
  python -u experiments/idealized_inlet_augmented_serial.py \
  --mode both \
  --coarse-mesh --nx 50 --ny 30 \
  --output-dir "$OUT_DIR" \
  --dt 600 --nt-ramp 24 --nt-da 6 --nt-forecast 6 \
  --vmax 10 --track-shift 0 --track-duration-s 28800 \
  --obs-fraction 0.10 --obs-frequency 3 --obs-noise-level 0.01 \
  --background-error-std 0.10 \
  --basis-type kl --kl-n-modes 6 --truth-kl-n-modes 6 \
  --kl-prior-std 0.5 \
  --reference-n 0.025 --n-bounds 0.01 0.08 \
  --truth-theta-coefficients 0.10 -0.05 0.08 0.0 -0.03 0.06 \
  --cost-method dc_wme \
  --dcwme-predictability-gamma 0.1 \
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
print('Cost method:', d['args'].get('cost_method','?'))
print('Basis type:', d['args'].get('basis_type','?'))
"

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

#!/bin/bash
# Run the augmented-control serial prototype on a COARSE generated mesh.
# The full Ideal_Inlet xdmf (~207k state DOFs) is too expensive for serial
# GMRES+ILU; the coarse rectangular mesh (~5k DOFs at default 50x30 cells)
# reproduces the same physical domain + boundary/bathymetry/friction logic
# but lets augmented mode actually complete in reasonable wall time.
# --------------------------------------------------------
#SBATCH -J aug_coarse
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o aug_coarse.%j.out
#SBATCH -e aug_coarse.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_DIR="$WORK/SWEMniCS/outputs/idealized_inlet_augmented_serial_coarse"
mkdir -p "$OUT_DIR"

mpirun -n 1 --bind-to core --timeout 7000 \
  python -u experiments/idealized_inlet_augmented_serial.py \
  --mode both \
  --coarse-mesh --nx 50 --ny 30 \
  --output-dir "$OUT_DIR" \
  --dt 600 --nt-ramp 24 --nt-da 6 --nt-forecast 6 \
  --vmax 10 --track-shift 0 --track-duration-s 28800 \
  --obs-fraction 0.10 --obs-frequency 3 --obs-noise-level 0.01 \
  --background-error-std 0.02 \
  --basis-shape 3 2 --basis-width-fraction 0.35 \
  --reference-n 0.025 --n-bounds 0.01 0.08 \
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
if 'augmented' in r:
    af = r['augmented'].get('mannings_field')
    if af:
        print('  Manning field: min=%.4f mean=%.4f max=%.4f' % (af['min'],af['mean'],af['max']))
    print('  theta =', r['augmented'].get('analysis_theta'))
"

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

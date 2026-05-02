#!/bin/bash
# Run the augmented-control serial prototype on Vista gh-dev.
# Single rank (np=1) — augmented controls are serial-only.
# Tests state_only vs augmented modes with forecast-after-analysis comparison.
# --------------------------------------------------------
#SBATCH -J aug_proto
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o aug_proto.%j.out
#SBATCH -e aug_proto.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_DIR="$WORK/SWEMniCS/outputs/idealized_inlet_augmented_serial"
mkdir -p "$OUT_DIR"

# Single-rank only. mpirun with np=1 to ensure serial-only paths fire.
mpirun -n 1 --bind-to core --timeout 3500 \
  python -u experiments/idealized_inlet_augmented_serial.py \
  --mode both \
  --output-dir "$OUT_DIR" \
  --dt 600 --nt-ramp 24 --nt-da 6 --nt-forecast 6 \
  --vmax 10 --track-shift 0 --track-duration-s 28800 \
  --obs-fraction 0.10 --obs-frequency 3 --obs-noise-level 0.01 \
  --background-error-std 0.02 \
  --basis-shape 3 2 --basis-width-fraction 0.35 \
  --reference-n 0.025 --n-bounds 0.01 0.08 \
  --max-iterations 10 --max-funcs 10
RET=$?

echo
echo "=== Summary JSON ==="
[[ -f "$OUT_DIR/summary.json" ]] && python3 -c "
import json
d = json.load(open('$OUT_DIR/summary.json'))
r = d['results']
print()
print(f\"{'metric':<28} {'state_only':>12} {'augmented':>12}\")
for k in ('bg_state_rmse','analysis_state_rmse','forecast_end_rmse','elapsed_s','iterations','theta_norm'):
    if k in r.get('state_only',{}) and k in r.get('augmented',{}):
        so = r['state_only'][k]
        au = r['augmented'][k]
        print(f\"{k:<28} {so:>12.4f} {au:>12.4f}\")
print()
if 'augmented' in r:
    af = r['augmented'].get('mannings_field')
    if af:
        print(f\"  Manning field (augmented): min={af['min']:.4f}  mean={af['mean']:.4f}  max={af['max']:.4f}\")
    print(f\"  theta = {r['augmented'].get('analysis_theta')}\")
"

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

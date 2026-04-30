#!/bin/bash
# Vista gh-dev: alpha-probe sweep on cycling_dump_w2 (window-3 bg).
# Forecast IC = truth_ic + alpha*(bg - truth_ic) for alpha in
# {0.0, 0.25, 0.5, 0.7, 0.85, 0.95, 1.0}. Holds spatial error structure
# constant; varies magnitude. Identifies the forward-basin threshold for
# this regime (peak hurricane, vmax=20, no model error).
# Each probe: 6 timesteps, ~30-60s wallclock. Total ~5 min.
# --------------------------------------------------------
#SBATCH -J alpha_w2_gg
#SBATCH -A ADCIRC
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 00:30:00
#SBATCH -o alpha_w2_gg.%j.out
#SBATCH -e alpha_w2_gg.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

DUMP_DIR="$WORK/SWEMniCS/results/idealized_inlet_da/cycling_dump_w2"
OUT_DIR="$WORK/SWEMniCS/results/idealized_inlet_da/alpha_sweep_w2"
mkdir -p "$OUT_DIR"

for ALPHA in 0.0 0.25 0.5 0.7 0.85 0.95 1.0; do
  TAG=$(printf "alpha_%s" "${ALPHA/./p}")  # 0.7 → 0p7
  echo
  echo "============================================================"
  echo "  ALPHA-PROBE  alpha=${ALPHA}  → ${TAG}"
  echo "============================================================"
  mpirun -n 8 --bind-to core --map-by socket --timeout 600 \
    python -u experiments/idealized_inlet_forecast_probe.py \
    --dump-dir "$DUMP_DIR" \
    --alpha "$ALPHA" \
    --nt-forecast 6 \
    --csv-out "$OUT_DIR/${TAG}.csv" \
    || echo "  [WARN] alpha=${ALPHA} returned non-zero, continuing"
done

echo
echo "=== ALL PROBES DONE ==="
echo "Results in: $OUT_DIR/"
ls -la "$OUT_DIR/"

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

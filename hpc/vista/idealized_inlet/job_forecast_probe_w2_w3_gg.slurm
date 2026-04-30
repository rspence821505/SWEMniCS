#!/bin/bash
# Forecast-sensitivity probe on cycling-DA window-2 and window-3 dumps
# from run B (682192). For each dump, load bg + truth_traj, run forward
# without DA, log per-step RMSE / min_h / max|u| / max|v| / Newton iters.
# CSVs land at <dump_dir>/forecast_probe.csv.
# --------------------------------------------------------
#SBATCH -J fcprobe_w2w3_gg
#SBATCH -A ADCIRC
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 00:30:00
#SBATCH -o fcprobe_w2w3_gg.%j.out
#SBATCH -e fcprobe_w2w3_gg.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

DUMP_BASE="$WORK/SWEMniCS/results/idealized_inlet_da"

for tag in w2 w3; do
  echo
  echo "============================================================"
  echo "  FORECAST PROBE on cycling_dump_${tag}/"
  echo "============================================================"
  mpirun -n 8 --bind-to core --map-by socket --timeout 1800 \
    python -u experiments/idealized_inlet_forecast_probe.py \
    --dump-dir "$DUMP_BASE/cycling_dump_${tag}" \
    --nt-forecast 6 \
    --csv-out "$DUMP_BASE/cycling_dump_${tag}/forecast_probe.csv" \
    || echo "  [WARN] probe ${tag} returned non-zero, continuing"
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

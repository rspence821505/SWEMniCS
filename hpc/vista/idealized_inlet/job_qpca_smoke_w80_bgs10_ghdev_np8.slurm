#!/bin/bash
# Vista gh-dev: QPCA-EnDCF smoke test, 80-min DA window, with localization
# (5 km Gaspari-Cohn) and background_error_std=0.10 to give the filter real
# signal. Tests:
#   - N=10 ensemble, κ=1, L=4 obs times, obs_frequency=2 (every 20 min)
#   - 500 obs/time at σ_obs=0.01 m
#   - 5 km GC localization on the cross-covariance gain
#   - Eigenvalue diagnostics logged in JSON
# --------------------------------------------------------
#SBATCH -J qpca_w80_bgs10
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o qpca_w80_bgs10.%j.out
#SBATCH -e qpca_w80_bgs10.%j.err

set -euxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

# Output subdir for this experiment — keeps results separated from other runs.
OUT_SUBDIR="idealized_inlet_qpca_endcf_vista_w80_bgs10"
mkdir -p "$WORK/SWEMniCS/results/$OUT_SUBDIR"

mpirun -n 8 --bind-to core --map-by socket --timeout 7000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke \
  --output-subdir "$OUT_SUBDIR"

echo
echo "=== result JSON ==="
ls -la "$WORK/SWEMniCS/results/$OUT_SUBDIR/result_qpca_endcf.json" || true
cat "$WORK/SWEMniCS/results/$OUT_SUBDIR/result_qpca_endcf.json" || true

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

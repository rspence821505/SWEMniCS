#!/bin/bash
# Vista gh-dev: QPCA-EnDCF smoke, κ=1 with localization DISABLED.
# Compares against the localized run 733575 to quantify how much the
# 5 km GC localization buys us at this bg_error_std=0.10 config — i.e.
# the actual cost of the rank-9 spurious cross-covariances we localized
# away in 733575.
# --------------------------------------------------------
#SBATCH -J qpca_w80_bgs10_noloc
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o qpca_w80_bgs10_noloc.%j.out
#SBATCH -e qpca_w80_bgs10_noloc.%j.err

set -euxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_SUBDIR="idealized_inlet_qpca_endcf_vista_w80_bgs10_noloc"
mkdir -p "$WORK/SWEMniCS/results/$OUT_SUBDIR"

mpirun -n 8 --bind-to core --map-by socket --timeout 7000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke \
  --override-loc-radius 0 \
  --output-subdir "$OUT_SUBDIR"

echo
echo "=== result JSON ==="
ls -la "$WORK/SWEMniCS/results/$OUT_SUBDIR/result_qpca_endcf.json" || true
cat "$WORK/SWEMniCS/results/$OUT_SUBDIR/result_qpca_endcf.json" || true

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

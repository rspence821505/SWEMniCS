#!/bin/bash
# LS6 dev: QPCA-EnDCF smoke, 4-window cycling at bg_error_std=0.10.
# Cycling tests whether the small per-window state improvement (+0.10 %
# in Vista run 733575) accumulates into a stable / improving filter
# over multiple consecutive 80-min DA windows, or whether spread
# collapses and the filter diverges.
#
# Each window: forecast ensemble through 80 min → localized QPCA-EnDCF
# analysis → use posterior as prior for next window.
# Total forecast cost: 4 windows × 10 members × 8 timesteps = 320 Newton
# solves. At ~5 s/solve on np=8 LS6 expect ~27 min wall time.
# --------------------------------------------------------
#SBATCH -J qpca_w80_bgs10_cyc4
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o qpca_w80_bgs10_cyc4.%j.out
#SBATCH -e qpca_w80_bgs10_cyc4.%j.err

set -uxo pipefail
shopt -s nullglob
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

OUT_SUBDIR="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4"
mkdir -p "$WORK/SWEMniCS/results/$OUT_SUBDIR"

ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke \
  --override-n-windows 4 \
  --output-subdir "$OUT_SUBDIR"

echo
echo "=== result JSON ==="
ls -la "$WORK/SWEMniCS/results/$OUT_SUBDIR/result_qpca_endcf.json" || true
cat "$WORK/SWEMniCS/results/$OUT_SUBDIR/result_qpca_endcf.json" || true

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

#!/bin/bash
# LS6 dev: BUNDLE — head-to-head comparison QPCA-EnDCF vs stochastic
# 4D-EnKF, both with 5 km Gaspari-Cohn localization and multiplicative
# inflation λ=1.05, 4-window cycling at the current best smoke config.
#
# Same forecasts, same observations, same R, same Gaspari-Cohn taper,
# same inflation, same cycling — only the analysis-update formula
# differs:
#   QPCA-EnDCF: whitened residuals + rank-κ spectral truncation + R-free
#               pseudoinverse gain, deterministic.
#   4D-EnKF   : standard R-stabilized Kalman gain + perturbed observations.
#
# This isolates the QPCA-EnDCF *contribution* (spectral truncation +
# deterministic update) from the localization+inflation machinery that
# every modern operational EnKF uses. If QPCA-EnDCF doesn't win this
# head-to-head, the spectral-truncation claim doesn't survive transfer
# from Lorenz-96 to the SWE PDE state.
#
# Per-run cost: 4 windows × 80 Newton solves on np=8 ≈ 26 min.
# Total bundle: ~52 min wall (well under 2 h dev limit).
# --------------------------------------------------------
#SBATCH -J qpca_vs_enkf4d_cyc4
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o qpca_vs_enkf4d_cyc4.%j.out
#SBATCH -e qpca_vs_enkf4d_cyc4.%j.err

set -uxo pipefail        # NOT -e: one failure should not abort the other
shopt -s nullglob
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

OUT_QPCA="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl105_qpca"
OUT_ENKF="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl105_enkf4d"
mkdir -p "$WORK/SWEMniCS/results/$OUT_QPCA" \
         "$WORK/SWEMniCS/results/$OUT_ENKF"

echo
echo "=== 1/2: QPCA-EnDCF (κ=1) + 5km loc + λ=1.05 + cycling 4w ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --output-subdir "$OUT_QPCA" \
  || echo "[bundle] QPCA-EnDCF exited non-zero"

echo
echo "=== 2/2: stochastic 4D-EnKF + 5km loc + λ=1.05 + cycling 4w ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method enkf4d --override-n-windows 4 --inflation 1.05 \
  --output-subdir "$OUT_ENKF" \
  || echo "[bundle] 4D-EnKF exited non-zero"

echo
echo "=== comparison results ==="
for SUB in "$OUT_QPCA" "$OUT_ENKF"; do
  echo "--- $SUB ---"
  ls -la "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  cat "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  echo
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

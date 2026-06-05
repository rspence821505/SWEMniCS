#!/bin/bash
# LS6 dev: BUNDLE — adaptive-κ vs fixed-κ head-to-head, QPCA-EnDCF.
# All three runs share 5 km Gaspari-Cohn localization + λ_infl=1.05 +
# 4-window cycling at the current best smoke config.
#
#   Fixed κ=1               control — paper baseline, κ stays at 1 every
#                           window even when the W1 whitened-residual
#                           spectrum is broad (top-1/trace ≈ 32%).
#   Adaptive κ, τ=0.85      pick smallest κ s.t. cumvar/trace ≥ 0.85.
#                           From the cycling-4 spectrum data we expect
#                           κ≈3-4 in W1 (broad), κ=1 in W2+ (post-collapse).
#   Adaptive κ, τ=0.95      same policy, more aggressive — likely κ≈5 in W1.
#
# The decisive read is whether adaptive κ (a) keeps a larger projector in
# the broad W1 spectrum and (b) collapses back to κ=1 once the spectrum
# spikes, beating the fixed control on mean state Δ% across the four
# windows. If τ=0.85 wins but τ=0.95 loses, projector-estimation
# variance dominated and we've localized the sweet spot.
#
# Per-run cost: 4 windows × 80 Newton solves on np=8 ≈ 26 min.
# Total bundle: ~80 min wall (well under 2 h dev limit).
# --------------------------------------------------------
#SBATCH -J qpca_adapk_cyc4
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o qpca_adapk_cyc4.%j.out
#SBATCH -e qpca_adapk_cyc4.%j.err

set -uxo pipefail        # NOT -e: one failure should not abort the others
shopt -s nullglob
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

OUT_FIXED="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl105_kfixed1"
OUT_AD085="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl105_kadap085"
OUT_AD095="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl105_kadap095"
mkdir -p "$WORK/SWEMniCS/results/$OUT_FIXED" \
         "$WORK/SWEMniCS/results/$OUT_AD085" \
         "$WORK/SWEMniCS/results/$OUT_AD095"

echo
echo "=== 1/3: QPCA-EnDCF, FIXED κ=1 (control) + 5km loc + λ=1.05 + cyc 4w ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --output-subdir "$OUT_FIXED" \
  || echo "[bundle] fixed κ=1 exited non-zero"

echo
echo "=== 2/3: QPCA-EnDCF, ADAPTIVE κ (τ=0.85) + 5km loc + λ=1.05 + cyc 4w ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --kappa-target 0.85 \
  --output-subdir "$OUT_AD085" \
  || echo "[bundle] adaptive κ τ=0.85 exited non-zero"

echo
echo "=== 3/3: QPCA-EnDCF, ADAPTIVE κ (τ=0.95) + 5km loc + λ=1.05 + cyc 4w ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --kappa-target 0.95 \
  --output-subdir "$OUT_AD095" \
  || echo "[bundle] adaptive κ τ=0.95 exited non-zero"

echo
echo "=== comparison results ==="
for SUB in "$OUT_FIXED" "$OUT_AD085" "$OUT_AD095"; do
  echo "--- $SUB ---"
  ls -la "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  cat "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  echo
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

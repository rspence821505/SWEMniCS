#!/bin/bash
# LS6 dev: BUNDLE of two QPCA-EnDCF smoke experiments back-to-back in one
# sbatch — pays the queue wait once for two experiments. Replaces the
# original Vista jobs 733616 (κ=2) and 733617 (no-localization) which
# were canceled. Each experiment is ~7 min at np=8, total ~16 min wall
# with overheads.
#
# 1. κ=2 (vs baseline κ=1 in Vista run 733575): tests if mode 2 of the
#    whitened residual covariance carries real signal as the spectrum
#    diagnostic in 733575 suggested (λ_2 = 75% of λ_1).
# 2. no-localization: paper-exact filter, no Gaspari-Cohn taper on the
#    gain. Quantifies what the 5 km localization in 733575 buys at this
#    bg_error_std=0.10 signal level.
# --------------------------------------------------------
#SBATCH -J qpca_w80_k2_noloc_bundle
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -o qpca_w80_k2_noloc_bundle.%j.out
#SBATCH -e qpca_w80_k2_noloc_bundle.%j.err

set -uxo pipefail        # NOT -e: one experiment failing should not abort the next
shopt -s nullglob
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

OUT_K2="idealized_inlet_qpca_endcf_ls6_w80_bgs10_k2"
OUT_NOLOC="idealized_inlet_qpca_endcf_ls6_w80_bgs10_noloc"
mkdir -p "$WORK/SWEMniCS/results/$OUT_K2" "$WORK/SWEMniCS/results/$OUT_NOLOC"

echo
echo "=== Experiment 1/2: κ=2 ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-k 2 --output-subdir "$OUT_K2" \
  || echo "[bundle] κ=2 experiment exited non-zero"

echo
echo "=== Experiment 2/2: no-localization ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-loc-radius 0 --output-subdir "$OUT_NOLOC" \
  || echo "[bundle] no-loc experiment exited non-zero"

echo
echo "=== κ=2 result ==="
ls -la "$WORK/SWEMniCS/results/$OUT_K2/result_qpca_endcf.json" 2>&1 || true
cat "$WORK/SWEMniCS/results/$OUT_K2/result_qpca_endcf.json" 2>&1 || true

echo
echo "=== no-loc result ==="
ls -la "$WORK/SWEMniCS/results/$OUT_NOLOC/result_qpca_endcf.json" 2>&1 || true
cat "$WORK/SWEMniCS/results/$OUT_NOLOC/result_qpca_endcf.json" 2>&1 || true

echo
echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

#!/bin/bash
# Vista gh-dev: BUNDLE — QPCA-EnDCF vs 4D LETKF *without localization*.
#
# Control experiment for the QPCA-vs-LETKF head-to-head (Vista job
# 737361 / chapter Table 1 LETKF column). The chapter's narrative now
# argues that QPCA's 8× end-of-cycling advantage over LETKF comes from
# spectral truncation. The natural counter-argument: QPCA uses
# B-localization (Schur on P_xy) while LETKF uses R-localization
# (per-DOF observation patches with weighted R^{-1}), so part of the
# 8× could be attributable to which localization scheme is better at
# this operating point, not to the analysis form itself.
#
# To isolate the analysis-form contribution from the localization-
# scheme contribution, this bundle re-runs both filters with
# localization *disabled* (--override-loc-radius 0). If QPCA still
# beats LETKF by a similar margin, the spectral truncation is the
# dominant factor regardless of localization. If the gap shrinks,
# localization scheme is doing real work and the chapter's mechanism
# claim needs to be qualified.
#
# Same operating point as 737361 otherwise:
#   --smoke baseline (vmax=15, track_shift=5 km, dt=600 s, nt_ramp=6,
#                     window_len=4, obs_frequency=2 → 80-min window,
#                     obs_fraction=0.043 → ~500 obs/time,
#                     bg_error_std=0.10, seed=2026).
#   cycling-4, λ_infl=1.05; QPCA uses fixed κ=1.
#
# Per-run cost: cycling-4 at N=10 without sparse Schur localization
# ≈ same as the localized baseline (~21 min on Vista) because the
# Schur multiply is cheap compared to the forecast. Bundle ≈ 45 min,
# fits the gh-dev 1 h cap.
# --------------------------------------------------------
#SBATCH -J qpca_vs_letkf_noloc
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -o qpca_vs_letkf_noloc.%j.out
#SBATCH -e qpca_vs_letkf_noloc.%j.err

set -uxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_QPCA="idealized_inlet_qpca_endcf_vista_w80_bgs10_cyc4_infl105_qpca_noloc"
OUT_LETKF="idealized_inlet_qpca_endcf_vista_w80_bgs10_cyc4_infl105_letkf_noloc"
mkdir -p "$WORK/SWEMniCS/results/$OUT_QPCA" \
         "$WORK/SWEMniCS/results/$OUT_LETKF"

echo
echo "=== 1/2: QPCA-EnDCF (κ=1) NO LOCALIZATION + λ=1.05 + cyc 4w (seed 2026) ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 3000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --override-loc-radius 0 \
  --ensemble-seed 2026 \
  --output-subdir "$OUT_QPCA" \
  || echo "[bundle] QPCA-EnDCF no-loc exited non-zero"

echo
echo "=== 2/2: 4D LETKF NO LOCALIZATION + λ=1.05 + cyc 4w (seed 2026) ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 3000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method letkf --override-n-windows 4 --inflation 1.05 \
  --override-loc-radius 0 \
  --ensemble-seed 2026 \
  --output-subdir "$OUT_LETKF" \
  || echo "[bundle] LETKF no-loc exited non-zero"

echo
echo "=== comparison results ==="
for SUB in "$OUT_QPCA" "$OUT_LETKF"; do
  echo "--- $SUB ---"
  ls -la "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  cat "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  echo
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

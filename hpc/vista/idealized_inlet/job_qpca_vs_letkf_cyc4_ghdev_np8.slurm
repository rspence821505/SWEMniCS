#!/bin/bash
# Vista gh-dev: BUNDLE — QPCA-EnDCF vs 4D LETKF head-to-head at the
# chapter's Section 4 cycling-4 operating point.
#
# Companion to the existing three-way comparison
# (QPCA vs 4D-EnKF vs SeqEnKF, LS6 jobs 3203539 + 3205065 — chapter
# Table 1). LETKF is the deterministic ensemble-square-root variant
# (Hunt, Kostelich, Szunyogh 2007), expected to occupy a *third*
# qualitative regime relative to the stochastic baselines and QPCA:
# deterministic (no perturbed obs), full-rank in ensemble space (no
# spectral truncation), with R-localization rather than B-
# localization. The interesting question is whether the mean-
# preserving square-root transform avoids the analysis-induced
# ensemble degradation that destabilizes 4D-EnKF and SeqEnKF, and
# whether the resulting trajectory tracks QPCA-EnDCF or falls between
# QPCA and the stochastic baselines.
#
# Shared config (matches chapter Table 1):
#   --smoke baseline (vmax=15, track_shift=5 km, dt=600 s, nt_ramp=6,
#                     window_len=4, obs_frequency=2 → 80-min window,
#                     obs_fraction=0.043 → ~500 obs/time,
#                     bg_error_std=0.10, seed=2026, 5 km loc).
#   cycling-4, λ_infl=1.05; QPCA uses fixed κ=1; LETKF uses the
#   sparse Schur taper as R-localization (per-state-DOF observation
#   patches).
#
# Per-run cost on Vista cycling-4 at N=10: ~21 min QPCA, ~22-25 min
# LETKF (LETKF per-DOF analysis loop adds a few seconds per window
# of overhead). Bundle ≈ 45-50 min, fits gh-dev 1 h cap.
# --------------------------------------------------------
#SBATCH -J qpca_vs_letkf_cyc4
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -o qpca_vs_letkf_cyc4.%j.out
#SBATCH -e qpca_vs_letkf_cyc4.%j.err

set -uxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_QPCA="idealized_inlet_qpca_endcf_vista_w80_bgs10_cyc4_infl105_qpca_letkf_bundle"
OUT_LETKF="idealized_inlet_qpca_endcf_vista_w80_bgs10_cyc4_infl105_letkf"
mkdir -p "$WORK/SWEMniCS/results/$OUT_QPCA" \
         "$WORK/SWEMniCS/results/$OUT_LETKF"

echo
echo "=== 1/2: QPCA-EnDCF (κ=1) + 5km loc + λ=1.05 + cyc 4w (seed 2026) ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 3000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --ensemble-seed 2026 \
  --output-subdir "$OUT_QPCA" \
  || echo "[bundle] QPCA-EnDCF exited non-zero"

echo
echo "=== 2/2: 4D LETKF + 5km R-loc + λ=1.05 + cyc 4w (seed 2026) ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 3000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method letkf --override-n-windows 4 --inflation 1.05 \
  --ensemble-seed 2026 \
  --output-subdir "$OUT_LETKF" \
  || echo "[bundle] LETKF exited non-zero"

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

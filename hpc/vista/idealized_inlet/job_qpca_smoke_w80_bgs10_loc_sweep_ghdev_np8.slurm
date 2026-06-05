#!/bin/bash
# Vista gh-dev: BUNDLE — localization-radius sweep at the current best
# QPCA-EnDCF smoke config. 4 runs back-to-back, ~7 min each → ~30 min
# wall in one queue wait.
#
# All four runs share: κ=1, bg_error_std=0.10, vmax=15, track_shift=5,
# obs_fraction=0.043 (~500 obs), σ_obs=0.01, window_len=4 obs_freq=2
# (80-min DA window), N=10, ensemble_seed=2026.
# Only --override-loc-radius differs.
#
# Sweep choices (in meters):
#   1000  — extreme-tight: most state DOFs have 0-1 obs in range (informative
#           breaking-point case where localization becomes useless)
#   3000  — tight: ~4 obs/DOF average; localized but enough information
#   5000  — baseline from Vista run 733575 (~38 obs/DOF)
#  10000  — loose: ~150 obs/DOF; localized but admits spurious mid-range
#           correlations
# --------------------------------------------------------
#SBATCH -J qpca_w80_loc_sweep
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -o qpca_w80_loc_sweep.%j.out
#SBATCH -e qpca_w80_loc_sweep.%j.err

set -uxo pipefail        # NOT -e: one failure should not abort the others
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_1KM="idealized_inlet_qpca_endcf_vista_w80_bgs10_loc1km"
OUT_3KM="idealized_inlet_qpca_endcf_vista_w80_bgs10_loc3km"
OUT_5KM="idealized_inlet_qpca_endcf_vista_w80_bgs10_loc5km"
OUT_10KM="idealized_inlet_qpca_endcf_vista_w80_bgs10_loc10km"
mkdir -p "$WORK/SWEMniCS/results/$OUT_1KM" \
         "$WORK/SWEMniCS/results/$OUT_3KM" \
         "$WORK/SWEMniCS/results/$OUT_5KM" \
         "$WORK/SWEMniCS/results/$OUT_10KM"

echo
echo "=== 1/4: L_loc = 1000 m ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 7000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-loc-radius 1000 --output-subdir "$OUT_1KM" \
  || echo "[bundle] L_loc=1km exited non-zero"

echo
echo "=== 2/4: L_loc = 3000 m ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 7000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-loc-radius 3000 --output-subdir "$OUT_3KM" \
  || echo "[bundle] L_loc=3km exited non-zero"

echo
echo "=== 3/4: L_loc = 5000 m ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 7000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-loc-radius 5000 --output-subdir "$OUT_5KM" \
  || echo "[bundle] L_loc=5km exited non-zero"

echo
echo "=== 4/4: L_loc = 10000 m ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 7000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-loc-radius 10000 --output-subdir "$OUT_10KM" \
  || echo "[bundle] L_loc=10km exited non-zero"

echo
echo "=== sweep results ==="
for SUB in "$OUT_1KM" "$OUT_3KM" "$OUT_5KM" "$OUT_10KM"; do
  echo "--- $SUB ---"
  ls -la "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  cat "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  echo
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

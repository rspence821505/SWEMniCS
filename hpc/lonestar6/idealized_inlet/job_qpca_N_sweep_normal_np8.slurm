#!/bin/bash
# LS6 normal: BUNDLE — ensemble-size sweep at the loc-sweep operating
# point. Adds N=15 to the N ∈ {10, 25, 50} Vista bundle (734016) so we
# have a denser 4-point slope. All four runs share fixed κ=1, 5 km
# Gaspari-Cohn localization, λ_infl=1.0, single-window smoke config
# (vmax=15, track_shift=5, obs_fraction=0.043, σ_obs=0.01, window_len=4,
# obs_freq=2, bg_error_std=0.10, seed=2026).
#
# Only --override-ensemble-size differs:
#   N = 10  baseline
#   N = 15  added vs Vista bundle — densifies the slope around the noise
#           floor we suspect lives between 10 and 25
#   N = 25  mid-range — projector estimation variance should drop ~2.5×
#   N = 50  upper end — variance should drop ~5×, exposes any
#           N-independent obstacle to the filter
#
# Per-run cost (single-window) ≈ 7 min × (N/10):
#   N=10 → ~7,  N=15 → ~10,  N=25 → ~17,  N=50 → ~35.
# Total ≈ 70 min wall. The 3 h request leaves comfortable margin for
# slower-than-expected member forecasts and queue overhead.
# --------------------------------------------------------
#SBATCH -J qpca_w80_N_sweep_norm
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 03:00:00
#SBATCH -o qpca_w80_N_sweep_norm.%j.out
#SBATCH -e qpca_w80_N_sweep_norm.%j.err

set -uxo pipefail        # NOT -e: one failure should not abort the others
shopt -s nullglob
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

OUT_N10="idealized_inlet_qpca_endcf_ls6_normal_w80_bgs10_loc5km_N10"
OUT_N15="idealized_inlet_qpca_endcf_ls6_normal_w80_bgs10_loc5km_N15"
OUT_N25="idealized_inlet_qpca_endcf_ls6_normal_w80_bgs10_loc5km_N25"
OUT_N50="idealized_inlet_qpca_endcf_ls6_normal_w80_bgs10_loc5km_N50"
mkdir -p "$WORK/SWEMniCS/results/$OUT_N10" \
         "$WORK/SWEMniCS/results/$OUT_N15" \
         "$WORK/SWEMniCS/results/$OUT_N25" \
         "$WORK/SWEMniCS/results/$OUT_N50"

echo
echo "=== 1/4: N = 10 (baseline) + 5km loc + fixed κ=1 ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-ensemble-size 10 --override-loc-radius 5000 \
  --output-subdir "$OUT_N10" \
  || echo "[bundle] N=10 exited non-zero"

echo
echo "=== 2/4: N = 15 + 5km loc + fixed κ=1 ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-ensemble-size 15 --override-loc-radius 5000 \
  --output-subdir "$OUT_N15" \
  || echo "[bundle] N=15 exited non-zero"

echo
echo "=== 3/4: N = 25 + 5km loc + fixed κ=1 ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-ensemble-size 25 --override-loc-radius 5000 \
  --output-subdir "$OUT_N25" \
  || echo "[bundle] N=25 exited non-zero"

echo
echo "=== 4/4: N = 50 + 5km loc + fixed κ=1 ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-ensemble-size 50 --override-loc-radius 5000 \
  --output-subdir "$OUT_N50" \
  || echo "[bundle] N=50 exited non-zero"

echo
echo "=== sweep results ==="
for SUB in "$OUT_N10" "$OUT_N15" "$OUT_N25" "$OUT_N50"; do
  echo "--- $SUB ---"
  ls -la "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  cat "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  echo
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

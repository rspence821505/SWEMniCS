#!/bin/bash
# Vista gh-dev: BUNDLE — ensemble-size sweep at the loc-sweep operating
# point. All three runs share fixed κ=1, 5 km Gaspari-Cohn localization,
# λ_infl=1.0, single-window smoke config (vmax=15, track_shift=5,
# obs_fraction=0.043, σ_obs=0.01, window_len=4, obs_freq=2,
# bg_error_std=0.10).
#
# Only --override-ensemble-size differs:
#   N = 10  baseline (matches loc 5 km row of 733722 sweep — sanity check)
#   N = 25  mid-range — projector estimation variance should drop ~2.5×
#   N = 50  upper end — variance should drop ~5×, should expose any
#           N-independent obstacle (rank deficiency in C_E truncation,
#           bg_error_std too small for filter to find signal, …)
#
# Cost is roughly linear in N: ~7 / 17 / 35 min based on previous N=10
# runs on this queue. Total ≈ 60 min, right at the gh-dev 1 h cap —
# largest run runs LAST so a walltime kill loses the most expensive
# point but preserves the cheap baseline + mid-range slope.
# --------------------------------------------------------
#SBATCH -J qpca_w80_N_sweep
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -o qpca_w80_N_sweep.%j.out
#SBATCH -e qpca_w80_N_sweep.%j.err

set -uxo pipefail        # NOT -e: one failure should not abort the others
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_N10="idealized_inlet_qpca_endcf_vista_w80_bgs10_loc5km_N10"
OUT_N25="idealized_inlet_qpca_endcf_vista_w80_bgs10_loc5km_N25"
OUT_N50="idealized_inlet_qpca_endcf_vista_w80_bgs10_loc5km_N50"
mkdir -p "$WORK/SWEMniCS/results/$OUT_N10" \
         "$WORK/SWEMniCS/results/$OUT_N25" \
         "$WORK/SWEMniCS/results/$OUT_N50"

echo
echo "=== 1/3: N = 10 (baseline) + 5km loc + fixed κ=1 ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 7000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-ensemble-size 10 --override-loc-radius 5000 \
  --output-subdir "$OUT_N10" \
  || echo "[bundle] N=10 exited non-zero"

echo
echo "=== 2/3: N = 25 + 5km loc + fixed κ=1 ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 7000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-ensemble-size 25 --override-loc-radius 5000 \
  --output-subdir "$OUT_N25" \
  || echo "[bundle] N=25 exited non-zero"

echo
echo "=== 3/3: N = 50 + 5km loc + fixed κ=1 ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 7000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-ensemble-size 50 --override-loc-radius 5000 \
  --output-subdir "$OUT_N50" \
  || echo "[bundle] N=50 exited non-zero"

echo
echo "=== sweep results ==="
for SUB in "$OUT_N10" "$OUT_N25" "$OUT_N50"; do
  echo "--- $SUB ---"
  ls -la "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  cat "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  echo
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

#!/bin/bash
# Vista gh: BUNDLE — three-way head-to-head extended from cycling-4 to
# cycling-9 (1 h spinup + 9 × 80-min windows = 13 h total simulated
# time). Companion to the LS6 cycling-4 head-to-head (job 3205065,
# Section 4 of the chapter) and the LS6 cycling-4 QPCA-vs-4D-EnKF
# (job 3203539, the original headline).
#
# Same operating point as the chapter's Section 4 baseline:
#   vmax=15, track_shift=5 km, dt=600, nt_ramp=6, window_len=4,
#   obs_frequency=2 (80-min windows), N=10, ensemble_seed=2026,
#   bg_error_std=0.10, σ_obs=0.01, 5 km Gaspari-Cohn localization,
#   λ_infl=1.05, κ=1 fixed (QPCA only).
# Only --override-n-windows differs: 4 → 9.
#
# Storm geometry: track_duration=28800 s (8 h), closest approach to
# inlet centroid at t≈4.6 h (W3 of cycling). Cycling-9 covers t = 1 h
# to t = 13 h, so W1-W4 cover storm approach + closest approach,
# W5-W9 cover post-storm quiescent dynamics with vortex parked at end
# of track. Expected behavior:
#   QPCA-EnDCF — bounded growth through W4 (storm peak), then
#                quiescent post-storm cycling through W5-W9; the
#                extended trajectory tests whether the filter
#                stabilizes once peak forcing has cleared.
#   4D-EnKF    — catastrophic at W2 (per Section 4 result), then
#                pinned at numerical overflow through W3-W9.
#   Seq EnKF   — catastrophic at W1 or W2 (per Section 4 result),
#                then pinned at floating-point infinity through the
#                rest of the cycle.
#
# Per-run cost: cycling-4 at N=10 was ~21 min on Vista. Cycling-9
# scales linearly in the forecast cost but the stochastic filters'
# Newton solves get cheaper once the ensemble overflows (the residual
# norm explodes, so Newton fails fast or returns immediately). Budget
# ~50 min for QPCA, ~30 min each for the stochastic baselines.
# Total bundle ~110 min wall. The 4 h walltime request leaves
# comfortable margin for slower-than-expected forecast segments
# (especially for the sequential filter, which does L = 4 forecast
# restarts per window).
# --------------------------------------------------------
#SBATCH -J qpca_3way_cyc9
#SBATCH -A ADCIRC
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 04:00:00
#SBATCH -o qpca_3way_cyc9.%j.out
#SBATCH -e qpca_3way_cyc9.%j.err

set -uxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_QPCA="idealized_inlet_qpca_endcf_vista_gh_w80_bgs10_cyc9_infl105_qpca"
OUT_ENKF="idealized_inlet_qpca_endcf_vista_gh_w80_bgs10_cyc9_infl105_enkf4d"
OUT_SEQ="idealized_inlet_qpca_endcf_vista_gh_w80_bgs10_cyc9_infl105_seq_enkf"
mkdir -p "$WORK/SWEMniCS/results/$OUT_QPCA" \
         "$WORK/SWEMniCS/results/$OUT_ENKF" \
         "$WORK/SWEMniCS/results/$OUT_SEQ"

echo
echo "=== 1/3: QPCA-EnDCF (κ=1) + cyc 9w at operating point (seed 2026) ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 13000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 9 --inflation 1.05 \
  --ensemble-seed 2026 \
  --output-subdir "$OUT_QPCA" \
  || echo "[bundle] QPCA-EnDCF exited non-zero"

echo
echo "=== 2/3: stochastic 4D-EnKF + cyc 9w at operating point (seed 2026) ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 13000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method enkf4d --override-n-windows 9 --inflation 1.05 \
  --ensemble-seed 2026 \
  --output-subdir "$OUT_ENKF" \
  || echo "[bundle] 4D-EnKF exited non-zero"

echo
echo "=== 3/3: sequential stochastic EnKF + cyc 9w at operating point (seed 2026) ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 13000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method seq_enkf --override-n-windows 9 --inflation 1.05 \
  --ensemble-seed 2026 \
  --output-subdir "$OUT_SEQ" \
  || echo "[bundle] sequential EnKF exited non-zero"

echo
echo "=== comparison results ==="
for SUB in "$OUT_QPCA" "$OUT_ENKF" "$OUT_SEQ"; do
  echo "--- $SUB ---"
  ls -la "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  cat "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  echo
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

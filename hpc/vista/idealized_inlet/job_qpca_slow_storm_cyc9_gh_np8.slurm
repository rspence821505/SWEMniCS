#!/bin/bash
# Vista gh: SINGLETON — slow-storm cycling-9 at the operating point.
#
# Probes the hypothesis that the cycling-4 W4 blowup is caused by the
# storm traversing too fast for the N=10 ensemble to track per 80-min
# window (per-window fc-RMSE growth jumps 4× at the surge peak). Slow
# the storm 50 % (--track-duration-s 43200 → 12 h vortex traversal)
# and extend cycling to 9 windows so the cycle covers the full storm.
#
# Expected timing in this config:
#   spinup     0 → 1 h  (wind ramps 0 → Vmax)
#   W1         1 → 2:20 h   pre-storm
#   W2         2:20 → 3:40  vortex approaching
#   W3         3:40 → 5:00
#   W4         5:00 → 6:20
#   W5         6:20 → 7:40  vortex CLOSEST APPROACH (peak wind, t≈6:50)
#   W6         7:40 → 9:00  surge building
#   W7         9:00 → 10:20 SURGE PEAK (expected)
#   W8        10:20 → 11:40 post-peak
#   W9        11:40 → 13:00 vortex parks at end of track, post-storm
#
# Track geometry unchanged from baseline (40,-20) → (25,50) km, so the
# inlet is at vortex closest-approach when frac ≈ 0.57, which is
# t = 0.57 × 12 h ≈ 6:50 h — well inside W5.
#
# Per-window cost: same as baseline (each window is still 8 timesteps ×
# N=10 forecasts). 9 windows × ~21 min ≈ 3 h 10 min wall. The 4 h
# request leaves margin for slower-than-expected steps and queue setup.
#
# Everything else matches the today's-best operating point:
#   --smoke baseline (vmax=15, track_shift=5 km, dt=600 s, nt_ramp=6,
#                     window_len=4, obs_frequency=2 → 80-min window,
#                     obs_fraction=0.043, σ_obs=0.01,
#                     bg_error_std=0.10, seed=2026, N=10, 5 km loc)
#   --method qpca, fixed κ=1, --inflation 1.05
# --------------------------------------------------------
#SBATCH -J qpca_slow_storm_cyc9
#SBATCH -A ADCIRC
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 04:00:00
#SBATCH -o qpca_slow_storm_cyc9.%j.out
#SBATCH -e qpca_slow_storm_cyc9.%j.err

set -uxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT="idealized_inlet_qpca_endcf_vista_gh_w80_cyc9_infl105_slow_storm_td43200"
mkdir -p "$WORK/SWEMniCS/results/$OUT"

echo
echo "=== QPCA-EnDCF, slow-storm probe + cyc 9w ==="
echo "    track_duration_s = 43200 (12 h vortex traversal, 50% slower)"
echo "    n_windows        = 9   (covers full storm + post-storm)"
echo "    Operating point: κ=1 fixed, λ_infl=1.05, 5 km loc, N=10"
mpirun -n 8 --bind-to core --map-by socket --timeout 14400 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca \
  --track-duration-s 43200 \
  --override-n-windows 9 \
  --inflation 1.05 \
  --output-subdir "$OUT" \
  || echo "[job] slow-storm cyc9 exited non-zero"

echo
echo "=== result ==="
ls -la "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true
cat "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

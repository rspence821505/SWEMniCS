#!/bin/bash
# Vista gh: SINGLETON — surge-in-W11 probe at seed=2028.
#
# Refines the slow-storm probe (734146 / cyc9 at track_duration=12 h):
# that one was supposed to push surge into W7 but instead saw W4 blow
# up because the vortex still closed within ~12 km of the inlet during
# W4 even at half-speed. To push closest-approach deeper into the
# cycle, slow the storm 3× from baseline (track_duration_s = 86400 s
# = 24 h) — the inlet now experiences closest-approach at
# t ≈ 0.571 × 24 h ≈ 13.7 h, which falls inside W10 (13:00 → 14:20).
# Surge response lags 1 window → expected blowup at W11.
#
# Cycling-12 covers t = 1 h to 17 h = 16 h of cycling, which spans the
# entire approach + closest-approach + post-peak window with margin.
#
# Critical: seed=2028 per user request. From the LS6 seed-repeat,
# seed=2028 was the seed where the filter at N=10 actually did
# meaningful W4 work (+17.52% reduction) and produced the best end
# RMSE (1.32) — it's our "least chaotic" seed at the operating point,
# so any blowup we see at W11 isn't attributable to a bad ensemble
# draw.
#
# Operating point preserved otherwise: κ=1 fixed, λ_infl=1.05, 5 km
# Gaspari-Cohn loc, N=10, vmax=15, dt=600, 80-min windows, 1 h spinup.
#
# Cost estimate: pre-surge windows are cheap (gentle storm → Newton
# converges in fewer iters). Slow-storm cyc9 (track_duration_s=43200)
# took 28:56 wall for 9 windows. cyc12 here should run ~35-50 min
# pre-surge plus 15-30 min of expensive late windows once surge hits
# → ~50-90 min wall. 2.5 h walltime leaves comfortable margin.
# --------------------------------------------------------
#SBATCH -J qpca_surge_W11_s2028
#SBATCH -A ADCIRC
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:30:00
#SBATCH -o qpca_surge_W11_s2028.%j.out
#SBATCH -e qpca_surge_W11_s2028.%j.err

set -uxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT="idealized_inlet_qpca_endcf_vista_gh_w80_cyc12_infl105_seed2028_td86400"
mkdir -p "$WORK/SWEMniCS/results/$OUT"

echo
echo "=== QPCA-EnDCF surge-in-W11 probe ==="
echo "    track_duration_s = 86400 (24 h storm, 3× slower than baseline)"
echo "    closest approach: t≈13.7 h → W10 (13:00–14:20)"
echo "    surge peak expected: W11 (14:20–15:40)"
echo "    n_windows = 12, seed = 2028"
mpirun -n 8 --bind-to core --map-by socket --timeout 9000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca \
  --track-duration-s 86400 \
  --override-n-windows 12 \
  --inflation 1.05 \
  --ensemble-seed 2028 \
  --output-subdir "$OUT" \
  || echo "[job] surge_W11 cyc12 seed=2028 exited non-zero"

echo
echo "=== result ==="
ls -la "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true
cat "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

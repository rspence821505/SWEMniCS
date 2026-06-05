#!/bin/bash
# Vista gh-dev: BUNDLE — surge-W11 bg-bracket at bg ∈ {0.15, 0.20}.
#
# Closes the goldilocks gap from cancelled 734178 (bg=0.10, filter
# irrelevant) and 734189 (bg=0.30, blew up at W1):
#   bg = 0.15  modest perturbation — should give visible per-window
#              filter work without immediate explosion
#   bg = 0.20  stronger perturbation — push toward the failure regime
#              to confirm bracket
#
# Each run takes ~16 min wall (from 734189 elapsed) since slow-storm
# dynamics make Newton converge cheaply pre-surge. Bundle ≈ 32 min,
# comfortably under gh-dev 1 h cap.
#
# All else matches the operating point at seed=2028:
#   vmax=15, track_shift=5 km, track_duration_s=86400 (24 h storm),
#   dt=600, nt_ramp=6, window_len=4, obs_freq=2 (80-min windows),
#   n_windows=12, N=10, κ=1, λ_infl=1.05, 5 km loc.
#
# Geometry reminder: closest approach at t ≈ 13.7 h → inside W10
# (13:00 → 14:20). Surge response lags ~1 window → expected blowup
# at W11 (14:20 → 15:40). Cycling-12 covers everything through W12
# (post-storm).
# --------------------------------------------------------
#SBATCH -J qpca_surge_W11_bgbrk
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -o qpca_surge_W11_bgbrk.%j.out
#SBATCH -e qpca_surge_W11_bgbrk.%j.err

set -uxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_BG15="idealized_inlet_qpca_endcf_vista_w80_cyc12_infl105_seed2028_td86400_bg15"
OUT_BG20="idealized_inlet_qpca_endcf_vista_w80_cyc12_infl105_seed2028_td86400_bg20"
mkdir -p "$WORK/SWEMniCS/results/$OUT_BG15" \
         "$WORK/SWEMniCS/results/$OUT_BG20"

echo
echo "=== 1/2: surge-W11 probe at bg_error_std = 0.15 ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 3000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --method qpca \
  --vmax 15.0 --track-shift 5.0 \
  --track-duration-s 86400 \
  --dt 600.0 --nt-ramp 6 \
  --window-len 4 --obs-frequency 2 --nt-da 8 \
  --n-windows 12 \
  --ensemble-size 10 --ensemble-seed 2028 \
  --k-modes 1 --inflation 1.05 \
  --loc-radius 5000 \
  --obs-fraction 0.043 --obs-noise-level 0.01 \
  --background-error-std 0.15 \
  --output-subdir "$OUT_BG15" \
  || echo "[bundle] bg=0.15 exited non-zero"

echo
echo "=== 2/2: surge-W11 probe at bg_error_std = 0.20 ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 3000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --method qpca \
  --vmax 15.0 --track-shift 5.0 \
  --track-duration-s 86400 \
  --dt 600.0 --nt-ramp 6 \
  --window-len 4 --obs-frequency 2 --nt-da 8 \
  --n-windows 12 \
  --ensemble-size 10 --ensemble-seed 2028 \
  --k-modes 1 --inflation 1.05 \
  --loc-radius 5000 \
  --obs-fraction 0.043 --obs-noise-level 0.01 \
  --background-error-std 0.20 \
  --output-subdir "$OUT_BG20" \
  || echo "[bundle] bg=0.20 exited non-zero"

echo
echo "=== bracket results ==="
for SUB in "$OUT_BG15" "$OUT_BG20"; do
  echo "--- $SUB ---"
  ls -la "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  cat "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  echo
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

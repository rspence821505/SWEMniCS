#!/bin/bash
# Vista gh-dev: BUNDLE — QPCA-EnDCF vs 4D-EnKF head-to-head at vmax=20.
#
# Generalizes the headline LS6 result (3203539 / qpca_vs_enkf4d_cyc4) at
# vmax=15 to a stronger storm. If QPCA still beats 4D-EnKF decisively at
# vmax=20, the head-to-head conclusion is robust across the
# moderate-to-strong-storm range, not specific to one storm magnitude.
#
# Same shared config as 3203539 except --vmax 20:
#   cycling-4, κ=1 fixed, λ_infl=1.05, 5 km Gaspari-Cohn loc, N=10,
#   seed=2026, track_shift=5 km, dt=600, 80-min windows, σ_obs=0.01,
#   bg_std=0.10.
#
# Per-run cost on Vista cycling-4 at N=10: ~21-25 min. Bundle ≈ 50 min,
# fits 1 h gh-dev cap comfortably.
# --------------------------------------------------------
#SBATCH -J qpca_vs_enkf4d_vmax20
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -o qpca_vs_enkf4d_vmax20.%j.out
#SBATCH -e qpca_vs_enkf4d_vmax20.%j.err

set -uxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_QPCA="idealized_inlet_qpca_endcf_vista_vmax20_w80_bgs10_cyc4_infl105_qpca"
OUT_ENKF="idealized_inlet_qpca_endcf_vista_vmax20_w80_bgs10_cyc4_infl105_enkf4d"
mkdir -p "$WORK/SWEMniCS/results/$OUT_QPCA" \
         "$WORK/SWEMniCS/results/$OUT_ENKF"

echo
echo "=== 1/2: QPCA-EnDCF (κ=1) at vmax=20 + 5km loc + λ=1.05 + cyc 4w ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 3000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --method qpca \
  --vmax 20.0 --track-shift 5.0 \
  --dt 600.0 --nt-ramp 6 \
  --window-len 4 --obs-frequency 2 --nt-da 8 \
  --n-windows 4 \
  --ensemble-size 10 --ensemble-seed 2026 \
  --k-modes 1 --inflation 1.05 \
  --loc-radius 5000 \
  --obs-fraction 0.043 --obs-noise-level 0.01 \
  --background-error-std 0.10 \
  --output-subdir "$OUT_QPCA" \
  || echo "[bundle] QPCA-EnDCF vmax=20 exited non-zero"

echo
echo "=== 2/2: stochastic 4D-EnKF at vmax=20 + 5km loc + λ=1.05 + cyc 4w ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 3000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --method enkf4d \
  --vmax 20.0 --track-shift 5.0 \
  --dt 600.0 --nt-ramp 6 \
  --window-len 4 --obs-frequency 2 --nt-da 8 \
  --n-windows 4 \
  --ensemble-size 10 --ensemble-seed 2026 \
  --inflation 1.05 \
  --loc-radius 5000 \
  --obs-fraction 0.043 --obs-noise-level 0.01 \
  --background-error-std 0.10 \
  --output-subdir "$OUT_ENKF" \
  || echo "[bundle] 4D-EnKF vmax=20 exited non-zero"

echo
echo "=== comparison results ==="
for SUB in "$OUT_QPCA" "$OUT_ENKF"; do
  echo "--- $SUB ---"
  ls -la "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  cat "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  echo
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

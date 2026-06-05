#!/bin/bash
# Vista gh-dev: SINGLETON — dt=150 cycling-4 at the operating point.
#
# Probes whether very fine timestepping changes the cycling outcome.
# dt quartered from baseline (600 → 150) means 4× timesteps per window.
#
# Cost estimate: ~70-90 min on Vista cycling-4 at N=10. The 1 h gh-dev
# cap is tight — if Newton iters don't drop substantially with smaller
# dt, this could overrun. Worth trying because:
#   a) Vista gh-dev queue is very light right now (typical wait < 15
#      min), so failure here costs only one queue wait.
#   b) CLAUDE.md notes dt=150 + np=8 has been used successfully for
#      state-only 4D-Var cycling at this problem.
#   c) If it overruns we know to bump to gh (3h+) next time — useful
#      data on the dt cost scaling.
#
# Same operating point as Vista 734045 kfixed1 otherwise. NOT using
# --smoke so we can pass dt + matched ramp/obs_frequency explicitly.
#
# Wall-time-equivalent config (80-min window, 1 h spinup):
#   dt          = 150 s
#   nt_ramp     = 24   (1 h spinup)
#   window_len  = 4    (4 obs per window)
#   obs_freq    = 8    (every 20 min — same wall-time as baseline)
#   nt_da       = 32   (80-min window — same wall-time)
#   n_windows   = 4
# --------------------------------------------------------
#SBATCH -J qpca_dt150_cyc4
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -o qpca_dt150_cyc4.%j.out
#SBATCH -e qpca_dt150_cyc4.%j.err

set -uxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT="idealized_inlet_qpca_endcf_vista_w80_cyc4_infl105_dt150"
mkdir -p "$WORK/SWEMniCS/results/$OUT"

echo
echo "=== QPCA-EnDCF, dt=150 + cyc 4w at operating point ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 3000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --method qpca \
  --vmax 15.0 --track-shift 5.0 \
  --dt 150.0 --nt-ramp 24 \
  --window-len 4 --obs-frequency 8 --nt-da 32 \
  --n-windows 4 \
  --ensemble-size 10 --ensemble-seed 2026 \
  --k-modes 1 --inflation 1.05 \
  --loc-radius 5000 \
  --obs-fraction 0.043 --obs-noise-level 0.01 \
  --background-error-std 0.10 \
  --output-subdir "$OUT" \
  || echo "[job] dt=150 cyc4 exited non-zero"

echo
echo "=== result ==="
ls -la "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true
cat "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

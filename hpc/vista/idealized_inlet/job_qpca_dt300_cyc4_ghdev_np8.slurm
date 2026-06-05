#!/bin/bash
# Vista gh-dev: SINGLETON — dt=300 cycling-4 at the operating point.
#
# Probes whether finer timestepping changes the cycling outcome at the
# storm peak. dt halved → 2× timesteps per window, Newton solves with
# milder per-step dynamics. If the W4 blowup is "the Newton solver
# can't track large per-step state changes during surge response,"
# halving dt should help.
#
# Same operating point as Vista 734045 kfixed1: κ=1, λ_infl=1.05,
# 5 km loc, N=10, vmax=15, seed=2026. NOT using --smoke so we can pass
# dt=300 + matched ramp/obs_frequency/nt_da explicitly.
#
# Wall-time-equivalent config (80-min window, 1 h spinup):
#   dt          = 300 s
#   nt_ramp     = 12   (1 h spinup)
#   window_len  = 4    (4 obs per window)
#   obs_freq    = 4    (every 20 min — same as dt=600 baseline)
#   nt_da       = 16   (80-min window — same as baseline)
#   n_windows   = 4
#
# Expected cost on Vista cycling-4 at N=10: ~30-40 min (1.5-2× baseline
# 21 min). Fits 1 h gh-dev cap with margin.
# --------------------------------------------------------
#SBATCH -J qpca_dt300_cyc4
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -o qpca_dt300_cyc4.%j.out
#SBATCH -e qpca_dt300_cyc4.%j.err

set -uxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT="idealized_inlet_qpca_endcf_vista_w80_cyc4_infl105_dt300"
mkdir -p "$WORK/SWEMniCS/results/$OUT"

echo
echo "=== QPCA-EnDCF, dt=300 + cyc 4w at operating point ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 3000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --method qpca \
  --vmax 15.0 --track-shift 5.0 \
  --dt 300.0 --nt-ramp 12 \
  --window-len 4 --obs-frequency 4 --nt-da 16 \
  --n-windows 4 \
  --ensemble-size 10 --ensemble-seed 2026 \
  --k-modes 1 --inflation 1.05 \
  --loc-radius 5000 \
  --obs-fraction 0.043 --obs-noise-level 0.01 \
  --background-error-std 0.10 \
  --output-subdir "$OUT" \
  || echo "[job] dt=300 cyc4 exited non-zero"

echo
echo "=== result ==="
ls -la "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true
cat "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

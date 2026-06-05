#!/bin/bash
# Vista gh: SINGLETON — surge-in-W11 probe with stronger bg perturbation.
#
# Successor to cancelled 734178. That run showed W1-W2 fc_rmse so small
# (0.09, 0.24) that the filter had essentially no work to do — state Δ
# was exactly +0.0% because the analysis correction was in the 4th
# decimal of RMSE. Slow-storm dynamics + smoke-default bg_std=0.10 ⇒
# filter-irrelevant pre-storm cycle.
#
# Fix: triple the bg perturbation (bg_error_std = 0.30) so the
# background ensemble starts ~3× further from truth, giving the filter
# meaningful initial error to chase from W1 onward. Storm geometry
# unchanged from 734178 — closest approach still at t≈13.7h (W10),
# expected surge blowup at W11.
#
# NOT using --smoke so bg_error_std and the dt/ramp/window config can
# all be set explicitly without smoke overriding them.
#
# Config:
#   vmax            = 15  (smoke baseline)
#   track_shift     = 5 km
#   track_duration  = 86400 s (24 h, 3× slower than baseline)
#   dt              = 600 s
#   nt_ramp         = 6 (1 h spinup)
#   window_len      = 4, obs_freq = 2, nt_da = 8 (80-min windows)
#   n_windows       = 12 (W11 covers t=14:20-15:40, post-closest-approach)
#   N               = 10
#   seed            = 2028
#   κ=1 fixed, λ_infl=1.05, 5 km Gaspari-Cohn loc
#   obs_fraction=0.043, σ_obs=0.01
#   bg_error_std    = 0.30  ← the changed knob
#
# Cost: similar to 734178 (~75-90 min wall) since dynamics are the
# same magnitude; just the prior ensemble is more spread. 2.5 h
# walltime keeps margin.
# --------------------------------------------------------
#SBATCH -J qpca_surge_W11_bg30
#SBATCH -A ADCIRC
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:30:00
#SBATCH -o qpca_surge_W11_bg30.%j.out
#SBATCH -e qpca_surge_W11_bg30.%j.err

set -uxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT="idealized_inlet_qpca_endcf_vista_gh_w80_cyc12_infl105_seed2028_td86400_bg30"
mkdir -p "$WORK/SWEMniCS/results/$OUT"

echo
echo "=== QPCA-EnDCF surge-in-W11 probe (bg_error_std=0.30) ==="
echo "    track_duration_s = 86400 (24 h storm)"
echo "    closest approach: t≈13.7 h → W10"
echo "    surge peak expected: W11"
echo "    bg_error_std = 0.30 (3× smoke baseline) — gives filter work to do"
echo "    n_windows = 12, seed = 2028"
mpirun -n 8 --bind-to core --map-by socket --timeout 9000 \
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
  --background-error-std 0.30 \
  --output-subdir "$OUT" \
  || echo "[job] surge_W11 bg=0.30 cyc12 seed=2028 exited non-zero"

echo
echo "=== result ==="
ls -la "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true
cat "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

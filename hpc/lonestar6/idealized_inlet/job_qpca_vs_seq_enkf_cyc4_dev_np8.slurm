#!/bin/bash
# LS6 dev: BUNDLE — QPCA-EnDCF vs traditional sequential stochastic EnKF
# at vmax=15, same operating point and same ensemble seed as the
# QPCA-vs-4D-EnKF head-to-head (LS6 3203539, the chapter's headline
# result). All three filters now have a same-cluster, same-seed,
# same-partition comparison.
#
# Sequential EnKF: one perturbed-observation Kalman update per
# observation time inside the window, with intermediate forecast
# segments propagating the *analyzed* ensemble between updates. This is
# the textbook sequential EnKF described in DA literature and ported
# here from QPCA-EnDCF-Paper/src/filters/seq_enkf.py (the L96 reference
# implementation), adapted to apply_H callable + sparse Schur
# localization for the SWE PDE state.
#
# Same shared config as 3203539:
#   --smoke baseline (vmax=15, track_shift=5 km, dt=600 s, nt_ramp=6,
#                     window_len=4, obs_frequency=2 → 80-min window,
#                     obs_fraction=0.043 → ~500 obs/time,
#                     bg_error_std=0.10, seed=2026, 5 km loc).
#   cycling-4, κ=1 fixed for QPCA, λ_infl=1.05.
#
# Cost: QPCA cycling-4 ≈ 25 min on LS6. Sequential EnKF does L=4
# segments per window instead of 1 forecast plus 1 update, with no
# additional total timesteps but ~4× the solver-restart overhead. Per
# window ≈ 30-35 min; total bundle ≈ 60-70 min. Fits 2 h dev cap with
# margin.
# --------------------------------------------------------
#SBATCH -J qpca_vs_seq_enkf
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o qpca_vs_seq_enkf.%j.out
#SBATCH -e qpca_vs_seq_enkf.%j.err

set -uxo pipefail
shopt -s nullglob
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

OUT_QPCA="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl105_qpca_seedrun"
OUT_SEQ="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl105_seq_enkf"
mkdir -p "$WORK/SWEMniCS/results/$OUT_QPCA" \
         "$WORK/SWEMniCS/results/$OUT_SEQ"

echo
echo "=== 1/2: QPCA-EnDCF (κ=1) + 5km loc + λ=1.05 + cyc 4w (seed 2026) ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --ensemble-seed 2026 \
  --output-subdir "$OUT_QPCA" \
  || echo "[bundle] QPCA-EnDCF exited non-zero"

echo
echo "=== 2/2: sequential stochastic EnKF + 5km loc + λ=1.05 + cyc 4w (seed 2026) ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method seq_enkf --override-n-windows 4 --inflation 1.05 \
  --ensemble-seed 2026 \
  --output-subdir "$OUT_SEQ" \
  || echo "[bundle] sequential EnKF exited non-zero"

echo
echo "=== comparison results ==="
for SUB in "$OUT_QPCA" "$OUT_SEQ"; do
  echo "--- $SUB ---"
  ls -la "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  cat "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  echo
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

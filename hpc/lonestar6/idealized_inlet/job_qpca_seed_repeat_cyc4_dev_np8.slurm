#!/bin/bash
# LS6 dev: BUNDLE — seed-repeat at the QPCA-EnDCF operating point.
# Three runs at the configuration already established by infl_sweep_ls6
# (best λ_infl = 1.05) and loc_sweep_vista (5 km Gaspari-Cohn), differing
# ONLY in --ensemble-seed. Combined with the existing λ=1.05 run at
# seed=2026 (in results/infl_sweep_ls6/infl105/) this gives a 4-sample
# variance estimate for the cycling-4 operating point.
#
# Purpose: every "winner" claim today (loc 10 km wins, λ_infl=1.05 wins,
# adaptive κ vs fixed κ, …) hinges on differences of ~0.1-3 % between
# runs. We have no seed-variance estimate to know whether those gaps
# are signal or noise. This bundle establishes the noise floor.
#
# Shared config (--smoke baseline): vmax=15, track_shift=5 km, dt=600 s,
# nt_ramp=6, window_len=4, obs_frequency=2 (80-min DA window), N=10,
# obs_fraction=0.043 (~500 obs), σ_obs=0.01, bg_error_std=0.10,
# 5 km Gaspari-Cohn localization, λ_infl=1.05, 4-window cycling,
# κ=1 (fixed), method=qpca.
#
# Per-run cost: 4 windows × 80 Newton solves on np=8 ≈ 27 min (from
# infl_sweep_ls6 step times: 27:29, 28:58, 26:46).
# Total bundle: ~82 min wall (well under 2 h dev limit).
# --------------------------------------------------------
#SBATCH -J qpca_seed_repeat_cyc4
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o qpca_seed_repeat_cyc4.%j.out
#SBATCH -e qpca_seed_repeat_cyc4.%j.err

set -uxo pipefail        # NOT -e: one failure should not abort the others
shopt -s nullglob
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

OUT_S2027="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl105_seed2027"
OUT_S2028="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl105_seed2028"
OUT_S2029="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl105_seed2029"
mkdir -p "$WORK/SWEMniCS/results/$OUT_S2027" \
         "$WORK/SWEMniCS/results/$OUT_S2028" \
         "$WORK/SWEMniCS/results/$OUT_S2029"

echo
echo "=== 1/3: seed = 2027 + 5km loc + λ=1.05 + cyc 4w (κ=1) ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --ensemble-seed 2027 \
  --output-subdir "$OUT_S2027" \
  || echo "[bundle] seed=2027 exited non-zero"

echo
echo "=== 2/3: seed = 2028 + 5km loc + λ=1.05 + cyc 4w (κ=1) ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --ensemble-seed 2028 \
  --output-subdir "$OUT_S2028" \
  || echo "[bundle] seed=2028 exited non-zero"

echo
echo "=== 3/3: seed = 2029 + 5km loc + λ=1.05 + cyc 4w (κ=1) ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --ensemble-seed 2029 \
  --output-subdir "$OUT_S2029" \
  || echo "[bundle] seed=2029 exited non-zero"

echo
echo "=== seed-repeat results ==="
for SUB in "$OUT_S2027" "$OUT_S2028" "$OUT_S2029"; do
  echo "--- $SUB ---"
  ls -la "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  cat "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  echo
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

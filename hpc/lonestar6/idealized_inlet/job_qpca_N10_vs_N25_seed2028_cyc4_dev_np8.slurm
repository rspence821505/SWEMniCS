#!/bin/bash
# LS6 dev: BUNDLE — controlled N=10 vs N=25 comparison at seed=2028.
#
# The Vista 734118 N=25 run hinted that N=25 hurts W4 filter performance
# (an_rmse 2.92 vs N=10's 2.47, W4 state Δ 1.81% vs 14.04%), but with
# only n=1 sample in a regime where end RMSE varies 1.32-4193 across
# seeds we can't tell signal from noise. This bundle pins the seed so
# both runs draw the same background ensemble structure → forecast
# trajectories should match closely → any state-Δ difference is
# attributable to the analysis update (i.e. to N).
#
# Why seed=2028? In the LS6 seed-repeat (3203487), seed=2028 was the
# seed where the filter at N=10 actually did meaningful W4 work
# (+17.52% W4 state Δ, end RMSE 1.32 — best of 4 seeds). If N=25 helps,
# it should make a bigger dent in that already-tractable W4 forecast;
# if N=25 hurts, the +17.52% drops. Either outcome is decisive vs the
# Vista 734118 ambiguity.
#
# Both runs share fixed κ=1, λ_infl=1.05, 5 km Gaspari-Cohn loc,
# --smoke baseline (vmax=15, track_shift=5, σ_obs=0.01, bg_std=0.10,
# 4-window cycling, 80-min windows). Only --override-ensemble-size
# differs.
#
# Per-run cost on LS6 cycling-4 ≈ 25 min (N=10, from seed-repeat
# elapsed) → N=25 ≈ 25 × 2.5 = ~62 min. Total bundle ~87 min, fits
# 2 h dev cap comfortably.
# --------------------------------------------------------
#SBATCH -J qpca_N10_v_N25_s2028
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o qpca_N10_v_N25_s2028.%j.out
#SBATCH -e qpca_N10_v_N25_s2028.%j.err

set -uxo pipefail
shopt -s nullglob
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

OUT_N10="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl105_seed2028_N10"
OUT_N25="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl105_seed2028_N25"
mkdir -p "$WORK/SWEMniCS/results/$OUT_N10" \
         "$WORK/SWEMniCS/results/$OUT_N25"

echo
echo "=== 1/2: N = 10, seed = 2028 + 5km loc + λ=1.05 + cyc 4w ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --ensemble-seed 2028 --override-ensemble-size 10 \
  --output-subdir "$OUT_N10" \
  || echo "[bundle] N=10 seed=2028 exited non-zero"

echo
echo "=== 2/2: N = 25, seed = 2028 + 5km loc + λ=1.05 + cyc 4w ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --ensemble-seed 2028 --override-ensemble-size 25 \
  --output-subdir "$OUT_N25" \
  || echo "[bundle] N=25 seed=2028 exited non-zero"

echo
echo "=== comparison results ==="
for SUB in "$OUT_N10" "$OUT_N25"; do
  echo "--- $SUB ---"
  ls -la "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  cat "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  echo
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

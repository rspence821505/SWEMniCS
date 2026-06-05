#!/bin/bash
# Vista gh-dev: SINGLETON — operating-point cycling-4 at N=25.
# Tests whether the larger ensemble materially changes outcomes in
# cycling (where the single-window N-sweep showed N=15/25 had no
# effect, but cycling exposes projector-variance failure modes that
# single-window doesn't).
#
# Config matches the today's-best operating point chosen from infl
# sweep + adaptive-κ partial: fixed κ=1, λ_infl=1.05, 5 km Gaspari-Cohn
# localization, 4-window cycling, vmax=15, track_shift=5, bg=0.10,
# seed=2026. Only --override-ensemble-size differs from Vista 734045's
# kfixed1 run (which had N=10 and gave end RMSE 2.47).
#
# Expected cost on Vista: ~21 min per cycling-4 run at N=10 (from 734045
# elapsed) × 2.5 ≈ 52 min. Fits 1 h gh-dev cap with ~8 min margin.
# --------------------------------------------------------
#SBATCH -J qpca_N25_cyc4
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -o qpca_N25_cyc4.%j.out
#SBATCH -e qpca_N25_cyc4.%j.err

set -uxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT="idealized_inlet_qpca_endcf_vista_w80_bgs10_cyc4_infl105_kfixed1_N25"
mkdir -p "$WORK/SWEMniCS/results/$OUT"

echo
echo "=== QPCA-EnDCF, FIXED κ=1 + 5km loc + λ=1.05 + cyc 4w + N=25 ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 3000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --override-ensemble-size 25 \
  --output-subdir "$OUT" \
  || echo "[job] N=25 cyc4 exited non-zero"

echo
echo "=== result ==="
ls -la "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true
cat "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

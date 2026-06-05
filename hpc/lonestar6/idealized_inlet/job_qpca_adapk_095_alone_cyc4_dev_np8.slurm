#!/bin/bash
# LS6 dev: SINGLETON — adaptive κ τ=0.95 alone.
# Companion to Vista 734015 (which is running κ=1 fixed + adaptive
# τ=0.85 but will exceed the gh-dev 1 h walltime before reaching the
# third τ=0.95 run). This recovers the missing τ=0.95 data point so
# the adaptive sweep is complete.
#
# Same config as the Vista bundle's third stanza: 4-window cycling,
# 5 km Gaspari-Cohn localization, λ_infl=1.05, --kappa-target 0.95,
# vmax=15 smoke baseline, N=10, seed=2026.
#
# Per-run cost: 4 windows × 80 Newton solves on np=8 ≈ 27 min (from
# infl_sweep_ls6 step times).
# --------------------------------------------------------
#SBATCH -J qpca_adap095_alone
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -o qpca_adap095_alone.%j.out
#SBATCH -e qpca_adap095_alone.%j.err

set -uxo pipefail        # NOT -e: failure should not silently mask
shopt -s nullglob
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

OUT_AD095="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl105_kadap095"
mkdir -p "$WORK/SWEMniCS/results/$OUT_AD095"

echo
echo "=== QPCA-EnDCF, ADAPTIVE κ (τ=0.95) + 5km loc + λ=1.05 + cyc 4w ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --kappa-target 0.95 \
  --output-subdir "$OUT_AD095" \
  || echo "[job] adaptive κ τ=0.95 exited non-zero"

echo
echo "=== result ==="
ls -la "$WORK/SWEMniCS/results/$OUT_AD095/result_qpca_endcf.json" 2>&1 || true
cat "$WORK/SWEMniCS/results/$OUT_AD095/result_qpca_endcf.json" 2>&1 || true

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

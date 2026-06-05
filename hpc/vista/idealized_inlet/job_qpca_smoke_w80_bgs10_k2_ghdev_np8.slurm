#!/bin/bash
# Vista gh-dev: QPCA-EnDCF smoke, same as the κ=1 baseline run 733575
# but with κ=2 (retain top-2 whitened-residual eigenmodes). Tests the
# hypothesis from the eigenvalue diagnostic that mode 2 carries real
# signal (λ_2 ≈ 75% of λ_1 at this bg_error_std=0.10 config).
# --------------------------------------------------------
#SBATCH -J qpca_w80_bgs10_k2
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o qpca_w80_bgs10_k2.%j.out
#SBATCH -e qpca_w80_bgs10_k2.%j.err

set -euxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_SUBDIR="idealized_inlet_qpca_endcf_vista_w80_bgs10_k2"
mkdir -p "$WORK/SWEMniCS/results/$OUT_SUBDIR"

mpirun -n 8 --bind-to core --map-by socket --timeout 7000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke \
  --override-k 2 \
  --output-subdir "$OUT_SUBDIR"

echo
echo "=== result JSON ==="
ls -la "$WORK/SWEMniCS/results/$OUT_SUBDIR/result_qpca_endcf.json" || true
cat "$WORK/SWEMniCS/results/$OUT_SUBDIR/result_qpca_endcf.json" || true

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

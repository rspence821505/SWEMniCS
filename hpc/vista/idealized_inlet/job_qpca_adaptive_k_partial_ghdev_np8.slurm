#!/bin/bash
# Vista gh-dev: PARTIAL BUNDLE — adaptive-κ resubmit (κ=1 fixed + τ=0.85
# only). Original 734015 bundle was lost to the UnboundLocalError fix.
# The τ=0.95 third run is covered separately by LS6 3203500 (PENDING)
# so we omit it here, which lets the 2-run bundle fit the gh-dev 1 h cap
# (~28 min × 2 ≈ 56 min, comfortable margin).
#
# Both runs share 5 km Gaspari-Cohn localization + λ_infl=1.05 +
# 4-window cycling at the smoke baseline. Comparison stays consistent
# across Vista (κ=1, τ=0.85) and LS6 (τ=0.95) — same code, same
# config, same seed; only κ-selection differs.
# --------------------------------------------------------
#SBATCH -J qpca_adapk_part
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -o qpca_adapk_part.%j.out
#SBATCH -e qpca_adapk_part.%j.err

set -uxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_FIXED="idealized_inlet_qpca_endcf_vista_w80_bgs10_cyc4_infl105_kfixed1"
OUT_AD085="idealized_inlet_qpca_endcf_vista_w80_bgs10_cyc4_infl105_kadap085"
mkdir -p "$WORK/SWEMniCS/results/$OUT_FIXED" \
         "$WORK/SWEMniCS/results/$OUT_AD085"

echo
echo "=== 1/2: QPCA-EnDCF, FIXED κ=1 (control) + 5km loc + λ=1.05 + cyc 4w ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 7000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --output-subdir "$OUT_FIXED" \
  || echo "[bundle] fixed κ=1 exited non-zero"

echo
echo "=== 2/2: QPCA-EnDCF, ADAPTIVE κ (τ=0.85) + 5km loc + λ=1.05 + cyc 4w ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 7000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --kappa-target 0.85 \
  --output-subdir "$OUT_AD085" \
  || echo "[bundle] adaptive κ τ=0.85 exited non-zero"

echo
echo "=== comparison results ==="
for SUB in "$OUT_FIXED" "$OUT_AD085"; do
  echo "--- $SUB ---"
  ls -la "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  cat "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  echo
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

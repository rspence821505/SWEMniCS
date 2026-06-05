#!/bin/bash
# LS6 dev: BUNDLE — multiplicative-inflation sweep with 4-window cycling.
# Three runs back-to-back at λ_infl ∈ {1.02, 1.05, 1.10}, all else at the
# current best smoke config (κ=1, bg=0.10, 5km loc, 80-min window, N=10,
# 500 obs, σ_obs=0.01).
#
# This is the response to the cycling divergence observed in LS6 run
# 3202978 (no inflation): ensemble spread grew 78× over 4 windows and
# the filter collapsed at W4. Standard EnKF fix is multiplicative
# inflation; the paper claims spectral truncation makes it unnecessary
# but our SWE-PDE cycling shows otherwise. This sweep tests whether
# even mild inflation (1.02) is enough to stabilize cycling, and
# whether more aggressive inflation (1.10) over-corrects.
#
# Per-run cost: 4 windows × 80 Newton solves on np=8 ≈ 27 min.
# Total bundle: ~80 min wall (well under 2 h dev limit).
# --------------------------------------------------------
#SBATCH -J qpca_w80_cyc4_infl_sweep
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o qpca_w80_cyc4_infl_sweep.%j.out
#SBATCH -e qpca_w80_cyc4_infl_sweep.%j.err

set -uxo pipefail        # NOT -e: one failure should not abort the others
shopt -s nullglob
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

OUT_102="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl102"
OUT_105="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl105"
OUT_110="idealized_inlet_qpca_endcf_ls6_w80_bgs10_cyc4_infl110"
mkdir -p "$WORK/SWEMniCS/results/$OUT_102" \
         "$WORK/SWEMniCS/results/$OUT_105" \
         "$WORK/SWEMniCS/results/$OUT_110"

echo
echo "=== 1/3: λ_infl = 1.02, n_windows = 4 ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-n-windows 4 --inflation 1.02 \
  --output-subdir "$OUT_102" \
  || echo "[bundle] λ_infl=1.02 exited non-zero"

echo
echo "=== 2/3: λ_infl = 1.05, n_windows = 4 ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-n-windows 4 --inflation 1.05 \
  --output-subdir "$OUT_105" \
  || echo "[bundle] λ_infl=1.05 exited non-zero"

echo
echo "=== 3/3: λ_infl = 1.10, n_windows = 4 ==="
ibrun python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --override-n-windows 4 --inflation 1.10 \
  --output-subdir "$OUT_110" \
  || echo "[bundle] λ_infl=1.10 exited non-zero"

echo
echo "=== sweep results ==="
for SUB in "$OUT_102" "$OUT_105" "$OUT_110"; do
  echo "--- $SUB ---"
  ls -la "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  cat "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  echo
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

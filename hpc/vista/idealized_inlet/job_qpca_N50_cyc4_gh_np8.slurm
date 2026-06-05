#!/bin/bash
# Vista gh: SINGLETON — N=50 cycling-4 at the operating point.
#
# Closes out the small-N hypothesis: at N=10 the QPCA-EnDCF cycling
# suffers filter-induced instability (per bg-bracket bracket runs) and
# 4D-EnKF blows up by 32 OOMs (per LS6 3203539). The argument for the
# thesis is "QPCA dominates at small N where spectral truncation is
# load-bearing." For the chapter we need evidence that the failure mode
# is N-dependent — i.e. that at sufficiently large N, the cycling
# stabilizes (or at least doesn't fail catastrophically).
#
# N=50 is 5× the operating-point N. At this size the projector
# eigenvectors are estimated from 5× as many samples and the empirical
# P_xz cross-covariance is correspondingly less noisy. If the
# operating-point cycling stabilizes at N=50, the small-N diagnosis is
# confirmed. If it still blows up at W4, the cycling instability is
# deeper than ensemble-size effects.
#
# Config matches today's-best operating point at seed=2026:
#   --smoke baseline (vmax=15, track_shift=5, σ_obs=0.01, bg_std=0.10,
#                     4-window cycling, 80-min windows)
#   --method qpca, κ=1, λ_infl=1.05, 5 km loc.
# Only --override-ensemble-size 50 differs.
#
# Per-run cost: cycling-4 at N=10 on Vista ≈ 21 min. Linear scaling
# in N → N=50 ≈ 105 min. 1.5 h walltime gives ~~20 min margin.
# --------------------------------------------------------
#SBATCH -J qpca_N50_cyc4
#SBATCH -A ADCIRC
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:30:00
#SBATCH -o qpca_N50_cyc4.%j.out
#SBATCH -e qpca_N50_cyc4.%j.err

set -uxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT="idealized_inlet_qpca_endcf_vista_w80_cyc4_infl105_N50"
mkdir -p "$WORK/SWEMniCS/results/$OUT"

echo
echo "=== QPCA-EnDCF, N=50 + cycling-4 at operating point ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 5000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --override-ensemble-size 50 \
  --output-subdir "$OUT" \
  || echo "[job] N=50 cyc4 exited non-zero"

echo
echo "=== result ==="
ls -la "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true
cat "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

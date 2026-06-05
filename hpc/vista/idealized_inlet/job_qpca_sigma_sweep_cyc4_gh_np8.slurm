#!/bin/bash
# Vista gh: BUNDLE — observation-noise sweep at the operating point.
# Tests whether obs informativeness is the bottleneck for cycling-4
# state-Δ improvement at fixed κ=1, λ_infl=1.05, 5 km loc, N=10.
#
# All four runs share the today's-best cycling config:
#   --smoke baseline (vmax=15, track_shift=5 km, dt=600 s, nt_ramp=6,
#                     window_len=4, obs_frequency=2 → 80-min window,
#                     obs_fraction=0.043 → ~500 obs/time,
#                     bg_error_std=0.10, seed=2026, 5 km loc)
#   --method qpca, --override-n-windows 4, --inflation 1.05
# Only --override-obs-noise-level differs:
#   σ = 0.001  10× tighter than current best — does very-informative
#              obs unlock state Δ that's been stuck at sub-percent
#              cycling-mean?
#   σ = 0.005  intermediate — finds the elbow if there is one
#   σ = 0.01   fresh control at the operating point (734045 kfixed1 is
#              the existing reference point — repeat at same seed in
#              this bundle to remove cross-bundle variance)
#   σ = 0.05   5× looser — confirms the expected degradation; sanity
#              check that the σ axis is monotonic
#
# Twin-experiment caveat: --obs-noise-level controls BOTH the synthetic
# obs noise (truth + ε, ε~N(0,σ²)) AND R passed to the filter, so each
# run is internally self-consistent. This is "what the math can do with
# this quality of obs," not "real-world obs at σ=0.001 would help."
#
# Per-run cost on Vista cycling-4 at N=10: ~21 min (from 734045 elapsed).
# Total bundle: 4 × ~21 ≈ 85 min, well under 3 h gh-queue request.
# --------------------------------------------------------
#SBATCH -J qpca_sigma_sweep_cyc4
#SBATCH -A ADCIRC
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 03:00:00
#SBATCH -o qpca_sigma_sweep_cyc4.%j.out
#SBATCH -e qpca_sigma_sweep_cyc4.%j.err

set -uxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT_S1="idealized_inlet_qpca_endcf_vista_gh_w80_cyc4_infl105_sigma0p001"
OUT_S5="idealized_inlet_qpca_endcf_vista_gh_w80_cyc4_infl105_sigma0p005"
OUT_S10="idealized_inlet_qpca_endcf_vista_gh_w80_cyc4_infl105_sigma0p01"
OUT_S50="idealized_inlet_qpca_endcf_vista_gh_w80_cyc4_infl105_sigma0p05"
mkdir -p "$WORK/SWEMniCS/results/$OUT_S1" \
         "$WORK/SWEMniCS/results/$OUT_S5" \
         "$WORK/SWEMniCS/results/$OUT_S10" \
         "$WORK/SWEMniCS/results/$OUT_S50"

echo
echo "=== 1/4: σ_obs = 0.001 (stress, 10× tighter) ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 3000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --override-obs-noise-level 0.001 \
  --output-subdir "$OUT_S1" \
  || echo "[bundle] σ=0.001 exited non-zero"

echo
echo "=== 2/4: σ_obs = 0.005 (intermediate) ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 3000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --override-obs-noise-level 0.005 \
  --output-subdir "$OUT_S5" \
  || echo "[bundle] σ=0.005 exited non-zero"

echo
echo "=== 3/4: σ_obs = 0.01 (control, current operating point) ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 3000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --override-obs-noise-level 0.01 \
  --output-subdir "$OUT_S10" \
  || echo "[bundle] σ=0.01 exited non-zero"

echo
echo "=== 4/4: σ_obs = 0.05 (5× looser, sanity check) ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 3000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --smoke --method qpca --override-n-windows 4 --inflation 1.05 \
  --override-obs-noise-level 0.05 \
  --output-subdir "$OUT_S50" \
  || echo "[bundle] σ=0.05 exited non-zero"

echo
echo "=== sweep results ==="
for SUB in "$OUT_S1" "$OUT_S5" "$OUT_S10" "$OUT_S50"; do
  echo "--- $SUB ---"
  ls -la "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  cat "$WORK/SWEMniCS/results/$SUB/result_qpca_endcf.json" 2>&1 || true
  echo
done

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

#!/bin/bash
# Vista gh: SINGLETON — dt=150 cycling-4 at the operating point (safe
# copy of 734164 / job_qpca_dt150_cyc4_ghdev_np8.slurm with 2.5 h
# walltime so we don't lose results if dt=150 overruns the gh-dev cap.
#
# Same config as the gh-dev twin (operating point, dt=150, 80-min
# windows, 1h spinup). Different output subdir tag (_gh suffix) so the
# two results don't overwrite each other and we can compare elapsed
# times if both complete.
# --------------------------------------------------------
#SBATCH -J qpca_dt150_cyc4_gh
#SBATCH -A ADCIRC
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:30:00
#SBATCH -o qpca_dt150_cyc4_gh.%j.out
#SBATCH -e qpca_dt150_cyc4_gh.%j.err

set -uxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

OUT="idealized_inlet_qpca_endcf_vista_gh_w80_cyc4_infl105_dt150"
mkdir -p "$WORK/SWEMniCS/results/$OUT"

echo
echo "=== QPCA-EnDCF, dt=150 + cyc 4w at operating point (gh, 2.5h) ==="
mpirun -n 8 --bind-to core --map-by socket --timeout 9000 \
  python -u experiments/idealized_inlet_qpca_endcf.py \
  --method qpca \
  --vmax 15.0 --track-shift 5.0 \
  --dt 150.0 --nt-ramp 24 \
  --window-len 4 --obs-frequency 8 --nt-da 32 \
  --n-windows 4 \
  --ensemble-size 10 --ensemble-seed 2026 \
  --k-modes 1 --inflation 1.05 \
  --loc-radius 5000 \
  --obs-fraction 0.043 --obs-noise-level 0.01 \
  --background-error-std 0.10 \
  --output-subdir "$OUT" \
  || echo "[job] dt=150 cyc4 (gh) exited non-zero"

echo
echo "=== result ==="
ls -la "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true
cat "$WORK/SWEMniCS/results/$OUT/result_qpca_endcf.json" 2>&1 || true

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

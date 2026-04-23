#!/bin/bash
# First-win search — Run 2: 4D-Var at obs_fraction=0.02 (moderate density).
# Matched-config baseline for Runs 3 and 4 at the same obs density.
# Expected wall: ~80 min (4D-Var optimization; no Gram).
# -----------------------------------------------------------------------------
#SBATCH -J inlet_r2_4dvar020
#SBATCH -o inlet_r2_4dvar020.%j.out
#SBATCH -e inlet_r2_4dvar020.%j.err
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 02:00:00

set -euxo pipefail
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh

echo "=== provenance ==="
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

(while sleep 300; do echo "[watchdog $(date +%T)] $(free -h | grep Mem:)"; done) &
WPID=$!; trap "kill $WPID 2>/dev/null || true" EXIT

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.02 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

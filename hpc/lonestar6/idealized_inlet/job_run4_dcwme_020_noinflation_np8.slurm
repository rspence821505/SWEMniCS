#!/bin/bash
# First-win search — Run 4: DC-WME at obs_fraction=0.02 with Eq 38 inflation OFF.
# Matched-config comparison against Run 2 (4D-Var) and Run 3 (DC-WME + Eq 38 ON).
# Expected wall: ~80 min (no Gram; similar cost to Run 2's 4D-Var).
# -----------------------------------------------------------------------------
#SBATCH -J inlet_r4_dcwme020_noinf
#SBATCH -o inlet_r4_dcwme020_noinf.%j.out
#SBATCH -e inlet_r4_dcwme020_noinf.%j.err
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh

echo "=== provenance ==="
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

(while sleep 300; do echo "[watchdog $(date +%T)] $(free -h | grep Mem:)"; done) &
WPID=$!; trap "kill $WPID 2>/dev/null || true" EXIT

ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.02 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --no-eq38-inflation \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

#!/bin/bash
# np=8 direct sanity run — tiny idealized_inlet_da.py config to see if the
# production code path works at np=8. Bypasses the broken parity harness.
#
# Config matches the bisector tiny run (nt_ramp=4, nt_da=4, obs_frac=0.005,
# max_iter=1) for ~10-min wall.
# -----------------------------------------------------------------------------
#SBATCH -J inlet_np8_sanity
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 00:20:00
#SBATCH -o inlet_np8_sanity.%j.out
#SBATCH -e inlet_np8_sanity.%j.err

set -euxo pipefail

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh

echo "=== provenance ==="
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

(while sleep 120; do echo "[watchdog $(date +%T)] $(free -h | grep Mem:)"; done) &
WPID=$!; trap "kill $WPID 2>/dev/null || true" EXIT

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 4 --nt-da 4 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 1 --max-funcs 1 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

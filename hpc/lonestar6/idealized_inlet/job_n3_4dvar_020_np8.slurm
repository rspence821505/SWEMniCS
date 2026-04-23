#!/bin/bash
# Controlled MPI study — n3_4dvar_020 at np=8. No Eq 38 inflation.
# --------------------------------------------------------
#SBATCH -J inlet_n3_4dvar_020
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o inlet_n3_4dvar_020.%j.out
#SBATCH -e inlet_n3_4dvar_020.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

(while sleep 120; do echo "[watchdog $(date +%T)] $(free -h | grep Mem:)"; done) &
WPID=$!; trap "kill $WPID 2>/dev/null || true" EXIT

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.02 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
   \
   \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

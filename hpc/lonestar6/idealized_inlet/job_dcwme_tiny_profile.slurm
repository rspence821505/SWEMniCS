#!/bin/bash
# Minimal DC-WME run for profiling Step 7b (_compute_static_L_wme) hang.
# nt_ramp=4, nt_da=4 so the pre-Step-7b phases are fast and we quickly
# see the timing-instrumented prints from _compute_static_L_wme.
#
# Goal: find exactly which line hangs when B is TLM-inflated (non-uniform).
# Wall time budget is 30 min (should not need more than 15 at tiny scale).
# -----------------------------------------------------------------------------
#SBATCH -J inlet_dcwme_tiny
#SBATCH -o inlet_dcwme_tiny.%j.out
#SBATCH -e inlet_dcwme_tiny.%j.err
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 2                         # Phase 1: 2 MPI ranks
#SBATCH -t 00:30:00

set -euxo pipefail

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export PYTHONUNBUFFERED=1

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh

cd $WORK/SWEMniCS
git rev-parse HEAD

# Same DC-WME config as primary, just with nt_ramp=4 nt_da=4 to make
# truth build fast. Step 7b is what we want to profile.
ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 4 --nt-da 4 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 3 --max-funcs 3 \
  --predictability-gamma 0.1 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

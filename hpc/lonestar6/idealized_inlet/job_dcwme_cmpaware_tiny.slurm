#!/bin/bash
# Tiny component-aware TLM Eq 38 sanity check (~15 min on dev queue).
# Verifies new code path runs cleanly at np=2 before the full production job.
# -----------------------------------------------------------------------------
#SBATCH -J inlet_dcwme_cmptiny
#SBATCH -o inlet_dcwme_cmptiny.%j.out
#SBATCH -e inlet_dcwme_cmptiny.%j.err
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 00:30:00

set -euxo pipefail

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export PYTHONUNBUFFERED=1

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh

cd $WORK/SWEMniCS
git rev-parse HEAD

ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 4 --nt-da 4 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 3 --max-funcs 3 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --eq38-component-aware \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

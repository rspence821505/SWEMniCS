#!/bin/bash
# Vista: DC-WME inflation-ON (TLM Eq 38) at obs=0.02, np=8.
# --------------------------------------------------------
#SBATCH -J vi_dcwme_002
#SBATCH -A ADCIRC
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 03:00:00
#SBATCH -o vi_dcwme_002.%j.out
#SBATCH -e vi_dcwme_002.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

mpirun -n 8 python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.02 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --eq38-component-aware \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

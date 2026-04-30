#!/bin/bash
# Vista gg: DC-WME static obs=0.1, NO Eq 38 inflation, np=8 — rerun of LS6 N8
# with MPI-safe Newton failure path. Previous LS6 attempts hit 2h timeout
# at eval 8 (16.0%); expected to complete 15 evals in ~30-40 min.
# --------------------------------------------------------
#SBATCH -J vn8_dcwme_010
#SBATCH -A ADCIRC
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:30:00
#SBATCH -o vn8_dcwme_010.%j.out
#SBATCH -e vn8_dcwme_010.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

mpirun -n 8 --bind-to core --map-by socket --timeout 7000 python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.1 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --obs-correlation-length 1500 --predictability-gamma 0.1 \
  --no-eq38-inflation \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

#!/bin/bash
# Vista gg: 4D-Var obs=0.005, no inflation, np=8 — rerun of LS6 N1 (6.5%) on
# Vista for cross-cluster apples-to-apples vs vn2 (DC-WME 0.005).
# Both use bg_rmse=0.152898 (Vista perturbation).
# --------------------------------------------------------
#SBATCH -J vn1_4dvar_0005
#SBATCH -A ADCIRC
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:30:00
#SBATCH -o vn1_4dvar_0005.%j.out
#SBATCH -e vn1_4dvar_0005.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

mpirun -n 8 --bind-to core --map-by socket --timeout 7000 python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

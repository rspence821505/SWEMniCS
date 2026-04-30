#!/bin/bash
# Vista gg: 4D-Var obs=0.05, no inflation, np=8 — rerun of LS6 N5 with MPI-safe
# Newton failure path. Previous LS6 attempt hit 2h timeout at iter 5 (24.8%);
# expected to complete 15 evals cleanly in ~25-35 min.
# --------------------------------------------------------
#SBATCH -J vn5_4dvar_005_f5
#SBATCH -A ADCIRC
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:30:00
#SBATCH -o vn5_4dvar_005_f5.%j.out
#SBATCH -e vn5_4dvar_005_f5.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

mpirun -n 8 --bind-to core --map-by socket --timeout 7000 python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

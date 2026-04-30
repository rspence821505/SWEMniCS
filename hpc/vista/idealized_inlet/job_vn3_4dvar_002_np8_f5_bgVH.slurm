#!/bin/bash
# Vista gg: 4D-Var obs=0.02, no inflation, np=8 — rerun of LS6 N3 with MPI-safe
# Newton failure path. Previous LS6 attempts hit 2h timeout at eval 6 (13.4%);
# expected to complete 15 evals in ~20-30 min on Vista like 679097 did.
# --------------------------------------------------------
#SBATCH -J vn3_4dvar_002_f5_bgVH
#SBATCH -A ADCIRC
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:30:00
#SBATCH -o vn3_4dvar_002_f5_bgVH.%j.out
#SBATCH -e vn3_4dvar_002_f5_bgVH.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

mpirun -n 8 --bind-to core --map-by socket --timeout 7000 python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.02 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.2 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

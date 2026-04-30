#!/bin/bash
# Vista gh-dev: 4D-Var obs=0.1, no inflation (parallels LS6 N7). np=8.
# Alternate of LS6 job_n7_4dvar_010_np8.slurm to burn through the
# LS6 QOS=1 sequential queue.
# --------------------------------------------------------
#SBATCH -J vn7_4dvar_010
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o vn7_4dvar_010.%j.out
#SBATCH -e vn7_4dvar_010.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

mpirun -n 8 --bind-to core --map-by socket --timeout 7000 python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.1 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

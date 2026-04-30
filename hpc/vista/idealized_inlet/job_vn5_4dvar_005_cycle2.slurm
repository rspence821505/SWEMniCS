#!/bin/bash
# Vista gg: 4D-Var obs=0.05, freq=5, bg=0.02 — CYCLING DA test, n_windows=2.
# First validation of cycling implementation in idealized_inlet_da.py.
# Window 1 should match the single-window vn5_f5 result (~26.8%).
# Window 2 uses Window 1's analysis as background + truth marched 12 more steps.
# --------------------------------------------------------
#SBATCH -J vn5_4d_005_c2
#SBATCH -A ADCIRC
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:30:00
#SBATCH -o vn5_4d_005_c2.%j.out
#SBATCH -e vn5_4d_005_c2.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

mpirun -n 8 --bind-to core --map-by socket --timeout 7000 python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --n-windows 2 \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

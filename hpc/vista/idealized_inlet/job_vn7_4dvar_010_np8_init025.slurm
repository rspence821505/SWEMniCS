#!/bin/bash
# Vista gh-dev: 4D-Var obs=0.1, no inflation, np=8 — WITH TAO init-step 0.25.
# Third stabilization probe after step-cap alone failed identically (678950 vs
# 678961). Armijo's first trial at α=init=0.25 (default 1.0) should reduce
# BLMVM's 2nd iteration's first step by 4x.
#
# Prior results at obs=0.1:
#   678950 (no knobs):        eval 2 → 5.2% RMSE, eval 3 Newton fail (bit-exact)
#   678961 (stepmax=1.0):     eval 2 → 5.2% RMSE, eval 3 Newton fail (bit-exact)
#   THIS (init_step=0.25):    expected: smaller eval-3 step, hopefully survives
# --------------------------------------------------------
#SBATCH -J vn7_4dvar_010_i025
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o vn7_4dvar_010_i025.%j.out
#SBATCH -e vn7_4dvar_010_i025.%j.err

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
  --ls-init-step 0.25 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

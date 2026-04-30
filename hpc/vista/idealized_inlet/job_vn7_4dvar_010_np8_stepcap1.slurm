#!/bin/bash
# Vista gh-dev: 4D-Var obs=0.1, no inflation, np=8 — WITH TAO step cap.
# A/B test vs job_vn7_4dvar_010_np8.slurm (no step cap).
# Hypothesis: --ls-step-max 1.0 prevents BLMVM's 2nd iteration from
# extrapolating into the forward-Newton-failure region that crashed
# the previous obs=0.1 attempts at eval 3 (min h → -20 m).
# See docs/idealized_inlet_newton_eval3_failure_audit (pending).
# --------------------------------------------------------
#SBATCH -J vn7_4dvar_010_sc1
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o vn7_4dvar_010_sc1.%j.out
#SBATCH -e vn7_4dvar_010_sc1.%j.err

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
  --ls-step-max 1.0 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

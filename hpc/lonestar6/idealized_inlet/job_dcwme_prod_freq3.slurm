#!/bin/bash
# DC-WME production variant — same as job_dcwme_prod.slurm but obs_frequency=3
# (4 obs times instead of 3, denser sampling of the DA window).
#
# Motivation: 3100482 with obs_frequency=4 showed J descending cleanly but
# RMSE_truth decoupling from J (cost-vs-truth mis-alignment hypothesis). More
# obs times give 4D-Var / WME more data AND increase N in L_wme = I + (N/σ²)·K,
# strengthening the predictability subtraction term.
#
# Phase 1 topology: np=2 × 64 threads per plan §0.
# -----------------------------------------------------------------------------
#SBATCH -J inlet_dcwme_freq3
#SBATCH -o inlet_dcwme_freq3.%j.out
#SBATCH -e inlet_dcwme_freq3.%j.err
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 02:00:00

set -euxo pipefail

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh

echo "=== provenance ==="
module list 2>&1
pip show fenics-dolfinx petsc4py mpi4py | grep -E "^(Name|Version|Location)"
git -C $WORK/SWEMniCS rev-parse HEAD
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  SLURM_NTASKS=$SLURM_NTASKS"

cd $WORK/SWEMniCS

(while sleep 300; do
    echo "[watchdog $(date +%T)] $(free -h | grep Mem:)"
done) &
WATCHDOG_PID=$!
trap "kill $WATCHDOG_PID 2>/dev/null || true" EXIT

# Only change from job_dcwme_prod.slurm: --obs-frequency 3 (was 4)
ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 3 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize 2>&1 || true

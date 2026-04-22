#!/bin/bash
# DC-WME production variant — component-aware TLM Eq 38.
#
# Tests the hypothesis that DC-WME's static-kernel failure on the idealized
# inlet is partly due to TLM Eq 38 producing a single scalar σ_b² that
# doesn't distinguish h vs u/v predictability. With --eq38-component-aware
# we compute separate G_h and G_uv from the SAME adjoint vectors (no extra
# PDE work), derive σ_b²_h and σ_b²_uv independently, and inflate each
# block to its own bound.
#
# Config otherwise identical to job_dcwme_prod.slurm.
# Phase 1 topology (np=2 × 64 threads).
# -----------------------------------------------------------------------------
#SBATCH -J inlet_dcwme_cmpaware
#SBATCH -o inlet_dcwme_cmpaware.%j.out
#SBATCH -e inlet_dcwme_cmpaware.%j.err
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

cd $WORK/SWEMniCS

(while sleep 300; do
    echo "[watchdog $(date +%T)] $(free -h | grep Mem:)"
done) &
WATCHDOG_PID=$!
trap "kill $WATCHDOG_PID 2>/dev/null || true" EXIT

# Only change from job_dcwme_prod.slurm: --eq38-component-aware flag added.
ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --eq38-component-aware \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize 2>&1 || true

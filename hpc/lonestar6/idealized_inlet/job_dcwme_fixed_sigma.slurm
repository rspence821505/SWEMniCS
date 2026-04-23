#!/bin/bash
# DC-WME with FIXED per-component σ_b² (no TLM Eq 38 Gram).
# Isolates the effect of the σ_b²_uv magnitude on the optimization,
# holding σ_b²_h at the post-fix TLM Eq 38 value (0.01073).
#
#   σ_b²_h  = 0.01073  (matches post-fix 3102561 TLM result — reproduces h block)
#   σ_b²_uv = 0.8731   (user-specified; 5× larger than the 3102561 TLM uv bound
#                       of 0.1731 — probes whether more uv background freedom
#                       helps or hurts).
#
# Expected wall: ~55 min (skips the 20-min Gram computation).
# -----------------------------------------------------------------------------
#SBATCH -J inlet_dcwme_fixsig
#SBATCH -o inlet_dcwme_fixsig.%j.out
#SBATCH -e inlet_dcwme_fixsig.%j.err
#SBATCH -A ADCIRC
#SBATCH -p development
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

ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --fixed-sigma-b-sq-h 0.01073 \
  --fixed-sigma-b-sq-uv 0.8731 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize 2>&1 || true

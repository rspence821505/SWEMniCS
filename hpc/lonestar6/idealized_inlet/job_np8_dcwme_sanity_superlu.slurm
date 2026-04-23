#!/bin/bash
# Phase 1 stabilization — Option B: switch LU factor solver from MUMPS to
# SuperLU_DIST in the adjoint transpose solve. MUMPS was unstable at np=8
# on this problem's DG-mixed-element Jacobian (PC_FAILED; all 3 solver
# strategies diverged). SuperLU_DIST uses a different factorization
# algorithm and pivoting strategy.
# -----------------------------------------------------------------------------
#SBATCH -J inlet_np8_dcwme_sl
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 00:25:00
#SBATCH -o inlet_np8_dcwme_sl.%j.out
#SBATCH -e inlet_np8_dcwme_sl.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Option B: use SuperLU_DIST instead of MUMPS for the adjoint LU factor.
export SWE4DVAR_ADJOINT_LU_SOLVER=superlu_dist

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh

echo "=== provenance ==="
git -C $WORK/SWEMniCS rev-parse HEAD
echo "SWE4DVAR_ADJOINT_LU_SOLVER=$SWE4DVAR_ADJOINT_LU_SOLVER"
cd $WORK/SWEMniCS

(while sleep 120; do echo "[watchdog $(date +%T)] $(free -h | grep Mem:)"; done) &
WPID=$!; trap "kill $WPID 2>/dev/null || true" EXIT

ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 4 --nt-da 4 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --obs-correlation-length 1500 \
  --predictability-gamma 0.1 \
  --eq38-component-aware \
  --max-iterations 1 --max-funcs 1 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

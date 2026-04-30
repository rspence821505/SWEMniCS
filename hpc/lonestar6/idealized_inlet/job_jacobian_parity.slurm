#!/bin/bash
# Jacobian parity harness — direct algebraic comparison of stored vs
# recomputed J_n at selected timesteps. Cheap (~7 min) — runs ONE forward
# solve with both legacy storage and the new replay-metadata capture, then
# diffs J_stored vs J_recomp via JacobianReplayContext.
#
# Goal: pass parity (relative Frobenius diff < 1e-10) at a handful of
# selected steps before re-enabling recompute mode in the production
# adjoint path.
# --------------------------------------------------------
#SBATCH -J jac_parity
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:30:00
#SBATCH -o jac_parity.%j.out
#SBATCH -e jac_parity.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

# Capture full ghosted form-visible state at each timestep that produces J.
export SWE4DVAR_CAPTURE_REPLAY_META=1
# Keep recompute mode OFF — this harness uses store_jacobians=True so the
# legacy J_n stays available as ground truth.
export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=0

ibrun python -u experiments/jacobian_parity_harness.py \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --steps 1 2 3 6 \
  --rtol 1e-10 \
  --matvec-trials 3
RET=$?

echo
echo "=== exit code: $RET ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

exit $RET

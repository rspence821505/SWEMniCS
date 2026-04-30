#!/bin/bash
# Replay-at-capture diagnostic.
#
# Runs the existing jacobian_parity_harness forward solve with the
# immediate capture-time diff enabled. For every timestep that emits a
# stored J_n, we also reassemble J_n via JacobianReplayContext from the
# replay metadata captured the same instant, and log:
#   |J_s|, |J_r|, |J_s - J_r|, rel-Frobenius, optional matvec ratio.
#
# Acceptance: rel < ~1e-12 at every step → replay is correct in
# isolation, so any DA-time mismatch lives in how replay-J is wired into
# the backward sweep (sweep-KSP, J_reg, transpose-cache).
# If rel matches the parity-harness floor (~2.6e-04), replay itself is
# the bug.
#
# This job does NOT touch the adjoint; it is purely forward-side.
# --------------------------------------------------------
#SBATCH -J replay_capdiag
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:20:00
#SBATCH -o replay_capdiag.%j.out
#SBATCH -e replay_capdiag.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

# Capture replay metadata (required so capture-diag has anything to
# replay against).
export SWE4DVAR_CAPTURE_REPLAY_META=1
# Arm the immediate capture-time diff.
export SWE4DVAR_REPLAY_CAPTURE_DIAG=1
# Include the random-vector matvec ratio for added confidence.
export SWE4DVAR_REPLAY_CAPTURE_DIAG_MATVEC=1
# Keep stored J ON so we have ground truth on hand.
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

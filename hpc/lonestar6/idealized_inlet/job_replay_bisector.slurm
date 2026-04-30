#!/bin/bash
# Stored-vs-replay one-step adjoint bisector.
#
# Runs ONE DA window with stored Jacobians ON (so both branches are
# available) and the bisector armed inside _solve_transpose_system.
# At each backward step (or the chosen step), it solves
#   J_n^T λ_n = forcing
# twice — once with J_stored, once with J_replay — both via FRESH
# direct-LU KSPs. Logs Frobenius diff and lambda diff so we can
# localize whether a parity gap survives the production transpose-solve
# pathway.
#
# Crucially this job does NOT enable recompute mode. The production
# adjoint still uses J_stored; the replay branch is observation-only.
#
# Knobs:
#   SWE4DVAR_ADJOINT_REPLAY_BISECTOR_STEP=K     # only run at step K
#   SWE4DVAR_ADJOINT_REPLAY_BISECTOR_NO_REG=1   # skip dry-node + εI
# --------------------------------------------------------
#SBATCH -J replay_bisect
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:30:00
#SBATCH -o replay_bisect.%j.out
#SBATCH -e replay_bisect.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

# Replay metadata must be captured during the forward solve.
export SWE4DVAR_CAPTURE_REPLAY_META=1
# Production adjoint stays on stored-J; bisector is observational.
export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=0
# Arm the bisector. Without _STEP set, it fires at every backward step
# that has both stored J and replay metadata — fine for nt_da=6.
export SWE4DVAR_ADJOINT_REPLAY_BISECTOR=1
# Keep the regularization match between the two branches; set to 1 if
# we want a stricter purely-algebraic comparison.
export SWE4DVAR_ADJOINT_REPLAY_BISECTOR_NO_REG=0
# Established Phase C defaults. Sweep-KSP/cache do NOT touch the bisector
# (it always uses fresh KSPs), so leaving them on is safe.
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

RES_DIR=$WORK/SWEMniCS/results/idealized_inlet_da
rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

# Just enough optimization to trigger one adjoint solve. The bisector
# logs from the FIRST adjoint sweep are sufficient.
ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 1 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 1 --max-funcs 2 \
  --mem-limit-gb 240
RET=$?

echo
echo "=== exit code: $RET ==="
echo "Look for [replay-bisector] lines above for n, |J_s|, |J_r|,"
echo "  ||J_s - J_r||/|J_s|, |lam_s|, |lam_r|, ||lam_s - lam_r||/|lam_s|."
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

exit $RET

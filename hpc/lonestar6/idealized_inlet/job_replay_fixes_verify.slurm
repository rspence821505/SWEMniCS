#!/bin/bash
# Verification of the two replay fixes:
#   Bug 1 — JacobianReplayContext._load/_restore now re-evaluates
#           problem.forcing at the saved t (wind/pressure refresh).
#   Bug 2 — cost_functions recompute branch now disables sweep-KSP
#           and transpose-cache so per-step replay-J Mats aren't
#           held by stale persistent KSPs.
#
# Three back-to-back jobs (one sbatch -> one Python invocation each
# is cheaper than three sbatch). Inside this script we run:
#   1) Parity harness (checks Bug 1 directly: rel-Frobenius at steps
#      1, 2, 3, 6 should drop from ~2.6e-04 to FP noise).
#   2) Bisector (one DA window, stored-J ON, bisector armed; J-diff
#      and λ-diff should both collapse).
#   3) recJ_w0 (recompute mode end-to-end; should produce non-zero
#      gradient and analysis RMSE within ~1% of baseline 0.1079).
# --------------------------------------------------------
#SBATCH -J replay_fixes
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 01:30:00
#SBATCH -o replay_fixes.%j.out
#SBATCH -e replay_fixes.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

RES_DIR=$WORK/SWEMniCS/results/idealized_inlet_da

# ---- (1) Parity harness — direct algebraic check of Bug 1 fix ----
echo
echo "================ STAGE 1: parity harness ================"
export SWE4DVAR_CAPTURE_REPLAY_META=1
export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=0
unset SWE4DVAR_ADJOINT_REPLAY_BISECTOR
unset SWE4DVAR_REPLAY_CAPTURE_DIAG
ibrun python -u experiments/jacobian_parity_harness.py \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --steps 1 2 3 6 \
  --rtol 1e-10 \
  --matvec-trials 3 || true   # don't abort the script on parity failure

# ---- (2) Bisector — checks that backward-walk λ-diff also collapses ----
echo
echo "================ STAGE 2: bisector ================"
export SWE4DVAR_CAPTURE_REPLAY_META=1
export SWE4DVAR_ADJOINT_REPLAY_BISECTOR=1
export SWE4DVAR_ADJOINT_REPLAY_BISECTOR_NO_REG=0
export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=0
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1
rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 1 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 1 --max-funcs 2 \
  --mem-limit-gb 240 || true

# ---- (3) recJ_w0 — full end-to-end recompute mode ----
echo
echo "================ STAGE 3: recJ_w0 end-to-end ================"
unset SWE4DVAR_ADJOINT_REPLAY_BISECTOR
export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
# Note: recompute branch in cost_functions now forces SWEEP_KSP=0 and
# TRANSPOSE_CACHE=0 internally, so the explicit settings here are
# documentary only (visible in env-dump).
export SWE4DVAR_ADJOINT_SWEEP_KSP=0
export SWE4DVAR_ADJOINT_TRANSPOSE_CACHE=0
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_MALLOC_TRIM=1
export SWE4DVAR_EVAL_MEM_DIAG=1
export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_recJ_w0_${SLURM_JOB_ID}_rank{rank}.csv"
rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 1 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo
echo "================ recJ_w0 result JSON ================"
cat $RES_DIR/result_4dvar_N_A_cycling_w0-0.json 2>/dev/null || true
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

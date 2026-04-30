#!/bin/bash
# Single-window DA validation for recompute-Jacobians-via-JacobianReplayContext.
# Per the parity-first plan: this runs ONE DA window with recompute enabled
# and reports W0 RMSE. Compare against the documented baseline (bg=0.1454,
# analysis=0.1079, +25.79%).
#
# Acceptance:
#   * non-zero gradient at iter 0
#   * optimizer takes >1 cost-eval
#   * analysis RMSE within ~1% of baseline (parity at 2.6e-04 should give
#     this; if much worse, the integration into _solve_transpose_system
#     is the issue, not replay correctness)
# --------------------------------------------------------
#SBATCH -J recJ_w0
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:45:00
#SBATCH -o recJ_w0.%j.out
#SBATCH -e recJ_w0.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

# Recompute mode (forces SWE4DVAR_CAPTURE_REPLAY_META=1 inside cost_functions
# and skips legacy stored-J)
export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
# Established Phase C defaults
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1
# Eval-memory diagnostic for sanity checking
export SWE4DVAR_EVAL_MEM_DIAG=1
RES_DIR=$WORK/SWEMniCS/results/idealized_inlet_da
export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_recJ_w0_${SLURM_JOB_ID}_rank{rank}.csv"

# Wipe stale chain bg files so this is a true cold-start W0
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
echo "=== W0 result ==="
cat $RES_DIR/result_4dvar_N_A_cycling_w0-0.json 2>/dev/null || true
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

#!/bin/bash
# v=20 + WD DC-WME W2-only chain with ILU sub-PC.
# Hypothesis: 3181227's W2 Newton failures (code -11 = KSP_DIVERGED_PC_FAILED)
# happen because the bjacobi outer PC's sub-blocks use inner GMRES with NO
# preconditioner. At storm peak the sub-block matrices become ill-conditioned
# and inner GMRES can't converge in 100 iterations.
#
# Fix: set sub_pc_type=ilu + nonzero shift (handles small/zero diagonals).
# Wired through new env var SWE4DVAR_FORWARD_SUB_PC.
# --------------------------------------------------------
#SBATCH -J wd_v20_ilu_w2
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o wd_v20_dcwme_w2_ilu.%j.out
#SBATCH -e wd_v20_dcwme_w2_ilu.%j.err

set -uxo pipefail
shopt -s nullglob
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

export SWE4DVAR_NEWTON_RELAX=0.3
unset SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS
unset SWE4DVAR_CAPTURE_REPLAY_META
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

# NEW: ILU on bjacobi sub-blocks with nonzero diagonal shift
export SWE4DVAR_FORWARD_SUB_PC=ilu
export SWE4DVAR_FORWARD_SUB_SHIFT=NONZERO
export SWE4DVAR_FORWARD_SUB_KSP_MAX_IT=500

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
TAG="wd_v20_dcwme_w2_ilupc"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 0 \
  --track-duration-s 43200 \
  --dt 150 \
  --nt-ramp 96 --nt-da 24 \
  --n-windows 1 --start-window 2 --chain-mode \
  --initial-bg-file "$RES_DIR/chain_bg_after_w1_rank{rank}.npy" \
  --obs-fraction 0.05 --obs-frequency 6 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --smooth-length 1500 \
  --max-iterations 3 --max-funcs 3 \
  --fixed-sigma-b-sq-h 0.01 --fixed-sigma-b-sq-uv 0.15 \
  --wd --wd-alpha 1.5 \
  --mem-limit-gb 240
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== v=20 + WD DC-WME W2 + ILU sub-PC ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}') for w in d['windows']]" 2>/dev/null
done
echo
echo "=== Newton failure summary ==="
grep -c 'Newton solver FAILED' /work/08398/tg876971/ls6/SWEMniCS/wd_v20_dcwme_w2_ilu.${SLURM_JOB_ID}.out 2>/dev/null || echo "0"
grep 'code -' /work/08398/tg876971/ls6/SWEMniCS/wd_v20_dcwme_w2_ilu.${SLURM_JOB_ID}.out 2>/dev/null | sort | uniq -c | head -5
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

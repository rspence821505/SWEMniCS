#!/bin/bash
# 10w slow-ramp follow-up to 3w diagnostic (3139864).
#
# 3w_slow showed promising eval-2 descent under slow ramp (12% drop
# vs iter14's 1500x blow-up). This run extends to 10 windows so we
# cover both the buildup phase AND the storm peak (peak now hits
# around W8-W9 with track-duration-s=157000, instead of W4 with
# 78500). Bumps max-funcs from 3 -> 5 because 3w_slow's eval #3 was
# rejected leaving the optimizer no room to recover.
#
# Goal: confirm whether slow ramp breaks the W3 ~0.29 cycling
# ceiling that iter13/iter14/obs30/5w_n03 all hit. If yes:
# slower forcing rate keeps 4D-Var inside the linearization
# radius. If no: cycling drift is unrelated to ramp rate.
#
# 8h walltime: estimate ~30min/window after W0 stabilizes
# (10 * ~30 + W0 setup ~60min ~= 6h baseline; 8h gives 33% margin).
# --------------------------------------------------------
#SBATCH -J 10w_slow
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 08:00:00
#SBATCH -o 10w_slow_ls6.%j.out
#SBATCH -e 10w_slow_ls6.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

export SWE4DVAR_NEWTON_RELAX=0.3
export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_10w_slow_ls6_${SLURM_JOB_ID}.csv"
rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 7 --track-shift 0 \
  --track-duration-s 157000 \
  --dt 300 \
  --nt-ramp 48 --nt-da 12 \
  --n-windows 10 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 6 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --smooth-length 1500 \
  --max-iterations 5 --max-funcs 5 \
  --mem-limit-gb 240
RET=$?

echo
echo "=== Per-window results ==="
for j in $RES_DIR/result_4dvar_N_A_cycling_w*-*.json; do
  [[ -f "$j" ]] || continue
  echo "  $(basename "$j")"
  python3 -c "import json; d=json.load(open('$j'));
[print(f\"    W{w['window']-1}: bg={w['bg_rmse']:.4f}  an={w['analysis_rmse']:.4f}  imp={w['improvement_pct']:+.2f}%  evals={w['n_func_evals']}\") for w in d['windows']]" 2>/dev/null
done
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

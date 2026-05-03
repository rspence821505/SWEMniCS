#!/bin/bash
# 3w slow-ramp diagnostic (LS6 development queue, fast turnaround).
#
# Diagnostic: iter14 baseline had W1->W2 bg_rmse jump 0.133->0.221
# (+66% in 1h). Hypothesis: the per-window forcing rate exceeds the
# 4D-Var tangent-linear linearization radius, so the gradient stops
# being a useful descent direction even though Newton converges. This
# run halves the wind intensification rate (track-duration-s 78500 ->
# 157000, i.e. Vmax=7 reached over ~8h instead of ~4h).
#
# Trimmed to 3 windows to fit dev queue (2h walltime). The diagnostic
# is just whether W1->W2 jump shrinks under slower forcing. If yes:
# follow up with full 10-window run on normal queue. If no: IC
# sensitivity / mesh-scale features dominate.
# --------------------------------------------------------
#SBATCH -J 3w_slow
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -o 3w_slow_ls6.%j.out
#SBATCH -e 3w_slow_ls6.%j.err

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
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_3w_slow_ls6_${SLURM_JOB_ID}.csv"
rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 7 --track-shift 0 \
  --track-duration-s 157000 \
  --dt 300 \
  --nt-ramp 48 --nt-da 12 \
  --n-windows 3 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 6 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --smooth-length 1500 \
  --max-iterations 3 --max-funcs 3 \
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

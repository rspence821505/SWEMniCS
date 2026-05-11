#!/bin/bash
# 3w 4D-Var with fixed σ_b² matching cmpawr DC-WME (3154342).
#
# Apples-to-apples control: 3154342 (DC-WME cmpawr) produced
# σ_b²_h=0.0209, σ_b²_uv=1630 (W0 values) and got W0 imp +1.86%.
# 4D-Var with default uniform σ_b² (3144074) got W0 imp +9.87%.
# This run pins 4D-Var's σ_b² to the DC-WME-derived values via
# --fixed-sigma-b-sq-h / --fixed-sigma-b-sq-uv, bypassing TLM Gram.
#
# Isolates the question: is DC-WME's 5x performance deficit from
#   (a) the inflated σ_b² (especially σ_b²_uv = 1630), or
#   (b) the L_wme machinery (predictability reweighting)?
# If 4D-Var with same σ_b² gets ~+1.86% like cmpawr: deficit is from
# inflation. If it gets back to ~+9.87%: deficit is from L_wme.
#
# All other config matches cmpawr: max-funcs=3, slow ramp, dt=300.
# --------------------------------------------------------
#SBATCH -J 3w_4dvfix
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH -o 3w_4dvfix_ls6.%j.out
#SBATCH -e 3w_4dvfix_ls6.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

export SWE4DVAR_NEWTON_RELAX=0.3
# Recompute-Jacobians-in-adjoint disabled for DC-WME (validated by
# 3150948 smoke test): Eq 38 TLM adjoint can't honor the replay-meta
# capture path on the truth-solver code path. Using stored Jacobians
# instead — ~1-2 GB more memory per window, but pipeline works
# end-to-end (Step 7a Gram cond 4-5.6 vs Shinnecock 1.0).
unset SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS
unset SWE4DVAR_CAPTURE_REPLAY_META
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_3w_4dvfix_ls6_${SLURM_JOB_ID}.csv"
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
  --fixed-sigma-b-sq-h 0.0209 \
  --fixed-sigma-b-sq-uv 1630.0 \
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

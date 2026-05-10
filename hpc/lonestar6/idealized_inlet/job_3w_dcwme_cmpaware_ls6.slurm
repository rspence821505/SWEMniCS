#!/bin/bash
# 3w static DC-WME with --eq38-component-aware (separate σ_b² for h vs uv).
#
# Follow-up to 3151680 (3w_dcwme uniform σ_b²): single σ_b² = 0.0172
# applied uniformly to all 35,196 inflated DOFs gave +1.64% W0 imp vs
# 4D-Var's +9.87% (6x worse). Hypothesis: the WSE (h) and velocity (uv)
# components have different predictability spectra, and forcing one
# σ_b² across both wastes information. Component-aware Eq 38 computes
# σ_b²_h and σ_b²_uv independently and inflates each block separately.
#
# Same other config as 3151680 (slow ramp, dt=300, nt-da=12, vmax=7).
# 4h walltime: 3151680 took 2:29, component-aware adds maybe 30-60s
# of Gram-matrix-on-subblocks overhead at Step 7a.
#
# Pass: σ_b²_h and σ_b²_uv differ materially (>2x); DC-WME W0 imp
# closes the gap to 4D-Var (>3%).
# --------------------------------------------------------
#SBATCH -J 3w_cmpawr
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 04:00:00
#SBATCH -o 3w_cmpawr_ls6.%j.out
#SBATCH -e 3w_cmpawr_ls6.%j.err

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
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_3w_cmpawr_ls6_${SLURM_JOB_ID}.csv"
rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 7 --track-shift 0 \
  --track-duration-s 157000 \
  --dt 300 \
  --nt-ramp 48 --nt-da 12 \
  --n-windows 3 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 6 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --smooth-length 1500 \
  --max-iterations 3 --max-funcs 3 \
  --eq38-component-aware \
  --mem-limit-gb 240
RET=$?

echo
echo "=== Per-window results ==="
for j in $RES_DIR/result_dcwme_*cycling_w*-*.json; do
  [[ -f "$j" ]] || continue
  echo "  $(basename "$j")"
  python3 -c "import json; d=json.load(open('$j'));
[print(f\"    W{w['window']-1}: bg={w['bg_rmse']:.4f}  an={w['analysis_rmse']:.4f}  imp={w['improvement_pct']:+.2f}%  evals={w['n_func_evals']}\") for w in d['windows']]" 2>/dev/null
done
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

#!/bin/bash
# DC-WME W0 probe #2: TLM-Gram-derived background covariance B.
# Drops --fixed-sigma-b-sq-h/uv to activate the TLM Gram path
# (_compute_eq38_from_tlm at idealized_inlet_da.py:902). This uses the
# tangent-linear adjoint sweep on the truth trajectory to estimate a
# flow-dependent σ_b², replacing the hand-picked scalar bypass. The
# resulting B carries genuine model-dynamical structure that L_wme can
# (in principle) pick up via H·B·H^T.
#
# Otherwise identical to the best previous DC-WME setting
# (γ=1.0, obs=0.10, mf=5, 1h window, TS=15).
# --------------------------------------------------------
#SBATCH -J thesis_dcw_tlmgram
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o thesis_dcwme_tlmgram_dev.%j.out
#SBATCH -e thesis_dcwme_tlmgram_dev.%j.err

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
export SWE4DVAR_FORWARD_NEWTON_REUSE=0
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_tlmgram_dev
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_tlmgram_dev"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

# NOTE: --fixed-sigma-b-sq-h/uv are OMITTED on purpose. With them absent
# and method=dcwme_static (so truth_jacobians ARE stored), Step 7a takes
# the TLM Gram branch at idealized_inlet_da.py:902.
bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 1 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 96 \
  --nt-da 24 \
  --method dcwme_static \
  --launcher ibrun \
  --extra "--track-shift 15 --track-duration-s 68400 --dt 150 --obs-fraction 0.10 --obs-frequency 6 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --wd --wd-alpha 1.5 --predictability-gamma 1.0 --eq38-component-aware --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME W0: TLM-Gram B (γ=1.0 dense-obs mf=5, no σ_b² bypass) ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== TLM Gram step + L_wme spectrum (check for non-flat ratio) ==="
grep -E 'Eq 38 via TLM|sigma_b_sq|\[L_wme spectrum\]' /work/08398/tg876971/ls6/SWEMniCS/thesis_dcwme_tlmgram_dev.${SLURM_JOB_ID}.out 2>/dev/null | head -8
echo
echo "=== Reference: γ=1.0 fixed-σ_b² W0 imp=+0.34% (3198818) ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

#!/bin/bash
# Smoke test for the new bg-feasibility preflight + interpolated-seed
# recovery ladder (commit 065db23) AND the new forward-evolve
# magnitude bound on the NaN guard (commit 0xx).
#
# Config picked to maximize the chance the recovery ladder actually
# fires: v=20 storm peak window forcing + track-shift model error +
# DC-WME cycling-stall. Late windows are the most likely to hit
# infeasible-bg.
#
# Pass criteria:
#   - [bg-feasibility] lines present, coherent per-window
#   - Either a window rescues via stricter-retry or seed (proves the
#     ladder works end-to-end) OR all windows are feasible (no regression)
#   - Cycling-aggregate JSON includes bg_status + bg_recovery_alpha
#   - No "TAO converged after 0 iterations" + non-zero gradient mismatch
# --------------------------------------------------------
#SBATCH -J bgfeas_smoke
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o bgfeas_smoke_v20_ts15.%j.out
#SBATCH -e bgfeas_smoke_v20_ts15.%j.err

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
# Validated DC-WME cycling fixes
export SWE4DVAR_FORWARD_NEWTON_REUSE=0
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

export SWE4DVAR_OUTPUT_SUBDIR=bgfeas_smoke_v20_ts15
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="bgfeas_smoke_v20_ts15_dcwme"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 3 \
  --np 8 \
  --vmax 20 \
  --nt-ramp 96 \
  --nt-da 24 \
  --method dcwme_static \
  --launcher ibrun \
  --extra "--track-shift 15 --track-duration-s 54000 --dt 150 --obs-fraction 0.05 --obs-frequency 6 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 3 --max-funcs 3 --fixed-sigma-b-sq-h 0.01 --fixed-sigma-b-sq-uv 0.15 --wd --wd-alpha 1.5 --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== bg-feasibility smoke: v=20 + TS=15 DC-WME 3w (chain+REUSE0) ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}  alpha={w.get(\"bg_recovery_alpha\")}') for w in d['windows']]" 2>/dev/null
done
echo
echo "=== bg-feasibility log lines per window ==="
grep '\[bg-feasibility' /work/08398/tg876971/ls6/SWEMniCS/bgfeas_smoke_v20_ts15.${SLURM_JOB_ID}.out 2>/dev/null | head -30 || true
echo
echo -n 'Hang check: '; grep -ch 'Error allocating' /work/08398/tg876971/ls6/SWEMniCS/bgfeas_smoke_v20_ts15.${SLURM_JOB_ID}.out 2>/dev/null || echo 0
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

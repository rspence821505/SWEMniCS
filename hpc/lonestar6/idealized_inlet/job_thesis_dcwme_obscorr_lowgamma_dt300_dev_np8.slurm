#!/bin/bash
# Re-run of 3200669 with dt=300 instead of dt=150.
# Identical config except all timestep counts halved to preserve
# physical durations:
#   dt=300, nt-ramp=48 (4h ramp), nt-da=12 (1h window),
#   obs-frequency=3 (every 15 min = 3 × 300 s; same cadence).
# Single W0 on dev queue.
# --------------------------------------------------------
#SBATCH -J thesis_dcw_dt300
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o thesis_dcwme_obscorr_lowgamma_dt300_dev.%j.out
#SBATCH -e thesis_dcwme_obscorr_lowgamma_dt300_dev.%j.err

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

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_obscorr_lowgamma_dt300_dev
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_obscorr_lowgamma_dt300_dev"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 1 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 48 \
  --nt-da 12 \
  --method dcwme_static \
  --launcher ibrun \
  --extra "--track-shift 15 --track-duration-s 68400 --dt 300 --obs-fraction 0.10 --obs-frequency 3 --obs-noise-level 0.01 --obs-correlation-length 2000 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --fixed-sigma-b-sq-h 0.01 --fixed-sigma-b-sq-uv 0.15 --wd --wd-alpha 1.5 --predictability-gamma 0.0001 --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME W0 dt=300: obs-corr=2000 + γ=0.0001 (full natural spectrum) ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== L_wme spectrum (compare to dt=150 ratio=3010) ==="
grep -E '\[L_wme spectrum\]|natural|Static adaptive' /work/08398/tg876971/ls6/SWEMniCS/thesis_dcwme_obscorr_lowgamma_dt300_dev.${SLURM_JOB_ID}.out 2>/dev/null | head -5
echo
echo "=== Reference dt=150 same config: W0 imp=+0.39% (3200669) ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

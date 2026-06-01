#!/bin/bash
# DC-WME W0 probe #3: obs-correlation-length=2000 m + γ=0.0001
# (preserves natural L_wme spectrum instead of floor-flattening it).
#
# From 3200432's log:
#   raw L_wme spectrum  : λ_max=3012, λ_min=1.0, ratio=3010
#   γ=1.0 → γ_eff=3012  → 1162/1163 modes floored (back to flat = degenerate)
#
# γ=0.0001 → γ_eff = 0.30 < λ_min = 1.0 → 0 modes floored.
# The full natural spectrum is preserved and DC-WME can actually
# discount in the under-predicted directions L_wme exposes.
#
# Same single-window W0 setup as 3200432 otherwise.
# --------------------------------------------------------
#SBATCH -J thesis_dcw_obsg4
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o thesis_dcwme_obscorr_lowgamma_dev.%j.out
#SBATCH -e thesis_dcwme_obscorr_lowgamma_dev.%j.err

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

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_obscorr_lowgamma_dev
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_obscorr_lowgamma_dev"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 1 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 96 \
  --nt-da 24 \
  --method dcwme_static \
  --launcher ibrun \
  --extra "--track-shift 15 --track-duration-s 68400 --dt 150 --obs-fraction 0.10 --obs-frequency 6 --obs-noise-level 0.01 --obs-correlation-length 2000 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --fixed-sigma-b-sq-h 0.01 --fixed-sigma-b-sq-uv 0.15 --wd --wd-alpha 1.5 --predictability-gamma 0.0001 --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME W0: obs-corr=2000 + γ=0.0001 (preserves natural spectrum) ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== L_wme spectrum + floor count (want 0/1163 floored) ==="
grep -E '\[L_wme spectrum\]|natural|Static adaptive' /work/08398/tg876971/ls6/SWEMniCS/thesis_dcwme_obscorr_lowgamma_dev.${SLURM_JOB_ID}.out 2>/dev/null | head -5
echo
echo "=== Reference: obs-corr=2000 + γ=1.0 W0 = ? (3200432), 4D-Var=+10.10% ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

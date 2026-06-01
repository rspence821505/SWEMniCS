#!/bin/bash
# DC-WME W0 probe #1: spatially-correlated observation term.
# Adds --obs-correlation-length 2000 (m) on top of the best previous
# DC-WME config (γ=1.0, obs=0.10, mf=5, 1h windows). Hypothesis: a
# non-trivial obs correlation length introduces directional structure
# into L_wme = H·B·H^T (currently flat-spectrum because B is
# diagonally bypassed via --fixed-sigma-b-sq-h/uv).
# --------------------------------------------------------
#SBATCH -J thesis_dcw_obscorr
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o thesis_dcwme_obscorr_dev.%j.out
#SBATCH -e thesis_dcwme_obscorr_dev.%j.err

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

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_obscorr_dev
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_obscorr_dev"
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
  --extra "--track-shift 15 --track-duration-s 68400 --dt 150 --obs-fraction 0.10 --obs-frequency 6 --obs-noise-level 0.01 --obs-correlation-length 2000 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --fixed-sigma-b-sq-h 0.01 --fixed-sigma-b-sq-uv 0.15 --wd --wd-alpha 1.5 --predictability-gamma 1.0 --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME W0: obs-correlation-length=2000 m (γ=1.0 dense-obs mf=5) ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== L_wme spectrum (look for non-flat ratio) ==="
grep '\[L_wme spectrum\]' /work/08398/tg876971/ls6/SWEMniCS/thesis_dcwme_obscorr_dev.${SLURM_JOB_ID}.out 2>/dev/null | head -3
echo
echo "=== Reference: γ=1.0 no-obs-corr W0 imp=+0.34% (3198818) ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

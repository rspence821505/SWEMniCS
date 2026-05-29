#!/bin/bash
# MODEL-ERROR test: DC-WME v=10 + WD + track-shift=15km, 3-window dev probe.
# Companion to the 4D-Var track-shift run. This is THE test of whether
# DC-WME's data-consistency machinery helps when there's genuine model
# error (vs the inverse-crime no-track-shift runs where it was flat).
#
# Uses chain-in-alloc + REUSE=0 (the validated DC-WME cycling fix).
# Config matches no-track-shift DC-WME baseline 3192942 EXCEPT track-shift.
# Dev 2h cap: expect W0-W1 clean (the meaningful comparison windows).
# --------------------------------------------------------
#SBATCH -J ts15_dcw_v10
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o ts15_v10_dcwme_chain.%j.out
#SBATCH -e ts15_v10_dcwme_chain.%j.err

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
# DC-WME cycling fix: fresh MUMPS factor per Newton call
export SWE4DVAR_FORWARD_NEWTON_REUSE=0
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

export SWE4DVAR_OUTPUT_SUBDIR=3w_ts15_v10_dcwme
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="3w_ts15_v10_dcwme"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 3 \
  --np 8 \
  --vmax 10 \
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
echo "=== v=10 + WD + track-shift=15km DC-WME 3w (model error, chain+REUSE0) ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}') for w in d['windows']]" 2>/dev/null
done
echo "=== Reference (no track-shift, 3192942): W0 +1.19%, W1 +0.00%, W2 +0.05% ==="
echo -n 'Hang check: '; grep -ch 'Error allocating' /work/08398/tg876971/ls6/SWEMniCS/ts15_v10_dcwme_chain.${SLURM_JOB_ID}.out 2>/dev/null || echo 0
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

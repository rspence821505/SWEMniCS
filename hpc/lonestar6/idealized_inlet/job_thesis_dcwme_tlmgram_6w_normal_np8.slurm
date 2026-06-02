#!/bin/bash
# Six-window cycling extension of 3201132 (the +3.36% TLM-Gram-B + dt=300
# result). Full pre-landfall surge cycle for DC-WME using the best
# single-W0 configuration measured to date.
#
# Same config as 3201132 except --n-windows 6. dt=300 nearly halves the
# per-window cost vs dt=150, so 12h walltime gives ~2h/window margin.
# --------------------------------------------------------
#SBATCH -J thesis_dcw_tlmgram_dt300_6w
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH -o thesis_dcwme_tlmgram_dt300_6w_normal.%j.out
#SBATCH -e thesis_dcwme_tlmgram_dt300_6w_normal.%j.err

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

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_tlmgram_dt300_6w_normal
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_tlmgram_dt300_6w_normal"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

# NOTE: --fixed-sigma-b-sq-h/uv are OMITTED on purpose. With them absent
# and method=dcwme_static (so truth_jacobians ARE stored), Step 7a takes
# the TLM Gram branch and produces a flow-dependent σ_b² instead of
# the diagonal scalar bypass used in the other thesis runs.
bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 6 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 48 \
  --nt-da 12 \
  --method dcwme_static \
  --launcher ibrun \
  --extra "--track-shift 15 --track-duration-s 68400 --dt 300 --obs-fraction 0.10 --obs-frequency 3 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --wd --wd-alpha 1.5 --predictability-gamma 1.0 --eq38-component-aware --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME 6-window cycling: TLM-Gram B + γ=1.0 + dt=300 (production thesis run) ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo
echo "=== Reference: dt=300 TLM-Gram W0 single-window = +3.36% (3201132) ==="
echo -n 'Hang check: '; grep -ch 'Error allocating' /work/08398/tg876971/ls6/SWEMniCS/thesis_dcwme_tlmgram_dt300_6w_normal.${SLURM_JOB_ID}.out 2>/dev/null || echo 0
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

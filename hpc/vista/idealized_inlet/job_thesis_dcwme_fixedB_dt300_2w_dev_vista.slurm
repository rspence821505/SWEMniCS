#!/bin/bash
# DC-WME 2-window cycling, dt=300, FIXED-B (non-TLM), Vista gh-dev.
# Replaces cancelled 733068 (dt=600 variant) with dt=300 so we get
# matched dt-coverage at dev cadence: dt=300 + dt=600 fixed-B pair.
#
# Same fixed σ_b² as 733066/733068 (σ²_h=5.0e-02, σ²_uv=2.0e+03) and
# R constant via eeb6332. dt=300 nt-ramp=48, nt-da=12, obs-frequency=3.
# --------------------------------------------------------
#SBATCH -J dcw_fixedB_dt300_2w_dev_v
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o thesis_dcwme_fixedB_dt300_2w_dev_vista.%j.out
#SBATCH -e thesis_dcwme_fixedB_dt300_2w_dev_vista.%j.err

set -euxo pipefail
shopt -s nullglob
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

export SWE4DVAR_NEWTON_RELAX=0.3
unset SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS
unset SWE4DVAR_CAPTURE_REPLAY_META
export SWE4DVAR_FORWARD_NEWTON_REUSE=0
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_fixedB_dt300_2w_dev_vista
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_fixedB_dt300_2w_dev_vista"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

# Fixed-B (TLM-Gram bypassed): σ²_h=5.0e-02, σ²_uv=2.0e+03.
# R constant via eeb6332 (obs_noise_level absolute, not fractional).
bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 2 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 48 \
  --nt-da 12 \
  --method dcwme_static \
  --launcher mpirun \
  --extra "--track-shift 15 --track-duration-s 68400 --dt 300 --obs-fraction 0.10 --obs-frequency 3 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --wd --wd-alpha 1.5 --fixed-sigma-b-sq-h 5.0e-02 --fixed-sigma-b-sq-uv 2.0e+03 --predictability-gamma 1.0 --eq38-component-aware --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME 2w dev cycling (Vista): FIXED-B (non-TLM) + dt=300 ==="
echo "=== σ²_h=5.0e-02, σ²_uv=2.0e+03, R constant via eeb6332 ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

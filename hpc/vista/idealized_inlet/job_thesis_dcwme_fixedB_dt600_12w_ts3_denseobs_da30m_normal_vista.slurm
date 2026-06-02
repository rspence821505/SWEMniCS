#!/bin/bash
# DC-WME 12-window cycling, dt=600, 30-min DA, ts=3, dense obs, vmax=10.
# Extension of 733911 (which reached W4 with positive cycling) to 12
# windows so the 30-min strategy covers the FULL 6-hour storm cycle
# through landfall.
#
# Storm peak lies at hour ~9.5 = end of W11 (in 30-min cycle terms).
# This is the first run designed to test whether 30-min cycling
# survives the storm peak.
#
# Walltime estimate: ~15 min/window × 12 = 3h. Request 6h for margin.
# --------------------------------------------------------
#SBATCH -J dcw_fixedB_dt600_12w_ts3_da30m_v
#SBATCH -A ADCIRC
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 06:00:00
#SBATCH -o thesis_dcwme_fixedB_dt600_12w_ts3_denseobs_da30m_vista.%j.out
#SBATCH -e thesis_dcwme_fixedB_dt600_12w_ts3_denseobs_da30m_vista.%j.err

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

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_fixedB_dt600_12w_ts3_denseobs_da30m_vista
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_fixedB_dt600_12w_ts3_denseobs_da30m_vista"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

# Same as 733911 EXCEPT --nwin 12 (was 6).
bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 12 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 24 \
  --nt-da 3 \
  --method dcwme_static \
  --launcher mpirun \
  --extra "--track-shift 3 --track-duration-s 68400 --dt 600 --obs-fraction 0.20 --obs-frequency 1 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --wd --wd-alpha 1.5 --fixed-sigma-b-sq-h 5.0e-02 --fixed-sigma-b-sq-uv 2.0e+03 --predictability-gamma 1.0 --eq38-component-aware --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME 12w cycling (Vista): fixed-B + dt=600 + 30-min DA + ts=3 + dense obs ==="
echo "=== Storm peak at W11. Test of whether 30-min cycling survives the peak. ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== Reference 733911 (6w, pre-peak): W0-W4 all positive, W5/W6 pending ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

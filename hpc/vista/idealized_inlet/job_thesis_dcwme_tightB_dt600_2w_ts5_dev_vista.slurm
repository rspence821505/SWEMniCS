#!/bin/bash
# DC-WME 2-window cycling, dt=600, ts=5, 1h DA, TIGHT fixed-B.
# Combo cell: combines the two best individual mitigations so far —
#   - track-shift 5 (vs 15): cut forecast amp by 37% (validated 733124)
#   - tight B σ²_h=5e-3, σ²_uv=2e+2 (vs 5e-2/2e+3): cut forecast amp
#                                                   by 19% (validated 733230)
#
# If both effects are independent, combined forecast amplification
# should drop to ~1.7× (vs 3.4× baseline) and W1 improvement should
# go meaningfully positive — first sustained cycling configuration.
#
# Vista gh-dev. 2 windows, 1h DA each.
# --------------------------------------------------------
#SBATCH -J dcw_tightB_dt600_2w_ts5_dev_v
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o thesis_dcwme_tightB_dt600_2w_ts5_dev_vista.%j.out
#SBATCH -e thesis_dcwme_tightB_dt600_2w_ts5_dev_vista.%j.err

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

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_tightB_dt600_2w_ts5_dev_vista
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_tightB_dt600_2w_ts5_dev_vista"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 2 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 24 \
  --nt-da 6 \
  --method dcwme_static \
  --launcher mpirun \
  --extra "--track-shift 5 --track-duration-s 68400 --dt 600 --obs-fraction 0.10 --obs-frequency 2 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --wd --wd-alpha 1.5 --fixed-sigma-b-sq-h 5.0e-03 --fixed-sigma-b-sq-uv 2.0e+02 --predictability-gamma 1.0 --eq38-component-aware --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME 2w dev cycling (Vista): TIGHT-B + dt=600 + ts=5 + 1h DA ==="
echo "=== σ²_h=5.0e-03, σ²_uv=2.0e+02 (combo of ts=5 + tight-B) ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== References ==="
echo "  733066 (orig-B, ts=15): W0 +33.60% W1 -1.14%  amp=3.4×"
echo "  733124 (orig-B, ts=5):  W0 +35.99% W1 +0.49%  amp=2.15×"
echo "  733230 (tight-B,ts=15): W0 +20.52% W1 -0.17%  amp=2.74×"
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

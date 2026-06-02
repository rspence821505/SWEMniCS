#!/bin/bash
# DC-WME 3-window cycling, dt=600, FIXED-B, track-shift=5, 2-hour DA windows.
# Vista gh production. Combines both diagnosed mitigations from the
# 733066/733124/3202296 series:
#   - track-shift 5  (vs 15): cuts forecast-step model-error amplification
#                              by ~37% (validated by 733124)
#   - nt-da 12 (2h)  (vs 6/1h): doubles obs per window (7 vs 4) and gives
#                              DA visibility into longer-horizon dynamics
#
# If W1 improvement comes in clearly positive (≥+5%), this is the first
# cycling configuration with sustained DA gains in this thesis.
#
# Reference matrix:
#   733066 (ts=15, da=1h): W0 +33.60%, W1 -1.14%
#   733124 (ts=5,  da=1h): W0 +35.99%, W1 +0.49%
#   3202296 (ts=15, da=2h): in progress
#   THIS   (ts=5,  da=2h): expected best
#
# Per-window cost ~2× the 1h-window case due to longer forward solves;
# Vista GH should still do 3 windows in <90 min so 4h walltime is generous.
# --------------------------------------------------------
#SBATCH -J dcw_fixedB_dt600_3w_ts5_da2h_v
#SBATCH -A ADCIRC
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 04:00:00
#SBATCH -o thesis_dcwme_fixedB_dt600_3w_ts5_da2h_vista.%j.out
#SBATCH -e thesis_dcwme_fixedB_dt600_3w_ts5_da2h_vista.%j.err

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

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_fixedB_dt600_3w_ts5_da2h_vista
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_fixedB_dt600_3w_ts5_da2h_vista"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

# Same fixed-B + R-const as 733124, but --nt-da 12 (2h DA windows).
bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 3 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 24 \
  --nt-da 12 \
  --method dcwme_static \
  --launcher mpirun \
  --extra "--track-shift 5 --track-duration-s 68400 --dt 600 --obs-fraction 0.10 --obs-frequency 2 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --wd --wd-alpha 1.5 --fixed-sigma-b-sq-h 5.0e-02 --fixed-sigma-b-sq-uv 2.0e+03 --predictability-gamma 1.0 --eq38-component-aware --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME 3w cycling (Vista): FIXED-B + dt=600 + ts=5 + 2h DA ==="
echo "=== σ²_h=5.0e-02, σ²_uv=2.0e+03, R constant via eeb6332 ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== Reference: 733066 ts=15/da=1h: W0 +33.60% W1 -1.14% ==="
echo "=== Reference: 733124 ts=5/da=1h:  W0 +35.99% W1 +0.49% ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

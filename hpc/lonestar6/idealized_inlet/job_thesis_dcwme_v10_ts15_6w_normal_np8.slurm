#!/bin/bash
# THESIS RUN: DC-WME pre-landfall surge nowcast cycle (6h, peak at landfall).
#
# Framing: 6h pre-landfall cycling DA on an idealized coastal inlet, tide-gauge
# obs assimilated hourly under a 15 km track perturbation (genuine forecast
# model error — not an inverse crime). Storm peak in the final cycling
# window (W5, "landfall hour").
#
# Set up to give DC-WME its best shot:
#  - vmax=10           — stable forward solver through the peak
#  - track-shift=15 km — model error for L_wme to exploit
#  - obs-fraction=0.10 — denser obs (2× baseline) gives L_wme more
#                        directional structure to use
#  - mf=5              — extra optimizer budget vs the mf3 baseline
#  - WD α=1.5          — Karna lift, repo convention
#
# All four cycling-robustness fixes engaged:
#   REUSE=0, chain-in-alloc, NaN+magnitude guard, bg-feasibility preflight.
#
# Walltime: ~75-90 min/window × 6 windows ≈ 8h; 12h budget = 33% margin.
# --------------------------------------------------------
#SBATCH -J thesis_dcw_v10
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH -o thesis_dcwme_v10_ts15_6w.%j.out
#SBATCH -e thesis_dcwme_v10_ts15_6w.%j.err

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

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_v10_ts15_6w
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_v10_ts15_6w"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

# track-duration-s=68400 → peak at t=9.5h sim = middle of W5 (landfall hour),
# given nt-ramp=96 (4h ramp at dt=150) + 5×nt-da=24 (5h into DA cycle).
bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 6 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 96 \
  --nt-da 24 \
  --method dcwme_static \
  --launcher ibrun \
  --extra "--track-shift 15 --track-duration-s 68400 --dt 150 --obs-fraction 0.10 --obs-frequency 6 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --fixed-sigma-b-sq-h 0.01 --fixed-sigma-b-sq-uv 0.15 --wd --wd-alpha 1.5 --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== THESIS: DC-WME v=10 + TS=15km + dense-obs 6w, peak-at-landfall ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo
echo -n 'Hang check: '; grep -ch 'Error allocating' /work/08398/tg876971/ls6/SWEMniCS/thesis_dcwme_v10_ts15_6w.${SLURM_JOB_ID}.out 2>/dev/null || echo 0
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

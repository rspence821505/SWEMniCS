#!/bin/bash
# DC-WME 2-window cycling, dt=600, ts=15, 1h DA, EVEN TIGHTER fixed-B.
# Next step in the B-tightness scan (100× tighter than original baseline):
#   σ²_h  = 5.0e-04  (vs 5.0e-03 in 733230; σ_h ≈ 0.022 m / 2.2 cm)
#   σ²_uv = 2.0e+01  (vs 2.0e+02 in 733230; σ_uv ≈ 4.5 m/s — first
#                     storm-realistic order)
#
# Continuation of the prior-tightness lever scan:
#   733066 baseline: σ²_h=5e-2, σ²_uv=2e+3  → W0 +33.6%, amp 3.4×
#   733230 tight:    σ²_h=5e-3, σ²_uv=2e+2  → W0 +20.5%, amp 2.7×
#   THIS  tighter:   σ²_h=5e-4, σ²_uv=2e+1  → ?
#
# If amp continues dropping ~linearly with log10(σ²), tighter B might
# eventually break the cycling collapse on its own. Risk: prior so tight
# DA does nothing, W0 imp → 0.
# --------------------------------------------------------
#SBATCH -J thesis_dcw_tighterB_dt600_2w_ts15_dev
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o thesis_dcwme_tighterB_dt600_2w_ts15_dev.%j.out
#SBATCH -e thesis_dcwme_tighterB_dt600_2w_ts15_dev.%j.err

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

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_tighterB_dt600_2w_ts15_dev
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_tighterB_dt600_2w_ts15_dev"
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
  --launcher ibrun \
  --extra "--track-shift 15 --track-duration-s 68400 --dt 600 --obs-fraction 0.10 --obs-frequency 2 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --wd --wd-alpha 1.5 --fixed-sigma-b-sq-h 5.0e-04 --fixed-sigma-b-sq-uv 2.0e+01 --predictability-gamma 1.0 --eq38-component-aware --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME 2w dev cycling (LS6): TIGHTER-B + dt=600 + ts=15 + 1h DA ==="
echo "=== σ²_h=5.0e-04, σ²_uv=2.0e+01 (100× tighter than 733066) ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== B-tightness scan ==="
echo "  733066 (σ²_h=5e-2): W0 +33.6%, W1 -1.14%, amp 3.4×"
echo "  733230 (σ²_h=5e-3): W0 +20.5%, W1 -0.17%, amp 2.7×"
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

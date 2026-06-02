#!/bin/bash
# DC-WME 3-window cycling, dt=600, FIXED-B, 2-hour DA windows, LS6 dev.
# Variant of Vista 733066 (cancelled) with --nt-da 12 (2h DA window
# vs the 1h windows used in every prior thesis run).
#
# Hypothesis: longer DA windows give more obs (7/window vs 4) and
# more time for the dynamics to spread IC perturbation into observable
# response, which should improve the W0 analysis AND potentially
# stabilize cycling by giving DA visibility into longer-horizon
# dynamics before propagation.
#
# Per-window cost ~2× longer than 1h-window (more forward steps in
# each eval). LS6 dev 2h cap means we'll likely get W0 and maybe W1
# only — per user request, that's acceptable.
# --------------------------------------------------------
#SBATCH -J thesis_dcw_fixedB_dt600_3w_da2h_dev
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o thesis_dcwme_fixedB_dt600_3w_da2h_dev.%j.out
#SBATCH -e thesis_dcwme_fixedB_dt600_3w_da2h_dev.%j.err

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

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_fixedB_dt600_3w_da2h_dev
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_fixedB_dt600_3w_da2h_dev"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

# Same fixed-B config as Vista 733066 EXCEPT --nt-da 12 (2h vs 1h).
# With obs-frequency=2 + nt-da=12 → 7 obs/window (vs 4 obs/window).
bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 3 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 24 \
  --nt-da 12 \
  --method dcwme_static \
  --launcher ibrun \
  --extra "--track-shift 15 --track-duration-s 68400 --dt 600 --obs-fraction 0.10 --obs-frequency 2 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --wd --wd-alpha 1.5 --fixed-sigma-b-sq-h 5.0e-02 --fixed-sigma-b-sq-uv 2.0e+03 --predictability-gamma 1.0 --eq38-component-aware --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME 3w dev cycling (LS6): FIXED-B + dt=600 + 2h DA windows ==="
echo "=== σ²_h=5.0e-02, σ²_uv=2.0e+03, track-shift=15, R-const (eeb6332) ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== Reference: Vista 733066 (1h DA windows): W0 +33.60%, W1 -1.14% ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

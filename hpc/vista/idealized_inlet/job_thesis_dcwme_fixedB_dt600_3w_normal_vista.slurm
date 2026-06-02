#!/bin/bash
# DC-WME 3-window cycling, dt=600, FIXED-B (non-TLM) configuration.
#
# Purpose: first run after the R-constant fix (commit eeb6332). Both
# B and R are now stationary across windows — the previous per-window
# TLM-Gram inflation is bypassed by --fixed-sigma-b-sq-h/uv, and the
# obs-noise generator no longer scales by signal magnitude.
#
# Fixed σ_b² values chosen to approximate the TLM-Gram W0-W2 outputs
# from Vista 732738 (dt=600, before R blew up at W3):
#   σ²_h  = 5.0e-02  (TLM-Gram h was 4.74-5.93e-02 across W0-W2)
#   σ²_uv = 2.0e+03  (TLM-Gram uv was 2.90-4.50e+03 across W0-W2)
# This holds the prior identical across all 3 windows so cycling
# convergence/divergence is interpretable without R or B drift.
#
# Vista gh production queue, 3 windows. dt=600 W0 dev probe was ~20 min
# so 3 cycling windows should fit in 4h with margin.
# --------------------------------------------------------
#SBATCH -J dcw_fixedB_dt600_3w_v
#SBATCH -A ADCIRC
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 04:00:00
#SBATCH -o thesis_dcwme_fixedB_dt600_3w_vista.%j.out
#SBATCH -e thesis_dcwme_fixedB_dt600_3w_vista.%j.err

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

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_fixedB_dt600_3w_vista
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_fixedB_dt600_3w_vista"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

# NOTE: --fixed-sigma-b-sq-h/uv PRESENT — bypasses TLM-Gram entirely.
# σ²_h = 5.0e-02 (σ_h ≈ 0.224 m); σ²_uv = 2.0e+03 (σ_uv ≈ 44.7 m/s,
# pathological by physics but mirrors the TLM-Gram uv values used in
# the prior thesis runs — keeps the comparison apples-to-apples).
# dt=600, nt-ramp=24 (4h), nt-da=6 (1h), obs-frequency=2 (4 obs/window).
bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 3 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 24 \
  --nt-da 6 \
  --method dcwme_static \
  --launcher mpirun \
  --extra "--track-shift 15 --track-duration-s 68400 --dt 600 --obs-fraction 0.10 --obs-frequency 2 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --wd --wd-alpha 1.5 --fixed-sigma-b-sq-h 5.0e-02 --fixed-sigma-b-sq-uv 2.0e+03 --predictability-gamma 1.0 --eq38-component-aware --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME 3w cycling (Vista): FIXED-B (non-TLM) + dt=600 ==="
echo "=== σ²_h = 5.0e-02, σ²_uv = 2.0e+03, R constant via eeb6332 ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== Reference: dt=600 TLM-Gram W0 single = +3.84% (ls6 3201419) ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

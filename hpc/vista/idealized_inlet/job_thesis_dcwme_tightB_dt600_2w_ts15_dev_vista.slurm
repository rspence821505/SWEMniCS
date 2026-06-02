#!/bin/bash
# DC-WME 2-window cycling, dt=600, ts=15, 1h DA window, TIGHTER fixed-B.
# Vista gh-dev. Companion B-tightness probe — same config as the prior
# fixed-B runs except σ_b² values are 10× smaller:
#   σ²_h  = 5.0e-03  (vs 5.0e-02 in 733066/733124/etc; σ_h ≈ 0.071 m)
#   σ²_uv = 2.0e+02  (vs 2.0e+03; σ_uv ≈ 14.1 m/s — still high but less
#                     pathological)
#
# Hypothesis: prior pulls analysis closer to background, reducing over-
# fit and improving cycling stability. Tradeoff: weaker W0 improvement
# because DA trusts bg more.
# --------------------------------------------------------
#SBATCH -J dcw_tightB_dt600_2w_ts15_dev_v
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o thesis_dcwme_tightB_dt600_2w_ts15_dev_vista.%j.out
#SBATCH -e thesis_dcwme_tightB_dt600_2w_ts15_dev_vista.%j.err

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

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_tightB_dt600_2w_ts15_dev_vista
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_tightB_dt600_2w_ts15_dev_vista"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

# 10× tighter B than the previous fixed-B baseline.
# σ²_h=5.0e-03 (vs 5.0e-02), σ²_uv=2.0e+02 (vs 2.0e+03).
bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 2 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 24 \
  --nt-da 6 \
  --method dcwme_static \
  --launcher mpirun \
  --extra "--track-shift 15 --track-duration-s 68400 --dt 600 --obs-fraction 0.10 --obs-frequency 2 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --wd --wd-alpha 1.5 --fixed-sigma-b-sq-h 5.0e-03 --fixed-sigma-b-sq-uv 2.0e+02 --predictability-gamma 1.0 --eq38-component-aware --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME 2w dev cycling (Vista): TIGHT fixed-B + dt=600 + ts=15 + 1h DA ==="
echo "=== σ²_h=5.0e-03, σ²_uv=2.0e+02 (10× tighter than 733066) ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== Reference: 733066 (σ²_h=5e-2, σ²_uv=2e+3): W0 +33.60%, W1 -1.14% ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

#!/bin/bash
# DC-WME 12-window cycling, dt=600, 30-min DA, ts=1, dense obs, TIGHT B.
# Vista gh production. Combines the two remaining unused mitigations:
#   - track-shift 1 (vs 3 in 734072): closer to inverse crime, smallest
#                                     non-zero forcing-error tested
#   - tight B σ²_h=5e-3, σ²_uv=2e+2: 10× tighter than 734072's standard.
#                                    Trades W0 improvement for cleaner
#                                    analysis-to-propagation behavior
#
# Reference: 734072 (ts=3, standard B) reached W6 feasible, crashed W7
# at bg=0.40. Hypothesis: ts=1 cuts per-cycle injection roughly in half
# (~0.02 m/cycle vs ~0.04 m/cycle), so the bg threshold of 0.40 should
# be reached around W14 — past the storm peak at W11.
#
# Walltime: tighter B + more windows = up to 5h. Request 6h for margin.
# --------------------------------------------------------
#SBATCH -J dcw_tightB_dt600_12w_ts1_da30m_v
#SBATCH -A ADCIRC
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 06:00:00
#SBATCH -o thesis_dcwme_tightB_dt600_12w_ts1_denseobs_da30m_vista.%j.out
#SBATCH -e thesis_dcwme_tightB_dt600_12w_ts1_denseobs_da30m_vista.%j.err

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

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_tightB_dt600_12w_ts1_denseobs_da30m_vista
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_tightB_dt600_12w_ts1_denseobs_da30m_vista"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

# Two new levers vs 734072:
#   --track-shift 1  (was 3)
#   --fixed-sigma-b-sq-h 5.0e-03  (was 5.0e-02)
#   --fixed-sigma-b-sq-uv 2.0e+02 (was 2.0e+03)
bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 12 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 24 \
  --nt-da 3 \
  --method dcwme_static \
  --launcher mpirun \
  --extra "--track-shift 1 --track-duration-s 68400 --dt 600 --obs-fraction 0.20 --obs-frequency 1 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --wd --wd-alpha 1.5 --fixed-sigma-b-sq-h 5.0e-03 --fixed-sigma-b-sq-uv 2.0e+02 --predictability-gamma 1.0 --eq38-component-aware --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME 12w cycling (Vista): TIGHT-B + dt=600 + 30-min DA + ts=1 + dense obs ==="
echo "=== σ²_h=5.0e-03, σ²_uv=2.0e+02, track-shift=1 km ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== Reference 734072 (ts=3, std B): W0-W6 feasible, W7 INFEASIBLE at bg=0.40 ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

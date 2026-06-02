#!/bin/bash
# DC-WME 6-window cycling, dt=600, ts=3, 1h DA, dense observations.
# Vista gh production. Goal: first 6-window DC-WME run where every
# window's analysis reduces the error vs its background.
#
# Builds on 733124 (the only positive-W1 DC-WME we have, +0.49%) with
# two additional unused free levers:
#   - track-shift 5 → 3  (extrapolating ts=15→5 amp trend: 3.4→2.15
#                         → ts=3 should hit ~1.5-1.6×)
#   - obs-fraction 0.10→0.20, obs-frequency 2→1 (~12 obs/window vs ~4)
#     denser obs reduces over-fit → cleaner IC → lower forecast amp
#
# All other knobs match 733124: vmax=10, standard fixed-B (5e-2/2e+3),
# WD α=1.5, 1h DA. Storm peak lands ~end of W5 with track-duration=19h.
# Walltime estimate: 17-22 min/window × 6 = 100-130 min; 4h gives margin.
# --------------------------------------------------------
#SBATCH -J dcw_fixedB_dt600_6w_ts3_denseobs_v
#SBATCH -A ADCIRC
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 04:00:00
#SBATCH -o thesis_dcwme_fixedB_dt600_6w_ts3_denseobs_vista.%j.out
#SBATCH -e thesis_dcwme_fixedB_dt600_6w_ts3_denseobs_vista.%j.err

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

export SWE4DVAR_OUTPUT_SUBDIR=thesis_dcwme_fixedB_dt600_6w_ts3_denseobs_vista
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="thesis_dcwme_fixedB_dt600_6w_ts3_denseobs_vista"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

# Two new levers vs 733124:
#   --track-shift 3  (was 5)
#   --obs-fraction 0.20 --obs-frequency 1  (were 0.10 / 2)
bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 6 \
  --np 8 \
  --vmax 10 \
  --nt-ramp 24 \
  --nt-da 6 \
  --method dcwme_static \
  --launcher mpirun \
  --extra "--track-shift 3 --track-duration-s 68400 --dt 600 --obs-fraction 0.20 --obs-frequency 1 --obs-noise-level 0.01 --background-error-std 0.02 --smooth-length 1500 --max-iterations 5 --max-funcs 5 --wd --wd-alpha 1.5 --fixed-sigma-b-sq-h 5.0e-02 --fixed-sigma-b-sq-uv 2.0e+03 --predictability-gamma 1.0 --eq38-component-aware --mem-limit-gb 240"
RET=$?

for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== DC-WME 6w cycling (Vista): fixed-B + dt=600 + ts=3 + dense obs + 1h DA ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
echo "=== Reference 733124 (ts=5, sparse obs): W0 +35.99%, W1 +0.49%, amp 2.15× ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

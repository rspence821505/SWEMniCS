#!/bin/bash
# MODEL-ERROR test: 4D-Var v=10 + WD + track-shift=15km, 3-window dev probe.
# Removes the inverse crime — DA assimilates with a 15km-shifted storm
# track while truth uses the true track. Genuine forcing model error,
# the regime DC-WME's data-consistency machinery is designed for.
#
# Config identical to no-track-shift baseline 3186945 EXCEPT track-shift,
# so the W0-W1(-W2) comparison isolates the model-error effect.
# Dev 2h cap: expect W0-W1 clean, W2 may walltime-cap (incremental save).
# --------------------------------------------------------
#SBATCH -J ts15_4dv_v10
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o ts15_v10_4dvar.%j.out
#SBATCH -e ts15_v10_4dvar.%j.err

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
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

export SWE4DVAR_OUTPUT_SUBDIR=3w_ts15_v10_4dvar
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="3w_ts15_v10_4dvar"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_4dvar_N_A_cycling_w*-*.json 2>/dev/null

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 15 \
  --track-duration-s 54000 \
  --dt 150 \
  --nt-ramp 96 --nt-da 24 \
  --n-windows 3 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 6 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --smooth-length 1500 \
  --max-iterations 3 --max-funcs 3 \
  --fixed-sigma-b-sq-h 0.01 --fixed-sigma-b-sq-uv 0.15 \
  --wd --wd-alpha 1.5 \
  --mem-limit-gb 240
RET=$?

for j in $RES_DIR/result_4dvar_N_A_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== v=10 + WD + track-shift=15km 4D-Var 3w (model error) ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}') for w in d['windows']]" 2>/dev/null
done
echo "=== Reference (no track-shift, 3186945): W0 +5.82%, W1 +0.55%, W2 +0.17% ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

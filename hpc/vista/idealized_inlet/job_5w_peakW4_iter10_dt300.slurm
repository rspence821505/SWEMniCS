#!/bin/bash
# Iter 10: 5w/peak-W4 with dt=300 (was 600).
#
# At dt=600 we needed α=0.3 to keep Newton stable through cycling
# propagation, but α=0.3 made each timestep need ~28 Newton iters,
# blowing past walltime. Halving dt halves the per-timestep state
# change, which lets Newton converge in ~13 iters/TS at α=0.5 — both
# faster AND more stable.
#
# Translation to keep windows the same physical length:
#   --dt 300       (was 600)
#   --nt-ramp 48   (was 24,   48*300s = 4h ramp, unchanged)
#   --nt-da 12     (was 6,   12*300s = 1h window, unchanged)
#   --obs-frequency 6  (was 3, gives slots [0,6,12]=3 per window unchanged)
#   --track-duration-s 78500  (unchanged — seconds-based, dt-independent)
#
# Other params unchanged:
#   --vmax 7  --smooth-length 1500
#   --max-iterations 5  --max-funcs 5
#   --background-error-std 0.02  --obs-fraction 0.05  --obs-noise-level 0.01
#   SWE4DVAR_NEWTON_RELAX=0.5 (relaxed from iter7's 0.3 — less needed at dt=300)
# --------------------------------------------------------
#SBATCH -J 5w_dt300
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 01:30:00
#SBATCH -o 5w_dt300.%j.out
#SBATCH -e 5w_dt300.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

export SWE4DVAR_NEWTON_RELAX=0.5

export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_5w_dt300_${SLURM_JOB_ID}.csv"
rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

mpirun -n 4 --bind-to core --map-by socket --timeout 4500 \
  -x SWE4DVAR_NEWTON_RELAX \
  -x SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS \
  -x SWE4DVAR_FORWARD_NEWTON_REUSE \
  -x SWE4DVAR_ADJOINT_SWEEP_KSP \
  -x SWE4DVAR_MALLOC_TRIM \
  -x SWE4DVAR_FORWARD_DIAG_CSV \
  python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 7 --track-shift 0 \
  --track-duration-s 78500 \
  --dt 300 \
  --nt-ramp 48 --nt-da 12 \
  --n-windows 5 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 6 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --smooth-length 1500 \
  --max-iterations 5 --max-funcs 5 \
  --mem-limit-gb 240
RET=$?

echo
echo "=== Per-window results ==="
for j in $RES_DIR/result_4dvar_N_A_cycling_w*-*.json; do
  [[ -f "$j" ]] || continue
  echo "  $(basename "$j")"
  python3 -c "import json; d=json.load(open('$j'));
[print(f\"    W{w['window']-1}: bg={w['bg_rmse']:.4f}  an={w['analysis_rmse']:.4f}  imp={w['improvement_pct']:+.2f}%  evals={w['n_func_evals']}\") for w in d['windows']]" 2>/dev/null
done
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

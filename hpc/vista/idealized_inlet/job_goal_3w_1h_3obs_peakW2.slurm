#!/bin/bash
# GOAL CONFIG: 3-window cycling DA, 1h windows, 3 obs/window, peak surge in W2.
#
# Storm timeline (default endpoints, td=60000):
#   t = 4.77h  storm crosses coast (y=0)              [W0: 4-5h]
#   t = 6.50h  closest approach (frac=0.39)           [W2: 6-7h]
#   t = 16.70h end of track
#
# DA window timeline (1h windows, nt_ramp=24 -> DA starts at t=4h):
#   ramp 0-4h  storm at sea, building, far from coast
#   W0   4-5h  landfall in W0 (storm crosses coast)
#   W1   5-6h  storm intensifying over shelf
#   W2   6-7h  PEAK forcing + surge (closest approach at 6.5h)
#
# Observation density: obs-frequency=3 -> 3 slots per 1h window: [0, 3, 6].
# This is the configuration the user wants validated.
#
# Vista gh-dev: works around LS6's Intel MPI/PETSc-3.19 Newton-divergence
# issue with obs_freq=3 (688936 RunA confirmed Vista handles 3obs cleanly).
# --------------------------------------------------------
#SBATCH -J goal_3w1h3o
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -o goal_3w1h3o.%j.out
#SBATCH -e goal_3w1h3o.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_EVAL_MEM_DIAG=1
export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_goal_3w1h3o_${SLURM_JOB_ID}_rank{rank}.csv"
# Forward-solve diagnostics so any failure is fully traceable
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_goal_3w1h3o_${SLURM_JOB_ID}.csv"
rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

mpirun -n 4 --bind-to core --map-by socket --timeout 7000 \
  -x SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS \
  -x SWE4DVAR_FORWARD_NEWTON_REUSE \
  -x SWE4DVAR_ADJOINT_SWEEP_KSP \
  -x SWE4DVAR_MALLOC_TRIM \
  -x SWE4DVAR_EVAL_MEM_DIAG \
  -x SWE4DVAR_EVAL_MEM_DIAG_CSV \
  -x SWE4DVAR_FORWARD_DIAG_CSV \
  python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --track-duration-s 60000 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 3 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 3 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240
RET=$?

echo
echo "=== Per-window result JSONs ==="
for j in $RES_DIR/result_4dvar_N_A_cycling_w*.json; do
  [[ -f "$j" ]] || continue
  echo "  $(basename "$j")"
  python3 -c "import json; d=json.load(open('$j')); w=d['windows'][0]; print(f\"    bg={w['bg_rmse']:.6f}  an={w['analysis_rmse']:.6f}  imp={w['improvement_pct']:+.2f}%  evals={w['n_func_evals']}\")" 2>/dev/null || cat "$j" | head -20
done

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

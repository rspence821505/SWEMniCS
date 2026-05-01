#!/bin/bash
# Iter 6: 5w/peak-W4 with combined levers.
#   NEWTON_RELAX=0.3 (was 0.7 default) — Newton overshoot guardrail
#   --max-funcs 5 / --max-iterations 5 (was 10/10) — reduces analysis-bg drift
#   --smooth-length 1500 — best W0 from sweep
#
# Iter5 (NEWTON_RELAX=0.5 + 10/10) walltime-capped in W3 — α=0.5 makes
# Newton ~2x slower per timestep, so a full 5w run needs >2h. Cutting
# the per-window optimization budget in half regains the time AND
# addresses cycling-divergence by keeping each window's analysis closer
# to its bg.
# --------------------------------------------------------
#SBATCH -J 5w_iter7
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 01:30:00
#SBATCH -o 5w_iter7.%j.out
#SBATCH -e 5w_iter7.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

export SWE4DVAR_NEWTON_RELAX=0.3

export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_5w_iter7_${SLURM_JOB_ID}.csv"
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
  --nt-ramp 24 --nt-da 6 \
  --n-windows 5 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 3 \
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

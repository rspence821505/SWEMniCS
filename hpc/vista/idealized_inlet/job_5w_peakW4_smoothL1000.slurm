#!/bin/bash
# 5-window cycling DA, 1h windows, 3 obs/window, peak surge in W4.
# Iter 1: lever 1 = stronger gradient smoother (L=1000m, was 500m default).
#
# Goal config: extends 3w/peak-W2 (689114 SUCCESS) to 5w/peak-W4 with
# stronger smoothing to suppress cycling-DA divergence in W1-W4.
#
# Storm timeline (track_duration_s=78500):
#   t = 6.24h  storm crosses coast               [W2: 6-7h]
#   t = 8.50h  closest approach (frac=0.39)      [W4: 8-9h]
#
# DA windows:
#   ramp 0-4h
#   W0   4-5h  far approach
#   W1   5-6h  approaching
#   W2   6-7h  coast crossing at 6.24h
#   W3   7-8h  intensifying
#   W4   8-9h  PEAK at 8.5h
#
# Acceptance:
#   - All 5 windows complete (no Newton failure)
#   - W0 imp ≥ -25% (matches 3w baseline)
#   - At least one of W1-W4 has imp > +5% (cycling not divergent)
# --------------------------------------------------------
#SBATCH -J 5w_smL1000
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 01:30:00
#SBATCH -o 5w_smL1000.%j.out
#SBATCH -e 5w_smL1000.%j.err

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
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_5w_smL1000_${SLURM_JOB_ID}.csv"
rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

mpirun -n 4 --bind-to core --map-by socket --timeout 4500 \
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
  --smooth-length 1000 \
  --max-iterations 10 --max-funcs 10 \
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

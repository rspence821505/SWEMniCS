#!/bin/bash
# Vista np=4 perf test: diagnose 135x slowdown vs LS6 in 10w_slow_v.
#
# Hypothesis: --bind-to core pins each rank to a single physical
# core; OMP_NUM_THREADS=16 + MKL_NUM_THREADS=16 then thrash 32
# threads on 1 core. Fix: bind-to numa (each rank gets ~36-core
# NUMA zone) + lower thread counts (8 each).
#
# Also adds DOLFINX_JIT_CFLAGS="-O3" so FFCX form-eval kernels
# compile with optimization (default conda gcc may emit -O0/-O2).
#
# Test config: very small (1 window, nt-ramp=12, nt-da=6,
# max-funcs=3) so we get per-step timing in <30 min on dev queue.
# Compare per-step wallclock vs LS6 3w_slow (3139864) ~22s/step.
# Goal: confirm per-step pace within ~2x of LS6 (vs 135x slowdown).
# --------------------------------------------------------
#SBATCH -J vista_perf
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -o vista_perf.%j.out
#SBATCH -e vista_perf.%j.err

set -euxo pipefail

# --- Threading: lower from 16 to 8 to fit Grace 36-core NUMA quadrant ---
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
# Let OMP do the pinning rather than mpirun --bind-to core
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# --- DOLFINx FFCX kernel optimization ---
export DOLFINX_JIT_CFLAGS="-O3 -march=armv9-a+sve2 -mtune=neoverse-v2"

source $WORK/SWEMniCS/hpc/vista/env.vista.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

export SWE4DVAR_NEWTON_RELAX=0.3
export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_vista_perf_${SLURM_JOB_ID}.csv"
rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"
rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

# --- Binding: numa instead of core, so OMP can spread across cores ---
mpirun -n 4 --bind-to numa --map-by ppr:1:numa --timeout 6000 \
  python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 7 --track-shift 0 \
  --track-duration-s 78500 \
  --dt 300 \
  --nt-ramp 12 --nt-da 6 \
  --n-windows 1 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 6 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --smooth-length 1500 \
  --max-iterations 3 --max-funcs 3 \
  --mem-limit-gb 240
RET=$?

echo
echo "=== Per-step pace check ==="
python3 -c "
import os, time
csv = os.environ.get('SWE4DVAR_FORWARD_DIAG_CSV','')
if csv and os.path.exists(csv):
    with open(csv) as f:
        n = sum(1 for _ in f) - 1
    print(f'  diag rows: {n}')
" 2>&1 || true

echo
echo "=== Window result ==="
for j in $RES_DIR/result_4dvar_N_A_cycling_w*-*.json; do
  [[ -f "$j" ]] || continue
  echo "  $(basename "$j")"
  python3 -c "import json; d=json.load(open('$j'));
[print(f\"    W{w['window']-1}: bg={w['bg_rmse']:.4f}  an={w['analysis_rmse']:.4f}  imp={w['improvement_pct']:+.2f}%  evals={w['n_func_evals']}\") for w in d['windows']]" 2>/dev/null
done
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

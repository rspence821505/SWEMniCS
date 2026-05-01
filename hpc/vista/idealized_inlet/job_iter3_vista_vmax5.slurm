#!/bin/bash
# Iter 3 (Vista): same as 688962 (goal config) but vmax=5 (was 10).
#
# 688962 failed at W2 timestep 1 (Newton divergence). The W1 analysis
# state, propagated 1h forward into peak-storm dynamics, produces an
# unstable state. Surge magnitude scales as Vmax^2, so halving Vmax
# from 10 m/s to 5 m/s gives ~1/4 the surge - much less violent
# dynamics and less stress on the W2 forward solve.
# Same other params: 3w / 1h / 3obs / td=60000 / peak in W2.
# --------------------------------------------------------
#SBATCH -J iter3_vmx5
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -o iter3_vmx5.%j.out
#SBATCH -e iter3_vmx5.%j.err

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
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_iter3_vmx5_${SLURM_JOB_ID}.csv"
rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

mpirun -n 4 --bind-to core --map-by socket --timeout 7000 \
  -x SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS \
  -x SWE4DVAR_FORWARD_NEWTON_REUSE \
  -x SWE4DVAR_ADJOINT_SWEEP_KSP \
  -x SWE4DVAR_MALLOC_TRIM \
  -x SWE4DVAR_FORWARD_DIAG_CSV \
  python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 5 --track-shift 0 \
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
  python3 -c "import json; d=json.load(open('$j')); w=d['windows'][0]; print(f\"    bg={w['bg_rmse']:.6f}  an={w['analysis_rmse']:.6f}  imp={w['improvement_pct']:+.2f}%  evals={w['n_func_evals']}\")" 2>/dev/null
done
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET

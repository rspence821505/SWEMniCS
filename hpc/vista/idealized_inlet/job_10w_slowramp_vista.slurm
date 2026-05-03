#!/bin/bash
# Vista port of 10w slow-ramp follow-up to 3w_slow diagnostic.
#
# Same physics + DA config as job_10w_slowramp_ls6.slurm. Differences:
#  - mpirun -n 4 instead of ibrun (Vista uses conda openmpi)
#  - source hpc/vista/env.vista.sh (conda PETSc + DOLFINx 0.10)
#  - Vista WORK path /work/08398/tg876971/vista/SWEMniCS
#  - -p gh (Vista normal queue), 10h walltime
#  - mpirun --timeout 34200 (9.5h) for safety against runaway
#
# Vista cycling DA at np=4 is uncharacterized — solver per-step
# timing will differ from LS6 (Intel MPI + system PETSc vs conda
# openmpi + conda PETSc on GH200 ARM nodes). 10h walltime gives
# 25% margin over LS6's 8h estimate to absorb the unknown.
# --------------------------------------------------------
#SBATCH -J 10w_slow_v
#SBATCH -A ADCIRC
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 10:00:00
#SBATCH -o 10w_slow_vista.%j.out
#SBATCH -e 10w_slow_vista.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
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
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_10w_slow_vista_${SLURM_JOB_ID}.csv"
rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

mpirun -n 4 --bind-to core --timeout 34200 \
  python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 7 --track-shift 0 \
  --track-duration-s 157000 \
  --dt 300 \
  --nt-ramp 48 --nt-da 12 \
  --n-windows 10 --start-window 0 --chain-mode \
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

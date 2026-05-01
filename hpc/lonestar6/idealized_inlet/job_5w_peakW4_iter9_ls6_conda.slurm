#!/bin/bash
# Iter 9 (LS6 with conda stack): mirror of Vista iter8 — vmax=10, α=0.3,
# max-funcs=5, 3-obs/window, peak-W4. Tests whether LS6 with the
# Vista-equivalent conda stack (PETSc 3.25 + OpenMPI 5 + MUMPS) handles
# the cycling DA the same way Vista does, vs the LS6 system stack
# (PETSc 3.19 + Intel MPI 19) which historically had issues at obs_freq=3.
# --------------------------------------------------------
#SBATCH -J 5w_iter9c
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 01:30:00
#SBATCH -o 5w_iter9c.%j.out
#SBATCH -e 5w_iter9c.%j.err

set -euxo pipefail
# Use the conda stack (PETSc 3.25 + OpenMPI 5), NOT the system Intel MPI.
source $WORK/SWEMniCS/hpc/lonestar6/env.ls6_conda.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

export SWE4DVAR_NEWTON_RELAX=0.3

export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_5w_iter9c_${SLURM_JOB_ID}.csv"
rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

# Conda OpenMPI uses mpirun, not ibrun (which is Intel-MPI specific).
# Single-node only (conda OpenMPI not multi-node-IB on LS6).
mpirun -n 4 --bind-to core --map-by socket --timeout 4500 \
  -x SWE4DVAR_NEWTON_RELAX \
  -x SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS \
  -x SWE4DVAR_FORWARD_NEWTON_REUSE \
  -x SWE4DVAR_ADJOINT_SWEEP_KSP \
  -x SWE4DVAR_MALLOC_TRIM \
  -x SWE4DVAR_FORWARD_DIAG_CSV \
  python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
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

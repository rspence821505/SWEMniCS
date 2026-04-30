#!/bin/bash
# W0 DA validation with sweep-KSP enabled. Verifies the persistent KSP +
# J_reg buffer produce a correct gradient (cost decreases, RMSE improves)
# matching the LU baseline's documented W0 result of ~26.2% improvement.
# --------------------------------------------------------
#SBATCH -J w0_sweep
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 01:00:00
#SBATCH -o w0_sweep.%j.out
#SBATCH -e w0_sweep.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

# sweep-KSP is ON by default — show that explicitly.
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

# Light memory diag (RSS-only is fine; PETSC_OPTIONS not needed for correctness check).
export SWE4DVAR_EVAL_MEM_DIAG=1
RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_w0sweep_${SLURM_JOB_ID}_rank{rank}.csv"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_w0sweep_${SLURM_JOB_ID}.csv"

# Wipe stale chain bg files so we start at the cold-start window 0.
rm -f "$RES_DIR"/chain_bg_after_w*_rank*.npy

# Single-window DA: same parameters as the documented W0 baseline (26.2% improvement).
ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 1 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo
echo "=== W0 result ==="
cat "$RES_DIR/result_4dvar_N_A_cycling_w0-0.json" 2>/dev/null || echo "no result json"

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

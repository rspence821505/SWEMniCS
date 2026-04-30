#!/bin/bash
# Repeated single-window eval reproducer with memory diagnostics.
# Runs the same value_gradient(m_background) call 12 times in one process,
# logging RSS + PETSc memory at 4 lifecycle points per eval. Used to
# classify per-eval memory growth as live PETSc state vs. allocator
# retention.
# --------------------------------------------------------
#SBATCH -J mem_repro_n
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:45:00
#SBATCH -o mem_repro_n.%j.out
#SBATCH -e mem_repro_n.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"

# --- Diagnostic env vars ------------------------------------------------
export SWE4DVAR_EVAL_MEM_DIAG=1
export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_repro_${SLURM_JOB_ID}_rank{rank}.csv"
export SWE4DVAR_REPEAT_EVALS=12
# Enable malloc_trim path so we can compare RSS before/after libc trim.
export SWE4DVAR_MALLOC_TRIM=1
# Optional: PETSc malloc-debug + memory-view (cheap)
# export SWE4DVAR_PETSC_MALLOC_DEBUG=1

# Forward-diag CSV — useful for cross-correlating eval_id with per-step Newton.
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_repro_${SLURM_JOB_ID}.csv"
rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"

ibrun python -u experiments/repeated_eval_reproducer.py \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 1 --max-funcs 1 \
  --mem-limit-gb 240

echo
echo "=== mem-diag CSV summary ==="
ls -la "$RES_DIR"/eval_mem_repro_${SLURM_JOB_ID}_rank*.csv 2>&1 || true
echo
echo "=== rank 0 head + tail ==="
RANK0_CSV="$RES_DIR/eval_mem_repro_${SLURM_JOB_ID}_rank0.csv"
if [[ -f "$RANK0_CSV" ]]; then
  head -1 "$RANK0_CSV"
  echo "---first 6 records (eval 1)---"
  head -7 "$RANK0_CSV" | tail -6
  echo "---last 16 records (eval 11-12)---"
  tail -16 "$RANK0_CSV"
fi

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

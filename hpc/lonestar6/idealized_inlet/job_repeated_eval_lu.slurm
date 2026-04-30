#!/bin/bash
# Part-2 experiment: repeated single-window eval with the ITERATIVE adjoint
# (GMRES + BJACOBI/ILU) instead of direct LU. Same reproducer harness as
# job_repeated_eval_repro.slurm; the only material change is
# SWE4DVAR_ADJOINT_ITERATIVE=1, which routes _solve_transpose_system through
# the iterative branch in implicit_adjoint.py.
#
# Goal: see whether removing the per-step LU factorization eliminates the
# dominant ~800 MB/eval pool retention attributed to repeated factor slabs.
#
# PETSc memory accounting is enabled here at the SHELL level via
# PETSC_OPTIONS — it MUST be set before Python imports petsc4py, which
# rules out the in-Python configure_petsc_malloc_logging() path.
# --------------------------------------------------------
#SBATCH -J mem_lu
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:45:00
#SBATCH -o mem_lu.%j.out
#SBATCH -e mem_lu.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"

# --- PETSc memory accounting: must be set here in the shell ----
# `-malloc_debug` makes PETSc.Log.getMemoryUsage() return real numbers.
# `-memory_view` prints a memory summary at PETSc finalize.
# `-log_view :file:ascii_info_detail` writes per-event counters incl. mem cols.
# `-log_view_memory` adds memory columns to the log_view summary.
export PETSC_OPTIONS="-malloc_debug -memory_view -log_view :$RES_DIR/petsc_log_lu_${SLURM_JOB_ID}.txt:ascii_info_detail -log_view_memory"

# --- Eval-memory diagnostic ----
export SWE4DVAR_EVAL_MEM_DIAG=1
export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_lu_${SLURM_JOB_ID}_rank{rank}.csv"
export SWE4DVAR_REPEAT_EVALS=12
export SWE4DVAR_MALLOC_TRIM=1

# --- The branch under test ----
# export SWE4DVAR_ADJOINT_ITERATIVE=0  # LU baseline
# export SWE4DVAR_ADJOINT_ITER_RTOL=1e-10
# export SWE4DVAR_ADJOINT_ITER_ATOL=1e-14
# export SWE4DVAR_ADJOINT_ITER_MAXIT=5000

export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_lu_${SLURM_JOB_ID}.csv"
rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"

ibrun python -u experiments/repeated_eval_reproducer.py \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 6 \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 1 --max-funcs 1 \
  --mem-limit-gb 240

echo
echo "=== mem-diag CSV summary (LU baseline) ==="
ls -la "$RES_DIR"/eval_mem_lu_${SLURM_JOB_ID}_rank*.csv 2>&1 || true
RANK0_CSV="$RES_DIR/eval_mem_lu_${SLURM_JOB_ID}_rank0.csv"
if [[ -f "$RANK0_CSV" ]]; then
  echo "---per-eval after_cleanup rows---"
  awk -F, 'NR==1 || $4=="after_cleanup"' "$RANK0_CSV"
  echo "---per-step (sample, eval=12)---"
  awk -F, 'NR==1 || ($2=="12" && $4 ~ /transpose_solve/)' "$RANK0_CSV"
fi

echo
echo "=== PETSc memory log tail ==="
tail -40 "$RES_DIR/petsc_log_lu_${SLURM_JOB_ID}.txt" 2>/dev/null || echo "(no PETSc log)"

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

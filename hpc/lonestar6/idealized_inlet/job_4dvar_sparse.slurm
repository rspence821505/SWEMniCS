#!/bin/bash
# Phase 1 (authoritative, np=2 × 64 threads) — matched 4D-Var sparse baseline.
# See docs/ls6_first_dcwme_production_run.md §0 and §3.2.
#
# Submit:  sbatch --test-only hpc/lonestar6/idealized_inlet/job_4dvar_sparse.slurm   # free
#          sbatch hpc/lonestar6/idealized_inlet/job_4dvar_sparse.slurm
# -----------------------------------------------------------------------------
#SBATCH -J inlet_4dvar_sparse
#SBATCH -o inlet_4dvar_sparse.%j.out
#SBATCH -e inlet_4dvar_sparse.%j.err
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 2                         # Phase 1: 2 MPI ranks
#SBATCH -t 02:00:00

set -euxo pipefail

# Force Phase 1 thread count (64 threads/rank); env.ls6.sh respects env already-set.
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh

# Provenance block — archive with the run output
echo "=== provenance ==="
module list 2>&1
pip show fenics-dolfinx petsc4py mpi4py | grep -E "^(Name|Version|Location)"
echo "TACC_DOLFINX_DIR=$TACC_DOLFINX_DIR"
echo "TACC_PETSC_DIR=$TACC_PETSC_DIR"
echo "I_MPI_ROOT=$I_MPI_ROOT"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "SLURM_NTASKS=$SLURM_NTASKS  SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE  SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
git -C $WORK/SWEMniCS rev-parse HEAD

cd $WORK/SWEMniCS

# 5-min memory watchdog in background
(while sleep 300; do
    echo "[watchdog $(date +%T)] $(free -h | grep Mem:)"
done) &
WATCHDOG_PID=$!
trap "kill $WATCHDOG_PID 2>/dev/null || true" EXIT

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,AllocCPUS,AllocNodes 2>&1 || true
echo "=== done ==="

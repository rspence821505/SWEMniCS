#!/bin/bash
# Plan §7 fallback — DC-WME with TLM Eq 38 ON, correlation kernel OFF.
# Primary config (job_dcwme_prod.slurm) TIMED OUT in Step 7b when TLM-inflated B
# was combined with obs_correlation_length=1500 (run 3098388, 2h wall).
# This fallback isolates the TLM mechanism from the kernel.
#
# Phase 1 topology (authoritative): np=2 × 64 threads per plan §0.
#
# Submit:  sbatch --test-only hpc/lonestar6/idealized_inlet/job_dcwme_fallback.slurm
#          sbatch -p development hpc/lonestar6/idealized_inlet/job_dcwme_fallback.slurm
# -----------------------------------------------------------------------------
#SBATCH -J inlet_dcwme_fallback
#SBATCH -o inlet_dcwme_fallback.%j.out
#SBATCH -e inlet_dcwme_fallback.%j.err
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 2                         # Phase 1: 2 MPI ranks
#SBATCH -t 02:00:00

set -euxo pipefail

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh

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

(while sleep 300; do
    echo "[watchdog $(date +%T)] $(free -h | grep Mem:)"
done) &
WATCHDOG_PID=$!
trap "kill $WATCHDOG_PID 2>/dev/null || true" EXIT

# Plan §7 fallback config:
#   - TLM Eq 38 ON (no --skip-tlm-eq38 flag)
#   - correlation kernel OFF (no --obs-correlation-length flag)
#   - everything else identical to primary
ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 20 --track-shift 10 \
  --nt-ramp 24 --nt-da 12 \
  --obs-fraction 0.005 --obs-frequency 4 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --predictability-gamma 0.1 \
  --mem-limit-gb 240

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,AllocCPUS,AllocNodes 2>&1 || true
echo "=== done ==="

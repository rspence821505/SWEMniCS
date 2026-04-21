#!/bin/bash
# -----------------------------------------------------------------------------
# Minimal sanity-check sbatch — one node, <5 min, development queue.
# Purpose: confirm the venv + system modules + ibrun all align before running
# anything that could burn SUs.
#
# Submit:  sbatch --test-only hpc/lonestar6/environment/test_cpu.slurm   # FREE dry-run
#          sbatch hpc/lonestar6/environment/test_cpu.slurm               # real
# -----------------------------------------------------------------------------

#SBATCH -J fenics_smoke
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4                        # 4 MPI ranks — enough to verify multi-rank, cheap
#SBATCH -t 00:05:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

set -euxo pipefail

# -----------------------------------------------------------------------------
# MODULES — must match the stack the venv was built against
# -----------------------------------------------------------------------------
module reset
module load gcc/13.2.0 impi/21.11 python/3.12.11 petsc/3.22 dolfinx/0.10.0.post5
module list

# Activate venv built with Strategy A
source $WORK/venvs/fenics-ls6/bin/activate
which python
python -c "import dolfinx, petsc4py, mpi4py; print('dolfinx', dolfinx.__version__, 'petsc4py', petsc4py.__version__, 'mpi4py', mpi4py.__version__)"

# -----------------------------------------------------------------------------
# Provenance — log what we're actually running against
# -----------------------------------------------------------------------------
echo "=== env provenance ==="
pip show fenics-dolfinx petsc4py mpi4py | grep -E "^(Name|Version|Location)" || true
echo "TACC_DOLFINX_DIR=$TACC_DOLFINX_DIR"
echo "TACC_PETSC_DIR=$TACC_PETSC_DIR"
echo "I_MPI_ROOT=$I_MPI_ROOT"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NTASKS=$SLURM_NTASKS"

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
cd $WORK/SWEMniCS

echo; echo "=== test_mpi.py ==="
ibrun python hpc/lonestar6/environment/test_mpi.py

echo; echo "=== test_dolfinx.py ==="
ibrun python hpc/lonestar6/environment/test_dolfinx.py

echo; echo "=== test_inlet_minimal.py ==="
ibrun python hpc/lonestar6/environment/test_inlet_minimal.py

echo; echo "=== ALL SMOKE TESTS PASSED ==="

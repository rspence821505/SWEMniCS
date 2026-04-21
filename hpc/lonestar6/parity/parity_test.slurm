#!/bin/bash
# -----------------------------------------------------------------------------
# parity_test.slurm — 5-minute sanity sbatch that runs every parity test on
# a real compute node and writes JSON into $SCRATCH/parity_runs/$SLURM_JOB_ID.
#
# Use for periodic re-checks (after module upgrades, after project code ports
# to dolfinx 0.10). Not needed for the original parity audit — login-node
# runs already produced deterministic output, as documented.
#
# Submit:  sbatch --test-only hpc/lonestar6/parity/parity_test.slurm   # free
#          sbatch hpc/lonestar6/parity/parity_test.slurm
# -----------------------------------------------------------------------------
#SBATCH -J parity_test
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 2                            # 2 ranks for MPI test; single-rank for the rest
#SBATCH -t 00:05:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

set -euxo pipefail

module reset
module load gcc/13.2.0 impi/21.12 python/3.12.11 \
            boost/1.86.0 pugixml/1.15 phdf5/2.0.0 parmetis/4.0.3 \
            ptscotch/7.0.7-i64 adios2/2.10.2 spdlog/1.17.0 \
            basix/0.10.0.post0 ffcx/0.10.1.post0 \
            petsc/3.22 dolfinx/0.10.0.post5
module list

source $WORK/venvs/fenics-ls6/bin/activate

OUT=$SCRATCH/parity_runs/$SLURM_JOB_ID
mkdir -p "$OUT"
cd $WORK/SWEMniCS

# Provenance
python hpc/lonestar6/parity/parity_test_imports.py > "$OUT/imports.json"
# 2-rank MPI sanity via ibrun
ibrun -n 2 python hpc/lonestar6/parity/parity_test_mpi.py > "$OUT/mpi.json"
# Single-rank PETSc + dolfinx (direct LU, deterministic)
python hpc/lonestar6/parity/parity_test_petsc.py > "$OUT/petsc.json"
python hpc/lonestar6/parity/parity_test_dolfinx_solve.py > "$OUT/dolfinx.json"

# Project-level tests — will fail until dolfinx 0.10 port lands
set +e
python hpc/lonestar6/parity/parity_4dvar_reduced.py > "$OUT/4dvar.json" || true
python hpc/lonestar6/parity/parity_dcwme_reduced.py > "$OUT/dcwme.json" || true
set -e

ls -la "$OUT"
echo "Results in $OUT — rsync home to compare with laptop output."

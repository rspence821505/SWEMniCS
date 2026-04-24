#!/bin/bash
# Vista dev-queue sanity template.
# 15-min job on gh-dev, np=2 OpenMPI, prints env info and runs a tiny DOLFINx
# mesh+assemble test. Use to verify the env is functional before production.
# -----------------------------------------------------------------------------
#SBATCH -J vista_sanity
#SBATCH -A ADCIRC
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 00:15:00
#SBATCH -o vista_sanity.%j.out
#SBATCH -e vista_sanity.%j.err

set -euxo pipefail
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
git rev-parse HEAD

# Conda OpenMPI uses `mpirun`, not `ibrun`. For multi-node later, swap to ibrun
# + system OpenMPI (see VISTA_PLAYBOOK §5).
mpirun -n 2 python -c "
from mpi4py import MPI
from dolfinx import mesh, fem
import swe4dvar
comm = MPI.COMM_WORLD
m = mesh.create_unit_square(comm, 32, 32)
V = fem.functionspace(m, ('Lagrange', 1))
print(f'[rank {comm.rank}/{comm.size}] dolfinx {mesh.__name__} OK, '
      f'global DOFs={V.dofmap.index_map.size_global}, '
      f'local DOFs={V.dofmap.index_map.size_local}', flush=True)
"

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

#!/bin/bash
# -----------------------------------------------------------------------------
# Lonestar6 short debug/validation job
# Partition: development (QoS qdevelopment — MaxNodes=8, MaxWall=2h)
# Use for: sanity-checking a Slurm script, reproducing a failure, smoke tests.
# -----------------------------------------------------------------------------
# Charges SUs at the same per-node rate as normal, so keep -t small.
# -----------------------------------------------------------------------------

#SBATCH -J swe_debug
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 16                        # small parallel run to exercise MPI
#SBATCH -t 00:15:00                  # keep it short
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

set -euxo pipefail                   # verbose for debugging

module reset
module load intel/19.1.1 impi/19.0.9 python3/3.9.7
module list

echo "Node(s): $SLURM_JOB_NODELIST"
echo "NTASKS:  $SLURM_NTASKS"
echo "NNODES:  $SLURM_NNODES"
nproc
free -h
env | grep -E '^(SLURM|TACC|WORK|SCRATCH|HOME)_?' | sort

# MPI hello world (swap in your code):
ibrun hostname

# Optional: test your project imports quickly
# ibrun python -c "from mpi4py import MPI; print('rank', MPI.COMM_WORLD.rank, 'of', MPI.COMM_WORLD.size)"

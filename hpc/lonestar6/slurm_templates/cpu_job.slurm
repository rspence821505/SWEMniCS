#!/bin/bash
# -----------------------------------------------------------------------------
# Lonestar6 CPU production job template
# Partition: normal  (QoS qnormal — MaxNodes=64, MaxWall=2d)
# -----------------------------------------------------------------------------
# Edit: -J, -N, -n, -t, and the run section at the bottom.
# Submit: sbatch --test-only cpu_job.slurm    (validates, no SUs)
#         sbatch cpu_job.slurm                (real submit)
# -----------------------------------------------------------------------------

#SBATCH -J swe_cpu                   # job name
#SBATCH -A ADCIRC                    # allocation
#SBATCH -p normal                    # partition
#SBATCH -N 2                         # nodes  (128 cores each on ls6 normal)
#SBATCH -n 256                       # total MPI tasks = 2 nodes * 128 cores
#SBATCH -t 02:00:00                  # wall clock  HH:MM:SS
#SBATCH -o %x.%j.out                 # stdout: jobname.jobid.out
#SBATCH -e %x.%j.err                 # stderr
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rylan.spence@utexas.edu

set -euo pipefail

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
module reset
module load intel/19.1.1 impi/19.0.9 python3/3.9.7
# module load petsc/3.19 hdf5/1.14.0 netcdf/4.9.2 parallel-netcdf/4.6.2
module list

# -----------------------------------------------------------------------------
# Run from $SCRATCH (TACC policy: never run from $HOME or $WORK)
# -----------------------------------------------------------------------------
RUN_DIR="$SCRATCH/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

# Stage inputs from $WORK if needed:
# cp -r $WORK/SWEMniCS/inputs/ ./inputs/

echo "=== job $SLURM_JOB_ID on $SLURM_JOB_NODELIST ==="
echo "running in $RUN_DIR"
date

# -----------------------------------------------------------------------------
# Launch: use ibrun for MPI (NOT mpirun, NOT srun for MPI binaries)
# -----------------------------------------------------------------------------
# Example: pure MPI
# ibrun python -m mpi4py /work/$USER/ls6/SWEMniCS/run_experiment.py --config my.yaml

# Example: hybrid (2 ranks per node, 64 threads per rank — one rank per NUMA):
# #SBATCH -N 2 -n 4 -c 64
# export OMP_NUM_THREADS=64
# ibrun python -m mpi4py /work/$USER/ls6/SWEMniCS/run_experiment.py --config my.yaml

# -----------------------------------------------------------------------------
# Stage results back to $WORK before the 10-day $SCRATCH purge
# -----------------------------------------------------------------------------
# mkdir -p $WORK/results/$SLURM_JOB_NAME
# rsync -a ./results/ $WORK/results/$SLURM_JOB_NAME/

date
echo "=== done ==="

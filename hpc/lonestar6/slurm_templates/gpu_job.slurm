#!/bin/bash
# -----------------------------------------------------------------------------
# Lonestar6 A100 GPU production job template
# Partition: gpu-a100  (QoS qa100 — MaxNodes=8, MaxWall=2d)
# Each node: 3× NVIDIA A100 PCIe 40 GB + 128 Milan cores + 256 GB RAM
# -----------------------------------------------------------------------------
# For a single-GPU shared slot (cheaper, 1 GPU), use partition gpu-a100-small
# (QoS qa100small, MaxNodes=1). For H100, use partition gpu-h100 (MaxNodes=1).
# -----------------------------------------------------------------------------

#SBATCH -J swe_gpu
#SBATCH -A ADCIRC
#SBATCH -p gpu-a100
#SBATCH -N 1                         # 1 node = 3 A100s
#SBATCH -n 3                         # 1 MPI rank per GPU
#SBATCH -t 02:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rylan.spence@utexas.edu

set -euo pipefail

# -----------------------------------------------------------------------------
# Environment (pick a CUDA-aware PETSc if you want device matvec)
# -----------------------------------------------------------------------------
module reset
module load intel/19.1.1 impi/19.0.9 python3/3.9.7
# module load petsc/3.19-cuda                 # CUDA-enabled PETSc (if using)
# module load cuda/12                         # if your code calls CUDA directly
module list
nvidia-smi                                    # confirm 3× A100 visible

# -----------------------------------------------------------------------------
# Run from $SCRATCH
# -----------------------------------------------------------------------------
RUN_DIR="$SCRATCH/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "=== job $SLURM_JOB_ID on $SLURM_JOB_NODELIST ==="
echo "GPUs: $(nvidia-smi -L)"
date

# -----------------------------------------------------------------------------
# GPU + MPI launch patterns
# -----------------------------------------------------------------------------
# (a) Single-rank single-GPU (simplest):
# export CUDA_VISIBLE_DEVICES=0
# python /work/$USER/ls6/SWEMniCS/run_experiment.py --config gpu.yaml

# (b) 3 MPI ranks, one GPU each (common for domain-decomp solvers):
# ibrun python /work/$USER/ls6/SWEMniCS/run_experiment.py --config gpu.yaml
# (CUDA_VISIBLE_DEVICES is set per rank automatically by ibrun on LS6)

# -----------------------------------------------------------------------------
# Archive
# -----------------------------------------------------------------------------
# rsync -a ./results/ $WORK/results/$SLURM_JOB_NAME/

date
echo "=== done ==="

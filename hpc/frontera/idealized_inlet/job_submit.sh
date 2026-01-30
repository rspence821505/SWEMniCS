#!/bin/bash
#==============================================================================
# SLURM Batch Script for Idealized Inlet DA Experiments on TACC Frontera
#==============================================================================
#
# This script submits parallel data assimilation experiments for the
# idealized inlet problem on Frontera.
#
# Usage:
#   sbatch job_submit.sh                     # Default: 1 node, normal queue
#   sbatch --nodes=4 job_submit.sh           # 4 nodes (224 ranks)
#   sbatch -p development job_submit.sh      # Development queue
#
# To run a scaling study across multiple node counts:
#   for N in 1 2 4 8; do
#       sbatch --nodes=$N --job-name="inlet_n${N}" job_submit.sh
#   done
#
#==============================================================================

#------------------------------------------------------------------------------
# SLURM Job Configuration
#------------------------------------------------------------------------------

# Job name - appears in squeue output
#SBATCH --job-name=inlet_4dvar

# Output file for stdout (%j = job ID, %N = node list)
#SBATCH --output=inlet_%j_%N.out

# Output file for stderr
#SBATCH --error=inlet_%j_%N.err

# Queue/partition selection:
#   - development: max 4 nodes, 2 hours (for testing)
#   - normal: max 512 nodes, 48 hours (production)
#   - large: 513+ nodes (requires special allocation)
#SBATCH --partition=normal

# Number of nodes (can be overridden via sbatch --nodes=N)
#SBATCH --nodes=1

# MPI tasks per node (Frontera CLX has 56 cores per node)
# Use all 56 cores for best performance
#SBATCH --ntasks-per-node=56

# Wall-clock time limit (hh:mm:ss)
# 4D-Var typically requires 2-4 hours depending on problem size
#SBATCH --time=04:00:00

# Memory per node (optional - Frontera has 192GB/node for CLX)
# Uncomment if you need to reserve specific memory
##SBATCH --mem=180G

# Email notifications (optional)
# Replace with your email if desired
##SBATCH --mail-user=your-email@utexas.edu
##SBATCH --mail-type=BEGIN,END,FAIL

# Allocation/project (REQUIRED on Frontera)
# Replace with your allocation name
#SBATCH --account=YOUR_ALLOCATION

#------------------------------------------------------------------------------
# Compute total MPI ranks
#------------------------------------------------------------------------------
# Calculate total ranks based on nodes and tasks-per-node
TOTAL_RANKS=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

echo "=============================================================="
echo "SLURM Job Information"
echo "=============================================================="
echo "Job ID:           $SLURM_JOB_ID"
echo "Job Name:         $SLURM_JOB_NAME"
echo "Nodes:            $SLURM_NNODES"
echo "Tasks per node:   $SLURM_NTASKS_PER_NODE"
echo "Total MPI ranks:  $TOTAL_RANKS"
echo "Partition:        $SLURM_JOB_PARTITION"
echo "Time limit:       $SLURM_TIMELIMIT"
echo "Working dir:      $(pwd)"
echo "=============================================================="

#------------------------------------------------------------------------------
# Environment Setup
#------------------------------------------------------------------------------
echo ""
echo "Loading environment..."

# Source the environment setup script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/environment_setup.sh"

# Verify the environment is set up correctly
if ! verify_environment; then
    echo "ERROR: Environment verification failed!"
    exit 1
fi

#------------------------------------------------------------------------------
# Experiment Configuration
#------------------------------------------------------------------------------
# These can be overridden via environment variables before submission

# Problem size (mesh refinement)
# Default values for production runs
NX=${NX:-100}
NY=${NY:-100}

# Time stepping
DT=${DT:-1200}                       # Time step in seconds (20 minutes)
FINAL_TIME=${FINAL_TIME:-345600}     # 4 days in seconds

# DA method: 4dvar or dcwme
METHOD=${METHOD:-4dvar}

# Observation configuration
OBS_FRACTION=${OBS_FRACTION:-0.5}    # 50% of spatial points
OBS_INTERVAL=${OBS_INTERVAL:-18}     # Every 18 timesteps = 6 hours with dt=1200s

# Output directory
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH/swe4dvar/inlet_results/$SLURM_JOB_ID}

echo ""
echo "Experiment Configuration"
echo "=============================================================="
echo "Grid size:        ${NX} x ${NY}"
echo "Time step:        ${DT} seconds"
echo "Final time:       ${FINAL_TIME} seconds ($(echo "scale=1; $FINAL_TIME/86400" | bc) days)"
echo "DA Method:        ${METHOD}"
echo "Obs fraction:     ${OBS_FRACTION}"
echo "Obs interval:     ${OBS_INTERVAL} timesteps"
echo "Output dir:       ${OUTPUT_DIR}"
echo "=============================================================="

#------------------------------------------------------------------------------
# Create Output Directories
#------------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/checkpoints"
mkdir -p "${OUTPUT_DIR}/logs"

# Copy this script to output for reproducibility
cp "${BASH_SOURCE[0]}" "${OUTPUT_DIR}/"

#------------------------------------------------------------------------------
# Run the Experiment
#------------------------------------------------------------------------------
echo ""
echo "Starting experiment at $(date)"
echo ""

# Change to project directory
cd ${PROJECT_ROOT}

# Enable Intel MPI process pinning for optimal performance
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=auto

# Enable PETSc logging for detailed performance analysis
export PETSC_OPTIONS="-log_view -log_view_memory"

# Run the experiment with ibrun (TACC's MPI launcher)
# ibrun automatically handles rank placement on Frontera
ibrun python "${SCRIPT_DIR}/run_experiment.py" \
    --nx ${NX} \
    --ny ${NY} \
    --dt ${DT} \
    --final-time ${FINAL_TIME} \
    --method ${METHOD} \
    --obs-fraction ${OBS_FRACTION} \
    --obs-interval ${OBS_INTERVAL} \
    --output-dir "${OUTPUT_DIR}" \
    --profile \
    --checkpoint-dir "${OUTPUT_DIR}/checkpoints"

# Capture exit status
EXIT_STATUS=$?

#------------------------------------------------------------------------------
# Post-Processing
#------------------------------------------------------------------------------
echo ""
echo "Experiment completed at $(date)"
echo "Exit status: ${EXIT_STATUS}"

# Move SLURM output files to the experiment output directory
mv inlet_${SLURM_JOB_ID}_*.out "${OUTPUT_DIR}/logs/" 2>/dev/null
mv inlet_${SLURM_JOB_ID}_*.err "${OUTPUT_DIR}/logs/" 2>/dev/null

# Generate summary
if [ ${EXIT_STATUS} -eq 0 ]; then
    echo ""
    echo "=============================================================="
    echo "SUCCESS: Experiment completed successfully"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "=============================================================="
else
    echo ""
    echo "=============================================================="
    echo "FAILED: Experiment failed with exit code ${EXIT_STATUS}"
    echo "Check error logs: ${OUTPUT_DIR}/logs/"
    echo "=============================================================="
fi

exit ${EXIT_STATUS}

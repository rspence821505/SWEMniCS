#!/bin/bash
#=============================================================================
# Shinnecock Inlet 4D-Var Data Assimilation on TACC Frontera
#=============================================================================
#
# SLURM batch script for running Shinnecock Inlet simulations with optional
# data assimilation (4D-Var or DC-WME) on TACC Frontera.
#
# Frontera Specifications:
#   - 8,368 compute nodes (CLX: Cascade Lake)
#   - 56 cores per node (2x Intel Xeon Platinum 8280)
#   - 192 GB RAM per node (~3.4 GB per core)
#   - 100 Gbps HDR InfiniBand interconnect
#
# Usage:
#   sbatch job_submit.sh                    # Default: 1 node forward run
#   sbatch --nodes=4 job_submit.sh          # 4 nodes (224 ranks)
#   DA_MODE=4dvar sbatch job_submit.sh      # Run 4D-Var DA
#   DA_MODE=dcwme NODES=8 sbatch job_submit.sh  # DC-WME on 8 nodes
#
#=============================================================================

#SBATCH --job-name=shinnecock_da      # Job name
#SBATCH --output=shinnecock_%j.out    # Standard output (%j = job ID)
#SBATCH --error=shinnecock_%j.err     # Standard error
#SBATCH --partition=normal            # Queue (normal, development, large)
#SBATCH --nodes=1                     # Number of nodes (override with --nodes=N)
#SBATCH --ntasks-per-node=56          # Tasks per node (full node)
#SBATCH --time=04:00:00               # Wall time (HH:MM:SS)
#SBATCH --account=YOUR_ALLOCATION     # TACC allocation (REQUIRED - replace this!)
#SBATCH --mail-type=ALL               # Email notifications
#SBATCH --mail-user=your@email.edu    # Email address (replace this!)

#=============================================================================
# Configuration (can be overridden via environment variables)
#=============================================================================

# Data assimilation mode: none, 4dvar, dcwme
DA_MODE=${DA_MODE:-none}

# Simulation parameters
DT=${DT:-600}                    # Time step in seconds
T_HOURS=${T_HOURS:-24}           # Simulation time in hours
SOLVER=${SOLVER:-dg}             # Solver: dg, supg, dgnc
ALPHA=${ALPHA:-1.5}              # Wetting/drying parameter

# DA-specific parameters (only used if DA_MODE != none)
OBS_FRACTION=${OBS_FRACTION:-0.5}      # Fraction of points to observe
OBS_FREQUENCY=${OBS_FREQUENCY:-6}      # Observe every N timesteps (~1 hour)
OBS_NOISE=${OBS_NOISE:-0.01}           # Observation noise level
BACKGROUND_ERROR=${BACKGROUND_ERROR:-0.1}  # Background error std
MAX_DA_ITER=${MAX_DA_ITER:-50}         # Max optimization iterations

# Output configuration
OUTPUT_PREFIX=${OUTPUT_PREFIX:-shinnecock_frontera}
VERBOSE=${VERBOSE:-true}
PROFILE=${PROFILE:-true}

#=============================================================================
# Environment Setup
#=============================================================================

echo "============================================================"
echo "Shinnecock Inlet Simulation on Frontera"
echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Nodes:         $SLURM_JOB_NUM_NODES"
echo "Tasks/Node:    $SLURM_NTASKS_PER_NODE"
echo "Total Tasks:   $SLURM_NTASKS"
echo "Start Time:    $(date)"
echo "Working Dir:   $SLURM_SUBMIT_DIR"
echo "============================================================"

# Source environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/environment_setup.sh"

# Change to project directory
cd "$SWE4DVAR_ROOT" || exit 1

#=============================================================================
# Verify Data Files
#=============================================================================

echo ""
echo "Verifying Shinnecock data files..."

# Check for required ADIOS files
ADIOS_BASE="${SWE4DVAR_ROOT}/examples/data/shinnecock_inlet"
REQUIRED_FILES=(
    "${ADIOS_BASE}_mesh.bp"
    "${ADIOS_BASE}_depth.bp"
    "${ADIOS_BASE}_boundary.json"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -e "$file" ]]; then
        echo "  [OK] Found: $(basename $file)"
    else
        echo "  [ERROR] Missing: $file"
        echo ""
        echo "Shinnecock data files not found!"
        echo "Please ensure the ADCIRC-to-ADIOS conversion has been run."
        echo "See: https://github.com/your-repo/swe4dvar/docs/data_preparation.md"
        exit 1
    fi
done

echo "All data files verified."

#=============================================================================
# Pre-flight Memory Check
#=============================================================================

echo ""
echo "Memory configuration:"
TOTAL_MEM_GB=192
MEM_PER_TASK=$((TOTAL_MEM_GB * 1024 / SLURM_NTASKS_PER_NODE))
echo "  Total per node: ${TOTAL_MEM_GB} GB"
echo "  Per task: ${MEM_PER_TASK} MB"

# Shinnecock mesh is larger than idealized cases - need more memory
# Estimate: ~50 MB per DOF, Shinnecock has ~5000 nodes, 3 DOFs per node
# Total state size: ~75 MB, trajectory storage: 75 MB * nt
# For DA with Jacobians: additional ~500 MB per timestep

if [[ "$DA_MODE" != "none" ]]; then
    echo ""
    echo "WARNING: DA mode requires significant memory for Jacobian storage."
    echo "Recommended: At least 2 GB per MPI rank for DA experiments."
    if [[ $MEM_PER_TASK -lt 2048 ]]; then
        echo "Consider using fewer tasks per node for DA experiments."
    fi
fi

#=============================================================================
# Build Command
#=============================================================================

# Build the Python command
PYTHON_CMD="python ${SWE4DVAR_ROOT}/examples/shinnecock.py"
PYTHON_CMD+=" --dt $DT"
PYTHON_CMD+=" --T $T_HOURS"
PYTHON_CMD+=" --solver $SOLVER"
PYTHON_CMD+=" --alpha $ALPHA"
PYTHON_CMD+=" --output-prefix $OUTPUT_PREFIX"
PYTHON_CMD+=" --da-mode $DA_MODE"

# Add DA-specific options
if [[ "$DA_MODE" != "none" ]]; then
    PYTHON_CMD+=" --obs-fraction $OBS_FRACTION"
    PYTHON_CMD+=" --obs-frequency $OBS_FREQUENCY"
    PYTHON_CMD+=" --obs-noise $OBS_NOISE"
    PYTHON_CMD+=" --background-error $BACKGROUND_ERROR"
    PYTHON_CMD+=" --max-da-iter $MAX_DA_ITER"
fi

# Add optional flags
if [[ "$VERBOSE" == "true" ]]; then
    PYTHON_CMD+=" --verbose"
fi

if [[ "$PROFILE" == "true" ]]; then
    PYTHON_CMD+=" --profile"
fi

# Suppress Newton console output for parallel runs (file logging instead)
PYTHON_CMD+=" --newton-quiet"
PYTHON_CMD+=" --newton-log ${OUTPUT_DIR}/newton_${SLURM_JOB_ID}.log"
PYTHON_CMD+=" --newton-store"

#=============================================================================
# Run Simulation
#=============================================================================

echo ""
echo "============================================================"
echo "Running Shinnecock Simulation"
echo "============================================================"
echo "DA Mode:       $DA_MODE"
echo "Time Step:     $DT s"
echo "Duration:      $T_HOURS hours"
echo "Solver:        $SOLVER"
echo "Output Prefix: $OUTPUT_PREFIX"
echo ""
echo "Command:"
echo "  ibrun $PYTHON_CMD"
echo "============================================================"
echo ""

# Create output directory if needed
mkdir -p "${OUTPUT_DIR}"

# Record start time
START_TIME=$(date +%s)

# Run with ibrun (Frontera's MPI launcher)
ibrun $PYTHON_CMD

# Capture exit code
EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

#=============================================================================
# Post-processing and Summary
#=============================================================================

echo ""
echo "============================================================"
echo "Job Summary"
echo "============================================================"
echo "Exit Code:     $EXIT_CODE"
echo "Wall Time:     ${ELAPSED} seconds ($((ELAPSED / 60)) minutes)"
echo "End Time:      $(date)"
echo "============================================================"

# List output files
if [[ -d "${OUTPUT_DIR}" ]]; then
    echo ""
    echo "Output files in ${OUTPUT_DIR}:"
    ls -lh "${OUTPUT_DIR}/${OUTPUT_PREFIX}"* 2>/dev/null || echo "  (no matching files)"
fi

# Check for errors
if [[ $EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "WARNING: Job completed with non-zero exit code: $EXIT_CODE"
    echo "Check error log: shinnecock_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE

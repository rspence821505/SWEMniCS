#!/bin/bash
# ============================================================================
# Run Parallel DA Experiments
# ============================================================================
# This script runs all parallel data assimilation experiments with MPI.
#
# Usage:
#   ./run_parallel_experiments.sh [--nprocs N] [--profile] [--verbose]
#
# Options:
#   --nprocs N    Number of MPI processes (default: 4)
#   --profile     Enable detailed timing profiling
#   --verbose     Enable verbose output
#   --help        Show this help message
#
# Example:
#   ./run_parallel_experiments.sh --nprocs 4 --profile
# ============================================================================

set -e  # Exit on error

# Default values
NPROCS=4
PROFILE=""
VERBOSE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nprocs)
            NPROCS="$2"
            shift 2
            ;;
        --profile)
            PROFILE="--profile"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help)
            echo "Usage: $0 [--nprocs N] [--profile] [--verbose]"
            echo ""
            echo "Options:"
            echo "  --nprocs N    Number of MPI processes (default: 4)"
            echo "  --profile     Enable detailed timing profiling"
            echo "  --verbose     Enable verbose output"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================================"
echo "Parallel Data Assimilation Experiments"
echo "============================================================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Script directory: ${SCRIPT_DIR}"
echo "MPI processes: ${NPROCS}"
echo "Profile: ${PROFILE:-disabled}"
echo "Verbose: ${VERBOSE:-disabled}"
echo "============================================================================"
echo ""

# Ensure we're in the project root for imports
cd "${PROJECT_ROOT}"

# Check if mpirun is available
if ! command -v mpirun &> /dev/null; then
    echo "Error: mpirun not found. Please ensure MPI is installed and in PATH."
    exit 1
fi

# Create output directories
mkdir -p outputs/data outputs/figures outputs/logs

echo "Running parallel experiments with ${NPROCS} MPI ranks..."
echo ""

# Track overall timing
START_TIME=$(date +%s)

# ============================================================================
# Tidal 4D-Var
# ============================================================================
echo "----------------------------------------"
echo "1. Tidal 4D-Var (Parallel)"
echo "----------------------------------------"
mpirun -n ${NPROCS} python "${SCRIPT_DIR}/tidal_4dvar_mpi.py" \
    --nx 10 --ny 5 --dt 3600 --final-time 86400 \
    ${PROFILE} ${VERBOSE} 2>&1 | tee outputs/logs/tidal_4dvar_mpi.log
echo ""

# ============================================================================
# Tidal DC-WME-4DVar
# ============================================================================
echo "----------------------------------------"
echo "2. Tidal DC-WME-4DVar (Parallel)"
echo "----------------------------------------"
mpirun -n ${NPROCS} python "${SCRIPT_DIR}/tidal_dcwme_mpi.py" \
    --nx 10 --ny 5 --dt 3600 --final-time 86400 \
    ${PROFILE} ${VERBOSE} 2>&1 | tee outputs/logs/tidal_dcwme_mpi.log
echo ""

# ============================================================================
# Dam Break 4D-Var
# ============================================================================
echo "----------------------------------------"
echo "3. Dam Break 4D-Var (Parallel)"
echo "----------------------------------------"
mpirun -n ${NPROCS} python "${SCRIPT_DIR}/dam_break_4dvar_mpi.py" \
    --nx 30 --ny 30 --dt 0.5 --final-time 20 --solver DG \
    ${PROFILE} ${VERBOSE} 2>&1 | tee outputs/logs/dam_break_4dvar_mpi.log
echo ""

# ============================================================================
# Dam Break DC-WME-4DVar
# ============================================================================
echo "----------------------------------------"
echo "4. Dam Break DC-WME-4DVar (Parallel)"
echo "----------------------------------------"
mpirun -n ${NPROCS} python "${SCRIPT_DIR}/dam_break_dcwme_mpi.py" \
    --nx 30 --ny 30 --dt 0.5 --final-time 20 --solver DG \
    ${PROFILE} ${VERBOSE} 2>&1 | tee outputs/logs/dam_break_dcwme_mpi.log
echo ""

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "============================================================================"
echo "All parallel experiments completed!"
echo "============================================================================"
echo "Total execution time: ${TOTAL_TIME} seconds"
echo ""
echo "Results saved to:"
echo "  - outputs/data/*.json"
echo "  - outputs/logs/*.log"
echo "============================================================================"

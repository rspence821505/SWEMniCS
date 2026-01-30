#!/bin/bash
# Run serial data assimilation experiments comparing 4D-Var vs DC-WME-4DVar
#
# This script runs all four experiments:
#   1. Tidal case with standard 4D-Var
#   2. Tidal case with DC-WME-4DVar
#   3. Dam break case with standard 4D-Var
#   4. Dam break case with DC-WME-4DVar
#
# Usage:
#   ./run_serial_experiments.sh [--quick] [--verbose]
#
# Options:
#   --quick    Use smaller grids and shorter time windows for testing
#   --verbose  Enable verbose output from experiments

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Parse arguments
QUICK_MODE=false
VERBOSE=""
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
    esac
done

# Configuration
if [ "$QUICK_MODE" = true ]; then
    echo "Running in QUICK mode (smaller grids, shorter time windows)"
    TIDAL_NX=5
    TIDAL_NY=3
    TIDAL_DT=7200
    TIDAL_FINAL=43200  # 12 hours

    DAM_NX=15
    DAM_NY=15
    DAM_DT=1.0
    DAM_FINAL=10  # 10 seconds

    MAX_ITER=20
else
    echo "Running in FULL mode"
    TIDAL_NX=10
    TIDAL_NY=5
    TIDAL_DT=3600
    TIDAL_FINAL=86400  # 24 hours

    DAM_NX=30
    DAM_NY=30
    DAM_DT=0.5
    DAM_FINAL=20  # 20 seconds

    MAX_ITER=50
fi

# Change to project root for imports
cd "$PROJECT_ROOT"

# Create output directories
mkdir -p outputs/data outputs/figures

echo ""
echo "========================================================================"
echo "Serial DA Experiments: 4D-Var vs DC-WME-4DVar"
echo "========================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Output directory: $PROJECT_ROOT/outputs"
echo ""

# Track overall timing
TOTAL_START=$(date +%s)

# ============================================================================
# Experiment 1: Tidal 4D-Var
# ============================================================================
echo ""
echo "========================================================================"
echo "Experiment 1/4: Tidal Case - Standard 4D-Var"
echo "========================================================================"
EXP1_START=$(date +%s)

python experiments/serial_da/tidal_4dvar.py \
    --nx $TIDAL_NX \
    --ny $TIDAL_NY \
    --dt $TIDAL_DT \
    --final-time $TIDAL_FINAL \
    --max-iter $MAX_ITER \
    $VERBOSE

EXP1_END=$(date +%s)
echo "Experiment 1 completed in $((EXP1_END - EXP1_START)) seconds"

# ============================================================================
# Experiment 2: Tidal DC-WME-4DVar
# ============================================================================
echo ""
echo "========================================================================"
echo "Experiment 2/4: Tidal Case - DC-WME-4DVar"
echo "========================================================================"
EXP2_START=$(date +%s)

python experiments/serial_da/tidal_dcwme.py \
    --nx $TIDAL_NX \
    --ny $TIDAL_NY \
    --dt $TIDAL_DT \
    --final-time $TIDAL_FINAL \
    --max-iter $MAX_ITER \
    $VERBOSE

EXP2_END=$(date +%s)
echo "Experiment 2 completed in $((EXP2_END - EXP2_START)) seconds"

# ============================================================================
# Experiment 3: Dam Break 4D-Var
# ============================================================================
echo ""
echo "========================================================================"
echo "Experiment 3/4: Dam Break Case - Standard 4D-Var"
echo "========================================================================"
EXP3_START=$(date +%s)

python experiments/serial_da/dam_break_4dvar.py \
    --nx $DAM_NX \
    --ny $DAM_NY \
    --dt $DAM_DT \
    --final-time $DAM_FINAL \
    --max-iter $MAX_ITER \
    --solver DG \
    $VERBOSE

EXP3_END=$(date +%s)
echo "Experiment 3 completed in $((EXP3_END - EXP3_START)) seconds"

# ============================================================================
# Experiment 4: Dam Break DC-WME-4DVar
# ============================================================================
echo ""
echo "========================================================================"
echo "Experiment 4/4: Dam Break Case - DC-WME-4DVar"
echo "========================================================================"
EXP4_START=$(date +%s)

python experiments/serial_da/dam_break_dcwme.py \
    --nx $DAM_NX \
    --ny $DAM_NY \
    --dt $DAM_DT \
    --final-time $DAM_FINAL \
    --max-iter $MAX_ITER \
    --solver DG \
    $VERBOSE

EXP4_END=$(date +%s)
echo "Experiment 4 completed in $((EXP4_END - EXP4_START)) seconds"

# ============================================================================
# Analysis
# ============================================================================
echo ""
echo "========================================================================"
echo "Running Comparison Analysis"
echo "========================================================================"
ANALYSIS_START=$(date +%s)

python experiments/serial_da/analyze_results.py

ANALYSIS_END=$(date +%s)
echo "Analysis completed in $((ANALYSIS_END - ANALYSIS_START)) seconds"

# ============================================================================
# Summary
# ============================================================================
TOTAL_END=$(date +%s)
echo ""
echo "========================================================================"
echo "All Experiments Complete"
echo "========================================================================"
echo "Total runtime: $((TOTAL_END - TOTAL_START)) seconds"
echo ""
echo "Individual experiment times:"
echo "  Tidal 4D-Var:        $((EXP1_END - EXP1_START)) seconds"
echo "  Tidal DC-WME:        $((EXP2_END - EXP2_START)) seconds"
echo "  Dam Break 4D-Var:    $((EXP3_END - EXP3_START)) seconds"
echo "  Dam Break DC-WME:    $((EXP4_END - EXP4_START)) seconds"
echo "  Analysis:            $((ANALYSIS_END - ANALYSIS_START)) seconds"
echo ""
echo "Results saved to:"
echo "  - outputs/data/serial_da_results.json"
echo "  - outputs/figures/"
echo ""
echo "========================================================================"

#!/bin/bash
# Run all notebooks in the notebooks/ directory
# Usage: ./scripts/run_all_notebooks.sh [--stop-on-error]
#
# This script executes all Jupyter notebooks in the notebooks/ directory
# and reports success/failure status for each.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTEBOOKS_DIR="$REPO_ROOT/notebooks"
OUTPUT_DIR="$REPO_ROOT/outputs/logs"

# Parse arguments
STOP_ON_ERROR=false
if [[ "$1" == "--stop-on-error" ]]; then
    STOP_ON_ERROR=true
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0
FAILED_NOTEBOOKS=()

echo "========================================"
echo "Running all notebooks in $NOTEBOOKS_DIR"
echo "========================================"
echo ""

# List of notebooks in recommended execution order
NOTEBOOKS=(
    "problems.ipynb"
    "solvers.ipynb"
    "solver_storage.ipynb"
    "timestep_manager.ipynb"
    "station_manager.ipynb"
    "metrics.ipynb"
    "observation_operator.ipynb"
    "tangent_linear.ipynb"
    "implicit_adjoint.ipynb"
    "checkpointing.ipynb"
    "covariance_demonstrations.ipynb"
    "cost_functions.ipynb"
    "qoi_maps.ipynb"
    "newton.ipynb"
)

for notebook in "${NOTEBOOKS[@]}"; do
    notebook_path="$NOTEBOOKS_DIR/$notebook"

    if [[ ! -f "$notebook_path" ]]; then
        echo -e "${YELLOW}SKIP${NC}: $notebook (not found)"
        continue
    fi

    echo -n "Running $notebook... "

    # Run notebook with nbconvert
    if jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=600 \
        --ExecutePreprocessor.kernel_name=python3 \
        "$notebook_path" 2>"$OUTPUT_DIR/${notebook%.ipynb}_error.log"; then
        echo -e "${GREEN}PASSED${NC}"
        ((PASSED++))
        # Remove empty error log
        rm -f "$OUTPUT_DIR/${notebook%.ipynb}_error.log"
    else
        echo -e "${RED}FAILED${NC}"
        ((FAILED++))
        FAILED_NOTEBOOKS+=("$notebook")
        echo "  Error log: $OUTPUT_DIR/${notebook%.ipynb}_error.log"

        if $STOP_ON_ERROR; then
            echo ""
            echo "Stopping due to --stop-on-error flag"
            break
        fi
    fi
done

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"

if [[ ${#FAILED_NOTEBOOKS[@]} -gt 0 ]]; then
    echo ""
    echo "Failed notebooks:"
    for nb in "${FAILED_NOTEBOOKS[@]}"; do
        echo "  - $nb"
    done
    exit 1
fi

echo ""
echo "All notebooks executed successfully!"
exit 0

#!/bin/bash
#=============================================================================
# Environment Setup for SWE4DVar on TACC Frontera
#=============================================================================
#
# This script sets up the environment for running SWE4DVar simulations
# on TACC Frontera. It handles module loads, conda activation, and
# environment variable configuration.
#
# Usage:
#   source environment_setup.sh
#
# This script can be sourced from job scripts or interactive sessions.
#=============================================================================

#=============================================================================
# Configuration Variables
#=============================================================================

# SWE4DVar installation root (adjust as needed)
export SWE4DVAR_ROOT="${SWE4DVAR_ROOT:-$WORK/swe4dvar}"

# Conda environment name
export SWE4DVAR_CONDA_ENV="${SWE4DVAR_CONDA_ENV:-swe4dvar}"

# Output directory (uses Frontera's scratch for large runs)
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/swe4dvar_outputs}"

# FEniCSx/DOLFINx version to use
export DOLFINX_VERSION="${DOLFINX_VERSION:-0.7.0}"

#=============================================================================
# Module Setup
#=============================================================================

echo "Setting up Frontera environment..."
echo "  SWE4DVAR_ROOT: $SWE4DVAR_ROOT"

# Reset modules to default state
module reset 2>/dev/null || module purge

# Load required modules for Frontera
# Note: Module names may change - check with `module spider` if errors occur

# Intel compiler and MPI (default on Frontera)
module load intel/19.1.1
module load impi/19.0.9

# CMake for building dependencies
module load cmake/3.24.2

# HDF5 with parallel I/O (required for ADIOS2)
module load phdf5/1.12.2

# PETSc with complex number support disabled (real-valued problems)
module load petsc/3.19

# ADIOS2 for mesh I/O
module load adios2/2.8.3

# Python via Anaconda
module load python3/3.9.7

echo "  Modules loaded successfully"

#=============================================================================
# Conda Environment Activation
#=============================================================================

# Initialize conda for shell
if [[ -f "$WORK/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$WORK/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "/opt/apps/intel19/python3/3.9.7/etc/profile.d/conda.sh" ]]; then
    source "/opt/apps/intel19/python3/3.9.7/etc/profile.d/conda.sh"
else
    echo "WARNING: Could not find conda initialization script"
    echo "Please ensure conda is installed and accessible"
fi

# Activate the SWE4DVar conda environment
if conda activate "$SWE4DVAR_CONDA_ENV" 2>/dev/null; then
    echo "  Activated conda environment: $SWE4DVAR_CONDA_ENV"
else
    echo "WARNING: Could not activate conda environment '$SWE4DVAR_CONDA_ENV'"
    echo "Please create it with: conda env create -f environment.yml"
    echo "Or manually install required packages"
fi

#=============================================================================
# Environment Variables for DOLFINx/FEniCSx
#=============================================================================

# Ensure DOLFINx can find PETSc
export PETSC_DIR="${TACC_PETSC_DIR:-/opt/apps/intel19/impi19_0/petsc/3.19}"
export PETSC_ARCH=""

# MPI configuration
export OMPI_MCA_btl="^openib"  # Disable deprecated BTL
export I_MPI_FABRICS="shm:ofi"  # Use shared memory and OFI (Frontera)

# Optimize MPI for Frontera's HDR InfiniBand
export I_MPI_OFI_PROVIDER="mlx"
export FI_PROVIDER="mlx"

# Thread configuration (1 thread per MPI rank for pure MPI)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# PETSc options for optimal performance
export PETSC_OPTIONS="-malloc_debug 0 -log_view"

# DOLFINx JIT compilation cache
export FENICS_CACHE_DIR="${SCRATCH}/.fenics_cache"
mkdir -p "${FENICS_CACHE_DIR}"

# XDG cache for additional DOLFINx caching
export XDG_CACHE_HOME="${SCRATCH}/.cache"
mkdir -p "${XDG_CACHE_HOME}"

#=============================================================================
# Memory and Performance Settings
#=============================================================================

# Increase stack size limit (needed for large problems)
ulimit -s unlimited 2>/dev/null || true

# Memory locking (helps with InfiniBand)
ulimit -l unlimited 2>/dev/null || true

# Disable core dumps (save space on scratch)
ulimit -c 0 2>/dev/null || true

#=============================================================================
# ADIOS2 Configuration for Real Data
#=============================================================================

# ADIOS2 engine for reading Shinnecock mesh/data
export ADIOS2_ENGINE="BP4"

# Data directory containing Shinnecock files
export SHINNECOCK_DATA_DIR="${SWE4DVAR_ROOT}/examples/data"

# Verify data files exist
verify_data_files() {
    local base="${SHINNECOCK_DATA_DIR}/shinnecock_inlet"
    local missing=0

    for ext in "_mesh.bp" "_depth.bp" "_boundary.json"; do
        if [[ ! -e "${base}${ext}" ]]; then
            echo "WARNING: Missing data file: ${base}${ext}"
            missing=1
        fi
    done

    if [[ $missing -eq 1 ]]; then
        echo ""
        echo "Shinnecock data files are missing!"
        echo "Please run the ADCIRC-to-ADIOS conversion script or"
        echo "copy pre-converted files to: $SHINNECOCK_DATA_DIR"
        return 1
    fi

    return 0
}

#=============================================================================
# Output Directory Setup
#=============================================================================

# Create output directories
mkdir -p "${OUTPUT_DIR}/data"
mkdir -p "${OUTPUT_DIR}/figures"
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/checkpoints"

# Set SWE4DVar output directory
export SWE4DVAR_OUTPUT_DIR="${OUTPUT_DIR}"

echo "  Output directory: $OUTPUT_DIR"

#=============================================================================
# Python Path Configuration
#=============================================================================

# Add SWE4DVar to Python path
export PYTHONPATH="${SWE4DVAR_ROOT}/src:${PYTHONPATH}"

# Add experiments to path (for DA utilities)
export PYTHONPATH="${SWE4DVAR_ROOT}/experiments:${PYTHONPATH}"

#=============================================================================
# Verification
#=============================================================================

verify_environment() {
    echo ""
    echo "Verifying environment setup..."

    # Check Python
    if command -v python &>/dev/null; then
        echo "  Python: $(python --version 2>&1)"
    else
        echo "  ERROR: Python not found"
        return 1
    fi

    # Check MPI
    if command -v mpirun &>/dev/null; then
        echo "  MPI: $(which mpirun)"
    else
        echo "  WARNING: mpirun not found (ibrun will be used on Frontera)"
    fi

    # Check required Python packages
    local packages=("mpi4py" "petsc4py" "dolfinx" "numpy" "adios4dolfinx")
    for pkg in "${packages[@]}"; do
        if python -c "import $pkg" 2>/dev/null; then
            echo "  $pkg: OK"
        else
            echo "  WARNING: $pkg not found"
        fi
    done

    # Check SWE4DVar
    if python -c "import swe4dvar" 2>/dev/null; then
        echo "  swe4dvar: OK"
    else
        echo "  WARNING: swe4dvar not found - ensure PYTHONPATH is set correctly"
    fi

    echo ""
    return 0
}

#=============================================================================
# Summary
#=============================================================================

echo ""
echo "============================================================"
echo "Environment Setup Complete"
echo "============================================================"
echo "  Modules:     intel/19.1.1, impi/19.0.9, petsc/3.19, adios2/2.8.3"
echo "  Conda Env:   $SWE4DVAR_CONDA_ENV"
echo "  SWE4DVAR:    $SWE4DVAR_ROOT"
echo "  Output:      $OUTPUT_DIR"
echo "  Threads:     OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "============================================================"

# Run verification if requested
if [[ "${VERIFY_ENV:-false}" == "true" ]]; then
    verify_environment
fi

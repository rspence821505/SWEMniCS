#!/bin/bash
#==============================================================================
# Environment Setup Script for SWE4DVar on TACC Frontera
#==============================================================================
#
# This script sets up the environment for running swe4dvar data assimilation
# experiments on Frontera. It handles:
#   - Module loading (Intel compiler, MPI, Python)
#   - Conda environment activation
#   - Path configuration
#   - Environment verification
#
# Usage:
#   source environment_setup.sh        # Source to set up environment
#   ./environment_setup.sh --install   # First-time installation
#
#==============================================================================

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

# Name of the conda environment
CONDA_ENV_NAME="swe4dvar"

# Path to project root (adjust if necessary)
# This assumes the HPC scripts are in hpc/frontera/idealized_inlet/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Frontera-specific paths
export SCRATCH=${SCRATCH:-/scratch1/$(whoami)}
export WORK=${WORK:-/work2/$(id -gn)/$(whoami)}

#------------------------------------------------------------------------------
# Module Setup
#------------------------------------------------------------------------------

setup_modules() {
    echo "Loading Frontera modules..."

    # Start with a clean environment
    module purge

    # Load Intel compiler suite (includes icc, icpc, ifort)
    # Intel 19.1.1 is well-tested on Frontera
    module load intel/19.1.1

    # Load Intel MPI for distributed computing
    module load impi/19.0.9

    # Load Python 3 (system Python for base, or use conda)
    module load python3/3.9.7

    # Optional: Load HDF5 for mesh I/O (parallel version)
    module load phdf5/1.10.4

    # Optional: Load PETSc if using system installation
    # Uncomment if you have a system PETSc build
    # module load petsc/3.16

    echo "Loaded modules:"
    module list
}

#------------------------------------------------------------------------------
# Conda Environment Setup
#------------------------------------------------------------------------------

setup_conda() {
    echo "Setting up conda environment..."

    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        # Try to initialize conda from common locations
        if [ -f "${WORK}/miniconda3/etc/profile.d/conda.sh" ]; then
            source "${WORK}/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
            source "${HOME}/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "/opt/apps/intel19/python3/3.9.7/etc/profile.d/conda.sh" ]; then
            source "/opt/apps/intel19/python3/3.9.7/etc/profile.d/conda.sh"
        else
            echo "WARNING: Conda not found. Install with: ./environment_setup.sh --install"
            return 1
        fi
    fi

    # Activate the swe4dvar environment
    if conda activate ${CONDA_ENV_NAME} 2>/dev/null; then
        echo "Activated conda environment: ${CONDA_ENV_NAME}"
    else
        echo "WARNING: Conda environment '${CONDA_ENV_NAME}' not found."
        echo "Create with: ./environment_setup.sh --install"
        return 1
    fi
}

#------------------------------------------------------------------------------
# Path Configuration
#------------------------------------------------------------------------------

setup_paths() {
    echo "Configuring paths..."

    # Add project source to Python path
    export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

    # Set data paths for mesh files
    export SWE4DVAR_DATA_DIR="${PROJECT_ROOT}/data"

    # Set output paths (use SCRATCH for large outputs)
    export SWE4DVAR_OUTPUT_DIR="${SCRATCH}/swe4dvar/outputs"
    mkdir -p "${SWE4DVAR_OUTPUT_DIR}"

    # PETSc configuration for optimal performance
    export PETSC_DIR=${PETSC_DIR:-"${WORK}/petsc"}
    export PETSC_ARCH=${PETSC_ARCH:-"arch-linux-c-opt"}

    # Intel MPI settings for Frontera
    export I_MPI_FABRICS=shm:ofi
    export I_MPI_OFI_PROVIDER=mlx

    # OMP settings (usually 1 for pure MPI)
    export OMP_NUM_THREADS=1
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close

    echo "PROJECT_ROOT:     ${PROJECT_ROOT}"
    echo "PYTHONPATH:       ${PYTHONPATH}"
    echo "Data directory:   ${SWE4DVAR_DATA_DIR}"
    echo "Output directory: ${SWE4DVAR_OUTPUT_DIR}"
}

#------------------------------------------------------------------------------
# Environment Verification
#------------------------------------------------------------------------------

verify_environment() {
    echo ""
    echo "Verifying environment..."
    local errors=0

    # Check Python
    if command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1)
        echo "  Python: ${PYTHON_VERSION}"
    else
        echo "  ERROR: Python not found"
        ((errors++))
    fi

    # Check MPI
    if command -v mpirun &> /dev/null; then
        MPI_VERSION=$(mpirun --version 2>&1 | head -1)
        echo "  MPI: ${MPI_VERSION}"
    else
        echo "  ERROR: MPI not found"
        ((errors++))
    fi

    # Check mpi4py
    if python -c "import mpi4py" 2>/dev/null; then
        MPI4PY_VERSION=$(python -c "import mpi4py; print(mpi4py.__version__)")
        echo "  mpi4py: ${MPI4PY_VERSION}"
    else
        echo "  ERROR: mpi4py not found"
        ((errors++))
    fi

    # Check petsc4py
    if python -c "import petsc4py" 2>/dev/null; then
        PETSC4PY_VERSION=$(python -c "import petsc4py; print(petsc4py.__version__)")
        echo "  petsc4py: ${PETSC4PY_VERSION}"
    else
        echo "  ERROR: petsc4py not found"
        ((errors++))
    fi

    # Check dolfinx
    if python -c "import dolfinx" 2>/dev/null; then
        DOLFINX_VERSION=$(python -c "import dolfinx; print(dolfinx.__version__)" 2>/dev/null || echo "unknown")
        echo "  dolfinx: ${DOLFINX_VERSION}"
    else
        echo "  ERROR: dolfinx not found"
        ((errors++))
    fi

    # Check swe4dvar
    if python -c "import swe4dvar" 2>/dev/null; then
        echo "  swe4dvar: installed"
    else
        echo "  WARNING: swe4dvar not importable (may need PYTHONPATH)"
    fi

    # Check mesh data file
    MESH_FILE="${SWE4DVAR_DATA_DIR}/Ideal_Inlet/Ideal_Inlet.xdmf"
    if [ -f "${MESH_FILE}" ]; then
        echo "  Mesh file: found"
    else
        echo "  WARNING: Mesh file not found at ${MESH_FILE}"
    fi

    echo ""
    if [ ${errors} -eq 0 ]; then
        echo "Environment verification: PASSED"
        return 0
    else
        echo "Environment verification: FAILED (${errors} errors)"
        return 1
    fi
}

#------------------------------------------------------------------------------
# Installation Function
#------------------------------------------------------------------------------

install_environment() {
    echo "=============================================================="
    echo "Installing SWE4DVar Environment on Frontera"
    echo "=============================================================="

    # Load modules first
    setup_modules

    # Install Miniconda if not present
    if ! command -v conda &> /dev/null; then
        echo ""
        echo "Installing Miniconda..."
        MINICONDA_DIR="${WORK}/miniconda3"

        if [ ! -d "${MINICONDA_DIR}" ]; then
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
            bash /tmp/miniconda.sh -b -p "${MINICONDA_DIR}"
            rm /tmp/miniconda.sh
        fi

        source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
    fi

    # Create conda environment
    echo ""
    echo "Creating conda environment: ${CONDA_ENV_NAME}"

    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Environment ${CONDA_ENV_NAME} already exists. Updating..."
        conda activate ${CONDA_ENV_NAME}
    else
        # Create environment with FEniCSx from conda-forge
        conda create -y -n ${CONDA_ENV_NAME} -c conda-forge \
            python=3.10 \
            fenics-dolfinx \
            petsc4py \
            mpi4py \
            numpy \
            scipy \
            matplotlib \
            h5py

        conda activate ${CONDA_ENV_NAME}
    fi

    # Install additional packages
    echo ""
    echo "Installing additional packages..."
    pip install --no-cache-dir \
        meshio \
        pyvista \
        tqdm

    # Install swe4dvar in development mode
    echo ""
    echo "Installing swe4dvar..."
    cd "${PROJECT_ROOT}"
    pip install -e .

    # Verify installation
    echo ""
    setup_paths
    verify_environment

    echo ""
    echo "=============================================================="
    echo "Installation complete!"
    echo ""
    echo "To use this environment in future sessions:"
    echo "  source ${SCRIPT_DIR}/environment_setup.sh"
    echo "=============================================================="
}

#------------------------------------------------------------------------------
# Main Entry Point
#------------------------------------------------------------------------------

# Check if running interactively or being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed directly
    case "$1" in
        --install|-i)
            install_environment
            ;;
        --verify|-v)
            setup_modules
            setup_conda
            setup_paths
            verify_environment
            ;;
        --help|-h)
            echo "Usage: source environment_setup.sh     # Set up environment"
            echo "       ./environment_setup.sh --install # First-time installation"
            echo "       ./environment_setup.sh --verify  # Verify environment"
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
else
    # Script is being sourced
    setup_modules
    setup_conda
    setup_paths
fi

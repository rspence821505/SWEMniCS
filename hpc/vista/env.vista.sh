#!/bin/bash
# env.vista.sh — activate the fenics-vista conda environment on TACC Vista.
#
# Vista architecture: aarch64 (NVIDIA Grace Neoverse-V2), 144 cores/node.
# Unlike LS6 (Intel x86_64 + Intel MPI + system DOLFINx), Vista's DOLFINx
# stack comes entirely from conda-forge — no system-installed dolfinx/basix/
# ffcx exists on this system. This script assumes the fenics-vista conda env
# has already been created (see hpc/vista/VISTA_PLAYBOOK.md for setup).
#
# Versions (as of 2026-04-23 env create):
#   - Python 3.12.13 (conda-forge)
#   - OpenMPI 5.0.10 (conda-bundled — NOT Vista's system openmpi/5.0.5)
#   - PETSc 3.25.0 (conda, compatible with dolfinx 0.10)
#   - DOLFINx 0.10.0, basix 0.10.0, ffcx 0.10.0
#
# IMPORTANT: conda-bundled OpenMPI is single-node only. For multi-node runs,
# swap to a system-MPI-linked mpi4py (see VISTA_PLAYBOOK §Multi-node).
# -----------------------------------------------------------------------------

# Guard against double-loading (set -u safe via :-)
if [[ -n "${SWEMNICS_VISTA_ENV_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi

# Conda initialization (miniforge3 installed at $WORK/miniforge3)
if [[ -z "${WORK:-}" ]]; then
  echo "ERROR: \$WORK is unset. Are you on Vista?" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "$WORK/miniforge3/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: miniforge3 not found at $WORK/miniforge3" >&2
  echo "       Run the install step in hpc/vista/VISTA_PLAYBOOK.md first." >&2
  return 1 2>/dev/null || exit 1
fi

source "$WORK/miniforge3/etc/profile.d/conda.sh"
conda activate fenics-vista

# Sanity: confirm the env is really active (set -u safe via :-)
if [[ "$(basename "${CONDA_PREFIX:-/none}")" != "fenics-vista" ]]; then
  echo "ERROR: fenics-vista env did not activate (CONDA_PREFIX=${CONDA_PREFIX:-unset})" >&2
  return 1 2>/dev/null || exit 1
fi

# Override $CC / $CXX so FFCX JIT compiles its generated C code with the
# conda-bundled gcc. Vista's login modules default $CC to NVIDIA's `nvc`,
# which cannot compile FFCX's C output (uses gcc-specific builtins).
if [[ -x "${CONDA_PREFIX:-}/bin/gcc" ]]; then
  export CC="$CONDA_PREFIX/bin/gcc"
  export CXX="$CONDA_PREFIX/bin/g++"
fi

# Thread pinning reasonable for single-node multi-rank runs on 144-core Grace
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

# OpenMPI runtime hints for conda-bundled OpenMPI 5 on TACC Vista login/compute.
# Disable CUDA-aware MPI probe (we're not using GPU matrix ops in this env).
export OMPI_MCA_opal_warn_on_missing_libcuda=0

export SWEMNICS_VISTA_ENV_LOADED=1

echo "=== fenics-vista env active ==="
echo "  python: $(python --version 2>&1)"
echo "  mpi:    $(python -c 'from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0])')"
echo "  petsc:  $(python -c 'from petsc4py import PETSc; print(PETSc.Sys.getVersion())')"
echo "  dolfinx: $(python -c 'import dolfinx; print(dolfinx.__version__)')"

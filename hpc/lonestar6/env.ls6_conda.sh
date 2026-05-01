#!/bin/bash
# env.ls6_conda.sh — activate the fenics-ls6 conda environment on TACC LS6.
#
# LS6 architecture: x86_64 (AMD Milan), 128-core nodes.
# Mirrors the Vista fenics-vista env (PETSc 3.25, dolfinx 0.10, OpenMPI 5,
# MUMPS) but for x86_64. Replaces the system PETSc 3.19 / Intel MPI stack
# whenever we need 3-obs/window or other configs that hit MUMPS-on-Intel-MPI
# numerical issues.
#
# IMPORTANT: conda-bundled OpenMPI is single-node only on LS6 too. Use this
# env only for 1-node jobs (np <= 128). For multi-node, use the original
# env.ls6.sh with Intel MPI.
# -----------------------------------------------------------------------------

if [[ -n "${SWEMNICS_LS6_CONDA_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi

if [[ -z "${WORK:-}" ]]; then
  echo "ERROR: \$WORK is unset. Are you on LS6?" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "$WORK/miniforge3/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: miniforge3 not found at $WORK/miniforge3" >&2
  echo "       Run the conda env install first." >&2
  return 1 2>/dev/null || exit 1
fi

# CRITICAL: LS6 login default env loads `python3/3.9.7` and its mpi4py via
# Intel MPI 19. That puts /opt/apps/.../python3/3.9.7/.../site-packages on
# PYTHONPATH, which shadows the conda env's mpi4py (built against the
# bundled OpenMPI 5). Without clearing it, `import mpi4py` resolves to the
# system Intel MPI build inside a Python that has conda's libraries — a
# broken hybrid.
module purge 2>/dev/null || true
unset PYTHONPATH
unset PYTHONHOME

source "$WORK/miniforge3/etc/profile.d/conda.sh"
conda activate fenics-ls6

if [[ "$(basename "${CONDA_PREFIX:-/none}")" != "fenics-ls6" ]]; then
  echo "ERROR: fenics-ls6 env did not activate (CONDA_PREFIX=${CONDA_PREFIX:-unset})" >&2
  return 1 2>/dev/null || exit 1
fi

# Use conda-bundled gcc for FFCX JIT compilation if available
if [[ -x "${CONDA_PREFIX:-}/bin/gcc" ]]; then
  export CC="$CONDA_PREFIX/bin/gcc"
  export CXX="$CONDA_PREFIX/bin/g++"
fi

# Thread pinning
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

# OpenMPI: disable CUDA probe (no GPU on this env)
export OMPI_MCA_opal_warn_on_missing_libcuda=0

export SWEMNICS_LS6_CONDA_LOADED=1

echo "=== fenics-ls6 (conda) env active ==="
echo "  python:  $(python --version 2>&1)"
echo "  mpi:     $(python -c 'from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0])')"
echo "  petsc:   $(python -c 'from petsc4py import PETSc; print(PETSc.Sys.getVersion())')"
echo "  dolfinx: $(python -c 'import dolfinx; print(dolfinx.__version__)')"

#!/bin/bash
# env.ls6.sh — sourced from every sbatch and interactive session on LS6.
# Matches WORKING_SETUP.md §3 (Per-session activation) + OMP pinning.
# Phase-split variables (see docs/ls6_first_dcwme_production_run.md §0):
#   - Phase 1 (np=2):  OMP_NUM_THREADS=64  (each rank owns one full socket)
#   - Phase 2 (np=8):  OMP_NUM_THREADS=16  (each rank owns one NUMA-internal chunk)
# Set OMP_NUM_THREADS **before** sourcing this file to override the np=2 default.

module reset
module load gcc/13.2.0 impi/21.12 python/3.12.11 \
            boost/1.86.0 pugixml/1.15 phdf5/2.0.0 parmetis/4.0.3 \
            ptscotch/7.0.7-i64 adios2/2.10.2 spdlog/1.17.0 \
            basix/0.10.0.post0 ffcx/0.10.1.post0 \
            petsc/3.22 dolfinx/0.10.0.post5
source $WORK/venvs/fenics-ls6/bin/activate

# MPI-4 release libmpi.so (IMPI 21.12 lib/ is MPI-3; only release/ exports persistent collectives that PETSc 3.22 needs)
export LD_LIBRARY_PATH=$I_MPI_ROOT/lib/release:$LD_LIBRARY_PATH

# FFCx JIT defaults to clang; LS6 only has gcc
export CC=gcc CXX=g++

# Default to Phase 1 (2 MPI ranks × 64 threads = all 128 cores on one Milan node).
# Override in the sbatch script before sourcing if doing np=8.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-64}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

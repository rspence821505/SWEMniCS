# WORKING_SETUP.md — FEniCSx on Lonestar6

**Verified**: 2026-04-21 | **Host**: `login1.ls6.tacc.utexas.edu` | **User**: `tg876971`

**Final verification on login node, 4 MPI ranks** (2026-04-21):
```
$ mpiexec -n 4 python hpc/lonestar6/environment/test_mpi.py
[rank 0/4] [rank 1/4] [rank 2/4] [rank 3/4] all on login1
allreduce sum of ranks+1 = 10  (expected 10)  OK
MPI library: Intel(R) MPI Library 2021.12 for Linux* OS

$ mpiexec -n 4 python hpc/lonestar6/environment/test_dolfinx.py
dolfinx 0.10.0.post5, petsc4py (3, 22, 4), mpi4py ranks=4
Poisson solved. global max(u) = 0.072783   OK
```

This is the **minimal working configuration** for `dolfinx 0.10` on LS6.
Every line has been executed on the live system; nothing is aspirational.

If you change anything (compiler, MPI, Python, PETSc, dolfinx version) you are off this page and back in WHY_IT_BROKE.md territory.

---

## 1. Exact module stack (REQUIRED, in this order)

```bash
module reset
module load gcc/13.2.0 impi/21.12 python/3.12.11 \
            boost/1.86.0 pugixml/1.15 phdf5/2.0.0 parmetis/4.0.3 \
            ptscotch/7.0.7-i64 adios2/2.10.2 spdlog/1.17.0 \
            basix/0.10.0.post0 ffcx/0.10.1.post0 \
            petsc/3.22 dolfinx/0.10.0.post5
```

### Why so many modules?
When you build the Python bindings, pip runs CMake with `find_package(DOLFINX)`, which transitively calls `find_dependency(...)` on **every library the system libdolfinx was compiled against**:

- `MPI` (provided by `impi/21.12`) — must have MPI-4.0 persistent-collective symbols (see §"Why 21.12 not 21.11").
- `spdlog` — logging library.
- `pugixml` — XML parser.
- `Boost` (≥ 1.70) — used by some mesh/IO routines.
- `HDF5` — parallel I/O (`phdf5` = parallel HDF5 variant).
- `parmetis`, `ptscotch` — mesh partitioning.
- `ADIOS2` (≥ 2.8.1) — parallel checkpointing.
- `ufcx`, `Basix` — provided by the Python packages `fenics-ffcx`, `fenics-basix`.

If any is missing from `CMAKE_PREFIX_PATH` when we build fenics-dolfinx's pybindings, configure aborts with "Could NOT find <X>". Loading each as a module both (a) sets the `TACC_<X>_DIR` env var we add to `CMAKE_PREFIX_PATH`, and (b) guarantees the shared-library version matches what system libdolfinx was linked against (avoiding runtime ABI breaks after a successful build).

### Environment variables these modules export (verified)
```
TACC_DOLFINX_DIR=/scratch/tacc/apps/gcc13_2/impi21/dolfinx/0.10.0.post5
TACC_PETSC_DIR=/scratch/tacc/apps/gcc13_2/impi21/petsc/3.22.4/3.22.4
I_MPI_ROOT=/scratch/projects/compilers/intel24.1/oneapi/mpi/2021.12     # Intel MPI 2021.12
```

### Why `impi/21.12` and **NOT** `impi/21.11`
`module spider dolfinx/0.10.0.post5` claims both are supported prerequisites. This is misleading. The shared `petsc/3.22` C library is linked against MPI-4.0 persistent-collective symbols (`MPI_Neighbor_alltoallv_init` and friends) which are:
- **Present** in `impi/21.12`'s `libmpi.so.12` (as weak symbols)
- **Absent** in `impi/21.11`'s `libmpi.so.12` — at both `lib/` and `lib/release/`

Loading `impi/21.11` gives an `ImportError: undefined symbol: MPI_Neighbor_alltoallv_init` at `from petsc4py import PETSc`. Always use `impi/21.12`. See FAILURE_LOG.md #8.

---

## 2. Build the Python environment (one-time, ~15 min)

```bash
# 1) Load modules first — THIS ORDER MATTERS.
module reset
module load gcc/13.2.0 impi/21.12 python/3.12.11 \
            boost/1.86.0 pugixml/1.15 phdf5/2.0.0 parmetis/4.0.3 \
            ptscotch/7.0.7-i64 adios2/2.10.2 spdlog/1.17.0 \
            basix/0.10.0.post0 ffcx/0.10.1.post0 \
            petsc/3.22 dolfinx/0.10.0.post5

# 2) Create the venv in $WORK (NOT $HOME — the 20-file quota will not tolerate a venv).
VENV=$WORK/venvs/fenics-ls6
mkdir -p $(dirname $VENV)
python3 -m venv --system-site-packages $VENV
source $VENV/bin/activate

# 3) Pin build deps.
#    - Cython 3.2.x crashes on petsc4py 3.22's .pyx files ("ExpressionWriter" crash).
#    - setuptools 80+ removed `dry_run=` from distutils.util.execute, which
#      petsc4py 3.22.5 `confpetsc.py` still passes. Must use setuptools < 70.
python -m pip install --upgrade pip
python -m pip install "cython>=3.0,<3.1" "setuptools<70" wheel \
    "pybind11>=2.12" "nanobind>=2" scikit-build-core cmake ninja numpy

# 4) mpi4py linked to system Intel MPI (force source build so it picks up mpicc).
env MPICC=mpicc python -m pip install --no-cache-dir --no-build-isolation \
    --no-binary=mpi4py "mpi4py>=4.0,<5"

# 5) petsc4py linked to system petsc/3.22.
export PETSC_DIR=$TACC_PETSC_DIR
export PETSC_ARCH=
python -m pip install --no-cache-dir --no-build-isolation \
    --no-binary=petsc4py "petsc4py==3.22.*"

# 6) FEniCSx Python — IMPORTANT: basix and ffcx Python packages MUST come from
#    the GitHub tag that matches the TACC C++ modules (`basix/0.10.0.post0`,
#    `ffcx/0.10.1.post0`), NOT from PyPI. PyPI only has `fenics-basix==0.10.0`
#    which is ABI-incompatible with dolfinx built against `0.10.0.post0`.
#    (See FAILURE_LOG.md #9.)
python -m pip install "fenics-ufl==2025.2.*"   # pure python, PyPI is fine
python -m pip install --no-build-isolation \
    "git+https://github.com/FEniCS/basix.git@v0.10.0.post0#subdirectory=python"
python -m pip install --no-build-isolation \
    "git+https://github.com/FEniCS/ffcx.git@v0.10.1.post0"

# 7) dolfinx Python bindings — NOT on PyPI; install from GitHub at matching tag.
#    CMAKE_PREFIX_PATH must include spdlog too (transitive dep of DOLFINXConfig.cmake).
export DOLFINX_DIR=$TACC_DOLFINX_DIR
export CMAKE_PREFIX_PATH=$TACC_DOLFINX_DIR:$TACC_PETSC_DIR:$TACC_SPDLOG_DIR:${CMAKE_PREFIX_PATH:-}
# Re-pin setuptools — fenics-ffcx bumped it in step 6.
python -m pip install --force-reinstall "setuptools<70"
python -m pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/FEniCS/dolfinx.git@v0.10.0.post5#subdirectory=python"

# 8) Project-specific pure-Python deps (no MPI touch).
python -m pip install -r $WORK/SWEMniCS/requirements.txt
```

---

## 3. Per-session activation (every ssh, every sbatch)

```bash
module reset
module load gcc/13.2.0 impi/21.12 python/3.12.11 \
            boost/1.86.0 pugixml/1.15 phdf5/2.0.0 parmetis/4.0.3 \
            ptscotch/7.0.7-i64 adios2/2.10.2 spdlog/1.17.0 \
            basix/0.10.0.post0 ffcx/0.10.1.post0 \
            petsc/3.22 dolfinx/0.10.0.post5
source $WORK/venvs/fenics-ls6/bin/activate

# CRITICAL for direct `python` or `mpiexec python` (but NOT for `ibrun`, which handles this):
# The MPI-4.0 symbols PETSc needs live in lib/release, not lib/.
export LD_LIBRARY_PATH=$I_MPI_ROOT/lib/release:$LD_LIBRARY_PATH
```

Save that in `$WORK/SWEMniCS/env.ls6.sh` so it's one line to source:
```bash
# $WORK/SWEMniCS/env.ls6.sh
module reset
module load gcc/13.2.0 impi/21.12 python/3.12.11 \
            boost/1.86.0 pugixml/1.15 phdf5/2.0.0 parmetis/4.0.3 \
            ptscotch/7.0.7-i64 adios2/2.10.2 spdlog/1.17.0 \
            basix/0.10.0.post0 ffcx/0.10.1.post0 \
            petsc/3.22 dolfinx/0.10.0.post5
source $WORK/venvs/fenics-ls6/bin/activate
```
Then every sbatch preamble is: `source $WORK/SWEMniCS/env.ls6.sh`.

---

## 4. Verification commands (copy-paste, should all pass)

```bash
# Activate env first as above, then:
python -c "import dolfinx; print(dolfinx.__version__)"
python -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"
python -c "from mpi4py import MPI; print(MPI.Get_library_version()[:80])"
python -c "import basix, ufl, ffcx; print(basix.__version__, ufl.__version__, ffcx.__version__)"

# Multi-rank via ibrun (LOGIN NODE, 4 ranks — ~1 sec, OK under TACC conduct policy)
cd $WORK/SWEMniCS
ibrun -n 4 python hpc/lonestar6/environment/test_mpi.py
ibrun -n 4 python hpc/lonestar6/environment/test_dolfinx.py

# Compute-node smoke (dry-run first, free):
sbatch --test-only hpc/lonestar6/environment/test_cpu.slurm
# If that passes:
sbatch hpc/lonestar6/environment/test_cpu.slurm
```

Expected `mpi4py` library line: starts with `Intel(R) MPI Library 2021.11` (or 21.12). If it says MPICH or OpenMPI, the module stack is wrong.

---

## 5. What each sbatch script MUST contain

```bash
#!/bin/bash
#SBATCH -A ADCIRC -p <partition> -N <n> -n <tasks> -t <hh:mm:ss>
#SBATCH -o %x.%j.out

set -euxo pipefail

source $WORK/SWEMniCS/env.ls6.sh     # modules + venv

# PROVENANCE — copy/paste into every sbatch
module list
pip show fenics-dolfinx petsc4py mpi4py | grep -E "^(Name|Version)"
echo "TACC_DOLFINX_DIR=$TACC_DOLFINX_DIR  TACC_PETSC_DIR=$TACC_PETSC_DIR"
echo "I_MPI_ROOT=$I_MPI_ROOT  SLURM_NTASKS=$SLURM_NTASKS"

cd $WORK/SWEMniCS
ibrun python <your_script>.py
```

If the `pip show` version output ever disagrees with what you developed against on your laptop, stop and rebuild the venv — see WHY_IT_BROKE.md §6.

---

## 6. Test files in this repo

| File | Purpose |
|---|---|
| `test_mpi.py` | Every rank prints its identity; verifies launcher sees all ranks; prints MPI library identity. |
| `test_dolfinx.py` | 8×8 unit square Poisson solve — shortest non-trivial end-to-end dolfinx call. |
| `test_inlet_minimal.py` | Imports project `swe4dvar`, builds a tiny mesh, runs a PETSc vec — smoke test for the project stack as a whole. |
| `test_cpu.slurm` | `-p development -N 1 -n 4 -t 00:05:00` sbatch that runs all three above via `ibrun`. Costs ~1 node-hour fraction. |

---

## 7. Do NOT

- Do **not** `conda env create` here — conda-forge dolfinx ships with MPICH, which is ABI-incompatible with LS6's IntelMPI under `ibrun`. (See WHY_IT_BROKE.md §"The real constraint".)
- Do **not** skip `module reset` — leftover modules from a previous session cause silent MPI mismatches.
- Do **not** `module load impi` without a version — default is `impi/19.0.9`, which is incompatible with `dolfinx/0.10`.
- Do **not** use `mpirun` / `srun` to launch MPI — on LS6 only `ibrun` wires PMI correctly for IntelMPI.
- Do **not** run heavy work on login. Use `idev` or `sbatch`. 4-rank smoke tests of a few seconds are OK.
- Do **not** put the venv in `$HOME`. Use `$WORK`.
- Do **not** run `ssh-keygen` on LS6 (unrelated to FEniCSx, but still — it breaks batch auth).

# FAILURE_LOG.md — `fenics-env3` on LS6 (Phase 1: deliberate breakage)

**Date**: 2026-04-21
**Host**: `login1.ls6.tacc.utexas.edu`
**Goal**: provoke every failure mode of the naive "just pip install fenics-dolfinx" path and record raw output so the diagnosis is grounded in evidence.

---

## Naive Attempt #0 — `conda env create -f environment.yml`

### Command
```
conda env create -f environment.yml
```

### Raw error
```
-bash: conda: command not found
```

### Classification
**ENVIRONMENT MISSING** — LS6 has no TACC-supported conda/mamba. User must install miniforge/miniconda themselves (into `$WORK`, not `$HOME` — 10 GB + 20-file quota). This is *before* we even get to any ABI question.

---

## Naive Attempt #1 — default module stack, `python -c "import dolfinx"`

### Setup
Login with default module set (auto-loaded by TACC on every ssh):
```
Currently Loaded Modules:
  1) intel/19.1.1   3) autotools/1.4   5) cmake/4.1.1   7) xalt/3.1
  2) impi/19.0.9    4) python3/3.9.7   6) pmix/3.2.3    8) TACC
```

### Commands + raw output
```
$ which python3
/opt/apps/intel19/python3/3.9.7/bin/python3
$ python3 --version
Python 3.9.7
$ python3 -c "import dolfinx"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'dolfinx'
```

### Classification
**ImportError (expected)**. Confirms the TACC default `python3/3.9.7` stack has no FEniCSx Python bindings installed.

### Why naive "pip install --user fenics-dolfinx" into this stack would also fail
- `fenics-dolfinx` on PyPI is a **source distribution** (there are no cross-platform wheels). Installing it requires building against:
  - C++17 compiler (intel 19.1 is **C++17 but pre-concepts**; dolfinx 0.10 uses C++20 features)
  - PETSc >= 3.20 with python bindings (default stack has no PETSc loaded, and system `petsc/3.22` module is NOT compatible with `intel/19.1.1 + impi/19.0.9` — it requires `gcc/13.2.0` or `intel/24.1`)
  - `pybind11`, `basix`, `ffcx`, `ufl` Python packages
- Result: compile fails with C++20 feature use (`concept`, `consteval`, `std::ranges`) under intel19, OR cmake cannot find `dolfinx-config.cmake` without the system module exporting `CMAKE_PREFIX_PATH`.

This attempt was not run to completion — the failure mode is predictable and running a 10-minute doomed build wastes SUs for no new information.

---

## Naive Attempt #2 — `pip install -r requirements.txt`

### Setup
Default stack, no FEniCSx module.

### What the file contains
```
numpy>=2.0, scipy>=1.14, matplotlib>=3.9, seaborn, h5py, pandas, tqdm, loguru>=0.7.0,
checkpoint-schedules>=1.0.0
# Note: FEniCSx ecosystem (dolfinx, basix, ffcx, ufl) and parallel computing
# (mpi4py, petsc4py) should be installed via conda from conda-forge channel.
```

### Outcome
`pip install -r requirements.txt` would succeed (pure-Python / manylinux wheels), but it installs **no FEniCSx at all**. The user then gets the same `ModuleNotFoundError: No module named 'dolfinx'` as Attempt #1.

### Classification
**SILENT MISCONFIGURATION** — the requirements file is designed to be paired with `environment.yml` (conda) but on LS6, without conda, the user sees no error, just missing functionality when they run experiments.

---

## Naive Attempt #3 — dolfinx module alone (no Python module)

### Setup
```
module reset
module load gcc/13.2.0 impi/21.11 dolfinx/0.10.0.post5
```

### Raw output
```
Inactive Modules:
  1) python3               # python3/3.9.7 deactivated (wrong impi version)

$ python --version
python: error while loading shared libraries: libpython3.9.so.1.0:
    cannot open shared object file: No such file or directory
```

### Classification
**BROKEN SYSTEM PYTHON**. Loading `gcc/13.2.0` **deactivates `python3/3.9.7`** (since that Python was compiled against `intel/19.1.1 + impi/19.0.9`). `/usr/bin/python` on LS6 is a TACC wrapper that hard-depends on `libpython3.9.so.1.0`; without that library on `LD_LIBRARY_PATH`, even `python --version` fails. Any user following naive "just load the dolfinx module" advice hits this wall.

### Fix implied
Must also load a compatible Python: `module load python/3.12.11` (the only Python module available under the gcc13+impi21 family).

---

## Naive Attempt #4 — `module load dolfinx` + `python/3.12` + `import dolfinx`

### Setup
```
module reset
module load gcc/13.2.0 impi/21.11 python/3.12.11 dolfinx/0.10.0.post5
```

### Raw output
```
$ which python3
/opt/apps/python/3.12.11/bin/python3
$ python3 --version
Python 3.12.11
$ python3 -c "import dolfinx"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'dolfinx'
$ python3 -c "from mpi4py import MPI"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'mpi4py'
$ python3 -c "import petsc4py"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'petsc4py'
$ python3 -c "import basix"
ModuleNotFoundError: No module named 'basix'
```

### Filesystem proof
```
$ ls /scratch/tacc/apps/gcc13_2/impi21/dolfinx/0.10.0.post5/lib64/python*
ls: cannot access '.../lib64/python*': No such file or directory
$ find /scratch/tacc/apps/gcc13_2/impi21/dolfinx/0.10.0.post5 -name 'dolfinx' -type d
.../lib64/cmake/dolfinx      # cmake exports for C++ consumers
.../lib64/dolfinx            # C++ .so libraries
.../include/dolfinx          # C++ headers
# NO python package anywhere
```

### Classification
**MISSING PYTHON BINDINGS**. The TACC `dolfinx/0.10.0.post5` module ships the **C++ library** only — `libdolfinx.so`, CMake configs, headers. It does **not** ship the Python bindings. Same story for `petsc/3.22` and `impi/21.11` — they are C/C++ layers.

To use FEniCSx from Python on LS6, the user must install the Python bindings themselves — compiled against these system C++ libraries.

---

## Naive Attempt #5 — `conda env create` from `environment.yml` (hypothetical)

Since miniforge isn't present, this was not run, but the outcome is predictable and needs recording because it's the most natural thing a new LS6 user would try after installing miniforge:

### Expected outcome
The conda env creates successfully. `environment.yml` pins:
- `fenics-dolfinx>=0.9.0,<0.10`
- `mpich`
- `petsc4py>=3.22`
- `mpi4py>=4.0`

### Predicted failure mode
Installs fine. `python -c "import dolfinx"` works. Single-process runs work. **Multi-node MPI under `ibrun` breaks**, because:
- conda-forge dolfinx is linked to **MPICH 4.x**
- LS6's launcher `ibrun` is a wrapper around **IntelMPI 21** (or Slurm's `srun` with IntelMPI's PMI client)
- The resulting binary-level MPI ABI is incompatible. Expected symptoms:
  - `PMI_Init` mismatch errors
  - Silent degeneration to 1 rank per node (`MPI.COMM_WORLD.size == 1` everywhere)
  - Hangs at the first collective
- Single-node `mpiexec` from conda's bundled launcher works, but cannot cross nodes on LS6's InfiniBand without the vendor MPI.

### Classification
**MPI ABI MISMATCH** — the ABI-breaking case. This is the silent killer: install succeeds, single-process works, single-node local-mpiexec works, and then production multi-node runs fail or (worse) produce silently-wrong results.

---

## Naive Attempt #6 — `pip install petsc4py==3.22.*` with default build isolation

### Context
After loading the correct module stack, created a venv, ran `pip install petsc4py==3.22.*` with default pip behavior (build isolation enabled).

### Raw error
```
error: Cython failure: 'petsc4py/PETSc.pyx' -> 'petsc4py/PETSc.c'
Compiler crash in ExpressionWriter
AttributeError: 'ExpressionWriter' object has no attribute 'emit_string'
```

### Root cause
Pip's build isolation creates an ephemeral env and installs **Cython 3.2.4** (latest) for the build. petsc4py 3.22.5 has `.pyx` files using Python-3.12 type-annotation syntax that triggers a crash in Cython 3.2.4's `ExpressionWriter`.

### Fix
Install Cython ≥ 3.0, < 3.1 in the venv, use `--no-build-isolation`:
```
pip install "cython>=3.0,<3.1"
pip install --no-build-isolation petsc4py==3.22.*
```

### Classification
**BUILD-DEP VERSION SKEW**. Not visible until you actually try to compile. The failure is at cythonize-time, not at runtime.

---

## Naive Attempt #7 — petsc4py with `setuptools>=80`

### Context
After fixing Cython, the petsc4py build advanced further but died in its custom setup logic.

### Raw error
```
File "confpetsc.py", line 692, in build_configuration
    execute(...)
TypeError: execute() got an unexpected keyword argument 'dry_run'
```

### Root cause
petsc4py 3.22.5's `conf/confpetsc.py` calls `distutils.util.execute(func, args, msg, dry_run=False)`. In **setuptools ≥ 80**, `setuptools._distutils.util.execute` no longer accepts `dry_run=` (the upstream CPython distutils removal landed, and setuptools' vendored copy followed). Pre-80 setuptools retained the old kwarg.

### Fix
Pin `setuptools < 70` in the venv:
```
pip install --force-reinstall "setuptools<70"
pip install --no-build-isolation petsc4py==3.22.*
```

### Classification
**SILENT UPSTREAM API REMOVAL**. setuptools does not advertise this as a breaking change in its CHANGELOG in a way petsc4py maintainers caught. Will be fixed in petsc4py ≥ 3.23, but pinned to 3.22.* because PETSc minor version must match `$TACC_PETSC_DIR`.

---

## Naive Attempt #8 — correct module stack with `impi/21.11` (as spider suggests)

### Context
After fixing Cython + setuptools, petsc4py built and installed successfully. Moving on to import-test.

### Module stack loaded
```
gcc/13.2.0, impi/21.11, python/3.12.11, petsc/3.22, dolfinx/0.10.0.post5
```

`module spider dolfinx/0.10.0.post5` explicitly advertises this combination as valid:
```
You will need to load all module(s) on any one of the lines below before the "dolfinx/0.10.0.post5" module is available to load.
  gcc/13.2.0  impi/21.11
  gcc/13.2.0  impi/21.12
```

### Raw error
```
$ python -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"
Traceback (most recent call last):
  File "/work/.../petsc4py/PETSc.py", line 4, in <module>
    PETSc = ImportPETSc(ARCH)
ImportError: /scratch/tacc/apps/gcc13_2/impi21/petsc/3.22.4/3.22.4/lib/libpetsc.so.3.22:
    undefined symbol: MPI_Neighbor_alltoallv_init
```

### Diagnosis
`MPI_Neighbor_alltoallv_init` is part of the **MPI-4.0 persistent collectives** group (introduced 2021). Checked every `libmpi.so.12` on LS6:

| Library | Has symbol? |
|---|---|
| `/scratch/.../2021.11/lib/libmpi.so.12` | **No** |
| `/scratch/.../2021.11/lib/release/libmpi.so.12` | **No** |
| `/scratch/intel24.1/.../2021.12/lib/release/libmpi.so.12` | **Yes (weak symbol)** |

So the shared `petsc/3.22` library was **built against `impi/21.12`** but TACC's module system advertises both `21.11` and `21.12` as compatible with `dolfinx`. They aren't — `dolfinx/0.10.0.post5` itself may load under `21.11`, but its petsc dependency will then crash on first symbol resolution.

### Fix
Replace `impi/21.11` with `impi/21.12` in the module load command. mpi4py built under 21.11 **does not need to be rebuilt** — the SONAME is `libmpi.so.12` in both versions, so dynamic linking resolves to 21.12 cleanly.

### Classification
**UPSTREAM MODULE METADATA ERROR**. `module spider` is lying (or its contract differs from what users assume: "either works for loading" vs "either works for using"). This is the most insidious failure because:
- Everything compiles.
- Everything loads.
- Import of specific packages crashes on specific missing symbols, and only some users hit it depending on which code paths they exercise.

### Moral
Always verify with `nm -D` which MPI library actually has the symbols your application needs. `module spider` is advisory, not contractual.

---

## Summary

| # | Attempt | Immediate failure | Category |
|---|---|---|---|
| 0 | `conda env create` | `conda: command not found` | Tooling |
| 1 | `python -c "import dolfinx"` (default stack) | `ModuleNotFoundError` | Import |
| 2 | `pip install -r requirements.txt` | Silent — no dolfinx installed | Silent misconfig |
| 3 | `module load dolfinx` only | `libpython3.9.so.1.0: cannot open` | System-Python break |
| 4 | `module load dolfinx + python/3.12` | `ModuleNotFoundError` for dolfinx/mpi4py/petsc4py | Missing Python bindings |
| 5 | conda-forge stack (predicted) | MPI ABI mismatch at `ibrun` time | **Runtime / ABI** |
| 6 | `pip install petsc4py` (default build-isolation) | Cython 3.2.4 `ExpressionWriter` crash | Build-dep skew |
| 7 | petsc4py with setuptools 80+ | `TypeError: execute() got unexpected kwarg 'dry_run'` | Upstream API removal |
| 8 | Build with impi/21.11 (module spider says OK) | `undefined symbol: MPI_Neighbor_alltoallv_init` | **Module metadata error** |
| 9 | fenics-basix==0.10.0 (PyPI) vs dolfinx built on TACC basix/0.10.0.post0 | `TypeError: CppElement incompatible function arguments` | **Basix Py/C++ patch-version skew** |

---

## Naive Attempt #9 — PyPI basix 0.10.0 vs dolfinx built against TACC basix 0.10.0.post0

### Context
Full install succeeded. Imports worked serially. Then running `fem.functionspace(domain, ("Lagrange", 1))` on 4 ranks:

### Raw error
```
File "dolfinx/fem/function.py", line 619, in functionspace
    return FiniteElement(CppElement(basix_e, value_shape, ufl_e.is_symmetric))
TypeError: __init__(): incompatible function arguments. The following argument types are supported:
    1. __init__(self, element: basix::FiniteElement<double>, ...)
    2. __init__(self, elements: collections.abc.Sequence[dolfinx.cpp.fem.FiniteElement_float64])
    3. __init__(self, cell_type: dolfinx.cpp.mesh.CellType, ...)

Invoked with types: dolfinx.cpp.fem.FiniteElement_float64, basix._basixcpp.FiniteElement_float64, NoneType, bool
```

### Root cause
- `fenics-basix 0.10.0` (PyPI, latest available) bundles basix C++ library at version 0.10.0.
- `fenics-dolfinx 0.10.0.post5` (GitHub tag) was built against TACC's `basix/0.10.0.post0` C++ library.
- nanobind-generated Python bindings carry type IDs that depend on the exact C++ class layout. Basix 0.10.0 and 0.10.0.post0 have subtly different class signatures (`block_shape` default, `symmetric` flag), so the dolfinx binding rejects Python objects coming from the "wrong" basix.

### Fix
Reinstall `fenics-basix` from the matching tag:
```
pip install --force-reinstall --no-build-isolation \
    "git+https://github.com/FEniCS/basix.git@v0.10.0.post0#subdirectory=python"
```
This builds basix C++ at `0.10.0.post0` to match dolfinx, and installs Python bindings that talk to that library.

### Classification
**PATCH-VERSION ABI SKEW**. The user-facing version string is `0.10.0` in both; only the "post0" suffix distinguishes them. PyPI does not carry post-release tags. You can only get them from GitHub. No `pip install fenics-*==0.10.*` command will produce a consistent stack on LS6 without specifying git tags explicitly.

### Moral
For FEniCSx on TACC systems, **all four Python packages** (basix, ufl, ffcx, dolfinx) must be installed from the exact GitHub tags that match the TACC module versions. PyPI is a trap.

See `DEPENDENCY_GRAPH.md` for the causal chain and `WHY_IT_BROKE.md` for the root-cause narrative.

# DEPENDENCY_GRAPH.md

## The ABI chain that every FEniCSx import on LS6 traverses

```
user code
  └── dolfinx (Python)                          [bindings; pip or conda-forge]
        └── libdolfinx.so                       [C++ library]
              ├── petsc4py (Python)             [bindings]
              │     └── libpetsc.so             [C library]
              │           └── libmpi.so.12      [MPI ABI — the fault line]
              │                  └── PMI client layer (Slurm-side)
              ├── mpi4py (Python)
              │     └── libmpi.so.12            [same fault line]
              └── basix / ffcx / ufl (Python, pure)  — independent of ABI

Parallel launch:
  ibrun / sbatch
    └── SLURM PMI-2 plugin
          └── MPI library chosen at user's `module load` time
                must match libmpi.so.12 linked into petsc / dolfinx
```

**The fault line**: every binary in the red path above must be linked against **the same MPI library family and version**.

LS6 reality: login nodes have `impi/19.0.9` by default (Intel MPI Library 2019u9), and the `dolfinx/0.10.0.post5` module requires `impi/21.11` or `impi/21.12`. Conda-forge `fenics-dolfinx` ships with `mpich 4.x`. These three MPI ABIs are mutually incompatible at the symbol level (`MPI_Init`, `PMI_Init`, datatype handles).

---

## Causal chain for each failure in FAILURE_LOG.md

### Failure #1: `ModuleNotFoundError: dolfinx` on default stack
```
default `module load` on login
  └── python3/3.9.7 (intel19 + impi19)
        └── no dolfinx Python bindings in site-packages
              └── import fails
```
**Root**: no Python bindings installed anywhere TACC's site-packages knows about.

---

### Failure #2: `conda: command not found`
```
naive instruction "conda env create -f environment.yml"
  └── conda binary absent from PATH
        └── TACC does not install conda system-wide
              └── user has never installed miniforge in $WORK
```
**Root**: assumed tooling absent on HPC. TACC's official path is Python-via-module, not conda.

---

### Failure #3: `libpython3.9.so.1.0: cannot open shared object file`
```
`module load gcc/13.2.0`
  └── Lmod auto-detects conflict with intel/19.1.1, swaps → gcc13
        └── Lmod marks `python3/3.9.7` INACTIVE (built for intel19)
              └── $LD_LIBRARY_PATH no longer contains /opt/apps/intel19/python3/3.9.7/lib
                    └── /usr/bin/python is a TACC-wrapped alias that dlopens libpython3.9.so.1.0
                          └── dlopen fails
                                └── even `python --version` dies
```
**Root**: the system /usr/bin/python is tightly bound to a specific Python module. Changing compiler modules blows it away. User must also load `python/3.12.11` (the matched Python for gcc13 stack).

---

### Failure #4: dolfinx module loaded, Python bindings still missing
```
dolfinx/0.10.0.post5 module
  └── exports only TACC_DOLFINX_{INC,LIB,DIR}
        └── /scratch/tacc/apps/gcc13_2/impi21/dolfinx/0.10.0.post5/
              ├── lib64/cmake/dolfinx/       [C++ cmake]
              ├── lib64/libdolfinx.so        [C++ library]
              ├── include/dolfinx/           [C++ headers]
              └── (no python/ or lib/python3.12/site-packages/ at all)
```
**Root**: by TACC convention, modules ship C/C++ layers. Python bindings are the user's responsibility to install (pip into a venv) so that multiple users' Python stacks don't collide.

---

### Failure #5 (predicted, not run): conda-forge dolfinx + `ibrun`
```
user installs conda-forge fenics-dolfinx (pulls mpich 4.x)
  └── libdolfinx.so → libpetsc.so → libmpi.so.12 (mpich build)
  └── Python launches fine, single-process
  └── user wraps with `ibrun -n 128 python run.py` in an sbatch script
        └── ibrun injects IntelMPI PMI env (I_MPI_PMI=pmi2, PMI_SIZE, …)
              └── mpich's PMI client doesn't understand IntelMPI PMI keys
                    └── OPTION A: collective hang at MPI_Init
                    └── OPTION B: MPI_Init succeeds trivially with COMM_WORLD.size=1
                                   per rank, and the 128 ranks never discover each other
                                   → silently wrong science (every rank does full run)
                    └── OPTION C: libmpi.so symbol resolution conflicts
                                   → `undefined symbol: MPIR_CVAR_*` or similar
```
**Root**: **MPI is not ABI-compatible across vendors**. MPICH, OpenMPI, IntelMPI all use different symbol layouts, PMI protocols, and datatype handles. A binary linked against one cannot run under another's launcher.

This is why the pragmatic choice on LS6 is **Strategy A (align to TACC's IntelMPI + modules)**, not Strategy B (conda-forge).

---

## The fix (Strategy A)

```
module reset
module load gcc/13.2.0 impi/21.11 python/3.12.11 petsc/3.22 dolfinx/0.10.0.post5
                ↓              ↓            ↓                ↓                ↓
             [compiler]   [system MPI]  [compatible Py]  [C++ PETSc]  [C++ dolfinx]

python -m venv $WORK/venvs/fenics-ls6 --system-site-packages
source $WORK/venvs/fenics-ls6/bin/activate

pip install --no-binary=mpi4py  mpi4py==4.*          # compiles against $I_MPI_ROOT
pip install --no-binary=petsc4py "petsc4py==3.22.*"  # compiles against $TACC_PETSC_DIR
pip install fenics-basix==0.10.* fenics-ufl fenics-ffcx==0.10.*   # pure python
pip install --no-build-isolation "fenics-dolfinx==0.10.*"          # builds against $TACC_DOLFINX_DIR
```

All four C-extension layers (mpi4py, petsc4py, dolfinx-bindings) are now **linked into the same IntelMPI 21.11** and **the same system PETSc 3.22**. `ibrun` can launch the result across nodes.

---

## Fragility diagram — what breaks this chain

```
[compiler] gcc/13.2.0 ──┐
[system MPI] impi/21.11 ┼──► TACC upgrades any of these → ABI break, rebuild venv
[C++ PETSc] petsc/3.22 ─┤
[C++ dolfinx] 0.10.0.post5┘

[Py]    python/3.12.11 ──► if Python module deprecated, venv invalid
[user]  pip install       ──► network flake during C++ build → partial venv, silent breakage
[launch] ibrun/sbatch     ──► wrong `module load` at job time → libmpi.so mismatch → OPTION B silent failure
```

**Mitigation**: pin exact versions in WORKING_SETUP.md; always `module list` inside sbatch scripts to log what was loaded; never leave `module load` calls out of the sbatch preamble.

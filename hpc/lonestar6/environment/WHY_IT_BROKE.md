# WHY_IT_BROKE.md

## One-line answer

`fenics-env3` was a conda-forge env built around **MPICH** on a laptop. LS6's MPI is **IntelMPI 21**. These two MPI libraries are not ABI-compatible, so any attempt to run a conda-forge `fenics-dolfinx` binary under LS6's `ibrun` / `srun` launcher will break — either loudly (symbol errors, PMI mismatch) or silently (all ranks think they're rank 0, science is wrong and you don't notice until your paper gets rejected).

That is the **real constraint**. Everything else is downstream.

---

## Why the naive attempts fail (condensed)

| Naive attempt | Why it fails |
|---|---|
| `conda env create -f environment.yml` | `conda` is not on LS6 — HPC sites don't ship it system-wide. |
| `pip install -r requirements.txt` | `requirements.txt` intentionally excludes FEniCSx (comment says "use environment.yml"). Silently installs no dolfinx. |
| `module load dolfinx/0.10.0.post5` alone | Loading `gcc/13.2.0` kicks out `python3/3.9.7`. System `/usr/bin/python` can't find `libpython3.9.so.1.0`. Even `python --version` dies. |
| `module load dolfinx + python/3.12` | TACC's dolfinx module is **C++ only** — no Python bindings. `import dolfinx` still fails. |
| Install conda-forge fenics-dolfinx | Installs fine; fails at `ibrun` time with MPI ABI mismatch. |

The *real* failure is #5. Everything before it is discoverable in seconds. #5 is the one that costs you an SU allocation before you notice.

---

## The real constraint: MPI ABI

When `dolfinx` / `petsc` / `mpi4py` are compiled, the compiler stamps the symbol table of `libmpi.so` it's linked against into the `.so` file. At load time, `ld.so` looks for those exact symbols. There are **three major incompatible MPI stacks** in play on LS6:

| Stack | Source | Where it appears |
|---|---|---|
| IntelMPI 2019u9 (`impi/19.0.9`) | TACC default | All default modules: `intel/19.1.1`, `python3/3.9.7`, all `petsc/3.15`-`3.19` built for intel19 |
| IntelMPI 2021.11 (`impi/21.11`) | TACC gcc13 stack | `gcc/13.2.0 + impi/21.11 + python/3.12.11 + petsc/3.22 + dolfinx/0.10.0.post5` — the new supported FEniCSx stack |
| MPICH 4.x (conda-forge) | User-installed conda | Anything installed from conda-forge with `mpich` in its build string |

A binary built against one **cannot** run inside a launcher that speaks another's PMI protocol. `ibrun` on LS6 only knows IntelMPI's PMI. Therefore conda-forge dolfinx is a dead end for multi-node work on LS6.

(Single-node with conda's own `mpiexec` does work. But you came to LS6 to scale, not to run on 128 cores of a single node.)

---

## Why the laptop env doesn't port

`fenics-env3` on the laptop is built around conda-forge's `mpich` because conda-forge doesn't ship IntelMPI binaries (Intel's license prohibits redistribution). Laptops don't have InfiniBand or a SLURM PMI plugin, so MPICH's bundled `mpiexec` is fine — you never notice the MPI choice matters.

The moment you cross from laptop to HPC, the MPI stack becomes a **contract between the application's linkage and the system's launcher**. Laptop: no contract, anything works. HPC: strict contract, enforced at `ibrun` time.

---

## What the future-you will hit again

1. **TACC upgrades `dolfinx` to 0.11 or drops 0.10** — your venv is now against a stale `libdolfinx.so`. Symptom: import-time `undefined symbol: _ZN7dolfinx...`. Fix: rebuild the venv against the new module. Mitigation: `module list` inside every sbatch script, log versions in every results directory.

2. **You update `impi/21.11` → `impi/21.12` in one terminal but not in the sbatch script** — `ibrun` launches with 21.12, petsc4py linked against 21.11 → `PMI_Init` mismatch or silent one-rank-per-node. Mitigation: always pin the exact `impi` version in sbatch preamble; never use `module load impi` without a version.

3. **You activate the venv before loading modules** — `PATH` and `LD_LIBRARY_PATH` get out of order; `python` may pick a stray `libpetsc.so`. Mitigation: load modules first, activate venv second, always.

4. **Someone edits `~/.bashrc` to auto-load modules** — hidden state. A teammate's sbatch script breaks at 3 am when the bashrc change is deployed. Mitigation: keep `~/.bashrc` minimal; do all env setup in sbatch preamble.

5. **You edit the code on laptop in a venv that has dolfinx 0.9; LS6 has 0.10** — API differences (e.g. `dolfinx.mesh.create_rectangle` signature, assembler namespaces) cause silent behavior divergence between "works locally" and "works on LS6". Mitigation: a local venv matching the LS6 version, OR a unit-test suite that pins API usage.

6. **Python 3.12 → 3.13 jump** — when TACC retires `python/3.12.11`, your venv is bound to a Python that's gone. Symptom: `venv/bin/python` errors on startup. Fix: rebuild the venv under the new Python module.

---

## What "done right" looks like

- The stack is: `module reset; module load gcc/13.2.0 impi/21.11 python/3.12.11 petsc/3.22 dolfinx/0.10.0.post5; source $WORK/venvs/fenics-ls6/bin/activate`. Those five lines are at the top of every sbatch script and documented in `WORKING_SETUP.md`.
- The venv contains only `mpi4py`, `petsc4py`, `fenics-basix`, `fenics-ufl`, `fenics-ffcx`, `fenics-dolfinx`, and the project's pip dependencies. No "accidental" MPI, PETSc, or dolfinx from other sources.
- Every sbatch script starts with `module list` + `pip show fenics-dolfinx | head` and pipes that into `$SLURM_JOB_ID.env.txt` so every run is provenance-traceable.
- The laptop env used for development pins the same major versions (`fenics-dolfinx==0.10.*`, `petsc4py==3.22.*`) so that API divergence between dev and HPC is impossible.

The system is predictable because the moving parts are enumerated.

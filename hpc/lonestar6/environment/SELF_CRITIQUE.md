# SELF_CRITIQUE.md — Phase 7

Strict honesty about the current working configuration: what's load-bearing, what's fragile, what we haven't proven, and what we'll get bitten by.

---

## What WILL fail on multi-node runs (predicted, not yet tested)

1. **PMI protocol drift between `ibrun` wrapper and `libmpi`.** The venv was built linked to `impi/21.11`. If your sbatch script loads `impi/21.12` (newer minor version, possibly loaded as default in 3 months when TACC rotates), the wire-protocol handshake may still work but: (a) non-blocking collectives can hang; (b) environment variable expectations differ; (c) some datatype handles change size. Reproduces as "works on 1 node, hangs at `MPI.COMM_WORLD.Barrier()` on 2 nodes".
   *Mitigation*: pin `#SBATCH` module loads to the EXACT `impi/21.11` used at venv build time.

2. **Inter-node `LD_LIBRARY_PATH` races.** Compute nodes on LS6 share the same NFS home and Lustre `$WORK`, but each node boots its own view of environment. If the sbatch preamble accidentally forks before `module load` completes (e.g. `&` at end of `module load` line), some nodes run with wrong library path. Reproduces as segfault on exactly one rank.
   *Mitigation*: never background `module load`; always `module list` immediately after to fail fast in the log.

3. **Fabric-level stalls under high rank counts.** LS6 HDR InfiniBand has a 24/16 oversubscribed topology. Tests so far are at N=4 ranks on one node. At N=1000+ ranks across 8 nodes, `MPI_Allreduce`-heavy workloads (which most PETSc iterative solvers are) will hit the oversubscription and stall. This is not a venv problem, it's a physics problem — but the first time you notice will be when a scaling curve flattens.
   *Mitigation*: benchmark `osu_allreduce` inside the venv at 2/4/8/16 node counts; report scaling curve in any paper.

4. **mpi4py process startup cost at N ~= 128 ranks/node.** mpi4py + dolfinx imports ~50 shared libraries per rank. Lustre's metadata server sees a thundering herd every job launch. Under contention, python startup alone can take 30–90 seconds per 128-rank node. Not a failure, just a surprise that eats wall-clock.
   *Mitigation*: pre-build Python bytecode caches (`python -m compileall $VENV`); for very large runs, consider building a Singularity/Apptainer image so the Python-module discovery is inside an overlay FS, not Lustre.

---

## What breaks if TACC changes modules

| Change | What breaks | Recovery |
|---|---|---|
| `dolfinx/0.10.0.post5` → `0.11.0` | `import dolfinx` fails (`undefined symbol: _ZN7dolfinx...`) | Rebuild venv (~15 min) |
| `petsc/3.22` → `petsc/3.23` | `from petsc4py import PETSc` fails (`undefined symbol: PetscInitialize`) | Rebuild venv or rebuild petsc4py against new dir |
| `impi/21.11` deprecated | `ibrun` launches but `mpi4py` crashes at `MPI_Init` | Rebuild mpi4py; must also rebuild anything that links `libmpi.so.12` |
| `python/3.12.11` → `python/3.13.x` | `$VENV/bin/python` segfaults on startup | Create a new venv under the new Python |
| `gcc/13.2.0` retired | `libdolfinx.so` linking against newer libstdc++ will ABI-break | Wait for new `dolfinx` module to land, then rebuild |
| TACC adds `fenics-dolfinx` as a module with bundled Python bindings | **Everything on this page is obsolete** | Delete the venv; use the module |

**Signal for each**: every sbatch script logs `module list` + `pip show fenics-dolfinx petsc4py mpi4py | head`. When those change silently, a diff against an older job's log tells you what moved.

---

## Assumptions that are currently fragile

1. **Only one compatible Python module exists** (`python/3.12.11`). If TACC adds a second, Lmod may pick the wrong one silently.

2. **The venv was built once, on login node**. It's never been exercised on a compute node. We assume `/work` is identically visible from compute nodes as from login (verified by TACC's docs, unverified by test).

3. **Cython 3.0.x works; 3.1+ might**. We pinned 3.0 because petsc4py 3.22.x documented crashes on Cython 3.2.4. When petsc4py 3.22.5+ ships with Cython 3.2 fixes, we can loosen this. For now, any `pip install` that pulls a newer Cython will re-break builds — this is why the venv lives with a constraint file, not an ad-hoc shell history.

4. **`--system-site-packages`** means pip packages at `/opt/apps/python/3.12.11/lib/python3.12/site-packages` shadow ours. If TACC upgrades a system `numpy` under us, we inherit it — may break petsc4py which was built against numpy ABI at install time. Safer would be a pure isolated venv (`python -m venv` without the flag) and pip-install numpy explicitly. **This is worth considering.**

5. **FEniCSx 0.9 (laptop) ≠ 0.10 (LS6)**. API drift between these two is not zero — `dolfinx.fem.assemble_matrix` signature and `dolfinx.nls.petsc.NewtonSolver` module path changed. Any code written against 0.9 and pushed unchanged to LS6 may fail silently (wrong namespace) or loudly (AttributeError). Dev-prod parity requires upgrading the laptop env to 0.10 OR pinning LS6 to 0.9 (requires building dolfinx from source — NOT recommended).

6. **Allocation (`ADCIRC`, 75 k SUs) is for a year**. If expired without renewal, `sbatch --test-only` fails at the allocation check and no runs are possible.

7. **No off-node persistent cache**. Every venv lives at `$WORK/venvs/fenics-ls6`. If `$WORK` hits quota (1 TB) or has a metadata incident, the venv is gone. The module stack can be re-bootstrapped in 15 min — but we should keep `WORKING_SETUP.md` version-controlled in-repo (this file is) so a fresh rebuild is reproducible.

---

## What we have NOT yet proven

- [ ] `ibrun -n <N> python test_dolfinx.py` on a **compute node** (will do in `test_cpu.slurm`).
- [ ] Multi-node run (2 nodes, 256 ranks) — pending.
- [ ] The actual `SWEMniCS` pipeline end-to-end on LS6 with real observation data. `test_inlet_minimal.py` is a smoke test, not a science test.
- [ ] Correctness parity between a laptop (0.9) run and an LS6 (0.10) run on the same input.
- [ ] Performance: we have **no** baseline wallclock for any SWEMniCS configuration on LS6 yet.

---

## Confidence rating

| Claim | Confidence |
|---|---|
| "imports work on login" | High — verified with `python -c ...` |
| "mpi4py links to Intel MPI 2021.11" | High — `MPI.Get_library_version()` confirmed |
| "petsc4py links to system petsc/3.22" | Pending — wait for STAGE C to finish |
| "`ibrun -n 4 python` runs 4 ranks that see each other" | Medium — MPI library is right, but not yet tested |
| "multi-node run scales correctly" | Unknown — untested |
| "conda-forge stack would work" | Low confidence (predicts failure, not verified) |
| "will survive a `impi` minor version bump" | Low — ABI-level dependence |

---

## The criterion we should be able to meet

*"If someone hands me a fresh LS6 account tomorrow and `WORKING_SETUP.md`, can they be running `test_cpu.slurm` in 30 minutes?"*

Current answer: **probably yes**, provided the Cython-pinning trick continues to work and the module names don't shift. The two risk points are (a) `petsc4py` source being incompatible with whatever Cython is available at the time, and (b) TACC rotating modules under us without notice.

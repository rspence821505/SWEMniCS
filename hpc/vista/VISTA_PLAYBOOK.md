# Vista (TACC) Operational Playbook

**System**: TACC Vista — NVIDIA Grace ARM CPU + Hopper H100 GPU superpod. 144-core Neoverse-V2 nodes. Paired with LS6 as a higher-throughput dev environment. SSH: `ssh vista`.

**Queue map** (vs LS6):

| Role | LS6 | Vista |
|---|---|---|
| Development | `development` (2h walltime, 8 nodes) | `gh-dev` (GH200 GPU+CPU; 20 nodes; wait typically 0 min) |
| Main production | `normal` | `gh` (576 GH200 nodes); `gg` (251 Grace-Grace CPU-only nodes) |

**Paths:**
- `$HOME = /home1/08398/tg876971` (10 GB, 500K files)
- `$WORK = /work/08398/tg876971/vista` (1 TB quota, code + env)
- `$SCRATCH = /scratch/08398/tg876971`

**Account**: `#SBATCH -A ADCIRC` (same as LS6, 24890 SUs)

---

## 1. Environment setup (one-time)

Vista has NO system DOLFINx. We use a conda-forge environment.

```bash
# From a login node on Vista
cd $WORK
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O miniforge.sh
bash miniforge.sh -b -p $WORK/miniforge3
rm miniforge.sh

source $WORK/miniforge3/etc/profile.d/conda.sh
# conda (not mamba) — mamba's parallel extraction hits Lustre race conditions
conda create -n fenics-vista -c conda-forge -y \
  python=3.12 \
  "fenics-dolfinx=0.10" \
  mpi4py petsc4py \
  numpy scipy matplotlib pytest ipython
```

Full install takes ~5 min for download + ~15 min extraction on Lustre. Result: 2.8 GB env at `$WORK/miniforge3/envs/fenics-vista`.

## 2. Per-session activation

Source [hpc/vista/env.vista.sh](env.vista.sh):

```bash
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
```

This activates the conda env and prints a version summary. Use this in every sbatch.

## 3. Project install

```bash
source $WORK/SWEMniCS/hpc/vista/env.vista.sh
cd $WORK/SWEMniCS
pip install -e .
```

Editable install, so future `git pull` picks up code changes without reinstall.

## 4. Current stack versions

| Component | Version | Source | Notes |
|---|---|---|---|
| Python | 3.12.13 | conda-forge | |
| OpenMPI | 5.0.10 | conda-forge | Bundled; NOT Vista's system openmpi/5.0.5 |
| PETSc | 3.25.0 | conda-forge | Newer than LS6's 3.22 |
| mpi4py | 4.1.1 | conda-forge | |
| petsc4py | 3.25.0 | conda-forge | |
| DOLFINx | 0.10.0 | conda-forge | Same minor as LS6's 0.10.0.post5 |
| basix | 0.10.0 | conda-forge | |
| ffcx | 0.10.0 | conda-forge | |
| NumPy | 2.4.3 | conda-forge | |

## 5. Single-node vs multi-node

**Current setup: single-node only.** The conda-bundled OpenMPI does not know about Vista's interconnect. Running across nodes with `ibrun` would fall back to Ethernet and be slow.

For multi-node eventually:

```bash
# Uninstall conda-bundled MPI packages
conda remove --force mpich openmpi mpi4py petsc4py
# Reinstall against system MPI
module load openmpi/5.0.5
env MPICC=mpicc pip install --no-binary :all: mpi4py petsc4py
```

Then use Vista's `ibrun` as the launcher instead of `mpirun`.

We don't need this yet — our 8-rank MPI runs are sub-node (8 < 144).

## 6. Submission cheatsheet

| Queue | Partition | Walltime | Typical wait |
|---|---|---|---|
| Dev (quick test) | `gh-dev` | up to 2h | ~0 min (often idle) |
| Prod GPU | `gh` | up to 2 days | 1–6h short; 12–24h+ long |
| Prod CPU | `gg` | up to 2 days | 1–4h short; 10–24h long |

Template sbatch files live in [hpc/vista/slurm_templates/](slurm_templates/). Running a tiny sanity:

```bash
sbatch --partition=gh-dev --time=00:15:00 hpc/vista/slurm_templates/dev_sanity.slurm
```

## 7. Running existing idealized-inlet experiments

Port-equivalent sbatch files live at `hpc/vista/idealized_inlet/`. For parity vs LS6, submit the same flags (they're CLI-compatible) — just the SLURM header and env source line differ:

- `#SBATCH -p gh-dev` (vs `-p development`)
- `source $WORK/SWEMniCS/hpc/vista/env.vista.sh` (vs LS6's module loads)
- Launcher: `mpirun` from conda OpenMPI (vs LS6's `ibrun` for Intel MPI)

## 8. Troubleshooting

- **Lustre extraction failures during `conda install`** → use `conda` (not `mamba`). Parallel extraction hits Lustre race conditions.
- **`ModuleNotFoundError: dolfinx`** → did you `source hpc/vista/env.vista.sh` before `pip install -e .`? Without the env active, pip installs to system Python 3.9.
- **`MPI_Abort` on login node** → don't run `mpirun -n 2` on the login node. Submit a 15-min sbatch to `gh-dev`.
- **Submit limit** — Vista QOS allows more concurrent jobs than LS6 (no `QOSMaxJobsPerUserLimit=1` constraint observed; can run several in parallel).

## 9. Port provenance

Environment set up 2026-04-23 (commit TBD) to offload experiments from LS6's congested dev queue. Initial smoke-test passed at np=1 with tiny unit-square mesh (25 DOFs). Full parity vs LS6 numbers not yet verified — see `hpc/vista/parity/` (to be added).

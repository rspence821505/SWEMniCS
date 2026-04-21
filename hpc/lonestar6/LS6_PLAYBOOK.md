# Lonestar6 Operational Playbook

**Generated**: 2026-04-21 | **User**: `tg876971` | **Allocation**: `ADCIRC` (75,000 SUs, expires 2027-03-31)
**Sources**: TACC Lonestar6 User Guide + Good Conduct Policies + live on-system inspection

Unless marked **UNVERIFIED**, every number in this document was confirmed by a direct command on `login1.ls6.tacc.utexas.edu`.

---

## 1. System Overview

| Component | Detail |
|---|---|
| CPU nodes | 560× 2-socket AMD EPYC 7763 Milan — **128 cores**, **256 GB RAM** (`RealMemory=257113` MB), 288 GB local SSD `/tmp`, 2.45 GHz (boost 3.5) |
| GPU A100 nodes | 84× 3× NVIDIA A100 PCIe 40 GB on Milan CPU/256 GB RAM hosts |
| GPU H100 nodes | 4× 2× NVIDIA H100 PCIe 80 GB on AMD EPYC 9454 Genoa — **96 cores**, 256 GB RAM |
| Dev nodes (CPU) | 18 reserved for `development` partition |
| Interconnect | Mellanox HDR InfiniBand, 200 Gb/s, fat-tree with 24/16 oversubscription |
| Login nodes | `login1`, `login2`, `login3` (round-robin via `ls6.tacc.utexas.edu`) |
| Slurm | 23.11.11 |
| Scheduler launcher | `ibrun` for MPI (NOT `mpirun`, NOT `srun` for MPI startup) |

---

## 2. Storage (Verified paths)

| Mount | Path | Quota | Purge | Use |
|---|---|---|---|---|
| `$HOME` | `/home1/08398/tg876971` | **10 GB, 20 files** | never | Scripts, env files. **Tight file quota — do not store repos here.** |
| `$WORK` / `$WORK2` | `/work/08398/tg876971/ls6` | 1 TB, 3M files | never | Code, compiled software, shared project data. **Not backed up.** |
| `$STOCKYARD` | `/work/08398/tg876971` | (same as $WORK) | never | Cross-system TACC work root |
| `$SCRATCH` | `/scratch/08398/tg876971` | no quota | **10-day atime purge** | Job I/O, temp runs. **Run from here, not $WORK.** |
| `$SCRATCH_S2` | `/scratch_S2/08398/tg876971` | — | — | Secondary scratch tier (UNVERIFIED purge) |
| Local `/tmp` | `/tmp` (per-node) | 288 GB | job-end | Per-node temp; fastest I/O |
| Archive | `ranch.tacc.utexas.edu` (`$ARCHIVER`) | — | — | Cold archive for finished runs |

### Practical rules
- **Never run jobs from `$HOME`** — the 20-file limit will block you in hours.
- **Always run from `$SCRATCH`** (TACC policy). Output directly there, `cp` critical results to `$WORK` when the run finishes, then `scp` or `globus` to laptop / `ranch` for long-term archive.
- Don't generate >10k files per directory — use HDF5/NetCDF or subdirs.
- Lustre `$WORK` default stripe is 1 stripe / 1 MB. For large files (>10 GB), stripe before writing: `lfs setstripe -c 8 <dir>`. BeeGFS `$SCRATCH` default is 4 targets / 512 KB chunk.

---

## 3. Slurm Partitions (VERIFIED via `scontrol show partition` + `sacctmgr show qos`)

| Partition | QoS | MaxNodes | MaxWall | Notes |
|---|---|---|---|---|
| `development` (default) | qdevelopment | **8** | **2 h** | CPU dev/debug; 18 nodes |
| `normal` | qnormal | **64** | **2 d** | Primary CPU production; 513 nodes |
| `large` | qlarge | **256** | **2 d** | Jobs needing >64 nodes; same hw as normal |
| `gpu-a100-dev` | qa100development | **2** | **2 h** | A100 dev; 4 nodes |
| `gpu-a100` | qa100 | **8** | **2 d** | A100 production; 73 nodes |
| `gpu-a100-small` | qa100small | **1** | **2 d** | Single-A100 shared VMs (24 nodes) |
| `gpu-h100` | qh100 | **1** | **2 d** | H100 production; 4 nodes |
| `vm-small` | qsmall | **1** | **2 d** | Shared CPU VMs (28 nodes) |

### SU charging
- Charging is **per-node-hour** — there is no partial-node discount on exclusive CPU partitions (`development`, `normal`, `large`, `gpu-a100`, `gpu-a100-dev`, `gpu-h100`). Requesting 1 task on 1 node charges the full node.
- **UNVERIFIED exact rates**: The TACC docs snapshot did not give a rate table. Historical LS6 rates have been approximately 1 SU / CPU-node-hour for `normal`; GPU rates are higher. Confirm via `/usr/local/etc/taccinfo` decrement after your first real run.
- `vm-small` and `gpu-a100-small` are VM-based → shared, likely charged by cores/GPU rather than whole node (UNVERIFIED — test with a small job).

### Default account
- Your only project is `ADCIRC`. Always include `#SBATCH -A ADCIRC`. Set `echo 'ACCOUNT=ADCIRC' > ~/.idevrc` to have `idev` default to it.

---

## 4. SSH + Workflow (Verified)

### From laptop
```bash
ssh ls6            # aliased in ~/.ssh/config with ControlMaster
# or explicitly:
ssh tg876971@ls6.tacc.utexas.edu
```
- First connection: TACC password + 6-digit MFA token.
- Additional terminals / `scp` / `rsync` for 4 h reuse the master with **no prompt**.
- `ssh -O check ls6` inspects master. `ssh -O exit ls6` closes it early.
- **DO NOT run `ssh-keygen` on ls6** — it breaks batch job launchers.

### Canonical workflow
1. **Edit on laptop**, sync with `rsync -avz --exclude='.git' ./ ls6:\$WORK/SWEMniCS/`.
2. **Submit from ls6 login**: `cd $SCRATCH/run_name && sbatch job.slurm`.
3. **Monitor**: `squeue -u $USER` (not in a tight loop!), `tail -f slurm-<jobid>.out`.
4. **Retrieve**: `rsync -avz ls6:\$SCRATCH/run_name/results/ ./results/`.
5. **Archive**: move finished results to `$WORK` or push to `ranch.tacc.utexas.edu` before the 10-day purge.

---

## 5. Job Execution: `sbatch` vs `srun` vs `idev`

| Tool | When to use |
|---|---|
| `sbatch job.slurm` | Production — non-interactive, queued, the only sensible way for >5 min jobs. |
| `srun` inside an sbatch script | Task launcher *for non-MPI* steps. **Do not use it to launch MPI binaries** on LS6 — use `ibrun`. |
| `ibrun ./mpi_binary` | TACC MPI launcher. Handles PMIx + InfiniBand rail binding correctly. Use inside sbatch scripts. |
| `idev -m 60 -N 1 -p development -A ADCIRC` | Interactive compute node for debugging. **Charges SUs** (1 node-hour min, 30 min default). Great for reproducing crashes, testing Python/FEniCSx imports. |
| `sbatch --test-only job.slurm` | Dry-run validation: checks queue, account, quotas, reports estimated start time. **Does not submit, no SUs charged.** Use this before every real sbatch. |

### MPI vs OpenMP vs hybrid
- Pure MPI (most of ADCIRC/FEniCSx): `#SBATCH -N <nodes> -n <total_tasks>`, then `ibrun ./bin`.
- Pure OpenMP: `#SBATCH -N 1 -n 1 -c <threads>` (or `-c 128`), then `OMP_NUM_THREADS=128 ./bin`.
- Hybrid: `#SBATCH -N <nodes> -n <ranks_total> -c <threads_per_rank>`, then `OMP_NUM_THREADS=<threads> ibrun ./bin`. With 128 cores / node and 2 NUMA domains, common patterns: 2 ranks × 64 threads, 8 ranks × 16 threads, 16 ranks × 8 threads.

---

## 6. Modules (Verified available)

Default loaded set:
```
intel/19.1.1  impi/19.0.9  python3/3.9.7  cmake/4.1.1  pmix/3.2.3  autotools/1.4  xalt/3.1  TACC
```

Relevant to thesis work (PETSc, dolfinx):
- `module avail petsc` → ~100 PETSc variants (3.15 … 3.22, with flavors `-complex`, `-i64`, `-single`, `-cuda`, `-debug`). For FEniCSx you typically want `petsc/3.19` or `petsc/3.22` with real/double (no suffix) OR an `-i64` build if you index large meshes.
- `hdf5/1.14.0`, `netcdf/4.9.2`, `parallel-netcdf/4.6.2` — parallel I/O.
- `adcirc/55.01` — pre-built ADCIRC.
- `hypre`, `mumps`, `parmetis_petsc`, `dealii`, `mfem`, `p4est` — standard FE/solver stack.
- `adios2/2.9.1` — parallel I/O framework.
- **FEniCSx / dolfinx is NOT in the default module tree** — you will need to `pip install fenics-dolfinx` or build it. For DG shallow-water work, confirm with `module spider dolfinx` before assuming absence.

### Saving a personal stack
```bash
module load petsc/3.19
module save swe            # stored at ~/.lmod.d/swe
# later:
module restore swe
```

---

## 7. Production Slurm Templates

See [slurm_templates/](slurm_templates/):
- `cpu_job.slurm` — MPI CPU production on `normal`
- `gpu_job.slurm` — A100 GPU job on `gpu-a100`
- `debug_job.slurm` — short `development` job for sanity checks

Common header fields explained:
```bash
#SBATCH -J <name>               # job name (appears in squeue)
#SBATCH -A ADCIRC               # only project
#SBATCH -p <partition>          # see §3 table
#SBATCH -N <nodes>              # nodes (counts toward MaxNodes QoS)
#SBATCH -n <total_mpi_tasks>    # across all nodes; ibrun reads this
#SBATCH -t HH:MM:SS             # wall clock; billed even if idle
#SBATCH -o %x.%j.out            # stdout → jobname.jobid.out
#SBATCH -e %x.%j.err            # stderr
#SBATCH --mail-type=END,FAIL    # optional email notifications
#SBATCH --mail-user=rylan.spence@utexas.edu
```

---

## 8. Anti-Patterns (Explicit, Strict)

**Running work on login nodes is grounds for account suspension.** TACC monitors — don't.

| Don't | Why | Do instead |
|---|---|---|
| `python run_big_script.py` on login | Uses login CPU, affects other users | `idev` → run there; or `sbatch` |
| `make -j 16` on login | Same | `make -j 4` max, or build in an idev session |
| Run VSCode remote-SSH that starts Python servers on login | Eats login memory; explicitly warned in TACC policy | Run VSCode on laptop, sync via rsync |
| `while true; do squeue -u $USER; sleep 1; done` | Schedulers hate polling loops | `squeue -u $USER -i 60` (watches every 60 s) or one-shot checks |
| Batch submission loops (`for i in ...; do sbatch; done` without delay) | Hammers scheduler | Use job arrays: `#SBATCH --array=1-100` |
| Running from `$HOME` | 20-file quota will break things silently | Run from `$SCRATCH` |
| Running from `$WORK` | Lustre is meant for storage, not high-IOPS job I/O | Run from `$SCRATCH`; stage final outputs to `$WORK` |
| Request more nodes "to be safe" | Charges you for full nodes idle | Profile first, right-size |
| Tens of thousands of tiny files in one dir | Kills Lustre/BeeGFS metadata | HDF5/NetCDF/tar archives |
| Use `mpirun` / `mpiexec` / `srun ./mpi_bin` | LS6 wants PMIx via `ibrun` | `ibrun ./mpi_bin` |
| `ssh-keygen` on ls6 | Breaks batch auth to compute nodes | Never run it. If you did: `rm -rf ~/.ssh && logout && login` |
| Multi-GB data staging without striping | Slow writes, metadata pressure | `lfs setstripe -c 8 dir` before bulk writes |

---

## 9. Useful Commands (Verified)

```bash
# allocation + quotas
/usr/local/etc/taccinfo

# queue state
sinfo                                  # all partitions
sinfo -p normal -o "%10P %5a %12l %5D %6t"
scontrol show partition normal
scontrol show node c301-009
squeue -u $USER
squeue -p gpu-a100 -t PD | wc -l       # # of pending jobs in a queue

# submitting & checking
sbatch --test-only job.slurm           # dry-run (FREE)
sbatch job.slurm
scancel <jobid>
scontrol show job <jobid>
sstat -j <jobid> --format=JobID,MaxRSS,AveCPU   # running job stats
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqTRES  # historical

# filesystem
lfs quota -h $WORK                     # $WORK Lustre quota
lfs getstripe <dir>                    # current stripe
lfs setstripe -c 8 <dir>               # restripe for large files
/usr/local/etc/taccinfo                # quota summary

# archive / transfer
rsync -avz ls6:$SCRATCH/run/ ./run/    # from laptop
scp -r ls6:$WORK/data/ ./              # single-shot
# use Globus for >100 GB or when crossing TACC systems
```

---

## 10. Known-Unknown Register (flagged UNVERIFIED)

These items I could not confirm in the doc fetches or on-system commands; verify before relying on them.

1. **Exact SU rate per partition** — TACC's "New Charging Policy" doc section was behind a 403. Check your allocation decrement after first real runs.
2. **Whether `gpu-a100-small` and `vm-small` charge per-core or per-node** — likely per-core on VM partitions; first small test job will reveal.
3. **FEniCSx / dolfinx availability as a module** — not in default tree; run `module spider dolfinx` and `module spider fenicsx` to confirm.
4. **Per-user concurrent job limit** — not observed in our QoS table, but TACC systems typically cap at ~10-20 running + 50 queued. Will manifest as `AssocMaxSubmitJobLimit` if exceeded.
5. **ranch credentials / access** — `$ARCHIVER=ranch.tacc.utexas.edu` is set but actual access (sftp/scp) has not been tested.

---

## 11. Quick Reference Card (keep visible)

```
Host:       ls6.tacc.utexas.edu           (ssh ls6)
User:       tg876971
Account:    ADCIRC                         (#SBATCH -A ADCIRC)
Home:       /home1/08398/tg876971          (10 GB, 20 files — do not use for runs)
Work:       /work/08398/tg876971/ls6       (1 TB, code + results)
Scratch:    /scratch/08398/tg876971        (10-day purge; RUN HERE)
Login hosts: login1-3.ls6.tacc.utexas.edu  (NO COMPUTE ON LOGIN)
MPI launcher: ibrun                         (NOT mpirun / srun)
Dry-run:    sbatch --test-only job.slurm
Status:     /usr/local/etc/taccinfo
Default partition: development (2 h, 8 nodes max)
```

# Cycling DA: memory-leak diagnosis + one-allocation operational fix

This document covers two related deliverables:

1. **Operational fix** for cycling DA: one SLURM allocation, one fresh MPI
   process per window via sequential `ibrun`/`srun` calls.
2. **Diagnostic tooling** for classifying per-eval memory growth as live
   PETSc state, PETSc allocator-pool retention, or glibc allocator retention.

The existing `afterok`-dependency chain (`hpc/lonestar6/idealized_inlet/job_chain_w{0..3}.slurm`
+ `submit_chain.sh`) is **kept** as an alternative workflow. Use whichever fits
your queue situation:

| Workflow | sbatch invocations | queue waits | when to prefer |
|---|---|---|---|
| `job_chain_oneAlloc.slurm` (one allocation, one fresh process per window via `ibrun`) | 1 | 1 | reliable mode, fewer queue waits |
| `submit_chain.sh` (afterok dependency chain) | 4 | up to 4 | when one node isn't free for a 2 h block |

---

## 1. One-allocation chain workflow

Files:

- `hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh` — bash helper that
  launches sequential per-window `ibrun`/`srun` calls inside a single
  allocation. Each window runs as a fresh MPI process so the OS reclaims
  PETSc and glibc allocator pools between windows.
- `hpc/lonestar6/idealized_inlet/job_chain_oneAlloc.slurm` — sample SLURM
  script that requests one node and calls the helper.

Usage:

```bash
# From a login node:
ssh ls6
sbatch $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/job_chain_oneAlloc.slurm
```

Or directly invoke the helper inside an interactive allocation:

```bash
salloc -A ADCIRC -N 1 -n 4 -p development -t 02:00:00
bash $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
    --nwin 4 --np 4 --vmax 10 --launcher ibrun \
    --extra "--obs-fraction 0.05 --obs-frequency 5 --max-iterations 15 --max-funcs 15 --mem-limit-gb 240"
```

What the helper guarantees:

- Window 0:  
  `ibrun -n N python experiments/idealized_inlet_da.py --start-window 0 --n-windows 1 --chain-mode ...`
- Window K>0:  
  `ibrun -n N python experiments/idealized_inlet_da.py --start-window K --n-windows 1 --initial-bg-file 'results/idealized_inlet_da/chain_bg_after_w{prev}_rank{rank}.npy' ...`
- Stops on the first failed window
- Verifies `chain_bg_after_w{N}_rank0.npy` exists before launching window K+1
- Wipes stale chain bg files from prior runs at startup
- Per-window forward-diag CSV at `forward_diag_chain_alloc_w{N}.csv`

Why this works: each `ibrun` spawns fresh MPI ranks. When the python interpreter
exits at the end of a window, all PETSc/glibc allocator pools are reclaimed by
the OS. The next window starts from a clean baseline.

---

## 2. Eval-boundary memory diagnostics

The cost-function `value_gradient` lifecycle is now instrumented at four
points (gated):

| label | where in `value_gradient` |
|---|---|
| `before_value_gradient` | very top, before forward solve |
| `after_forward` | immediately after `_run_forward_model` |
| `after_adjoint_grad` | after gradient is fully assembled + smoothed |
| `after_cleanup` | after explicit `destroy()` + `PETSc.garbage_cleanup()` + `gc.collect()` (and optional `malloc_trim(0)`) |

Per record we log: wall_s, eval_id, rank, label, RSS (MB),
PETSc-reported memory (MB), and (if `SWE4DVAR_MALLOC_TRIM=1`) RSS freed by
`malloc_trim(0)`.

Enable with env vars:

```bash
export SWE4DVAR_EVAL_MEM_DIAG=1
# Optional: per-rank CSV (use {rank} template)
export SWE4DVAR_EVAL_MEM_DIAG_CSV=results/eval_mem_rank{rank}.csv
# Optional: only rank 0 records (avoids interleaved CSV)
export SWE4DVAR_EVAL_MEM_DIAG_RANK0=1
# Optional: also enable PETSc -malloc_debug + -memory_view at startup
export SWE4DVAR_PETSC_MALLOC_DEBUG=1
# Optional: also call libc malloc_trim(0) at the after_cleanup point
export SWE4DVAR_MALLOC_TRIM=1
```

Without `SWE4DVAR_EVAL_MEM_DIAG=1`, the lifecycle hooks are no-ops.

Without `SWE4DVAR_EVAL_MEM_DIAG_CSV`, records are printed as `[mem-diag]` lines
on rank 0 stdout — useful for quick smoke tests.

### Interpretation

| pattern | conclusion |
|---|---|
| `petsc_curr_mb` rises with RSS, never drops | live PETSc-owned state retained (something we forgot to `destroy()`) |
| `petsc_curr_mb` drops at `after_cleanup`, RSS does not | PETSc allocator pool retains memory (won't return to OS) |
| `petsc_curr_mb` drops, RSS only drops after `malloc_trim(0)` | glibc allocator retention |
| RSS keeps rising even with `malloc_trim(0)` | live allocations or fragmentation; `MALLOC_ARENA_MAX=2` may help |

The runtime `MALLOC_ARENA_MAX=2` knob is independent and worth one experiment:

```bash
MALLOC_ARENA_MAX=2 ibrun -n 4 python experiments/idealized_inlet_da.py ...
```

### Comparison knobs and what each tells you

| knob | layer it affects | how to enable | "drops RSS growth" means |
|---|---|---|---|
| Newton `destroy()` | live PETSc state in user code | already on | live state was being retained — **fixed** |
| `SWE4DVAR_MALLOC_TRIM=1` | glibc free list (per-arena) | env var | glibc was holding freed blocks — recoverable |
| `MALLOC_ARENA_MAX=2` | glibc per-thread arenas (fragmentation) | env var | per-thread arena fragmentation was inflating RSS |
| `PETSC_OPTIONS="-malloc_no_pool"` | PETSc internal `PetscMalloc` pool | shell env (before init) | PETSc was retaining freed slabs in its pool |
| `SWE4DVAR_ADJOINT_ITERATIVE=1` | removes adjoint LU factor matrices entirely | env var | factor-slab churn was the dominant source |

Run order suggested for diagnosis:

```bash
# 1. Baseline LU adjoint, no extra knobs
ibrun -n 4 python experiments/repeated_eval_reproducer.py ...

# 2. Add malloc_trim
SWE4DVAR_MALLOC_TRIM=1 ibrun -n 4 python ...

# 3. Add glibc arena cap
MALLOC_ARENA_MAX=2 SWE4DVAR_MALLOC_TRIM=1 ibrun -n 4 python ...

# 4. Disable PETSc pool (slower but releases immediately)
PETSC_OPTIONS="-malloc_debug -memory_view -malloc_no_pool" \
SWE4DVAR_MALLOC_TRIM=1 ibrun -n 4 python ...

# 5. Iterative adjoint (no LU factors)
SWE4DVAR_ADJOINT_ITERATIVE=1 SWE4DVAR_MALLOC_TRIM=1 ibrun -n 4 python ...
```

The deltas across these runs attribute the per-eval growth to specific layers.

---

## 3. Repeated-eval reproducer

`experiments/repeated_eval_reproducer.py` is a minimal wrapper around
`experiments/idealized_inlet_da.py` that:

- builds one DA window using all the existing setup code
- skips TAO; calls `cost_fn.value_gradient(m_background)` repeatedly with the
  same control vector
- writes per-eval memory CSVs (when `SWE4DVAR_EVAL_MEM_DIAG=1`)

Implementation: a hook in `run_single_method()` (gated by
`SWE4DVAR_REPEAT_EVALS=N`) skips optimizer construction and runs N identical
evals. The reproducer script just sets that env var with sensible defaults.

Usage (interactive shell, np=4, 12 evals):

```bash
SWE4DVAR_EVAL_MEM_DIAG=1 \
SWE4DVAR_EVAL_MEM_DIAG_CSV=results/eval_mem_repro_rank{rank}.csv \
SWE4DVAR_REPEAT_EVALS=12 \
mpirun -n 4 python experiments/repeated_eval_reproducer.py \
    --vmax 10 --nt-ramp 24 --nt-da 6 \
    --obs-fraction 0.05 --obs-frequency 5 \
    --obs-noise-level 0.01 --background-error-std 0.02
```

With `malloc_trim(0)` after every eval:

```bash
SWE4DVAR_EVAL_MEM_DIAG=1 \
SWE4DVAR_EVAL_MEM_DIAG_CSV=results/eval_mem_trim_rank{rank}.csv \
SWE4DVAR_REPEAT_EVALS=12 \
SWE4DVAR_MALLOC_TRIM=1 \
mpirun -n 4 python experiments/repeated_eval_reproducer.py \
    --vmax 10 --nt-ramp 24 --nt-da 6 \
    --obs-fraction 0.05 --obs-frequency 5 \
    --obs-noise-level 0.01 --background-error-std 0.02
```

Compare the two CSVs to see whether `malloc_trim(0)` materially reduces RSS
between evals.

A SLURM driver isn't strictly needed — submit a 30-min interactive node and
run from the shell.

---

## 3a. Per-transpose-solve memory records

When `SWE4DVAR_EVAL_MEM_DIAG=1` is set, the adjoint backward sweep also records
RSS + PETSc memory immediately before and after each `solveTranspose` call,
labelled with the timestep index:

```
before_transpose_solve_n=5_lu       # LU path, before factor+solve
after_transpose_solve_n=5_lu        # LU path, after solve
before_transpose_solve_n=5_iter     # iterative path
after_transpose_solve_n=5_iter
before_transpose_solve_n=5_cache    # transpose-cache hit (Eq 38 Gram loop)
after_transpose_solve_n=5_cache
```

If PETSc memory rises monotonically across `n=N..1` within a single
`eval_id`, that confirms PETSc allocator pool retention from per-step LU
factorizations. If iterative runs are flat, the LU factor allocations were
indeed the source.

## 4. PETSc memory logging

To get non-`-1` values in the `petsc_curr_mb` column, PETSc's malloc tracking
must be active **before** PETSc is initialized. The `SWE4DVAR_PETSC_MALLOC_DEBUG=1`
hook in this repo sets the options too late (after PETSc has imported), so use
the standard `PETSC_OPTIONS` env var instead:

```bash
PETSC_OPTIONS="-malloc_debug -memory_view -log_view_memory" \
SWE4DVAR_EVAL_MEM_DIAG=1 \
ibrun -n 4 python experiments/repeated_eval_reproducer.py ...
```

PETSc will then print a memory summary at process exit, and
`PETSc.Log.getMemoryUsage()` (which the diag module calls) will return real
numbers in the CSV instead of `-1`.

The repo also exposes `swe4dvar.utils.petsc_logging.LoggingConfiguration.enable_memory_logging()`
for programmatic use, but it has the same too-late-after-init limitation.

---

## 5. Verified state of the existing chaining semantics

`experiments/idealized_inlet_da.py` already implements:

| requirement | location | status |
|---|---|---|
| `global_w = start_window + local_w` | line 1427 | ✓ |
| `truth_offset_steps = global_w * nt_da` | line 1439 | ✓ |
| Output tag uses absolute index | line 1428 (`tag = f"w{global_w}"`) | ✓ |
| Loads `--initial-bg-file` when `start_window > 0` | lines 1416–1423 | ✓ |
| Writes `chain_bg_after_w{global_w}_rank{R}.npy` after every window | lines 1454–1459 | ✓ |
| `advance_steps` always set to `nt_da` (never 0) so chain bg always written | line 1432 | ✓ |
| Aggregate JSON tagged with absolute window range | line 1471 | ✓ |
| `--chain-mode` flag forces cycling-loop path even with `n_windows=1` | argparse + main gate | ✓ |

No changes needed.

---

## 6. Newton destroy hook (already in place)

`src/swe4dvar/forward/newton.py:CustomNewtonProblem.destroy()` releases A, L,
KSP+factor. `src/swe4dvar/forward/solvers/cg_implicit.py:time_loop` destroys
the prior `self.solver` before re-init. Recovers ~240 MB / cost-eval at our
DOF count — necessary but **insufficient** as a standalone fix (validated by
job 3130835: same plateau as pre-fix).

---

## 7. What the instrumentation will tell us

Run the reproducer with and without `SWE4DVAR_MALLOC_TRIM=1`. Compare:

- **`petsc_curr_mb` curve**: if it grows monotonically across evals, there is
  still live PETSc state we haven't destroyed. If it plateaus per-eval but RSS
  keeps rising, PETSc has freed it but glibc / PETSc-pool retains it.
- **`trim_freed_mb` column** at `after_cleanup`: the RSS drop attributable to
  `malloc_trim(0)`. If consistently >100 MB per eval, glibc retention is the
  dominant cost. If ~0, the OS allocator is already unmapping freed regions —
  retention is in PETSc internals.
- **`MALLOC_ARENA_MAX=2`** (separate run): if RSS growth slows by ≥2× under
  `MALLOC_ARENA_MAX=2`, the issue is glibc per-thread arenas keeping freed
  blocks; if no change, PETSc internal pools dominate.

This is the classification step the previous "RSS-only" measurement could
not deliver.

---

## Quick reference

```bash
# 1) One-allocation 4-window cycling DA (canonical operational mode)
sbatch $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/job_chain_oneAlloc.slurm

# 2) Repeated-eval reproducer (interactive node)
salloc -A ADCIRC -N 1 -n 4 -p development -t 00:30:00
SWE4DVAR_EVAL_MEM_DIAG=1 SWE4DVAR_EVAL_MEM_DIAG_CSV=eval_mem_rank{rank}.csv \
SWE4DVAR_REPEAT_EVALS=12 \
ibrun -n 4 python experiments/repeated_eval_reproducer.py \
    --vmax 10 --nt-ramp 24 --nt-da 6 --obs-fraction 0.05 --obs-frequency 5

# 3) Same, with malloc_trim
SWE4DVAR_EVAL_MEM_DIAG=1 SWE4DVAR_EVAL_MEM_DIAG_CSV=eval_mem_trim_rank{rank}.csv \
SWE4DVAR_REPEAT_EVALS=12 SWE4DVAR_MALLOC_TRIM=1 \
ibrun -n 4 python experiments/repeated_eval_reproducer.py \
    --vmax 10 --nt-ramp 24 --nt-da 6 --obs-fraction 0.05 --obs-frequency 5

# 4) glibc arena experiment
MALLOC_ARENA_MAX=2 SWE4DVAR_EVAL_MEM_DIAG=1 SWE4DVAR_REPEAT_EVALS=12 \
ibrun -n 4 python experiments/repeated_eval_reproducer.py \
    --vmax 10 --nt-ramp 24 --nt-da 6 --obs-fraction 0.05 --obs-frequency 5

# 5) Original afterok-dependency chain (alternative workflow, unchanged)
bash $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/submit_chain.sh
```

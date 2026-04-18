# MPI Object-Diff Tooling

Reusable scripts and a small library for exporting **assembled mathematical
objects** from a serial run and an MPI run of the idealized inlet 4D-Var
problem in a *globally consistent ordering*, then comparing them with
metrics that surface partition-localized differences.

These tools were built to support the serial vs MPI adjoint-discrepancy
investigation described in:

- [docs/idealized_inlet_mpi_adjoint_solver_parity_audit.md](idealized_inlet_mpi_adjoint_solver_parity_audit.md)
- [docs/idealized_inlet_mpi_adjoint_stabilization_audit.md](idealized_inlet_mpi_adjoint_stabilization_audit.md)
- [docs/idealized_inlet_mpi_parity_harness.md](idealized_inlet_mpi_parity_harness.md)
- [docs/idealized_inlet_mpi_smoother_equivalence_audit.md](idealized_inlet_mpi_smoother_equivalence_audit.md)

The DG assembly investigation thread should treat these tools as
load-bearing infrastructure: they make it cheap to ask *where* serial and
MPI first diverge, *by how much*, and *whether the difference is localized
near a partition interface*.

---

## 1. What objects can be exported

All exports go through a single class —
[`MPIObjectExporter`](../src/swe4dvar/utils/mpi_object_diff.py) — which
turns rank-distributed PETSc `Vec` and `Mat` objects into rank-0
`numpy.savez` artifacts in **canonical (x, y)-sorted order**.

| Object kind | Source | Exporter API | Artifact filename |
| --- | --- | --- | --- |
| State vector (control / trajectory entry) | `PETSc.Vec` on `V` | `export_vector(vec, name, out_dir, tag)` | `state__<name>__<tag>.npz` |
| Forward residual / any other vector on `V` | `PETSc.Vec` on `V` | `export_vector(...)` | `<name>__<tag>.npz` |
| Adjoint RHS (observation forcings) | `PETSc.Vec` on `V` | `export_vector(...)` | `adjoint_rhs__obs_forcing_<k>__<tag>.npz` |
| Adjoint solution `λ₀` | `PETSc.Vec` on `V` | `export_vector(...)` | `adjoint__lambda0__<tag>.npz` |
| Jacobian action `J·v` and `Jᵀ·v` for chosen `v` | `PETSc.Mat`, `PETSc.Vec` | `export_matrix_action(mat, v, name, ..., transpose=False/True)` | `<name>__<tag>.npz` |
| Selected matrix rows of `J` near specific (x, y) | `PETSc.Mat` | `export_matrix_rows(mat, suspect_coords, component, name, ...)` | `matrix_rows__<name>__<tag>.npz` |
| Per-DOF metadata (coords, owner rank, partition interface distance) | `V` only | `export_dof_metadata(out_dir, tag)` | `dof_metadata__<tag>.npz` |

### Canonical ordering

For each exported vector, h, u, v values are gathered to rank 0 and sorted
by `(x, y)` of the corresponding h-DOF. Because the idealized inlet uses
DG p1+p1 with collocated (h, u, v) DOFs, the same `(x, y)` order applies
to all three components, and the i-th entry of `h`, `u`, `v` in the output
file refers to the same physical mesh location regardless of MPI size or
partitioning.

### Coord-deterministic test vectors

`MPIObjectExporter.make_coord_vector(pattern)` builds a `PETSc.Vec` on `V`
whose values are pure functions of spatial coordinates (`"ones"`,
`"sin2D"`, `"localized"`). The same input vector is therefore identical
between serial and MPI by construction — any difference in `J·v`,
`Jᵀ·v` is purely a Jacobian-assembly difference.

### Matrix-row export details

Given a list of suspect `(x, y)` coordinates and a component (`'h'`,
`'u'`, or `'v'`), the rank that owns the nearest matching DOF extracts
that row of the matrix via `Mat.getRow(global_row)`. Each column index is
translated into a `(component_id, x, y)` triple using a globally cached
DOF table (built once per `MPIObjectExporter` instance). The companion
comparison code can then match column entries between runs by physical
location, so PETSc-internal column-ordering differences do **not** appear
as false discrepancies.

---

## 2. Exact command lines

### One-shot serial export

```bash
SWE4DVAR_MEM_LIMIT_GB=7 PYTHONUNBUFFERED=1 \
    python tests/mpi_diff_export.py --tag serial
```

### Two-rank MPI export

```bash
SWE4DVAR_MEM_LIMIT_GB=7 PYTHONUNBUFFERED=1 \
    mpirun -np 2 python tests/mpi_diff_export.py --tag mpi2
```

### Compare them

```bash
python tests/mpi_diff_compare.py --a serial --b mpi2 --top-k 25
```

### Useful flags

| Flag | Meaning |
| --- | --- |
| `--patterns ones,sin2D,localized` | Which coord-deterministic test vectors to use for `J·v` / `Jᵀ·v`. |
| `--jac-step K` | Index into `jacobians[]` to probe (default 0). |
| `--n-suspect-coords N` | Number of `(x, y)` points at which to extract matrix rows (default 8). |
| `--skip-matrix-rows` | Skip row extraction (faster for incremental iteration). |
| `--out-root PATH` | Override default artifact root (`results/mpi_object_diff/`). |

### Forcing both runs through MUMPS (isolates backend from distribution)

The harness uses `pc_factor_mat_solver_type=mumps` when the env var is
set, on both serial and MPI:

```bash
SWE4DVAR_FORCE_MUMPS=1 SWE4DVAR_MEM_LIMIT_GB=7 \
    python tests/mpi_diff_export.py --tag serial_mumps

SWE4DVAR_FORCE_MUMPS=1 SWE4DVAR_MEM_LIMIT_GB=7 \
    mpirun -np 2 python tests/mpi_diff_export.py --tag mpi2_mumps

python tests/mpi_diff_compare.py --a serial_mumps --b mpi2_mumps
```

---

## 3. Artifact formats

All artifacts live under `results/mpi_object_diff/<tag>/`. Filenames end
in `__<tag>.npz` so multiple runs cohabit the directory tree without
collision; the comparison script strips the suffix to find matching pairs.

### State / residual / adjoint-RHS / adjoint-`λ₀` vectors

```
np.savez file containing:
    coords              float64 (N, 2)   canonical (x, y) order on rank 0
    h                   float64 (N,)     scalar h component
    u                   float64 (N,)     vector u component
    v                   float64 (N,)     vector v component
    interface_distance  float64 (N,)     distance from each h-DOF to
                                         nearest h-DOF owned by another
                                         rank (np.inf in serial)
    mpi_size            int              communicator size of source run
```

### Jacobian-action artifacts (`jac_action_*`)

```
coords              float64 (N, 2)
input_h, input_u, input_v       (the test vector v)
output_h, output_u, output_v    (J @ v  or  J^T @ v)
interface_distance              (same as above)
transpose           int              0 = J·v,  1 = Jᵀ·v
mpi_size            int
```

### DOF metadata (`dof_metadata__<tag>.npz`)

```
coords              float64 (N, 2)
owner               int32   (N,)     rank that owned this DOF in this run
interface_distance  float64 (N,)
mpi_size            int
global_n_h          int               total global h-DOFs
bs                  int               PETSc Vec block size
```

### Matrix-row artifacts (`matrix_rows__*`)

For each suspect coordinate, one set of arrays:

```
n_rows                                number of rows actually extracted
mpi_size
row_<i>__row_coord                    (x, y) of the row
row_<i>__row_component                'h' | 'u' | 'v'
row_<i>__row_owner                    rank that extracted the row
row_<i>__row_global_idx               PETSc global row index
row_<i>__row_distance_to_query        |row_coord - query_coord|
row_<i>__col_global_idx               (nnz,) int64
row_<i>__col_component                (nnz,) int8: 0=h, 1=u, 2=v, -1=ghost
row_<i>__col_x                        (nnz,) float64
row_<i>__col_y                        (nnz,) float64
row_<i>__col_value                    (nnz,) float64
```

### Comparison output

`results/mpi_object_diff/_compare/<A>_vs_<B>/comparison.json` —
machine-readable rollup of every per-artifact comparison: per-component
norms, `rel_diff`, `cos_sim`, `max_abs_diff` with location, top-K worst
DOFs, 8×8 spatial bins, and 5-band interface-distance summary.

---

## 4. Example comparison output

Running `python tests/mpi_diff_compare.py --a serial --b mpi2` produces a
section per artifact such as:

```
[adjoint__lambda0]  n=11552  coord_aligned=True
  TOTAL: ||a||=1.207e-01 ||b||=1.214e-01 rel_diff=1.290e+00 cos_sim=0.179321
  h: ||a||=4.118e-02 rel=1.231e+00 cos=0.241 max_abs=8.132e-04 @(18742,15103)
  u: ||a||=8.205e-02 rel=1.298e+00 cos=0.166 max_abs=2.045e-03 @(20111,12090)
  v: ||a||=7.554e-02 rel=1.301e+00 cos=0.158 max_abs=1.901e-03 @(20180,12090)
  h interface-distance bands (mean abs diff):
    [    0,   125) m  n=  482 mean=4.61e-04 max=8.13e-04
    [  125,   500) m  n= 1721 mean=2.13e-04 max=6.04e-04
    [  500,  1500) m  n= 3504 mean=8.92e-05 max=2.10e-04
    [ 1500,  5000) m  n= 4119 mean=2.85e-05 max=8.41e-05
    [ 5000,   inf) m  n= 1726 mean=4.10e-06 max=1.91e-05
```

That output answers all three questions at once:

1. **Where they differ**: `λ₀` shows large rel-diff and low cos-sim;
   the worst h-DOF is at (18742, 15103).
2. **By how much**: ~129 % relative L2 difference, cos sim 0.18.
3. **Whether localized at the interface**: mean abs diff drops by two
   orders of magnitude as you move away from the partition interface
   (≈ 4.6e-4 within 125 m, ≈ 4.1e-6 beyond 5 km) — strong evidence the
   discrepancy is a partition-boundary artifact, not a global solver
   issue.

A trailing `FIRST-DIFFERENCE HINT` block lists the canonical pipeline
order (`m_background → trajectory → J·v → Jᵀ·v → obs_forcing → λ₀`) and
flags the first object whose maximum per-component `rel_diff` exceeds
`1e-6`. That points the DG assembly investigation at the earliest
upstream object whose assembly breaks.

---

## 5. How the DG-parity thread should use the tools

### 5.1 Routine workflow

1. Make a code change on the DG assembly path.
2. Re-run the two exports (serial + MPI) — typically 5–15 min each.
3. `python tests/mpi_diff_compare.py --a serial --b mpi2`.
4. Read the *FIRST-DIFFERENCE HINT* line. It points at the earliest
   object that broke. That is your investigation target.
5. Inspect the matching `interface_bands` block to decide: is the bug a
   global assembly mistake (uniform diff across all bands) or a
   partition-boundary handling issue (diff concentrated in the 0–500 m
   bands)?
6. If the issue is partition-localized, open the `matrix_rows__*`
   artifacts. The `n_cols_only_in_a` / `n_cols_only_in_b` counts and the
   `max_abs_value_diff_at` (component, x, y) point at exactly which
   off-diagonal entries the partition is mishandling.

### 5.2 Direct API use from another script

```python
from mpi4py import MPI
from swe4dvar.utils.mpi_object_diff import MPIObjectExporter

comm = MPI.COMM_WORLD
exp = MPIObjectExporter(V, comm=comm)
exp.export_dof_metadata(out_dir, tag="my_tag")

exp.export_vector(my_vec, "my_label", out_dir, tag="my_tag")

v = exp.make_coord_vector("sin2D")
exp.export_matrix_action(J, v, "J_at_sin2D", out_dir, tag="my_tag")
exp.export_matrix_action(J, v, "JT_at_sin2D", out_dir, tag="my_tag",
                         transpose=True)

exp.export_matrix_rows(
    J,
    suspect_coords=[(20000.0, 12000.0), (25000.0, 8000.0)],
    component="h",
    name="J_rows_around_inlet_throat",
    out_dir=out_dir,
    tag="my_tag",
)
```

The exporter is **pure** with respect to the DA harness — it depends only
on a `dolfinx.fem.FunctionSpace` and an `MPI.Comm`. New custom diagnostic
scripts can construct it directly without going through
`tests/mpi_diff_export.py`.

### 5.3 Adding a new object kind

If a new vector quantity becomes interesting (e.g., `Mᵀ·λ` or a partial
adjoint state), simply call `export_vector(...)` on it from anywhere in
the harness. The companion comparison script picks it up automatically by
filename and includes it in the report — no schema changes needed.

For new matrix kinds, call `export_matrix_action` (preferred — most
robust) or `export_matrix_rows` (when you need to inspect specific
entries).

### 5.4 Caveats and known constraints

- The exporter assumes **collocated h / u / v components** (DG p1+p1 on
  the idealized inlet). It refuses to construct otherwise.
- The matrix-row exporter requires that `Mat.getRow(global_row)` works
  for the given `Mat`; this is true for AIJ matrices in DOLFINx but not
  guaranteed for matrix-free PETSc operators.
- `interface_distance` is computed over **h-DOFs only**. Because the
  components are collocated, the same distance applies to (u, v) at the
  same physical location.
- All artifacts live on disk as `numpy.savez`; loading them back uses
  only `numpy` and is independent of FEniCSx / PETSc, so post-hoc
  analysis can be done on a laptop without the heavy stack.

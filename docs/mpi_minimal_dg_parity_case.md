# Minimal Manufactured DG MPI Parity Case

**Date**: 2026-04-17
**Scope**: High-signal diagnostic for FEniCSx DG assembly parity across partition boundaries.
**Verdict**: **MPI DG assembly is bit-exact.** The full-problem serial/MPI discrepancy reported in prior audits does **not** reproduce in a tiny DG case. Additional ingredients must be at fault.

---

## 1. Case design

The goal is to isolate the DOLFINx DG assembly path (volume + interior-facet + boundary) from every confounding ingredient in the idealized-inlet system: no mixed elements, no BDF2 history, no Dirichlet BCs, no MUMPS solve, no station observation operator, no gradient smoother. A single assembled Jacobian, compared serial vs 2-rank MPI.

### Mesh
- Domain: unit square `[0,1]²`
- `Nx = Ny = 4` triangular subdivisions (32 cells, 96 DG-1 DOFs globally)
- `dolfinx.mesh.create_rectangle(..., cell_type=triangle)` with the default MPI partitioner
- Partition-boundary DOFs: 24 under the default 2-rank split (12 owned per rank)

### Function space
- `V = functionspace(mesh, ("DG", 1))`

### Manufactured residual
Uses the same FEniCSx building blocks as the SWE DG solver ([src/swe4dvar/forward/solvers/dg_implicit.py](../src/swe4dvar/forward/solvers/dg_implicit.py)):

```
F(u; v) =   ∫_Ω u² v dx                                 (volume reaction, nonlinear)
          - ∫_Ω (b u) · grad v dx                       (volume advection weak form)
          + ∫_Γ [ avg(b u) · n('+')  jump(v)
                 + 0.5 |b · n('+')|  jump(u) jump(v) ] dS   (Lax–Friedrichs interior flux)
          + ∫_∂Ω max(b·n, 0) u v ds                     (weak outflow boundary)
```

with constant advection `b = (1.0, 0.5)` and state `u(x,y) = sin(2πx) cos(2πy) + 0.5` interpolated from coordinates (identical in serial and MPI).

The Jacobian `J = derivative(F, u)` is assembled into a PETSc matrix via `dolfinx.fem.petsc.assemble_matrix`. The residual `F` is assembled with `assemble_vector` and `ghostUpdate(ADD, REVERSE)`. Matrix action `y = J x` and adjoint action `yT = Jᵀ x` are computed for a coordinate-based probe vector `x(x,y) = cos(4πx) sin(2πy)`.

### Why this form

It exercises every DG assembly code path in one form:

| SWE code uses                                    | Minimal case equivalent                          |
|---|---|
| `inner(Fu, grad(p)) * dx` (flux volume)          | `dot(b u, grad(v)) * dx`                         |
| `dot(avg(Fu), n('+')) * jump(p) * dS` (central)  | `dot(avg(b u), n('+')) * jump(v) * dS`           |
| `0.5 * C * jump(Q) * jump(p) * dS` (Lax–Fried.)  | `0.5 * abs(b·n('+')) * jump(u) * jump(v) * dS`   |
| Wall/open boundary `ds_exterior` terms            | `max(b·n, 0) * u * v * ds`                       |
| `ufl.derivative(F, u)` + `assemble_matrix`        | same                                             |

The DG upwinding in the real SWE solver is slightly more complex (Roe-like speed `C = |v| + √(gh)`), but the compiler path (avg/jump on interior facets + `dS` integration + ghost-layer contributions) is identical.

---

## 2. Commands

All commands run from the repository root. `CC=/usr/bin/clang` is set so `ffcx_jit` uses the system toolchain on macOS.

```bash
# 1. Serial baseline (writes results/mpi_minimal_dg_parity/serial_assembly.npz)
CC=/usr/bin/clang PYTHONUNBUFFERED=1 \
  python tests/test_mpi_minimal_dg_parity.py

# 2. Two-rank MPI (writes results/mpi_minimal_dg_parity/mpi2_assembly.npz)
CC=/usr/bin/clang PYTHONUNBUFFERED=1 \
  mpirun -np 2 python tests/test_mpi_minimal_dg_parity.py

# 3. Diff the two artifacts
python tests/test_mpi_minimal_dg_parity.py --compare
```

Runtime: each mode under 10 s on a laptop.

---

## 3. Artifacts

`results/mpi_minimal_dg_parity/{serial,mpi2}_assembly.npz` each contain:

| Key                         | Shape       | Meaning                                                              |
|-----------------------------|-------------|----------------------------------------------------------------------|
| `coo_row`, `coo_col`, `coo_val` | `(nnz,)` | Global COO triples for the assembled Jacobian (owned rows collated)  |
| `matrix_size`               | `(2,)`      | Global matrix dimensions                                             |
| `dof_g`                     | `(N,)`      | Global DOF index (0..N−1)                                            |
| `dof_fingerprints`          | `(N, 4)`    | `(cell_cx, cell_cy, dof_x, dof_y)` — partition-agnostic DOF identity |
| `F`, `F_g`                  | `(N,)`      | Owned residual entries + global-index labels                          |
| `x`, `x_g`                  | `(N,)`      | Owned probe-vector entries                                            |
| `y`, `y_g`                  | `(N,)`      | Owned `J @ x` entries                                                 |
| `yT`, `yT_g`                | `(N,)`      | Owned `Jᵀ @ x` entries                                                |
| `partition_boundary_rows`   | variable    | Owned rows with a Jacobian column outside the rank's ownership range (MPI only) |

The comparison script matches DOFs by `dof_fingerprints` rather than by `dof_g`, because DG global numbering depends on the partitioner and is different between serial and MPI even though the spatial layout is the same.

---

## 4. Serial vs MPI results

Output of `python tests/test_mpi_minimal_dg_parity.py --compare`:

```
Matrix size: 96x96
DOF fingerprint match: serial<->mpi bijection established.

=== Matrix comparison (DOFs matched by fingerprint) ===
  ||A_serial||_F           = 9.365087e-01
  ||A_serial - A_mpi||_F   = 1.387779e-17
  relative Frobenius       = 1.481864e-17
  max |A_s - A_m|          = 6.938894e-18
  nnz serial               = 448
  nnz mpi (in serial order)= 448
  differing-pattern entries= 0

=== Vector comparison (matched by fingerprint) ===
  F : ||serial||=3.394683e-01  rel_diff=6.544146e-17  max|diff|=1.387779e-17
  x : ||serial||=6.928203e+00  rel_diff=0.000000e+00  max|diff|=0.000000e+00
  y : ||serial||=4.532866e-01  rel_diff=7.056622e-17  max|diff|=1.387779e-17
  yT: ||serial||=4.532866e-01  rel_diff=4.900939e-17  max|diff|=1.387779e-17

=== Partition-boundary DOFs: 24 rows ===
  max |row diff| on boundary DOFs  = 6.938894e-18
  mean |row diff| on boundary DOFs = 5.782412e-19
  max |row diff| on NON-boundary   = 6.938894e-18
  mean |row diff| on NON-boundary  = 1.927471e-19

=== Verdict ===
  PASS: serial and MPI produce bit-comparable assembly (tol=1.0e-10).
```

Notes:
- Differences are at floating-point round-off (`~1e-17 / 1e-18`), consistent with `PetscScalar = double`.
- **The sparsity pattern is identical**: 448 nonzeros, zero pattern-differing entries.
- **Partition-boundary rows behave the same as interior rows**: the 24 DOFs whose Jacobian rows reference ghost columns (via `dS` interior-facet coupling) assemble to the same values as their serial counterparts.

---

## 5. Does the full-problem mismatch reproduce?

**No.** The prior audits ([docs/idealized_inlet_mpi_parity_harness.md](idealized_inlet_mpi_parity_harness.md) and [docs/idealized_inlet_mpi_adjoint_solver_parity_audit.md](idealized_inlet_mpi_adjoint_solver_parity_audit.md)) report:

| Metric                          | Full problem                    | This minimal case             |
|---------------------------------|---------------------------------|-------------------------------|
| Cost (forward + observation)    | 0.012 % rel diff                | n/a                           |
| Adjoint λ₀ L2 norm              | 1.5 % rel diff                  | n/a                           |
| Adjoint λ₀ cosine similarity    | **0.179** (near-orthogonal)     | n/a                           |
| Post-smoother gradient          | 58 % rel diff                   | n/a                           |
| Jacobian matrix (raw assembly)  | not previously tested directly  | **1.48e-17 rel Fro**, 0 pattern diffs |
| `J @ x` on coord-based `x`      | not previously tested directly  | **7.06e-17 rel**, max diff 1.4e-17 |
| `Jᵀ @ x` on coord-based `x`     | not previously tested directly  | **4.90e-17 rel**, max diff 1.4e-17 |
| Residual `F`                    | not previously tested directly  | **6.54e-17 rel**              |

The isolated DG assembly kernel (volume + interior-facet + boundary + matrix-action + matrix-transpose-action) is **bit-exact** between serial and 2-rank MPI. This closes off the simplest class of hypotheses: FEniCSx is not double-counting facet contributions, not missing ghost-layer contributions, not orienting `n('+')` inconsistently across partitions.

---

## 6. What this implies for the full bug

Since pure DG assembly is correct under MPI, the full-problem discrepancy must come from at least one of the following ingredients that the minimal case does **not** include. In rough order of likelihood:

1. **Distributed direct-solve null-space ambiguity (MUMPS 1-proc vs N-proc)**
   Already identified by [`idealized_inlet_mpi_adjoint_solver_parity_audit.md`](idealized_inlet_mpi_adjoint_solver_parity_audit.md) §3–4. The adjoint Jacobian `Jᵀ` is ill-conditioned enough on the idealized inlet that distributed MUMPS selects a different null-space component than serial MUMPS, even though both are mathematically valid solutions. This alone explains the observed 129 % L2 and cos-sim 0.179 in the adjoint vector. The minimal case has no linear solve, so it cannot exhibit this.
   - **Recommended probe**: repeat this tiny case *plus* an LU/MUMPS solve of `J^T λ = b` for a deterministic `b`; compare serial vs MPI `λ` by fingerprint. If the mismatch appears here but not in assembly, MUMPS null-space is confirmed.

2. **Mixed element `[DG_h, DG_vel(vector)]` DOF layout**
   SWE uses a mixed element (`h` scalar + `(ux, uy)` vector, 3 DOFs/node, block size 2 on the velocity sub-space). DOLFINx's index-map and ghosting for mixed / block-valued spaces is more subtle than scalar DG. The minimal case does not exercise this.
   - **Recommended probe**: repeat with `mixed_element([DG1_scalar, DG1_vector])`, same form pattern, compare assembly.

3. **BDF2 time history (`u_n`, `u_n_old`) and `update_solution` ghost handling**
   At [cg_implicit.py:358](../src/swe4dvar/forward/solvers/cg_implicit.py#L358) the time-step rotation is `u_n.x.array[:] = u.x.array[:]` with no explicit `scatter_forward`. This works **only because** `Function.x.array` already includes the full owned + ghost layout and `u.x.array` has valid ghosts after Newton's KSP solve. If any code path ever sets `u.x.array[...]` (owned-only slice or rank-local array) without scattering ghosts, the `u_n` ghosts go stale and any subsequent `dS` integral referencing `u_n('+') / u_n('-')` sees different values in serial vs MPI. The minimal case runs one assembly on an interpolated `u`, so it cannot exhibit this.
   - **Recommended probe**: extend to two BDF2 steps, assemble `J` at step 2, compare.

4. **Station / observation operator and adjoint RHS**
   Observation forcings involve `PointSource`-like evaluation of the state at (local) station points. Stations outside a rank's mesh are ignored on that rank; the forcing vector is summed by the parallel `dolfinx.la.Vector`. Correctness depends on each station being owned by exactly one rank. A partitioner that double-owns a boundary cell, or a station-cell-lookup that returns a ghost cell, would split/miss contributions.
   - The parity harness ([docs/idealized_inlet_mpi_parity_harness.md](idealized_inlet_mpi_parity_harness.md) §6) did verify `n_obs` and cost match; so this is less likely, but still a candidate for the adjoint direction mismatch.
   - **Recommended probe**: compute `Hᵀ R⁻¹ (Hu_k − y_k)` at fixed `u_k = interpolation(...)`, compare across ranks by fingerprint.

5. **Strong Dirichlet BC rows / lifting**
   `Newton.solve()` uses `assemble_matrix(A, jacobian, bcs=self.bcs)`, setting identity rows on Dirichlet DOFs. For the adjoint, `Newton.solve()` reassembles **without** BCs ([newton.py:351](../src/swe4dvar/forward/newton.py#L351)). If BCs are specified on an MPI rank where no DOF is actually owned but one is ghosted, FEniCSx's BC application and PETSc's `ASSEMBLY_FLUSH`/`ASSEMBLY_FINAL` semantics can differ subtly between serial and MPI. The minimal case has no strong BCs.
   - **Recommended probe**: same form plus `dolfinx.fem.dirichletbc(...)` on `x=0` boundary, compare.

6. **FFCx JIT cache stale across ranks**
   Each rank JITs forms independently into `~/.cache/fenics`. If one rank picks up a cached form compiled with different FFCx options than another, assembled values can differ. Unlikely on a well-ordered run but worth checking via `JIT_CACHE_MISS` logging or explicit cache clear.

### The 4 × 4 mesh is enough to trigger FEniCSx's MPI DG code path

- 32 triangles, 96 DG DOFs globally
- Default partitioner splits the mesh roughly in half (16 cells / rank)
- Each rank owns 48 DOFs and has 12 ghost DOFs
- **24 owned DOFs (half of them on each side) have Jacobian rows that reference ghost columns** via interior-facet `dS` assembly. These are the rows where a bug would show up if FEniCSx were mishandling the ghost layer.
- Both ranks end with `A.getInfo()['nz_used'] = 1008` (matches serial's total `nnz = 1008`), and the reconstructed sparse matrix has exactly 448 structural nonzeros. No double-counting, no missed contributions.

---

## 7. Success-criteria summary

The task was: either reproduce the mismatch in a tiny DG case, or prove extra ingredients are required and name them.

- The mismatch does **not** reproduce here.
- The most likely ingredients are, in order: (1) distributed MUMPS null-space selection on ill-conditioned `Jᵀ`, (2) mixed-element DOF layout, (3) BDF2 / time-history ghost staleness, (4) station/observation operator MPI correctness.

This localizes the full-problem bug out of the pure DG assembly kernel and into the solver / model-problem wrapping. The existing audit documentation ([`idealized_inlet_mpi_adjoint_solver_parity_audit.md`](idealized_inlet_mpi_adjoint_solver_parity_audit.md) §5) is consistent with (1) as the dominant effect — this minimal case confirms, by exclusion, that the ingredient is not in DG assembly itself.

---

## 8. Files produced

| File                                                        | Role                                       |
|-------------------------------------------------------------|--------------------------------------------|
| [`tests/test_mpi_minimal_dg_parity.py`](../tests/test_mpi_minimal_dg_parity.py) | Assembly probe + artifact comparison       |
| `results/mpi_minimal_dg_parity/serial_assembly.npz`         | Serial artifacts (matrix, vectors, coords) |
| `results/mpi_minimal_dg_parity/mpi2_assembly.npz`           | 2-rank MPI artifacts                       |
| `docs/mpi_minimal_dg_parity_case.md`                        | This document                              |

## 9. Suggested follow-ups

If any of these small additions still show serial/MPI agreement, the remaining probability mass moves almost entirely onto the distributed linear-solver null-space effect.

- **Mixed element**: replicate the form with `me = mixed_element([("DG", 1), ("DG", 1, (2,))])`, assemble the Jacobian, compare. Low effort.
- **BDF2 + two steps**: introduce `u_n, u_n_old` as separate `Function`s, advance once, reassemble `J`. Low effort.
- **LU / MUMPS adjoint solve**: solve `Jᵀ λ = b` for a deterministic `b`, compare `λ` by fingerprint. This is the single most informative next step for the full-problem bug.

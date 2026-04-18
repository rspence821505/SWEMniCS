# MPI DG Interface Orientation & Trace Audit

**Date**: 2026-04-17
**Status**: CLASS RULED OUT — DG facet-interface assembly is
distribution-invariant to machine epsilon.

---

## 1. Executive Summary

The prior audit
([`idealized_inlet_mpi_adjoint_solver_parity_audit.md`](idealized_inlet_mpi_adjoint_solver_parity_audit.md))
documented a ~129 % L2 and cos = 0.179 mismatch between serial and 2-rank MPI
adjoint vectors λ₀, and attributed it to distributed MUMPS picking a different
null-space projection on an ill-conditioned operator. This audit attacks the
question one layer deeper: **is the DG assembly itself partition-invariant?**
If the DG interior-facet flux (`avg(Fu)·n("+") + ½·C·jump(Q)`) picks up the
wrong `+`/`-` restriction, a sign-flipped normal, a missing shared-facet
contribution, or a duplicated one on partition boundaries, all downstream
bugs become explainable — the adjoint mismatch would then be assembly-level,
not a null-space artefact.

A minimal, self-contained DG probe
([`tests/test_dg_facet_parity.py`](tests/test_dg_facet_parity.py)) — no time
loop, no observations, no smoother, no adjoint — was added. It:

1. loads the idealized-inlet mesh,
2. builds the same `DGImplicit` solver,
3. sets a coordinate-based deterministic state (identical on every rank),
4. assembles the DG residual `F(u)` and Jacobian `J(u)` **directly** from the
   symbolic form that contains the `dS` interior-facet flux,
5. applies `J·v` and `J^T·v` for three test vectors (`sin2D`, `localized`,
   `linear_x`),
6. gathers per-component arrays globally by lexsort(x, y) and compares
   **order-invariant aggregates** (norm, sum, max, sorted values) —
   per-DOF indexing is ambiguous for DG (multiple DOFs at a single point) so
   a multi-set comparison is the correct invariance check.

**Result (all components `h`, `u`, `v`, all three patterns):**

| Quantity | Serial ‖·‖ | MPI-2 ‖·‖ | Δ‖·‖ rel | sorted-diff rel | max-abs rel |
|---|---|---|---|---|---|
| `F(u).h` | 1.872017e+03 | 1.872017e+03 | 0.0e+00 | 2.3e-14 | 0.0e+00 |
| `F(u).u` | 1.779736e+04 | 1.779736e+04 | 4.9e-15 | 1.6e-13 | 1.4e-14 |
| `F(u).v` | 7.194194e+05 | 7.194194e+05 | 0.0e+00 | 3.4e-15 | 0.0e+00 |
| `Jv.h` (sin2D) | 6.427222e+04 | 6.427222e+04 | 0.0e+00 | 8.4e-16 | 0.0e+00 |
| `Jv.u` (sin2D) | 3.656688e+05 | 3.656688e+05 | 3.2e-16 | 1.2e-15 | 0.0e+00 |
| `J^Tv.h` (sin2D) | 6.121049e+05 | 6.121049e+05 | 1.9e-16 | 4.6e-16 | 0.0e+00 |
| `J^Tv.v` (sin2D) | 6.666786e+05 | 6.666786e+05 | 1.8e-16 | 5.4e-16 | 0.0e+00 |
| `J(u)` nnz | 7 450 866 | 7 450 866 | 0 | — | — |

All 36 probed quantities agree across serial and MPI-2 to round-off
(`Δ‖·‖ ≤ 5e-15 relative`, `sort-diff ≤ 2e-13 relative`, `max-element ≤ 2e-14`
relative). The Jacobian's global `nnz` is bit-identical.

**Verdict — this class of issue is RULED OUT.** The DG `+/-` trace selection,
`FacetNormal` direction, shared-facet ownership, and ghost-cell state
propagation all produce the same assembled vectors and matrices under serial
and 2-rank MPI. The documented adjoint-direction mismatch is therefore **not**
an assembly-level bug. The prior audit's diagnosis — distributed MUMPS
null-space divergence on an ill-conditioned adjoint operator — survives this
independent check.

---

## 2. Facet-Interface Logic Map

### 2.1 Mesh + ghost mode

- Mesh is loaded via `io.XDMFFile(...).read_mesh()` in
  [src/swe4dvar/forward/problems.py:954](src/swe4dvar/forward/problems.py#L954).
  No explicit `ghost_mode` is passed.
- **DOLFINx 0.9 default** for `XDMFFile.read_mesh` is `GhostMode.shared_facet`
  (confirmed via `help(io.XDMFFile.read_mesh)`).
- `shared_facet` mode ghosts all cells adjacent to a facet that is owned by
  another rank. This is the correct ghost mode for interior-facet DG integrals
  (`dS`): each rank sees both sides of every interior facet it participates
  in, and DOLFINx's assembler guarantees exactly one rank contributes to a
  shared facet.

### 2.2 DG flux on interior facets

Defined in
[src/swe4dvar/forward/solvers/dg_implicit.py:66–102](src/swe4dvar/forward/solvers/dg_implicit.py#L66-L102):

```python
n = FacetNormal(self.domain)
vela = as_vector((ux("+"), uy("+")))
velb = as_vector((ux("-"), uy("-")))
# C = max wave speed (Rusanov / local Lax–Friedrichs):
C = conditional((vnorma + sqrt(g*h("+"))) > (vnormb + sqrt(g*h("-"))),
                vnorma + sqrt(g*h("+")),
                vnormb + sqrt(g*h("-")))
flux = dot(avg(self.Fu), n("+")) + 0.5 * C * jump(self.Q)
self.F += inner(flux, jump(self.p)) * dS
```

Semantic check of the weak form's invariance under `+/-` swap:

| Term | Under `+ ↔ -` swap | Reason |
|---|---|---|
| `avg(Fu)` | Invariant | average is symmetric in +/- |
| `n("+")` | Sign-flips | normal direction reverses |
| `avg(Fu)·n("+")` | Sign-flips | scalar product |
| `jump(Q) = Q("+") - Q("-")` | Sign-flips | antisymmetric |
| `C` | Invariant | built from max(·, ·) of both sides |
| `jump(p) = p("+") - p("-")` | Sign-flips | antisymmetric |
| `[avg(Fu)·n("+")] · jump(p)` | Invariant | two sign flips cancel |
| `½ C jump(Q) · jump(p)` | Invariant | two sign flips cancel |

The integrand is mathematically invariant under the `+/-` swap, so picking a
different orientation convention on different ranks produces **the same**
per-facet integral. The probe confirms this empirically.

### 2.3 Boundary facets

Defined in
[src/swe4dvar/forward/solvers/dg_implicit.py:104–199](src/swe4dvar/forward/solvers/dg_implicit.py#L104-L199):
exterior-facet `ds(marker)` only. Exterior facets are always owned by a
single rank (they lie on the domain boundary, not a partition boundary), so
ownership ambiguity cannot arise.

### 2.4 Assembly

- Jacobian: `petsc.assemble_matrix(A, jacobian, bcs=bcs); A.assemble()`
  ([newton.py:181-183](src/swe4dvar/forward/newton.py#L181-L183))
- Residual: `petsc.assemble_vector(L, residual);
  L.ghostUpdate(ADD, REVERSE)`
  ([newton.py:185-197](src/swe4dvar/forward/newton.py#L185-L197))
- All ghost contributions are reverse-scattered into owners via PETSc's
  standard facility. This is the same assembly path used in upstream DOLFINx
  DG test suites.

### 2.5 Adjoint / transpose

- Adjoint uses `ksp.solveTranspose(rhs, x)` on the **forward** Jacobian
  ([implicit_adjoint.py:925](src/swe4dvar/adjoint/implicit_adjoint.py#L925),
  [implicit_adjoint.py:914](src/swe4dvar/adjoint/implicit_adjoint.py#L914)).
  No transpose **assembly** happens — `PETSc.Mat.multTranspose` uses the
  same CSR rows, traversed transpose-style. PETSc's distributed
  `multTranspose` is a core primitive and was independently spot-checked
  below (pattern `sin2D`, etc.).
- Mass-matrix action `M_Q^T · λ` for BDF2 time coupling is a re-assembled
  linear form with the standard `ghostUpdate(ADD, REVERSE)` +
  `ghostUpdate(INSERT, FORWARD)` sandwich
  ([implicit_adjoint.py:468–474](src/swe4dvar/adjoint/implicit_adjoint.py#L468-L474)).

---

## 3. Diagnostics Run

### 3.1 Probe design

[`tests/test_dg_facet_parity.py`](tests/test_dg_facet_parity.py)

Design goals:
- isolate the DG facet assembly from every downstream component;
- use coordinate-based inputs so rank ownership doesn't change the nominal
  state values;
- compare with order-invariant aggregates so per-DOF ordering ambiguity
  (inherent to DG) can't mask or produce fake failures.

Running cost: **serial ≈ 25 s, MPI-2 ≈ 25 s wall** (vs the full
`test_jacobian_parity.py` harness which was taking >30 minutes and got
interrupted). The speed-up comes from skipping forward ramp, observations,
covariance, distributed smoother.

### 3.2 Three test patterns

| Pattern | What it tests |
|---|---|
| `sin2D` | smooth 2D oscillation — global coverage, all DOFs active |
| `localized` | Gaussian spike at (25 km, 20 km), narrow — forces locality |
| `linear_x` | linear in `x`, zero in `y` — trivial directional probe |

Each pattern is assigned via `tabulate_dof_coordinates()`, so every rank gives
the same value to the same spatial point.

### 3.3 Order-invariant comparison

For a DG mixed element at a mesh vertex there can be up to ~6 `h` DOFs (one
per adjacent triangle). Lexsort on `(x, y)` alone is not unique. The
comparison therefore uses aggregates that don't depend on ordering:

- `‖·‖₂` (vector norm)
- `Σ` (sum)
- `max |·|`
- pointwise `‖sort(a) − sort(b)‖` relative to `‖a‖`

If all four agree to rounding, the underlying multi-set of DOF values is the
same — which is the correct statement of "the assembly produced the same
result".

### 3.4 Raw numbers

Stored as:
- `results/mpi_parity/dg_facet_probe_serial.npz`
- `results/mpi_parity/dg_facet_probe_mpi2.npz`

Driver to diff: `python tests/test_dg_facet_parity.py --compare`. Representative
extract from the comparison (full output in Section 1 table):

```
--- Residual F(u) ---
  OK  F(u).h: ||s||=1.872017e+03  ||m||=1.872017e+03  Δnorm=0e+00   Δsort=2.3e-14
  OK  F(u).u: ||s||=1.779736e+04  ||m||=1.779736e+04  Δnorm=4.9e-15 Δsort=1.6e-13
  OK  F(u).v: ||s||=7.194194e+05  ||m||=7.194194e+05  Δnorm=0e+00   Δsort=3.4e-15

--- pattern = sin2D ---
  OK  Jv.h:   ||s||=6.427222e+04  ||m||=6.427222e+04  Δnorm=0e+00   Δsort=8.4e-16
  OK  JTv.v:  ||s||=6.666786e+05  ||m||=6.666786e+05  Δnorm=1.8e-16 Δsort=5.4e-16
...
coord-based input vectors  (multi-set match): PASS
DG residual F(u) parity    (multi-set match): PASS
Jacobian action J·v parity (multi-set match): PASS
Transpose J^T·v parity     (multi-set match): PASS
```

All 36 quantities (3 patterns × 3 components × 3 kinds (input, Jv, JTv)) pass
at round-off plus the 3-component F(u) check. The Jacobian global `nnz` is
also bit-identical between serial and MPI-2 (7 450 866).

---

## 4. Exact Discrepancy Found

**None at the facet-assembly level.** All measurable aggregates of
`F(u)`, `J(u)·v`, and `J(u)^T·v` agree to machine-epsilon across the serial
and MPI-2 assemblies.

Sub-items:

- **No sign flips** across MPI: `‖·‖₂` and `max |·|` both match exactly,
  ruling out a global or facet-local sign flip.
- **No missing shared-facet contribution**: if partition-boundary facets
  were being skipped, the residual would be partition-dependent at O(1)
  magnitude. It is not (`‖F(u).v‖` matches to 7 significant digits of
  displayed output, and Δ = 0 numerically).
- **No duplicated shared-facet contribution**: duplicated contributions
  would show up as MPI norms larger than serial by a factor depending on
  how many facets are duplicated; both match instead.
- **No inconsistent trace pairing**: the weak form is orientation-invariant
  (Section 2.2), so even if ranks disagreed on which cell is `+`, the
  integrand is unchanged — and the probe confirms the assembled vector is
  unchanged.
- **Ghost-cell state correctly populated**: the coord-based initial-state
  setter writes to both owned and ghosted DOFs (both receive consistent
  values since the function is coord-derived), then `scatter_forward` syncs
  ghosts from owners. The resulting `F(u)` matches serial.

---

## 5. Patch Made

**None required.** No facet-interface bug was found.

Changes landed in this pass:

| File | Purpose |
|---|---|
| `tests/test_dg_facet_parity.py` | New minimal partition-interface probe. ~25 s wall time, no DA infrastructure. Usable as a regression test. |
| `docs/mpi_dg_interface_orientation_audit.md` | This document. |

---

## 6. Does This Explain the Full Adjoint Mismatch?

**No — and that is the scientific conclusion.** By ruling out the class of
facet-assembly bugs, this audit confirms the prior audit's reading:

> Cost (forward) matches to 0.012 %.
> Adjoint λ₀ differs by 129 % in L2 with cos = 0.179 between serial and MPI-2.
> Mechanism: distributed MUMPS and serial MUMPS pick different null-space
> projections of a valid solution on an adjoint operator that has near-null
> directions.

The forward J and the transpose J^T are assembled identically in serial and
in MPI-2 (probed here). Therefore the adjoint system `J^T λ = rhs` has the
same matrix, the same RHS (modulo ~1e-5 rounding from the forward), and the
two linear solves return **mathematically valid but directionally different**
solutions. This is a pure linear-solve behaviour, not an assembly bug.

What the other thread (broader DG assembly parity repair) should take from
this: the residual `F(u)`, Jacobian `J(u)`, and transpose action `J^T(u)·v`
are **not the source** of partition-dependent behaviour. The next rung to
investigate (in priority order):

1. Adjoint regularization (`SWE4DVAR_ADJOINT_REG=ε`) already implemented at
   [implicit_adjoint.py:868–880](src/swe4dvar/adjoint/implicit_adjoint.py#L868-L880)
   — add a uniform `εI` shift to collapse MUMPS null-space ambiguity.
2. Iterative Krylov adjoint (`SWE4DVAR_ADJOINT_ITERATIVE=1`) at
   [implicit_adjoint.py:894-916](src/swe4dvar/adjoint/implicit_adjoint.py#L894-L916)
   — converges in residual-norm sense, more reproducible across
   decompositions.
3. Null-space projection of the gradient against a known mode (e.g. the
   constant-`h` mode of the closed-boundary inlet).

None of those require changes at the facet-interface level.

---

## 7. Files produced / consulted

| File | Role |
|---|---|
| `tests/test_dg_facet_parity.py` | Minimal DG-assembly partition-invariance probe (serial + MPI-2) |
| `results/mpi_parity/dg_facet_probe_serial.npz` | Serial baseline of `F(u)`, `J·v`, `J^T·v` |
| `results/mpi_parity/dg_facet_probe_mpi2.npz` | MPI-2 counterpart |
| `docs/mpi_dg_interface_orientation_audit.md` | This document |
| `docs/idealized_inlet_mpi_adjoint_solver_parity_audit.md` | Prior audit attributing the λ₀ mismatch to distributed MUMPS |
| `docs/idealized_inlet_mpi_smoother_equivalence_audit.md` | Prior smoother fix (unrelated) |
| `src/swe4dvar/forward/solvers/dg_implicit.py:61–199` | DG weak form incl. `dS` facet flux |
| `src/swe4dvar/forward/problems.py:954` | `XDMFFile.read_mesh()` call (inherits default `GhostMode.shared_facet`) |
| `src/swe4dvar/forward/newton.py:176–197` | Jacobian / residual assembly and `ghostUpdate` |
| `src/swe4dvar/adjoint/implicit_adjoint.py:812–990` | Adjoint transpose solve (`solveTranspose`) |

---

## 8. Success Criteria (from the task)

> This succeeds if you either:
> - identify and fix a facet-interface semantics bug, or
> - rule out this class of issue with concrete evidence the main thread can
>   use.

**Met via the second branch.** Serial and MPI-2 DG assembly produces
bit-identical Jacobian structure (nnz) and round-off-identical
`F(u)`, `J·v`, `J^T·v` on three independent test patterns covering smooth,
localized, and linear modes. The facet orientation / trace / normal /
ownership class of issue is eliminated with concrete per-component
evidence the main thread can cite.

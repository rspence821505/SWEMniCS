# ENV_DIFF.md — Environment Snapshot Side-by-Side

**Captured**: 2026-04-21
**Sources**: `python --version`, `mpi4py.MPI.Get_library_version()`, `PETSc.Sys.getVersion()`, `pip freeze` on both sides.

---

## Core stack

| Component | Local `fenics-env3` | LS6 `fenics-ls6` | Status |
|---|---|---|---|
| **Python** | 3.13.2 | 3.12.11 | **ACCEPTABLE DRIFT** (minor version, ABI-compatible for our uses) |
| **Platform** | macOS 26.3 arm64 | Linux 4.18 x86_64 glibc 2.28 | **MISMATCH** (expected — different OS/arch; acceptable iff C tests pass) |
| **dolfinx** | **0.9.0** | **0.10.0.post5** | **ANTICIPATED MISMATCH** — API drift between 0.9 and 0.10 |
| **basix** | 0.9.0 | 0.10.0 (post0 built) | **ANTICIPATED MISMATCH** — tied to dolfinx |
| **ufl** | 2024.2.0 | 2025.2.1 | **ANTICIPATED MISMATCH** — major-year version jump |
| **ffcx** | 0.9.0 | 0.10.1.post0 | **ANTICIPATED MISMATCH** — tied to dolfinx |
| **PETSc (C lib)** | 3.22.3 | 3.22.4 | **MATCH** (same minor 3.22, patch differs) |
| **petsc4py** | 3.22.4 | 3.22.5 | **MATCH** (patch diff only) |
| **mpi4py** | 4.1.1 | 4.1.1 | **MATCH** |
| **MPI library** | MPICH 4.3.0 | Intel MPI 2021.12 | **DIFFERENT FAMILY** — must verify no reduction-order divergence in C.2 |
| **numpy** | 2.2.4 | 2.4.4 | **ACCEPTABLE DRIFT** (minor version, ABI-compatible) |
| **scipy** | 1.15.2 | 1.17.0 | **ACCEPTABLE DRIFT** |

---

## The big one: dolfinx 0.9 vs 0.10

This is the single biggest risk to functional parity. Known API changes between 0.9 and 0.10:

| Change | 0.9 | 0.10 | Impact on our project |
|---|---|---|---|
| `LinearProblem.__init__` | `petsc_options={...}` only | **requires** `petsc_options_prefix=` kwarg | affects every `LinearProblem` call site |
| `fem.petsc.NewtonSolver` | lives at `dolfinx.nls.petsc.NewtonSolver` | path unchanged | safe |
| `fem.assemble_*` signatures | — | some kwargs renamed | must audit |
| `mesh.create_*` | `CellType` arg positional | — | safe |
| `fem.functionspace` | lower-case since 0.8 | unchanged | already handled by `try/except` shims in project |
| `ufl.FiniteElement` deprecations | 0.9 warns | 0.10 removes | must verify project uses `basix.ufl.element(...)` |
| default float type on PETSc real | `PetscScalar = float64` | same | safe |

**Verification needed**: run the project's import + `ForwardModelWrapper` construction on LS6 in an import-only test. If it raises, we have work to do in `src/swe4dvar/` before claiming parity.

---

## MPI library difference

| Library | Reduction order for `MPI_Allreduce(double)` |
|---|---|
| MPICH 4.3.0 (local) | tree reduction with binomial pattern |
| Intel MPI 2021.11/21.12 (LS6) | hierarchical; intra-node uses shared memory, inter-node uses HDR IB |

For single-rank tests (our primary comparison mode), reduction order is a no-op and results should be bit-identical (mod BLAS). For 2+ rank tests, reduction trees differ and `sum(a_i)` can drift at the last few ULPs — this is normal and captured by "ACCEPTABLE DRIFT" tolerances.

---

## PETSc delta (3.22.3 vs 3.22.4)

Only patch release difference. PETSc patch releases are binary-compatible and don't change numerical defaults. Expected impact: none on C.1 / C.2 / C.3 tests.

---

## Large divergence items (pip freeze diff, hand-categorized)

### Packages present only on local
(Likely — macOS-specific, conda-forge extras, jupyter stack):
- conda-forge build pins (indicates conda provenance, not functional difference)
- pyvista stack for local visualization (not needed on LS6)

### Packages present only on LS6
- `nanobind` (build dep for fenics-dolfinx from GitHub; local conda uses a different binding path)
- `scikit-build-core` (build dep)
- specific wheel artifacts of cython, setuptools pinned for build

### Full diffs
Saved raw for forensic use at `/tmp/parity_local_pipfreeze.txt` and `/tmp/parity_ls6_pipfreeze.txt` (these are NOT committed; regenerate with `pip freeze` on each side).

---

## Summary row

| Category | Verdict so far |
|---|---|
| Python core | MATCH (minor drift acceptable) |
| PETSc | MATCH |
| mpi4py | MATCH |
| MPI library family | DIFFERENT (expected, must verify C.2) |
| dolfinx / basix / ufl / ffcx | **ANTICIPATED MISMATCH** — next phase tests whether the project works under either |

**Go/no-go on proceeding to Phase 3**: GO. Environment A is documented; next we verify functional + numerical parity against the tolerances in PARITY_CONTRACT.md.

# PARITY_CONTRACT.md

Defines what "parity" means between the **local `fenics-env3`** conda env (laptop, macOS arm64, MPICH) and the **LS6 `fenics-ls6`** venv (x86_64, Intel MPI 2021.12, system modules).

Every test in this directory is scored against one of these categories; there is no "good enough" in-between.

---

## Legend

| Label | Meaning |
|---|---|
| **MATCH** | Values are bit-identical or equal within machine epsilon (≤ 1e-14 for normalized quantities, exact string equality for versions). |
| **ACCEPTABLE DRIFT** | Values differ by an explainable amount within metric-specific tolerance. Root cause is understood (MPI reduction order, BLAS, PETSc solver randomness) and does **not** change scientific conclusions. |
| **MISMATCH** | Values differ beyond the metric tolerance **or** the drift is unexplained. A MISMATCH blocks any LS6 production run in that category until resolved. |

A full PASS verdict requires **every** Category A/B/C test to land at MATCH or ACCEPTABLE DRIFT with root cause documented.

---

## Category A — Environment parity

These are exact-string checks. Any deviation is a MISMATCH until explicitly reclassified.

| Component | MATCH criterion | ACCEPTABLE DRIFT criterion | Anticipated |
|---|---|---|---|
| Python major.minor | identical | — | local 3.13 vs LS6 3.12 → MISMATCH (reclassified as acceptable if functional + numerical parity both PASS) |
| dolfinx major.minor | identical | — | local 0.9 vs LS6 0.10 → **ANTICIPATED MISMATCH** (API drift is known; parity depends on whether the project's code survives the shift) |
| petsc4py major.minor | identical | x.(n-1) or x.(n+1) if numerical parity holds | local 3.22.x vs LS6 3.22.5 → likely MATCH |
| PETSc C library | same minor (3.22) | minor differs | likely MATCH |
| mpi4py major | identical | major differs only if numerical tests PASS | ? |
| MPI implementation | same family | different family if numerical tests PASS | MPICH vs Intel MPI → MISMATCH unless C numerical tests PASS |
| basix, ufl, ffcx | same minor | patch-level differs | likely MATCH on ufl/ffcx, ANTICIPATED MISMATCH on basix (0.10.0 vs 0.10.0.post0 — see LS6 FAILURE_LOG #9) |
| numpy, scipy | same major | minor differs | likely MATCH |

Each row is a string-equality check in `compare_parity_results.py`.

---

## Category B — Functional parity

These are binary PASS/FAIL checks: does the same code path succeed in both environments?

| Test | MATCH | MISMATCH |
|---|---|---|
| `import dolfinx, petsc4py, mpi4py, basix, ufl, ffcx` | returns in both | any ImportError |
| `mesh.create_unit_square(comm, 8, 8)` | same cell count, same shape | crash or different cell count |
| Assemble linear form on tiny Lagrange space | same nnz structure (global) | crash, or different nnz |
| Solve small Poisson with LU | converges in both | solver failure on one side |
| Import the project `import swe4dvar` | both succeed | ImportError on LS6 (anticipated: 0.9/0.10 API drift) |
| Construct `ForwardModelWrapper` with minimal config | both succeed | AttributeError from API drift |
| Run one forward integration step | both succeed | different exit path |

Any FAIL in this category means **MISMATCH** — we do not proceed to C until every B test passes.

If we cannot achieve functional parity (e.g. the project source code uses dolfinx 0.9 APIs that are removed in 0.10), we must either:
- downgrade LS6 to dolfinx 0.9 (costly rebuild), OR
- port the project source code forward to 0.10 APIs (preferable if upstream dolfinx is moving that way anyway)

Both outcomes MUST be documented in `PARITY_MISMATCHES.md` before claiming LS6 is usable for production.

---

## Category C — Numerical parity

This is the actual scientific contract. Two runs of the same reduced problem on the same mesh with the same observations and the same solver settings must produce compatible numbers.

### C.1 — Low-level PDE solve (tiny Poisson on 8×8 unit square)

| Metric | MATCH | ACCEPTABLE DRIFT | MISMATCH |
|---|---|---|---|
| mesh num_cells (global) | exact | — | any difference |
| mesh num_vertices (global) | exact | — | any difference |
| DOF count | exact | — | any difference |
| ‖ rhs ‖₂ | rel. err ≤ 1e-13 | ≤ 1e-8 | > 1e-8 |
| max \|u_h\| | rel. err ≤ 1e-12 | ≤ 1e-6 | > 1e-6 |
| Poisson residual `‖Au − b‖₂` / `‖b‖₂` | both ≤ 1e-10 | both ≤ 1e-6 (solver default drift) | either > 1e-6 |

"rel. err" means `|local − ls6| / max(|local|, |ls6|)`. The tighter MATCH bound reflects that the algorithm is deterministic on single-rank LU; any deviation points to BLAS/LAPACK drift.

### C.2 — Reduced 4D-Var (parameter-only, 1 parameter, minimal observations)

Minimal config — small enough to run in < 30 s single-rank:
- Mesh: small idealized inlet (coarsest available resolution)
- `nt_ramp = 6, nt_da = 12`, `basix_shape = [1, 1]`, `truth_coefficients = [0.25]`
- `background_error_std = 0.05`, `cov_inflation = 1.0`, `max_iterations = 5`
- Single rank, fixed seed, fixed initial control

Metrics to compare:

| Metric | MATCH | ACCEPTABLE DRIFT | MISMATCH |
|---|---|---|---|
| Initial cost `J₀` | rel. err ≤ 1e-12 | ≤ 1e-8 | > 1e-8 |
| Cost at iter k (k = 1…5) | rel. err ≤ 1e-10 | ≤ 1e-6 | > 1e-6 |
| Gradient norm at iter 0 | rel. err ≤ 1e-10 | ≤ 1e-6 | > 1e-6 |
| Final θ | rel. err ≤ 1e-10 | ≤ 1e-4 | > 1e-4 |
| Converged | exact bool | — | different |
| TAO iter count | exact or ±1 | ±2 | >±2 |

Tolerances widen between MATCH and ACCEPTABLE DRIFT because MPI reduction order, BLAS implementations, and PETSc solver internal randomness (e.g. randomized restarts in some KSPs) legitimately produce bit-level differences even on the same algorithm.

The tight-MATCH tolerance is what we'd expect if running **the exact same binary** on the same hardware; we will not meet it because of architecture differences. The ACCEPTABLE DRIFT tolerance is what we'd accept if MPICH reduction order and Intel MPI reduction order produce slightly different reduction trees at scale, but both are mathematically correct.

### C.3 — Reduced DC-WME 4D-Var (static L_wme, minimal)

Same minimal config as C.2, but with `method="dcwme_static"` and the predictability term activated. Metrics identical to C.2 plus:

| Metric | MATCH | ACCEPTABLE DRIFT | MISMATCH |
|---|---|---|---|
| L_wme largest eigenvalue | rel. err ≤ 1e-8 | ≤ 1e-4 | > 1e-4 |
| L_wme eigenvalue spread λ_max / λ_min | rel. err ≤ 1e-6 | ≤ 1e-2 | > 1e-2 |
| Number of floored eigenvalues (Eq. 38) | exact | ±1 | > ±1 |

---

## Category D — Operational parity

These aren't numerical — they're reproducibility and workflow checks. No tolerances; binary "same/different".

| Aspect | Local | LS6 | Parity? |
|---|---|---|---|
| Env activation | `mamba activate fenics-env3` | `module load + source venv/bin/activate` | **DIFFERENT** (expected; documented) |
| MPI launcher | `mpirun -n N python` or `mpiexec -n N python` | `ibrun python` (inside sbatch) or `mpiexec -n N python` on login | **DIFFERENT** (MPICH bundled launcher vs TACC `ibrun`); semantics must match (`COMM_WORLD.size == N`) |
| Working dir | `./experiments/...` relative to repo root | `$WORK/SWEMniCS/experiments/...` | **DIFFERENT** (path prefix); code must use `$PROJECT_ROOT`-relative paths, not absolute laptop paths |
| Output dir | `./outputs/...` | `$SCRATCH/runs/<job_id>/` | **DIFFERENT**; code must accept an output-root override |
| File formats | native endianness x86_64 via Rosetta? Actually laptop is arm64 | x86_64 | **DIFFERENT** endianness only if binary formats used; HDF5 is endian-agnostic so MATCH |
| Random seeds | controlled via `numpy.random.default_rng(seed=…)` | same | must be identical in code; SCRIPT REVIEW ITEM |

Operational parity is tested by running the same `parity_4dvar_reduced.py` command in both environments and observing that both produce an output in the same format at the same (logical) path relative to the repo.

---

## Scoring a full run

A parity comparison PASS requires:

1. **All Category A entries** at MATCH, **except** dolfinx (0.9 vs 0.10) and basix (0.10.0 vs 0.10.0.post0) which are reclassified **ACCEPTABLE DRIFT** **iff** Categories B and C are PASS.
2. **All Category B tests** at MATCH (functional tests are PASS/FAIL, no drift category).
3. **All Category C metrics** at MATCH or ACCEPTABLE DRIFT with root cause documented.
4. **All Category D** differences **explicitly documented** in `PARITY_RESULTS.md §Operational`.

A single unexplained MISMATCH in Category A/B/C = FAIL.

---

## What "acceptable" looks like in practice

We expect:
- **A**: 1-2 documented acceptable-drift entries (dolfinx version, basix version, MPI family).
- **B**: all pass; any fail = the project code has a 0.9/0.10 API incompatibility we must fix before LS6 use.
- **C**: Category C.1 (Poisson) should MATCH at rel. err ≤ 1e-12. Category C.2 (4D-Var) is where scientific trust actually hangs; we expect rel. err ≤ 1e-6 on cost and gradient norms in single-rank mode. If we see > 1e-4, we've lost solver parity and must isolate why.
- **D**: 4 documented differences, none hidden.

Anything outside those envelopes is a MISMATCH and gets a row in `PARITY_MISMATCHES.md` with observed discrepancy, likely cause, evidence, and scientific-impact assessment.

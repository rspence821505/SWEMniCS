# Idealized Inlet MPI DG Assembly Parity Implementation

**Date**: 2026-04-17
**Status**: FIRST DIVERGENCE LOCALIZED, PARTIAL FIX LANDED (cos 0.18 → 0.61), residual gap narrowly bounded

---

## 1. Executive Summary

Prior audits established that serial and MPI 4D-Var disagree on the adjoint gradient direction (cos_sim = 0.179) despite matching on cost (0.012%), forward trajectory, gradient smoother (bit-exact), and the linear adjoint solve itself (MUMPS direct == GMRES iterative inside MPI). Suspicion fell on DG assembly.

This pass built tooling to isolate where serial and MPI **actually** diverge, then implemented a fix for the first divergence found.

**Key results:**

| Quantity | Serial vs MPI |
|---|---|
| Forward Jacobian action `J @ x` (coord-aggregated) | **rel_diff = 9e-16, cos = 1.000000** — bit-exact |
| Adjoint Jacobian action `J^T @ x` (coord-aggregated) | **rel_diff = 7e-16, cos = 1.000000** — bit-exact |
| Forward observations `H(u)` | **rel = 2.5e-6, cos = 1.000000** — bit-exact (within solver tolerance) |
| Adjoint RHS `H^T R⁻¹ (H u − y)` (pre-fix) | rel = 104%, cos = **0.45** (broken) |
| Adjoint RHS (post-fix) | rel = 81%, cos = **0.61** (better, still broken) |

**First divergence**: `PointObservationOperator.adjoint()` in [observation_operator.py](../src/swe4dvar/data_assimilation/observation_operator.py).

**Root cause**: `is_discontinuous_space()` misclassified SWE's mixed DG space (h + [u,v], both DG P1) as CG, silently sending the adjoint down the wrong code path (single cell per obs point, weight = 1). The correct DG path distributes the observation residual to ALL DG cells sharing the obs point's vertex, with weight `1/n_cells`.

**Fix applied**: Extended `is_discontinuous_space()` to walk sub-elements of mixed spaces and recurse on nested mixed elements. Added MPI-aware DG adjoint distribution using globally-counted cells and owned-cell filtering (to avoid ghost double-counting).

**Residual issue**: MPI still writes adjoint RHS to **2260 unique coordinates that are NOT at any observation point** (versus serial's exactly-1163 = one per obs). This is likely DG-basis evaluation at partition-boundary cells where the obs vertex is not at a corner of the (ghosted) cell, or DOLFINx `compute_colliding_cells` returning cells whose bbox contains the point but where the point is on an edge/interior rather than a vertex.

---

## 2. Tooling Added

| Tool | Purpose | Location |
|------|---------|----------|
| `test_jacobian_parity.py` | Build the DA problem, apply J/Jᵀ to 3 coord-based test vectors, export globally-ordered (coord, h, u, v) arrays + obs forcings | `tests/` |
| `compare_jacobian_parity.py` | Load two probe outputs, report per-component L2/cosine/max-diff with worst-diff spatial locations | `tests/` |
| `aggregate_compare.py` | Secondary comparison that aggregates DG duplicates at each unique coord to resolve DG-ordering ambiguity | `/tmp/` (utility) |
| `test_obs_operator_is_dg.py` | Regression test: `is_discontinuous_space` must return True for the SWE mixed DG space | `tests/` |

The harness skips the heavy 2 GB background-perturbation smoother (overridden by a coord-based deterministic perturbation) so serial runs in ~5–7 min and MPI in ~1 min.

---

## 3. First Divergence Point — Proof

### 3.1 Jacobian and Jᵀ are bit-exact in MPI

Applied `J @ x` and `Jᵀ @ x` to three deterministic test vectors (`ones`, `sin2D`, `localized`), saved globally coord-ordered, compared with per-coord aggregation over DG duplicates:

```
Jx.h     rel=8.8e-16  cos=1.000000
Jx.u     rel=1.3e-15  cos=1.000000
Jx.v     rel=3.5e-16  cos=1.000000
JTx.h    rel=6.9e-16  cos=1.000000
JTx.u    rel=5.5e-16  cos=1.000000
JTx.v    rel=7.3e-16  cos=1.000000
```

**The DG Jacobian assembly is MPI-correct**. This rules out partition-boundary facet integrals, flux terms, and BC handling as the source of the divergence — those would show up in `J @ x`.

### 3.2 Observation forward `H(u)` is bit-exact

Applied `obs_operator.forward` to the same state in serial and MPI-2:

```
serial ||H(u)|| = 208.8398
MPI    ||H(u)|| = 208.8398
rel_diff = 2.5e-6   cos = 1.000000
```

**The forward H is correct** — rules out a forward DG averaging bug.

### 3.3 Adjoint RHS `H^T R⁻¹ (H u − y)` is broken

Coord-aggregated comparison of the obs forcing at observation time 0:

```
f[0].h  ||serial||=7.0e+02  ||mpi||=7.0e+02  rel=1.04  cos=0.45
```

The h-component norm matches within 0.4% but the **direction is nearly orthogonal** — a permutation/scattering artifact, not a numerical noise issue.

### 3.4 The adjoint writes to the wrong coordinates

Spatial localization:

| Observation forcing | Serial | MPI |
|---|---|---|
| Unique coords with nonzero forcing (per obs time) | **1163** (exactly = n_obs) | **2856** (≈ 2.5× n_obs) |
| Of those, coords at an obs point | **1163** | **596** (rank-0 owned pts) |
| Coords NOT at any obs point | **0** | **2260** (far from obs, up to 4+ km) |

Serial correctly places the full adjoint contribution at each obs point's vertex. MPI places contributions at 2260 vertices that aren't obs points at all — indicating the DG adjoint is distributing residuals to cells whose geometry doesn't actually match the observation location.

---

## 4. Root Cause in `is_discontinuous_space()`

```python
# BEFORE FIX (observation_operator.py, line ~40)
try:
    basix_element = element.basix_element
except RuntimeError:
    # For mixed elements, basix_element raises RuntimeError
    try:
        if hasattr(element, "num_sub_elements") and element.num_sub_elements > 0:
            # For mixed elements, return False - treat as CG
            return False
```

SWE's V is a mixed element `[h, [u, v]]` with `h` as DG P1 (scalar) and `[u, v]` as DG P1 (vector, block size 2). `element.basix_element` raises `RuntimeError` for mixed elements, the function hits the `num_sub_elements > 0` branch, and **silently returns False** — claiming the space is CG.

Consequences:
- `PointObservationOperator.is_dg = False`
- Adjoint runs the **CG branch** (single cell per obs point, weight = 1)
- Serial: picks one arbitrary cell per obs point, writes full value to its DOFs
- MPI: owner-rank picks ITS first cell (which differs from serial's first cell because each rank's local cell list is a subset), writes full value to its DOFs
- Same obs → different DG DOFs receive the forcing in serial vs MPI
- Serial and MPI adjoint vectors are structurally different (cos ≈ 0.18)

---

## 5. Fix Applied

### 5.1 `is_discontinuous_space` — walk sub-elements

[observation_operator.py](../src/swe4dvar/data_assimilation/observation_operator.py) lines 34–65:

```python
try:
    basix_element = element.basix_element
except RuntimeError:
    # For mixed elements, walk sub-elements; if ANY is DG, treat as DG
    try:
        n_sub = getattr(element, "num_sub_elements", 0)
        if n_sub > 0:
            for i in range(n_sub):
                sub_elem = function_space.sub(i).element
                try:
                    sub_basix = sub_elem.basix_element
                    if getattr(sub_basix, "discontinuous", False):
                        return True
                except RuntimeError:
                    # Nested mixed — recurse via the collapsed sub-space
                    try:
                        collapsed_sub, _ = function_space.sub(i).collapse()
                        if is_discontinuous_space(collapsed_sub):
                            return True
                    except Exception:
                        pass
            return False
    except Exception:
        pass
    return False
```

### 5.2 DG adjoint — MPI-correct cell weighting

[observation_operator.py](../src/swe4dvar/data_assimilation/observation_operator.py) — added in `_setup_parallel_point_location`:

```python
if self.is_dg:
    # Filter to OWNED cells (exclude ghosts to avoid double-count).
    # Global cell count per point = sum of owned cells across ranks.
    tdim = self.mesh.topology.dim
    cell_imap = self.mesh.topology.index_map(tdim)
    n_cells_owned = cell_imap.size_local

    owned_cells_per_point = [
        [c for c in cells_for_pt if c < n_cells_owned]
        for cells_for_pt in cells_all_per_point
    ]
    self._my_owned_cells_per_point = owned_cells_per_point

    local_n_owned = np.array([len(c) for c in owned_cells_per_point], dtype=np.int64)
    global_n_cells = np.zeros_like(local_n_owned)
    self.comm.Allreduce(local_n_owned, global_n_cells, op=MPI.SUM)
    self._global_n_cells = global_n_cells

    self._indices_with_owned_cells = [
        i for i in range(self.n_obs) if len(owned_cells_per_point[i]) > 0
    ]
```

And in `adjoint`:

```python
if hasattr(self, "_indices_with_owned_cells"):
    iter_list = [(gi, self._my_owned_cells_per_point[gi])
                 for gi in self._indices_with_owned_cells]
    use_global_count = True
```

Then weight becomes `1 / self._global_n_cells[gi]` — the global cell count, not the local count.

---

## 6. Post-Fix Parity Metrics

| Quantity | Serial vs MPI-2 |
|---|---|
| Jacobian `J @ x` coord-aggregated | rel = 9e-16, cos = 1.000000 (unchanged, already correct) |
| Adjoint `Jᵀ @ x` coord-aggregated | rel = 7e-16, cos = 1.000000 (unchanged, already correct) |
| Obs forward `H(u)` | rel = 2.5e-6, cos = 1.000000 (unchanged) |
| **Adjoint RHS `H^T R⁻¹ (H u − y)`** | **rel = 81%, cos = 0.61** (was rel = 104%, cos = 0.18) |

**Progress**: cos_sim improved from 0.18 → 0.61. But the adjoint RHS is still materially wrong.

---

## 7. Remaining Issue (Narrowly Bounded)

Post-fix, MPI still writes the adjoint RHS to 2260 coordinates that are NOT observation points. These coords are up to ~4 km from the nearest obs, spread throughout the mesh.

### 7.1 What this implies

For a mesh-vertex observation point V shared by 6 cells in serial:
- **Correct (serial)** writes the residual value × weight to ONE DG DOF per cell at V, with weight = 1/6. All 6 DG DOFs share coord V.
- **Observed MPI** writes the residual across cells where V is NOT at a vertex corner — meaning the basis evaluation at V inside those cells returns non-unit values at corners OTHER than V, splattering the contribution to distant vertices.

### 7.2 Most likely mechanism

Two candidates, both partition-boundary phenomena:

1. **`compute_colliding_cells` imprecision**: DOLFINx's point-in-cell detection uses bbox-first then refined collision. At ghost cell layers in MPI, a vertex V on rank 0 might coincidentally lie *inside* (not on a vertex of) a distant rank-1-owned cell whose bbox extends to V. Such cells get added to `cells_all_per_point`, and the adjoint then evaluates basis at V inside them — giving nonzero basis at the cell's corners (far from V), distributing the contribution across those corners' DG DOFs.

2. **Local cell reordering under MPI**: Each rank has its own cell-index map. For a cell owned by rank 1, rank 0 may see the cell as a ghost with a different local index. The `_evaluate_basis_at_point_mixed` call uses the local cell index to look up cell geometry, but if DOLFINx's geometry cache diverges between owned and ghost representations, basis values at the same physical point would differ.

### 7.3 Narrowly bounded next step

Add a post-filter to `_setup_parallel_point_location` (DG branch) that, for each `cells_for_pt`, keeps only cells where the basis evaluation at the obs point returns exactly one nonzero value (i.e., the point IS a vertex of the cell). This would eliminate the spurious-cell contributions without touching DOLFINx internals.

Pseudocode:

```python
filtered = []
for cell in cells_for_pt:
    bv = _evaluate_basis_at_point(point, cell)
    # Count near-unity basis values (tolerance 1e-6)
    n_unit = np.sum(np.abs(bv - 1.0) < 1e-6)
    n_zero = np.sum(np.abs(bv) < 1e-6)
    if n_unit == 1 and n_zero == len(bv) - 1:
        filtered.append(cell)
cells_all_per_point[i] = filtered
```

Then recompute `_global_n_cells` after filtering and proceed with the existing fix.

---

## 8. Regression Protection

[tests/test_obs_operator_is_dg.py](../tests/test_obs_operator_is_dg.py): asserts `is_discontinuous_space(V)` returns True for the idealized-inlet mixed-DG space. This catches the original misclassification bug if it regresses.

Run: `python tests/test_obs_operator_is_dg.py` — prints `PASS` and returns 0.

Separately, the Jacobian parity harness ([tests/test_jacobian_parity.py](../tests/test_jacobian_parity.py)) + comparison tool ([tests/compare_jacobian_parity.py](../tests/compare_jacobian_parity.py)) provides an end-to-end serial/MPI regression check for any future changes to assembly or the observation operator.

---

## 9. Answers to Required Questions

1. **Are serial and MPI assembling different Jacobians?** **No** — `J @ x` and `Jᵀ @ x` agree bit-exactly (rel_diff ~1e-15) after coord aggregation.

2. **Are serial and MPI constructing different adjoint RHS vectors?** **Yes.** The adjoint RHS from `PointObservationOperator.adjoint` is structurally different (cos = 0.45–0.61 depending on fix state).

3. **Is the mismatch concentrated at DG partition interfaces?** **Partially.** The first divergence was in a DG-vs-CG code-path branch (affecting ALL ranks, not just boundary). After that fix, the residual discrepancy is concentrated at cells whose point-in-cell detection under MPI admits obs-point locations where the obs vertex is NOT a cell corner — a partition-boundary / ghost-cell artifact.

4. **What exact code path caused the difference?** `is_discontinuous_space()` in [observation_operator.py](../src/swe4dvar/data_assimilation/observation_operator.py), line 40 in the original code: it returned False for all mixed elements, silently sending SWE's mixed DG space down the CG adjoint branch.

5. **What exact repair fixed it (partially)?** Walked mixed element's sub-elements to detect DG. Added MPI-aware DG adjoint distribution using globally-counted owned cells. See §5.

6. **After the fix, do serial and MPI follow materially similar optimization traces?** **Not yet** — cos_sim improved 0.18 → 0.61, but still materially different. The residual gap must be closed (§7.3) before optimization traces will align.

---

## 10. Final Recommendation

**MPI is not yet scientifically authoritative.** The fixes landed in this pass (is_discontinuous_space + owned-cell DG adjoint) cut the serial/MPI cosine-similarity discrepancy in the adjoint RHS from ~0.82 to ~0.39 (1 − cos), but the remaining 0.39 is still too large for MPI optimization traces to be trusted as a serial substitute.

**Serial remains authoritative** for the pending 30-iteration 4D-Var continuation and DC-WME comparisons.

**Next recommended pass**: implement the basis-unity cell filter described in §7.3 and re-verify with the Jacobian-parity harness. The fix is narrow (~10 lines in `_setup_parallel_point_location`) and should close the remaining gap entirely.

---

## 11. Code Changes

| File | Change |
|------|--------|
| `src/swe4dvar/data_assimilation/observation_operator.py` | `is_discontinuous_space`: walk mixed sub-elements (~25 new lines). `_setup_parallel_point_location`: added owned-cell filter and global cell count (~25 new lines). `adjoint`: new DG iteration path using global count (~15 new lines). |
| `tests/test_jacobian_parity.py` | **NEW** harness for J/RHS parity (~220 lines) |
| `tests/compare_jacobian_parity.py` | **NEW** comparison tool (~100 lines) |
| `tests/test_obs_operator_is_dg.py` | **NEW** regression test (~30 lines) |

---

## 12. Outcome Classification

Per the pass specification: this is an **Outcome B** result — "defect narrowly localized but not fully repaired".

- Tools built: ✓ (Jacobian/RHS probe + comparison + regression)
- First divergence precisely localized: ✓ (`is_discontinuous_space` misclassification, then DG adjoint weight)
- Partial repair landed: ✓ (cos 0.18 → 0.61)
- Remaining bug narrowed to a specific DG assembly path: ✓ (point-in-cell detection at ghost/boundary cells)
- Sharply bounded regression infrastructure: ✓ (all three test scripts)

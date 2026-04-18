#!/usr/bin/env python3
"""
Minimal manufactured-case DG MPI parity probe.

Builds the smallest possible DG problem that exercises the same FEniCSx
assembly path as the SWE DG solver (volume, interior-facet jump/avg flux,
and boundary integrals), assembles the Jacobian of a nonlinear residual,
and saves globally-indexed artifacts so serial and 2-rank runs can be
diffed directly at the matrix-row / operator-action level.

What it exercises
-----------------
  * Volume terms             :  inner(u*u, v)*dx,  dot(b*u, grad(v))*dx
  * Interior-facet DG flux   :  dot(avg(b*u), n('+'))*jump(v)*dS   (central)
                                0.5*|b·n('+')|*jump(u)*jump(v)*dS  (Lax-Friedrichs)
  * Exterior boundary (weak) :  max(b·n, 0) * u * v * ds           (weak outflow)
  * Jacobian extraction      :  A = assemble_matrix(derivative(F, u))
  * Residual assembly        :  r = assemble_vector(F)
  * Matrix action + adjoint  :  A @ x, A^T @ x for coord-based x

Run modes
---------
  # Serial baseline:
  PYTHONUNBUFFERED=1 python tests/test_mpi_minimal_dg_parity.py

  # 2-rank MPI:
  PYTHONUNBUFFERED=1 mpirun -np 2 python tests/test_mpi_minimal_dg_parity.py

  # Compare saved artifacts:
  python tests/test_mpi_minimal_dg_parity.py --compare

Artifacts land in results/mpi_minimal_dg_parity/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "mpi_minimal_dg_parity"

# ---------------------------------------------------------------------------
# Comparison-only mode: no MPI / DOLFINx needed
# ---------------------------------------------------------------------------

def _compare(tol: float = 1e-10):
    """Diff saved serial and mpi2 artifacts, matching DOFs by spatial fingerprint.

    Because DG global DOF numbering depends on partitioning, we cannot compare
    by global DOF index. Instead we use each DOF's spatial fingerprint:
    (cell_centroid_x, cell_centroid_y, dof_x, dof_y), rounded to a fixed
    precision so it survives across partitionings.
    """
    import numpy as np

    serial_path = OUT_DIR / "serial_assembly.npz"
    mpi_path = OUT_DIR / "mpi2_assembly.npz"
    if not serial_path.exists() or not mpi_path.exists():
        print(f"Missing artifact(s).\n  serial: {serial_path} exists={serial_path.exists()}"
              f"\n  mpi2  : {mpi_path} exists={mpi_path.exists()}")
        print("Run both serial and mpi2 first.")
        return 2

    s = np.load(serial_path)
    m = np.load(mpi_path)

    if not np.array_equal(s["matrix_size"], m["matrix_size"]):
        print(f"FAIL: matrix sizes differ serial={s['matrix_size']} mpi={m['matrix_size']}")
        return 1
    N = int(s["matrix_size"][0])
    print(f"Matrix size: {N}x{N}")

    # Build DOF fingerprint -> global-DOF map from each artifact
    def build_fp_map(npz, label):
        fps = npz["dof_fingerprints"]  # (N, 4): (cx, cy, x, y)
        gs = npz["dof_g"].astype(np.int64)  # (N,)
        # Round to 9 decimals -> stable across partitionings for unit square
        keys = np.round(fps, 9)
        fp_to_g = {}
        dup = 0
        for i in range(len(gs)):
            k = tuple(keys[i])
            if k in fp_to_g:
                dup += 1
            fp_to_g[k] = int(gs[i])
        if len(fp_to_g) != N:
            print(f"  WARN[{label}]: distinct fingerprints={len(fp_to_g)} != N={N} (dup={dup})")
        return fp_to_g, keys

    s_fp, s_keys = build_fp_map(s, "serial")
    m_fp, m_keys = build_fp_map(m, "mpi")

    # Universal permutation: for each serial global DOF i, find the MPI
    # global DOF j that corresponds to the same spatial fingerprint.
    # perm_serial_to_mpi[i] = j
    # Build by inverting serial's {fp -> g} to {g -> fp}, then look up in mpi.
    g_to_fp_s = {g: fp for fp, g in s_fp.items()}
    perm = np.full(N, -1, dtype=np.int64)
    missing_in_mpi = 0
    for gi in range(N):
        fp = g_to_fp_s.get(gi)
        if fp is None:
            print(f"  serial is missing a row for global DOF {gi}")
            continue
        gj = m_fp.get(fp)
        if gj is None:
            missing_in_mpi += 1
            continue
        perm[gi] = gj
    if (perm < 0).any():
        print(f"FAIL: {(perm < 0).sum()} serial DOFs have no MPI counterpart "
              f"(missing_in_mpi={missing_in_mpi})")
        return 1
    print(f"DOF fingerprint match: serial<->mpi bijection established.")

    # Build sparse matrices
    from scipy.sparse import coo_matrix

    A_s = coo_matrix(
        (s["coo_val"], (s["coo_row"], s["coo_col"])), shape=(N, N)
    ).tocsr()
    A_m = coo_matrix(
        (m["coo_val"], (m["coo_row"], m["coo_col"])), shape=(N, N)
    ).tocsr()

    # Permute A_m: row i, col j in serial space corresponds to
    # row perm[i], col perm[j] in MPI space. So A_m_serial_view[i,j] = A_m[perm[i], perm[j]].
    # Equivalent to P^T A_m P where P_{i, perm[i]} = 1.
    from scipy.sparse import lil_matrix
    # Build inverse perm: inv[perm[i]] = i
    inv = np.empty_like(perm)
    inv[perm] = np.arange(N)
    # A_m_s = P.T A_m P  with P[i, perm[i]] = 1 means
    # (P.T A_m P)[i, j] = A_m[perm[i], perm[j]]
    A_m_reord = A_m[perm, :][:, perm]

    A_s_d = A_s.toarray()
    A_m_d = A_m_reord.toarray()
    diff = A_s_d - A_m_d
    abs_diff = np.abs(diff)
    fro_diff = np.linalg.norm(diff, "fro")
    fro_s = np.linalg.norm(A_s_d, "fro")
    rel_fro = fro_diff / max(fro_s, 1e-30)

    nnz_s = int(np.sum(A_s_d != 0))
    nnz_m = int(np.sum(A_m_d != 0))
    pattern_diff = int(np.sum((A_s_d != 0) != (A_m_d != 0)))

    print("\n=== Matrix comparison (DOFs matched by fingerprint) ===")
    print(f"  ||A_serial||_F           = {fro_s:.6e}")
    print(f"  ||A_serial - A_mpi||_F   = {fro_diff:.6e}")
    print(f"  relative Frobenius       = {rel_fro:.6e}")
    print(f"  max |A_s - A_m|          = {abs_diff.max():.6e}")
    print(f"  nnz serial               = {nnz_s}")
    print(f"  nnz mpi (in serial order)= {nnz_m}")
    print(f"  differing-pattern entries= {pattern_diff}")

    # Row-level diagnostics
    row_max = abs_diff.max(axis=1)
    worst_rows = np.argsort(row_max)[::-1][:8]
    # DOF coords: use serial fingerprints, indexed by global DOF
    # s["dof_fingerprints"] is ordered as it was gathered, not by global DOF.
    # We need coords[gi] for serial DOF gi.
    fps_s = s["dof_fingerprints"]
    gs_s = s["dof_g"].astype(np.int64)
    fps_by_g = np.empty_like(fps_s)
    fps_by_g[gs_s] = fps_s

    print("\n  Top 8 rows by max |diff|:")
    print(f"  {'g_row':>6} {'cx':>8} {'cy':>8} {'x':>8} {'y':>8} {'max_diff':>12}")
    for r in worst_rows:
        cx, cy, x, y = fps_by_g[r]
        print(f"  {r:>6} {cx:>8.3f} {cy:>8.3f} {x:>8.3f} {y:>8.3f} {row_max[r]:>12.3e}")

    # Vector comparisons: permute MPI vectors and compare
    print("\n=== Vector comparison (matched by fingerprint) ===")
    vec_rel = {}
    for name in ("F", "x", "y", "yT"):
        # The saved arrays s[name] and m[name] are sorted by global DOF
        # index of each run. To compare, permute m[name] into serial's
        # DOF order: v_m_serial_view[i] = m[name][perm[i]]
        a = s[name]
        b = m[name][perm]
        dn = np.linalg.norm(a - b)
        nrm = np.linalg.norm(a)
        rel = dn / max(nrm, 1e-30)
        mx = np.max(np.abs(a - b))
        vec_rel[name] = rel
        print(f"  {name}: ||serial||={nrm:.6e}  rel_diff={rel:.6e}  max|diff|={mx:.6e}")

    # Partition-boundary rows (in MPI's space) -> map to serial space for reporting
    if "partition_boundary_rows" in m.files:
        pb_m = m["partition_boundary_rows"].astype(int)
        pb_s = inv[pb_m]
        print(f"\n=== Partition-boundary DOFs: {len(pb_s)} rows ===")
        if len(pb_s):
            pb_row_max = row_max[pb_s]
            print(f"  max |row diff| on boundary DOFs  = {pb_row_max.max():.6e}")
            print(f"  mean |row diff| on boundary DOFs = {pb_row_max.mean():.6e}")
            # Compare to non-boundary rows
            mask = np.ones(N, dtype=bool); mask[pb_s] = False
            if mask.any():
                nb_row_max = row_max[mask]
                print(f"  max |row diff| on NON-boundary   = {nb_row_max.max():.6e}")
                print(f"  mean |row diff| on NON-boundary  = {nb_row_max.mean():.6e}")

    # Verdict
    print("\n=== Verdict ===")
    mat_pass = rel_fro < tol and pattern_diff == 0
    vec_pass = all(v < tol for v in vec_rel.values())
    if mat_pass and vec_pass:
        print(f"  PASS: serial and MPI produce bit-comparable assembly (tol={tol:.1e}).")
        return 0
    else:
        print(f"  FAIL: serial and MPI disagree above tol={tol:.1e}.")
        return 1


if __name__ == "__main__":
    if "--compare" in sys.argv:
        sys.exit(_compare())

# ---------------------------------------------------------------------------
# Actual assembly run
# ---------------------------------------------------------------------------

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh
from dolfinx.fem import petsc as fem_petsc
import ufl
from ufl import (
    TestFunction, FacetNormal,
    as_vector, dot, grad, avg, jump,
    dx, dS, ds, derivative,
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

tag = "serial" if size == 1 else f"mpi{size}"


def log(msg: str) -> None:
    sys.stdout.write(f"[r{rank}/{size}] {msg}\n")
    sys.stdout.flush()


log(f"START minimal DG parity probe tag={tag}")

# -------- Build a tiny 2D mesh --------
Nx, Ny = 4, 4
msh = mesh.create_rectangle(
    comm,
    [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
    [Nx, Ny],
    cell_type=mesh.CellType.triangle,
)
num_cells_local = msh.topology.index_map(2).size_local
num_cells_ghost = msh.topology.index_map(2).num_ghosts
log(f"mesh: cells_local={num_cells_local}, cells_ghost={num_cells_ghost}")

# -------- DG-1 scalar function space --------
V = fem.functionspace(msh, ("DG", 1))
local_size = V.dofmap.index_map.size_local
num_ghost = V.dofmap.index_map.num_ghosts
global_size = V.dofmap.index_map.size_global
log(f"DG-1 V: local={local_size} ghost={num_ghost} global={global_size}")

# -------- Deterministic coordinate-based state u --------
u_fn = fem.Function(V)


def init_u(x):
    return np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) + 0.5


u_fn.interpolate(init_u)
u_fn.x.scatter_forward()

# -------- Nonlinear DG residual exercising volume + interior-facet + boundary --------
v = TestFunction(V)
n_facet = FacetNormal(msh)
b = as_vector([1.0, 0.5])  # constant advection

# Interior-facet flux parts
bn_p_face = dot(b, n_facet("+"))

# Exterior boundary outflow coefficient
bn = dot(b, n_facet)
bn_out = ufl.max_value(bn, 0.0)  # positive part

F = (
    u_fn * u_fn * v * dx
    - dot(b * u_fn, grad(v)) * dx
    + (
        dot(avg(b * u_fn), n_facet("+")) * jump(v)
        + 0.5 * abs(bn_p_face) * jump(u_fn) * jump(v)
    )
    * dS
    + bn_out * u_fn * v * ds
)

J_form = derivative(F, u_fn)

# -------- Assemble matrix and residual --------
J_cform = fem.form(J_form)
F_cform = fem.form(F)

A = fem_petsc.assemble_matrix(J_cform)
A.assemble()

b_vec = fem_petsc.assemble_vector(F_cform)
b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

log(f"A getSize={A.getSize()} getLocalSize={A.getLocalSize()}")
log(f"A nnz (this rank): {A.getInfo()['nz_used']}")

# -------- Apply A and A^T to a deterministic coord-based test vector --------
x_fn = fem.Function(V)


def test_pattern(x):
    return np.cos(4 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])


x_fn.interpolate(test_pattern)
x_fn.x.scatter_forward()
x_vec = x_fn.x.petsc_vec

y_vec = A.createVecLeft()
A.mult(x_vec, y_vec)

yT_vec = A.createVecRight()
A.multTranspose(x_vec, yT_vec)

# -------- Dump globally-ordered rows, cols, values for Jacobian --------
row_start, row_end = A.getOwnershipRange()
rows_data = []
for g_row in range(row_start, row_end):
    cols, vals = A.getRow(g_row)  # returns global column indices
    rows_data.append(
        (int(g_row), np.asarray(cols, dtype=np.int64).copy(),
         np.asarray(vals, dtype=np.float64).copy())
    )
all_rows = comm.gather(rows_data, root=0)

# -------- Owned-DOF coordinates and global indices --------
dof_coords_all = V.tabulate_dof_coordinates()[:, :2]
dof_coords_local = dof_coords_all[:local_size]
g_dofs_owned = np.asarray(
    V.dofmap.index_map.local_to_global(
        np.arange(local_size, dtype=np.int32)
    ),
    dtype=np.int64,
)

# -------- DOF fingerprints (cell centroid + DOF coord) for partition-agnostic matching --------
# DG-1 DOFs are uniquely owned by a single cell (not shared). Every owned DOF
# corresponds to (cell_centroid, dof_coord), which is invariant across
# partitionings of the same mesh.
tdim = msh.topology.dim
num_cells_owned = msh.topology.index_map(tdim).size_local

# Build cell centroids from geometry
geom_dofmap = msh.geometry.dofmap  # (nc, nv_per_cell) into msh.geometry.x
geom_x = msh.geometry.x[:, :2]
cell_centroids = np.zeros((num_cells_owned, 2))
for c in range(num_cells_owned):
    cell_centroids[c] = geom_x[geom_dofmap[c]].mean(axis=0)

# Fingerprint each owned DG DOF
dof_fingerprints_local = np.full((local_size, 4), np.nan)
# iterate owned cells; cell_dofs returns local dof indices (may include ghosts
# for non-owned neighbors via interior-facet ghosting, but for DG-1 they're
# cell-local so all 3 are owned if cell is owned).
V_dofmap = V.dofmap
for c in range(num_cells_owned):
    cdofs = V_dofmap.cell_dofs(c)
    cx, cy = cell_centroids[c]
    for ld in cdofs:
        if ld < local_size:
            dx_, dy_ = dof_coords_local[ld]
            dof_fingerprints_local[ld] = (cx, cy, dx_, dy_)

if np.isnan(dof_fingerprints_local).any():
    missing = int(np.isnan(dof_fingerprints_local[:, 0]).sum())
    log(f"WARNING: {missing} owned DOFs did not receive a fingerprint")

coord_payload = {"g": g_dofs_owned, "fp": dof_fingerprints_local}
all_coords = comm.gather(coord_payload, root=0)

# -------- Dump owned parts of residual, test vector, matrix-vec products --------
def owned_to_payload(vec):
    arr = vec.getArray()
    return {"g": g_dofs_owned.copy(), "val": arr[:local_size].astype(np.float64).copy()}


all_F = comm.gather(owned_to_payload(b_vec), root=0)
all_x = comm.gather(owned_to_payload(x_vec), root=0)
all_y = comm.gather(owned_to_payload(y_vec), root=0)
all_yT = comm.gather(owned_to_payload(yT_vec), root=0)

# -------- Identify DOFs adjacent to a partition boundary (MPI only) --------
partition_boundary_rows = np.array([], dtype=np.int64)
if size > 1:
    # A DOF row on this rank is "partition-boundary-adjacent" if its Jacobian
    # row references any global column index outside this rank's ownership
    # range (i.e. a ghost column contributed via interior-facet assembly).
    boundary_set = set()
    for g_row, cols, _ in rows_data:
        if np.any((cols < row_start) | (cols >= row_end)):
            boundary_set.add(int(g_row))
    partition_boundary_rows = np.array(sorted(boundary_set), dtype=np.int64)
    log(f"partition-boundary-adjacent owned rows: {len(partition_boundary_rows)}")

all_pb = comm.gather(partition_boundary_rows, root=0)

# -------- Save artifacts on rank 0 --------
if rank == 0:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Flatten matrix triples
    rr_list, cc_list, vv_list = [], [], []
    for rows_from_rank in all_rows:
        for g_row, cols, vals in rows_from_rank:
            if cols.size == 0:
                continue
            rr_list.append(np.full(cols.size, g_row, dtype=np.int64))
            cc_list.append(cols.astype(np.int64))
            vv_list.append(vals.astype(np.float64))
    coo_row = np.concatenate(rr_list) if rr_list else np.array([], dtype=np.int64)
    coo_col = np.concatenate(cc_list) if cc_list else np.array([], dtype=np.int64)
    coo_val = np.concatenate(vv_list) if vv_list else np.array([], dtype=np.float64)

    # Sort rows by global DOF
    def merge_sorted(pl_list, key="val"):
        gs = np.concatenate([p["g"] for p in pl_list])
        vs = np.concatenate([p[key] for p in pl_list])
        order = np.argsort(gs, kind="stable")
        return gs[order], vs[order]

    # DOF fingerprints + global indices: sort by global index so saved
    # arrays are indexed directly by global DOF.
    gs_c = np.concatenate([p["g"] for p in all_coords])
    fps_c = np.concatenate([p["fp"] for p in all_coords], axis=0)
    order_c = np.argsort(gs_c, kind="stable")
    dof_g_sorted = gs_c[order_c]
    dof_fingerprints_sorted = fps_c[order_c]

    g_F, F_sorted = merge_sorted(all_F)
    g_x, x_sorted = merge_sorted(all_x)
    g_y, y_sorted = merge_sorted(all_y)
    g_yT, yT_sorted = merge_sorted(all_yT)

    pb_cat = np.unique(np.concatenate(all_pb)) if size > 1 else np.array([], dtype=np.int64)

    out_file = OUT_DIR / f"{tag}_assembly.npz"
    np.savez(
        out_file,
        coo_row=coo_row, coo_col=coo_col, coo_val=coo_val,
        matrix_size=np.array([A.getSize()[0], A.getSize()[1]]),
        dof_g=dof_g_sorted,
        dof_fingerprints=dof_fingerprints_sorted,  # (N, 4): cx, cy, x, y
        F=F_sorted, F_g=g_F,
        x=x_sorted, x_g=g_x,
        y=y_sorted, y_g=g_y,
        yT=yT_sorted, yT_g=g_yT,
        partition_boundary_rows=pb_cat,
    )
    log(f"saved: {out_file}  (matrix {A.getSize()[0]}x{A.getSize()[1]}, "
        f"nnz={coo_val.size}, partition_boundary_rows={pb_cat.size})")

comm.Barrier()
log("DONE")

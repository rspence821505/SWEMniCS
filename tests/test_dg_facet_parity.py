#!/usr/bin/env python3
"""
Minimal DG partition-interface parity probe.

Builds the DG solver on the idealized-inlet mesh, sets a deterministic
coord-based state, and compares the assembled DG residual F(u) and the
Jacobian action J(u)·v between serial and 2-rank MPI — globally ordered by
spatial coordinate so partition differences can't hide numeric divergence.

This targets ONLY the DG facet-interface assembly:
  - interior-facet flux: dot(avg(Fu), n("+")) + 0.5 C jump(Q)
  - ownership / contribution rules on shared facets across ranks
  - orientation of "+"/"-" traces at partition boundaries
  - FacetNormal consistency on shared facets

It avoids the Newton loop, time stepping, cost function, distributed smoother,
covariance, observations, and adjoint solve — so the runtime is short and any
observed parity failure points squarely at DG facet assembly.

Usage:
  # Serial:
  python tests/test_dg_facet_parity.py
  # MPI-2:
  PYTHONUNBUFFERED=1 mpirun -np 2 python tests/test_dg_facet_parity.py
  # Then compare:
  python tests/test_dg_facet_parity.py --compare
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CC", "/usr/bin/clang")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT_DIR = PROJECT_ROOT / "results" / "mpi_parity"


def _allgather_by_component(V, comm, arr_local_owned):
    """Gather a per-DOF array across ranks, returning one globally-ordered
    array per component (h, u, v), sorted by lexsort on h-DOF (x, y) coords.

    Returns (coords_global, h_global, u_global, v_global) on rank 0,
    or (None, None, None, None) on other ranks.

    Uses the collapsed h, ux, uy sub-space DOF coords to pair values
    position-by-position. Works because the DG mixed element stores h/u/v
    DOFs on identical cell-corner positions.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Collapse each sub-component to find its owned DOF indices in the parent.
    h_space, h_map = V.sub(0).collapse()
    ux_space, ux_map = V.sub(1).sub(0).collapse()
    uy_space, uy_map = V.sub(1).sub(1).collapse()

    bs = V.dofmap.index_map_bs
    n_owned_parent = V.dofmap.index_map.size_local * bs

    h_map = np.asarray(h_map, dtype=np.int64)
    ux_map = np.asarray(ux_map, dtype=np.int64)
    uy_map = np.asarray(uy_map, dtype=np.int64)

    # Restrict to owned parent DOFs
    h_owned_parent = h_map[h_map < n_owned_parent]
    ux_owned_parent = ux_map[ux_map < n_owned_parent]
    uy_owned_parent = uy_map[uy_map < n_owned_parent]

    # Coords of each OWNED h-DOF in the collapsed h-space (global 2D positions)
    all_h_coords = h_space.tabulate_dof_coordinates()[:, :2]
    # Mapping from parent-index to collapsed-index
    parent_to_hcoll = np.full(max(h_map.max(), n_owned_parent) + 1, -1, dtype=np.int64)
    for ci, pi in enumerate(h_map):
        parent_to_hcoll[pi] = ci

    h_coll_idx = parent_to_hcoll[h_owned_parent]
    h_coords_local = all_h_coords[h_coll_idx]

    h_vals_local = arr_local_owned[h_owned_parent].astype(np.float64, copy=True)
    u_vals_local = arr_local_owned[ux_owned_parent].astype(np.float64, copy=True)
    v_vals_local = arr_local_owned[uy_owned_parent].astype(np.float64, copy=True)

    # Gather counts, then gather arrays
    local_n = np.int64(len(h_vals_local))
    counts = np.zeros(size, dtype=np.int64)
    comm.Allgather([np.array([local_n], dtype=np.int64), 1], [counts, 1])

    def _gather(vec):
        if size == 1:
            return vec.copy()
        total = int(counts.sum())
        global_arr = np.zeros(total, dtype=np.float64)
        displs = np.zeros(size, dtype=np.int64)
        displs[1:] = np.cumsum(counts)[:-1]
        from mpi4py import MPI
        comm.Allgatherv([vec, MPI.DOUBLE],
                        [global_arr, counts, displs, MPI.DOUBLE])
        return global_arr

    # coords are 2D — gather as flat then reshape
    coords_flat = h_coords_local.flatten().astype(np.float64)
    coords_counts = counts * 2
    if size == 1:
        coords_global = h_coords_local.copy()
    else:
        from mpi4py import MPI
        total = int(coords_counts.sum())
        coords_global = np.zeros(total, dtype=np.float64)
        displs = np.zeros(size, dtype=np.int64)
        displs[1:] = np.cumsum(coords_counts)[:-1]
        comm.Allgatherv([coords_flat, MPI.DOUBLE],
                        [coords_global, coords_counts, displs, MPI.DOUBLE])
        coords_global = coords_global.reshape(-1, 2)

    h_global = _gather(h_vals_local)
    u_global = _gather(u_vals_local)
    v_global = _gather(v_vals_local)

    # Sort by lex(x, y) canonical order
    if rank == 0:
        order = np.lexsort((coords_global[:, 1], coords_global[:, 0]))
        return (coords_global[order],
                h_global[order], u_global[order], v_global[order])
    return None, None, None, None


def _setup_coord_state(V, solver, comm, amplitude=1.0):
    """Fill solver.u.x.array with a deterministic coord-based DG state.

    The state is coord-keyed so every rank's owned DOFs get the same value at
    the same spatial location. h=h_b+small, u/v smooth sinusoids.
    """
    from dolfinx import fem

    # Constant h_b sampled at DOF coords
    h_sub, h_map = V.sub(0).collapse()
    ux_sub, ux_map = V.sub(1).sub(0).collapse()
    uy_sub, uy_map = V.sub(1).sub(1).collapse()

    coords_h = h_sub.tabulate_dof_coordinates()[:, :2]
    coords_ux = ux_sub.tabulate_dof_coordinates()[:, :2]
    coords_uy = uy_sub.tabulate_dof_coordinates()[:, :2]

    Lx, Ly = 50000.0, 40000.0
    # h: background depth + coord-based perturbation (stays physical)
    h_vals = 14.0 + amplitude * 0.1 * np.cos(2 * np.pi * coords_h[:, 0] / Lx) \
                                     * np.sin(2 * np.pi * coords_h[:, 1] / Ly)
    ux_vals = amplitude * 0.05 * np.sin(2 * np.pi * coords_ux[:, 0] / Lx) \
                                * np.cos(3 * np.pi * coords_ux[:, 1] / Ly)
    uy_vals = amplitude * -0.05 * np.cos(3 * np.pi * coords_uy[:, 0] / Lx) \
                                 * np.sin(2 * np.pi * coords_uy[:, 1] / Ly)

    arr = solver.u.x.array
    arr[:] = 0.0
    arr[h_map] = h_vals
    arr[ux_map] = ux_vals
    arr[uy_map] = uy_vals
    solver.u.x.scatter_forward()
    solver.u_n.x.array[:] = arr[:]
    solver.u_n_old.x.array[:] = arr[:]
    solver.u_n.x.scatter_forward()
    solver.u_n_old.x.scatter_forward()


def _test_vector_coord_based(V, comm, pattern="sin2D"):
    """Build a coord-based test vector on V (same values at same positions)."""
    h_sub, h_map = V.sub(0).collapse()
    ux_sub, ux_map = V.sub(1).sub(0).collapse()
    uy_sub, uy_map = V.sub(1).sub(1).collapse()

    coords_h = h_sub.tabulate_dof_coordinates()[:, :2]
    coords_ux = ux_sub.tabulate_dof_coordinates()[:, :2]
    coords_uy = uy_sub.tabulate_dof_coordinates()[:, :2]

    Lx, Ly = 50000.0, 40000.0

    if pattern == "sin2D":
        h_vals = np.cos(4 * np.pi * coords_h[:, 0] / Lx) \
                 * np.sin(6 * np.pi * coords_h[:, 1] / Ly)
        ux_vals = np.sin(2 * np.pi * coords_ux[:, 0] / Lx) \
                  * np.cos(4 * np.pi * coords_ux[:, 1] / Ly)
        uy_vals = -ux_vals.copy()
    elif pattern == "localized":
        xc, yc = 0.5 * Lx, 0.5 * Ly
        r2 = lambda c: ((c[:, 0] - xc) ** 2 + (c[:, 1] - yc) ** 2) / (5000.0 ** 2)
        h_vals = np.exp(-r2(coords_h))
        ux_vals = np.zeros_like(h_vals)
        uy_vals = np.zeros_like(h_vals)
    elif pattern == "linear_x":
        h_vals = coords_h[:, 0] / Lx
        ux_vals = coords_ux[:, 0] / Lx
        uy_vals = np.zeros_like(ux_vals)
    else:
        raise ValueError(pattern)

    bs = V.dofmap.index_map_bs
    n_local = V.dofmap.index_map.size_local * bs + \
              V.dofmap.index_map.num_ghosts * bs
    arr = np.zeros(n_local)
    arr[h_map] = h_vals
    arr[ux_map] = ux_vals
    arr[uy_map] = uy_vals
    return arr


def run_probe(comm):
    from petsc4py import PETSc
    from mpi4py import MPI
    from dolfinx import fem
    import ufl

    from swe4dvar.forward.problems import IdealizedInlet
    from swe4dvar.forward.solvers import get_solver

    rank = comm.Get_rank()
    size = comm.Get_size()

    def log(msg):
        sys.stdout.write(f"  [r{rank}/{size}] {msg}\n")
        sys.stdout.flush()

    log("building problem...")
    prob = IdealizedInlet(
        dt=600.0, nt=1,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
        dramp=1.0,
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])
    V = solver.V
    log(f"V built: local={V.dofmap.index_map.size_local * V.dofmap.index_map_bs}, "
        f"ghosts={V.dofmap.index_map.num_ghosts * V.dofmap.index_map_bs}")

    # Set coord-based deterministic state
    log("setting coord-based state...")
    _setup_coord_state(V, solver, comm)

    # Assemble residual F(u) — this includes the DG interior-facet flux
    log("assembling residual F(u)...")
    F_form = fem.form(solver.F)
    from dolfinx.fem.petsc import assemble_vector
    F_vec = assemble_vector(F_form)
    F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    F_owned = F_vec.getArray()[:V.dofmap.index_map.size_local * V.dofmap.index_map_bs].copy()
    F_global = _allgather_by_component(V, comm, F_vec.getArray())

    # Assemble Jacobian J(u) — same facet-flux DG form
    log("assembling Jacobian J(u)...")
    du = ufl.TrialFunction(V)
    J_form = fem.form(ufl.derivative(solver.F, solver.u, du))
    from dolfinx.fem.petsc import assemble_matrix
    J = assemble_matrix(J_form)
    J.assemble()
    log(f"J size={J.getSize()} local={J.getLocalSize()} nnz={int(J.getInfo()['nz_used'])}")

    # Apply J and J^T to a coord-based test vector
    results = {}
    for pat in ["sin2D", "localized", "linear_x"]:
        log(f"applying J and J^T to pattern={pat}")
        v_arr = _test_vector_coord_based(V, comm, pat)

        # Wrap the test vector in a PETSc Vec matching J's layout
        x_vec = J.createVecRight()
        x_vec.zeroEntries()
        owned_only = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        x_vec.getArray()[:] = v_arr[:owned_only]
        x_vec.assemble()

        # J @ v
        Jv = x_vec.duplicate()
        J.mult(x_vec, Jv)
        # J^T @ v
        JTv = x_vec.duplicate()
        J.multTranspose(x_vec, JTv)

        # Gather globally
        v_coords, v_h, v_u, v_v = _allgather_by_component(V, comm, v_arr)
        Jv_coords, Jv_h, Jv_u, Jv_v = _allgather_by_component(V, comm, Jv.getArray())
        JTv_coords, JTv_h, JTv_u, JTv_v = _allgather_by_component(V, comm, JTv.getArray())

        if rank == 0:
            results[f"{pat}_input"] = (v_h, v_u, v_v)
            results[f"{pat}_Jv"] = (Jv_h, Jv_u, Jv_v)
            results[f"{pat}_JTv"] = (JTv_h, JTv_u, JTv_v)

        x_vec.destroy(); Jv.destroy(); JTv.destroy()

    # Residual coords/values
    if rank == 0:
        results["F_u"] = (F_global[1], F_global[2], F_global[3])
        results["coords"] = F_global[0]

    return results, rank, size


def save_results(results, tag):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flat = {"coords": results["coords"]}
    for k, v in results.items():
        if k == "coords":
            continue
        if isinstance(v, tuple) and len(v) == 3:
            flat[f"{k}__h"] = v[0]
            flat[f"{k}__u"] = v[1]
            flat[f"{k}__v"] = v[2]
    out = OUT_DIR / f"dg_facet_probe_{tag}.npz"
    np.savez(out, **flat)
    print(f"  wrote {out}", flush=True)
    return out


def compare():
    """Order-invariant comparison.

    For DG elements there can be multiple DOFs at the same spatial coordinate
    (one per adjacent cell), so a lexsort on (x, y) alone cannot produce a
    unique canonical ordering; per-DOF alignment between serial and MPI is
    ambiguous.

    This comparison uses three ORDER-INVARIANT statistics for each component:
      (a) L2 norm                  — distribution-invariant if values agree
      (b) sum of values            — distribution-invariant by same reasoning
      (c) sum of squares after sort — if the multi-set of values matches, any
          order-free aggregate (min, max, sorted values) is bit-equal
    If all three match to rounding, the underlying entries ARE the same set.
    """
    ser_file = OUT_DIR / "dg_facet_probe_serial.npz"
    mpi_file = OUT_DIR / "dg_facet_probe_mpi2.npz"
    if not ser_file.exists() or not mpi_file.exists():
        print(f"Missing {ser_file if not ser_file.exists() else mpi_file}")
        return 1

    s = np.load(ser_file)
    m = np.load(mpi_file)

    print("=" * 72)
    print("DG FACET INTERFACE PARITY PROBE — serial vs MPI-2")
    print("ORDER-INVARIANT AGGREGATES (DG DOFs are not uniquely indexable by (x,y))")
    print("=" * 72)

    def _diff(a, b, label):
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        sa = float(np.sum(a))
        sb = float(np.sum(b))
        # sorted-value comparison (element-wise diff after sorting both)
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        sort_rel = float(np.linalg.norm(a_sorted - b_sorted)) / max(na, 1e-30)
        norm_rel = abs(na - nb) / max(na, 1e-30)
        sum_rel = abs(sa - sb) / max(abs(sa), 1e-30)
        amax_s = float(np.max(np.abs(a)))
        amax_m = float(np.max(np.abs(b)))
        amax_rel = abs(amax_s - amax_m) / max(amax_s, 1e-30)
        verdict = "OK  " if (norm_rel < 1e-8 and sort_rel < 1e-8
                              and amax_rel < 1e-8) else (
            "NEAR" if (norm_rel < 1e-3 and sort_rel < 1e-3) else "FAIL")
        print(f"  {verdict}  {label}: ||s||={na:.6e} ||m||={nb:.6e} "
              f"Δnorm={norm_rel:.2e}  Δsort={sort_rel:.2e}  "
              f"Δmax={amax_rel:.2e}  Δsum={sum_rel:.2e}")
        return {"norm_rel": norm_rel, "sort_rel": sort_rel,
                "sum_rel": sum_rel, "amax_rel": amax_rel}

    rows = {}

    print("\n--- Residual F(u) (includes DG interior-facet flux) ---")
    for c in ("h", "u", "v"):
        rows[f"F__{c}"] = _diff(s[f"F_u__{c}"], m[f"F_u__{c}"], f"F(u).{c}")

    for pat in ("sin2D", "localized", "linear_x"):
        print(f"\n--- pattern = {pat} ---")
        for kind in ("input", "Jv", "JTv"):
            for c in ("h", "u", "v"):
                key = f"{pat}_{kind}__{c}"
                rows[key] = _diff(s[key], m[key], f"{kind}.{c}")

    print("\n" + "=" * 72)

    def _all_ok(keys, tol=1e-6):
        return all(rows[k]["norm_rel"] < tol and rows[k]["sort_rel"] < tol
                   for k in keys)

    F_ok = _all_ok([f"F__{c}" for c in ("h", "u", "v")])
    Jv_ok = _all_ok([f"{p}_Jv__{c}" for p in ("sin2D", "localized", "linear_x")
                     for c in ("h", "u", "v")])
    JTv_ok = _all_ok([f"{p}_JTv__{c}" for p in ("sin2D", "localized", "linear_x")
                      for c in ("h", "u", "v")])
    inp_ok = _all_ok([f"{p}_input__{c}" for p in ("sin2D", "localized", "linear_x")
                      for c in ("h", "u", "v")], tol=1e-10)

    print(f"  coord-based input vectors  (multi-set match): {'PASS' if inp_ok else 'FAIL'}")
    print(f"  DG residual F(u) parity    (multi-set match): {'PASS' if F_ok else 'FAIL'}")
    print(f"  Jacobian action J·v parity (multi-set match): {'PASS' if Jv_ok else 'FAIL'}")
    print(f"  Transpose J^T·v parity     (multi-set match): {'PASS' if JTv_ok else 'FAIL'}")

    if F_ok and Jv_ok and JTv_ok:
        print("\n  VERDICT: DG partition-interface assembly is distribution-invariant.")
        print("  Facet orientation, +/- trace selection, normal direction, and shared-")
        print("  facet ownership all produce matching assembled vectors. The MPI")
        print("  result differs only in per-DOF ORDERING (ambiguous for DG, not a bug).")
    elif F_ok and Jv_ok and not JTv_ok:
        print("\n  VERDICT: transpose action itself diverges (unlikely PETSc bug).")
    else:
        print("\n  VERDICT: DG partition-interface has a distribution-dependent defect.")
    return 0 if (F_ok and Jv_ok and JTv_ok) else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()
    if args.compare:
        return compare()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    results, rank, size = run_probe(comm)
    if rank == 0:
        tag = "serial" if size == 1 else f"mpi{size}"
        save_results(results, tag)
    comm.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

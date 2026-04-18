"""
Serial-vs-MPI object diff tooling.

Reusable export helpers for comparing assembled mathematical objects
(state vectors, residuals, adjoint RHS, Jacobian actions, selected matrix
rows) between a serial run and an MPI run on the SAME nominal problem.

All exports use a *canonical* global ordering keyed by spatial coordinate,
so that artifacts from different communicator sizes / partitions are
directly comparable byte-for-byte (modulo numerical noise) without
permutation alignment.

Design assumptions (verified for DG p1+p1 on the idealized inlet):
  * V is a mixed function space with sub(0) = scalar h and
    sub(1) = vector (u, v).
  * h, u, v DOFs are collocated and per-rank counts agree:
        len(h_owned) == len(u_owned) == len(v_owned).
  * The PETSc layout of vectors/matrices on V matches
    V.dofmap.index_map (block_size = V.dofmap.index_map_bs).

Artifacts are written as numpy .npz files under a per-run directory.
The companion script tests/mpi_diff_compare.py reads them and computes
per-component metrics, coordinate-localized differences, and partition
interface-distance summaries.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


def _get_component_dof_indices(V, owned_only: bool):
    """V-local DOF indices for h, u, v components.

    Mirrors swe4dvar.utils.distributed_smoother._get_component_dof_indices
    (kept separate so this module has no smoother dependency).
    """
    h_dofs = V.sub(0).dofmap.list.flatten()
    uv_dofs = V.sub(1).dofmap.list.flatten()
    h_indices = np.unique(h_dofs)
    uv_indices = np.unique(uv_dofs)
    u_indices = uv_indices[0::2]
    v_indices = uv_indices[1::2]
    if owned_only:
        owned_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        h_indices = h_indices[h_indices < owned_size]
        u_indices = u_indices[u_indices < owned_size]
        v_indices = v_indices[v_indices < owned_size]
    return h_indices, u_indices, v_indices


class MPIObjectExporter:
    """Export PETSc Vec/Mat objects in a globally coordinate-sorted form.

    On construction, performs the one-time work of building:
      * per-rank owned (h, u, v) V-DOF indices,
      * coordinates of each owned DOF (collocated h/u/v assumption),
      * global canonical ordering (lexsort on (x, y) within each component),
      * global PETSc DOF index for every owned DOF,
      * partition interface distance per owned h-DOF (distance to the
        nearest h-DOF owned by ANOTHER rank; +inf in serial),
      * a global "PETSc-row-index → (component, x, y, owner_rank)" table,
        cached on every rank for matrix-row interpretation.

    Per-export work is then cheap: one allgather per component vector,
    one mat-vec for Jacobian actions, one row extraction for matrix rows.

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        Mixed h+(u,v) function space matching the PETSc Vec/Mat layout.
    comm : MPI.Comm, optional
        Communicator. Defaults to MPI.COMM_WORLD.
    """

    def __init__(self, V, comm=None):
        from mpi4py import MPI
        self._MPI = MPI

        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.V = V

        idx_map = V.dofmap.index_map
        self.bs = V.dofmap.index_map_bs
        self.n_owned_blocks = idx_map.size_local
        self.n_owned = self.n_owned_blocks * self.bs

        self.h_owned, self.u_owned, self.v_owned = _get_component_dof_indices(V, owned_only=True)

        if not (len(self.h_owned) == len(self.u_owned) == len(self.v_owned)):
            raise RuntimeError(
                f"MPIObjectExporter requires collocated h/u/v components, but got "
                f"|h|={len(self.h_owned)}, |u|={len(self.u_owned)}, |v|={len(self.v_owned)}. "
                "This tooling targets DG p1+p1 mixed elements."
            )

        h_sub = V.sub(0)
        h_space, h_map = h_sub.collapse()
        h_map_arr = np.asarray(h_map, dtype=np.int64)
        v_to_hspace = {int(v): i for i, v in enumerate(h_map_arr)}
        all_h_coords = h_space.tabulate_dof_coordinates()[:, :2]

        self.h_coords_local = np.zeros((len(self.h_owned), 2))
        for i, vdof in enumerate(self.h_owned):
            self.h_coords_local[i] = all_h_coords[v_to_hspace[int(vdof)]]

        local_n = np.array([len(self.h_owned)], dtype=np.int64)
        all_sizes = np.zeros(self.size, dtype=np.int64)
        self.comm.Allgather(local_n, all_sizes)
        self.local_sizes = all_sizes
        self.local_offsets = np.concatenate([[0], np.cumsum(all_sizes)[:-1]])
        self.global_n = int(all_sizes.sum())
        self.local_size = int(all_sizes[self.rank])
        self.local_offset = int(self.local_offsets[self.rank])

        self.global_coords = self._allgather_coords(self.h_coords_local)
        self.canonical_order = np.lexsort((self.global_coords[:, 1], self.global_coords[:, 0]))
        self.canonical_coords = self.global_coords[self.canonical_order]

        self.owner_per_global = np.zeros(self.global_n, dtype=np.int32)
        for r in range(self.size):
            s0 = int(self.local_offsets[r])
            s1 = s0 + int(self.local_sizes[r])
            self.owner_per_global[s0:s1] = r

        self.interface_distance = self._compute_interface_distance()

        self._build_global_petsc_dof_table(idx_map)

    # ------------------------------------------------------------------
    # internal: allgather + interface-distance + global PETSc DOF table
    # ------------------------------------------------------------------
    def _allgather_coords(self, local_coords):
        MPI = self._MPI
        sendbuf = np.ascontiguousarray(local_coords, dtype=np.float64).ravel()
        local_count = int(len(local_coords) * 2)
        all_counts = (self.local_sizes * 2).astype(np.int64)
        all_displs = np.concatenate([[0], np.cumsum(all_counts)[:-1]])
        recvbuf = np.zeros(int(all_counts.sum()), dtype=np.float64)
        self.comm.Allgatherv(
            [sendbuf, local_count, MPI.DOUBLE],
            [recvbuf, all_counts, all_displs, MPI.DOUBLE],
        )
        return recvbuf.reshape(-1, 2)

    def _allgather_component(self, local_vals):
        MPI = self._MPI
        sendbuf = np.ascontiguousarray(local_vals, dtype=np.float64)
        all_counts = self.local_sizes.astype(np.int64)
        all_displs = np.concatenate([[0], np.cumsum(all_counts)[:-1]])
        recvbuf = np.zeros(int(all_counts.sum()), dtype=np.float64)
        self.comm.Allgatherv(
            [sendbuf, int(all_counts[self.rank]), MPI.DOUBLE],
            [recvbuf, all_counts, all_displs, MPI.DOUBLE],
        )
        return recvbuf

    def _compute_interface_distance(self):
        """For each owned h-DOF, distance to nearest h-DOF on a DIFFERENT rank.

        +inf for all DOFs in serial.
        """
        if self.size == 1:
            return np.full(self.local_size, np.inf)
        s0, s1 = self.local_offset, self.local_offset + self.local_size
        my_coords = self.global_coords[s0:s1]
        other_coords = np.vstack([self.global_coords[:s0], self.global_coords[s1:]])
        if len(other_coords) == 0:
            return np.full(self.local_size, np.inf)
        tree = cKDTree(other_coords)
        dists, _ = tree.query(my_coords, k=1)
        return dists

    def _build_global_petsc_dof_table(self, idx_map):
        """Build a (global_n_owned * bs)-length table:
            global_dof_table[g_petsc_idx] = (component_id, x, y, owner_rank)

        component_id: 0 for h, 1 for u, 2 for v.

        Required by export_matrix_rows so each rank can interpret column
        indices coming back from PETSc Mat.getRow() as physical (component,
        x, y) tuples.
        """
        MPI = self._MPI
        bs = self.bs
        n_local = self.n_owned

        local_block = np.arange(self.n_owned_blocks, dtype=np.int32)
        block_global = idx_map.local_to_global(local_block).astype(np.int64)
        local_g_idx = (block_global[:, None] * bs + np.arange(bs)[None, :]).ravel()

        comp_local = np.full(n_local, -1, dtype=np.int8)
        x_local = np.zeros(n_local, dtype=np.float64)
        y_local = np.zeros(n_local, dtype=np.float64)
        for i, vdof in enumerate(self.h_owned):
            comp_local[int(vdof)] = 0
            x_local[int(vdof)] = self.h_coords_local[i, 0]
            y_local[int(vdof)] = self.h_coords_local[i, 1]
        for i, vdof in enumerate(self.u_owned):
            comp_local[int(vdof)] = 1
            x_local[int(vdof)] = self.h_coords_local[i, 0]
            y_local[int(vdof)] = self.h_coords_local[i, 1]
        for i, vdof in enumerate(self.v_owned):
            comp_local[int(vdof)] = 2
            x_local[int(vdof)] = self.h_coords_local[i, 0]
            y_local[int(vdof)] = self.h_coords_local[i, 1]

        owner_local = np.full(n_local, self.rank, dtype=np.int32)

        local_count = np.array([n_local], dtype=np.int64)
        all_counts = np.zeros(self.size, dtype=np.int64)
        self.comm.Allgather(local_count, all_counts)
        all_displs = np.concatenate([[0], np.cumsum(all_counts)[:-1]])
        total = int(all_counts.sum())

        def _gather_int8(arr):
            buf = np.zeros(total, dtype=np.int8)
            self.comm.Allgatherv(
                [np.ascontiguousarray(arr, dtype=np.int8), int(all_counts[self.rank]), MPI.BYTE],
                [buf, all_counts, all_displs, MPI.BYTE],
            )
            return buf

        def _gather_int32(arr):
            buf = np.zeros(total, dtype=np.int32)
            self.comm.Allgatherv(
                [np.ascontiguousarray(arr, dtype=np.int32), int(all_counts[self.rank]), MPI.INT],
                [buf, all_counts, all_displs, MPI.INT],
            )
            return buf

        def _gather_int64(arr):
            buf = np.zeros(total, dtype=np.int64)
            self.comm.Allgatherv(
                [np.ascontiguousarray(arr, dtype=np.int64), int(all_counts[self.rank]), MPI.LONG_LONG],
                [buf, all_counts, all_displs, MPI.LONG_LONG],
            )
            return buf

        def _gather_f64(arr):
            buf = np.zeros(total, dtype=np.float64)
            self.comm.Allgatherv(
                [np.ascontiguousarray(arr, dtype=np.float64), int(all_counts[self.rank]), MPI.DOUBLE],
                [buf, all_counts, all_displs, MPI.DOUBLE],
            )
            return buf

        all_g_idx = _gather_int64(local_g_idx)
        all_comp = _gather_int8(comp_local)
        all_owner = _gather_int32(owner_local)
        all_x = _gather_f64(x_local)
        all_y = _gather_f64(y_local)

        global_dof_count = idx_map.size_global * bs
        self.global_dof_count = int(global_dof_count)
        self.global_dof_component = np.full(global_dof_count, -1, dtype=np.int8)
        self.global_dof_owner = np.full(global_dof_count, -1, dtype=np.int32)
        self.global_dof_x = np.zeros(global_dof_count, dtype=np.float64)
        self.global_dof_y = np.zeros(global_dof_count, dtype=np.float64)
        self.global_dof_component[all_g_idx] = all_comp
        self.global_dof_owner[all_g_idx] = all_owner
        self.global_dof_x[all_g_idx] = all_x
        self.global_dof_y[all_g_idx] = all_y

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def vector_to_canonical(self, vec) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Return (coords, h, u, v) sorted by (x, y) on rank 0; (None,)*4 elsewhere.

        Input: PETSc.Vec laid out on V (rank-local, size n_owned).
        """
        owned_arr = vec.getArray()
        h_local = owned_arr[self.h_owned]
        u_local = owned_arr[self.u_owned]
        v_local = owned_arr[self.v_owned]
        h_g = self._allgather_component(h_local)
        u_g = self._allgather_component(u_local)
        v_g = self._allgather_component(v_local)
        order = self.canonical_order
        return self.canonical_coords, h_g[order], u_g[order], v_g[order]

    def export_vector(self, vec, name: str, out_dir: Path, tag: str) -> Optional[Path]:
        """Save a PETSc Vec as canonical (coords, h, u, v).npz.

        Returns the written path on rank 0; None on other ranks.
        """
        coords, h, u, v = self.vector_to_canonical(vec)
        if self.rank != 0:
            return None
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}__{tag}.npz"
        np.savez(
            path,
            coords=coords, h=h, u=u, v=v,
            interface_distance=self._gather_interface_distance(),
            mpi_size=self.size,
        )
        return path

    def _gather_interface_distance(self):
        """Gather per-h-DOF interface distance into canonical order on rank 0."""
        global_iface = self._allgather_component(self.interface_distance)
        return global_iface[self.canonical_order]

    def export_matrix_action(
        self,
        mat,
        test_vec,
        name: str,
        out_dir: Path,
        tag: str,
        transpose: bool = False,
    ) -> Optional[Path]:
        """Apply mat (or mat^T) to test_vec and export the result canonically.

        Both the input and the output are saved together so the test vector
        can be verified identical across runs.
        """
        result = test_vec.duplicate()
        if transpose:
            mat.multTranspose(test_vec, result)
        else:
            mat.mult(test_vec, result)
        in_coords, in_h, in_u, in_v = self.vector_to_canonical(test_vec)
        out_coords, out_h, out_u, out_v = self.vector_to_canonical(result)
        result.destroy()
        if self.rank != 0:
            return None
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}__{tag}.npz"
        np.savez(
            path,
            coords=in_coords,
            input_h=in_h, input_u=in_u, input_v=in_v,
            output_h=out_h, output_u=out_u, output_v=out_v,
            interface_distance=self._gather_interface_distance(),
            transpose=int(transpose),
            mpi_size=self.size,
        )
        return path

    def make_coord_vector(self, pattern: str, amp_h: float = 1.0, amp_uv: float = 0.3):
        """Build a deterministic PETSc Vec on V whose values are functions of
        spatial coordinates only — identical between serial and MPI by
        construction.

        Patterns:
          * "ones"      : 1.0 everywhere (h = 1, u = +amp_uv, v = -amp_uv)
          * "sin2D"     : smooth tensor-product mode
          * "localized" : Gaussian spike near domain center
        """
        from petsc4py import PETSc  # noqa: F401  (ensures init)
        from dolfinx import la

        Lx, Ly = 50000.0, 40000.0
        x = self.h_coords_local[:, 0] / Lx
        y = self.h_coords_local[:, 1] / Ly

        if pattern == "ones":
            vals_h = np.ones_like(x)
            vals_uv = np.ones_like(x)
        elif pattern == "sin2D":
            vals_h = np.cos(4 * np.pi * x) * np.sin(6 * np.pi * y)
            vals_uv = np.sin(2 * np.pi * x) * np.cos(4 * np.pi * y)
        elif pattern == "localized":
            vals_h = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.01)
            vals_uv = np.zeros_like(vals_h)
        else:
            raise ValueError(f"Unknown pattern: {pattern!r}")

        vec = la.create_petsc_vector(self.V.dofmap.index_map, self.V.dofmap.index_map_bs)
        arr = np.zeros(self.n_owned)
        arr[self.h_owned] = vals_h * amp_h
        arr[self.u_owned] = vals_uv * amp_uv
        arr[self.v_owned] = vals_uv * (-amp_uv)
        vec.setArray(arr)
        vec.assemble()
        return vec

    def export_matrix_rows(
        self,
        mat,
        suspect_coords: Sequence[Tuple[float, float]],
        component: str,
        name: str,
        out_dir: Path,
        tag: str,
        max_cols_per_row: int = 4096,
    ) -> Optional[Path]:
        """Extract rows of `mat` near `suspect_coords` and save with column
        coordinates resolved.

        For each (xq, yq), finds the nearest owned DOF of the requested
        component (h/u/v). The owning rank extracts the row from `mat` via
        Mat.getRow(global_row) and translates each global column index to
        (component_id, x, y) using the cached global DOF table.

        Output (rank 0) — list of dicts (one per suspect coord), each with:
            row_coord:  (x_row, y_row)
            row_component: 'h'|'u'|'v'
            row_owner: int rank
            row_global_idx: int
            col_global_idx: (n_nnz,) int64
            col_component: (n_nnz,) int8
            col_x: (n_nnz,) float64
            col_y: (n_nnz,) float64
            col_value: (n_nnz,) float64
        """
        comp_to_id = {"h": 0, "u": 1, "v": 2}
        comp_id_query = comp_to_id[component]
        owned_local = {"h": self.h_owned, "u": self.u_owned, "v": self.v_owned}[component]
        my_coords = self.h_coords_local

        idx_map = self.V.dofmap.index_map
        bs = self.bs
        local_block = np.arange(self.n_owned_blocks, dtype=np.int32)
        block_global = idx_map.local_to_global(local_block).astype(np.int64)

        rows_out: List[dict] = []

        for (xq, yq) in suspect_coords:
            d2 = (my_coords[:, 0] - xq) ** 2 + (my_coords[:, 1] - yq) ** 2
            local_min = float(np.min(d2)) if len(d2) > 0 else float("inf")
            local_argmin = int(np.argmin(d2)) if len(d2) > 0 else -1

            all_mins = self.comm.allgather(local_min)
            owner = int(np.argmin(all_mins))
            global_min = all_mins[owner]

            if self.rank == owner and local_argmin >= 0:
                v_dof = int(owned_local[local_argmin])
                block = v_dof // bs
                offset = v_dof % bs
                g_row = int(block_global[block]) * bs + offset

                cols_g, vals = mat.getRow(g_row)
                cols_g = np.asarray(cols_g, dtype=np.int64)
                vals = np.asarray(vals, dtype=np.float64)
                if len(cols_g) > max_cols_per_row:
                    order = np.argsort(-np.abs(vals))[:max_cols_per_row]
                    cols_g = cols_g[order]
                    vals = vals[order]

                in_table = cols_g < self.global_dof_count
                col_comp = np.full(len(cols_g), -1, dtype=np.int8)
                col_x = np.zeros(len(cols_g), dtype=np.float64)
                col_y = np.zeros(len(cols_g), dtype=np.float64)
                col_comp[in_table] = self.global_dof_component[cols_g[in_table]]
                col_x[in_table] = self.global_dof_x[cols_g[in_table]]
                col_y[in_table] = self.global_dof_y[cols_g[in_table]]

                payload = {
                    "row_coord": (float(my_coords[local_argmin, 0]), float(my_coords[local_argmin, 1])),
                    "row_component": component,
                    "row_owner": int(owner),
                    "row_global_idx": int(g_row),
                    "row_distance_to_query": float(np.sqrt(global_min)),
                    "col_global_idx": cols_g,
                    "col_component": col_comp,
                    "col_x": col_x,
                    "col_y": col_y,
                    "col_value": vals,
                }
            else:
                payload = None

            payloads = self.comm.gather(payload, root=0)
            if self.rank == 0:
                for p in payloads:
                    if p is not None:
                        rows_out.append(p)
                        break

        if self.rank != 0:
            return None
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}__{tag}.npz"
        flat = {"n_rows": np.array(len(rows_out), dtype=np.int64), "mpi_size": np.array(self.size, dtype=np.int32)}
        for i, r in enumerate(rows_out):
            flat[f"row_{i}__row_coord"] = np.asarray(r["row_coord"], dtype=np.float64)
            flat[f"row_{i}__row_component"] = np.array(r["row_component"], dtype="<U1")
            flat[f"row_{i}__row_owner"] = np.array(r["row_owner"], dtype=np.int32)
            flat[f"row_{i}__row_global_idx"] = np.array(r["row_global_idx"], dtype=np.int64)
            flat[f"row_{i}__row_distance_to_query"] = np.array(r["row_distance_to_query"], dtype=np.float64)
            flat[f"row_{i}__col_global_idx"] = r["col_global_idx"]
            flat[f"row_{i}__col_component"] = r["col_component"]
            flat[f"row_{i}__col_x"] = r["col_x"]
            flat[f"row_{i}__col_y"] = r["col_y"]
            flat[f"row_{i}__col_value"] = r["col_value"]
        np.savez(path, **flat)
        return path

    def export_dof_metadata(self, out_dir: Path, tag: str) -> Optional[Path]:
        """Save canonical DOF metadata: coords, owner-per-DOF (in canonical
        order), interface distance per h-DOF, and basic config.

        The owner field reflects THIS run's partitioning. In serial all
        owners are 0; in MPI N>1 it shows which rank owned which canonical
        position.
        """
        if self.rank != 0:
            return None
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"dof_metadata__{tag}.npz"
        owner_canonical = self.owner_per_global[self.canonical_order]
        iface_canonical = self._gather_interface_distance() if self.size > 1 else \
            np.full(self.global_n, np.inf)
        np.savez(
            path,
            coords=self.canonical_coords,
            owner=owner_canonical,
            interface_distance=iface_canonical,
            mpi_size=self.size,
            global_n_h=self.global_n,
            bs=self.bs,
        )
        return path

    # convenience: human-readable summary written to stdout on rank 0
    def summarize(self):
        if self.rank != 0:
            return
        n_iface = int(np.sum(np.isfinite(self.interface_distance)))
        med = float(np.median(self.interface_distance[np.isfinite(self.interface_distance)])) \
            if n_iface > 0 else float("nan")
        print(
            f"[MPIObjectExporter] size={self.size} rank0_local_h={self.local_size} "
            f"global_h={self.global_n} bs={self.bs} "
            f"interface_dist median (rank0)={med:.1f} m",
            flush=True,
        )

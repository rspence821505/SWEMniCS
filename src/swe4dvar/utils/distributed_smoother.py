"""
Distributed Gaussian gradient smoother for MPI-parallel DA.

Rationale
---------
A local-only smoother truncates neighborhoods at partition boundaries,
producing O(1) gradient-norm differences between serial and MPI
(~58% at L=500m on the idealized inlet). DOLFINx's default ghost
halo is ~1 cell wide — far too narrow for a smoother kernel with
3L ~ 1500m radius. On the 69K-DOF idealized inlet, 25.8% of owned
DOFs are within 3L of a partition boundary and miss ~500 neighbors
each under the default halo.

Fix
---
Since the global gradient is small (~560 KB for 70K DOFs in float64),
we allgather the gradient to every rank, apply the full-size smoother
locally on each rank, then extract owned values. This is bit-exact
with serial by construction.

Memory: one n_global x n_global sparse smoother, same as serial.
Time: one MPI_Allgather per apply (negligible for this problem size).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse


def _get_component_dof_indices(V, owned_only: bool):
    """Same logic as TwinExperiment._get_component_dof_indices."""
    h_dofs = V.sub(0).dofmap.list.flatten()
    uv_dofs = V.sub(1).dofmap.list.flatten()
    h_indices = np.unique(h_dofs)
    uv_indices = np.unique(uv_dofs)
    u_indices = uv_indices[0::2]
    v_indices = uv_indices[1::2]
    if owned_only:
        u_owned_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        h_indices = h_indices[h_indices < u_owned_size]
        u_indices = u_indices[u_indices < u_owned_size]
        v_indices = v_indices[v_indices < u_owned_size]
    return h_indices, u_indices, v_indices


class DistributedGradientSmoother:
    """Allgather-based Gaussian gradient smoother — bit-exact with serial.

    Each rank gathers the full global h, u, v gradient vectors via
    MPI allgather, applies the full-size smoother locally, and extracts
    its owned DOF values.

    Parameters
    ----------
    solver : object with attribute V (FEniCSx MixedFunctionSpace)
    correlation_length : float
        Gaussian kernel length scale (meters).
    comm : MPI.Comm, optional
        MPI communicator.
    """

    def __init__(self, solver, correlation_length: float, comm=None):
        from mpi4py import MPI
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.solver = solver
        self.V = solver.V
        self.L = correlation_length

        self.n_owned = self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs

        # OWNED component V-DOF indices on this rank
        self.h_owned, self.u_owned, self.v_owned = _get_component_dof_indices(self.V, owned_only=True)

        # Build h-space coordinate lookup paired with h_owned.
        # h_owned[i] is a V-DOF; we need the spatial coord at that DOF.
        # Use h_map (from h_sub.collapse()) to go V-DOF -> h-space index -> coord.
        h_sub = self.V.sub(0)
        h_space, h_map = h_sub.collapse()
        h_map_arr = np.asarray(h_map, dtype=np.int64)
        v_to_hspace = {int(v): i for i, v in enumerate(h_map_arr)}
        all_h_coords = h_space.tabulate_dof_coordinates()[:, :2]

        h_coords_local = np.zeros((len(self.h_owned), 2))
        for i, v_dof in enumerate(self.h_owned):
            h_coords_local[i] = all_h_coords[v_to_hspace[int(v_dof)]]

        # Gather per-rank sizes
        local_n = np.array([len(self.h_owned)], dtype=np.int64)
        all_sizes = np.zeros(self.size, dtype=np.int64)
        self.comm.Allgather(local_n, all_sizes)

        self.global_n_h = int(all_sizes.sum())
        self.local_offsets = np.concatenate([[0], np.cumsum(all_sizes)[:-1]])
        self.local_size = int(all_sizes[self.rank])
        self.local_offset = int(self.local_offsets[self.rank])

        # Allgather coords so every rank has the same global h-coord array,
        # with coords paired 1-to-1 with the allgathered gradient values.
        self.global_h_coords = self._allgather_coords(h_coords_local, all_sizes)

        # Build THIS RANK's rows of the smoother matrix only:
        # shape (local_size, global_n_h). Saves ~(size-1)/size of the memory
        # compared to building the full n_global x n_global matrix on every rank.
        self.G = self._build_owned_rows(self.global_h_coords)

    def _allgather_coords(self, local_coords, all_sizes):
        """Allgather per-rank local coordinate arrays (N_i, 2) -> (N_global, 2)."""
        # Flatten local coords to 1D (2*N_i), allgatherv, reshape.
        sendbuf = np.ascontiguousarray(local_coords, dtype=np.float64).ravel()
        local_count = len(local_coords) * 2
        all_counts = np.asarray(all_sizes * 2, dtype=np.int64)
        all_displs = np.concatenate([[0], np.cumsum(all_counts)[:-1]])

        recvbuf = np.zeros(int(all_counts.sum()), dtype=np.float64)
        from mpi4py import MPI
        self.comm.Allgatherv(
            [sendbuf, local_count, MPI.DOUBLE],
            [recvbuf, all_counts, all_displs, MPI.DOUBLE],
        )
        return recvbuf.reshape(-1, 2)

    def _build_owned_rows(self, coords):
        """Build this rank's OWNED rows of the Gaussian smoother.

        Shape: (local_size, global_n). Each row i corresponds to this
        rank's i-th owned h-DOF (coord at coords[local_offset + i]).
        Columns span all global DOFs. This gives us exact serial/MPI
        equivalence while avoiding the memory cost of a replicated
        full-size matrix.
        """
        n_global = len(coords)
        s0 = self.local_offset
        s1 = s0 + self.local_size
        my_coords = coords[s0:s1]

        radius = 3.0 * self.L
        two_L_sq = 2.0 * self.L ** 2

        tree = cKDTree(coords)
        rows, cols, data = [], [], []
        for i in range(self.local_size):
            neighbors = tree.query_ball_point(my_coords[i], radius)
            if len(neighbors) == 0:
                neighbors = [s0 + i]
            neighbors = np.asarray(neighbors)
            dists_sq = np.sum((coords[neighbors] - my_coords[i]) ** 2, axis=1)
            weights = np.exp(-dists_sq / two_L_sq)
            rows.extend([i] * len(neighbors))
            cols.extend(neighbors.tolist())
            data.extend(weights.tolist())

        G = sparse.csr_matrix((data, (rows, cols)), shape=(self.local_size, n_global))

        # Row normalization (same formula; each row normalized by its own sum of squares)
        G_sq = G.multiply(G)
        row_norms = np.sqrt(np.asarray(G_sq.sum(axis=1)).flatten())
        row_norms[row_norms == 0] = 1.0
        inv_norms = sparse.diags(1.0 / row_norms)
        G = inv_norms @ G
        return G

    def _allgather_component(self, local_vals, all_sizes):
        """Allgather component values (N_i,) -> (N_global,)."""
        from mpi4py import MPI
        sendbuf = np.ascontiguousarray(local_vals, dtype=np.float64)
        all_counts = np.asarray(all_sizes, dtype=np.int64)
        all_displs = np.concatenate([[0], np.cumsum(all_counts)[:-1]])
        recvbuf = np.zeros(int(all_counts.sum()), dtype=np.float64)
        self.comm.Allgatherv(
            [sendbuf, int(all_counts[self.rank]), MPI.DOUBLE],
            [recvbuf, all_counts, all_displs, MPI.DOUBLE],
        )
        return recvbuf

    def apply(self, grad_vec):
        """Apply smoothing to distributed PETSc gradient vector in-place."""
        # Recompute all_sizes (could cache from __init__)
        all_sizes = np.diff(np.concatenate([self.local_offsets, [self.global_n_h]]))
        all_sizes_int = all_sizes.astype(np.int64)

        owned_arr = grad_vec.getArray()

        # Extract per-component owned values (position-paired)
        h_local = owned_arr[self.h_owned]
        u_local = owned_arr[self.u_owned]
        v_local = owned_arr[self.v_owned]

        # Allgather to global
        h_global = self._allgather_component(h_local, all_sizes_int)
        u_global = self._allgather_component(u_local, all_sizes_int)
        v_global = self._allgather_component(v_local, all_sizes_int)

        # Apply: G is (local_size, global_n), so result is this rank's slice directly.
        h_smoothed = self.G @ h_global
        u_smoothed = self.G @ u_global
        v_smoothed = self.G @ v_global

        # Write back to owned entries
        result = owned_arr.copy()
        result[self.h_owned] = h_smoothed
        result[self.u_owned] = u_smoothed
        result[self.v_owned] = v_smoothed

        grad_vec.setArray(result)

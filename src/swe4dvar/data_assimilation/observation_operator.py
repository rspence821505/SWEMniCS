"""
Observation operator implementations for 4D-Var.

Maps model state to observation space: H_k: V → R^{m_k}
with MPI-aware point location and parallel assembly.

Automatically handles both Continuous Galerkin (CG) and
Discontinuous Galerkin (DG) function spaces.

FIXED VERSION: Corrects adjoint consistency issues.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
import dolfinx
import dolfinx.geometry


def is_discontinuous_space(function_space) -> bool:
    """
    Determine if a function space uses DG elements.

    Args:
        function_space: FEniCSx function space

    Returns:
        True if space uses DG (discontinuous) elements
    """
    element = function_space.element

    # Try to get the basix_element - this may fail for mixed elements
    try:
        basix_element = element.basix_element
    except RuntimeError:
        # For mixed elements, basix_element raises RuntimeError.
        # Walk the sub-elements: if ANY is DG, treat the whole mixed
        # space as DG (so the adjoint uses the multi-cell path).
        # Previous code returned False for all mixed spaces, causing
        # the adjoint to misclassify SWE's DG mixed space as CG.
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

    # Check if element has discontinuous property (newer Basix API)
    if hasattr(basix_element, "discontinuous"):
        return basix_element.discontinuous

    # Fallback: check family_name if available (older API)
    if hasattr(basix_element, "family_name"):
        family_name = basix_element.family_name
        return "Discontinuous" in family_name or family_name == "DG"

    # Alternative: check element family enum
    try:
        import basix

        if hasattr(basix, "ElementFamily"):
            # Get the family from the element
            family = basix_element.family
            # DG elements typically have family == ElementFamily.P and discontinuous=True
            # But we need the discontinuous flag which should be available
            return False  # Default to continuous if we can't determine
    except:
        pass

    return False


class ObservationOperator(ABC):
    """
    Abstract base class for observation operators.

    Defines interface for H and H^T operations needed
    for cost function evaluation and adjoint computation.
    """

    def __init__(self, function_space, comm: MPI.Comm = None):
        """
        Initialize observation operator.

        Args:
            function_space: FEniCSx function space
            comm: MPI communicator
        """
        self.function_space = function_space
        self.comm = comm or MPI.COMM_WORLD

    @abstractmethod
    def forward(self, state: PETSc.Vec) -> PETSc.Vec:
        """
        Apply observation operator: H(u).

        Args:
            state: Model state vector

        Returns:
            Observation vector
        """
        pass

    @abstractmethod
    def adjoint(self, innovation: PETSc.Vec) -> PETSc.Vec:
        """
        Apply adjoint operator: H^T(d).

        Args:
            innovation: Observation-space vector (e.g., H(u) - y)

        Returns:
            State-space vector
        """
        pass

    def get_num_observations(self) -> int:
        """Return number of observations."""
        raise NotImplementedError("Must be implemented by subclass")


class PointObservationOperator(ObservationOperator):
    """
    Point-wise observation operator.

    Observes state values at specified spatial locations.
    Uses MPI-aware point location to handle distributed meshes.

    **Automatically handles both CG and DG spaces**:
    - CG: Single value per point (continuous across elements)
    - DG: Averages over multiple cells at element boundaries

    This ensures adjoint consistency for both discretizations.
    """

    def __init__(
        self,
        function_space,
        observation_points: np.ndarray,
        component_indices: Optional[List[int]] = None,
        comm: MPI.Comm = None,
        dg_averaging: str = "arithmetic",
        sub_component: Optional[int] = None,
    ):
        """
        Initialize point observation operator.

        Args:
            function_space: FEniCSx function space (CG or DG)
            observation_points: Array of (x, y) coordinates, shape (n_obs, 2)
            component_indices: Which components to observe (for mixed spaces).
                             e.g. [0] for h, [1] for (u,v) in a mixed (h, (u,v)) space.
            comm: MPI communicator
            dg_averaging: For DG spaces, how to handle multiple values:
                         'arithmetic' - simple average (default)
                         'volume_weighted' - weight by cell volume
                         'first' - take first cell only (breaks adjoint consistency!)
            sub_component: For vector sub-spaces, which scalar to extract.
                          e.g. component_indices=[1] selects (u,v), then
                          sub_component=0 extracts u, sub_component=1 extracts v.
                          If None, extracts first scalar (backward-compatible).
        """
        super().__init__(function_space, comm)
        self.obs_points = observation_points
        self.n_obs = len(observation_points)
        self.components = component_indices
        self.sub_component = sub_component
        self.dg_averaging = dg_averaging

        # Determine spatial dimension from mesh
        self.mesh = function_space.mesh
        self.gdim = self.mesh.geometry.dim

        # Check if this is a mixed element space
        element = function_space.element
        try:
            _ = element.basix_element
            self.is_mixed = False
        except RuntimeError:
            self.is_mixed = True

        # Check if this is a DG space
        self.is_dg = is_discontinuous_space(function_space)

        # MPI-aware point location data
        if self.is_dg:
            # DG: Store ALL cells containing each point
            self._local_points: List[np.ndarray] = []
            self._local_cells_all: List[List[int]] = []  # Multiple cells per point
            self._local_cells: List[int] = []  # First cell (for ownership)
            self._local_indices: List[int] = []
            self._point_ownership: np.ndarray = None
        else:
            # CG: Store single cell per point
            self._local_points: List[np.ndarray] = []
            self._local_cells: List[int] = []
            self._local_indices: List[int] = []
            self._point_ownership: np.ndarray = None

        self._setup_parallel_point_location()

    def _setup_parallel_point_location(self):
        """
        Determine which rank owns each observation point.

        For CG: Finds single cell containing each point
        For DG: Finds ALL cells containing each point (for proper averaging)

        Uses dolfinx collision detection to find cells containing
        observation points in distributed mesh.
        """
        mesh = self.mesh

        # Ensure points have correct dimension
        points = self.obs_points.copy()
        # DOLFINx geometry functions always expect 3D points, even for 2D meshes
        if points.shape[1] < 3:
            # Pad with zeros to make points 3D
            padding = np.zeros((self.n_obs, 3 - points.shape[1]))
            points = np.hstack([points, padding])
        elif points.shape[1] > 3:
            # Truncate to 3D
            points = points[:, :3]

        # Find collision candidates: which cells might contain points
        bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)

        # Compute collisions for all points at once
        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(
            mesh, cell_candidates, points
        )

        cells_per_point = []
        cells_all_per_point = [] if self.is_dg else None

        for i in range(len(points)):
            # Get cells containing point i
            cells_for_point = colliding_cells.links(i)

            if len(cells_for_point) > 0:
                if self.is_dg:
                    # DG: Store ALL cells
                    cells_all_per_point.append(list(cells_for_point))
                    cells_per_point.append(cells_for_point[0])  # First for ownership
                else:
                    # CG: Store first cell only
                    cells_per_point.append(cells_for_point[0])
            else:
                if self.is_dg:
                    cells_all_per_point.append([])
                cells_per_point.append(-1)

        cells_per_point = np.array(cells_per_point, dtype=np.int32)

        # Gather ownership information across all ranks
        all_cells = self.comm.allgather(cells_per_point)

        # Build global ownership map
        self._point_ownership = np.full(self.n_obs, -1, dtype=np.int32)
        for rank, rank_cells in enumerate(all_cells):
            for i, cell in enumerate(rank_cells):
                if cell >= 0 and self._point_ownership[i] == -1:
                    self._point_ownership[i] = rank

        # Verify all points were found
        missing_points = np.where(self._point_ownership == -1)[0]
        if len(missing_points) > 0:
            # Make sure error is raised on all ranks
            error_msg = (
                f"Observation points not found in mesh: {missing_points}\n"
                f"Point coordinates: {self.obs_points[missing_points]}"
            )
            # Broadcast error to all ranks
            has_error = self.comm.allreduce(len(missing_points) > 0, op=MPI.LOR)
            if has_error:
                raise RuntimeError(error_msg)

        # Store local point data for this rank
        local_mask = self._point_ownership == self.comm.rank
        self._local_indices = np.where(local_mask)[0].tolist()
        self._local_cells = [cells_per_point[i] for i in self._local_indices]
        self._local_points = [points[i] for i in self._local_indices]

        if self.is_dg:
            # DG: Also store all cells for each local point
            self._local_cells_all = [
                cells_all_per_point[i] for i in self._local_indices
            ]

            # --- MPI parity fix for DG adjoint ---
            # The adjoint must distribute each observation to ALL cells
            # globally containing the point, not just this rank's LOCAL
            # cells. Otherwise partition-boundary obs have partial
            # contributions and the adjoint direction is partition-dependent.
            #
            # Critical: filter to OWNED cells only (exclude ghost cells)
            # so we don't double-count: ghost cells are owned by another
            # rank which will independently write to those cells' DOFs.
            # Writing from both sides would double the contribution.
            #
            # Global cell count per point = sum of OWNED cells containing
            # the point across all ranks (each cell counted exactly once).

            # Get number of owned cells on this rank (rest are ghost)
            tdim = self.mesh.topology.dim
            cell_imap = self.mesh.topology.index_map(tdim)
            n_cells_owned = cell_imap.size_local

            # Filter local cells to OWNED only, per obs point
            owned_cells_per_point = [
                [c for c in cells_for_pt if c < n_cells_owned]
                for cells_for_pt in cells_all_per_point
            ]

            # --- Basis-unity cell filter (MPI correctness for DG point obs) ---
            # compute_colliding_cells returns cells whose geometry contains
            # the point, but at partition boundaries DOLFINx may admit cells
            # where the obs-point lies strictly inside the cell (not at a
            # vertex) — often because adjacent ghost cells have bounding
            # boxes extending over the vertex. Writing basis * value to
            # those cells splatters the adjoint RHS across corners that
            # aren't at the observation vertex.
            #
            # For VERTEX-located observations (our case: obs points are
            # mesh vertices), a valid cell has exactly one basis value
            # near 1 and the rest near 0 when evaluated at the point.
            # Keep only cells satisfying this "one-hot" pattern.
            #
            # Tolerance: 1e-4 is loose enough to survive float error in
            # point-in-reference-element mapping, tight enough to reject
            # cells where the point is visibly interior (basis values
            # ~1/3 in a uniform triangle center, or various splits along
            # edges/medians).
            bunity_tol = 1e-4
            coord_tol = 1e-6  # tolerance (m) for DOF-coord vs obs-coord match
            # Determine the sub-space to use for basis evaluation
            if self.is_mixed:
                comp_idx = self.components[0] if self.components else 0
                bunity_sub_space = self.function_space.sub(comp_idx)
                if self.sub_component is not None:
                    bunity_sub_space = bunity_sub_space.sub(self.sub_component)
            else:
                bunity_sub_space = None

            # Pre-build: (V-DOF index) -> (h-space coord) map for coord-match test
            if bunity_sub_space is not None:
                bunity_space, bunity_map = bunity_sub_space.collapse()
            else:
                bunity_space, bunity_map = self.function_space.collapse() if hasattr(self.function_space, "collapse") else (self.function_space, list(range(self.function_space.dofmap.index_map.size_local + self.function_space.dofmap.index_map.num_ghosts)))
            bunity_map_arr = np.asarray(bunity_map, dtype=np.int64)
            bunity_all_coords = bunity_space.tabulate_dof_coordinates()[:, :2]
            bunity_v_to_h = {int(v): i for i, v in enumerate(bunity_map_arr)}

            # For dofmap lookup we need the sub-space dofmap (mixed case) or self's dofmap
            if bunity_sub_space is not None:
                bunity_dofmap = bunity_sub_space.dofmap
            else:
                bunity_dofmap = self.function_space.dofmap

            def _is_one_hot(bv):
                n_near_one = int(np.sum(np.abs(np.asarray(bv) - 1.0) < bunity_tol))
                n_near_zero = int(np.sum(np.abs(np.asarray(bv)) < bunity_tol))
                return (n_near_one == 1) and (n_near_zero == len(bv) - 1)

            filtered_owned = []
            n_filtered_out = 0
            n_filtered_wrong_vertex = 0
            for i, cells_for_pt in enumerate(owned_cells_per_point):
                if len(cells_for_pt) == 0:
                    filtered_owned.append([])
                    continue
                pt_i = points[i]
                pt_xy = np.asarray(pt_i[:2])
                keep = []
                for cell in cells_for_pt:
                    try:
                        if self.is_mixed:
                            bv = self._evaluate_basis_at_point_mixed(
                                pt_i, cell, bunity_sub_space
                            )
                        else:
                            bv = self._evaluate_basis_at_point(pt_i, cell)
                        if not _is_one_hot(bv):
                            n_filtered_out += 1
                            continue
                        # Verify the one-hot DG DOF's coord matches obs point
                        j_hot = int(np.argmax(np.asarray(bv)))
                        cell_dofs = bunity_dofmap.cell_dofs(cell)
                        dof = int(cell_dofs[j_hot])
                        if dof in bunity_v_to_h:
                            dof_coord = bunity_all_coords[bunity_v_to_h[dof]]
                            if np.all(np.abs(dof_coord - pt_xy) < coord_tol):
                                keep.append(cell)
                            else:
                                # Cell geometrically contains point but the
                                # one-hot vertex is at a DIFFERENT coord —
                                # reject to prevent adjoint splatter.
                                n_filtered_wrong_vertex += 1
                        else:
                            n_filtered_out += 1
                    except Exception:
                        n_filtered_out += 1
                filtered_owned.append(keep)

            owned_cells_per_point = filtered_owned
            self._my_owned_cells_per_point = owned_cells_per_point
            self._n_cells_filtered_out = n_filtered_out
            self._n_cells_filtered_wrong_vertex = n_filtered_wrong_vertex

            # --- Coord-based DG DOF lookup (MPI-robust) ---
            # For each obs point, find all OWNED DG DOFs on this rank whose
            # spatial coordinate matches the obs point. These are the
            # "correct" DG DOFs for vertex-located observations: one per
            # cell sharing the vertex.
            #
            # This avoids issues with cell_dofs/basis ordering under MPI
            # partitioning (where cell reordering can cause cell_dofs[j_hot]
            # to refer to a DG DOF at a different vertex than expected).
            n_owned_V = self.function_space.dofmap.index_map.size_local \
                * self.function_space.dofmap.index_map_bs
            # KD-tree over owned h-DOF coords
            from scipy.spatial import cKDTree as _kd
            owned_h_parent = []  # parent V-DOF indices for owned h
            owned_h_coords = []
            for vdof, hidx in bunity_v_to_h.items():
                if vdof < n_owned_V:
                    owned_h_parent.append(vdof)
                    owned_h_coords.append(bunity_all_coords[hidx])
            owned_h_parent = np.asarray(owned_h_parent, dtype=np.int64)
            owned_h_coords = np.asarray(owned_h_coords)
            if len(owned_h_coords) > 0:
                self._owned_h_tree = _kd(owned_h_coords)
                self._owned_h_parents = owned_h_parent
            else:
                self._owned_h_tree = None
                self._owned_h_parents = owned_h_parent

            # Per-obs: owned DG DOFs at the obs coord (coord match)
            obs_local_dofs = []
            for i in range(self.n_obs):
                pt_xy = np.asarray(points[i][:2])
                if self._owned_h_tree is not None:
                    dists, idxs = self._owned_h_tree.query(
                        pt_xy, k=min(20, len(owned_h_coords))
                    )
                    if np.isscalar(dists):
                        dists = np.array([dists]); idxs = np.array([idxs])
                    matching = idxs[dists < coord_tol]
                    obs_local_dofs.append([int(self._owned_h_parents[m]) for m in matching])
                else:
                    obs_local_dofs.append([])
            self._obs_owned_h_dofs = obs_local_dofs

            # Global count of DG DOFs at each obs vertex (allreduce)
            local_dof_count = np.array(
                [len(d) for d in obs_local_dofs], dtype=np.int64
            )
            global_dof_count = np.zeros_like(local_dof_count)
            self.comm.Allreduce(local_dof_count, global_dof_count, op=MPI.SUM)
            self._obs_global_dof_count = global_dof_count

            import os as _os
            if _os.environ.get("SWE4DVAR_OBS_ADJ_DEBUG", "0") == "1":
                n_kept = sum(len(c) for c in owned_cells_per_point)
                n_dof_total = int(np.sum(local_dof_count))
                # Count unique DOFs we'll write to
                all_my_write_dofs = set()
                for dof_list in obs_local_dofs:
                    for d in dof_list:
                        all_my_write_dofs.add(int(d))
                # Check how many are < n_owned_V (owned V-DOFs)
                n_in_owned = sum(1 for d in all_my_write_dofs if d < n_owned_V)
                print(f"  [obs_adj_debug] rank{self.comm.Get_rank()} cell filter: "
                      f"kept={n_kept}, filtered_out={n_filtered_out}, "
                      f"filtered_wrong_vertex={n_filtered_wrong_vertex}, "
                      f"obs_DOF_lookup_total={n_dof_total}, "
                      f"unique_write_dofs={len(all_my_write_dofs)}, "
                      f"in_owned_V={n_in_owned}, n_owned_V={n_owned_V}", flush=True)

            # Global cell count = sum of OWNED cell counts (no double-count)
            local_n_owned = np.array(
                [len(c) for c in owned_cells_per_point], dtype=np.int64
            )
            global_n_cells = np.zeros_like(local_n_owned)
            self.comm.Allreduce(local_n_owned, global_n_cells, op=MPI.SUM)
            self._global_n_cells = global_n_cells

            # Indices of obs points this rank has ANY OWNED cell for
            self._indices_with_owned_cells = [
                i for i in range(self.n_obs) if len(owned_cells_per_point[i]) > 0
            ]

    def forward(self, state: PETSc.Vec) -> PETSc.Vec:
        """
        Extract point values from state.

        For CG: Evaluates at single cell per point
        For DG: Averages over all cells sharing each point

        For distributed state, this involves:
        1. Evaluate state at local observation points
        2. (DG only) Average over multiple cells if needed
        3. Communicate to gather all observations on all ranks
        4. Return global observation vector

        Args:
            state: Distributed state vector (from forward model)

        Returns:
            Global observation vector (replicated on all ranks)
        """
        # Create FEniCSx function from PETSc vector
        u = dolfinx.fem.Function(self.function_space)

        # Copy state values to function - handle various vector types
        # Get sizes to determine how to copy
        # Account for block size (e.g., 2 for 2D vector spaces)
        state_local_size = state.getLocalSize()
        bs = self.function_space.dofmap.index_map_bs
        u_owned_size = self.function_space.dofmap.index_map.size_local * bs
        u_total_size = u_owned_size + self.function_space.dofmap.index_map.num_ghosts * bs

        if state_local_size == u_total_size:
            # State includes ghosts - direct copy
            try:
                state.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                with state.localForm() as loc_state, u.x.petsc_vec.localForm() as loc_u:
                    loc_u[:] = loc_state[:]
            except Exception:
                # Fallback if ghostUpdate fails
                u.x.array[:] = state.getArray()[:]
        elif state_local_size == u_owned_size:
            # State has only owned DOFs - copy to owned portion
            u.x.array[:u_owned_size] = state.getArray()[:]
        else:
            # Try to handle other cases - might be sequential vector
            try:
                arr = state.getArray()
                if len(arr) == u_total_size:
                    u.x.array[:] = arr
                elif len(arr) == u_owned_size:
                    u.x.array[:u_owned_size] = arr
                else:
                    raise ValueError(
                        f"State vector size {len(arr)} does not match "
                        f"function space (owned={u_owned_size}, total={u_total_size})"
                    )
            except Exception as e:
                raise ValueError(
                    f"Could not copy state vector to function: {e}"
                )

        u.x.scatter_forward()

        # For mixed spaces, extract subfunction for evaluation
        if self.is_mixed:
            # Default to first component (h) if no component specified
            comp_idx = self.components[0] if self.components else 0
            u_sub = u.sub(comp_idx)
            # Need to collapse to get evaluable function
            u_eval = u_sub.collapse()
            # For vector sub-spaces (e.g. velocity (u,v)), further extract
            # a scalar component if sub_component is set.
            if self.sub_component is not None:
                u_eval = u_eval.sub(self.sub_component).collapse()
        else:
            u_eval = u

        # Evaluate at local points
        n_local = len(self._local_points)
        local_values = np.zeros(n_local)

        if self.is_dg:
            # DG: Average over all cells containing point
            for i, (point, cells) in enumerate(
                zip(self._local_points, self._local_cells_all)
            ):
                if len(cells) == 0:
                    continue

                # Evaluate at point in ALL cells
                values = []
                weights = []

                for cell in cells:
                    value = u_eval.eval(point.reshape(1, -1), cell)

                    # For collapsed mixed space, result is scalar
                    val = value[0, 0] if value.ndim > 1 else value[0]

                    values.append(val)

                    # Compute weights for averaging
                    if self.dg_averaging == "volume_weighted":
                        cell_volume = self._compute_cell_volume(cell)
                        weights.append(cell_volume)
                    else:
                        weights.append(1.0)

                # Average over all cells
                if self.dg_averaging == "first":
                    local_values[i] = values[0]
                else:
                    weights = np.array(weights)
                    local_values[i] = np.average(values, weights=weights)
        else:
            # CG: Single value per point (standard evaluation)
            for i, (point, cell) in enumerate(
                zip(self._local_points, self._local_cells)
            ):
                # Evaluate at point in cell
                value = u_eval.eval(point.reshape(1, -1), cell)

                # For collapsed mixed space or scalar space, result is scalar
                if self.is_mixed:
                    local_values[i] = value[0, 0] if value.ndim > 1 else value[0]
                elif self.components is not None:
                    # Handle both 1D and 2D arrays from eval
                    if value.ndim > 1:
                        local_values[i] = value[0, self.components[0]]
                    else:
                        local_values[i] = value[self.components[0]]
                else:
                    # Scalar function space
                    local_values[i] = value[0, 0] if value.ndim > 1 else value[0]

        # Gather all observations on all ranks
        local_counts = self.comm.allgather(n_local)
        recvcounts = np.array(local_counts, dtype=np.int32)
        displacements = np.zeros(self.comm.size, dtype=np.int32)
        displacements[1:] = np.cumsum(recvcounts)[:-1]

        # Allocate global observation vector
        global_values = np.zeros(self.n_obs)

        # Gather local values to proper positions
        self.comm.Allgatherv(
            [local_values, n_local, MPI.DOUBLE],
            [global_values, recvcounts, displacements, MPI.DOUBLE],
        )

        # Reorder to match original point ordering
        reordered_values = np.zeros(self.n_obs)
        offset = 0
        for rank in range(self.comm.size):
            rank_indices = np.where(self._point_ownership == rank)[0]
            rank_count = len(rank_indices)
            reordered_values[rank_indices] = global_values[offset : offset + rank_count]
            offset += rank_count

        # Create PETSc Vec for observations (replicated)
        obs_vec = PETSc.Vec().createSeq(self.n_obs, comm=PETSc.COMM_SELF)
        obs_vec.setArray(reordered_values)
        obs_vec.assemble()

        return obs_vec

    def adjoint(self, innovation: PETSc.Vec) -> PETSc.Vec:
        """
        Distribute innovation back to state space.

        For CG: Distributes to basis functions in single cell
        For DG: Distributes to ALL cells sharing each point

        This implements H^T where H extracts point values.
        The adjoint distributes point values back to basis functions.

        Args:
            innovation: Observation-space vector (e.g., y - H(u))
                       Typically sequential on each rank

        Returns:
            State-space vector with distributed contributions
        """
        # Create state-space vector using dolfinx.la
        from dolfinx import la

        adj_state = la.create_petsc_vector(
            self.function_space.dofmap.index_map,
            self.function_space.dofmap.index_map_bs,
        )
        adj_state.set(0.0)  # Zero all entries including ghosts
        adj_state.assemble()

        # Get innovation values
        innov_array = innovation.getArray()

        # For mixed spaces, we need to get the sub-function space for adjoint
        if self.is_mixed:
            comp_idx = self.components[0] if self.components else 0
            sub_space = self.function_space.sub(comp_idx)
            sub_dofmap = sub_space.dofmap
            # For vector sub-spaces with sub_component, we need to further
            # select which scalar DOFs within the vector block to write to.
            if self.sub_component is not None:
                sub_sub_space = sub_space.sub(self.sub_component)
                sub_dofmap = sub_sub_space.dofmap
                sub_space = sub_sub_space
        else:
            sub_space = None
            sub_dofmap = None

        if self.is_dg:
            # DG: Distribute to ALL cells globally containing each point.
            #
            # MPI parity fix: iterate over EVERY point that has ANY local
            # cell on this rank (not just owned points), and use the
            # GLOBAL cell count as the divisor. Each rank writes to its
            # local cells' DOFs; together all ranks produce the same
            # distribution as serial.
            # --- COORD-LOOKUP PATH (MPI-robust DG adjoint) ---
            # For each obs point, identify OWNED DG DOFs at the obs
            # coordinate (precomputed in _setup_parallel_point_location).
            # Write innov/global_n_dofs to each. Avoids cell-iteration
            # pitfalls where cell_dofs[j_hot] under MPI may refer to a DG
            # DOF at a vertex different from the observation point.
            if hasattr(self, "_obs_owned_h_dofs") and self.dg_averaging != "first":
                import os as _dbg_os
                dbg_on = _dbg_os.environ.get("SWE4DVAR_OBS_ADJ_DEBUG", "0") == "1"
                n_writes = 0
                unique_dofs_written = set()
                for global_idx in range(self.n_obs):
                    my_dofs = self._obs_owned_h_dofs[global_idx]
                    if len(my_dofs) == 0:
                        continue
                    global_n_dofs = int(self._obs_global_dof_count[global_idx])
                    if global_n_dofs == 0:
                        global_n_dofs = len(my_dofs)
                    weight = 1.0 / global_n_dofs
                    value = innov_array[global_idx]
                    for dof in my_dofs:
                        adj_state.setValueLocal(
                            int(dof),
                            value * weight,
                            addv=PETSc.InsertMode.ADD,
                        )
                        n_writes += 1
                        unique_dofs_written.add(int(dof))
                adj_state.assemble()
                if dbg_on:
                    arr = adj_state.getArray()
                    n_nonzero_pre = int(np.sum(np.abs(arr) > 1e-12))
                    print(f"  [obs_adj_debug] rank{self.comm.Get_rank()} "
                          f"writes={n_writes} unique_dofs={len(unique_dofs_written)} "
                          f"nonzero_after_assemble={n_nonzero_pre}", flush=True)
                adj_state.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                adj_state.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                if dbg_on:
                    arr = adj_state.getArray()
                    n_nonzero_post = int(np.sum(np.abs(arr) > 1e-12))
                    print(f"  [obs_adj_debug] rank{self.comm.Get_rank()} "
                          f"nonzero_after_scatter={n_nonzero_post}", flush=True)
                return adj_state

            # Fallback: cell-iteration path (dg_averaging="first" or legacy)
            if hasattr(self, "_indices_with_owned_cells"):
                iter_list = [
                    (gi, self._my_owned_cells_per_point[gi])
                    for gi in self._indices_with_owned_cells
                ]
                use_global_count = True
            else:
                iter_list = [
                    (gi, self._local_cells_all[li])
                    for li, gi in enumerate(self._local_indices)
                ]
                use_global_count = False

            # The point coord lookup: for owned points we have _local_points
            # keyed by local_idx; for non-owned we rebuild from self.obs_points.
            for global_idx, cells in iter_list:
                # Coord for this obs point (3D)
                pt3 = np.zeros(3)
                pt3[:self.obs_points.shape[1]] = self.obs_points[global_idx, :self.obs_points.shape[1]]
                point = pt3
                value = innov_array[global_idx]

                if len(cells) == 0:
                    continue

                # Weight for distributing to multiple cells
                # Must use 1/(global n_cells) for MPI parity
                if self.dg_averaging == "first":
                    weight = 1.0
                    cells_to_use = [cells[0]]
                elif use_global_count:
                    global_n = int(self._global_n_cells[global_idx])
                    if global_n == 0:
                        global_n = len(cells)  # safety
                    weight = 1.0 / global_n
                    cells_to_use = cells
                else:
                    weight = 1.0 / len(cells)
                    cells_to_use = cells

                for cell in cells_to_use:
                    if self.is_mixed:
                        # For mixed spaces, only modify the observed component
                        cell_dofs = sub_dofmap.cell_dofs(cell)
                        basis_values = self._evaluate_basis_at_point_mixed(point, cell, sub_space)
                        for j, dof in enumerate(cell_dofs):
                            adj_state.setValueLocal(
                                dof,
                                value * basis_values[j] * weight,
                                addv=PETSc.InsertMode.ADD,
                            )
                    else:
                        # Get cell DOFs
                        cell_dofs = self.function_space.dofmap.cell_dofs(cell)
                        basis_values = self._evaluate_basis_at_point(point, cell)

                        # Add weighted contribution to state DOFs
                        if self.components is not None:
                            bs = self.function_space.dofmap.bs
                            for j in range(len(cell_dofs)):
                                dof_component = j % bs
                                if dof_component == self.components[0]:
                                    basis_idx = j // bs
                                    adj_state.setValueLocal(
                                        cell_dofs[j],
                                        value * basis_values[basis_idx] * weight,
                                        addv=PETSc.InsertMode.ADD,
                                    )
                        else:
                            for j, dof in enumerate(cell_dofs):
                                adj_state.setValueLocal(
                                    dof,
                                    value * basis_values[j] * weight,
                                    addv=PETSc.InsertMode.ADD,
                                )
        else:
            # CG: Single cell per point (standard adjoint)
            for local_idx, global_idx in enumerate(self._local_indices):
                point = self._local_points[local_idx]
                cell = self._local_cells[local_idx]
                value = innov_array[global_idx]

                if self.is_mixed:
                    # For mixed spaces, only modify the observed component
                    cell_dofs = sub_dofmap.cell_dofs(cell)
                    basis_values = self._evaluate_basis_at_point_mixed(point, cell, sub_space)
                    for j, dof in enumerate(cell_dofs):
                        adj_state.setValueLocal(
                            dof,
                            value * basis_values[j],
                            addv=PETSc.InsertMode.ADD,
                        )
                else:
                    # Get cell DOFs
                    cell_dofs = self.function_space.dofmap.cell_dofs(cell)
                    basis_values = self._evaluate_basis_at_point(point, cell)

                    # Add contribution to state DOFs
                    if self.components is not None:
                        bs = self.function_space.dofmap.bs
                        for j in range(len(cell_dofs)):
                            dof_component = j % bs
                            if dof_component == self.components[0]:
                                basis_idx = j // bs
                                adj_state.setValueLocal(
                                    cell_dofs[j],
                                    value * basis_values[basis_idx],
                                    addv=PETSc.InsertMode.ADD,
                                )
                    else:
                        # Scalar space - distribute to all DOFs
                        for j, dof in enumerate(cell_dofs):
                            adj_state.setValueLocal(
                                dof, value * basis_values[j], addv=PETSc.InsertMode.ADD
                            )

        # Assemble to finalize local additions
        adj_state.assemble()

        # Critical: Use scatter_reverse to properly accumulate ghost contributions
        # This sends ghost values back to their owners and adds them
        adj_state.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # After reverse scatter, we need to forward scatter to update ghosts
        # so the returned vector has consistent ghost values
        adj_state.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        return adj_state

    def _evaluate_basis_at_point(self, point: np.ndarray, cell: int) -> np.ndarray:
        """
        Evaluate all basis functions in a cell at a given point.

        This is needed for the adjoint operator to distribute
        observation values back to DOFs.

        Args:
            point: Physical coordinates of evaluation point
            cell: Cell index containing the point

        Returns:
            Array of basis function values at point
        """
        # Get reference element and mapping
        element = self.function_space.element
        mesh = self.mesh

        # Get cell geometry
        cell_vertices = mesh.geometry.x[mesh.geometry.dofmap[cell]]

        # Map physical point to reference coordinates
        ref_point = self._physical_to_reference(point, cell_vertices)

        # Evaluate basis functions at reference point
        # For P1 elements in 2D: [1-ξ-η, ξ, η]
        if element.basix_element.degree == 1:
            if self.gdim == 2:
                xi, eta = ref_point[0], ref_point[1]
                basis = np.array([1.0 - xi - eta, xi, eta])
            elif self.gdim == 3:
                xi, eta, zeta = ref_point[0], ref_point[1], ref_point[2]
                basis = np.array([1.0 - xi - eta - zeta, xi, eta, zeta])
            else:
                raise NotImplementedError(f"Dimension {self.gdim} not supported")
        else:
            # For higher-order elements, use basix directly
            basis = element.basix_element.tabulate(0, ref_point.reshape(1, -1))[0, :, 0]

        return basis

    def _evaluate_basis_at_point_mixed(
        self, point: np.ndarray, cell: int, sub_space
    ) -> np.ndarray:
        """
        Evaluate basis functions for a sub-space of a mixed element at a point.

        Args:
            point: Physical coordinates of evaluation point
            cell: Cell index containing the point
            sub_space: The sub-function space for the observed component

        Returns:
            Array of basis function values at point
        """
        mesh = self.mesh

        # Get cell geometry
        cell_vertices = mesh.geometry.x[mesh.geometry.dofmap[cell]]

        # Map physical point to reference coordinates
        ref_point = self._physical_to_reference(point, cell_vertices)

        # Get the element from the sub-space
        element = sub_space.element

        # Try to get basix_element, if that fails use simple P1 basis
        try:
            basix_element = element.basix_element
            degree = basix_element.degree
        except RuntimeError:
            # Default to P1 for sub-elements
            degree = 1

        # Evaluate basis functions at reference point
        # For P1 elements in 2D: [1-xi-eta, xi, eta]
        if degree == 1:
            if self.gdim == 2:
                xi, eta = ref_point[0], ref_point[1]
                basis = np.array([1.0 - xi - eta, xi, eta])
            elif self.gdim == 3:
                xi, eta, zeta = ref_point[0], ref_point[1], ref_point[2]
                basis = np.array([1.0 - xi - eta - zeta, xi, eta, zeta])
            else:
                raise NotImplementedError(f"Dimension {self.gdim} not supported")
        else:
            # For higher-order elements, use basix directly
            basis = basix_element.tabulate(0, ref_point.reshape(1, -1))[0, :, 0]

        return basis

    def _physical_to_reference(
        self, point: np.ndarray, cell_vertices: np.ndarray
    ) -> np.ndarray:
        """
        Map physical coordinates to reference element coordinates.

        For a triangle with vertices v0, v1, v2:
        x = v0 + ξ(v1-v0) + η(v2-v0)

        Solving for (ξ, η) given x.

        Args:
            point: Physical coordinates
            cell_vertices: Vertices of the cell

        Returns:
            Reference coordinates
        """
        if self.gdim == 2:
            v0 = cell_vertices[0, :2]
            v1 = cell_vertices[1, :2]
            v2 = cell_vertices[2, :2]

            # Solve linear system: [v1-v0, v2-v0][ξ, η]^T = x - v0
            A = np.column_stack([v1 - v0, v2 - v0])
            b = point[:2] - v0

            ref_coords = np.linalg.solve(A, b)
            return ref_coords

        elif self.gdim == 3:
            # Similar for tetrahedra
            v0 = cell_vertices[0, :]
            v1 = cell_vertices[1, :]
            v2 = cell_vertices[2, :]
            v3 = cell_vertices[3, :]

            A = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
            b = point - v0

            ref_coords = np.linalg.solve(A, b)
            return ref_coords

        else:
            raise NotImplementedError(f"Dimension {self.gdim} not supported")

    def _compute_cell_volume(self, cell: int) -> float:
        """
        Compute volume (area in 2D) of a cell.

        Used for volume-weighted averaging in DG spaces.

        Args:
            cell: Cell index

        Returns:
            Cell volume/area
        """
        mesh = self.mesh

        # Get cell vertices
        cell_vertices = mesh.geometry.x[mesh.geometry.dofmap[cell]]

        if self.gdim == 2:
            # Triangle area: 0.5 * |cross product|
            v0 = cell_vertices[0, :2]
            v1 = cell_vertices[1, :2]
            v2 = cell_vertices[2, :2]

            area = 0.5 * abs(
                (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
            )
            return area

        elif self.gdim == 3:
            # Tetrahedron volume: |det(matrix)| / 6
            v0 = cell_vertices[0, :]
            v1 = cell_vertices[1, :]
            v2 = cell_vertices[2, :]
            v3 = cell_vertices[3, :]

            matrix = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
            volume = abs(np.linalg.det(matrix)) / 6.0
            return volume

        else:
            raise NotImplementedError(f"Volume computation for dim={self.gdim}")

    def get_num_observations(self) -> int:
        """Return total number of observation points."""
        return self.n_obs


class IntegralObservationOperator(ObservationOperator):
    """
    Integral observation operator.

    Observes spatial integrals or averages over regions:
    y_i = ∫_{Ω_i} u dx  or  y_i = (1/|Ω_i|) ∫_{Ω_i} u dx
    """

    def __init__(
        self,
        function_space,
        observation_regions: List,
        weights: Optional[List[float]] = None,
        normalize: bool = True,
        comm: MPI.Comm = None,
    ):
        """
        Initialize integral observation operator.

        Args:
            function_space: FEniCSx function space
            observation_regions: List of subdomain markers or measures
            weights: Optional weights for each region
            normalize: If True, compute averages instead of integrals
            comm: MPI communicator
        """
        super().__init__(function_space, comm)
        self.regions = observation_regions
        self.weights = weights or [1.0] * len(observation_regions)
        self.normalize = normalize

        # Precomputed assembly data
        self._assembly_matrices = None

        self._precompute_assembly_matrices()

    def _precompute_assembly_matrices(self):
        """
        Precompute integration matrices for each region.

        Builds sparse matrices H_i such that y_i = H_i · u.
        """
        # TODO: Implement using dolfinx assembly
        # This will be implemented in a later sprint when needed
        raise NotImplementedError(
            "IntegralObservationOperator to be implemented in Sprint 2"
        )

    def forward(self, state: PETSc.Vec) -> PETSc.Vec:
        """Compute regional integrals/averages."""
        raise NotImplementedError(
            "IntegralObservationOperator to be implemented in Sprint 2"
        )

    def adjoint(self, innovation: PETSc.Vec) -> PETSc.Vec:
        """Apply transposed integration matrices."""
        raise NotImplementedError(
            "IntegralObservationOperator to be implemented in Sprint 2"
        )

    def get_num_observations(self) -> int:
        """Return number of observation regions."""
        return len(self.regions)


class CompositeObservationOperator(ObservationOperator):
    """
    Composite observation operator combining multiple operators.

    Useful for heterogeneous observation types (points + integrals).
    """

    def __init__(self, operators: List[ObservationOperator], comm: MPI.Comm = None):
        """
        Initialize composite operator.

        Args:
            operators: List of observation operators to combine
            comm: MPI communicator
        """
        super().__init__(operators[0].function_space, comm)
        self.operators = operators

    def forward(self, state: PETSc.Vec) -> PETSc.Vec:
        """Apply all operators and concatenate results into a single vector."""
        sub_vecs = [op.forward(state) for op in self.operators]
        total_size = sum(v.getSize() for v in sub_vecs)
        result = PETSc.Vec().createSeq(total_size, comm=PETSc.COMM_SELF)
        offset = 0
        for v in sub_vecs:
            arr = v.getArray(readonly=True)
            result.getArray()[offset:offset + arr.size] = arr
            offset += arr.size
            v.destroy()
        result.assemble()
        return result

    def adjoint(self, innovation: PETSc.Vec) -> PETSc.Vec:
        """Split innovation across sub-operators, apply adjoints, and sum."""
        innov_arr = innovation.getArray(readonly=True)
        offset = 0
        adj_state = None
        for op in self.operators:
            n = op.get_num_observations()
            sub_innov = PETSc.Vec().createSeq(n, comm=PETSc.COMM_SELF)
            sub_innov.setArray(innov_arr[offset:offset + n].copy())
            sub_innov.assemble()
            offset += n

            sub_adj = op.adjoint(sub_innov)
            sub_innov.destroy()

            if adj_state is None:
                adj_state = sub_adj
            else:
                adj_state.axpy(1.0, sub_adj)
                sub_adj.destroy()
        return adj_state

    def get_num_observations(self) -> int:
        """Return total number of observations."""
        return sum(op.get_num_observations() for op in self.operators)

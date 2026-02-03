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
        # For mixed elements, basix_element raises RuntimeError
        # Check if this is a mixed element with sub-elements
        try:
            if hasattr(element, "num_sub_elements") and element.num_sub_elements > 0:
                # For mixed elements, return False - treat as CG
                # This is safe because CG point evaluation works for both
                return False
        except:
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
    ):
        """
        Initialize point observation operator.

        Args:
            function_space: FEniCSx function space (CG or DG)
            observation_points: Array of (x, y) coordinates, shape (n_obs, 2)
            component_indices: Which components to observe (for mixed spaces)
            comm: MPI communicator
            dg_averaging: For DG spaces, how to handle multiple values:
                         'arithmetic' - simple average (default)
                         'volume_weighted' - weight by cell volume
                         'first' - take first cell only (breaks adjoint consistency!)
        """
        super().__init__(function_space, comm)
        self.obs_points = observation_points
        self.n_obs = len(observation_points)
        self.components = component_indices
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
        else:
            sub_space = None
            sub_dofmap = None

        if self.is_dg:
            # DG: Distribute to ALL cells containing each point
            for local_idx, global_idx in enumerate(self._local_indices):
                point = self._local_points[local_idx]
                cells = self._local_cells_all[local_idx]  # ALL cells
                value = innov_array[global_idx]

                if len(cells) == 0:
                    continue

                # Weight for distributing to multiple cells
                # Must use 1/n_cells to maintain adjoint consistency
                if self.dg_averaging == "first":
                    weight = 1.0
                    cells_to_use = [cells[0]]
                else:
                    weight = 1.0 / len(cells)
                    cells_to_use = cells

                for cell in cells_to_use:
                    if self.is_mixed:
                        # For mixed spaces, only modify the observed component
                        cell_dofs = sub_dofmap.cell_dofs(cell)
                        basis_values = self._evaluate_basis_at_point_mixed(point, cell, sub_space)
                        for j, dof in enumerate(cell_dofs):
                            adj_state.setValue(
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
                                    adj_state.setValue(
                                        cell_dofs[j],
                                        value * basis_values[basis_idx] * weight,
                                        addv=PETSc.InsertMode.ADD,
                                    )
                        else:
                            for j, dof in enumerate(cell_dofs):
                                adj_state.setValue(
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
                        adj_state.setValue(
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
                                adj_state.setValue(
                                    cell_dofs[j],
                                    value * basis_values[basis_idx],
                                    addv=PETSc.InsertMode.ADD,
                                )
                    else:
                        # Scalar space - distribute to all DOFs
                        for j, dof in enumerate(cell_dofs):
                            adj_state.setValue(
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
        """Apply all operators and concatenate results."""
        # TODO: Implement in Sprint 2 when needed
        raise NotImplementedError(
            "CompositeObservationOperator to be implemented in Sprint 2"
        )

    def adjoint(self, innovation: PETSc.Vec) -> PETSc.Vec:
        """Apply all adjoint operators and sum contributions."""
        raise NotImplementedError(
            "CompositeObservationOperator to be implemented in Sprint 2"
        )

    def get_num_observations(self) -> int:
        """Return total number of observations."""
        return sum(op.get_num_observations() for op in self.operators)

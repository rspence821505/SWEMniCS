"""
Observation operator implementations for 4D-Var.

Maps model state to observation space: H_k: V → R^{m_k}
with MPI-aware point location and parallel assembly.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
import dolfinx


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
    """

    def __init__(
        self,
        function_space,
        observation_points: np.ndarray,
        component_indices: Optional[List[int]] = None,
        comm: MPI.Comm = None,
    ):
        """
        Initialize point observation operator.

        Args:
            function_space: FEniCSx function space
            observation_points: Array of (x, y) coordinates, shape (n_obs, 2)
            component_indices: Which components to observe (for mixed spaces)
            comm: MPI communicator
        """
        super().__init__(function_space, comm)
        self.obs_points = observation_points
        self.components = component_indices

        # MPI-aware point location data
        self._local_points = None
        self._local_cells = None
        self._owning_ranks = None

        self._setup_parallel_point_location()

    def _setup_parallel_point_location(self):
        """
        Determine which rank owns each observation point.

        Uses dolfinx collision detection to find cells containing
        observation points in distributed mesh.
        """
        # TODO: Implement MPI-aware point location using dolfinx
        # - Use dolfinx.geometry.compute_collisions
        # - Create communication pattern for ghost points
        # - Build local interpolation matrices
        pass

    def forward(self, state: PETSc.Vec) -> PETSc.Vec:
        """
        Extract point values from state.

        For distributed state, this involves:
        1. Evaluate state at local observation points
        2. Communicate to gather all observations on all ranks
        3. Return global observation vector
        """
        # TODO: Implement parallel point evaluation
        pass

    def adjoint(self, innovation: PETSc.Vec) -> PETSc.Vec:
        """
        Distribute innovation back to state space.

        For distributed mesh, this involves:
        1. Distribute global innovation to owning ranks
        2. Add contributions to local state DOFs
        3. Sum contributions across processors
        """
        # TODO: Implement parallel adjoint assembly
        pass

    def get_num_observations(self) -> int:
        """Return total number of observation points."""
        return len(self.obs_points)


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
        pass

    def forward(self, state: PETSc.Vec) -> PETSc.Vec:
        """Compute regional integrals/averages."""
        # TODO: Apply precomputed matrices
        pass

    def adjoint(self, innovation: PETSc.Vec) -> PETSc.Vec:
        """Apply transposed integration matrices."""
        # TODO: Apply matrix transposes
        pass

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
        # TODO: Apply each operator and stack results
        pass

    def adjoint(self, innovation: PETSc.Vec) -> PETSc.Vec:
        """Apply all adjoint operators and sum contributions."""
        # TODO: Split innovation and apply adjoints
        pass

    def get_num_observations(self) -> int:
        """Return total number of observations."""
        return sum(op.get_num_observations() for op in self.operators)

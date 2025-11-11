"""
Checkpointing strategies for adjoint computations.

Implements:
- Full trajectory storage (small problems)
- State-only checkpointing (medium problems)
- Binomial checkpointing (large problems)
- Jacobian storage management
"""

from abc import ABC, abstractmethod
import petsc4py.PETSc as PETSc
from enum import Enum


class CheckpointingStrategy(Enum):
    """Available checkpointing strategies."""

    FULL_TRAJECTORY = "full"  # Store all states + Jacobians
    STATE_ONLY = "state_only"  # Store states, recompute Jacobians
    BINOMIAL = "binomial"  # Optimal O(log N) checkpoints


class CheckpointerBase(ABC):
    """
    Abstract base class for checkpointing.

    Manages storage and retrieval of forward trajectory data
    needed for adjoint computation.
    """

    @abstractmethod
    def store_forward_data(self, time_index, state, jacobian=None):
        """
        Store forward solve data at given time.

        Args:
            time_index: Time step index
            state: PETSc.Vec state vector
            jacobian: Optional PETSc.Mat Jacobian matrix
        """
        pass

    @abstractmethod
    def retrieve_forward_data(self, time_index):
        """
        Retrieve forward data for adjoint computation.

        Args:
            time_index: Time step index

        Returns:
            tuple: (state, jacobian or None)
        """
        pass

    @abstractmethod
    def get_memory_usage(self):
        """
        Estimate memory usage in bytes.

        Returns:
            int: Approximate memory footprint
        """
        pass


class FullTrajectoryCheckpointer(CheckpointerBase):
    """
    Full trajectory storage: fastest adjoint, highest memory.

    Stores all states and Jacobians during forward solve.
    Best for problems with N < 500 time steps.
    """

    def __init__(self, num_steps):
        """
        Initialize full trajectory checkpointer.

        Args:
            num_steps: Total number of time steps
        """
        self.num_steps = num_steps
        self.states = {}
        self.jacobians = {}

    def store_forward_data(self, time_index, state, jacobian=None):
        """Store state and Jacobian."""
        # TODO: Implement in Week 4, Day 16
        raise NotImplementedError("To be implemented in Sprint 2, Week 4")

    def retrieve_forward_data(self, time_index):
        """Retrieve stored state and Jacobian."""
        # TODO: Implement in Week 4, Day 16
        raise NotImplementedError("To be implemented in Sprint 2, Week 4")

    def get_memory_usage(self):
        """Estimate memory: N × (state_size + jacobian_size)"""
        # TODO: Implement in Week 4, Day 16
        raise NotImplementedError("To be implemented in Sprint 2, Week 4")


class StateOnlyCheckpointer(CheckpointerBase):
    """
    State-only checkpointing: moderate speed, low memory.

    Stores only states during forward solve.
    Recomputes Jacobians during adjoint sweep.
    Best for 500 < N < 2000 time steps.
    """

    def __init__(self, num_steps, forward_model):
        """
        Initialize state-only checkpointer.

        Args:
            num_steps: Total number of time steps
            forward_model: Reference to forward model for Jacobian recomputation
        """
        self.num_steps = num_steps
        self.forward_model = forward_model
        self.states = {}

    def store_forward_data(self, time_index, state, jacobian=None):
        """Store state only (ignore Jacobian)."""
        # TODO: Implement in Week 4, Day 16
        raise NotImplementedError("To be implemented in Sprint 2, Week 4")

    def retrieve_forward_data(self, time_index):
        """Retrieve state and recompute Jacobian."""
        # TODO: Implement in Week 4, Day 16
        raise NotImplementedError("To be implemented in Sprint 2, Week 4")

    def get_memory_usage(self):
        """Estimate memory: N × state_size only"""
        # TODO: Implement in Week 4, Day 16
        raise NotImplementedError("To be implemented in Sprint 2, Week 4")


class BinomialCheckpointer(CheckpointerBase):
    """
    Binomial checkpointing (Griewank algorithm): slowest, lowest memory.

    Stores O(log N) checkpoints optimally.
    Recomputes states and Jacobians as needed during adjoint.
    Best for N > 2000 time steps.

    References:
        Griewank & Walther (2000), "Algorithm 799: Revolve"
    """

    def __init__(self, num_steps, max_checkpoints, forward_model):
        """
        Initialize binomial checkpointer.

        Args:
            num_steps: Total number of time steps
            max_checkpoints: Maximum number of checkpoints to store
            forward_model: Reference for recomputation
        """
        self.num_steps = num_steps
        self.max_checkpoints = max_checkpoints
        self.forward_model = forward_model
        self.checkpoints = {}
        self._schedule = None

    def _compute_schedule(self):
        """
        Compute optimal checkpoint schedule.

        Uses revolve algorithm to determine when to store/restore.
        """
        # TODO: Implement in Week 4, Day 16 (optional)
        raise NotImplementedError("Optional: implement if large N required")

    def store_forward_data(self, time_index, state, jacobian=None):
        """Store checkpoint according to schedule."""
        # TODO: Implement in Week 4, Day 16 (optional)
        raise NotImplementedError("Optional: implement if large N required")

    def retrieve_forward_data(self, time_index):
        """Retrieve or recompute data as needed."""
        # TODO: Implement in Week 4, Day 16 (optional)
        raise NotImplementedError("Optional: implement if large N required")

    def get_memory_usage(self):
        """Estimate memory: O(log N) × (state_size + jacobian_size)"""
        # TODO: Implement in Week 4, Day 16 (optional)
        raise NotImplementedError("Optional: implement if large N required")


class CheckpointerFactory:
    """
    Factory for creating appropriate checkpointer based on problem size.

    Automatically selects strategy based on available memory and N.
    """

    @staticmethod
    def create(num_steps, forward_model=None, strategy=None, max_memory_gb=None):
        """
        Create appropriate checkpointer.

        Args:
            num_steps: Number of time steps
            forward_model: ImplicitForwardModel instance (for recomputation)
            strategy: Optional explicit strategy choice
            max_memory_gb: Optional memory constraint in GB

        Returns:
            CheckpointerBase: Configured checkpointer instance
        """
        if strategy is not None:
            if strategy == CheckpointingStrategy.FULL_TRAJECTORY:
                return FullTrajectoryCheckpointer(num_steps)
            elif strategy == CheckpointingStrategy.STATE_ONLY:
                return StateOnlyCheckpointer(num_steps, forward_model)
            elif strategy == CheckpointingStrategy.BINOMIAL:
                max_cp = 20  # Default
                return BinomialCheckpointer(num_steps, max_cp, forward_model)

        # Auto-select based on num_steps
        if num_steps < 500:
            return FullTrajectoryCheckpointer(num_steps)
        elif num_steps < 2000:
            return StateOnlyCheckpointer(num_steps, forward_model)
        else:
            return BinomialCheckpointer(num_steps, 20, forward_model)

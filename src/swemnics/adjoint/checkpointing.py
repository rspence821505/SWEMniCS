"""
Checkpointing strategies for adjoint computations.

Implements:
- Full trajectory storage (small problems, N < 500)
- State-only checkpointing (medium problems, 500 < N < 2000)
- Binomial checkpointing (large problems, N > 2000)
- Jacobian storage management

This module provides memory-efficient strategies for storing forward solve
data needed during adjoint computation. For implicit BDF2 schemes, both
state vectors and Jacobian matrices can be cached to avoid recomputation.

Performance Targets:
- Full trajectory: Adjoint cost ≈ 0.5× forward (using cached Jacobians)
- State-only: Adjoint cost ≈ 1.5× forward (recompute Jacobians)
- Binomial: Adjoint cost ≈ 2-3× forward (O(log N) recomputations)
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple, Dict
import numpy as np
import petsc4py.PETSc as PETSc


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
    def store_forward_data(
        self, time_index: int, state: PETSc.Vec, jacobian: Optional[PETSc.Mat] = None
    ):
        """
        Store forward solve data at given time.

        Args:
            time_index: Time step index
            state: PETSc.Vec state vector
            jacobian: Optional PETSc.Mat Jacobian matrix
        """
        pass

    @abstractmethod
    def retrieve_forward_data(
        self, time_index: int
    ) -> Tuple[PETSc.Vec, Optional[PETSc.Mat]]:
        """
        Retrieve forward data for adjoint computation.

        Args:
            time_index: Time step index

        Returns:
            tuple: (state, jacobian or None)
        """
        pass

    @abstractmethod
    def get_memory_usage(self) -> int:
        """
        Estimate memory usage in bytes.

        Returns:
            int: Approximate memory footprint
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear all stored data to free memory."""
        pass


class FullTrajectoryCheckpointer(CheckpointerBase):
    """
    Full trajectory storage: fastest adjoint, highest memory.

    Stores all states and Jacobians during forward solve.
    Best for problems with N < 500 time steps.

    Memory: O(N × (state_size + jacobian_size))
    Adjoint cost: ~0.5× forward (using cached Jacobians)
    """

    def __init__(self, num_steps: int):
        """
        Initialize full trajectory checkpointer.

        Args:
            num_steps: Total number of time steps
        """
        self.num_steps = num_steps
        self.states: Dict[int, PETSc.Vec] = {}
        self.jacobians: Dict[int, PETSc.Mat] = {}

    def store_forward_data(
        self, time_index: int, state: PETSc.Vec, jacobian: Optional[PETSc.Mat] = None
    ):
        """
        Store state and Jacobian.

        Creates deep copies to avoid overwriting during forward solve.
        """
        if time_index < 0 or time_index >= self.num_steps:
            raise ValueError(
                f"Time index {time_index} out of range [0, {self.num_steps})"
            )

        # Store state (deep copy)
        self.states[time_index] = state.copy()

        # Store Jacobian if provided (duplicate matrix structure)
        if jacobian is not None:
            self.jacobians[time_index] = jacobian.duplicate(copy=True)

    def retrieve_forward_data(
        self, time_index: int
    ) -> Tuple[PETSc.Vec, Optional[PETSc.Mat]]:
        """
        Retrieve stored state and Jacobian.

        Returns:
            tuple: (state vector, Jacobian matrix or None)

        Raises:
            KeyError: If time_index was not stored
        """
        if time_index not in self.states:
            raise KeyError(f"No stored data for time index {time_index}")

        state = self.states[time_index]
        jacobian = self.jacobians.get(time_index, None)

        return state, jacobian

    def get_memory_usage(self) -> int:
        """
        Estimate memory: N × (state_size + jacobian_size).

        Returns:
            int: Memory usage in bytes
        """
        memory_bytes = 0

        # State vectors: num_states × size × sizeof(double)
        if self.states:
            example_state = next(iter(self.states.values()))
            state_size = example_state.getSize()
            memory_bytes += len(self.states) * state_size * 8

        # Jacobian matrices: num_jacobians × nnz × 2 × sizeof(double)
        # (factor of 2 for values + indices)
        if self.jacobians:
            for jacobian in self.jacobians.values():
                info = jacobian.getInfo()
                nnz = int(info["nz_used"])
                memory_bytes += nnz * 2 * 8

        return memory_bytes

    def clear(self):
        """Clear all stored data."""
        for vec in self.states.values():
            vec.destroy()
        for mat in self.jacobians.values():
            mat.destroy()
        self.states.clear()
        self.jacobians.clear()

    def __del__(self):
        """Destructor to ensure PETSc objects are destroyed."""
        self.clear()


class StateOnlyCheckpointer(CheckpointerBase):
    """
    State-only checkpointing: moderate speed, low memory.

    Stores only states during forward solve.
    Recomputes Jacobians during adjoint sweep.
    Best for 500 < N < 2000 time steps.

    Memory: O(N × state_size)
    Adjoint cost: ~1.5× forward (recompute Jacobians)
    """

    def __init__(self, num_steps: int, forward_model):
        """
        Initialize state-only checkpointer.

        Args:
            num_steps: Total number of time steps
            forward_model: Reference to forward model for Jacobian recomputation
        """
        self.num_steps = num_steps
        self.forward_model = forward_model
        self.states: Dict[int, PETSc.Vec] = {}

    def store_forward_data(
        self, time_index: int, state: PETSc.Vec, jacobian: Optional[PETSc.Mat] = None
    ):
        """
        Store state only (ignore Jacobian).

        Args:
            time_index: Time step index
            state: State vector to store
            jacobian: Ignored (will be recomputed on retrieval)
        """
        if time_index < 0 or time_index >= self.num_steps:
            raise ValueError(
                f"Time index {time_index} out of range [0, {self.num_steps})"
            )

        # Store only state (deep copy)
        self.states[time_index] = state.copy()

    def retrieve_forward_data(
        self, time_index: int
    ) -> Tuple[PETSc.Vec, Optional[PETSc.Mat]]:
        """
        Retrieve state and recompute Jacobian.

        Args:
            time_index: Time step index

        Returns:
            tuple: (state, recomputed Jacobian)

        Raises:
            KeyError: If time_index was not stored
        """
        if time_index not in self.states:
            raise KeyError(f"No stored data for time index {time_index}")

        state = self.states[time_index]

        # Recompute Jacobian at this state
        # NOTE: This requires forward_model to have a method to compute
        # Jacobian at a given state without advancing the solution
        jacobian = None
        if hasattr(self.forward_model, "compute_jacobian"):
            jacobian = self.forward_model.compute_jacobian(state, time_index)

        return state, jacobian

    def get_memory_usage(self) -> int:
        """
        Estimate memory: N × state_size only.

        Returns:
            int: Memory usage in bytes
        """
        if not self.states:
            return 0

        example_state = next(iter(self.states.values()))
        state_size = example_state.getSize()
        return len(self.states) * state_size * 8

    def clear(self):
        """Clear all stored states."""
        for vec in self.states.values():
            vec.destroy()
        self.states.clear()

    def __del__(self):
        """Destructor to ensure PETSc objects are destroyed."""
        self.clear()


class BinomialCheckpointer(CheckpointerBase):
    """
    Binomial checkpointing (Griewank algorithm): slowest, lowest memory.

    Stores O(log N) checkpoints optimally.
    Recomputes states and Jacobians as needed during adjoint.
    Best for N > 2000 time steps.

    Memory: O(log N × (state_size + jacobian_size))
    Adjoint cost: ~2-3× forward (O(log N) recomputations per step)

    References:
        Griewank & Walther (2000), "Algorithm 799: Revolve"
        DOI: 10.1145/347837.347846
    """

    def __init__(self, num_steps: int, max_checkpoints: int, forward_model):
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
        self.checkpoints: Dict[int, Tuple[PETSc.Vec, Optional[PETSc.Mat]]] = {}
        self._schedule: Optional[list] = None

        # Compute and validate checkpoint schedule
        self._compute_schedule()

    def _compute_schedule(self):
        """
        Compute optimal checkpoint schedule.

        Uses a simplified binomial checkpointing strategy:
        - Distribute checkpoints geometrically through time
        - Denser checkpoints near the end (where adjoint starts)

        For production use, consider implementing full Revolve algorithm.
        """
        if self.max_checkpoints >= self.num_steps:
            # Store everything if we have enough space
            self._schedule = list(range(self.num_steps))
            return

        # Geometric distribution: more checkpoints near end
        # This is a simplified heuristic; full Revolve is more sophisticated
        schedule = []
        for i in range(self.max_checkpoints):
            # Quadratic spacing: denser near end
            fraction = (i / (self.max_checkpoints - 1)) ** 2
            checkpoint_idx = int(fraction * (self.num_steps - 1))
            schedule.append(checkpoint_idx)

        # Ensure first and last are included
        schedule = sorted(set(schedule + [0, self.num_steps - 1]))
        self._schedule = schedule[: self.max_checkpoints]

    def store_forward_data(
        self, time_index: int, state: PETSc.Vec, jacobian: Optional[PETSc.Mat] = None
    ):
        """
        Store checkpoint according to schedule.

        Only stores data if time_index is in the computed schedule.
        """
        if time_index not in self._schedule:
            return  # Not a checkpoint time

        # Store both state and Jacobian
        state_copy = state.copy()
        jacobian_copy = jacobian.duplicate(copy=True) if jacobian is not None else None

        self.checkpoints[time_index] = (state_copy, jacobian_copy)

    def retrieve_forward_data(
        self, time_index: int
    ) -> Tuple[PETSc.Vec, Optional[PETSc.Mat]]:
        """
        Retrieve or recompute data as needed.

        If time_index is a checkpoint, return stored data.
        Otherwise, find nearest earlier checkpoint and recompute forward.

        Args:
            time_index: Time step index

        Returns:
            tuple: (state, jacobian or None)
        """
        # Direct checkpoint hit
        if time_index in self.checkpoints:
            return self.checkpoints[time_index]

        # Find nearest earlier checkpoint
        earlier_checkpoints = [cp for cp in self._schedule if cp < time_index]
        if not earlier_checkpoints:
            raise ValueError(
                f"Cannot retrieve data for time {time_index}: no earlier checkpoint"
            )

        start_idx = max(earlier_checkpoints)
        start_state, _ = self.checkpoints[start_idx]

        # Recompute forward from start_idx to time_index
        if not hasattr(self.forward_model, "recompute_forward"):
            raise NotImplementedError(
                "Forward model must implement 'recompute_forward' for binomial checkpointing"
            )

        state, jacobian = self.forward_model.recompute_forward(
            start_state, start_idx, time_index
        )

        return state, jacobian

    def get_memory_usage(self) -> int:
        """
        Estimate memory: O(log N) × (state_size + jacobian_size).

        Returns:
            int: Memory usage in bytes
        """
        memory_bytes = 0

        for state, jacobian in self.checkpoints.values():
            # State contribution
            if state is not None:
                state_size = state.getSize()
                memory_bytes += state_size * 8

            # Jacobian contribution
            if jacobian is not None:
                info = jacobian.getInfo()
                nnz = int(info["nz_used"])
                memory_bytes += nnz * 2 * 8

        return memory_bytes

    def clear(self):
        """Clear all checkpoints."""
        for state, jacobian in self.checkpoints.values():
            if state is not None:
                state.destroy()
            if jacobian is not None:
                jacobian.destroy()
        self.checkpoints.clear()

    def __del__(self):
        """Destructor to ensure PETSc objects are destroyed."""
        self.clear()


class CheckpointerFactory:
    """
    Factory for creating appropriate checkpointer based on problem size.

    Automatically selects strategy based on available memory and N.
    """

    @staticmethod
    def create(
        num_steps: int,
        forward_model=None,
        strategy: Optional[CheckpointingStrategy] = None,
        max_memory_gb: Optional[float] = None,
    ) -> CheckpointerBase:
        """
        Create appropriate checkpointer.

        Args:
            num_steps: Number of time steps
            forward_model: ImplicitForwardModel instance (for recomputation)
            strategy: Optional explicit strategy choice
            max_memory_gb: Optional memory constraint in GB

        Returns:
            CheckpointerBase: Configured checkpointer instance

        Example:
            >>> # Auto-select based on num_steps
            >>> checkpointer = CheckpointerFactory.create(num_steps=300, forward_model=model)
            >>> # Returns FullTrajectoryCheckpointer
            >>>
            >>> # Explicit strategy
            >>> checkpointer = CheckpointerFactory.create(
            ...     num_steps=1500,
            ...     forward_model=model,
            ...     strategy=CheckpointingStrategy.STATE_ONLY
            ... )
        """
        # Explicit strategy requested
        if strategy is not None:
            if strategy == CheckpointingStrategy.FULL_TRAJECTORY:
                return FullTrajectoryCheckpointer(num_steps)

            elif strategy == CheckpointingStrategy.STATE_ONLY:
                if forward_model is None:
                    raise ValueError(
                        "forward_model required for STATE_ONLY checkpointing"
                    )
                return StateOnlyCheckpointer(num_steps, forward_model)

            elif strategy == CheckpointingStrategy.BINOMIAL:
                if forward_model is None:
                    raise ValueError(
                        "forward_model required for BINOMIAL checkpointing"
                    )
                max_cp = 20  # Default
                return BinomialCheckpointer(num_steps, max_cp, forward_model)

        # Auto-select based on num_steps and memory constraints
        if max_memory_gb is not None:
            # Estimate memory for full trajectory
            # Rough estimate: 100k DoFs, 10 nonzeros per row
            state_memory_mb = num_steps * 100_000 * 8 / 1e6  # MB
            jacobian_memory_mb = num_steps * 100_000 * 10 * 2 * 8 / 1e6  # MB
            total_memory_gb = (state_memory_mb + jacobian_memory_mb) / 1000

            if total_memory_gb <= max_memory_gb:
                return FullTrajectoryCheckpointer(num_steps)
            elif total_memory_gb / 10 <= max_memory_gb:  # States only ~10% of total
                if forward_model is None:
                    raise ValueError(
                        "forward_model required for memory-constrained problem"
                    )
                return StateOnlyCheckpointer(num_steps, forward_model)
            else:
                if forward_model is None:
                    raise ValueError("forward_model required for large problems")
                return BinomialCheckpointer(num_steps, 20, forward_model)

        # Auto-select based on num_steps only
        if num_steps < 500:
            return FullTrajectoryCheckpointer(num_steps)
        elif num_steps < 2000:
            if forward_model is None:
                raise ValueError("forward_model required for N >= 500")
            return StateOnlyCheckpointer(num_steps, forward_model)
        else:
            if forward_model is None:
                raise ValueError("forward_model required for N >= 2000")
            return BinomialCheckpointer(num_steps, 20, forward_model)


# Convenience function for common use case
def create_checkpointer(
    num_steps: int,
    forward_model=None,
    strategy: Optional[str] = None,
    max_memory_gb: Optional[float] = None,
) -> CheckpointerBase:
    """
    Convenience function to create checkpointer.

    Args:
        num_steps: Number of time steps
        forward_model: Forward model instance
        strategy: Strategy name ("full", "state_only", "binomial") or None for auto
        max_memory_gb: Optional memory constraint in GB

    Returns:
        CheckpointerBase: Configured checkpointer

    Example:
        >>> checkpointer = create_checkpointer(num_steps=300)
        >>> checkpointer = create_checkpointer(num_steps=1500, strategy="state_only")
    """
    if strategy is not None:
        strategy_enum = CheckpointingStrategy(strategy)
    else:
        strategy_enum = None

    return CheckpointerFactory.create(
        num_steps, forward_model, strategy_enum, max_memory_gb
    )

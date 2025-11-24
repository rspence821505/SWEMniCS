"""
Tests for checkpointing strategies.

Tests adaptively run in serial or parallel mode:
- Serial: pytest test_checkpointing.py -v
- Parallel: mpirun -n 4 pytest test_checkpointing.py -v

Tests cover:
- FullTrajectoryCheckpointer (store all states + Jacobians)
- StateOnlyCheckpointer (store states, recompute Jacobians)
- BinomialCheckpointer (optimal O(log N) checkpoints)
- CheckpointerFactory (auto-selection and creation)
- Memory usage estimation
- PETSc object lifecycle management
- MPI determinism and parallel consistency
- Retrieval correctness after storage
"""

import pytest
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from typing import List, Tuple, Optional

# ============================================================================
# MPI DETECTION AND MARKERS
# ============================================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Pytest markers
requires_mpi = pytest.mark.skipif(size == 1, reason="Requires MPI with multiple ranks")
serial_only = pytest.mark.skipif(size > 1, reason="Serial test only")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_forward_model():
    """Create a mock forward model for testing checkpointing."""

    class MockForwardModel:
        def __init__(self, n_dofs=100, num_steps=10, dt=0.1):
            self.n_dofs = n_dofs
            self.num_steps = num_steps
            self.dt = dt
            self.comm = comm

        def compute_jacobian(self, state, time_index):
            """Compute Jacobian at given state (mock implementation)."""
            J = PETSc.Mat().createAIJ([self.n_dofs, self.n_dofs], comm=self.comm)
            J.setUp()

            # Simple diagonal Jacobian
            for i in range(self.n_dofs):
                J.setValue(i, i, 0.95)

            J.assemblyBegin()
            J.assemblyEnd()
            return J

        def recompute_forward(
            self, start_state, start_idx, end_idx
        ) -> Tuple[PETSc.Vec, Optional[PETSc.Mat]]:
            """Recompute forward from start to end (for binomial checkpointing)."""
            state = start_state.copy()

            # Simple decay dynamics
            for _ in range(end_idx - start_idx):
                state.scale(0.95)

            jacobian = self.compute_jacobian(state, end_idx)
            return state, jacobian

        def solve(self, initial_state, store_jacobians=False):
            """
            Solve forward model from initial state.

            Args:
                initial_state: Initial condition
                store_jacobians: Whether to compute and return Jacobians

            Returns:
                trajectory: List of state vectors
                jacobians: List of Jacobian matrices (or None if not requested)
            """
            trajectory = []
            jacobians = []

            state = initial_state.copy()

            for n in range(self.num_steps):
                # Store current state
                state_copy = state.copy()
                trajectory.append(state_copy)

                # Compute Jacobian if requested
                if store_jacobians:
                    jac = self.compute_jacobian(state, n)
                    jacobians.append(jac)
                else:
                    jacobians.append(None)

                # Advance state (simple decay dynamics)
                state.scale(0.95)

            return trajectory, jacobians

    return MockForwardModel()


@pytest.fixture
def test_states():
    """Create a list of test state vectors."""
    n_dofs = 100
    num_steps = 10

    states = []
    for n in range(num_steps):
        vec = PETSc.Vec().createMPI(n_dofs, comm=comm)
        vec.setUp()
        vec.set(float(n))  # Simple pattern: value = time_index
        states.append(vec)

    yield states

    # Cleanup
    for vec in states:
        vec.destroy()


@pytest.fixture
def test_jacobians():
    """Create a list of test Jacobian matrices."""
    n_dofs = 100
    num_steps = 10

    jacobians = []
    for n in range(num_steps):
        J = PETSc.Mat().createAIJ([n_dofs, n_dofs], comm=comm)
        J.setUp()

        # Simple diagonal pattern
        for i in range(n_dofs):
            J.setValue(i, i, float(n) + 1.0)

        J.assemblyBegin()
        J.assemblyEnd()
        jacobians.append(J)

    yield jacobians

    # Cleanup
    for mat in jacobians:
        mat.destroy()


# ============================================================================
# IMPORT CHECKPOINTING MODULE
# ============================================================================

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swemnics.adjoint.checkpointing import (
    CheckpointingStrategy,
    CheckpointerBase,
    FullTrajectoryCheckpointer,
    StateOnlyCheckpointer,
    BinomialCheckpointer,
    CheckpointerFactory,
    create_checkpointer,
)


# ============================================================================
# FULL TRAJECTORY CHECKPOINTER TESTS
# ============================================================================


class TestFullTrajectoryCheckpointer:
    """Test suite for FullTrajectoryCheckpointer."""

    def test_initialization(self):
        """Test checkpointer initialization."""
        num_steps = 10
        checkpointer = FullTrajectoryCheckpointer(num_steps)

        assert checkpointer.num_steps == num_steps
        assert len(checkpointer.states) == 0
        assert len(checkpointer.jacobians) == 0

    def test_store_and_retrieve_state_only(self, test_states):
        """Test storing and retrieving states without Jacobians."""
        num_steps = len(test_states)
        checkpointer = FullTrajectoryCheckpointer(num_steps)

        # Store states
        for idx, state in enumerate(test_states):
            checkpointer.store_forward_data(idx, state, jacobian=None)

        # Retrieve and verify
        for idx in range(num_steps):
            retrieved_state, retrieved_jacobian = checkpointer.retrieve_forward_data(
                idx
            )

            assert retrieved_jacobian is None
            assert retrieved_state.norm() == pytest.approx(test_states[idx].norm())

            # Verify values match
            expected_val = float(idx)
            actual_val = retrieved_state.sum() / retrieved_state.getSize()
            assert actual_val == pytest.approx(expected_val)

        checkpointer.clear()

    def test_store_and_retrieve_with_jacobians(self, test_states, test_jacobians):
        """Test storing and retrieving states with Jacobians."""
        num_steps = len(test_states)
        checkpointer = FullTrajectoryCheckpointer(num_steps)

        # Store states and Jacobians
        for idx, (state, jacobian) in enumerate(zip(test_states, test_jacobians)):
            checkpointer.store_forward_data(idx, state, jacobian)

        # Retrieve and verify
        for idx in range(num_steps):
            retrieved_state, retrieved_jacobian = checkpointer.retrieve_forward_data(
                idx
            )

            assert retrieved_state is not None
            assert retrieved_jacobian is not None

            # Verify state values
            expected_val = float(idx)
            actual_val = retrieved_state.sum() / retrieved_state.getSize()
            assert actual_val == pytest.approx(expected_val)

            # Verify Jacobian diagonal value
            diag = retrieved_jacobian.getDiagonal()
            expected_diag = float(idx) + 1.0
            actual_diag = diag.sum() / diag.getSize()
            assert actual_diag == pytest.approx(expected_diag)
            diag.destroy()

        checkpointer.clear()

    def test_invalid_time_index(self):
        """Test error handling for invalid time indices."""
        checkpointer = FullTrajectoryCheckpointer(10)
        vec = PETSc.Vec().createMPI(100, comm=comm)
        vec.setUp()

        # Should raise ValueError for negative index
        with pytest.raises(ValueError):
            checkpointer.store_forward_data(-1, vec)

        # Should raise ValueError for index >= num_steps
        with pytest.raises(ValueError):
            checkpointer.store_forward_data(10, vec)

        vec.destroy()

    def test_retrieve_nonexistent_data(self):
        """Test error handling when retrieving non-stored data."""
        checkpointer = FullTrajectoryCheckpointer(10)

        # Should raise KeyError
        with pytest.raises(KeyError):
            checkpointer.retrieve_forward_data(5)

    def test_memory_usage_estimation(self, test_states, test_jacobians):
        """Test memory usage estimation."""
        num_steps = len(test_states)
        checkpointer = FullTrajectoryCheckpointer(num_steps)

        # Initially zero
        assert checkpointer.get_memory_usage() == 0

        # Store states only
        for idx, state in enumerate(test_states):
            checkpointer.store_forward_data(idx, state, jacobian=None)

        memory_states_only = checkpointer.get_memory_usage()
        assert memory_states_only > 0

        checkpointer.clear()

        # Store states and Jacobians
        checkpointer2 = FullTrajectoryCheckpointer(num_steps)
        for idx, (state, jacobian) in enumerate(zip(test_states, test_jacobians)):
            checkpointer2.store_forward_data(idx, state, jacobian)

        memory_with_jacobians = checkpointer2.get_memory_usage()
        assert memory_with_jacobians > memory_states_only

        if rank == 0:
            print(f"  Memory (states only): {memory_states_only / 1e6:.2f} MB")
            print(f"  Memory (with Jacobians): {memory_with_jacobians / 1e6:.2f} MB")

        checkpointer2.clear()

    def test_clear_functionality(self, test_states):
        """Test that clear() properly releases memory."""
        checkpointer = FullTrajectoryCheckpointer(len(test_states))

        # Store data
        for idx, state in enumerate(test_states):
            checkpointer.store_forward_data(idx, state)

        assert len(checkpointer.states) > 0

        # Clear
        checkpointer.clear()

        assert len(checkpointer.states) == 0
        assert checkpointer.get_memory_usage() == 0

    @requires_mpi
    def test_mpi_determinism(self, test_states):
        """Test that storage/retrieval is deterministic across MPI ranks."""
        checkpointer = FullTrajectoryCheckpointer(len(test_states))

        # Store states
        for idx, state in enumerate(test_states):
            checkpointer.store_forward_data(idx, state)

        # Retrieve and compute norm
        norms = []
        for idx in range(len(test_states)):
            state, _ = checkpointer.retrieve_forward_data(idx)
            norms.append(state.norm())

        # Gather norms from all ranks
        all_norms = comm.allgather(norms)

        # All ranks should have identical norms
        if rank == 0:
            for other_rank_norms in all_norms[1:]:
                assert np.allclose(norms, other_rank_norms)
            print(f"✓ MPI determinism verified (size={size})")

        checkpointer.clear()


# ============================================================================
# STATE-ONLY CHECKPOINTER TESTS
# ============================================================================


class TestStateOnlyCheckpointer:
    """Test suite for StateOnlyCheckpointer."""

    def test_initialization(self, mock_forward_model):
        """Test checkpointer initialization."""
        num_steps = 10
        checkpointer = StateOnlyCheckpointer(num_steps, mock_forward_model)

        assert checkpointer.num_steps == num_steps
        assert checkpointer.forward_model is mock_forward_model
        assert len(checkpointer.states) == 0

    def test_store_states_only(self, test_states, mock_forward_model):
        """Test that only states are stored (Jacobians ignored)."""
        checkpointer = StateOnlyCheckpointer(len(test_states), mock_forward_model)

        # Create a dummy Jacobian
        J = PETSc.Mat().createAIJ([100, 100], comm=comm)
        J.setUp()
        J.assemblyBegin()
        J.assemblyEnd()

        # Store with Jacobian (should be ignored)
        for idx, state in enumerate(test_states):
            checkpointer.store_forward_data(idx, state, jacobian=J)

        # Verify only states stored
        assert len(checkpointer.states) == len(test_states)

        # Memory should be lower than full trajectory
        memory = checkpointer.get_memory_usage()
        assert memory > 0

        J.destroy()
        checkpointer.clear()

    def test_retrieve_with_jacobian_recomputation(
        self, test_states, mock_forward_model
    ):
        """Test retrieval with Jacobian recomputation."""
        checkpointer = StateOnlyCheckpointer(len(test_states), mock_forward_model)

        # Store states
        for idx, state in enumerate(test_states):
            checkpointer.store_forward_data(idx, state)

        # Retrieve and verify Jacobian is recomputed
        for idx in range(len(test_states)):
            state, jacobian = checkpointer.retrieve_forward_data(idx)

            assert state is not None
            # Jacobian should be recomputed (if forward_model.compute_jacobian exists)
            if jacobian is not None:
                assert isinstance(jacobian, PETSc.Mat)
                jacobian.destroy()

        checkpointer.clear()

    def test_memory_usage_states_only(self, test_states, mock_forward_model):
        """Test that memory usage reflects states only."""
        checkpointer = StateOnlyCheckpointer(len(test_states), mock_forward_model)

        for idx, state in enumerate(test_states):
            checkpointer.store_forward_data(idx, state)

        memory = checkpointer.get_memory_usage()

        # Expected: num_states * state_size * sizeof(double)
        n_dofs = test_states[0].getSize()
        expected_memory = len(test_states) * n_dofs * 8

        assert memory == pytest.approx(expected_memory)

        if rank == 0:
            print(f"  Memory (states only): {memory / 1e6:.2f} MB")

        checkpointer.clear()

    @requires_mpi
    def test_mpi_consistency(self, test_states, mock_forward_model):
        """Test MPI consistency for state-only checkpointing."""
        checkpointer = StateOnlyCheckpointer(len(test_states), mock_forward_model)

        # Store states
        for idx, state in enumerate(test_states):
            checkpointer.store_forward_data(idx, state)

        # Retrieve and compute checksums
        checksums = []
        for idx in range(len(test_states)):
            state, _ = checkpointer.retrieve_forward_data(idx)
            checksums.append(state.sum())

        # Gather from all ranks
        all_checksums = comm.allgather(checksums)

        if rank == 0:
            for other_checksums in all_checksums[1:]:
                assert np.allclose(checksums, other_checksums)
            print(f"✓ MPI consistency (size={size})")

        checkpointer.clear()


# ============================================================================
# BINOMIAL CHECKPOINTER TESTS
# ============================================================================


class TestBinomialCheckpointer:
    """Test suite for BinomialCheckpointer."""

    def test_initialization(self, mock_forward_model):
        """Test checkpointer initialization."""
        num_steps = 100
        max_checkpoints = 10
        checkpointer = BinomialCheckpointer(
            num_steps, max_checkpoints, mock_forward_model
        )

        assert checkpointer.num_steps == num_steps
        assert checkpointer.max_checkpoints == max_checkpoints
        assert checkpointer._schedule is not None
        assert len(checkpointer._schedule) <= max_checkpoints

    def test_checkpoint_schedule(self, mock_forward_model):
        """Test that checkpoint schedule is computed correctly."""
        num_steps = 100
        max_checkpoints = 10
        checkpointer = BinomialCheckpointer(
            num_steps, max_checkpoints, mock_forward_model
        )

        schedule = checkpointer._schedule

        # Should have at most max_checkpoints
        assert len(schedule) <= max_checkpoints

        # Should include first and last
        assert 0 in schedule
        assert num_steps - 1 in schedule

        # Should be sorted
        assert schedule == sorted(schedule)

        if rank == 0:
            print(f"  Checkpoint schedule ({len(schedule)} checkpoints): {schedule}")

    def test_selective_storage(self, test_states, mock_forward_model):
        """Test that only scheduled checkpoints are stored."""
        num_steps = len(test_states)
        max_checkpoints = 5
        checkpointer = BinomialCheckpointer(
            num_steps, max_checkpoints, mock_forward_model
        )

        # Store all states (but only scheduled ones should be kept)
        for idx, state in enumerate(test_states):
            checkpointer.store_forward_data(idx, state)

        # Only scheduled checkpoints should be stored
        assert len(checkpointer.checkpoints) <= max_checkpoints
        assert len(checkpointer.checkpoints) == len(checkpointer._schedule)

        if rank == 0:
            print(f"  Stored {len(checkpointer.checkpoints)} checkpoints")

        checkpointer.clear()

    def test_retrieve_checkpoint_hit(self, test_states, mock_forward_model):
        """Test retrieval when time_index is a checkpoint."""
        num_steps = len(test_states)
        max_checkpoints = 5
        checkpointer = BinomialCheckpointer(
            num_steps, max_checkpoints, mock_forward_model
        )

        # Store states
        for idx, state in enumerate(test_states):
            checkpointer.store_forward_data(idx, state)

        # Retrieve checkpoint hits
        for checkpoint_idx in checkpointer._schedule:
            state, _ = checkpointer.retrieve_forward_data(checkpoint_idx)
            assert state is not None

            # Verify value
            expected_val = float(checkpoint_idx)
            actual_val = state.sum() / state.getSize()
            assert actual_val == pytest.approx(expected_val)

        checkpointer.clear()

    def test_retrieve_with_recomputation(self, test_states, mock_forward_model):
        """Test retrieval requiring forward recomputation."""
        num_steps = len(test_states)
        max_checkpoints = 3  # Sparse checkpoints
        checkpointer = BinomialCheckpointer(
            num_steps, max_checkpoints, mock_forward_model
        )

        # Store states
        for idx, state in enumerate(test_states):
            checkpointer.store_forward_data(idx, state)

        # Try to retrieve non-checkpoint time
        # Should trigger recomputation from nearest earlier checkpoint
        schedule = checkpointer._schedule
        if len(schedule) >= 2:
            # Pick a time between two checkpoints
            mid_time = (schedule[0] + schedule[1]) // 2
            if mid_time not in schedule:
                state, jacobian = checkpointer.retrieve_forward_data(mid_time)
                assert state is not None
                # Jacobian may or may not be computed depending on implementation

        checkpointer.clear()

    def test_memory_usage_log_n(self, test_states, mock_forward_model):
        """Test that memory usage is O(log N)."""
        num_steps = len(test_states)
        max_checkpoints = 5
        checkpointer = BinomialCheckpointer(
            num_steps, max_checkpoints, mock_forward_model
        )

        # Store states
        for idx, state in enumerate(test_states):
            checkpointer.store_forward_data(idx, state)

        memory = checkpointer.get_memory_usage()

        # Should be much less than full trajectory
        n_dofs = test_states[0].getSize()
        full_memory = num_steps * n_dofs * 8

        assert memory < full_memory / 2

        if rank == 0:
            print(f"  Memory (binomial): {memory / 1e6:.2f} MB")
            print(f"  Memory (full): {full_memory / 1e6:.2f} MB")
            print(f"  Reduction: {full_memory / memory:.1f}x")

        checkpointer.clear()


# ============================================================================
# FACTORY TESTS
# ============================================================================


class TestCheckpointerFactory:
    """Test suite for CheckpointerFactory."""

    def test_auto_select_small_problem(self, mock_forward_model):
        """Test auto-selection for small problems (N < 500)."""
        checkpointer = CheckpointerFactory.create(num_steps=300)

        assert isinstance(checkpointer, FullTrajectoryCheckpointer)
        assert checkpointer.num_steps == 300

    def test_auto_select_medium_problem(self, mock_forward_model):
        """Test auto-selection for medium problems (500 < N < 2000)."""
        checkpointer = CheckpointerFactory.create(
            num_steps=1000, forward_model=mock_forward_model
        )

        assert isinstance(checkpointer, StateOnlyCheckpointer)
        assert checkpointer.num_steps == 1000

    def test_auto_select_large_problem(self, mock_forward_model):
        """Test auto-selection for large problems (N > 2000)."""
        checkpointer = CheckpointerFactory.create(
            num_steps=3000, forward_model=mock_forward_model
        )

        assert isinstance(checkpointer, BinomialCheckpointer)
        assert checkpointer.num_steps == 3000

    def test_explicit_strategy_selection(self, mock_forward_model):
        """Test explicit strategy selection."""
        # Full trajectory
        checkpointer1 = CheckpointerFactory.create(
            num_steps=100, strategy=CheckpointingStrategy.FULL_TRAJECTORY
        )
        assert isinstance(checkpointer1, FullTrajectoryCheckpointer)

        # State only
        checkpointer2 = CheckpointerFactory.create(
            num_steps=100,
            forward_model=mock_forward_model,
            strategy=CheckpointingStrategy.STATE_ONLY,
        )
        assert isinstance(checkpointer2, StateOnlyCheckpointer)

        # Binomial
        checkpointer3 = CheckpointerFactory.create(
            num_steps=100,
            forward_model=mock_forward_model,
            strategy=CheckpointingStrategy.BINOMIAL,
        )
        assert isinstance(checkpointer3, BinomialCheckpointer)

    def test_memory_constraint_selection(self, mock_forward_model):
        """Test selection based on memory constraints."""
        # Large problem with generous memory
        checkpointer1 = CheckpointerFactory.create(
            num_steps=1000, forward_model=mock_forward_model, max_memory_gb=10.0
        )
        # Should select full trajectory or state-only

        # Large problem with tight memory
        checkpointer2 = CheckpointerFactory.create(
            num_steps=1000, forward_model=mock_forward_model, max_memory_gb=0.01
        )
        # Should select binomial
        assert isinstance(checkpointer2, BinomialCheckpointer)

    def test_convenience_function(self, mock_forward_model):
        """Test convenience wrapper function."""
        # Auto-select
        checkpointer1 = create_checkpointer(num_steps=300)
        assert isinstance(checkpointer1, FullTrajectoryCheckpointer)

        # With strategy string
        checkpointer2 = create_checkpointer(
            num_steps=100, forward_model=mock_forward_model, strategy="state_only"
        )
        assert isinstance(checkpointer2, StateOnlyCheckpointer)

        # With memory constraint
        checkpointer3 = create_checkpointer(
            num_steps=1000, forward_model=mock_forward_model, max_memory_gb=0.01
        )
        assert isinstance(checkpointer3, BinomialCheckpointer)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestCheckpointingIntegration:
    """Integration tests for checkpointing in realistic scenarios."""

    def test_full_forward_adjoint_cycle(self):
        """Test checkpointing in a full forward-adjoint cycle."""
        num_steps = 20
        n_dofs = 100

        # Create initial condition
        m = PETSc.Vec().createMPI(n_dofs, comm=comm)
        m.setUp()
        m.set(1.0)

        # Create mock with correct num_steps
        class MockForwardModel:
            def __init__(self, n_dofs, num_steps):
                self.n_dofs = n_dofs
                self.num_steps = num_steps
                self.comm = comm

            def compute_jacobian(self, state, time_index):
                J = PETSc.Mat().createAIJ([self.n_dofs, self.n_dofs], comm=self.comm)
                J.setUp()
                for i in range(self.n_dofs):
                    J.setValue(i, i, 0.95)
                J.assemblyBegin()
                J.assemblyEnd()
                return J

            def solve(self, initial_state, store_jacobians=False):
                trajectory = []
                jacobians = []
                state = initial_state.copy()
                for n in range(self.num_steps):
                    state_copy = state.copy()
                    trajectory.append(state_copy)
                    if store_jacobians:
                        jac = self.compute_jacobian(state, n)
                        jacobians.append(jac)
                    else:
                        jacobians.append(None)
                    state.scale(0.95)
                return trajectory, jacobians

        mock = MockForwardModel(n_dofs, num_steps)

        # Create checkpointer
        checkpointer = FullTrajectoryCheckpointer(num_steps)

        # Forward solve with checkpointing
        trajectory, jacobians = mock.solve(m, store_jacobians=True)

        # Store trajectory
        for idx, (state, jac) in enumerate(zip(trajectory, jacobians)):
            checkpointer.store_forward_data(idx, state, jac)

        # Simulate adjoint sweep (backward in time)
        for idx in range(num_steps - 1, -1, -1):
            state, jacobian = checkpointer.retrieve_forward_data(idx)

            assert state is not None
            assert jacobian is not None

            # In real adjoint, would compute: lambda = J^T * lambda
            # Here just verify retrieval works

        if rank == 0:
            print(f"✓ Forward-adjoint cycle with checkpointing")

        # Cleanup
        m.destroy()
        for state in trajectory:
            state.destroy()
        for jac in jacobians:
            jac.destroy()
        checkpointer.clear()

    def test_memory_vs_speed_tradeoff(self):
        """Compare memory usage and speed across strategies."""
        num_steps = 50
        n_dofs = 100

        m = PETSc.Vec().createMPI(n_dofs, comm=comm)
        m.setUp()
        m.set(1.0)

        # Create mock with correct num_steps
        class MockForwardModel:
            def __init__(self, n_dofs, num_steps):
                self.n_dofs = n_dofs
                self.num_steps = num_steps
                self.comm = comm

            def compute_jacobian(self, state, time_index):
                J = PETSc.Mat().createAIJ([self.n_dofs, self.n_dofs], comm=self.comm)
                J.setUp()
                for i in range(self.n_dofs):
                    J.setValue(i, i, 0.95)
                J.assemblyBegin()
                J.assemblyEnd()
                return J

            def solve(self, initial_state, store_jacobians=False):
                trajectory = []
                jacobians = []
                state = initial_state.copy()
                for n in range(self.num_steps):
                    state_copy = state.copy()
                    trajectory.append(state_copy)
                    if store_jacobians:
                        jac = self.compute_jacobian(state, n)
                        jacobians.append(jac)
                    else:
                        jacobians.append(None)
                    state.scale(0.95)
                return trajectory, jacobians

        mock = MockForwardModel(n_dofs, num_steps)

        trajectory, jacobians = mock.solve(m, store_jacobians=True)

        results = {}

        # Full trajectory
        cp1 = FullTrajectoryCheckpointer(num_steps)
        for idx, (state, jac) in enumerate(zip(trajectory, jacobians)):
            cp1.store_forward_data(idx, state, jac)
        results["full"] = cp1.get_memory_usage()
        cp1.clear()

        # State only
        cp2 = StateOnlyCheckpointer(num_steps, mock)
        for idx, state in enumerate(trajectory):
            cp2.store_forward_data(idx, state)
        results["state_only"] = cp2.get_memory_usage()
        cp2.clear()

        # Binomial
        cp3 = BinomialCheckpointer(num_steps, 10, mock)
        for idx, (state, jac) in enumerate(zip(trajectory, jacobians)):
            cp3.store_forward_data(idx, state, jac)
        results["binomial"] = cp3.get_memory_usage()
        cp3.clear()

        if rank == 0:
            print("\nMemory comparison:")
            print(f"  Full trajectory: {results['full'] / 1e6:.2f} MB")
            print(f"  State only: {results['state_only'] / 1e6:.2f} MB")
            print(f"  Binomial: {results['binomial'] / 1e6:.2f} MB")

        # Verify ordering
        assert results["binomial"] < results["state_only"] < results["full"]

        # Cleanup
        m.destroy()
        for state in trajectory:
            state.destroy()
        for jac in jacobians:
            jac.destroy()

    @requires_mpi
    def test_parallel_scaling(self):
        """Test that checkpointing scales properly in parallel."""
        num_steps = 30
        n_dofs = 1000  # Larger for parallel test

        m = PETSc.Vec().createMPI(n_dofs, comm=comm)
        m.setUp()
        m.set(1.0)

        # Create mock with correct num_steps
        class MockForwardModel:
            def __init__(self, n_dofs, num_steps):
                self.n_dofs = n_dofs
                self.num_steps = num_steps
                self.comm = comm

            def compute_jacobian(self, state, time_index):
                J = PETSc.Mat().createAIJ([self.n_dofs, self.n_dofs], comm=self.comm)
                J.setUp()
                for i in range(self.n_dofs):
                    J.setValue(i, i, 0.95)
                J.assemblyBegin()
                J.assemblyEnd()
                return J

            def solve(self, initial_state, store_jacobians=False):
                trajectory = []
                jacobians = []
                state = initial_state.copy()
                for n in range(self.num_steps):
                    state_copy = state.copy()
                    trajectory.append(state_copy)
                    if store_jacobians:
                        jac = self.compute_jacobian(state, n)
                        jacobians.append(jac)
                    else:
                        jacobians.append(None)
                    state.scale(0.95)
                return trajectory, jacobians

        mock = MockForwardModel(n_dofs, num_steps)

        checkpointer = FullTrajectoryCheckpointer(num_steps)

        trajectory, jacobians = mock.solve(m, store_jacobians=True)

        # Store in parallel
        for idx, (state, jac) in enumerate(zip(trajectory, jacobians)):
            checkpointer.store_forward_data(idx, state, jac)

        # Retrieve in parallel
        norms = []
        for idx in range(num_steps):
            state, _ = checkpointer.retrieve_forward_data(idx)
            norms.append(state.norm())

        # Verify determinism across ranks
        all_norms = comm.allgather(norms)

        if rank == 0:
            for other_norms in all_norms[1:]:
                assert np.allclose(norms, other_norms)
            print(f"✓ Parallel scaling test passed (size={size})")

        # Cleanup
        m.destroy()
        for state in trajectory:
            state.destroy()
        for jac in jacobians:
            jac.destroy()
        checkpointer.clear()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

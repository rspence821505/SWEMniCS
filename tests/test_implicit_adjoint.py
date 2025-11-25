"""
Comprehensive test suite for ImplicitAdjointSolver.

Tests all key features in both serial and parallel modes:
- Transpose system solves
- BDF2 time-coupling
- Observation forcing integration
- Adjoint consistency
- MPI determinism
- Checkpointing integration

Run tests:
    Serial: pytest test_implicit_adjoint.py -v
    Parallel: mpirun -n 4 pytest test_implicit_adjoint.py -v
"""

import pytest
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from typing import List, Optional, Tuple

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
# MOCK CLASSES FOR TESTING
# ============================================================================


class MockForwardModel:
    """
    Mock forward model for testing adjoint solver.

    Provides minimal interface needed for adjoint computation:
    - get_mass_matrix() or mass_matrix attribute
    - compute_jacobian() for recomputation
    """

    def __init__(self, n_dofs: int = 100, dt: float = 0.1):
        """
        Initialize mock forward model.

        Parameters
        ----------
        n_dofs : int
            Number of degrees of freedom.
        dt : float
            Time step size.
        """
        self.n_dofs = n_dofs
        self.dt = dt
        self.comm = comm
        self._mass_matrix = None
        self._jacobian_calls = 0

    def get_mass_matrix(self) -> PETSc.Mat:
        """Get mass matrix (identity for simplicity)."""
        if self._mass_matrix is None:
            M = PETSc.Mat().createAIJ([self.n_dofs, self.n_dofs], comm=self.comm)
            M.setUp()

            start, end = M.getOwnershipRange()
            for i in range(start, end):
                M.setValue(i, i, 1.0)

            M.assemblyBegin()
            M.assemblyEnd()

            self._mass_matrix = M

        return self._mass_matrix

    def compute_jacobian(
        self,
        state: PETSc.Vec,
        state_prev: Optional[PETSc.Vec],
        state_prev_prev: Optional[PETSc.Vec],
        n: int,
    ) -> PETSc.Mat:
        """
        Compute Jacobian at given state.

        Creates a simple tridiagonal Jacobian for testing.
        """
        self._jacobian_calls += 1

        J = PETSc.Mat().createAIJ([self.n_dofs, self.n_dofs], comm=self.comm)
        J.setUp()

        start, end = J.getOwnershipRange()

        # Simple tridiagonal structure
        for i in range(start, end):
            # Diagonal (depends on time step for variety)
            J.setValue(i, i, 2.0 + 0.1 * n)

            # Off-diagonals (if in range)
            if i > 0:
                J.setValue(i, i - 1, -0.5)
            if i < self.n_dofs - 1:
                J.setValue(i, i + 1, -0.5)

        J.assemblyBegin()
        J.assemblyEnd()

        return J

    def solve(
        self, initial_state: PETSc.Vec, store_jacobians: bool = False
    ) -> Tuple[List[PETSc.Vec], Optional[List[PETSc.Mat]]]:
        """
        Mock forward solve.

        Creates a simple trajectory for testing.
        """
        num_steps = 10
        trajectory = []
        jacobians = [] if store_jacobians else None

        # Initial state
        state = initial_state.copy()
        trajectory.append(state.copy())

        # Forward steps
        for n in range(num_steps):
            # Simple decay dynamics
            state.scale(0.95)
            trajectory.append(state.copy())

            if store_jacobians:
                if n == 0:
                    J = self.compute_jacobian(state, None, None, n)
                elif n == 1:
                    J = self.compute_jacobian(state, trajectory[n - 1], None, n)
                else:
                    J = self.compute_jacobian(
                        state, trajectory[n - 1], trajectory[n - 2], n
                    )
                jacobians.append(J)

        return trajectory, jacobians


class MockCheckpointer:
    """Mock checkpointer for testing."""

    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self._states = {}
        self._jacobians = {}

    def store_forward_data(
        self, time_index: int, state: PETSc.Vec, jacobian: Optional[PETSc.Mat] = None
    ):
        """Store forward data."""
        self._states[time_index] = state.copy()
        if jacobian is not None:
            self._jacobians[time_index] = jacobian.duplicate(copy=True)

    def retrieve_forward_data(
        self, time_index: int
    ) -> Tuple[PETSc.Vec, Optional[PETSc.Mat]]:
        """Retrieve forward data."""
        state = self._states.get(time_index)
        jacobian = self._jacobians.get(time_index)
        return state, jacobian

    def clear(self):
        """Clear all stored data."""
        for vec in self._states.values():
            vec.destroy()
        for mat in self._jacobians.values():
            mat.destroy()
        self._states.clear()
        self._jacobians.clear()


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_forward_model():
    """Create a mock forward model."""
    model = MockForwardModel(n_dofs=100, dt=0.1)
    yield model
    # Cleanup
    if model._mass_matrix is not None:
        model._mass_matrix.destroy()


@pytest.fixture
def test_trajectory():
    """Create a test trajectory."""
    n_steps = 10
    n_dofs = 100
    trajectory = []

    for n in range(n_steps + 1):
        vec = PETSc.Vec().createMPI(n_dofs, comm=comm)
        vec.setUp()
        vec.set(1.0 + 0.1 * n)  # Simple pattern
        trajectory.append(vec)

    yield trajectory

    # Cleanup
    for vec in trajectory:
        vec.destroy()


@pytest.fixture
def test_jacobians(mock_forward_model):
    """Create test Jacobian matrices."""
    n_steps = 10
    jacobians = []

    for n in range(n_steps):
        J = mock_forward_model.compute_jacobian(None, None, None, n)
        jacobians.append(J)

    yield jacobians

    # Cleanup
    for mat in jacobians:
        mat.destroy()


@pytest.fixture
def mock_checkpointer():
    """Create a mock checkpointer."""
    checkpointer = MockCheckpointer(num_steps=10)
    yield checkpointer
    checkpointer.clear()


# ============================================================================
# IMPORT IMPLEMENTATION
# ============================================================================

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swemnics.adjoint.implicit_adjoint import (
    ImplicitAdjointSolver,
    ImplicitAdjointStepAnalyzer,
    CheckpointedImplicitAdjoint,
)


# ============================================================================
# BASIC FUNCTIONALITY TESTS (RUN IN BOTH SERIAL AND PARALLEL)
# ============================================================================


class TestImplicitAdjointSolverBasic:
    """Test basic functionality of ImplicitAdjointSolver."""

    def test_initialization(self, mock_forward_model, test_trajectory, test_jacobians):
        """Test solver initialization."""
        solver = ImplicitAdjointSolver(
            mock_forward_model, test_trajectory, test_jacobians, dt=0.1
        )

        assert solver.num_steps == 10
        assert solver.dt == 0.1
        assert len(solver.trajectory) == 11
        assert len(solver.jacobians) == 10

        if rank == 0:
            print(f"\n✓ Initialization test passed (size={size})")

        solver.cleanup()

    def test_invalid_initialization(self, mock_forward_model, test_trajectory):
        """Test that initialization fails with mismatched lengths."""
        # Create wrong number of Jacobians
        jacobians = []
        for n in range(5):  # Wrong length!
            J = mock_forward_model.compute_jacobian(None, None, None, n)
            jacobians.append(J)

        with pytest.raises(ValueError, match="Expected 10 Jacobians"):
            ImplicitAdjointSolver(
                mock_forward_model, test_trajectory, jacobians, dt=0.1
            )

        # Cleanup
        for mat in jacobians:
            mat.destroy()

        if rank == 0:
            print(f"✓ Invalid initialization test passed (size={size})")

    def test_mass_matrix_retrieval(
        self, mock_forward_model, test_trajectory, test_jacobians
    ):
        """Test mass matrix retrieval."""
        solver = ImplicitAdjointSolver(
            mock_forward_model, test_trajectory, test_jacobians, dt=0.1
        )

        M = solver._get_mass_matrix()

        assert M is not None
        assert M.getSize() == (100, 100)

        # Check that it's identity
        test_vec = test_trajectory[0].duplicate()
        test_vec.set(1.0)
        result = test_vec.duplicate()
        M.mult(test_vec, result)

        assert np.allclose(result.sum(), test_vec.sum())

        test_vec.destroy()
        result.destroy()

        if rank == 0:
            print(f"✓ Mass matrix retrieval test passed (size={size})")

        solver.cleanup()

    def test_transpose_solve(self, mock_forward_model, test_trajectory, test_jacobians):
        """Test transpose system solve."""
        solver = ImplicitAdjointSolver(
            mock_forward_model, test_trajectory, test_jacobians, dt=0.1
        )

        # Create test forcing
        forcing = test_trajectory[0].duplicate()
        forcing.set(1.0)

        # Solve transpose system
        lambda_n = solver._solve_transpose_system(5, forcing)

        # Check that result is reasonable
        assert lambda_n.getSize() == 100
        norm = lambda_n.norm()
        assert norm > 0.0

        if rank == 0:
            print(f"✓ Transpose solve test passed (size={size}, norm={norm:.6f})")

        # Cleanup
        lambda_n.destroy()
        forcing.destroy()
        solver.cleanup()

    def test_bdf2_time_coupling(
        self, mock_forward_model, test_trajectory, test_jacobians
    ):
        """Test BDF2 time-coupling assembly."""
        solver = ImplicitAdjointSolver(
            mock_forward_model, test_trajectory, test_jacobians, dt=0.1
        )

        # Create test adjoint variables
        lambda_next = test_trajectory[0].duplicate()
        lambda_next.set(1.0)

        lambda_next_next = test_trajectory[0].duplicate()
        lambda_next_next.set(0.5)

        # Assemble forcing (no observation forcing)
        forcing = solver._assemble_adjoint_forcing(
            5, lambda_next, lambda_next_next, None
        )

        # Check structure
        assert forcing.getSize() == 100

        # Expected: (4/(2*0.1))*1.0 - (1/(2*0.1))*0.5 = 20 - 2.5 = 17.5
        norm = forcing.norm()
        expected_norm = np.sqrt(100 * 17.5**2)
        assert np.isclose(norm, expected_norm, rtol=1e-10)

        if rank == 0:
            print(f"✓ BDF2 time-coupling test passed (size={size})")

        # Cleanup
        forcing.destroy()
        lambda_next.destroy()
        lambda_next_next.destroy()
        solver.cleanup()

    def test_full_adjoint_solve(
        self, mock_forward_model, test_trajectory, test_jacobians
    ):
        """Test complete adjoint solve."""
        solver = ImplicitAdjointSolver(
            mock_forward_model, test_trajectory, test_jacobians, dt=0.1
        )

        # Terminal condition (zero)
        terminal = test_trajectory[-1].duplicate()
        terminal.set(0.0)

        # Observation forcings (one at middle time)
        obs_forcings = [None] * 11
        obs_forcings[5] = test_trajectory[0].duplicate()
        obs_forcings[5].set(1.0)

        # Solve adjoint
        lambda_0 = solver.solve(terminal, obs_forcings)

        # Check result
        assert lambda_0.getSize() == 100
        norm = lambda_0.norm()
        assert norm > 0.0  # Should be non-zero due to observation forcing

        if rank == 0:
            print(f"✓ Full adjoint solve test passed (size={size}, norm={norm:.6f})")

        # Cleanup
        lambda_0.destroy()
        terminal.destroy()
        obs_forcings[5].destroy()
        solver.cleanup()


# ============================================================================
# ADJOINT CONSISTENCY TESTS
# ============================================================================


class TestAdjointConsistency:
    """Test adjoint consistency: ⟨J·δu, λ⟩ = ⟨δu, J^T·λ⟩."""

    def test_single_step_consistency(
        self, mock_forward_model, test_trajectory, test_jacobians
    ):
        """Test adjoint consistency for a single step."""
        analyzer = ImplicitAdjointStepAnalyzer(mock_forward_model, dt=0.1)

        # Test middle step
        error = analyzer.verify_adjoint_step(
            n=5, trajectory=test_trajectory, jacobians=test_jacobians, rtol=1e-10
        )

        assert error < 1e-10

        if rank == 0:
            print(
                f"✓ Single step consistency test passed (size={size}, error={error:.2e})"
            )

    def test_multiple_steps_consistency(
        self, mock_forward_model, test_trajectory, test_jacobians
    ):
        """Test adjoint consistency for multiple steps."""
        analyzer = ImplicitAdjointStepAnalyzer(mock_forward_model, dt=0.1)

        errors = []
        for n in [0, 3, 5, 7, 9]:
            error = analyzer.verify_adjoint_step(
                n=n, trajectory=test_trajectory, jacobians=test_jacobians, rtol=1e-10
            )
            errors.append(error)

        max_error = max(errors)
        assert max_error < 1e-10

        if rank == 0:
            print(
                f"✓ Multiple steps consistency test passed (size={size}, max_error={max_error:.2e})"
            )

    def test_time_coupling_verification(self, mock_forward_model, test_trajectory):
        """Test BDF2 time-coupling verification utility."""
        analyzer = ImplicitAdjointStepAnalyzer(mock_forward_model, dt=0.1)

        # Create test adjoint variables
        lambda_n = test_trajectory[0].duplicate()
        lambda_n.set(0.0)  # Not used in computation

        lambda_next = test_trajectory[0].duplicate()
        lambda_next.set(1.0)

        lambda_next_next = test_trajectory[0].duplicate()
        lambda_next_next.set(0.5)

        M = mock_forward_model.get_mass_matrix()

        # Compute time coupling
        coupling = analyzer.verify_time_coupling(
            lambda_n, lambda_next, lambda_next_next, M
        )

        # Check result
        norm = coupling.norm()
        expected_norm = np.sqrt(100 * 17.5**2)
        assert np.isclose(norm, expected_norm, rtol=1e-10)

        if rank == 0:
            print(f"✓ Time coupling verification test passed (size={size})")

        # Cleanup
        lambda_n.destroy()
        lambda_next.destroy()
        lambda_next_next.destroy()
        coupling.destroy()


# ============================================================================
# MPI DETERMINISM TESTS (PARALLEL ONLY)
# ============================================================================


@requires_mpi
class TestMPIDeterminism:
    """Test that parallel execution is deterministic."""

    def test_parallel_determinism(
        self, mock_forward_model, test_trajectory, test_jacobians
    ):
        """Test that result is identical across ranks."""
        solver = ImplicitAdjointSolver(
            mock_forward_model, test_trajectory, test_jacobians, dt=0.1
        )

        # Terminal condition
        terminal = test_trajectory[-1].duplicate()
        terminal.set(0.0)

        # Observation forcings
        obs_forcings = [None] * 11
        obs_forcings[5] = test_trajectory[0].duplicate()
        obs_forcings[5].set(1.0)

        # Solve adjoint
        lambda_0 = solver.solve(terminal, obs_forcings)

        # Compute norm on each rank
        norm = lambda_0.norm()

        # Gather norms from all ranks
        all_norms = comm.allgather(norm)

        # Check that all norms are identical
        for other_norm in all_norms:
            assert np.isclose(norm, other_norm, rtol=1e-12)

        if rank == 0:
            print(f"✓ MPI determinism test passed (size={size}, norm={norm:.10f})")
            print(f"  All ranks produced identical result")

        # Cleanup
        lambda_0.destroy()
        terminal.destroy()
        obs_forcings[5].destroy()
        solver.cleanup()

    def test_parallel_consistency_multiple_runs(self, mock_forward_model):
        """Test that multiple parallel runs give identical results."""
        # Create trajectory
        n_dofs = 100
        n_steps = 10
        trajectory = []
        for n in range(n_steps + 1):
            vec = PETSc.Vec().createMPI(n_dofs, comm=comm)
            vec.setUp()
            vec.set(1.0 + 0.1 * n)
            trajectory.append(vec)

        # Create Jacobians
        jacobians = []
        for n in range(n_steps):
            J = mock_forward_model.compute_jacobian(None, None, None, n)
            jacobians.append(J)

        # Run adjoint solve multiple times
        norms = []
        for run in range(3):
            solver = ImplicitAdjointSolver(
                mock_forward_model, trajectory, jacobians, dt=0.1
            )

            terminal = trajectory[-1].duplicate()
            terminal.set(0.0)

            obs_forcings = [None] * (n_steps + 1)
            obs_forcings[5] = trajectory[0].duplicate()
            obs_forcings[5].set(1.0)

            lambda_0 = solver.solve(terminal, obs_forcings)
            norms.append(lambda_0.norm())

            lambda_0.destroy()
            terminal.destroy()
            obs_forcings[5].destroy()
            solver.cleanup()

        # Check that all runs gave same result
        for norm in norms[1:]:
            assert np.isclose(norms[0], norm, rtol=1e-12)

        if rank == 0:
            print(f"✓ Multiple parallel runs consistency test passed (size={size})")
            print(f"  {len(norms)} runs all produced identical result")

        # Cleanup
        for vec in trajectory:
            vec.destroy()
        for mat in jacobians:
            mat.destroy()


# ============================================================================
# CHECKPOINTING TESTS
# ============================================================================


class TestCheckpointedImplicitAdjoint:
    """Test checkpointed adjoint solver."""

    def test_initialization(self, mock_forward_model, mock_checkpointer):
        """Test checkpointed adjoint initialization."""
        solver = CheckpointedImplicitAdjoint(
            mock_forward_model, mock_checkpointer, dt=0.1
        )

        assert solver.dt == 0.1
        assert solver.checkpointer is mock_checkpointer

        if rank == 0:
            print(f"✓ Checkpointed adjoint initialization test passed (size={size})")

    def test_checkpointed_solve(self, mock_forward_model, mock_checkpointer):
        """Test checkpointed adjoint solve."""
        # Create and store trajectory
        n_dofs = 100
        n_steps = 10
        trajectory = []
        jacobians = []

        for n in range(n_steps + 1):
            vec = PETSc.Vec().createMPI(n_dofs, comm=comm)
            vec.setUp()
            vec.set(1.0 + 0.1 * n)
            trajectory.append(vec)

            if n < n_steps:
                J = mock_forward_model.compute_jacobian(None, None, None, n)
                jacobians.append(J)
                mock_checkpointer.store_forward_data(n, vec, J)

        # Store final state
        mock_checkpointer.store_forward_data(n_steps, trajectory[-1], None)

        # Create solver
        solver = CheckpointedImplicitAdjoint(
            mock_forward_model, mock_checkpointer, dt=0.1
        )

        # Terminal condition
        terminal = trajectory[-1].duplicate()
        terminal.set(0.0)

        # Observation forcings
        obs_forcings = [None] * (n_steps + 1)
        obs_forcings[5] = trajectory[0].duplicate()
        obs_forcings[5].set(1.0)

        # Solve adjoint
        lambda_0 = solver.solve(terminal, obs_forcings)

        # Check result
        assert lambda_0.getSize() == n_dofs
        norm = lambda_0.norm()
        assert norm > 0.0

        if rank == 0:
            print(f"✓ Checkpointed solve test passed (size={size}, norm={norm:.6f})")

        # Cleanup
        lambda_0.destroy()
        terminal.destroy()
        obs_forcings[5].destroy()
        for vec in trajectory:
            vec.destroy()
        for mat in jacobians:
            mat.destroy()


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_observation_forcings(
        self, mock_forward_model, test_trajectory, test_jacobians
    ):
        """Test adjoint solve with no observation forcings."""
        solver = ImplicitAdjointSolver(
            mock_forward_model, test_trajectory, test_jacobians, dt=0.1
        )

        terminal = test_trajectory[-1].duplicate()
        terminal.set(0.0)

        # No observation forcings (None or all None)
        lambda_0 = solver.solve(terminal, None)

        # Result should be close to zero
        norm = lambda_0.norm()
        assert norm < 1e-10

        if rank == 0:
            print(f"✓ No observation forcings test passed (size={size})")

        lambda_0.destroy()
        terminal.destroy()
        solver.cleanup()

    def test_single_observation_forcing(
        self, mock_forward_model, test_trajectory, test_jacobians
    ):
        """Test with single observation forcing."""
        solver = ImplicitAdjointSolver(
            mock_forward_model, test_trajectory, test_jacobians, dt=0.1
        )

        terminal = test_trajectory[-1].duplicate()
        terminal.set(0.0)

        # Single observation at final time
        obs_forcings = [None] * 11
        obs_forcings[-1] = test_trajectory[0].duplicate()
        obs_forcings[-1].set(1.0)

        lambda_0 = solver.solve(terminal, obs_forcings)

        norm = lambda_0.norm()
        assert norm > 0.0

        if rank == 0:
            print(f"✓ Single observation forcing test passed (size={size})")

        lambda_0.destroy()
        terminal.destroy()
        obs_forcings[-1].destroy()
        solver.cleanup()

    def test_invalid_observation_forcings_length(
        self, mock_forward_model, test_trajectory, test_jacobians
    ):
        """Test error handling for wrong observation forcings length."""
        solver = ImplicitAdjointSolver(
            mock_forward_model, test_trajectory, test_jacobians, dt=0.1
        )

        terminal = test_trajectory[-1].duplicate()
        terminal.set(0.0)

        # Wrong length
        obs_forcings = [None] * 5

        with pytest.raises(ValueError, match="observation_forcings must have length"):
            solver.solve(terminal, obs_forcings)

        terminal.destroy()
        solver.cleanup()

        if rank == 0:
            print(f"✓ Invalid observation forcings length test passed (size={size})")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self, mock_forward_model):
        """Test complete forward-adjoint pipeline."""
        # Create initial state
        n_dofs = 100
        u0 = PETSc.Vec().createMPI(n_dofs, comm=comm)
        u0.setUp()
        u0.set(1.0)

        # Forward solve
        trajectory, jacobians = mock_forward_model.solve(u0, store_jacobians=True)

        # Create adjoint solver
        solver = ImplicitAdjointSolver(
            mock_forward_model, trajectory, jacobians, dt=mock_forward_model.dt
        )

        # Terminal condition
        terminal = trajectory[-1].duplicate()
        terminal.set(0.0)

        # Create observation forcings at multiple times
        obs_forcings = [None] * len(trajectory)
        for k in [3, 6, 9]:
            obs_forcings[k] = trajectory[0].duplicate()
            obs_forcings[k].set(0.1 * k)

        # Solve adjoint
        lambda_0 = solver.solve(terminal, obs_forcings)

        # Verify adjoint consistency
        analyzer = ImplicitAdjointStepAnalyzer(mock_forward_model, dt=0.1)
        for n in range(len(jacobians)):
            error = analyzer.verify_adjoint_step(n, trajectory, jacobians, rtol=1e-10)
            assert error < 1e-10

        if rank == 0:
            print(f"✓ Full pipeline integration test passed (size={size})")
            print(f"  Forward solve: {len(trajectory)} states")
            print(f"  Adjoint solve: λ_0 norm = {lambda_0.norm():.6f}")
            print(f"  Consistency: max error < 1e-10")

        # Cleanup
        u0.destroy()
        lambda_0.destroy()
        terminal.destroy()
        for obs in obs_forcings:
            if obs is not None:
                obs.destroy()
        for vec in trajectory:
            vec.destroy()
        for mat in jacobians:
            mat.destroy()
        solver.cleanup()


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


if __name__ == "__main__":
    # Print test environment
    if rank == 0:
        print("=" * 70)
        print("ImplicitAdjointSolver Test Suite")
        print("=" * 70)
        print(f"MPI size: {size}")
        print(f"PETSc version: {PETSc.Sys.getVersion()}")
        print("=" * 70)
        print()

    # Run pytest
    pytest.main([__file__, "-v", "-s"])

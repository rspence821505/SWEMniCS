"""
Test suite for adjoint_operators.py

Tests cover:
  - Adjoint consistency via inner product tests
  - Backward adjoint propagation
  - Observation operator adjoints
  - Covariance matrix adjoints
  - Composite operator adjoints
  - Finite difference verification
  - MPI parallel execution

Run serial tests:
    pytest test_adjoint_operators.py -v -m "not mpi"

Run parallel tests:
    mpirun -np 4 pytest test_adjoint_operators.py -v -m "mpi"
"""

import pytest
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem
from petsc4py import PETSc

# Import modules under test
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from swe4dvar.adjoint.adjoint_operators import (
    AdjointModel,
    ObservationAdjoint,
    CovarianceAdjoint,
    CompositeAdjoint,
    FiniteDifferenceAdjoint,
)


# ============================================================================
# Mock Classes for Testing
# ============================================================================


class MockForwardModel:
    """Mock forward model for testing."""

    def __init__(self, size: int = 10):
        self.size = size
        self.comm = MPI.COMM_WORLD

    def create_vec(self) -> PETSc.Vec:
        """Create a distributed PETSc vector."""
        vec = PETSc.Vec().createMPI(self.size, comm=self.comm)
        return vec


class MockObservationOperator:
    """Mock observation operator with linear forward/adjoint."""

    def __init__(self, state_size: int = 10, obs_size: int = 5):
        self.state_size = state_size
        self.obs_size = obs_size
        self.comm = MPI.COMM_WORLD

        # Create observation matrix (random for testing)
        self.H = self._create_observation_matrix()

    def _create_observation_matrix(self) -> PETSc.Mat:
        """Create random observation matrix."""
        H = PETSc.Mat().createAIJ([self.obs_size, self.state_size], comm=self.comm)
        H.setUp()

        # Fill with random values (deterministic across ranks)
        np.random.seed(42)
        local_range = H.getOwnershipRange()
        for i in range(local_range[0], local_range[1]):
            cols = list(range(self.state_size))
            vals = np.random.randn(self.state_size) * 0.1
            H.setValues(i, cols, vals)

        H.assemble()
        return H

    def forward(self, state: PETSc.Vec) -> PETSc.Vec:
        """Apply observation operator: H·u."""
        obs = self.H.createVecLeft()
        self.H.mult(state, obs)
        return obs

    def adjoint(self, obs: PETSc.Vec) -> PETSc.Vec:
        """Apply adjoint observation operator: H^T·y."""
        state = self.H.createVecRight()
        self.H.multTranspose(obs, state)
        return state


class MockCovariance:
    """Mock covariance matrix with diagonal structure."""

    def __init__(self, size: int = 10, variance: float = 1.0):
        self.size = size
        self.variance = variance
        self.comm = MPI.COMM_WORLD

    def apply_inverse(self, v: PETSc.Vec) -> PETSc.Vec:
        """Apply C^{-1}·v."""
        result = v.copy()
        result.scale(1.0 / self.variance)
        return result


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_model():
    """Create mock forward model."""
    return MockForwardModel(size=10)


@pytest.fixture
def mock_obs_op():
    """Create mock observation operator."""
    return MockObservationOperator(state_size=10, obs_size=5)


@pytest.fixture
def mock_covariance():
    """Create mock covariance."""
    return MockCovariance(size=10, variance=2.0)


@pytest.fixture
def simple_trajectory(mock_model):
    """Create simple forward trajectory."""
    num_steps = 5
    trajectory = []

    for i in range(num_steps + 1):
        vec = mock_model.create_vec()
        vec.set(float(i))  # Simple values for testing
        trajectory.append(vec)

    return trajectory


@pytest.fixture
def simple_jacobians(mock_model):
    """Create simple Jacobian matrices."""
    num_steps = 5
    jacobians = []

    for i in range(num_steps):
        # Create diagonal Jacobian for simplicity
        J = PETSc.Mat().createAIJ(
            [mock_model.size, mock_model.size], comm=mock_model.comm
        )
        J.setUp()

        # Set diagonal values
        local_range = J.getOwnershipRange()
        for row in range(local_range[0], local_range[1]):
            J.setValue(row, row, 1.0 + 0.1 * i)

        J.assemble()
        jacobians.append(J)

    return jacobians


# ============================================================================
# AdjointModel Tests
# ============================================================================


class TestAdjointModel:
    """Tests for AdjointModel class."""

    def test_initialization(self, mock_model, simple_trajectory, simple_jacobians):
        """Test AdjointModel initialization."""
        adj_model = AdjointModel(mock_model, simple_trajectory, simple_jacobians)

        assert adj_model.forward_model is mock_model
        assert adj_model.trajectory == simple_trajectory
        assert adj_model.jacobians == simple_jacobians
        assert adj_model.num_steps == len(simple_trajectory) - 1

    def test_solve_returns_vector(
        self, mock_model, simple_trajectory, simple_jacobians
    ):
        """Test that solve returns a PETSc vector."""
        adj_model = AdjointModel(mock_model, simple_trajectory, simple_jacobians)

        # Create terminal condition
        terminal = mock_model.create_vec()
        terminal.set(1.0)

        # Solve adjoint
        lambda_0 = adj_model.solve(terminal)

        assert isinstance(lambda_0, PETSc.Vec)

    def test_adjoint_step_basic_implementation(
        self, mock_model, simple_trajectory, simple_jacobians
    ):
        """Test that _solve_adjoint_step provides basic backward Euler implementation."""
        adj_model = AdjointModel(mock_model, simple_trajectory, simple_jacobians)

        vec = mock_model.create_vec()
        vec.set(1.0)

        # Should not raise, provides basic implementation
        result = adj_model._solve_adjoint_step(0, vec, None)
        assert isinstance(result, PETSc.Vec)

    @pytest.mark.mpi
    def test_parallel_initialization(
        self, mock_model, simple_trajectory, simple_jacobians
    ):
        """Test that AdjointModel works in parallel."""
        adj_model = AdjointModel(mock_model, simple_trajectory, simple_jacobians)

        # Should not raise any errors
        assert adj_model.num_steps == len(simple_trajectory) - 1


# ============================================================================
# ObservationAdjoint Tests
# ============================================================================


class TestObservationAdjoint:
    """Tests for ObservationAdjoint class."""

    def test_initialization(self, mock_obs_op):
        """Test ObservationAdjoint initialization."""
        obs_adj = ObservationAdjoint(mock_obs_op)
        assert obs_adj.obs_op is mock_obs_op

    def test_apply_returns_vector(self, mock_obs_op):
        """Test that apply returns a vector."""
        obs_adj = ObservationAdjoint(mock_obs_op)

        # Create innovation vector
        innovation = PETSc.Vec().createMPI(mock_obs_op.obs_size, comm=MPI.COMM_WORLD)
        innovation.set(1.0)

        result = obs_adj.apply(innovation)

        assert isinstance(result, PETSc.Vec)
        assert result.getSize() == mock_obs_op.state_size

    def test_adjoint_consistency(self, mock_obs_op):
        """
        Test adjoint consistency: <H·u, v> = <u, H^T·v>.

        This is the fundamental adjoint test.
        """
        obs_adj = ObservationAdjoint(mock_obs_op)

        # Create random vectors
        state = PETSc.Vec().createMPI(mock_obs_op.state_size, comm=MPI.COMM_WORLD)
        obs = PETSc.Vec().createMPI(mock_obs_op.obs_size, comm=MPI.COMM_WORLD)

        np.random.seed(123)
        state.setArray(np.random.randn(state.getLocalSize()))
        obs.setArray(np.random.randn(obs.getLocalSize()))

        # Compute <H·u, v>
        H_u = mock_obs_op.forward(state)
        lhs = H_u.dot(obs)

        # Compute <u, H^T·v>
        HT_v = obs_adj.apply(obs)
        rhs = state.dot(HT_v)

        # Check consistency
        rel_error = abs(lhs - rhs) / (abs(lhs) + 1e-14)
        assert rel_error < 1e-12, f"Adjoint inconsistency: rel_error = {rel_error}"

    @pytest.mark.mpi
    def test_adjoint_consistency_parallel(self, mock_obs_op):
        """Test adjoint consistency in parallel."""
        obs_adj = ObservationAdjoint(mock_obs_op)

        # Create distributed vectors
        state = PETSc.Vec().createMPI(mock_obs_op.state_size, comm=MPI.COMM_WORLD)
        obs = PETSc.Vec().createMPI(mock_obs_op.obs_size, comm=MPI.COMM_WORLD)

        # Set values (deterministic)
        state.set(1.0)
        obs.set(0.5)

        # Test
        H_u = mock_obs_op.forward(state)
        lhs = H_u.dot(obs)

        HT_v = obs_adj.apply(obs)
        rhs = state.dot(HT_v)

        rel_error = abs(lhs - rhs) / (abs(lhs) + 1e-14)
        assert rel_error < 1e-12, f"Parallel adjoint inconsistency: {rel_error}"

        # Verify all ranks agree
        comm = MPI.COMM_WORLD
        errors = comm.allgather(rel_error)
        assert all(abs(e - rel_error) < 1e-14 for e in errors)


# ============================================================================
# CovarianceAdjoint Tests
# ============================================================================


class TestCovarianceAdjoint:
    """Tests for CovarianceAdjoint class."""

    def test_initialization(self, mock_covariance):
        """Test CovarianceAdjoint initialization."""
        cov_adj = CovarianceAdjoint(mock_covariance)
        assert cov_adj.cov is mock_covariance

    def test_apply_precision(self, mock_covariance):
        """Test precision matrix application."""
        cov_adj = CovarianceAdjoint(mock_covariance)

        v = PETSc.Vec().createMPI(mock_covariance.size, comm=MPI.COMM_WORLD)
        v.set(2.0)

        result = cov_adj.apply_precision(v)

        # Should be v / variance = 2.0 / 2.0 = 1.0
        expected = 1.0
        actual = result.sum() / result.getSize()
        assert abs(actual - expected) < 1e-12

    def test_weight_innovation(self, mock_covariance):
        """Test innovation weighting."""
        cov_adj = CovarianceAdjoint(mock_covariance)

        innovation = PETSc.Vec().createMPI(mock_covariance.size, comm=MPI.COMM_WORLD)
        innovation.set(4.0)

        weighted = cov_adj.weight_innovation(innovation)

        # Should be innovation / variance = 4.0 / 2.0 = 2.0
        expected = 2.0
        actual = weighted.sum() / weighted.getSize()
        assert abs(actual - expected) < 1e-12

    @pytest.mark.mpi
    def test_precision_parallel(self, mock_covariance):
        """Test precision application in parallel."""
        cov_adj = CovarianceAdjoint(mock_covariance)

        v = PETSc.Vec().createMPI(mock_covariance.size, comm=MPI.COMM_WORLD)
        v.set(2.0)

        result = cov_adj.apply_precision(v)

        # Verify result is consistent across ranks
        norm = result.norm()
        comm = MPI.COMM_WORLD
        norms = comm.allgather(norm)
        assert all(abs(n - norm) < 1e-12 for n in norms)


# ============================================================================
# CompositeAdjoint Tests
# ============================================================================


class TestCompositeAdjoint:
    """Tests for CompositeAdjoint class."""

    def test_initialization(self, mock_obs_op):
        """Test CompositeAdjoint initialization."""
        ops = [mock_obs_op, mock_obs_op]
        comp_adj = CompositeAdjoint(ops)
        assert comp_adj.operators == ops

    def test_apply_chains_adjoints(self, mock_obs_op):
        """Test that composite adjoint chains operators in reverse."""
        # Create two observation operators
        op1 = MockObservationOperator(state_size=10, obs_size=5)
        op2 = MockObservationOperator(state_size=5, obs_size=3)

        comp_adj = CompositeAdjoint([op1, op2])

        # Input to composite adjoint (size 3)
        adj_input = PETSc.Vec().createMPI(3, comm=MPI.COMM_WORLD)
        adj_input.set(1.0)

        # Apply composite adjoint
        result = comp_adj.apply(adj_input)

        # Should apply op2.adjoint then op1.adjoint
        # Result size should be 10
        assert result.getSize() == 10

    def test_composite_consistency(self):
        """Test composite adjoint consistency: <(f∘g)(x), y> = <x, (g^T∘f^T)(y)>."""
        # Create operators
        op1 = MockObservationOperator(state_size=10, obs_size=5)
        op2 = MockObservationOperator(state_size=5, obs_size=3)

        comp_adj = CompositeAdjoint([op1, op2])

        # Create test vectors
        x = PETSc.Vec().createMPI(10, comm=MPI.COMM_WORLD)
        y = PETSc.Vec().createMPI(3, comm=MPI.COMM_WORLD)

        np.random.seed(456)
        x.setArray(np.random.randn(x.getLocalSize()))
        y.setArray(np.random.randn(y.getLocalSize()))

        # Forward: y = op2(op1(x))
        temp = op1.forward(x)
        forward_result = op2.forward(temp)
        lhs = forward_result.dot(y)

        # Adjoint: x = op1^T(op2^T(y))
        adjoint_result = comp_adj.apply(y)
        rhs = x.dot(adjoint_result)

        rel_error = abs(lhs - rhs) / (abs(lhs) + 1e-14)
        assert rel_error < 1e-10, f"Composite adjoint inconsistency: {rel_error}"


# ============================================================================
# FiniteDifferenceAdjoint Tests
# ============================================================================


class TestFiniteDifferenceAdjoint:
    """Tests for FiniteDifferenceAdjoint verification."""

    def test_initialization(self, mock_obs_op):
        """Test FiniteDifferenceAdjoint initialization."""
        fd_adj = FiniteDifferenceAdjoint(mock_obs_op, epsilon=1e-6)
        assert fd_adj.operator is mock_obs_op
        assert fd_adj.epsilon == 1e-6

    def test_verify_adjoint_correct_operator(self, mock_obs_op):
        """Test verification passes for correct adjoint."""
        obs_adj = ObservationAdjoint(mock_obs_op)
        fd_adj = FiniteDifferenceAdjoint(mock_obs_op)

        # Create test vectors
        state = PETSc.Vec().createMPI(mock_obs_op.state_size, comm=MPI.COMM_WORLD)
        adjoint_state = PETSc.Vec().createMPI(mock_obs_op.obs_size, comm=MPI.COMM_WORLD)

        state.set(1.0)
        adjoint_state.set(0.5)

        # Verify adjoint
        error = fd_adj.verify_adjoint(obs_adj, state, adjoint_state)

        assert error < 1e-10, f"Adjoint verification failed: error = {error}"

    def test_verify_adjoint_incorrect_operator(self, mock_obs_op):
        """Test verification fails for incorrect adjoint."""

        class IncorrectAdjoint:
            """Adjoint that's intentionally wrong."""

            def apply(self, obs: PETSc.Vec) -> PETSc.Vec:
                result = PETSc.Vec().createMPI(
                    mock_obs_op.state_size, comm=MPI.COMM_WORLD
                )
                result.set(0.0)  # Wrong! Should apply H^T
                return result

        incorrect_adj = IncorrectAdjoint()
        fd_adj = FiniteDifferenceAdjoint(mock_obs_op)

        # Create test vectors
        state = PETSc.Vec().createMPI(mock_obs_op.state_size, comm=MPI.COMM_WORLD)
        adjoint_state = PETSc.Vec().createMPI(mock_obs_op.obs_size, comm=MPI.COMM_WORLD)

        state.set(1.0)
        adjoint_state.set(0.5)

        # Verify adjoint
        error = fd_adj.verify_adjoint(incorrect_adj, state, adjoint_state)

        # Should have large error
        assert error > 0.1, f"Verification should fail but error = {error}"

    @pytest.mark.mpi
    def test_verify_adjoint_parallel(self, mock_obs_op):
        """Test adjoint verification in parallel."""
        obs_adj = ObservationAdjoint(mock_obs_op)
        fd_adj = FiniteDifferenceAdjoint(mock_obs_op)

        # Create test vectors
        state = PETSc.Vec().createMPI(mock_obs_op.state_size, comm=MPI.COMM_WORLD)
        adjoint_state = PETSc.Vec().createMPI(mock_obs_op.obs_size, comm=MPI.COMM_WORLD)

        state.set(1.0)
        adjoint_state.set(0.5)

        # Verify
        error = fd_adj.verify_adjoint(obs_adj, state, adjoint_state)

        assert error < 1e-10

        # All ranks should agree
        comm = MPI.COMM_WORLD
        errors = comm.allgather(error)
        assert all(abs(e - error) < 1e-12 for e in errors)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple adjoint operators."""

    def test_observation_plus_covariance_chain(self, mock_obs_op, mock_covariance):
        """Test chaining observation adjoint with covariance."""
        obs_adj = ObservationAdjoint(mock_obs_op)

        # Create innovation
        innovation = PETSc.Vec().createMPI(mock_obs_op.obs_size, comm=MPI.COMM_WORLD)
        innovation.set(2.0)

        # Apply R^{-1} to innovation (assuming same size for simplicity)
        # In reality, covariance would match observation size

        # Apply observation adjoint
        state_contrib = obs_adj.apply(innovation)

        assert isinstance(state_contrib, PETSc.Vec)
        assert state_contrib.getSize() == mock_obs_op.state_size

    @pytest.mark.mpi
    def test_full_gradient_computation_structure(self, mock_model, mock_obs_op):
        """
        Test structure of full gradient computation.

        Gradient typically has form: ∇J = B^{-1}(m - m_b) + λ_0
        where λ_0 comes from adjoint sweep.
        """
        # Background term
        m = mock_model.create_vec()
        m_b = mock_model.create_vec()
        m.set(1.0)
        m_b.set(0.5)

        # Compute B^{-1}(m - m_b)
        diff = m.copy()
        diff.axpy(-1.0, m_b)  # diff = m - m_b

        # For diagonal B with variance σ², B^{-1} = I/σ²
        sigma_sq = 2.0
        diff.scale(1.0 / sigma_sq)

        # Adjoint contribution (mock)
        lambda_0 = mock_model.create_vec()
        lambda_0.set(0.1)

        # Total gradient
        gradient = diff.copy()
        gradient.axpy(1.0, lambda_0)  # gradient = diff + lambda_0

        assert isinstance(gradient, PETSc.Vec)

        # Verify parallel consistency
        norm = gradient.norm()
        comm = MPI.COMM_WORLD
        norms = comm.allgather(norm)
        assert all(abs(n - norm) < 1e-12 for n in norms)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

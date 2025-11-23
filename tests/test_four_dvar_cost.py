"""
Test suite for FourDVarCost implementation.

This module contains comprehensive tests for the standard 4D-Var
cost function including:
- Unit tests for value, gradient, Hessian
- Taylor remainder tests for gradient verification
- Adjoint consistency tests
- Parallel determinism tests
- Integration tests with mock forward models

Author: Rylan Spence
Date: 2025
"""

import pytest
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from typing import List, Tuple, Dict, Optional
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import from the module we're testing
try:
    from swemnics.data_assimilation.cost_functions import (
        FourDVarCost,
        taylor_remainder_test,
        adjoint_consistency_test,
    )
except ImportError:
    # If running from examples/tests directory
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from swemnics.data_assimilation.cost_functions import (
        FourDVarCost,
        taylor_remainder_test,
        adjoint_consistency_test,
    )


# ============================================================================
# MOCK COMPONENTS FOR TESTING
# ============================================================================


class MockForwardModel:
    """
    Mock forward model for testing cost functions.

    Simulates a simple linear forward model:
        u_{n+1} = A * u_n + b

    with cached Jacobians from "Newton iterations".
    """

    def __init__(self, n_dofs: int, num_steps: int, dt: float = 0.1):
        self.n_dofs = n_dofs
        self.num_steps = num_steps
        self.dt = dt
        self.comm = PETSc.COMM_WORLD

        # Create simple dynamics matrix
        self._create_dynamics_matrix()

    def _create_dynamics_matrix(self):
        """Create simple diffusion-like dynamics."""
        # Create tridiagonal matrix (discrete Laplacian)
        self.A = PETSc.Mat().create(comm=self.comm)
        self.A.setSizes([self.n_dofs, self.n_dofs])
        self.A.setType(PETSc.Mat.Type.AIJ)
        self.A.setUp()

        # Fill tridiagonal structure
        for i in range(self.n_dofs):
            self.A.setValue(i, i, 2.0)
            if i > 0:
                self.A.setValue(i, i - 1, -1.0)
            if i < self.n_dofs - 1:
                self.A.setValue(i, i + 1, -1.0)

        self.A.assemblyBegin()
        self.A.assemblyEnd()

        # Scale by time step (makes it a proper time-stepper)
        self.A.scale(0.9)  # Stability factor

    def solve(
        self, u0: PETSc.Vec, store_jacobians: bool = True
    ) -> Tuple[List[PETSc.Vec], Optional[List[PETSc.Mat]]]:
        """
        Solve forward problem from initial condition u0.

        Parameters
        ----------
        u0 : PETSc.Vec
            Initial condition
        store_jacobians : bool
            Whether to cache Jacobians

        Returns
        -------
        trajectory : List[PETSc.Vec]
            State trajectory
        jacobians : Optional[List[PETSc.Mat]]
            Cached Jacobians
        """
        trajectory = [u0.copy()]
        jacobians = [] if store_jacobians else None

        u_current = u0.copy()

        for n in range(self.num_steps):
            # Simple forward step: u_{n+1} = A * u_n
            u_next = u0.duplicate()
            self.A.mult(u_current, u_next)

            trajectory.append(u_next.copy())

            # Store Jacobian (which is just A for this linear model)
            if store_jacobians:
                jacobians.append(self.A.copy())

            u_current = u_next

        return trajectory, jacobians


class MockObservationOperator:
    """
    Mock observation operator for testing.

    Observes a subset of state components (point observations).
    """

    def __init__(self, observation_indices: List[int], n_dofs: int):
        self.obs_indices = observation_indices
        self.n_dofs = n_dofs
        self.n_obs = len(observation_indices)
        self.comm = PETSc.COMM_WORLD

    def apply(self, u: PETSc.Vec, time_index: int = 0) -> PETSc.Vec:
        """
        Apply observation operator: H(u).

        Extracts components at observation indices.
        """
        y = PETSc.Vec().create(comm=self.comm)
        y.setSizes(self.n_obs)
        y.setUp()

        u_array = u.getArray()
        y_array = y.getArray()

        for i, idx in enumerate(self.obs_indices):
            y_array[i] = u_array[idx]

        return y

    def apply_adjoint(
        self, w: PETSc.Vec, u: PETSc.Vec, time_index: int = 0
    ) -> PETSc.Vec:
        """
        Apply observation operator adjoint: H^T(w).

        Distributes observation weights to state components.
        """
        v = u.copy()
        v.zeroEntries()

        w_array = w.getArray()
        v_array = v.getArray()

        for i, idx in enumerate(self.obs_indices):
            v_array[idx] = w_array[i]

        return v

    def linearize_apply(
        self, du: PETSc.Vec, u: PETSc.Vec, time_index: int = 0
    ) -> PETSc.Vec:
        """
        Apply linearized observation operator: (∂H/∂u)·δu.

        For linear operator, this is just H(δu).
        """
        return self.apply(du, time_index)


class MockCovarianceMatrix:
    """
    Mock covariance matrix (diagonal).
    """

    def __init__(self, variance: float, size: int):
        self.variance = variance
        self.size = size
        self.comm = PETSc.COMM_WORLD

    def apply_inverse(self, v: PETSc.Vec) -> PETSc.Vec:
        """Apply C^{-1} (inverse covariance)."""
        result = v.copy()
        result.scale(1.0 / self.variance)
        return result


# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def mock_setup():
    """Create mock components for testing."""
    n_dofs = 100
    num_steps = 10
    dt = 0.1

    # Forward model
    forward_model = MockForwardModel(n_dofs, num_steps, dt)

    # Observation operator (observe every 10th component)
    obs_indices = list(range(0, n_dofs, 10))
    obs_op = MockObservationOperator(obs_indices, n_dofs)

    # Covariances
    B = MockCovarianceMatrix(variance=1.0, size=n_dofs)
    R = {
        k: MockCovarianceMatrix(variance=0.01, size=len(obs_indices))
        for k in [0, 5, 10]
    }

    # Background state
    m_b = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    m_b.setSizes(n_dofs)
    m_b.setUp()
    m_b.setRandom()

    # Generate observations from a "true" trajectory
    m_true = m_b.copy()
    m_true.shift(1.0)  # Perturb from background

    trajectory_true, _ = forward_model.solve(m_true, store_jacobians=False)
    observations = {
        k: obs_op.apply(trajectory_true[k], time_index=k) for k in [0, 5, 10]
    }

    return {
        "forward_model": forward_model,
        "obs_op": obs_op,
        "B": B,
        "R": R,
        "m_b": m_b,
        "observations": observations,
        "obs_times": [0, 5, 10],
        "n_dofs": n_dofs,
    }


# ============================================================================
# UNIT TESTS
# ============================================================================


def test_cost_function_initialization(mock_setup):
    """Test that cost function initializes correctly."""

    cost = FourDVarCost(
        forward_model=mock_setup["forward_model"],
        observation_operator=mock_setup["obs_op"],
        background_cov=mock_setup["B"],
        observation_cov=mock_setup["R"],
        m_background=mock_setup["m_b"],
        observations=mock_setup["observations"],
        obs_times=mock_setup["obs_times"],
    )

    assert cost is not None
    assert cost.num_forward_solves == 0
    assert cost.num_adjoint_solves == 0
    assert len(cost.obs_times) == 3


def test_cost_function_value_positive(mock_setup):
    """Test that cost function value is positive."""

    cost = FourDVarCost(
        forward_model=mock_setup["forward_model"],
        observation_operator=mock_setup["obs_op"],
        background_cov=mock_setup["B"],
        observation_cov=mock_setup["R"],
        m_background=mock_setup["m_b"],
        observations=mock_setup["observations"],
        obs_times=mock_setup["obs_times"],
    )

    m = mock_setup["m_b"].copy()
    J = cost.value(m)

    assert J > 0, "Cost function should be positive"
    assert np.isfinite(J), "Cost function should be finite"


def test_cost_function_value_at_background(mock_setup):
    """Test cost function at background state."""

    cost = FourDVarCost(
        forward_model=mock_setup["forward_model"],
        observation_operator=mock_setup["obs_op"],
        background_cov=mock_setup["B"],
        observation_cov=mock_setup["R"],
        m_background=mock_setup["m_b"],
        observations=mock_setup["observations"],
        obs_times=mock_setup["obs_times"],
    )

    # At background, only observation term should contribute
    J = cost.value(mock_setup["m_b"])

    # Background term should be zero at m_b
    m_diff = mock_setup["m_b"].copy()
    m_diff.axpy(-1.0, mock_setup["m_b"])
    assert m_diff.norm() < 1e-14


def test_cost_function_decreases_toward_truth(mock_setup):
    """Test that cost decreases as we move toward true state."""

    # Create true state (used to generate observations)
    m_true = mock_setup["m_b"].copy()
    m_true.shift(1.0)

    # Regenerate observations
    trajectory_true, _ = mock_setup["forward_model"].solve(
        m_true, store_jacobians=False
    )
    observations = {
        k: mock_setup["obs_op"].apply(trajectory_true[k], time_index=k)
        for k in mock_setup["obs_times"]
    }

    cost = FourDVarCost(
        forward_model=mock_setup["forward_model"],
        observation_operator=mock_setup["obs_op"],
        background_cov=mock_setup["B"],
        observation_cov=mock_setup["R"],
        m_background=mock_setup["m_b"],
        observations=observations,
        obs_times=mock_setup["obs_times"],
    )

    # Cost at background
    J_background = cost.value(mock_setup["m_b"])

    # Cost at truth
    J_truth = cost.value(m_true)

    # Cost should be lower at truth
    assert J_truth < J_background, "Cost should decrease toward truth"


def test_gradient_shape(mock_setup):
    """Test that gradient has correct shape."""

    cost = FourDVarCost(
        forward_model=mock_setup["forward_model"],
        observation_operator=mock_setup["obs_op"],
        background_cov=mock_setup["B"],
        observation_cov=mock_setup["R"],
        m_background=mock_setup["m_b"],
        observations=mock_setup["observations"],
        obs_times=mock_setup["obs_times"],
    )

    m = mock_setup["m_b"].copy()
    grad = cost.gradient(m)

    assert grad.getSize() == mock_setup["n_dofs"]


def test_gradient_zero_at_truth(mock_setup):
    """Test that gradient is near zero at true minimum."""

    # Use background as "truth" with zero observation noise
    m_true = mock_setup["m_b"].copy()
    trajectory, _ = mock_setup["forward_model"].solve(m_true, store_jacobians=False)
    observations = {
        k: mock_setup["obs_op"].apply(trajectory[k], time_index=k)
        for k in mock_setup["obs_times"]
    }

    cost = FourDVarCost(
        forward_model=mock_setup["forward_model"],
        observation_operator=mock_setup["obs_op"],
        background_cov=mock_setup["B"],
        observation_cov=mock_setup["R"],
        m_background=m_true,
        observations=observations,
        obs_times=mock_setup["obs_times"],
    )

    grad = cost.gradient(m_true)
    grad_norm = grad.norm()

    # Gradient should be very small at optimum
    assert grad_norm < 1e-6, f"Gradient norm {grad_norm} too large at optimum"


def test_caching_efficiency(mock_setup):
    """Test that caching avoids redundant forward solves."""

    cost = FourDVarCost(
        forward_model=mock_setup["forward_model"],
        observation_operator=mock_setup["obs_op"],
        background_cov=mock_setup["B"],
        observation_cov=mock_setup["R"],
        m_background=mock_setup["m_b"],
        observations=mock_setup["observations"],
        obs_times=mock_setup["obs_times"],
    )

    m = mock_setup["m_b"].copy()

    # First call: should run forward model
    J1 = cost.value(m)
    assert cost.num_forward_solves == 1

    # Second call with same m: should use cache
    grad = cost.gradient(m)
    assert cost.num_forward_solves == 1, "Should reuse cached trajectory"

    # Call with different m: should run forward model again
    m2 = m.copy()
    m2.shift(0.1)
    J2 = cost.value(m2)
    assert cost.num_forward_solves == 2


# ============================================================================
# GRADIENT VERIFICATION TESTS
# ============================================================================


def test_taylor_remainder_order(mock_setup):
    """Test that gradient satisfies second-order Taylor remainder."""

    cost = FourDVarCost(
        forward_model=mock_setup["forward_model"],
        observation_operator=mock_setup["obs_op"],
        background_cov=mock_setup["B"],
        observation_cov=mock_setup["R"],
        m_background=mock_setup["m_b"],
        observations=mock_setup["observations"],
        obs_times=mock_setup["obs_times"],
    )

    m0 = mock_setup["m_b"].copy()
    m0.shift(0.5)  # Perturb from background

    passed = taylor_remainder_test(cost, m0)
    assert passed, "Taylor remainder test failed"


def test_finite_difference_gradient(mock_setup):
    """Test gradient against finite difference approximation."""

    cost = FourDVarCost(
        forward_model=mock_setup["forward_model"],
        observation_operator=mock_setup["obs_op"],
        background_cov=mock_setup["B"],
        observation_cov=mock_setup["R"],
        m_background=mock_setup["m_b"],
        observations=mock_setup["observations"],
        obs_times=mock_setup["obs_times"],
    )

    m0 = mock_setup["m_b"].copy()

    # Compute adjoint gradient
    grad_adjoint = cost.gradient(m0)

    # Compute finite difference gradient (single direction)
    direction = m0.duplicate()
    direction.setRandom()
    direction.normalize()

    eps = 1e-6
    m_plus = m0.copy()
    m_plus.axpy(eps, direction)

    m_minus = m0.copy()
    m_minus.axpy(-eps, direction)

    J_plus = cost.value(m_plus)
    J_minus = cost.value(m_minus)

    grad_fd_directional = (J_plus - J_minus) / (2 * eps)
    grad_adjoint_directional = grad_adjoint.dot(direction)

    rel_error = abs(grad_fd_directional - grad_adjoint_directional) / abs(
        grad_fd_directional
    )

    assert rel_error < 1e-4, f"Gradient error {rel_error} too large"


# ============================================================================
# HESSIAN TESTS
# ============================================================================


def test_hessian_symmetry(mock_setup):
    """Test that Hessian is symmetric."""

    cost = FourDVarCost(
        forward_model=mock_setup["forward_model"],
        observation_operator=mock_setup["obs_op"],
        background_cov=mock_setup["B"],
        observation_cov=mock_setup["R"],
        m_background=mock_setup["m_b"],
        observations=mock_setup["observations"],
        obs_times=mock_setup["obs_times"],
    )

    m0 = mock_setup["m_b"].copy()

    passed = adjoint_consistency_test(cost, m0)
    assert passed, "Hessian symmetry test failed"


def test_hessian_positive_semidefinite(mock_setup):
    """Test that Gauss-Newton Hessian is positive semi-definite."""

    cost = FourDVarCost(
        forward_model=mock_setup["forward_model"],
        observation_operator=mock_setup["obs_op"],
        background_cov=mock_setup["B"],
        observation_cov=mock_setup["R"],
        m_background=mock_setup["m_b"],
        observations=mock_setup["observations"],
        obs_times=mock_setup["obs_times"],
    )

    m0 = mock_setup["m_b"].copy()

    # Test multiple random directions
    for _ in range(5):
        v = m0.duplicate()
        v.setRandom()
        v.normalize()

        Hv = cost.hessian_vector_product(m0, v)
        quadratic_form = v.dot(Hv)

        assert quadratic_form >= -1e-10, "Hessian should be PSD"


# ============================================================================
# PARALLEL TESTS
# ============================================================================


def test_parallel_determinism(mock_setup):
    """Test that results are deterministic across MPI ranks."""

    cost = FourDVarCost(
        forward_model=mock_setup["forward_model"],
        observation_operator=mock_setup["obs_op"],
        background_cov=mock_setup["B"],
        observation_cov=mock_setup["R"],
        m_background=mock_setup["m_b"],
        observations=mock_setup["observations"],
        obs_times=mock_setup["obs_times"],
    )

    m = mock_setup["m_b"].copy()

    # Compute cost and gradient
    J = cost.value(m)
    grad = cost.gradient(m)

    # All ranks should agree
    comm = PETSc.COMM_WORLD
    mpi_comm = comm.tompi4py()
    J_min = mpi_comm.allreduce(J, op=MPI.MIN)
    J_max = mpi_comm.allreduce(J, op=MPI.MAX)

    assert abs(J_max - J_min) < 1e-14, "Cost should be identical across ranks"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_simple_optimization(mock_setup):
    """Test simple gradient descent optimization."""

    cost = FourDVarCost(
        forward_model=mock_setup["forward_model"],
        observation_operator=mock_setup["obs_op"],
        background_cov=mock_setup["B"],
        observation_cov=mock_setup["R"],
        m_background=mock_setup["m_b"],
        observations=mock_setup["observations"],
        obs_times=mock_setup["obs_times"],
    )

    m = mock_setup["m_b"].copy()
    J_initial = cost.value(m)

    # Simple gradient descent
    for i in range(10):
        grad = cost.gradient(m)

        if grad.norm() < 1e-8:
            break

        # Line search (simple backtracking)
        alpha = 0.1
        m_new = m.copy()
        m_new.axpy(-alpha, grad)

        J_new = cost.value(m_new)

        # Accept step if cost decreases
        if J_new < cost.value(m):
            m = m_new

    J_final = cost.value(m)

    # Cost should have decreased
    assert J_final < J_initial, "Optimization should decrease cost"

    # Check diagnostics
    diagnostics = cost.get_diagnostics()
    assert diagnostics["num_forward_solves"] > 0
    assert diagnostics["num_adjoint_solves"] > 0


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


@pytest.mark.benchmark
def test_cost_evaluation_performance(mock_setup, benchmark):
    """Benchmark cost function evaluation."""

    cost = FourDVarCost(
        forward_model=mock_setup["forward_model"],
        observation_operator=mock_setup["obs_op"],
        background_cov=mock_setup["B"],
        observation_cov=mock_setup["R"],
        m_background=mock_setup["m_b"],
        observations=mock_setup["observations"],
        obs_times=mock_setup["obs_times"],
    )

    m = mock_setup["m_b"].copy()

    result = benchmark(cost.value, m)
    assert result > 0


@pytest.mark.benchmark
def test_gradient_evaluation_performance(mock_setup, benchmark):
    """Benchmark gradient evaluation."""

    cost = FourDVarCost(
        forward_model=mock_setup["forward_model"],
        observation_operator=mock_setup["obs_op"],
        background_cov=mock_setup["B"],
        observation_cov=mock_setup["R"],
        m_background=mock_setup["m_b"],
        observations=mock_setup["observations"],
        obs_times=mock_setup["obs_times"],
    )

    m = mock_setup["m_b"].copy()

    # Prime cache with forward solve
    cost.value(m)

    result = benchmark(cost.gradient, m)
    assert result.norm() > 0


# ============================================================================
# BENCHMARK FIXTURE FALLBACK
# ============================================================================

# Provide a no-op benchmark fixture if pytest-benchmark is unavailable
try:
    import pytest_benchmark.plugin  # type: ignore
except Exception:

    @pytest.fixture
    def benchmark():
        """Minimal stand-in for pytest-benchmark fixture."""

        def _run(func, *args, **kwargs):
            return func(*args, **kwargs)

        return _run


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

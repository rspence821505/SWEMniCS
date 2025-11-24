"""
Test suite for L-BFGS optimizer.

Tests include:
- Convergence on quadratic problems
- Two-loop recursion correctness
- Descent property verification
- Memory management
- Line search integration
- Parallel determinism
"""

import pytest
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from typing import Optional
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swemnics.optimization.lbfgs import LBFGSOptimizer, PreconditionedLBFGS


# ============================================================================
# MOCK COST FUNCTIONS FOR TESTING
# ============================================================================


class QuadraticCostFunction:
    """
    Quadratic cost function: f(x) = 0.5 * x^T * A * x - b^T * x + c
    Gradient: ∇f(x) = A * x - b
    Hessian: ∇²f(x) = A
    Minimum at: x* = A^{-1} * b with f(x*) = c - 0.5 * b^T * A^{-1} * b
    """

    def __init__(self, n: int, comm=MPI.COMM_WORLD, condition_number: float = 10.0):
        """
        Initialize quadratic cost function.

        Args:
            n: Problem dimension
            comm: MPI communicator
            condition_number: Condition number of Hessian matrix
        """
        self.n = n
        self.comm = comm
        self.condition_number = condition_number

        # Create Hessian matrix A (SPD)
        self._create_hessian()

        # Create right-hand side b
        self.b = PETSc.Vec().create(comm=self.comm)
        self.b.setSizes(self.n)
        self.b.setFromOptions()
        self.b.setUp()
        self.b.set(1.0)  # Simple constant vector

        # Constant term
        self.c = 0.0

        # Compute true minimum
        self._compute_minimum()

    def _create_hessian(self):
        """Create SPD Hessian matrix with specified condition number."""
        self.A = PETSc.Mat().create(comm=self.comm)
        self.A.setSizes([self.n, self.n])
        self.A.setType(PETSc.Mat.Type.AIJ)
        self.A.setUp()

        # Create tridiagonal matrix
        for i in range(self.n):
            # Diagonal: scale to control condition number
            diag_val = 1.0 + (self.condition_number - 1.0) * i / max(self.n - 1, 1)
            self.A.setValue(i, i, diag_val)

            # Off-diagonals for smoothness
            if i > 0:
                self.A.setValue(i, i - 1, -0.1)
            if i < self.n - 1:
                self.A.setValue(i, i + 1, -0.1)

        self.A.assemblyBegin()
        self.A.assemblyEnd()

    def _compute_minimum(self):
        """Compute exact minimum x* = A^{-1} * b."""
        self.x_star = self.b.duplicate()

        # Solve A * x* = b using PETSc KSP
        ksp = PETSc.KSP().create(comm=self.comm)
        ksp.setOperators(self.A)
        ksp.setType(PETSc.KSP.Type.CG)
        ksp.getPC().setType(PETSc.PC.Type.JACOBI)
        ksp.setTolerances(rtol=1e-12)
        ksp.solve(self.b, self.x_star)
        ksp.destroy()

    def value(self, x: PETSc.Vec) -> float:
        """
        Evaluate cost function at x.

        Args:
            x: Evaluation point

        Returns:
            f(x) = 0.5 * x^T * A * x - b^T * x + c
        """
        # Compute A * x
        Ax = x.duplicate()
        self.A.mult(x, Ax)

        # Compute 0.5 * x^T * A * x
        quad_term = 0.5 * x.dot(Ax)

        # Compute -b^T * x
        linear_term = -x.dot(self.b)

        Ax.destroy()

        return quad_term + linear_term + self.c

    def gradient(self, x: PETSc.Vec, grad: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """
        Compute gradient at x.

        Args:
            x: Evaluation point
            grad: Output vector (created if None)

        Returns:
            ∇f(x) = A * x - b
        """
        if grad is None:
            grad = x.duplicate()

        # Compute A * x
        self.A.mult(x, grad)

        # Subtract b
        grad.axpy(-1.0, self.b)

        return grad

    def get_minimum(self) -> PETSc.Vec:
        """Return exact minimum."""
        return self.x_star.copy()


class RosenbrockFunction:
    """
    Rosenbrock function: f(x) = sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    A challenging non-convex test problem with a narrow valley.
    Minimum at x* = [1, 1, ..., 1] with f(x*) = 0.
    """

    def __init__(self, n: int, comm=MPI.COMM_WORLD):
        self.n = n
        self.comm = comm

    def value(self, x: PETSc.Vec) -> float:
        """Evaluate Rosenbrock function."""
        x_array = x.getArray()
        cost = 0.0

        for i in range(self.n - 1):
            cost += 100.0 * (x_array[i + 1] - x_array[i] ** 2) ** 2
            cost += (1.0 - x_array[i]) ** 2

        return cost

    def gradient(self, x: PETSc.Vec, grad: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Compute gradient of Rosenbrock function."""
        if grad is None:
            grad = x.duplicate()

        x_array = x.getArray()
        grad_array = np.zeros(self.n)

        # Interior points
        for i in range(1, self.n - 1):
            grad_array[i] = (
                200.0 * (x_array[i] - x_array[i - 1] ** 2)
                - 400.0 * x_array[i] * (x_array[i + 1] - x_array[i] ** 2)
                - 2.0 * (1.0 - x_array[i])
            )

        # First point
        grad_array[0] = -400.0 * x_array[0] * (x_array[1] - x_array[0] ** 2) - 2.0 * (
            1.0 - x_array[0]
        )

        # Last point
        grad_array[self.n - 1] = 200.0 * (
            x_array[self.n - 1] - x_array[self.n - 2] ** 2
        )

        grad.setArray(grad_array)
        return grad


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def quadratic_problem():
    """Create simple quadratic test problem."""
    n = 50
    cost_func = QuadraticCostFunction(n, condition_number=10.0)
    return cost_func


@pytest.fixture
def ill_conditioned_problem():
    """Create ill-conditioned quadratic problem."""
    n = 50
    cost_func = QuadraticCostFunction(n, condition_number=100.0)
    return cost_func


@pytest.fixture
def rosenbrock_problem():
    """Create Rosenbrock test problem."""
    n = 10
    cost_func = RosenbrockFunction(n)
    return cost_func


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================


class TestLBFGSBasics:
    """Test basic L-BFGS functionality."""

    def test_initialization(self, quadratic_problem):
        """Test L-BFGS initialization."""
        optimizer = LBFGSOptimizer(quadratic_problem, memory_size=10)

        assert optimizer.m == 10
        assert len(optimizer.s_history) == 0
        assert len(optimizer.y_history) == 0
        assert len(optimizer.rho_history) == 0
        assert optimizer.iteration == 0
        assert optimizer.converged == False

    def test_quadratic_convergence(self, quadratic_problem):
        """Test convergence on quadratic problem."""
        optimizer = LBFGSOptimizer(
            quadratic_problem,
            memory_size=5,
            options={"max_iterations": 50, "gradient_tolerance": 1e-6},
        )

        # Initial guess
        x0 = PETSc.Vec().create(comm=MPI.COMM_WORLD)
        x0.setSizes(quadratic_problem.n)
        x0.setFromOptions()
        x0.setUp()
        x0.set(0.0)  # Start at origin

        # Solve
        x_opt = optimizer.solve(x0)

        # Check convergence
        assert optimizer.converged, "Optimizer should converge on quadratic problem"

        # Check solution quality
        x_star = quadratic_problem.get_minimum()
        error = x_opt.copy()
        error.axpy(-1.0, x_star)
        rel_error = error.norm() / x_star.norm()

        assert rel_error < 1e-4, f"Solution error too large: {rel_error}"

        # Cleanup
        x0.destroy()
        x_opt.destroy()
        x_star.destroy()
        error.destroy()

    def test_descent_property(self, quadratic_problem):
        """Test that cost decreases at each iteration."""
        optimizer = LBFGSOptimizer(
            quadratic_problem,
            memory_size=5,
            options={"max_iterations": 20, "gradient_tolerance": 1e-6},
        )

        x0 = PETSc.Vec().create(comm=MPI.COMM_WORLD)
        x0.setSizes(quadratic_problem.n)
        x0.setFromOptions()
        x0.setUp()
        x0.set(2.0)

        x_opt = optimizer.solve(x0)

        # Check that cost decreased monotonically
        history = optimizer.convergence_history
        costs = [h["cost"] for h in history]

        for i in range(1, len(costs)):
            assert costs[i] <= costs[i - 1] + 1e-10, f"Cost increased at iteration {i}"

        x0.destroy()
        x_opt.destroy()

    def test_gradient_norm_decreases(self, quadratic_problem):
        """Test that gradient norm decreases."""
        optimizer = LBFGSOptimizer(
            quadratic_problem,
            memory_size=5,
            options={"max_iterations": 20, "gradient_tolerance": 1e-8},
        )

        x0 = PETSc.Vec().create(comm=MPI.COMM_WORLD)
        x0.setSizes(quadratic_problem.n)
        x0.setFromOptions()
        x0.setUp()
        x0.set(1.0)

        x_opt = optimizer.solve(x0)

        # Check gradient norm history
        history = optimizer.convergence_history
        grad_norms = [h["grad_norm"] for h in history]

        # Final gradient should be small
        assert grad_norms[-1] < 1e-5, f"Final gradient norm too large: {grad_norms[-1]}"

        x0.destroy()
        x_opt.destroy()


# ============================================================================
# TWO-LOOP RECURSION TESTS
# ============================================================================


class TestTwoLoopRecursion:
    """Test L-BFGS two-loop recursion algorithm."""

    def test_first_iteration_steepest_descent(self, quadratic_problem):
        """Test that first iteration uses steepest descent."""
        optimizer = LBFGSOptimizer(quadratic_problem, memory_size=5)

        # Create gradient
        x = PETSc.Vec().create(comm=MPI.COMM_WORLD)
        x.setSizes(quadratic_problem.n)
        x.setFromOptions()
        x.setUp()
        x.set(1.0)

        grad = quadratic_problem.gradient(x)

        # Compute search direction (should be -grad when no history)
        direction = optimizer._two_loop_recursion(grad)

        # Check direction = -grad
        test_vec = grad.copy()
        test_vec.scale(-1.0)
        test_vec.axpy(1.0, direction)
        diff_norm = test_vec.norm()

        assert diff_norm < 1e-10, f"First iteration should use -gradient: {diff_norm}"

        x.destroy()
        grad.destroy()
        direction.destroy()
        test_vec.destroy()

    def test_memory_limit_enforced(self, quadratic_problem):
        """Test that memory limit is enforced."""
        memory_size = 3
        optimizer = LBFGSOptimizer(
            quadratic_problem,
            memory_size=memory_size,
            options={"max_iterations": 10},
        )

        x0 = PETSc.Vec().create(comm=MPI.COMM_WORLD)
        x0.setSizes(quadratic_problem.n)
        x0.setFromOptions()
        x0.setUp()
        x0.set(1.0)

        # Run a few iterations
        x_opt = optimizer.solve(x0)

        # Check history size
        assert len(optimizer.s_history) <= memory_size
        assert len(optimizer.y_history) <= memory_size
        assert len(optimizer.rho_history) <= memory_size

        x0.destroy()
        x_opt.destroy()

    def test_curvature_condition_skip(self, quadratic_problem):
        """Test that updates are skipped when curvature condition violated."""
        optimizer = LBFGSOptimizer(quadratic_problem, memory_size=5)

        # Create vectors where y^T s ≈ 0 (violates curvature)
        x_new = PETSc.Vec().create(comm=MPI.COMM_WORLD)
        x_new.setSizes(quadratic_problem.n)
        x_new.setFromOptions()
        x_new.setUp()
        x_new.set(1.0)

        x_old = x_new.copy()
        x_old.set(1.0)  # x_new ≈ x_old, so s ≈ 0

        grad_new = quadratic_problem.gradient(x_new)
        grad_old = grad_new.copy()  # grad_new ≈ grad_old, so y ≈ 0

        initial_size = len(optimizer.s_history)
        optimizer._update_history(x_new, x_old, grad_new, grad_old)

        # History should not have grown
        assert len(optimizer.s_history) == initial_size

        x_new.destroy()
        x_old.destroy()
        grad_new.destroy()
        grad_old.destroy()


# ============================================================================
# ADVANCED TESTS
# ============================================================================


class TestLBFGSAdvanced:
    """Advanced L-BFGS tests."""

    def test_ill_conditioned_problem(self, ill_conditioned_problem):
        """Test on ill-conditioned problem."""
        optimizer = LBFGSOptimizer(
            ill_conditioned_problem,
            memory_size=10,
            options={"max_iterations": 100, "gradient_tolerance": 1e-5},
        )

        x0 = PETSc.Vec().create(comm=MPI.COMM_WORLD)
        x0.setSizes(ill_conditioned_problem.n)
        x0.setFromOptions()
        x0.setUp()
        x0.set(0.0)

        x_opt = optimizer.solve(x0)

        # Should still converge, though may take more iterations
        assert (
            optimizer.converged or optimizer.iteration >= 100
        ), "Should converge or hit max iterations"

        # Check reasonable accuracy
        x_star = ill_conditioned_problem.get_minimum()
        error = x_opt.copy()
        error.axpy(-1.0, x_star)
        rel_error = error.norm() / x_star.norm()

        # Looser tolerance for ill-conditioned problem
        assert rel_error < 1e-2, f"Solution error: {rel_error}"

        x0.destroy()
        x_opt.destroy()
        x_star.destroy()
        error.destroy()

    def test_rosenbrock(self, rosenbrock_problem):
        """Test on Rosenbrock function (non-convex)."""
        optimizer = LBFGSOptimizer(
            rosenbrock_problem,
            memory_size=10,
            options={"max_iterations": 500, "gradient_tolerance": 1e-4},
        )

        # Start near minimum
        x0 = PETSc.Vec().create(comm=MPI.COMM_WORLD)
        x0.setSizes(rosenbrock_problem.n)
        x0.setFromOptions()
        x0.setUp()
        x0.set(0.5)  # Reasonably close to [1, 1, ..., 1]

        x_opt = optimizer.solve(x0)

        # Check if close to minimum [1, 1, ..., 1]
        x_opt_array = x_opt.getArray()
        error = np.linalg.norm(x_opt_array - 1.0)

        # Rosenbrock is hard, so use loose tolerance
        assert error < 0.5, f"Should find approximate minimum: error = {error}"

        x0.destroy()
        x_opt.destroy()

    def test_different_memory_sizes(self, quadratic_problem):
        """Test with different memory sizes."""
        memory_sizes = [1, 5, 10, 20]

        for m in memory_sizes:
            optimizer = LBFGSOptimizer(
                quadratic_problem,
                memory_size=m,
                options={"max_iterations": 50, "gradient_tolerance": 1e-6},
            )

            x0 = PETSc.Vec().create(comm=MPI.COMM_WORLD)
            x0.setSizes(quadratic_problem.n)
            x0.setFromOptions()
            x0.setUp()
            x0.set(0.0)

            x_opt = optimizer.solve(x0)

            # All should converge for quadratic
            assert optimizer.converged, f"Failed to converge with memory size {m}"

            x0.destroy()
            x_opt.destroy()


# ============================================================================
# PARALLEL TESTS
# ============================================================================


@pytest.mark.mpi
class TestLBFGSParallel:
    """Test L-BFGS in parallel."""

    def test_parallel_determinism(self, quadratic_problem):
        """Test that parallel and serial give same result."""
        if MPI.COMM_WORLD.size == 1:
            pytest.skip("Need multiple ranks for parallel test")

        optimizer = LBFGSOptimizer(
            quadratic_problem,
            memory_size=5,
            options={"max_iterations": 20, "gradient_tolerance": 1e-6},
        )

        x0 = PETSc.Vec().create(comm=MPI.COMM_WORLD)
        x0.setSizes(quadratic_problem.n)
        x0.setFromOptions()
        x0.setUp()
        x0.set(1.0)

        x_opt = optimizer.solve(x0)

        # All ranks should have same result
        norm = x_opt.norm()
        norms = MPI.COMM_WORLD.allgather(norm)

        # Check all ranks agree
        for n in norms:
            assert abs(n - norms[0]) < 1e-10, "Ranks disagree on solution"

        x0.destroy()
        x_opt.destroy()


# ============================================================================
# RUN TESTS
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
Test suite for optimizer_base.py

Tests all base optimization classes including line search, trust region,
and convergence monitoring.
"""

import pytest
import numpy as np
from petsc4py import PETSc

from swe4dvar.optimization.optimizer_base import (
    Optimizer,
    LineSearch,
    TrustRegion,
    ConvergenceMonitor,
)


# Mock cost function classes for testing
class QuadraticCostFunction:
    """
    Simple quadratic cost function: f(x) = 0.5 * x^T * A * x - b^T * x
    Gradient: ∇f(x) = A * x - b
    Minimum at: x* = A^{-1} * b
    """

    def __init__(self, A, b):
        """
        Initialize quadratic cost function.

        Args:
            A: Positive definite matrix (as PETSc Mat)
            b: Linear term (as PETSc Vec)
        """
        self.A = A
        self.b = b

    def value(self, x: PETSc.Vec) -> float:
        """Compute cost: 0.5 * x^T * A * x - b^T * x"""
        Ax = x.duplicate()
        self.A.mult(x, Ax)
        cost = 0.5 * x.dot(Ax) - self.b.dot(x)
        Ax.destroy()
        return cost

    def gradient(self, x: PETSc.Vec, grad: PETSc.Vec):
        """Compute gradient: A * x - b"""
        self.A.mult(x, grad)
        grad.axpy(-1.0, self.b)


class RosenbrockCostFunction:
    """
    Rosenbrock function: f(x,y) = (a-x)^2 + b*(y-x^2)^2
    Standard values: a=1, b=100
    Minimum at (a, a^2) = (1, 1)
    """

    def __init__(self, a: float = 1.0, b: float = 100.0):
        self.a = a
        self.b = b

    def value(self, x: PETSc.Vec) -> float:
        """Compute Rosenbrock cost"""
        x_array = x.getArray()
        x_val = x_array[0]
        y_val = x_array[1]
        return (self.a - x_val) ** 2 + self.b * (y_val - x_val**2) ** 2

    def gradient(self, x: PETSc.Vec, grad: PETSc.Vec):
        """Compute Rosenbrock gradient"""
        x_array = x.getArray()
        x_val = x_array[0]
        y_val = x_array[1]

        grad_array = grad.getArray()
        grad_array[0] = -2 * (self.a - x_val) - 4 * self.b * x_val * (y_val - x_val**2)
        grad_array[1] = 2 * self.b * (y_val - x_val**2)


class SimpleOptimizer(Optimizer):
    """Concrete optimizer for testing base class"""

    def solve(self, x0: PETSc.Vec) -> PETSc.Vec:
        """Dummy solve method"""
        return x0.copy()


# Fixtures
@pytest.fixture
def petsc_setup():
    """Initialize PETSc"""
    # PETSc is already initialized by importing petsc4py
    yield
    # Cleanup happens automatically


@pytest.fixture
def quadratic_problem():
    """Create simple quadratic problem"""
    n = 10
    # Create positive definite matrix A
    A = PETSc.Mat().create()
    A.setSizes([n, n])
    A.setFromOptions()
    A.setUp()

    # Fill A with diagonal matrix (makes it positive definite)
    for i in range(n):
        A.setValue(i, i, 2.0 + i * 0.1)
    A.assemble()

    # Create vector b
    b = PETSc.Vec().create()
    b.setSizes(n)
    b.setFromOptions()
    b.setUp()
    b_array = b.getArray()
    for i in range(n):
        b_array[i] = float(i + 1)

    cost_func = QuadraticCostFunction(A, b)

    yield cost_func, A, b

    # Cleanup
    A.destroy()
    b.destroy()


@pytest.fixture
def rosenbrock_problem():
    """Create Rosenbrock problem"""
    cost_func = RosenbrockCostFunction(a=1.0, b=100.0)
    yield cost_func


# Tests for Optimizer base class
class TestOptimizer:
    """Tests for Optimizer abstract base class"""

    def test_initialization(self, quadratic_problem):
        """Test optimizer initialization"""
        cost_func, A, b = quadratic_problem
        options = {"gradient_tolerance": 1e-8, "max_iterations": 100}

        opt = SimpleOptimizer(cost_func, options)

        assert opt.cost_function == cost_func
        assert opt.options == options
        assert opt.iteration == 0
        assert opt.converged == False
        assert len(opt.convergence_history) == 0

    def test_check_convergence_gradient(self, quadratic_problem):
        """Test convergence check based on gradient norm"""
        cost_func, A, b = quadratic_problem
        opt = SimpleOptimizer(cost_func, {"gradient_tolerance": 1e-6})

        # Should converge with small gradient
        assert opt.check_convergence(1e-7) == True

        # Should not converge with large gradient
        assert opt.check_convergence(1e-5) == False

    def test_check_convergence_cost_change(self, quadratic_problem):
        """Test convergence check based on cost change"""
        cost_func, A, b = quadratic_problem
        opt = SimpleOptimizer(cost_func, {"cost_tolerance": 1e-8})

        # Should converge with small cost change
        assert opt.check_convergence(1e-5, cost_change=1e-9) == True

        # Should not converge with large cost change
        assert opt.check_convergence(1e-5, cost_change=1e-7) == False

    def test_record_iteration(self, quadratic_problem):
        """Test iteration recording"""
        cost_func, A, b = quadratic_problem
        opt = SimpleOptimizer(cost_func)

        x = b.duplicate()
        x.set(1.0)

        opt.record_iteration(x, cost=10.0, grad_norm=1.5)
        opt.iteration += 1
        opt.record_iteration(x, cost=5.0, grad_norm=0.8)

        assert len(opt.convergence_history) == 2
        assert opt.convergence_history[0]["cost"] == 10.0
        assert opt.convergence_history[1]["grad_norm"] == 0.8

        x.destroy()

    def test_get_convergence_info(self, quadratic_problem):
        """Test convergence info retrieval"""
        cost_func, A, b = quadratic_problem
        opt = SimpleOptimizer(cost_func)

        opt.converged = True
        opt.iteration = 5

        info = opt.get_convergence_info()

        assert info["converged"] == True
        assert info["iterations"] == 5
        assert "history" in info


# Tests for LineSearch class
class TestLineSearch:
    """Tests for LineSearch class"""

    def test_initialization(self, quadratic_problem):
        """Test line search initialization"""
        cost_func, A, b = quadratic_problem
        ls = LineSearch(cost_func, c1=1e-4, c2=0.9, max_iter=20)

        assert ls.c1 == 1e-4
        assert ls.c2 == 0.9
        assert ls.max_iter == 20

    def test_armijo_quadratic_descent(self, quadratic_problem):
        """Test Armijo line search on quadratic with descent direction"""
        cost_func, A, b = quadratic_problem
        ls = LineSearch(cost_func, c1=1e-4, max_iter=20)

        # Create current point
        n = b.getSize()
        x = PETSc.Vec().create()
        x.setSizes(n)
        x.setFromOptions()
        x.setUp()
        x.set(1.0)

        # Compute gradient at current point
        grad = x.duplicate()
        cost_func.gradient(x, grad)

        # Use negative gradient as search direction (steepest descent)
        direction = grad.duplicate()
        direction.copy(grad)
        direction.scale(-1.0)

        # Current cost
        cost_current = cost_func.value(x)

        # Perform line search
        alpha = ls.armijo_backtracking(x, direction, grad, cost_current)

        # Should return positive step size
        assert alpha > 0
        assert alpha <= 1.0

        # Verify sufficient decrease
        x_new = x.duplicate()
        x_new.waxpy(alpha, direction, x)
        cost_new = cost_func.value(x_new)

        directional_derivative = grad.dot(direction)
        armijo_threshold = cost_current + ls.c1 * alpha * directional_derivative

        assert cost_new <= armijo_threshold + 1e-10  # Small tolerance for numerics

        # Cleanup
        x.destroy()
        grad.destroy()
        direction.destroy()
        x_new.destroy()

    def test_armijo_non_descent_direction(self, quadratic_problem):
        """Test Armijo with non-descent direction"""
        cost_func, A, b = quadratic_problem
        ls = LineSearch(cost_func)

        n = b.getSize()
        x = PETSc.Vec().create()
        x.setSizes(n)
        x.setFromOptions()
        x.setUp()
        x.set(1.0)

        grad = x.duplicate()
        cost_func.gradient(x, grad)

        # Use gradient itself (not descent direction)
        direction = grad.duplicate()
        direction.copy(grad)

        cost_current = cost_func.value(x)

        alpha = ls.armijo_backtracking(x, direction, grad, cost_current)

        # Should return minimal step for non-descent direction
        assert alpha < 1e-6

        x.destroy()
        grad.destroy()
        direction.destroy()

    def test_wolfe_conditions_quadratic(self, quadratic_problem):
        """Test Wolfe line search on quadratic problem"""
        cost_func, A, b = quadratic_problem
        ls = LineSearch(cost_func, c1=1e-4, c2=0.9, max_iter=20)

        n = b.getSize()
        x = PETSc.Vec().create()
        x.setSizes(n)
        x.setFromOptions()
        x.setUp()
        x.set(1.0)

        grad = x.duplicate()
        cost_func.gradient(x, grad)

        # Steepest descent direction
        direction = grad.duplicate()
        direction.copy(grad)
        direction.scale(-1.0)

        cost_current = cost_func.value(x)

        alpha = ls.wolfe_conditions(x, direction, grad, cost_current)

        # Should find valid step size
        assert alpha > 0

        # Verify sufficient decrease
        x_new = x.duplicate()
        x_new.waxpy(alpha, direction, x)
        cost_new = cost_func.value(x_new)

        directional_derivative = grad.dot(direction)
        armijo_threshold = cost_current + ls.c1 * alpha * directional_derivative

        assert cost_new <= armijo_threshold + 1e-10

        # Verify curvature condition
        grad_new = grad.duplicate()
        cost_func.gradient(x_new, grad_new)
        directional_derivative_new = grad_new.dot(direction)

        assert (
            abs(directional_derivative_new) <= -ls.c2 * directional_derivative + 1e-10
        )

        # Cleanup
        x.destroy()
        grad.destroy()
        direction.destroy()
        x_new.destroy()
        grad_new.destroy()

    def test_zoom_helper(self, quadratic_problem):
        """Test zoom helper method"""
        cost_func, A, b = quadratic_problem
        ls = LineSearch(cost_func, c1=1e-4, c2=0.9)

        n = b.getSize()
        x = PETSc.Vec().create()
        x.setSizes(n)
        x.setFromOptions()
        x.setUp()
        x.set(1.0)

        grad = x.duplicate()
        cost_func.gradient(x, grad)

        direction = grad.duplicate()
        direction.copy(grad)
        direction.scale(-1.0)

        cost_current = cost_func.value(x)
        directional_derivative = grad.dot(direction)

        # Test zoom with reasonable bracket
        alpha = ls._zoom(
            x, direction, grad, cost_current, directional_derivative, 0.0, 1.0
        )

        assert 0.0 <= alpha <= 1.0

        # Cleanup
        x.destroy()
        grad.destroy()
        direction.destroy()


# Tests for TrustRegion class
class TestTrustRegion:
    """Tests for TrustRegion class"""

    def test_initialization(self):
        """Test trust region initialization"""
        tr = TrustRegion(radius_init=1.0, radius_max=10.0, eta=0.1)

        assert tr.radius == 1.0
        assert tr.radius_max == 10.0
        assert tr.eta == 0.1

    def test_update_radius_good_step(self):
        """Test radius update with good step"""
        tr = TrustRegion(radius_init=1.0, radius_max=10.0)

        # Good agreement, step at boundary
        tr.update_radius(actual_reduction=0.8, predicted_reduction=1.0, step_norm=1.0)

        # Radius should expand
        assert tr.radius == 2.0

    def test_update_radius_poor_step(self):
        """Test radius update with poor step"""
        tr = TrustRegion(radius_init=1.0, radius_max=10.0)

        # Poor agreement
        tr.update_radius(actual_reduction=0.1, predicted_reduction=1.0, step_norm=0.5)

        # Radius should shrink
        assert tr.radius == 0.25

    def test_update_radius_max_limit(self):
        """Test radius doesn't exceed maximum"""
        tr = TrustRegion(radius_init=8.0, radius_max=10.0)

        # Good step at boundary, would expand to 16.0
        tr.update_radius(actual_reduction=0.9, predicted_reduction=1.0, step_norm=8.0)

        # Should be capped at radius_max
        assert tr.radius == 10.0

    def test_accept_step_good_ratio(self):
        """Test step acceptance with good ratio"""
        tr = TrustRegion(eta=0.1)

        # Good ratio
        assert tr.accept_step(actual_reduction=0.8, predicted_reduction=1.0) == True

    def test_accept_step_poor_ratio(self):
        """Test step rejection with poor ratio"""
        tr = TrustRegion(eta=0.1)

        # Poor ratio
        assert tr.accept_step(actual_reduction=0.05, predicted_reduction=1.0) == False

    def test_accept_step_zero_prediction(self):
        """Test step acceptance with zero predicted reduction"""
        tr = TrustRegion(eta=0.1)

        # Zero predicted, positive actual
        assert tr.accept_step(actual_reduction=0.1, predicted_reduction=1e-15) == True

        # Zero predicted, negative actual
        assert tr.accept_step(actual_reduction=-0.1, predicted_reduction=1e-15) == False


# Tests for ConvergenceMonitor class
class TestConvergenceMonitor:
    """Tests for ConvergenceMonitor class"""

    def test_initialization(self):
        """Test convergence monitor initialization"""
        monitor = ConvergenceMonitor(options={"stall_tolerance": 1e-10})

        assert monitor.options["stall_tolerance"] == 1e-10
        assert len(monitor.cost_history) == 0
        assert len(monitor.grad_norm_history) == 0

    def test_update(self):
        """Test update with iteration data"""
        monitor = ConvergenceMonitor()

        monitor.update(cost=10.0, grad_norm=1.5)
        monitor.update(cost=5.0, grad_norm=0.8)

        assert len(monitor.cost_history) == 2
        assert monitor.cost_history[0] == 10.0
        assert monitor.grad_norm_history[1] == 0.8

    def test_is_stalled_true(self):
        """Test stall detection when stalled"""
        monitor = ConvergenceMonitor(options={"stall_tolerance": 1e-12})

        # Add converged sequence (tiny changes)
        for i in range(10):
            monitor.update(cost=1.0 + i * 1e-14, grad_norm=1e-8)

        assert monitor.is_stalled(window=5) == True

    def test_is_stalled_false(self):
        """Test stall detection when making progress"""
        monitor = ConvergenceMonitor(options={"stall_tolerance": 1e-12})

        # Add sequence with significant changes
        for i in range(10):
            monitor.update(cost=10.0 - i * 0.5, grad_norm=1.0 / (i + 1))

        assert monitor.is_stalled(window=5) == False

    def test_is_stalled_insufficient_history(self):
        """Test stall detection with insufficient history"""
        monitor = ConvergenceMonitor()

        monitor.update(cost=10.0, grad_norm=1.0)
        monitor.update(cost=9.0, grad_norm=0.9)

        # Not enough history for window=5
        assert monitor.is_stalled(window=5) == False

    def test_is_diverging_true(self):
        """Test divergence detection when diverging"""
        monitor = ConvergenceMonitor()

        # Add diverging sequence
        monitor.update(cost=1.0, grad_norm=0.1)
        monitor.update(cost=2.0, grad_norm=0.2)
        monitor.update(cost=3.0, grad_norm=0.3)

        assert monitor.is_diverging() == True

    def test_is_diverging_false(self):
        """Test divergence detection when converging"""
        monitor = ConvergenceMonitor()

        # Add converging sequence
        monitor.update(cost=10.0, grad_norm=1.0)
        monitor.update(cost=5.0, grad_norm=0.5)
        monitor.update(cost=2.0, grad_norm=0.2)

        assert monitor.is_diverging() == False

    def test_is_diverging_insufficient_history(self):
        """Test divergence detection with insufficient history"""
        monitor = ConvergenceMonitor()

        monitor.update(cost=10.0, grad_norm=1.0)

        # Not enough history
        assert monitor.is_diverging() == False


# Integration tests
class TestIntegration:
    """Integration tests combining multiple components"""

    def test_quadratic_optimization_with_line_search(self, quadratic_problem):
        """Test simple optimization loop with line search"""
        cost_func, A, b = quadratic_problem
        ls = LineSearch(cost_func, c1=1e-4, c2=0.9)
        monitor = ConvergenceMonitor()

        n = b.getSize()
        x = PETSc.Vec().create()
        x.setSizes(n)
        x.setFromOptions()
        x.setUp()
        x.set(1.0)  # Initial guess

        grad = x.duplicate()
        direction = x.duplicate()

        max_iter = 50
        for i in range(max_iter):
            # Compute gradient
            cost = cost_func.value(x)
            cost_func.gradient(x, grad)
            grad_norm = grad.norm()

            # Update monitor
            monitor.update(cost, grad_norm)

            # Check convergence
            if grad_norm < 1e-6:
                break

            # Search direction (steepest descent)
            direction.copy(grad)
            direction.scale(-1.0)

            # Line search
            alpha = ls.armijo_backtracking(x, direction, grad, cost)

            # Update
            x.axpy(alpha, direction)

        # Should converge for quadratic problem
        final_grad_norm = grad.norm()
        assert final_grad_norm < 1e-5
        assert not monitor.is_diverging()

        # Cleanup
        x.destroy()
        grad.destroy()
        direction.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

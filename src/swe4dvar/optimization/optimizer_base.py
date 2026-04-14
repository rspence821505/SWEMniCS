"""
Base classes for optimization algorithms.

Defines common interface for all optimizers used in 4D-Var.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable
from petsc4py import PETSc
import numpy as np


class Optimizer(ABC):
    """
    Abstract base class for optimization algorithms.

    Defines standard interface for minimizing cost functions.
    """

    def __init__(self, cost_function, options: Optional[Dict] = None):
        """
        Initialize optimizer.

        Args:
            cost_function: Cost function to minimize (must have value/gradient methods)
            options: Optimizer-specific options
        """
        self.cost_function = cost_function
        self.options = options or {}

        # Convergence monitoring
        self.iteration = 0
        self.converged = False
        self.convergence_history = []
        self.iteration_callback = self.options.get("iteration_callback")

    @abstractmethod
    def solve(self, x0: PETSc.Vec) -> PETSc.Vec:
        """
        Minimize cost function starting from initial guess.

        Args:
            x0: Initial guess

        Returns:
            Optimal solution
        """
        pass

    def check_convergence(
        self, grad_norm: float, cost_change: Optional[float] = None
    ) -> bool:
        """
        Check if optimization has converged.

        Args:
            grad_norm: Current gradient norm
            cost_change: Change in cost function value

        Returns:
            True if converged
        """
        # Gradient tolerance
        grad_tol = self.options.get("gradient_tolerance", 1e-6)
        if grad_norm < grad_tol:
            return True

        # Cost change tolerance
        if cost_change is not None:
            cost_tol = self.options.get("cost_tolerance", 1e-8)
            if abs(cost_change) < cost_tol:
                return True

        return False

    def record_iteration(self, x: PETSc.Vec, cost: float, grad_norm: float):
        """
        Record iteration history.

        Args:
            x: Current iterate
            cost: Cost function value
            grad_norm: Gradient norm
        """
        entry = {"iteration": self.iteration, "cost": cost, "grad_norm": grad_norm}

        if self.iteration_callback is not None:
            extra = self.iteration_callback(x, self.iteration, cost, grad_norm)
            if isinstance(extra, dict):
                entry.update(extra)

        self.convergence_history.append(entry)

    def get_convergence_info(self) -> Dict:
        """
        Get convergence information.

        Returns:
            Dictionary with convergence data
        """
        return {
            "converged": self.converged,
            "iterations": self.iteration,
            "history": self.convergence_history,
        }


class LineSearch:
    """
    Line search for step size determination.

    Implements Armijo backtracking and Wolfe conditions.
    """

    def __init__(
        self, cost_function, c1: float = 1e-4, c2: float = 0.9, max_iter: int = 20
    ):
        """
        Initialize line search.

        Args:
            cost_function: Cost function with value/gradient methods
            c1: Armijo parameter (sufficient decrease)
            c2: Wolfe parameter (curvature condition)
            max_iter: Maximum line search iterations
        """
        self.cost_function = cost_function
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter

    def armijo_backtracking(
        self,
        x: PETSc.Vec,
        direction: PETSc.Vec,
        grad: PETSc.Vec,
        cost_current: float,
        alpha_init: float = 1.0,
        rho: float = 0.5,
    ) -> float:
        """
        Armijo backtracking line search.

        Finds step size α satisfying:
        f(x + α·p) ≤ f(x) + c₁·α·∇f(x)ᵀ·p

        Args:
            x: Current point
            direction: Search direction p
            grad: Current gradient
            cost_current: f(x)
            alpha_init: Initial step size
            rho: Backtracking factor

        Returns:
            Step size α
        """
        # Compute directional derivative: ∇f(x)ᵀ·p
        directional_derivative = grad.dot(direction)

        # If direction is not a descent direction, return small step
        if directional_derivative >= 0:
            return 1e-8

        alpha = alpha_init
        x_trial = x.duplicate()

        for i in range(self.max_iter):
            # Compute trial point: x_trial = x + α·p
            x_trial.waxpy(alpha, direction, x)  # x_trial = x + alpha * direction

            # Evaluate cost at trial point
            cost_trial = self.cost_function.value(x_trial)

            # Check Armijo condition: f(x+α·p) ≤ f(x) + c₁·α·∇fᵀ·p
            armijo_threshold = cost_current + self.c1 * alpha * directional_derivative

            if cost_trial <= armijo_threshold:
                # Sufficient decrease achieved
                x_trial.destroy()
                return alpha

            # Reduce step size
            alpha *= rho

        # If no suitable step found, return minimal step
        x_trial.destroy()
        return alpha

    def wolfe_conditions(
        self,
        x: PETSc.Vec,
        direction: PETSc.Vec,
        grad: PETSc.Vec,
        cost_current: float,
        alpha_init: float = 1.0,
    ) -> float:
        """
        Line search satisfying strong Wolfe conditions.

        Finds α satisfying both:
        1. Sufficient decrease: f(x+α·p) ≤ f(x) + c₁·α·∇fᵀ·p
        2. Curvature: |∇f(x+α·p)ᵀ·p| ≤ c₂·|∇f(x)ᵀ·p|

        Args:
            x: Current point
            direction: Search direction
            grad: Current gradient
            cost_current: Current cost
            alpha_init: Initial step size

        Returns:
            Step size α
        """
        # Compute directional derivative at current point
        directional_derivative = grad.dot(direction)

        # If not a descent direction, return small step
        if directional_derivative >= 0:
            return 1e-8

        # Initialize step size bounds
        alpha = alpha_init
        alpha_prev = 0.0
        cost_prev = cost_current

        x_trial = x.duplicate()
        grad_trial = grad.duplicate()

        for i in range(self.max_iter):
            # Compute trial point
            x_trial.waxpy(alpha, direction, x)

            # Evaluate cost and gradient at trial point
            cost_trial = self.cost_function.value(x_trial)
            grad_trial_new = self.cost_function.gradient(x_trial)
            grad_trial.copy(grad_trial_new)
            grad_trial_new.destroy()

            # Check sufficient decrease (Armijo condition)
            armijo_threshold = cost_current + self.c1 * alpha * directional_derivative

            # Sufficient decrease violated or cost increased from previous step
            if cost_trial > armijo_threshold or (i > 0 and cost_trial >= cost_prev):
                # Zoom in the interval [alpha_prev, alpha]
                alpha_result = self._zoom(
                    x,
                    direction,
                    grad,
                    cost_current,
                    directional_derivative,
                    alpha_prev,
                    alpha,
                )
                x_trial.destroy()
                grad_trial.destroy()
                return alpha_result

            # Check curvature condition
            directional_derivative_trial = grad_trial.dot(direction)

            # Strong Wolfe curvature condition satisfied
            if abs(directional_derivative_trial) <= -self.c2 * directional_derivative:
                x_trial.destroy()
                grad_trial.destroy()
                return alpha

            # Directional derivative positive, zoom in [alpha, alpha_prev]
            if directional_derivative_trial >= 0:
                alpha_result = self._zoom(
                    x,
                    direction,
                    grad,
                    cost_current,
                    directional_derivative,
                    alpha,
                    alpha_prev,
                )
                x_trial.destroy()
                grad_trial.destroy()
                return alpha_result

            # Update for next iteration
            alpha_prev = alpha
            cost_prev = cost_trial
            alpha *= 2.0  # Increase step size

        # Max iterations reached, return current alpha
        x_trial.destroy()
        grad_trial.destroy()
        return alpha

    def _zoom(
        self,
        x: PETSc.Vec,
        direction: PETSc.Vec,
        grad: PETSc.Vec,
        cost_current: float,
        directional_derivative: float,
        alpha_lo: float,
        alpha_hi: float,
    ) -> float:
        """
        Zoom phase for Wolfe line search.

        Refines step size within bracket [alpha_lo, alpha_hi].

        Args:
            x: Current point
            direction: Search direction
            grad: Current gradient
            cost_current: Current cost
            directional_derivative: ∇f(x)ᵀ·p
            alpha_lo: Lower bound of bracket
            alpha_hi: Upper bound of bracket

        Returns:
            Step size satisfying Wolfe conditions
        """
        x_trial = x.duplicate()
        grad_trial = grad.duplicate()

        for i in range(self.max_iter):
            # Interpolate to find trial step (bisection for simplicity)
            alpha = 0.5 * (alpha_lo + alpha_hi)

            # Evaluate at trial point
            x_trial.waxpy(alpha, direction, x)
            cost_trial = self.cost_function.value(x_trial)
            grad_trial_new = self.cost_function.gradient(x_trial)
            grad_trial.copy(grad_trial_new)
            grad_trial_new.destroy()

            # Evaluate cost at alpha_lo for comparison
            x_lo = x.duplicate()
            x_lo.waxpy(alpha_lo, direction, x)
            cost_lo = self.cost_function.value(x_lo)
            x_lo.destroy()

            # Check sufficient decrease
            armijo_threshold = cost_current + self.c1 * alpha * directional_derivative

            if cost_trial > armijo_threshold or cost_trial >= cost_lo:
                # Sufficient decrease violated, narrow to [alpha_lo, alpha]
                alpha_hi = alpha
            else:
                # Check curvature condition
                directional_derivative_trial = grad_trial.dot(direction)

                if (
                    abs(directional_derivative_trial)
                    <= -self.c2 * directional_derivative
                ):
                    # Both conditions satisfied
                    x_trial.destroy()
                    grad_trial.destroy()
                    return alpha

                # Check sign of directional derivative
                if directional_derivative_trial * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo

                alpha_lo = alpha

        # Return best alpha found
        x_trial.destroy()
        grad_trial.destroy()
        return 0.5 * (alpha_lo + alpha_hi)


class TrustRegion:
    """
    Trust region framework for optimization.

    Alternative to line search for step size control.
    """

    def __init__(
        self, radius_init: float = 1.0, radius_max: float = 10.0, eta: float = 0.1
    ):
        """
        Initialize trust region.

        Args:
            radius_init: Initial trust region radius
            radius_max: Maximum trust region radius
            eta: Acceptance threshold for step
        """
        self.radius = radius_init
        self.radius_max = radius_max
        self.eta = eta

    def update_radius(
        self, actual_reduction: float, predicted_reduction: float, step_norm: float
    ):
        """
        Update trust region radius based on step quality.

        Args:
            actual_reduction: Actual decrease in cost
            predicted_reduction: Predicted decrease from model
            step_norm: Norm of computed step
        """
        # Ratio of actual to predicted reduction
        if abs(predicted_reduction) < 1e-14:
            ratio = 0.0
        else:
            ratio = actual_reduction / predicted_reduction

        # Update radius based on ratio
        if ratio < 0.25:
            # Poor agreement, shrink radius
            self.radius *= 0.25
        elif ratio > 0.75 and abs(step_norm - self.radius) < 1e-10:
            # Good agreement and step at boundary, expand radius
            self.radius = min(2.0 * self.radius, self.radius_max)

    def accept_step(self, actual_reduction: float, predicted_reduction: float) -> bool:
        """
        Decide whether to accept step.

        Args:
            actual_reduction: Actual cost reduction
            predicted_reduction: Predicted cost reduction

        Returns:
            True if step should be accepted
        """
        if abs(predicted_reduction) < 1e-14:
            return actual_reduction > 0

        ratio = actual_reduction / predicted_reduction
        return ratio > self.eta


class ConvergenceMonitor:
    """
    Monitor convergence diagnostics during optimization.

    Tracks various metrics and detects stalling/divergence.
    """

    def __init__(self, options: Optional[Dict] = None):
        """
        Initialize convergence monitor.

        Args:
            options: Monitoring options
        """
        self.options = options or {}

        # History
        self.cost_history = []
        self.grad_norm_history = []

    def update(self, cost: float, grad_norm: float):
        """
        Update with current iteration data.

        Args:
            cost: Cost function value
            grad_norm: Gradient norm
        """
        self.cost_history.append(cost)
        self.grad_norm_history.append(grad_norm)

    def is_stalled(self, window: int = 5) -> bool:
        """
        Check if optimization has stalled.

        Args:
            window: Number of iterations to check

        Returns:
            True if stalled
        """
        if len(self.cost_history) < window + 1:
            return False

        recent_costs = self.cost_history[-window - 1 :]
        cost_changes = [
            abs(recent_costs[i + 1] - recent_costs[i])
            for i in range(len(recent_costs) - 1)
        ]

        # Stalled if all changes are tiny
        stall_tol = self.options.get("stall_tolerance", 1e-12)
        return all(change < stall_tol for change in cost_changes)

    def is_diverging(self) -> bool:
        """
        Check if optimization is diverging.

        Returns:
            True if cost is increasing
        """
        if len(self.cost_history) < 3:
            return False

        # Check for sustained cost increase
        recent = self.cost_history[-3:]
        return recent[-1] > recent[-2] > recent[-3]

"""
Limited-memory BFGS (L-BFGS) optimizer.

Efficient quasi-Newton method for large-scale unconstrained optimization.
Uses two-loop recursion to compute search directions.
"""

from typing import List, Tuple, Optional
from petsc4py import PETSc
import numpy as np

from .optimizer_base import Optimizer, LineSearch


class LBFGSOptimizer(Optimizer):
    """
    Limited-memory BFGS optimizer.

    Approximates Hessian inverse using limited history of
    gradient differences. Memory efficient for large problems.
    """

    def __init__(
        self, cost_function, memory_size: int = 10, options: Optional[dict] = None
    ):
        """
        Initialize L-BFGS optimizer.

        Args:
            cost_function: Cost function to minimize
            memory_size: Number of correction pairs to store (m)
            options: Optimizer options
        """
        super().__init__(cost_function, options)

        self.m = memory_size  # Memory size

        # Storage for correction pairs (s_k, y_k)
        self.s_history: List[PETSc.Vec] = []  # x_{k+1} - x_k
        self.y_history: List[PETSc.Vec] = []  # ∇f_{k+1} - ∇f_k
        self.rho_history: List[float] = []  # 1 / (y_k^T s_k)

        # Line search
        self.line_search = LineSearch(cost_function)

    def solve(self, x0: PETSc.Vec) -> PETSc.Vec:
        """
        Minimize cost function using L-BFGS.

        Args:
            x0: Initial guess

        Returns:
            Optimal solution
        """
        # Initialize
        x = x0.copy()
        grad = self.cost_function.gradient(x)
        cost = self.cost_function.value(x)

        max_iter = self.options.get("max_iterations", 100)

        for k in range(max_iter):
            self.iteration = k

            # Check convergence
            grad_norm = grad.norm()
            if k > 0:
                cost_change = cost - cost_prev
                if self.check_convergence(grad_norm, cost_change):
                    self.converged = True
                    break

            # Record iteration
            self.record_iteration(x, cost, grad_norm)

            # Compute search direction via two-loop recursion
            direction = self._two_loop_recursion(grad)

            # Line search
            alpha = self.line_search.armijo_backtracking(x, direction, grad, cost)

            # Update
            x_new = x.copy()
            x_new.axpy(alpha, direction)

            grad_new = self.cost_function.gradient(x_new)
            cost_prev = cost
            cost = self.cost_function.value(x_new)

            # Store correction pair
            self._update_history(x_new, x, grad_new, grad)

            # Prepare for next iteration
            x = x_new
            grad = grad_new

        return x

    def _two_loop_recursion(self, grad: PETSc.Vec) -> PETSc.Vec:
        """
        Compute search direction using L-BFGS two-loop recursion.

        Implicitly applies H_k · ∇f where H_k ≈ (∇²f)^{-1}.

        Args:
            grad: Current gradient

        Returns:
            Search direction (negative for minimization)
        """
        q = grad.copy()

        # Number of stored corrections
        num_corrections = len(self.s_history)

        # First loop: backward
        alphas = []
        for i in range(num_corrections - 1, -1, -1):
            alpha_i = self.rho_history[i] * self.s_history[i].dot(q)
            alphas.insert(0, alpha_i)
            q.axpy(-alpha_i, self.y_history[i])

        # Apply initial Hessian approximation H_0 = γI
        if num_corrections > 0:
            # γ = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
            s_last = self.s_history[-1]
            y_last = self.y_history[-1]
            gamma = s_last.dot(y_last) / y_last.dot(y_last)
        else:
            gamma = 1.0

        r = q.copy()
        r.scale(gamma)

        # Second loop: forward
        for i in range(num_corrections):
            beta = self.rho_history[i] * self.y_history[i].dot(r)
            r.axpy(alphas[i] - beta, self.s_history[i])

        # Return negative for descent direction
        r.scale(-1.0)
        return r

    def _update_history(
        self,
        x_new: PETSc.Vec,
        x_old: PETSc.Vec,
        grad_new: PETSc.Vec,
        grad_old: PETSc.Vec,
    ):
        """
        Update correction pair history.

        Args:
            x_new: New iterate
            x_old: Previous iterate
            grad_new: New gradient
            grad_old: Previous gradient
        """
        # Compute s_k = x_{k+1} - x_k
        s = x_new.copy()
        s.axpy(-1.0, x_old)

        # Compute y_k = ∇f_{k+1} - ∇f_k
        y = grad_new.copy()
        y.axpy(-1.0, grad_old)

        # Compute ρ_k = 1 / (y_k^T s_k)
        y_dot_s = y.dot(s)
        if abs(y_dot_s) < 1e-14:
            # Skip update if curvature condition violated
            return

        rho = 1.0 / y_dot_s

        # Add to history
        self.s_history.append(s)
        self.y_history.append(y)
        self.rho_history.append(rho)

        # Enforce memory limit
        if len(self.s_history) > self.m:
            self.s_history.pop(0)
            self.y_history.pop(0)
            self.rho_history.pop(0)


class PreconditionedLBFGS(LBFGSOptimizer):
    """
    Preconditioned L-BFGS optimizer.

    Uses a preconditioner P in the initial Hessian approximation:
    H_0 = P instead of H_0 = γI
    """

    def __init__(
        self,
        cost_function,
        preconditioner,
        memory_size: int = 10,
        options: Optional[dict] = None,
    ):
        """
        Initialize preconditioned L-BFGS.

        Args:
            cost_function: Cost function
            preconditioner: Preconditioner operator
            memory_size: L-BFGS memory
            options: Optimizer options
        """
        super().__init__(cost_function, memory_size, options)
        self.preconditioner = preconditioner

    def _two_loop_recursion(self, grad: PETSc.Vec) -> PETSc.Vec:
        """
        Two-loop recursion with preconditioner.

        Replaces H_0 = γI with H_0 = P.
        """
        q = grad.copy()

        # First loop: backward
        num_corrections = len(self.s_history)
        alphas = []
        for i in range(num_corrections - 1, -1, -1):
            alpha_i = self.rho_history[i] * self.s_history[i].dot(q)
            alphas.insert(0, alpha_i)
            q.axpy(-alpha_i, self.y_history[i])

        # Apply preconditioner instead of γI
        r = self.preconditioner.apply(q)

        # Second loop: forward
        for i in range(num_corrections):
            beta = self.rho_history[i] * self.y_history[i].dot(r)
            r.axpy(alphas[i] - beta, self.s_history[i])

        # Return negative for descent direction
        r.scale(-1.0)
        return r


class BoundedLBFGS(LBFGSOptimizer):
    """
    L-BFGS with box constraints.

    Implements L-BFGS-B algorithm for bound-constrained optimization:
    min f(x)  s.t.  l ≤ x ≤ u
    """

    def __init__(
        self,
        cost_function,
        lower_bounds: Optional[PETSc.Vec] = None,
        upper_bounds: Optional[PETSc.Vec] = None,
        memory_size: int = 10,
        options: Optional[dict] = None,
    ):
        """
        Initialize L-BFGS-B.

        Args:
            cost_function: Cost function
            lower_bounds: Lower bounds l (None = -∞)
            upper_bounds: Upper bounds u (None = +∞)
            memory_size: L-BFGS memory
            options: Optimizer options
        """
        super().__init__(cost_function, memory_size, options)
        self.lower = lower_bounds
        self.upper = upper_bounds

    def solve(self, x0: PETSc.Vec) -> PETSc.Vec:
        """
        Minimize with box constraints.

        Uses projected gradient and Cauchy point algorithm.
        """
        # TODO: Implement L-BFGS-B algorithm
        # 1. Identify active set
        # 2. Compute Cauchy point
        # 3. Subspace minimization
        # 4. Line search with projection
        pass

    def _project_onto_bounds(self, x: PETSc.Vec) -> PETSc.Vec:
        """
        Project onto feasible region.

        Args:
            x: Input vector

        Returns:
            Projected vector
        """
        x_proj = x.copy()

        if self.lower is not None:
            # x = max(x, l)
            x_proj.pointwiseMax(x_proj, self.lower)

        if self.upper is not None:
            # x = min(x, u)
            x_proj.pointwiseMin(x_proj, self.upper)

        return x_proj

    def _compute_projected_gradient(self, x: PETSc.Vec, grad: PETSc.Vec) -> PETSc.Vec:
        """
        Compute projected gradient.

        Sets gradient components to zero at active bounds.

        Args:
            x: Current point
            grad: Gradient

        Returns:
            Projected gradient
        """
        # TODO: Implement projected gradient computation
        pass

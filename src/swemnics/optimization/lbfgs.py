"""
Limited-memory BFGS (L-BFGS) optimizer.

Efficient quasi-Newton method for large-scale unconstrained optimization.
Uses two-loop recursion to compute search directions.

References:
    Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for
    large scale optimization. Mathematical programming, 45(1-3), 503-528.
"""

from typing import List, Tuple, Optional
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI

from .optimizer_base import Optimizer, LineSearch


class LBFGSOptimizer(Optimizer):
    """
    Limited-memory BFGS optimizer.

    Approximates Hessian inverse using limited history of
    gradient differences. Memory efficient for large problems.

    The L-BFGS algorithm maintains m correction pairs (s_k, y_k):
        s_k = x_{k+1} - x_k  (step)
        y_k = ∇f_{k+1} - ∇f_k  (gradient difference)

    These are used to implicitly represent an approximation to H_k ≈ (∇²f)^{-1}
    via the two-loop recursion algorithm.
    """

    def __init__(
        self,
        cost_function,
        memory_size: int = 10,
        options: Optional[dict] = None,
    ):
        """
        Initialize L-BFGS optimizer.

        Args:
            cost_function: Cost function to minimize (must have value/gradient methods)
            memory_size: Number of correction pairs to store (m). Default: 10
            options: Optimizer options:
                - max_iterations: Maximum optimization iterations (default: 100)
                - gradient_tolerance: Convergence tolerance on gradient norm (default: 1e-6)
                - cost_tolerance: Convergence tolerance on cost change (default: 1e-8)
                - line_search_c1: Armijo parameter for line search (default: 1e-4)
                - line_search_c2: Wolfe parameter for line search (default: 0.9)
                - line_search_max_iter: Max line search iterations (default: 20)
                - verbose: Print iteration info (default: False)
        """
        super().__init__(cost_function, options)

        self.m = memory_size  # Memory size

        # Storage for correction pairs (s_k, y_k)
        self.s_history: List[PETSc.Vec] = []  # x_{k+1} - x_k
        self.y_history: List[PETSc.Vec] = []  # ∇f_{k+1} - ∇f_k
        self.rho_history: List[float] = []  # 1 / (y_k^T s_k)

        # Line search with customizable parameters
        c1 = self.options.get("line_search_c1", 1e-4)
        c2 = self.options.get("line_search_c2", 0.9)
        max_ls_iter = self.options.get("line_search_max_iter", 20)
        self.line_search = LineSearch(cost_function, c1=c1, c2=c2, max_iter=max_ls_iter)

        # Verbose output
        self.verbose = self.options.get("verbose", False)
        self.comm = MPI.COMM_WORLD

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

        if self.verbose and self.comm.rank == 0:
            print(f"\n{'=' * 70}")
            print(f"L-BFGS Optimization (memory_size={self.m})")
            print(f"{'=' * 70}")
            print(f"{'Iter':>6} {'Cost':>15} {'||grad||':>15} {'α':>12} {'ΔCost':>15}")
            print(f"{'-' * 70}")

        cost_prev = cost

        for k in range(max_iter):
            self.iteration = k

            # Check convergence
            grad_norm = grad.norm()

            if k > 0:
                cost_change = cost - cost_prev

                if self.verbose and self.comm.rank == 0:
                    print(
                        f"{k:6d} {cost:15.8e} {grad_norm:15.8e} {alpha:12.5e} {cost_change:15.8e}"
                    )

                if self.check_convergence(grad_norm, cost_change):
                    self.converged = True
                    if self.verbose and self.comm.rank == 0:
                        print(f"\nConverged after {k} iterations!")
                        print(f"Final cost: {cost:.8e}")
                        print(f"Final ||grad||: {grad_norm:.8e}")
                    break
            else:
                if self.verbose and self.comm.rank == 0:
                    print(
                        f"{k:6d} {cost:15.8e} {grad_norm:15.8e} {'--':>12} {'--':>15}"
                    )

            # Record iteration
            self.record_iteration(x, cost, grad_norm)

            # Compute search direction via two-loop recursion
            direction = self._two_loop_recursion(grad)

            # Verify descent direction
            directional_deriv = grad.dot(direction)
            if directional_deriv >= 0:
                if self.verbose and self.comm.rank == 0:
                    print(
                        f"Warning: Non-descent direction detected. Resetting to steepest descent."
                    )
                direction.copy(grad)
                direction.scale(-1.0)
                # Clear history to restart
                self._clear_history()

            # Line search
            alpha = self.line_search.armijo_backtracking(x, direction, grad, cost)

            # Check if line search failed
            if alpha < 1e-16:
                if self.verbose and self.comm.rank == 0:
                    print(f"Warning: Line search returned very small step. Stopping.")
                break

            # Update
            x_new = x.copy()
            x_new.axpy(alpha, direction)

            grad_new = self.cost_function.gradient(x_new)
            cost_prev = cost
            cost = self.cost_function.value(x_new)

            # Store correction pair
            self._update_history(x_new, x, grad_new, grad)

            # Prepare for next iteration
            x.destroy()  # Clean up old iterate
            grad.destroy()  # Clean up old gradient
            direction.destroy()  # Clean up direction
            x = x_new
            grad = grad_new

        if self.verbose and self.comm.rank == 0:
            print(f"{'=' * 70}\n")

        # Clean up
        grad.destroy()

        return x

    def _two_loop_recursion(self, grad: PETSc.Vec) -> PETSc.Vec:
        """
        Compute search direction using L-BFGS two-loop recursion.

        Implicitly applies H_k · ∇f where H_k ≈ (∇²f)^{-1}.

        Algorithm (Liu & Nocedal, 1989):
        1. q ← ∇f_k
        2. For i = k-1, k-2, ..., k-m:
               α_i ← ρ_i * s_i^T * q
               q ← q - α_i * y_i
        3. r ← H_0^k * q  (where H_0^k = γ_k * I)
        4. For i = k-m, k-m+1, ..., k-1:
               β ← ρ_i * y_i^T * r
               r ← r + s_i * (α_i - β)
        5. Return p = -r (negative for minimization)

        Args:
            grad: Current gradient ∇f_k

        Returns:
            Search direction p (descent direction)
        """
        q = grad.copy()

        # Number of stored corrections
        num_corrections = len(self.s_history)

        # First loop: backward recursion
        alphas = []
        for i in range(num_corrections - 1, -1, -1):
            alpha_i = self.rho_history[i] * self.s_history[i].dot(q)
            alphas.insert(0, alpha_i)
            q.axpy(-alpha_i, self.y_history[i])

        # Apply initial Hessian approximation H_0 = γI
        # γ = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
        # This scaling improves performance significantly
        if num_corrections > 0:
            s_last = self.s_history[-1]
            y_last = self.y_history[-1]
            s_dot_y = s_last.dot(y_last)
            y_dot_y = y_last.dot(y_last)

            # Avoid division by zero
            if y_dot_y > 1e-16:
                gamma = s_dot_y / y_dot_y
            else:
                gamma = 1.0

            # Ensure gamma is positive (should be by curvature condition)
            gamma = max(gamma, 1e-8)
        else:
            gamma = 1.0

        r = q.copy()
        r.scale(gamma)

        # Second loop: forward recursion
        for i in range(num_corrections):
            beta = self.rho_history[i] * self.y_history[i].dot(r)
            r.axpy(alphas[i] - beta, self.s_history[i])

        # Return negative for descent direction
        r.scale(-1.0)

        q.destroy()  # Clean up temporary

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

        Computes and stores:
            s_k = x_{k+1} - x_k
            y_k = ∇f_{k+1} - ∇f_k
            ρ_k = 1 / (y_k^T s_k)

        The curvature condition y_k^T s_k > 0 must be satisfied
        for L-BFGS to maintain positive definiteness. If violated,
        the update is skipped.

        Args:
            x_new: New iterate x_{k+1}
            x_old: Previous iterate x_k
            grad_new: New gradient ∇f_{k+1}
            grad_old: Previous gradient ∇f_k
        """
        # Compute s_k = x_{k+1} - x_k
        s = x_new.copy()
        s.axpy(-1.0, x_old)

        # Compute y_k = ∇f_{k+1} - ∇f_k
        y = grad_new.copy()
        y.axpy(-1.0, grad_old)

        # Compute y_k^T s_k (curvature)
        y_dot_s = y.dot(s)

        # Check curvature condition: y^T s > 0
        # This is required for maintaining positive definiteness
        if y_dot_s < 1e-14:
            if self.verbose and self.comm.rank == 0:
                print(
                    f"  Warning: Curvature condition violated (y^T s = {y_dot_s:.2e}). Skipping update."
                )
            s.destroy()
            y.destroy()
            return

        # Compute ρ_k = 1 / (y_k^T s_k)
        rho = 1.0 / y_dot_s

        # Add to history
        self.s_history.append(s)
        self.y_history.append(y)
        self.rho_history.append(rho)

        # Enforce memory limit (FIFO: remove oldest)
        if len(self.s_history) > self.m:
            # Destroy oldest vectors before removing
            old_s = self.s_history.pop(0)
            old_y = self.y_history.pop(0)
            old_s.destroy()
            old_y.destroy()
            self.rho_history.pop(0)

    def _clear_history(self):
        """
        Clear correction pair history.

        Used when restarting the algorithm (e.g., after detecting
        non-descent direction or other numerical issues).
        """
        # Destroy all stored vectors
        for s in self.s_history:
            s.destroy()
        for y in self.y_history:
            y.destroy()

        # Clear lists
        self.s_history.clear()
        self.y_history.clear()
        self.rho_history.clear()

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self._clear_history()


class PreconditionedLBFGS(LBFGSOptimizer):
    """
    Preconditioned L-BFGS optimizer.

    Uses a preconditioner P in the initial Hessian approximation:
    H_0 = P instead of H_0 = γI

    This can significantly improve convergence for problems where
    a good preconditioner is available (e.g., from background
    error covariance in 4D-Var).

    Example preconditioners:
        - Diagonal scaling (Jacobi)
        - Approximate inverse Hessian
        - Background error covariance B in 4D-Var context
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
            cost_function: Cost function to minimize
            preconditioner: Preconditioner operator with apply(v) method
                           that returns P * v
            memory_size: L-BFGS memory size
            options: Optimizer options
        """
        super().__init__(cost_function, memory_size, options)
        self.preconditioner = preconditioner

    def _two_loop_recursion(self, grad: PETSc.Vec) -> PETSc.Vec:
        """
        Two-loop recursion with preconditioner.

        Replaces H_0 = γI with H_0 = P where P is the preconditioner.
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

        q.destroy()

        return r


class BoundedLBFGS(LBFGSOptimizer):
    """
    L-BFGS with box constraints (L-BFGS-B).

    Implements L-BFGS-B algorithm for bound-constrained optimization:
        min f(x)  s.t.  l ≤ x ≤ u

    References:
        Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory
        algorithm for bound constrained optimization. SIAM Journal on
        scientific computing, 16(5), 1190-1208.

    Note: Full L-BFGS-B implementation is complex. This is a simplified
    version using projected gradient and projected line search.
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
            cost_function: Cost function to minimize
            lower_bounds: Lower bounds l (None = -∞)
            upper_bounds: Upper bounds u (None = +∞)
            memory_size: L-BFGS memory size
            options: Optimizer options
        """
        super().__init__(cost_function, memory_size, options)
        self.lower = lower_bounds
        self.upper = upper_bounds

    def solve(self, x0: PETSc.Vec) -> PETSc.Vec:
        """
        Minimize with box constraints.

        Uses projected gradient and projected line search.

        Note: This is a simplified implementation. Full L-BFGS-B
        would include:
            1. Active set identification
            2. Cauchy point computation
            3. Subspace minimization
            4. More sophisticated line search

        For production use, consider using PETSc's TAO (Toolkit for
        Advanced Optimization) which has a full L-BFGS-B implementation.
        """
        # Project initial guess onto feasible region
        x = self._project_onto_bounds(x0)

        grad = self.cost_function.gradient(x)
        cost = self.cost_function.value(x)

        max_iter = self.options.get("max_iterations", 100)

        if self.verbose and self.comm.rank == 0:
            print(f"\nL-BFGS-B Optimization (with box constraints)")
            print(f"{'=' * 70}")

        for k in range(max_iter):
            self.iteration = k

            # Compute projected gradient
            grad_proj = self._compute_projected_gradient(x, grad)
            grad_norm = grad_proj.norm()

            cost_prev = cost

            if k > 0:
                cost_change = cost - cost_prev
                if self.check_convergence(grad_norm, cost_change):
                    self.converged = True
                    break

            self.record_iteration(x, cost, grad_norm)

            # Compute search direction (in feasible space)
            direction = self._two_loop_recursion(grad)

            # Project search direction
            # TODO: Implement proper gradient projection in subspace
            # For now, use simple approach

            # Line search with projection
            alpha = self._projected_line_search(x, direction, grad, cost)

            # Update with projection
            x_new = x.copy()
            x_new.axpy(alpha, direction)
            x_new = self._project_onto_bounds(x_new)

            grad_new = self.cost_function.gradient(x_new)
            cost = self.cost_function.value(x_new)

            # Update history
            self._update_history(x_new, x, grad_new, grad)

            x.destroy()
            grad.destroy()
            grad_proj.destroy()
            direction.destroy()
            x = x_new
            grad = grad_new

        grad.destroy()

        return x

    def _project_onto_bounds(self, x: PETSc.Vec) -> PETSc.Vec:
        """
        Project onto feasible region [l, u].

        Args:
            x: Input vector

        Returns:
            Projected vector: x_proj[i] = max(l[i], min(x[i], u[i]))
        """
        x_proj = x.copy()

        if self.lower is not None:
            # x_proj = max(x_proj, lower)
            x_proj.pointwiseMax(x_proj, self.lower)

        if self.upper is not None:
            # x_proj = min(x_proj, upper)
            x_proj.pointwiseMin(x_proj, self.upper)

        return x_proj

    def _compute_projected_gradient(self, x: PETSc.Vec, grad: PETSc.Vec) -> PETSc.Vec:
        """
        Compute projected gradient.

        Sets gradient components to zero at active bounds:
            - If x[i] = l[i] and grad[i] > 0: grad_proj[i] = 0
            - If x[i] = u[i] and grad[i] < 0: grad_proj[i] = 0

        Args:
            x: Current point
            grad: Gradient

        Returns:
            Projected gradient
        """
        grad_proj = grad.copy()
        x_array = x.getArray()
        grad_array = grad_proj.getArray()

        tol = 1e-10  # Tolerance for checking active bounds

        if self.lower is not None:
            lower_array = self.lower.getArray()
            for i in range(len(x_array)):
                # At lower bound with positive gradient
                if abs(x_array[i] - lower_array[i]) < tol and grad_array[i] > 0:
                    grad_array[i] = 0.0

        if self.upper is not None:
            upper_array = self.upper.getArray()
            for i in range(len(x_array)):
                # At upper bound with negative gradient
                if abs(x_array[i] - upper_array[i]) < tol and grad_array[i] < 0:
                    grad_array[i] = 0.0

        grad_proj.setArray(grad_array)
        return grad_proj

    def _projected_line_search(
        self, x: PETSc.Vec, direction: PETSc.Vec, grad: PETSc.Vec, cost_current: float
    ) -> float:
        """
        Line search with projection onto feasible region.

        Args:
            x: Current point
            direction: Search direction
            grad: Current gradient
            cost_current: Current cost value

        Returns:
            Step size α
        """
        alpha = 1.0
        rho = 0.5
        c1 = 1e-4

        directional_derivative = grad.dot(direction)

        if directional_derivative >= 0:
            return 1e-8

        x_trial = x.duplicate()

        for i in range(20):
            # Trial point with projection
            x_trial.waxpy(alpha, direction, x)
            x_trial = self._project_onto_bounds(x_trial)

            # Evaluate cost
            cost_trial = self.cost_function.value(x_trial)

            # Check Armijo condition
            if cost_trial <= cost_current + c1 * alpha * directional_derivative:
                x_trial.destroy()
                return alpha

            alpha *= rho

        x_trial.destroy()
        return alpha

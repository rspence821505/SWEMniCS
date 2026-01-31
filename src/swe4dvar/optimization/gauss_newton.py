"""
Gauss-Newton optimizer for nonlinear least-squares problems.

Exploits special structure of 4D-Var cost function for
efficient Hessian approximation and Newton steps.

WARNING: This module is EXPERIMENTAL. The TLM/Adjoint integration
required for Hessian-vector products is not yet implemented.
Use LBFGSOptimizer for production 4D-Var optimization.
"""

import warnings
from typing import Optional
from petsc4py import PETSc

from .optimizer_base import Optimizer, TrustRegion


class GaussNewtonOptimizer(Optimizer):
    """
    Gauss-Newton method for 4D-Var optimization.

    Approximates Hessian as: H ≈ J^T · R^{-1} · J + B^{-1}
    where J is the linearized observation operator (via TLM).

    This is exact for linear least-squares and a good
    approximation when residuals are small.

    .. warning::
        EXPERIMENTAL: This optimizer is incomplete. The TLM and adjoint
        methods required for Hessian-vector products are not implemented.
        Use :class:`LBFGSOptimizer` instead for production use.
    """

    def __init__(self, cost_function, options: Optional[dict] = None):
        """
        Initialize Gauss-Newton optimizer.

        Args:
            cost_function: 4D-Var cost function
            options: Optimizer options

        Raises:
            UserWarning: This optimizer is experimental and incomplete.
        """
        warnings.warn(
            "GaussNewtonOptimizer is EXPERIMENTAL and incomplete. "
            "The TLM/Adjoint methods for Hessian-vector products are not implemented. "
            "Use LBFGSOptimizer for production 4D-Var optimization.",
            UserWarning,
            stacklevel=2
        )
        super().__init__(cost_function, options)

        # Trust region for globalization
        if options is None:
            options = {}
        self.trust_region = TrustRegion(radius_init=options.get("trust_radius", 1.0))

        # CG solver for inner iterations
        self.ksp = None

    def solve(self, x0: PETSc.Vec) -> PETSc.Vec:
        """
        Minimize using Gauss-Newton with trust region.

        Args:
            x0: Initial guess

        Returns:
            Optimal solution
        """
        x = x0.copy()

        max_iter = self.options.get("max_iterations", 50)

        for k in range(max_iter):
            self.iteration = k

            # Evaluate cost and gradient
            cost = self.cost_function.value(x)
            grad = self.cost_function.gradient(x)
            grad_norm = grad.norm()

            # Check convergence
            if k > 0:
                cost_change = cost - cost_prev
                if self.check_convergence(grad_norm, cost_change):
                    self.converged = True
                    break

            # Record iteration
            self.record_iteration(x, cost, grad_norm)

            # Solve trust region subproblem
            # min_{‖p‖ ≤ Δ} m(p) = f + g^T·p + ½p^T·H·p
            step = self._solve_trust_region_subproblem(x, grad)

            # Evaluate step quality
            x_new = x.copy()
            x_new.axpy(1.0, step)
            cost_new = self.cost_function.value(x_new)

            actual_reduction = cost - cost_new
            predicted_reduction = self._evaluate_model_reduction(grad, step)

            # Update trust region
            step_norm = step.norm()
            self.trust_region.update_radius(
                actual_reduction, predicted_reduction, step_norm
            )

            # Accept or reject step
            if self.trust_region.accept_step(actual_reduction, predicted_reduction):
                x = x_new
                cost_prev = cost
            else:
                cost_prev = cost

        return x

    def _solve_trust_region_subproblem(
        self, x: PETSc.Vec, grad: PETSc.Vec
    ) -> PETSc.Vec:
        """
        Solve trust region subproblem using Steihaug-CG.

        Approximately minimizes quadratic model:
        m(p) = ½p^T·H·p + g^T·p
        subject to ‖p‖ ≤ Δ

        Args:
            x: Current point
            grad: Current gradient

        Returns:
            Trust region step
        """
        # TODO: Implement Steihaug conjugate gradient algorithm
        # 1. Run CG on Gauss-Newton system
        # 2. Stop if negative curvature detected
        # 3. Stop if trust region boundary reached
        # 4. Return approximate solution
        pass

    def _apply_gauss_newton_hessian(self, x: PETSc.Vec, v: PETSc.Vec) -> PETSc.Vec:
        """
        Apply Gauss-Newton Hessian approximation: H·v.

        H·v = B^{-1}·v + Σ_k (DQ_k)^T · R_k^{-1} · (DQ_k·v)

        where DQ_k is linearized QoI map (TLM + observation operator).

        Args:
            x: Current point (for linearization)
            v: Direction vector

        Returns:
            H·v
        """
        # Background term: B^{-1}·v
        result = self.cost_function.B.apply_inverse(v)

        # Observation terms: Σ_k (DQ_k)^T · R_k^{-1} · (DQ_k·v)
        for k in self.cost_function.obs_times:
            # Apply TLM: δu_k = TLM_{k:0}·v
            delta_u_k = self._apply_tlm(v, k)

            # Apply observation operator: δy_k = H_k·δu_k
            delta_y_k = self.cost_function.obs_op.forward(delta_u_k)

            # Weight by R_k^{-1}
            weighted = self.cost_function.R[k].apply_inverse(delta_y_k)

            # Apply adjoint: (H_k^T · ADJ_{k:0})·weighted
            adjoint_contrib = self._apply_adjoint(weighted, k)

            result.axpy(1.0, adjoint_contrib)

        return result

    def _apply_tlm(self, v: PETSc.Vec, time_index: int) -> PETSc.Vec:
        """
        Apply tangent linear model to propagate perturbation.

        Args:
            v: Initial perturbation
            time_index: Target time

        Returns:
            Perturbation at target time
        """
        # TODO: Use TangentLinearModel from adjoint module
        pass

    def _apply_adjoint(self, v: PETSc.Vec, time_index: int) -> PETSc.Vec:
        """
        Apply adjoint model to propagate backward.

        Args:
            v: Adjoint forcing at time_index
            time_index: Source time

        Returns:
            Adjoint at initial time
        """
        # TODO: Use AdjointModel from adjoint module
        pass

    def _evaluate_model_reduction(self, grad: PETSc.Vec, step: PETSc.Vec) -> float:
        """
        Evaluate predicted reduction from quadratic model.

        pred = -g^T·p - ½p^T·H·p

        Args:
            grad: Gradient
            step: Proposed step

        Returns:
            Predicted reduction
        """
        # Linear term: g^T·p
        linear_term = grad.dot(step)

        # Quadratic term: ½p^T·H·p
        Hp = self.cost_function.hessian_vector_product(None, step)
        quadratic_term = 0.5 * step.dot(Hp)

        return -(linear_term + quadratic_term)


class InexactGaussNewton(GaussNewtonOptimizer):
    """
    Inexact Gauss-Newton with adaptive CG tolerance.

    Uses Eisenstat-Walker forcing terms to adaptively
    set CG tolerance based on Newton iteration progress.
    """

    def __init__(self, cost_function, options: Optional[dict] = None):
        """
        Initialize inexact Gauss-Newton.

        Args:
            cost_function: Cost function
            options: Optimizer options
        """
        super().__init__(cost_function, options)

        # Previous gradient norm for adaptive tolerance
        self.prev_grad_norm = None

    def _get_cg_tolerance(self, grad_norm: float) -> float:
        """
        Compute adaptive CG tolerance using Eisenstat-Walker.

        η_k = min(η_max, max(η_min, γ·(‖g_k‖/‖g_{k-1}‖)^α))

        Args:
            grad_norm: Current gradient norm

        Returns:
            CG tolerance
        """
        eta_min = self.options.get("cg_tol_min", 1e-6)
        eta_max = self.options.get("cg_tol_max", 0.5)
        gamma = self.options.get("cg_tol_gamma", 0.9)
        alpha = self.options.get("cg_tol_alpha", 1.0)

        if self.prev_grad_norm is None:
            eta = eta_max
        else:
            ratio = grad_norm / self.prev_grad_norm
            eta = min(eta_max, max(eta_min, gamma * (ratio**alpha)))

        self.prev_grad_norm = grad_norm
        return eta


class MatrixFreeGaussNewton(GaussNewtonOptimizer):
    """
    Matrix-free Gauss-Newton using shell matrices.

    Never explicitly forms Hessian, only applies it via
    TLM and adjoint operations. Memory efficient for
    large problems.
    """

    def __init__(self, cost_function, options: Optional[dict] = None):
        """
        Initialize matrix-free Gauss-Newton.

        Args:
            cost_function: Cost function
            options: Optimizer options
        """
        super().__init__(cost_function, options)

        # Create shell matrix for Hessian
        self._setup_shell_matrix()

    def _setup_shell_matrix(self):
        """
        Create PETSc shell matrix for Hessian.

        Defines matrix-vector product via callback.
        """
        # TODO: Implement PETSc.Mat.createPython with mult callback
        pass

    def _hessian_mult_callback(self, mat, x: PETSc.Vec, y: PETSc.Vec):
        """
        Callback for shell matrix-vector product.

        Computes y = H·x using TLM/adjoint.

        Args:
            mat: Shell matrix (unused)
            x: Input vector
            y: Output vector (modified in-place)
        """
        result = self._apply_gauss_newton_hessian(None, x)
        y.copy(result)

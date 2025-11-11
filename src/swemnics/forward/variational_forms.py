"""
Variational forms module for SWEMniCS forward solver.

This module exposes Jacobians computed during Newton iterations
for efficient adjoint computation.
"""

from typing import Tuple, Optional
import dolfinx
from petsc4py import PETSc


class VariationalForm:
    """
    Base class for variational forms with Jacobian access.

    Provides interface for assembling residuals and Jacobians
    needed for both forward Newton solver and adjoint computation.
    """

    def __init__(self, function_space, dt: float):
        """
        Initialize variational form.

        Args:
            function_space: FEniCSx function space
            dt: Time step size
        """
        self.function_space = function_space
        self.dt = dt

    def assemble_residual(self, u_next, u_n, u_nm1) -> PETSc.Vec:
        """
        Assemble residual R(u^{n+1}; u^n, u^{n-1}).

        For BDF2: R = (3u^{n+1} - 4u^n + u^{n-1})/(2Δt) + F(u^{n+1})

        Args:
            u_next: State at time n+1
            u_n: State at time n
            u_nm1: State at time n-1

        Returns:
            Assembled residual vector
        """
        raise NotImplementedError("Must be implemented by subclass")

    def assemble_jacobian(self, u_next, u_n, u_nm1) -> PETSc.Mat:
        """
        Assemble Jacobian ∂R/∂u^{n+1}.

        For BDF2: J = (3/(2Δt))·M + ∂F/∂u

        Args:
            u_next: State at time n+1
            u_n: State at time n
            u_nm1: State at time n-1

        Returns:
            Assembled Jacobian matrix
        """
        raise NotImplementedError("Must be implemented by subclass")


class SWEVariationalForm(VariationalForm):
    """
    Variational form for shallow water equations.

    Implements BDF2 time discretization with nonlinear spatial operators.
    """

    def __init__(
        self, function_space, dt: float, g: float = 9.81, friction: float = 0.0
    ):
        """
        Initialize SWE variational form.

        Args:
            function_space: Mixed function space for (H, u, v)
            dt: Time step size
            g: Gravitational acceleration
            friction: Manning friction coefficient
        """
        super().__init__(function_space, dt)
        self.g = g
        self.friction = friction

    def assemble_residual(self, u_next, u_n, u_nm1) -> PETSc.Vec:
        """Assemble SWE residual with BDF2 time discretization."""
        # TODO: Implement SWE-specific residual assembly
        pass

    def assemble_jacobian(self, u_next, u_n, u_nm1) -> PETSc.Mat:
        """Assemble SWE Jacobian using automatic differentiation."""
        # TODO: Implement SWE-specific Jacobian assembly
        pass


class LinearizedVariationalForm:
    """
    Linearized variational form for tangent linear model.

    Provides J·δu operations needed for TLM integration.
    """

    def __init__(self, base_form: VariationalForm):
        """
        Initialize from base variational form.

        Args:
            base_form: The nonlinear variational form to linearize
        """
        self.base_form = base_form

    def apply_jacobian(self, u_base, delta_u, u_n, u_nm1) -> PETSc.Vec:
        """
        Apply Jacobian to perturbation: J(u_base)·δu.

        Args:
            u_base: Base state for linearization
            delta_u: Perturbation
            u_n: Previous time state
            u_nm1: Two-steps-back state

        Returns:
            J·δu
        """
        raise NotImplementedError("Must be implemented by subclass")

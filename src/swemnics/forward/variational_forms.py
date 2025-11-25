"""
Variational forms module for SWEMniCS forward solver.

This module exposes Jacobians and mass matrices computed during Newton
iterations for efficient adjoint computation. It handles BDF2-specific
time-stepping logic including:
  - First step: Backward Euler (α=0)
  - Subsequent steps: Full BDF2 (α=1)
  - Support for adaptive time-stepping
  - Wetting/drying transitions

Author: Rylan Spence
Date: 2025
"""

from typing import Tuple, Optional, Union
import numpy as np
import dolfinx
from dolfinx import fem
from ufl import (
    TestFunction,
    TrialFunction,
    dx,
    grad,
    div,
    dot,
    inner,
    FacetNormal,
    ds,
)
from petsc4py import PETSc


class VariationalForm:
    """
    Base class for variational forms with Jacobian access.

    Provides interface for assembling residuals and Jacobians
    needed for both forward Newton solver and adjoint computation.

    Attributes
    ----------
    function_space : dolfinx.fem.FunctionSpace
        FEniCSx function space.
    dt : float or list
        Time step size(s). Can be scalar for constant dt or list for adaptive.
    use_bdf2 : bool
        Whether to use full BDF2 (True) or Backward Euler (False).
        Typically False for first step, True thereafter.
    """

    def __init__(
        self,
        function_space: dolfinx.fem.FunctionSpace,
        dt: Union[float, list],
        use_bdf2: bool = True,
    ):
        """
        Initialize variational form.

        Parameters
        ----------
        function_space : dolfinx.fem.FunctionSpace
            FEniCSx function space.
        dt : float or list
            Time step size. If list, supports adaptive time-stepping.
        use_bdf2 : bool, optional
            If True, use BDF2; if False, use Backward Euler (default: True).
        """
        self.function_space = function_space
        self.dt = dt
        self.use_bdf2 = use_bdf2

        # Cache for assembled mass matrix
        self._mass_matrix = None

    def get_dt(self, step: int) -> float:
        """
        Get time step size for given step index.

        Parameters
        ----------
        step : int
            Time step index.

        Returns
        -------
        float
            Time step size.
        """
        if isinstance(self.dt, (list, tuple, np.ndarray)):
            return self.dt[step]
        else:
            return self.dt

    def set_time_scheme(self, use_bdf2: bool):
        """
        Switch between BDF2 and Backward Euler.

        Parameters
        ----------
        use_bdf2 : bool
            If True, use full BDF2; if False, use Backward Euler.

        Notes
        -----
        Typically called after first time step to switch from
        Backward Euler startup to full BDF2.
        """
        self.use_bdf2 = use_bdf2

    def assemble_mass_matrix(self) -> PETSc.Mat:
        """
        Assemble mass matrix M for the function space.

        Returns
        -------
        PETSc.Mat
            Assembled mass matrix.

        Notes
        -----
        The mass matrix is cached after first assembly for efficiency.
        """
        if self._mass_matrix is not None:
            return self._mass_matrix

        # Define trial and test functions
        u = TrialFunction(self.function_space)
        v = TestFunction(self.function_space)

        # Mass matrix form: M[i,j] = ∫ φ_j · φ_i dx
        mass_form = inner(u, v) * dx

        # Assemble into PETSc matrix
        self._mass_matrix = fem.petsc.assemble_matrix(fem.form(mass_form))
        self._mass_matrix.assemble()

        return self._mass_matrix

    def assemble_residual(
        self,
        u_next: dolfinx.fem.Function,
        u_n: dolfinx.fem.Function,
        u_nm1: Optional[dolfinx.fem.Function],
        step: int = 0,
    ) -> PETSc.Vec:
        """
        Assemble residual R(u^{n+1}; u^n, u^{n-1}).

        For BDF2:
            R = (3u^{n+1} - 4u^n + u^{n-1})/(2Δt) + F(u^{n+1})

        For Backward Euler:
            R = (u^{n+1} - u^n)/Δt + F(u^{n+1})

        Parameters
        ----------
        u_next : dolfinx.fem.Function
            State at time n+1.
        u_n : dolfinx.fem.Function
            State at time n.
        u_nm1 : dolfinx.fem.Function or None
            State at time n-1 (required for BDF2, can be None for Backward Euler).
        step : int, optional
            Time step index (for adaptive dt).

        Returns
        -------
        PETSc.Vec
            Assembled residual vector.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def assemble_jacobian(
        self,
        u_next: dolfinx.fem.Function,
        u_n: dolfinx.fem.Function,
        u_nm1: Optional[dolfinx.fem.Function],
        step: int = 0,
    ) -> PETSc.Mat:
        """
        Assemble Jacobian ∂R/∂u^{n+1}.

        For BDF2:
            J = (3/(2Δt))·M + ∂F/∂u

        For Backward Euler:
            J = (1/Δt)·M + ∂F/∂u

        Parameters
        ----------
        u_next : dolfinx.fem.Function
            State at time n+1.
        u_n : dolfinx.fem.Function
            State at time n.
        u_nm1 : dolfinx.fem.Function or None
            State at time n-1.
        step : int, optional
            Time step index (for adaptive dt).

        Returns
        -------
        PETSc.Mat
            Assembled Jacobian matrix.
        """
        raise NotImplementedError("Must be implemented by subclass")


class SWEVariationalForm(VariationalForm):
    """
    Variational form for shallow water equations.

    Implements BDF2 time discretization with nonlinear spatial operators.
    Supports wetting/drying via positivity-preserving transformations.

    Attributes
    ----------
    g : float
        Gravitational acceleration (m/s²).
    friction : float
        Manning friction coefficient.
    wetting_drying : bool
        Whether to apply wetting/drying positivity scheme.
    h_min : float
        Minimum depth for wetting/drying transitions.
    """

    def __init__(
        self,
        function_space: dolfinx.fem.FunctionSpace,
        dt: Union[float, list],
        g: float = 9.81,
        friction: float = 0.0,
        use_bdf2: bool = True,
        wetting_drying: bool = False,
        h_min: float = 0.01,
    ):
        """
        Initialize SWE variational form.

        Parameters
        ----------
        function_space : dolfinx.fem.FunctionSpace
            Mixed function space for (H, u, v).
        dt : float or list
            Time step size(s).
        g : float, optional
            Gravitational acceleration (default: 9.81 m/s²).
        friction : float, optional
            Manning friction coefficient (default: 0.0).
        use_bdf2 : bool, optional
            Use BDF2 (True) or Backward Euler (False).
        wetting_drying : bool, optional
            Apply wetting/drying scheme (default: False).
        h_min : float, optional
            Minimum depth threshold (default: 0.01 m).
        """
        super().__init__(function_space, dt, use_bdf2)
        self.g = g
        self.friction = friction
        self.wetting_drying = wetting_drying
        self.h_min = h_min

    def apply_wetting_drying_transform(
        self, H: dolfinx.fem.Function
    ) -> dolfinx.fem.Function:
        """
        Apply positivity-preserving transformation for wetting/drying.

        H̃ = H + f(H) where f(H) ensures H̃ > h_min even when H ≤ 0.

        Parameters
        ----------
        H : dolfinx.fem.Function
            Raw water depth.

        Returns
        -------
        dolfinx.fem.Function
            Transformed depth H̃ > h_min.

        Notes
        -----
        This implements the α-scheme from Kärna (2020).
        The Jacobian ∂f/∂H has a discontinuity at H = 0,
        which must be handled carefully in the adjoint.
        """
        # TODO: Implement α-scheme transformation
        # For now, return identity (no wetting/drying)
        return H

    def assemble_residual(
        self,
        u_next: dolfinx.fem.Function,
        u_n: dolfinx.fem.Function,
        u_nm1: Optional[dolfinx.fem.Function],
        step: int = 0,
    ) -> PETSc.Vec:
        """
        Assemble SWE residual with BDF2 or Backward Euler time discretization.

        Parameters
        ----------
        u_next : dolfinx.fem.Function
            State at time n+1.
        u_n : dolfinx.fem.Function
            State at time n.
        u_nm1 : dolfinx.fem.Function or None
            State at time n-1 (required for BDF2).
        step : int, optional
            Time step index.

        Returns
        -------
        PETSc.Vec
            Residual vector.

        Raises
        ------
        ValueError
            If BDF2 is requested but u_nm1 is None.
        """
        if self.use_bdf2 and u_nm1 is None:
            raise ValueError("BDF2 requires u_nm1, but None was provided")

        dt = self.get_dt(step)
        v = TestFunction(self.function_space)

        # Time derivative term
        if self.use_bdf2:
            # BDF2: (3u^{n+1} - 4u^n + u^{n-1})/(2Δt)
            time_deriv = (3.0 * u_next - 4.0 * u_n + u_nm1) / (2.0 * dt)
        else:
            # Backward Euler: (u^{n+1} - u^n)/Δt
            time_deriv = (u_next - u_n) / dt

        # Spatial operator F(u^{n+1})

        # TODO: Add full SWE spatial terms (advection, pressure, friction)
        spatial_op = 0  # Placeholder

        # Total residual form
        residual_form = inner(time_deriv + spatial_op, v) * dx

        # Assemble into PETSc vector
        residual_vec = fem.petsc.assemble_vector(fem.form(residual_form))
        residual_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )

        return residual_vec

    def assemble_jacobian(
        self,
        u_next: dolfinx.fem.Function,
        u_n: dolfinx.fem.Function,
        u_nm1: Optional[dolfinx.fem.Function],
        step: int = 0,
    ) -> PETSc.Mat:
        """
        Assemble SWE Jacobian using automatic differentiation.

        Parameters
        ----------
        u_next : dolfinx.fem.Function
            State at time n+1.
        u_n : dolfinx.fem.Function
            State at time n.
        u_nm1 : dolfinx.fem.Function or None
            State at time n-1.
        step : int, optional
            Time step index.

        Returns
        -------
        PETSc.Mat
            Jacobian matrix J = ∂R/∂u^{n+1}.
        """
        dt = self.get_dt(step)

        # Get mass matrix
        M = self.assemble_mass_matrix()

        # Time derivative Jacobian
        if self.use_bdf2:
            # BDF2: J_time = (3/(2Δt))·M
            time_coeff = 3.0 / (2.0 * dt)
        else:
            # Backward Euler: J_time = (1/Δt)·M
            time_coeff = 1.0 / dt

        # Spatial Jacobian: ∂F/∂u (computed via automatic differentiation)
        # TODO: Implement full SWE spatial Jacobian
        spatial_jacobian = M.copy()  # Placeholder - should be ∂F/∂u
        spatial_jacobian.scale(0.0)  # Zero it out for now

        # Total Jacobian: J = time_coeff·M + ∂F/∂u
        J = M.copy()
        J.scale(time_coeff)
        J.axpy(1.0, spatial_jacobian)  # Add spatial contribution

        return J


class LinearizedVariationalForm:
    """
    Linearized variational form for tangent linear model.

    Provides J·δu operations needed for TLM integration without
    explicitly forming the full Jacobian matrix.
    """

    def __init__(self, base_form: VariationalForm):
        """
        Initialize from base variational form.

        Parameters
        ----------
        base_form : VariationalForm
            The nonlinear variational form to linearize.
        """
        self.base_form = base_form

    def apply_jacobian(
        self,
        u_base: dolfinx.fem.Function,
        delta_u: dolfinx.fem.Function,
        u_n: dolfinx.fem.Function,
        u_nm1: Optional[dolfinx.fem.Function],
        step: int = 0,
    ) -> PETSc.Vec:
        """
        Apply Jacobian to perturbation: J(u_base)·δu.

        This is the core operation for tangent linear model integration.

        Parameters
        ----------
        u_base : dolfinx.fem.Function
            Base state for linearization.
        delta_u : dolfinx.fem.Function
            Perturbation.
        u_n : dolfinx.fem.Function
            Previous time state.
        u_nm1 : dolfinx.fem.Function or None
            Two-steps-back state.
        step : int, optional
            Time step index.

        Returns
        -------
        PETSc.Vec
            Result of J·δu.
        """
        # Assemble Jacobian at base state
        J = self.base_form.assemble_jacobian(u_base, u_n, u_nm1, step)

        # Apply to perturbation
        result = delta_u.vector.copy()
        J.mult(delta_u.vector, result)

        return result


class BDF2TimeCoefficients:
    """
    Helper class for BDF2 time-coupling coefficients.

    Provides correct coefficients for:
      - Forward time-stepping
      - Adjoint time-coupling
      - Tangent linear model

    Handles both constant and adaptive time-stepping.
    """

    def __init__(self, dt: Union[float, list], use_bdf2: bool = True):
        """
        Initialize BDF2 coefficients.

        Parameters
        ----------
        dt : float or list
            Time step size(s).
        use_bdf2 : bool, optional
            Use BDF2 (True) or Backward Euler (False).
        """
        self.dt = dt
        self.use_bdf2 = use_bdf2

    def get_forward_coeffs(self, step: int) -> Tuple[float, float, float]:
        """
        Get forward time-stepping coefficients.

        For BDF2: (3u^{n+1} - 4u^n + u^{n-1})/(2Δt)
        Returns: (c_{n+1}, c_n, c_{n-1}) = (3/(2Δt), -4/(2Δt), 1/(2Δt))

        For Backward Euler: (u^{n+1} - u^n)/Δt
        Returns: (c_{n+1}, c_n, c_{n-1}) = (1/Δt, -1/Δt, 0)

        Parameters
        ----------
        step : int
            Time step index.

        Returns
        -------
        tuple of float
            (c_{n+1}, c_n, c_{n-1})
        """
        dt = self.dt if isinstance(self.dt, float) else self.dt[step]

        if self.use_bdf2:
            return (3.0 / (2.0 * dt), -4.0 / (2.0 * dt), 1.0 / (2.0 * dt))
        else:
            return (1.0 / dt, -1.0 / dt, 0.0)

    def get_adjoint_coeffs(self, step: int) -> Tuple[float, float]:
        """
        Get adjoint time-coupling coefficients.

        For BDF2: (4/(2Δt))·M·λ^{n+1} - (1/(2Δt))·M·λ^{n+2}
        Returns: (c_{n+1}, c_{n+2}) = (4/(2Δt), -1/(2Δt))

        For Backward Euler: (1/Δt)·M·λ^{n+1}
        Returns: (c_{n+1}, c_{n+2}) = (1/Δt, 0)

        Parameters
        ----------
        step : int
            Time step index.

        Returns
        -------
        tuple of float
            (c_{n+1}, c_{n+2})
        """
        dt = self.dt if isinstance(self.dt, float) else self.dt[step]

        if self.use_bdf2:
            return (4.0 / (2.0 * dt), -1.0 / (2.0 * dt))
        else:
            return (1.0 / dt, 0.0)

    def get_jacobian_coeff(self, step: int) -> float:
        """
        Get Jacobian time derivative coefficient.

        For BDF2: J = (3/(2Δt))·M + ∂F/∂u
        Returns: 3/(2Δt)

        For Backward Euler: J = (1/Δt)·M + ∂F/∂u
        Returns: 1/Δt

        Parameters
        ----------
        step : int
            Time step index.

        Returns
        -------
        float
            Coefficient multiplying mass matrix in Jacobian.
        """
        dt = self.dt if isinstance(self.dt, float) else self.dt[step]

        if self.use_bdf2:
            return 3.0 / (2.0 * dt)
        else:
            return 1.0 / dt

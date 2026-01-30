"""DG steady-state solver for shallow water equations.

This module contains the DGSolver class which implements a discontinuous
Galerkin method for steady-state shallow water equations.
"""

from dolfinx import fem as fe

try:
    from dolfinx.fem import functionspace
except ImportError:
    from dolfinx.fem import FunctionSpace as functionspace

from ufl import (
    TestFunction,
    FacetNormal,
    dot,
    inner,
    avg,
    jump,
    dS,
)

from petsc4py import PETSc

from swe4dvar.utils.fem_utilities import create_element, create_mixed_element
from .base_solver import BaseSolver


class DGSolver(BaseSolver):
    """DG steady-state solver."""

    def init_fields(self):
        """Initialize the variables"""
        self.p_type = "DG"
        if self.p_degree[0] != self.p_degree[1]:
            raise RuntimeError("DG solver requires equal polynomial degrees")

        # Use refactored element creation
        el_h = create_element(self.domain, self.p_type, self.p_degree[0])
        el_vel = create_element(self.domain, self.p_type, self.p_degree[1], shape=(2,))
        me = create_mixed_element([el_h, el_vel])
        self.V = functionspace(self.domain, me)

        self.V_scalar = functionspace(self.domain, (self.p_type, self.p_degree[0]))

        self.u = fe.Function(self.V)
        self.p = TestFunction(self.V)

    def init_weak_form(self):
        """Initialize the weak form"""
        super().init_weak_form()

        # add DG upwinding
        C = fe.Constant(self.domain, PETSc.ScalarType(1.0))
        n = FacetNormal(self.domain)
        flux = dot(avg(self.Fu), n("+")) - 0.5 * C * jump(self.u)
        self.F += inner(flux, jump(self.p)) * dS

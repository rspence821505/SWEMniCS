"""DGCG implicit solver for shallow water equations.

This module contains the DGCGImplicit class which implements a mixed
DG continuity / CG momentum formulation with SUPG stabilization.
"""

from dolfinx import fem as fe

try:
    from dolfinx.fem import functionspace
except ImportError:
    from dolfinx.fem import FunctionSpace as functionspace

from ufl import (
    TestFunctions,
    as_vector,
)

from swe4dvar.utils.fem_utilities import create_element, create_mixed_element
from .dg_implicit import DGImplicit


class DGCGImplicit(DGImplicit):
    """DG continuity and CG momentum with SUPG."""

    def init_fields(self):
        """Initialize the variables with CG for momentum."""
        self.p_type = "CG"

        # Use refactored element creation
        el_h = create_element(self.domain, self.p_type, self.p_degree[0])
        el_vel = create_element(self.domain, self.p_type, self.p_degree[1], shape=(2,))
        me = create_mixed_element([el_h, el_vel])
        self.V = functionspace(self.domain, me)

        self.V_vel = self.V.sub(1).collapse()[0]
        self.V_scalar = self.V.sub(0).collapse()[0]
        if self.verbose:
            self.log("V scalar", self.V_scalar)

        self.u = fe.Function(self.V)
        self.hel, self.vel_sol = self.u.split()

        self.p1, self.p2 = TestFunctions(self.V)
        self.p = as_vector((self.p1, self.p2[0], self.p2[1]))

        self.u_n = fe.Function(self.V)
        self.u_n.name = "u_n"
        self.u_n_old = fe.Function(self.V)
        self.u_n_old.name = "u_n_old"

    def init_weak_form(self):
        """Initialize weak form - adds SUPG from parent."""
        super().init_weak_form()

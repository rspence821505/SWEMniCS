"""DG implicit time-stepping solver for shallow water equations.

This module contains the DGImplicit class which implements a discontinuous
Galerkin method for time-dependent shallow water equations.
"""

from dolfinx import fem as fe

try:
    from dolfinx.fem import functionspace
except ImportError:
    from dolfinx.fem import FunctionSpace as functionspace

from ufl import (
    TestFunctions,
    FacetNormal,
    as_vector,
    dot,
    inner,
    avg,
    jump,
    sqrt,
    conditional,
    dS,
)

from swemnics.physics.constants import g, R
from swemnics.utils.fem_utilities import create_element, create_mixed_element

from .cg_implicit import CGImplicit


class DGImplicit(CGImplicit):
    """DG implicit time-stepping solver."""

    def init_fields(self):
        """Initialize the variables"""
        self.p_type = "DG"

        # Use refactored element creation
        el_h = create_element(self.domain, self.p_type, self.p_degree[0])
        el_vel = create_element(self.domain, self.p_type, self.p_degree[1], shape=(2,))
        me = create_mixed_element([el_h, el_vel])
        self.V = functionspace(self.domain, me)

        # for plotting
        self.V_vel = self.V.sub(1).collapse()[0]
        self.V_scalar = self.V.sub(0).collapse()[0]

        self.u = fe.Function(self.V)
        self.hel, self.vel_sol = self.u.split()

        self.p1, self.p2 = TestFunctions(self.V)
        self.p = as_vector((self.p1, self.p2[0], self.p2[1]))

        self.u_n = fe.Function(self.V)
        self.u_n.name = "u_n"
        self.u_n_old = fe.Function(self.V)
        self.u_n_old.name = "u_n_old"

    def init_weak_form(self):
        """Initialize the weak form with DG upwinding"""
        super().init_weak_form()

        # add DG upwinding
        eps = 1e-16
        n = FacetNormal(self.domain)

        h, ux, uy = self.problem._get_standard_vars(self.u, "h")
        vela = as_vector((ux("+"), uy("+")))
        velb = as_vector((ux("-"), uy("-")))

        vnorma = conditional(dot(vela, vela) > eps, sqrt(dot(vela, vela)), 0.0)
        vnormb = conditional(dot(velb, velb) > eps, sqrt(dot(velb, velb)), 0.0)

        if self.swe_type == "full":
            C = conditional(
                (vnorma + sqrt(g * h("+"))) > (vnormb + sqrt(g * h("-"))),
                (vnorma + sqrt(g * h("+"))),
                (vnormb + sqrt(g * h("-"))),
            )
        elif self.swe_type == "linear":
            h_b = self.problem.get_h_b(self.u)
            C = conditional(
                (sqrt(g * h_b("+"))) > (sqrt(g * h_b("-"))),
                (sqrt(g * h_b("+"))),
                (sqrt(g * h_b("-"))),
            )

        if self.problem.spherical:
            if self.problem.projected:
                if self.verbose:
                    self.log("spherical projected DG!!")
                flux = dot(avg(self.Fu), n("+")) + 0.5 * C * jump(self.Q)
            else:
                flux = dot(avg(self.Fu), n("+")) + 0.5 * C * avg(
                    self.problem.S**2 / R
                ) * jump(self.Q)
        else:
            flux = dot(avg(self.Fu), n("+")) + 0.5 * C * jump(self.Q)

        self.F += inner(flux, jump(self.p)) * dS

    def add_bcs_to_weak_form(self):
        """Add boundary integrals for DG."""
        super().add_bcs_to_weak_form()
        boundary_conditions = self.problem.boundary_conditions
        ds_exterior = self.problem.ds
        n = FacetNormal(self.domain)

        if self.p_type == "DG":
            if self.swe_type == "full":
                if self.verbose:
                    self.log("Adding DG boundary conditions weakly")
                h, ux, uy = self.problem._get_standard_vars(self.u, "h")
                h_bc, ux_bc, uy_bc = self.problem._get_standard_vars(self.u_bc, "h")

                vel = as_vector((ux, uy))
                un = dot(vel, n)
                eps = 1e-8
                vnorm = conditional(dot(vel, vel) > eps, sqrt(dot(vel, vel)), 0.0)

                jump_Q_wall = as_vector((0, 2 * h * un * n[0], 2 * h * un * n[1]))
                C_wall = vnorm + sqrt(g * h)

                u_wall = as_vector(
                    (
                        self.u[0],
                        self.u[1] * n[1] * n[1]
                        - self.u[1] * n[0] * n[0]
                        - 2 * self.u[2] * n[0] * n[1],
                        self.u[2] * n[0] * n[0]
                        - self.u[2] * n[1] * n[1]
                        - 2 * self.u[1] * n[0] * n[1],
                    )
                )
                Fu_wall_ext = self.problem.make_Fu(u_wall)

                for condition in boundary_conditions:
                    if condition.type == "Open":
                        self.F += dot(dot(self.Fu_open, n), self.p) * ds_exterior(
                            condition.marker
                        )
                    if condition.type == "Wall":
                        self.F += dot(
                            0.5 * dot(self.Fu, n) + 0.5 * dot(Fu_wall_ext, n), self.p
                        ) * ds_exterior(condition.marker) + dot(
                            0.5 * C_wall * jump_Q_wall, self.p
                        ) * ds_exterior(
                            condition.marker
                        )

            elif self.swe_type == "linear":
                if self.verbose:
                    self.log("Adding linearized DG boundary conditions weakly")
                h, ux, uy = self.problem._get_standard_vars(self.u, "h")
                h_bc, ux_bc, uy_bc = self.problem._get_standard_vars(self.u_bc, "h")
                h_b = self.problem.get_h_b(self.u)

                vel = as_vector((ux, uy))
                un = dot(vel, n)
                eps = 1e-16
                vnorm = conditional(dot(vel, vel) > eps, sqrt(dot(vel, vel)), 0.0)

                jump_Q_wall = as_vector((0, 2 * un * n[0], 2 * un * n[1]))
                C_wall = sqrt(g * h_b)

                u_wall = as_vector(
                    (
                        self.u[0],
                        self.u[1] * n[1] * n[1]
                        - self.u[1] * n[0] * n[0]
                        - 2 * self.u[2] * n[0] * n[1],
                        self.u[2] * n[0] * n[0]
                        - self.u[2] * n[1] * n[1]
                        - 2 * self.u[1] * n[0] * n[1],
                    )
                )
                Fu_wall_ext = self.problem.make_Fu_linearized(u_wall)
                jump_Q_open = as_vector((h - h_bc, ux - ux_bc, uy - uy_bc))
                C_open = sqrt(g * h_b)

                for condition in boundary_conditions:
                    if condition.type == "Open":
                        self.F += dot(
                            0.5 * dot(self.Fu_open, n) + 0.5 * dot(self.Fu, n), self.p
                        ) * ds_exterior(condition.marker) + dot(
                            0.5 * C_open * jump_Q_open, self.p
                        ) * ds_exterior(
                            condition.marker
                        )
                    if condition.type == "Wall":
                        self.F += dot(
                            0.5 * dot(self.Fu, n) + 0.5 * dot(Fu_wall_ext, n), self.p
                        ) * ds_exterior(condition.marker) + dot(
                            0.5 * C_wall * jump_Q_wall, self.p
                        ) * ds_exterior(
                            condition.marker
                        )

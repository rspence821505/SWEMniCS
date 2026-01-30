"""DG implicit non-conservative solver for shallow water equations.

This module contains the DGImplicitNonConservative class which implements
a discontinuous Galerkin method with non-conservative formulation.
"""

from dolfinx import fem as fe

from ufl import (
    FacetNormal,
    as_vector,
    dot,
    inner,
    avg,
    jump,
    sqrt,
    conditional,
    grad,
    dx,
    dS,
)

from petsc4py import PETSc

from swe4dvar.physics.constants import g, R
from .dg_implicit import DGImplicit


class DGImplicitNonConservative(DGImplicit):
    """DG implicit solver with non-conservative formulation."""

    def init_weak_form(self):
        """Initialize the weak form with non-conservative formulation."""
        theta = self.theta
        self.set_initial_condition()

        self.u_bc = as_vector((self.problem.u_bc[0], self.u[1], self.u[2]))
        if self.swe_type == "full":
            print("Creating NONCONSERVATIVE DG FORM\n\n")
            self.Fu = Fu = self.problem.make_Fu_nonconservative(self.u)
            self.Fu_wall = self.problem.make_Fu_nonconservative_wall(self.u)
            self.Fu_open = self.problem.make_Fu_nonconservative(self.u_bc)
            self.S = self.problem.make_Source(self.u, mom_form="nonconservative")
        elif self.swe_type == "linear":
            raise Exception(
                "Sorry, swe_type must be full for DGImplicitNonConservative, not %s"
                % self.swe_type
            )
        else:
            raise Exception(
                "Sorry, swe_type must either be linear or full, not %s" % self.swe_type
            )

        self.theta1 = theta1 = fe.Constant(self.domain, PETSc.ScalarType(theta))
        self.F = -inner(self.Fu, grad(self.p)) * dx
        self.add_bcs_to_weak_form()

        self.dt = self.problem.dt
        self.F += inner(self.S, self.p) * dx

        h_b = self.problem.h_b
        if self.swe_type == "full":
            self.Q = as_vector(self.problem._get_standard_vars(self.u, "h"))
            self.Qn = as_vector(self.problem._get_standard_vars(self.u_n, "h"))
            self.Qn_old = as_vector(self.problem._get_standard_vars(self.u_n_old, "h"))
        else:
            raise Exception(
                "Sorry, swe_type must either be linear or full, not %s" % self.swe_type
            )

        # BDF2
        self.dQdt = theta1 * fe.Constant(self.domain, PETSc.ScalarType(1 / self.dt)) * (
            1.5 * self.Q - 2 * self.Qn + 0.5 * self.Qn_old
        ) + (1 - theta1) * fe.Constant(self.domain, PETSc.ScalarType(1 / self.dt)) * (
            self.Q - self.Qn
        )

        self.F += inner(self.dQdt, self.p) * dx

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
        """Add boundary integrals for non-conservative DG."""
        boundary_conditions = self.problem.boundary_conditions
        ds_exterior = self.problem.ds
        n = FacetNormal(self.domain)

        if self.p_type == "DG":
            if self.swe_type == "full":
                if self.verbose:
                    self.log("Adding DG boundary conditions weakly")
                h, ux, uy = self.problem._get_standard_vars(self.u, "h")
                h_bc, ux_bc, uy_bc = self.problem._get_standard_vars(self.u_bc, "h")

                for condition in boundary_conditions:
                    if condition.type == "Open":
                        self.F += dot(dot(self.Fu_open, n), self.p) * ds_exterior(
                            condition.marker
                        )
                    if condition.type == "Wall":
                        self.F += dot(dot(self.Fu_wall, n), self.p) * ds_exterior(
                            condition.marker
                        )

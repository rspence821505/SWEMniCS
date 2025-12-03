"""SUPG stabilized implicit solver for shallow water equations.

This module contains the SUPGImplicit class which implements a
Streamline-Upwind Petrov-Galerkin (SUPG) stabilized solver.
"""

from dolfinx import fem as fe, cpp

try:
    from dolfinx.fem import functionspace
except ImportError:
    from dolfinx.fem import FunctionSpace as functionspace

from ufl import (
    TestFunction,
    TrialFunction,
    FacetNormal,
    as_matrix,
    as_vector,
    inner,
    sqrt,
    dx,
)

import numpy as np

from swemnics.physics.constants import g, R
from .cg_implicit import CGImplicit


class SUPGImplicit(CGImplicit):
    """SUPG stabilized implicit solver."""

    def project_L2(self, f, V):
        """Project function to L2 space."""
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(u, v) * dx
        L = inner(f, v) * dx
        problem = fe.petsc.LinearProblem(a, L, petsc_options={"ksp_type": "cg"})
        ux = problem.solve()
        ux.vector.ghostUpdate()
        return ux

    def init_weak_form(self):
        """Initialize weak form with SUPG stabilization."""
        super().init_weak_form()

        n = FacetNormal(self.domain)
        eps = 1e-8
        theta = self.theta
        dQdt = self.dQdt
        dQ_ncdt = self.dQ_ncdt

        # get element height
        tdim = self.domain.topology.dim
        self.domain.topology.create_connectivity(tdim, tdim)
        num_cells1 = self.domain.topology.index_map(tdim).size_local
        cells = np.arange(num_cells1, dtype=np.int32)

        try:
            h = cpp.mesh.h(self.domain, tdim, range(num_cells1))
        except TypeError:
            h = cpp.mesh.h(self.domain._cpp_object, tdim, cells)

        self.cellwise = functionspace(self.domain, ("DG", 0))
        height1 = fe.Function(self.cellwise)
        height1.x.array[:num_cells1] = h
        height1.x.petsc_vec.ghostUpdate()

        alpha = 0.25
        spherical = self.problem.spherical

        if self.problem.solution_var == "h":
            h, ux, uy = self.problem._get_standard_vars(self.u, "h")
            h_n, ux_n, uy_n = self.problem._get_standard_vars(self.u_n, "h")

            if self.swe_type == "full":
                factor = sqrt(ux_n * ux_n + uy_n * uy_n + g * (h_n))
                T1 = as_matrix(((ux, h, 0), (g, ux, 0), (0, 0, ux)))
                T2 = as_matrix(((uy, 0, h), (0, uy, 0), (g, 0, uy)))

                if self.wd:
                    S_nc = self.problem.make_Source(self.u, form="canonical")
                else:
                    S_temp = self.problem.make_Source(self.u, form="canonical")
                    S_nc = as_vector((S_temp[0], S_temp[1] / h, S_temp[2] / h))

            elif self.swe_type == "linear":
                alpha = 0.1
                h_b = self.problem.get_h_b(self.u)
                factor = sqrt(ux_n * ux_n + uy_n * uy_n + g * (h_b))
                T1 = as_matrix(((0, h_b, 0), (g, 0, 0), (0, 0, 0)))
                T2 = as_matrix(((0, 0, h_b), (0, 0, 0), (g, 0, 0)))
                S_temp = self.problem.make_Source_linearized(self.u, form="canonical")
                S_nc = as_vector((S_temp[0], S_temp[1], S_temp[2]))

            if spherical:
                if self.problem.projected:
                    factor = sqrt(
                        self.problem.S**2
                        * (self.u_n[1] * self.u_n[1] + self.u_n[2] * self.u_n[2])
                        + g * (self.u_n[0])
                    )
                    T1 = T1 * self.problem.S
                else:
                    T1 = T1 * self.problem.S / R
                    T2 = T2 / R

            tau_SUPG = as_vector(
                (
                    alpha * height1 / factor,
                    alpha * height1 / factor,
                    alpha * height1 / factor,
                )
            )

            # petrov terms for SUPG
            temp_x = as_vector(
                (
                    tau_SUPG[0] * self.p[0].dx(0),
                    tau_SUPG[1] * self.p[1].dx(0),
                    tau_SUPG[2] * self.p[2].dx(0),
                )
            )
            temp_y = as_vector(
                (
                    tau_SUPG[0] * self.p[0].dx(1),
                    tau_SUPG[1] * self.p[1].dx(1),
                    tau_SUPG[2] * self.p[2].dx(1),
                )
            )

            if spherical:
                self.F += (
                    inner(
                        dQ_ncdt + T1 * self.u.dx(0) + T2 * self.u.dx(1) + S_nc,
                        T1 * temp_x + T2 * temp_y,
                    )
                    * dx
                )
            else:
                self.F += (
                    inner(
                        dQ_ncdt + T1 * self.u.dx(0) + T2 * self.u.dx(1) + S_nc,
                        (T1 * temp_x + T2 * temp_y),
                    )
                    * dx
                )

"""Base solver class for steady-state shallow water equations.

This module contains the BaseSolver class which defines the foundation
for all solver classes in SWEMniCS.
"""

from dolfinx import fem as fe, nls, log
import sys

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
    grad,
    dx,
    ds,
)

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from swemnics.physics.constants import g, R
from swemnics.utils.fem_utilities import create_element, create_mixed_element
from swemnics.utils.solver_storage import SolverStateStorage

from typing import Literal


class BaseSolver:
    """Defines a base solver class that solves the steady-state shallow-water equations."""

    def __init__(
        self,
        problem,
        theta=0.5,
        p_degree=[1, 1],
        p_type: Literal["CG", "DG"] = "CG",
        swe_type="full",
        verbose=True,
    ):
        r"""Initialize the solver.

        Args:
          problem: Problem class defining the mesh and boundary conditions.
          theta: Time stepping scheme parameter. The temporal derivative is approximated as
        .. math:: \frac{\partial Q}{\partial t} = \Delta t \theta (\frac{3}{2}Q_n - 2Q_{n-1} + \frac{1}{2}Q_{n-2}) + \Delta t(1-\theta)(Q_n-Q_{n-1}).

            Consequently, the scheme is Implicit Euler if theta is 0, Crank-Nicolson if theta is 1, and BDF2 if theta is .5.
          p_degree: A tuple with two integers. The first indicates the polynomial degree for the mass variable, and the second the degree for the momentum variable.
          p_type: Type of element to use - either 'CG' for continuous Galerkin or 'DG' for discontinuous Galerkin.
          This is usually set by a subclass and not the user.
          swe_type: Form of the shallow water equations to solve. Either 'full' for the full nonlinear equations (default) or 'linear' for the linearized equations. In general, 'linear' should only be used in very specific circumstances, such as verifying convergence rates to an analytic solution.
        """

        self.mpi_rank = MPI.COMM_WORLD.Get_rank()
        self.problem = problem
        self.theta = theta
        self.p_degree = p_degree
        self.p_type = p_type
        self.names = ["eta", "u", "v"]
        self.swe_type = swe_type
        self.F_no_dt = None
        self.verbose = verbose

        # Use refactored storage
        self.storage = SolverStateStorage()

        # Backward compatibility: provide direct access to storage arrays
        # This allows existing code to continue using self.saved_states, etc.
        self.saved_adjoints = self.storage.saved_adjoints
        self.saved_states = self.storage.saved_states
        self.dry_nodes = self.storage.dry_nodes
        self.saved_true_bathy = self.storage.saved_true_bathy
        self.saved_bathy = self.storage.saved_bathy
        self.saved_jacobians = self.storage.saved_jacobians

        self.init_fields()
        self.init_weak_form()

    @property
    def TAU(self):
        return self.problem.TAU

    @property
    def domain(self):
        return self.problem.mesh

    @property
    def wd(self):
        return self.problem.wd

    def init_fields(self):
        """Initialize the relevant elements, functions, and function spaces."""

        # Use refactored element creation
        el_h = create_element(self.domain, self.p_type, self.p_degree[0])
        el_vel = create_element(self.domain, self.p_type, self.p_degree[1], shape=(2,))
        me = create_mixed_element([el_h, el_vel])
        self.V = functionspace(self.domain, me)

        # for plotting
        self.V_vel = self.V.sub(1).collapse()[0]
        self.V_scalar = self.V.sub(0).collapse()[0]

        # split these up
        self.u = fe.Function(self.V)
        self.hel, self.vel_sol = self.u.split()

        self.p1, self.p2 = TestFunctions(self.V)

        # try this to minimize rewrite but may want to change in future
        self.p = as_vector((self.p1, self.p2[0], self.p2[1]))

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, new_v):
        self._V = new_v
        self.problem.init_V(new_v)

    def init_weak_form(self):
        """Initialize the weak form."""

        if self.swe_type == "full":
            self.Fu = Fu = self.problem.make_Fu(self.u)
        elif self.swe_type == "linear":
            self.Fu = Fu = self.problem.make_Fu_linearized(self.u)
        n = FacetNormal(self.domain)
        self.F = (
            -inner(Fu, grad(self.p)) * dx
            + dot(dot(Fu, n), self.p) * ds
            - dot(self.problem.get_rhs(), self.p) * dx
        )

    def solve(self, u_init=lambda x: np.ones(x.shape), solver_parameters={}):
        """Solve the steady-state equation and save the result in u_sol."""

        # set initial guess
        self.u.interpolate(u_init)

        prob = fe.petsc.NonlinearProblem(self.F, self.u, bcs=self.problem.dirichlet_bcs)

        # the problem appears to be that the residual is humongous. . .
        res = fe.form(self.F)
        test_res = fe.petsc.create_vector(res)
        fe.petsc.assemble_vector(test_res, res)
        print(f"Calling NewtonSolver", file=sys.stdout)
        solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, prob)
        print("Solver created", file=sys.stdout)
        for k, v in solver_parameters.items():
            setattr(solver, k, v)
        solver.report = True
        solver.convergence_criterion = "incremental"
        solver.error_on_nonconvergence = False
        log.set_log_level(log.LogLevel.INFO)
        solver.solve(self.u)

        return self.u

    def log(self, *msg):
        if self.mpi_rank == 0 and self.verbose:
            print(*msg)

    def print_config(self):
        """Print solver configuration information.

        Prints solver settings including SWE type, wetting/drying status,
        state vector dimensions, polynomial degrees, and time stepping parameters.
        Only prints from MPI rank 0.
        """
        if self.mpi_rank != 0:
            return

        print("\n" + "=" * 70)
        print("Solver Configuration")
        print("=" * 70)

        # SWE type
        print(f"SWE type:            {self.swe_type}")

        # Wetting/drying
        wd_status = "activated" if self.wd else "NOT activated"
        print(f"Wetting/drying:      {wd_status}")

        # Polynomial degrees
        print(f"Polynomial degrees:  h={self.p_degree[0]}, vel={self.p_degree[1]}")

        # Element type
        print(f"Element type:        {self.p_type}")

        # State vector size
        state_size = self.u.x.array.shape[0]
        num_components = 3  # h, u, v
        print(f"State vector size:   {state_size}, {num_components} (h, u, v)")

        # Time stepping info if available
        if hasattr(self, 'problem') and hasattr(self.problem, 'nt'):
            print(f"Number of timesteps: {self.problem.nt}")
        if hasattr(self, 'problem') and hasattr(self.problem, 'dt'):
            print(f"Time step size:      {self.problem.dt}")

        # Theta parameter
        theta_schemes = {0: "Implicit Euler", 0.5: "BDF2", 1.0: "Crank-Nicolson"}
        scheme_name = theta_schemes.get(self.theta, f"Custom (theta={self.theta})")
        print(f"Time scheme:         {scheme_name}")

        # Data saving configuration (if time_loop has been called)
        # Check if we have data manager information from a time_loop call
        if hasattr(self, '_last_data_manager_config'):
            config = self._last_data_manager_config
            save_mode = config.get('mode', 'unknown')
            if save_mode == 'observation':
                n_obs = config.get('n_observations', 0)
                print(f"Timesteps saved:     {n_obs} observations")
            elif save_mode == 'all_timesteps':
                print(f"Timesteps saved:     All timesteps")
            else:
                print(f"Timesteps saved:     None (forward only)")

        print("=" * 70)

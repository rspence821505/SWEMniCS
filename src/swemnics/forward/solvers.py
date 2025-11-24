"""Solver classes for steady-state and time-dependent problems.

Each Solver requires a Problem class to initialize, and Solvers inherit from one another.
New numerical methods can be implemented by inheriting from the classes in this file.
"""

from pathlib import Path
from dolfinx import fem as fe, nls, log, geometry, io, cpp, mesh
import sys

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
    dot,
    inner,
    grad,
    dx,
    ds,
    dS,
    jump,
    avg,
    sqrt,
    conditional,
    div,
    elem_mult,
    TestFunctions,
)

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from swemnics.forward.newton import CustomNewtonProblem
from swemnics.physics.constants import g, R

# Import refactored modules
from swemnics.utils.fem_utilities import create_element, create_mixed_element
from swemnics.utils.solver_storage import SolverStateStorage
from swemnics.utils.timestep_manager import TimeStepDataManager
from swemnics.utils.observation_stations import StationManager
from swemnics.utils.visualization import SolverVisualizer

from petsc4py.PETSc import ScalarType
from typing import Literal, Optional, Sequence


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

        if self.verbose:
            self.log("SWE TYPE", self.swe_type)

        if self.wd:
            if self.verbose:
                self.log("Wetting drying activated \n")
        else:
            if self.verbose:
                self.log("Wetting drying NOT activated \n")

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


class CGImplicit(BaseSolver):
    """Base class for all time stepping solvers."""

    def __init__(self, *args, **kwargs):
        """Initialize CGImplicit solver with station manager and visualizer."""
        super().__init__(*args, **kwargs)

        # Initialize station manager (will be configured in init_stations)
        self.station_manager = None

        # Initialize visualizer (will be configured in initialize_video)
        self.visualizer = None

    def init_fields(self):
        super().init_fields()
        self.u_n = fe.Function(self.V)
        self.u_n.name = "u_n"
        # for second order timestep need n-1
        self.u_n_old = fe.Function(self.V)
        self.u_n_old.name = "u_n_old"

    def add_bcs_to_weak_form(self):
        """Add boundary integrals to the variational form.

        This method may need to be overridden when implementing a solver with trace variables or an alternate approach to boundary conditions.
        """
        boundary_conditions = self.problem.boundary_conditions
        ds_exterior = self.problem.ds
        n = FacetNormal(self.domain)

        if self.p_type == "CG":
            if self.verbose:
                self.log("Adding CG boundary conditions weakly")
            # loop through boundary conditions
            for condition in boundary_conditions:
                if condition.type == "Open":
                    self.F += dot(dot(self.Fu_open, n), self.p) * ds_exterior(
                        condition.marker
                    )
                if condition.type == "Wall":
                    self.F += dot(dot(self.Fu_wall, n), self.p) * ds_exterior(
                        condition.marker
                    )

    def set_initial_condition(self):
        """Set the initial condition.

        The water column height is assumed to be equal to the bathymetry unless the Problem specifies a different initial condition.
        If the Problem doesn't specify a velocity initial condition, it is assumed to be zero.
        """
        if self.problem.solution_var == "h" or self.problem.solution_var == "flux":
            if self.verbose:
                self.log("setting initial condition")
            # if the initial condition is specified set this, if not assume level starting condition
            if self.problem.h_init is None:
                self.u_n.sub(0).interpolate(
                    fe.Expression(
                        self.problem.h_b, self.V.sub(0).element.interpolation_points()
                    )
                )
            else:
                self.u_n.sub(0).interpolate(
                    fe.Expression(
                        self.problem.h_init,
                        self.V.sub(0).element.interpolation_points(),
                    )
                )
            if self.problem.vel_init is None:
                # by default assume 0 velocity everywhere
                self.u_n.sub(1).interpolate(
                    fe.Expression(
                        as_vector(
                            [
                                fe.Constant(self.domain, ScalarType(0.0)),
                                fe.Constant(self.domain, ScalarType(0.0)),
                            ]
                        ),
                        self.V.sub(1).element.interpolation_points(),
                    )
                )
            else:
                self.u_n.sub(1).interpolate(
                    fe.Expression(
                        self.problem.vel_init,
                        self.V.sub(1).element.interpolation_points(),
                    )
                )

        if self.problem.solution_var == "eta":
            if self.problem.h_init is not None:
                self.u_n.sub(0).interpolate(
                    fe.Expression(
                        self.problem.h_init - self.problem.h_b,
                        self.V.sub(0).element.interpolation_points(),
                    )
                )

        # apply dirichlet conditions
        self.problem.update_boundary()
        if self.problem.dof_open.size != 0:
            self.u_n.x.array[self.problem.dof_open] = self.problem.u_bc.x.array[
                self.problem.dof_open
            ]
        if self.problem.uy_dofs_closed.size != 0:
            self.u_n.x.array[self.problem.uy_dofs_closed] = self.problem.u_bc.x.array[
                self.problem.uy_dofs_closed
            ]
        if self.problem.ux_dofs_closed.size != 0:
            self.u_n.x.array[self.problem.ux_dofs_closed] = self.problem.u_bc.x.array[
                self.problem.ux_dofs_closed
            ]

        self.u_n_old.sub(0).x.array[:] = self.u_n.sub(0).x.array[:]
        self.u_n_old.sub(1).x.array[:] = self.u_n.sub(1).x.array[:]

    def init_weak_form(self):
        """Initialize the weak form.

        This method is typically overridden by any child class implementing a different numerical method.
        """
        theta = self.theta
        self.set_initial_condition()
        # create fluxes
        self.u_bc = as_vector((self.problem.u_bc[0], self.u[1], self.u[2]))
        if self.swe_type == "full":
            self.Fu = self.problem.make_Fu(self.u)
            self.Fu_wall = self.problem.make_Fu_wall(self.u)
            self.Fu_open = self.problem.make_Fu(self.u_bc)
            self.S = self.problem.make_Source(self.u)
        elif self.swe_type == "linear":
            self.Fu = self.problem.make_Fu_linearized(self.u)
            self.Fu_wall = self.problem.make_Fu_wall_linearized(self.u)
            self.Fu_open = self.problem.make_Fu_linearized(self.u_bc)
            self.S = self.problem.make_Source_linearized(self.u)
        else:
            raise Exception(
                "Sorry, swe_type must either be linear or full, not %s" % self.swe_type
            )

        # weak form
        self.theta1 = theta1 = fe.Constant(self.domain, PETSc.ScalarType(theta))

        # start adding to residual
        self.F = -inner(self.Fu, grad(self.p)) * dx

        self.dt = self.problem.dt

        # add RHS to residual
        self.F += inner(self.S, self.p) * dx

        # add contribution from time step
        if self.swe_type == "full":
            self.Q = as_vector(self.problem._get_standard_vars(self.u, "flux"))
            self.Qn = as_vector(self.problem._get_standard_vars(self.u_n, "flux"))
            self.Qn_old = as_vector(
                self.problem._get_standard_vars(self.u_n_old, "flux")
            )
        elif self.swe_type == "linear":
            self.Q = as_vector((self.u[0], self.u[1], self.u[2]))
            self.Qn = as_vector((self.u_n[0], self.u_n[1], self.u_n[2]))
            self.Qn_old = as_vector((self.u_n_old[0], self.u_n_old[1], self.u_n_old[2]))
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
        u = as_vector(self.problem._get_standard_vars(self.u, "h"))
        u_n = as_vector(self.problem._get_standard_vars(self.u_n, "h"))
        u_n_old = as_vector(self.problem._get_standard_vars(self.u_n_old, "h"))
        self.dQ_ncdt = theta1 * fe.Constant(
            self.domain, PETSc.ScalarType(1 / self.dt)
        ) * (1.5 * u - 2 * u_n + 0.5 * u_n_old) + (1 - theta1) * fe.Constant(
            self.domain, PETSc.ScalarType(1 / self.dt)
        ) * (
            u - u_n
        )

        self.add_bcs_to_weak_form()
        self.F += inner(self.dQdt, self.p) * dx

    def solve_init(self, solver_parameters={}):
        """Initialize the Newton solver"""
        Newton_obj = CustomNewtonProblem(self, solver_parameters=solver_parameters)
        return Newton_obj

    def solve_timestep(self, solver, store_jacobian=False):
        """Solve the nonlinear problem at the current time step.

        Args:
        solver: Newton solver (CustomNewtonProblem instance).
        store_jacobian: If True, request Jacobian from Newton solver (for 4D-Var).

        Returns:
        J: Jacobian matrix if store_jacobian=True, else None.
            The Jacobian is ∂R/∂u evaluated at the converged solution.

        Raises:
        RuntimeError: If negative water depths detected.
        ValueError: If store_jacobian=True but Jacobian extraction fails.

        Notes:
        For 4D-Var data assimilation, the Jacobian is automatically copied by
        the Newton solver to prevent overwriting in subsequent timesteps.
        """
        try:
            if store_jacobian:
                _, J = solver.solve(self.u, return_jacobian=True)

                # Validate Jacobian was successfully extracted
                if J is None:
                    raise ValueError(
                        "Jacobian extraction failed: Newton solver returned None. "
                        "Ensure CustomNewtonProblem is being used (not NewtonSolver)."
                    )

                # Validate it's a proper PETSc matrix
                if not isinstance(J, PETSc.Mat):
                    raise ValueError(
                        f"Expected PETSc.Mat for Jacobian, got {type(J)}. "
                        "This indicates an issue with Newton solver implementation."
                    )

                # Verify matrix is assembled and has nonzero size
                if J.assembled == False:
                    raise ValueError(
                        "Jacobian matrix is not assembled. "
                        "This indicates an issue with Newton solver implementation."
                    )

                size = J.getSize()
                if size[0] == 0 or size[1] == 0:
                    raise ValueError(
                        f"Jacobian has invalid size {size}. "
                        "Expected non-zero square matrix."
                    )

                # Optional: verify it's square (required for adjoint solve)
                if size[0] != size[1]:
                    raise ValueError(
                        f"Jacobian must be square for adjoint computation, got {size}"
                    )

                # Log success in verbose mode
                if self.verbose and self.mpi_rank == 0:
                    nnz = J.getInfo()["nz_used"]
                    self.log(
                        f"  Jacobian extracted: {size[0]}x{size[1]}, nnz={int(nnz)}"
                    )

                # NOTE: Newton solver already returns a copy (J_final = A.copy()),
                # so we don't need to copy again here. However, to be
                # extra defensive, you could uncomment the following line:
                # J = J.copy()

                return J
            else:
                solver.solve(self.u, return_jacobian=False)
                return None

        except RuntimeError as e:
            # Handle negative water depth errors
            h_fun = self.u.sub(0).collapse()
            hvals = h_fun.x.array[:]
            min_h = hvals.min()
            print(f"Min h on process {self.mpi_rank}, {min_h}")
            bad_h = hvals < 0
            coords = h_fun.function_space.tabulate_dof_coordinates()[:, :2]
            coords = self.problem.reverse_projection(coords)
            print(f"first coords of negative h on {self.mpi_rank}", coords[bad_h][:1])
            if not self.mpi_rank:
                raise
        except ValueError as e:
            # Re-raise validation errors with context
            if "Jacobian" in str(e):
                print(f"ERROR on rank {self.mpi_rank}: {e}")
                if not self.mpi_rank:
                    raise
            else:
                raise

    def update_solution(self):
        """Advance solution to next time step."""
        self.u_n_old.x.array[:] = self.u_n.x.array[:]
        self.u_n.x.array[:] = self.u.x.array[:]

        # dirichlet boundary
        self.problem.advance_time()

        # update any possible dirichlet boundaries
        if self.problem.dof_open.size != 0:
            self.u.x.array[self.problem.dof_open] = self.problem.u_bc.x.array[
                self.problem.dof_open
            ]
        if self.problem.uy_dofs_closed.size != 0:
            self.u.x.array[self.problem.uy_dofs_closed] = self.problem.u_bc.x.array[
                self.problem.uy_dofs_closed
            ]
        if self.problem.ux_dofs_closed.size != 0:
            self.u.x.array[self.problem.ux_dofs_closed] = self.problem.u_bc.x.array[
                self.problem.ux_dofs_closed
            ]

    # Station management methods (delegated to StationManager)
    def init_stations(self, points):
        """Initialize recording stations. Delegates to StationManager."""
        if self.station_manager is None:
            self.station_manager = StationManager(
                self.domain, self.V_scalar, self.problem.h_b, verbose=self.verbose
            )

        local_points = self.station_manager.init_stations(points)
        # Keep backward compatibility by storing attributes
        self.cells = self.station_manager.cells
        self.station_index = self.station_manager.station_index
        self.station_bathy = self.station_manager.station_bathy

        return local_points

    def record_stations(self, u_sol, points_on_proc):
        """Record time series at stations. Delegates to StationManager."""
        if self.station_manager is None:
            raise RuntimeError("Must call init_stations() before record_stations()")
        return self.station_manager.record_stations(u_sol, self.problem.solution_var)

    def check_dry_nodes(
        self, solution, evaluation_points, save_bathy=False, save_true_bathy=False
    ):
        """Check for dry nodes at observation points. Delegates to StationManager."""
        if self.station_manager is None:
            raise RuntimeError("Must call init_stations() before check_dry_nodes()")

        water_height, dry_node_indices = self.station_manager.check_dry_nodes(
            solution, evaluation_points, save_bathy, save_true_bathy
        )

        # Save to storage if requested
        if save_true_bathy:
            bathy = self.problem.h_b.eval(evaluation_points, self.station_manager.cells)
            self.storage.save_bathymetry(bathy, is_true_bathy=True)
            self.storage.save_dry_nodes(dry_node_indices)
        if save_bathy:
            bathy = self.problem.h_b.eval(evaluation_points, self.station_manager.cells)
            self.storage.save_bathymetry(bathy, is_true_bathy=False)

        return water_height, dry_node_indices

    def gather_station(self, root, local_stats, local_vals):
        """Gather station data to root process. Delegates to StationManager."""
        if self.station_manager is None:
            raise RuntimeError("Must call init_stations() before gather_station()")
        return self.station_manager.gather_station(root, local_stats, local_vals)

    # Visualization methods (delegated to SolverVisualizer)
    def initialize_video(self, filename):
        """Initialize video output. Delegates to SolverVisualizer."""
        if self.visualizer is None:
            self.visualizer = SolverVisualizer(
                self.domain,
                self.V_scalar,
                self.V_vel,
                self.problem,
                verbose=self.verbose,
            )
        self.visualizer.initialize_video(filename)

        # Keep backward compatibility by storing plot functions as attributes
        self.eta_plot = self.visualizer.eta_plot
        self.h_plot = self.visualizer.h_plot
        self.vel_plot = self.visualizer.vel_plot
        self.bathy_plot = self.visualizer.bathy_plot
        self.wse_writer = self.visualizer.wse_writer
        self.h_writer = self.visualizer.h_writer
        self.vel_writer = self.visualizer.vel_writer
        self.bathy_writer = self.visualizer.bathy_writer

    def plot_frame(self):
        """Plot a frame of the state. Delegates to SolverVisualizer."""
        if self.visualizer is None:
            raise RuntimeError("Must call initialize_video() before plot_frame()")
        self.visualizer.plot_frame(self.u, self.problem.t)

    def finalize_video(self):
        """Close video writers. Delegates to SolverVisualizer."""
        if self.visualizer is not None:
            self.visualizer.finalize_video()

    def plot_func(self, func, name="eta"):
        """Plot a function interactively with PyVista. Delegates to SolverVisualizer."""
        if self.visualizer is None:
            self.visualizer = SolverVisualizer(
                self.domain,
                self.V_scalar,
                self.V_vel,
                self.problem,
                verbose=self.verbose,
            )
        self.visualizer.plot_func_interactive(func, name)

    # Storage methods (delegated to SolverStateStorage)
    def save_adjoints(self):
        """Save adjoint Jacobian for parallel adjoint solve."""
        A_tlm = self.solver.assemble_A()
        A_adjoint = A_tlm.transpose()
        A_adjoint.assemble()
        self.storage.save_adjoint(A_adjoint)

    def save_jacobians(self, J):
        """Save Jacobian matrix for 4D-Var adjoint computation."""
        if J is not None:
            self.storage.save_jacobian(J)

    def save_states(self, water_height=None, dry_node_indices=None):
        """Save global state vector with optional wetting/drying adjustments."""
        u_sol = self.u.x.array.copy().flatten()

        if dry_node_indices is not None and water_height is not None:
            V_sub = self.problem.V.sub(0)
            _, sub_map = V_sub.collapse()
            water_height[dry_node_indices] = 0.0
            u_sol[sub_map] = water_height.copy().flatten()

        self.storage.save_state(u_sol)

    def time_loop(
        self,
        solver_parameters,
        stations=[],
        plot_every=999999,
        plot_name="debug_tide",
        u_0=None,
        save_state=False,
        adjoint_method=False,
        save_bathy=False,
        save_true_bathy=False,
        make_wet=False,
        store_jacobians=False,
        observation_times=None,
    ):
        """Time-stepping loop with optional Jacobian storage for 4D-Var.

        Args:
            solver_parameters: Parameters passed to the nonlinear solver.
            stations: Monitoring station coordinates.
            plot_every: Plotting cadence.
            plot_name: Base name for visualization output.
            u_0: Optional initial condition override.
            save_state: Save every state (legacy mode).
            adjoint_method: Whether to assemble adjoint matrices.
            save_bathy: Save bathymetry samples when applying wetting/drying.
            save_true_bathy: Save true bathymetry and dry node indices.
            make_wet: Apply wetting/drying adjustments before saving.
            store_jacobians: Store Jacobians for 4D-Var adjoint computation.
            observation_times: Optional iterable of timestep indices to save.
        """

        if store_jacobians:
            self.storage.saved_jacobians = []
            if self.verbose:
                self.log("4D-Var mode: Jacobians will be stored during forward solve")

        if self.verbose:
            self.log("calling time loop")

        self.points_on_proc = local_points = self.init_stations(stations)
        self.station_data = np.zeros((self.problem.nt + 1, local_points.shape[0], 3))

        # set initial guess for the first time step
        if u_0 is None:
            self.u.x.array[:] = self.u_n.x.array[:]
        else:
            self.u_n.x.array[:] = u_0.x.array[:]
            self.u.x.array[:] = self.u_n.x.array[:]

        self.solver = solver = self.solve_init(solver_parameters=solver_parameters)

        if self.verbose:
            self.log("plot every", plot_every)
            self.log("nt", self.problem.nt)

        if plot_every <= self.problem.nt:
            if self.verbose:
                self.log("creating video")
            self.initialize_video(plot_name)
            self.plot_frame()

        effective_save_state = save_state or (observation_times is not None)
        data_manager = TimeStepDataManager(
            solver=self,
            save_state=effective_save_state,
            make_wet=make_wet,
            save_bathy=save_bathy,
            save_true_bathy=save_true_bathy,
            store_jacobians=store_jacobians,
            save_adjoints=adjoint_method,
            observation_times=observation_times,
            verbose=self.verbose,
        )

        data_manager.save_timestep(timestep=0, local_points=local_points)

        # take first 2 steps with implicit Euler
        self.theta1.value = 0
        for a in range(min(2, self.problem.nt)):
            if self.verbose:
                self.log("Time Step Number", a, "Out of", self.problem.nt)
                self.log(a / self.problem.nt * 100, "% Complete")
            self.update_solution()

            should_get_jacobian = store_jacobians and data_manager.should_save_at(a + 1)
            J = self.solve_timestep(solver, store_jacobian=should_get_jacobian)

            if a % plot_every == 0 and plot_every <= self.problem.nt:
                self.plot_frame()

            data_manager.save_timestep(
                timestep=a + 1,
                J=J,
                local_points=local_points,
            )

        # switch to high order time stepping (BDF2)
        self.theta1.value = self.theta
        for a in range(2, self.problem.nt):
            if self.verbose:
                self.log("Time Step Number", a, "Out of", self.problem.nt)
                self.log(a / self.problem.nt * 100, "% Complete")
            self.update_solution()
            should_get_jacobian = store_jacobians and data_manager.should_save_at(a + 1)
            J = self.solve_timestep(solver, store_jacobian=should_get_jacobian)

            data_manager.save_timestep(
                timestep=a + 1,
                J=J,
                local_points=local_points,
            )

            if a % plot_every == 0:
                self.plot_frame()

        if plot_every <= self.problem.nt:
            self.finalize_video()

        inds, vals = None, None
        self.vals = vals
        self.inds = inds

        if self.verbose:
            summary = data_manager.get_summary()
            self.log(f"Time loop complete: {summary}")
            if store_jacobians:
                self.log(
                    f"Stored {len(self.storage.saved_jacobians)} Jacobians for 4D-Var"
                )

        # Optionally evaluate and print L2 error
        if self.problem.check_solution_def is not None:
            print("Checking solution at ", self.problem.t)
            e0 = self.problem.check_solution(self.u, self.V, self.problem.t)
            print("L2 error at t=", str(self.problem.t), " is ", str(e0))

        return (self.u, vals)


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


# Solver factory
_get_solver = {
    "CG": CGImplicit,
    "SUPG": SUPGImplicit,
    "DGCG": DGCGImplicit,
    "DG": DGImplicit,
    "DGNC": DGImplicitNonConservative,
}


def get_solver(solver_type: str) -> BaseSolver:
    """Get solver class by type string.

    Args:
        solver_type: One of 'CG', 'SUPG', 'DGCG', 'DG', 'DGNC'

    Returns:
        Solver class

    Raises:
        ValueError: If solver_type is unknown
    """
    try:
        return _get_solver[solver_type.upper()]
    except KeyError:
        raise ValueError(
            f"Unknown solver type {solver_type}, options available are: {list(_get_solver.keys())}"
        )

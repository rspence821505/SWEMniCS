"""CG implicit time-stepping solver for shallow water equations.

This module contains the CGImplicit class which serves as the base class
for all time-stepping solvers in SWE4DVar.
"""

from dolfinx import fem as fe

try:
    from dolfinx.fem import functionspace
except ImportError:
    from dolfinx.fem import FunctionSpace as functionspace

from ufl import (
    FacetNormal,
    as_vector,
    dot,
    inner,
    grad,
    dx,
)

from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
import numpy as np
import os

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from swe4dvar.forward.newton import CustomNewtonProblem
from swe4dvar.utils.timestep_manager import TimeStepDataManager
from swe4dvar.utils.observation_stations import StationManager
from swe4dvar.utils.visualization import SolverVisualizer
from swe4dvar.utils.compat import interpolation_points as _ipts, create_vector_from_form as _cvf

from .base_solver import BaseSolver


# ============================================================================
# Forward-solve diagnostics (env-gated). When SWE4DVAR_FORWARD_DIAG_CSV is set
# to a path, every solve_timestep call appends a row containing:
#   eval_id, timestep, t_seconds, newton_iters, newton_converged,
#   correction_norm, residual_norm, min_h_global, max_u_global, max_v_global
#
# eval_id auto-increments at every solve_init() (one per forward-solve start).
# Only rank 0 writes; min_h / max_u / max_v reductions are collective.
#
# Used to diagnose cycling-DA Newton failures inside the actual cost-function
# forward path — see docs/idealized_inlet_cycling_basin_diagnosis.md.
# ============================================================================
_FWD_DIAG = {
    "csv_path": os.environ.get("SWE4DVAR_FORWARD_DIAG_CSV", None),
    "eval_id": 0,
    "header_written": False,
}


def _fwd_diag_enabled():
    return _FWD_DIAG["csv_path"] is not None


def _fwd_diag_bump_eval_id():
    """Increment eval_id at the start of each forward solve (solve_init)."""
    if _fwd_diag_enabled():
        _FWD_DIAG["eval_id"] += 1


def _fwd_diag_log_step(comm, mpi_rank, solver_obj, newton_solver,
                      timestep, t_seconds,
                      h_local_indices, uv_local_indices,
                      newton_converged, n_iter, correction_norm, residual_norm):
    """Append one diagnostic row for this timestep.

    Collective: all ranks call this; min/max reductions use comm.allreduce.
    Only rank 0 writes the row.
    """
    if not _fwd_diag_enabled():
        return

    arr = solver_obj.u_n.x.array[:]
    if h_local_indices is not None and h_local_indices.size > 0:
        h_local = arr[h_local_indices]
        local_min_h = float(h_local.min())
    else:
        local_min_h = float("inf")
    if uv_local_indices is not None and uv_local_indices.size > 0:
        uv_local = arr[uv_local_indices]
        # Interleaved [u0, v0, u1, v1, ...]
        local_max_u = float(np.max(np.abs(uv_local[0::2]))) if uv_local.size > 1 else 0.0
        local_max_v = float(np.max(np.abs(uv_local[1::2]))) if uv_local.size > 1 else 0.0
    else:
        local_max_u = 0.0
        local_max_v = 0.0

    from mpi4py import MPI as _MPI
    global_min_h = comm.allreduce(local_min_h, op=_MPI.MIN)
    global_max_u = comm.allreduce(local_max_u, op=_MPI.MAX)
    global_max_v = comm.allreduce(local_max_v, op=_MPI.MAX)

    if mpi_rank != 0:
        return

    # Lazy CSV write — header on first call
    path = _FWD_DIAG["csv_path"]
    if not _FWD_DIAG["header_written"]:
        try:
            with open(path, "w") as f:
                f.write("eval_id,timestep,t_seconds,newton_iters,"
                        "newton_converged,correction_norm,residual_norm,"
                        "min_h_global,max_u_global,max_v_global\n")
            _FWD_DIAG["header_written"] = True
        except Exception as e:
            print(f"[fwd-diag] failed to write header to {path}: {e}",
                  flush=True)
            return

    row = (f"{_FWD_DIAG['eval_id']},{int(timestep)},{float(t_seconds):.6f},"
           f"{int(n_iter)},{int(bool(newton_converged))},"
           f"{float(correction_norm):.6e},{float(residual_norm):.6e},"
           f"{float(global_min_h):.6e},{float(global_max_u):.6e},"
           f"{float(global_max_v):.6e}\n")
    try:
        with open(path, "a") as f:
            f.write(row)
    except Exception as e:
        print(f"[fwd-diag] failed to append row: {e}", flush=True)


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
            # if the initial condition is specified set this, if not assume level starting condition
            if self.problem.h_init is None:
                self.u_n.sub(0).interpolate(
                    fe.Expression(
                        self.problem.h_b, _ipts(self.V.sub(0).element)
                    )
                )
            else:
                self.u_n.sub(0).interpolate(
                    fe.Expression(
                        self.problem.h_init,
                        _ipts(self.V.sub(0).element),
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
                        _ipts(self.V.sub(1).element),
                    )
                )
            else:
                self.u_n.sub(1).interpolate(
                    fe.Expression(
                        self.problem.vel_init,
                        _ipts(self.V.sub(1).element),
                    )
                )

        if self.problem.solution_var == "eta":
            if self.problem.h_init is not None:
                self.u_n.sub(0).interpolate(
                    fe.Expression(
                        self.problem.h_init - self.problem.h_b,
                        _ipts(self.V.sub(0).element),
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

        # Initialize current solution to match initial condition
        # This ensures solve_timestep works when called directly (without time_loop)
        self.u.x.array[:] = self.u_n.x.array[:]

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

    def solve_init(self, solver_parameters={}, newton_diagnostics_config=None):
        """Initialize the Newton solver with optional diagnostics configuration.

        Args:
            solver_parameters: Parameters for the Newton solver
            newton_diagnostics_config: Optional dict with diagnostics configuration.
                Keys: 'print_to_console', 'log_file', 'store_history', 'verbose'
                If None, uses default diagnostics based on self.verbose

        Returns:
            CustomNewtonProblem instance
        """
        # Forward-solve diagnostics: bump eval_id every time a new forward
        # solve begins (solve_init is called once at the top of time_loop).
        # Lets the diag CSV correlate timesteps to specific cost-function
        # evaluations.
        _fwd_diag_bump_eval_id()
        from swe4dvar.utils.newton_diagnostics import NewtonDiagnostics

        # Create diagnostics instance if config provided
        diagnostics = None
        if newton_diagnostics_config is not None:
            diagnostics = NewtonDiagnostics(**newton_diagnostics_config)
            # Print configuration message (MPI-aware)
            diagnostics.print_config()

        Newton_obj = CustomNewtonProblem(
            self, solver_parameters=solver_parameters, diagnostics=diagnostics
        )
        return Newton_obj

    def solve_timestep(self, solver, store_jacobian=False, timestep=None, time=None):
        """Solve the nonlinear problem at the current time step.

        Args:
        solver: Newton solver (CustomNewtonProblem instance).
        store_jacobian: If True, request Jacobian from Newton solver (for 4D-Var).
        timestep: Optional timestep number for diagnostics tracking.
        time: Optional simulation time for diagnostics tracking.

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
        # MPI-safe failure path. Newton.solve raises on ALL ranks when Newton
        # doesn't converge (correction_norm is a collective PETSc op, so
        # `converged` is consistent), so each rank enters the except block.
        # Previously only rank 0 re-raised after the local diagnostics — other
        # ranks swallowed the exception and returned to time_loop, causing
        # rank-0 to unwind through Python while ranks 1..N-1 proceeded to the
        # next timestep's collective assemble and deadlocked waiting for rank 0
        # (observed on Vista OpenMPI in runs 678950 / 678961).
        #
        # Collective-safe pattern: capture local status in the except handler,
        # do no raise there, then allreduce the status across ranks and have
        # EVERY rank raise an identical exception when any rank reports failure.
        _runtime_fail = 0
        _value_fail = 0
        _local_err_msg = ""
        _local_result = None

        # Compute h/uv local DOF indices once (cheap; cached on self after first
        # call) so the per-step diagnostic logger can grab min(h)/max(|u|,|v|)
        # without re-introspecting the function space each step.
        _diag_h_idx = None
        _diag_uv_idx = None
        if _fwd_diag_enabled():
            cached = getattr(self, "_fwd_diag_idx_cache", None)
            if cached is None:
                n_local = (self.V.dofmap.index_map.size_local
                           * self.V.dofmap.index_map_bs)
                _, h_map = self.V.sub(0).collapse()
                h_idx = np.asarray(h_map, dtype=np.int64)
                h_idx = h_idx[h_idx < n_local]
                _, uv_map = self.V.sub(1).collapse()
                uv_idx = np.asarray(uv_map, dtype=np.int64)
                uv_idx = uv_idx[uv_idx < n_local]
                self._fwd_diag_idx_cache = (h_idx, uv_idx)
                cached = self._fwd_diag_idx_cache
            _diag_h_idx, _diag_uv_idx = cached

        try:
            if store_jacobian:
                _, J = solver.solve(
                    self.u, return_jacobian=True, timestep=timestep, time=time
                )

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
                # NOTE: J.getInfo() is collective in PETSc - must call on all ranks
                if self.verbose:
                    nnz = J.getInfo()["nz_used"]
                    if self.mpi_rank == 0:
                        self.log(
                            f"  Jacobian extracted: {size[0]}x{size[1]}, nnz={int(nnz)}"
                        )

                # NOTE: Newton solver already returns a copy (J_final = A.copy()),
                # so we don't need to copy again here. However, to be
                # extra defensive, you could uncomment the following line:
                # J = J.copy()

                _local_result = J
            else:
                solver.solve(
                    self.u, return_jacobian=False, timestep=timestep, time=time
                )
                _local_result = None

            # Forward-diag logging on the SUCCESS path. Newton's last_*
            # attributes are populated by newton.py whether converged or not,
            # so reading them here is safe.
            if _fwd_diag_enabled():
                _fwd_diag_log_step(
                    MPI.COMM_WORLD, self.mpi_rank, self, solver,
                    timestep=timestep if timestep is not None else 0,
                    t_seconds=time if time is not None else self.problem.t,
                    h_local_indices=_diag_h_idx,
                    uv_local_indices=_diag_uv_idx,
                    newton_converged=getattr(solver, "last_converged", True),
                    n_iter=getattr(solver, "last_n_iters", -1),
                    correction_norm=getattr(solver, "last_correction_norm", float("nan")),
                    residual_norm=getattr(solver, "last_residual_norm", float("nan")),
                )

        except RuntimeError as e:
            # Typical cause: Newton did not converge + raise_on_failure=True.
            # Per-rank diagnostic printing (bare print, no collective ops).
            _runtime_fail = 1
            _local_err_msg = str(e)
            try:
                h_fun = self.u.sub(0).collapse()
                hvals = h_fun.x.array[:]
                min_h = hvals.min()
                print(f"Min h on process {self.mpi_rank}, {min_h}")
                bad_h = hvals < 0
                coords = h_fun.function_space.tabulate_dof_coordinates()[:, :2]
                coords = self.problem.reverse_projection(coords)
                print(f"first coords of negative h on {self.mpi_rank}",
                      coords[bad_h][:1])
            except Exception:
                pass

            # Forward-diag logging on the FAILURE path — captures the state
            # at the failed Newton iterate. This is the row that actually
            # diagnoses cycling-DA failures.
            if _fwd_diag_enabled():
                try:
                    _fwd_diag_log_step(
                        MPI.COMM_WORLD, self.mpi_rank, self, solver,
                        timestep=timestep if timestep is not None else 0,
                        t_seconds=time if time is not None else self.problem.t,
                        h_local_indices=_diag_h_idx,
                        uv_local_indices=_diag_uv_idx,
                        newton_converged=False,
                        n_iter=getattr(solver, "last_n_iters", -1),
                        correction_norm=getattr(solver, "last_correction_norm", float("nan")),
                        residual_norm=getattr(solver, "last_residual_norm", float("nan")),
                    )
                except Exception:
                    pass
        except ValueError as e:
            # Jacobian validation / other checks above.
            _value_fail = 1
            _local_err_msg = str(e)
            print(f"ERROR on rank {self.mpi_rank}: {e}")

        # Collective agreement. Newton's last ops are collective and complete
        # before Python reaches this point, so MPI state is clean here. Any
        # non-zero sum means at least one rank raised; we make ALL ranks raise
        # identically to avoid rank-divergent control flow.
        total_runtime_fail = MPI.COMM_WORLD.allreduce(_runtime_fail, op=MPI.SUM)
        total_value_fail = MPI.COMM_WORLD.allreduce(_value_fail, op=MPI.SUM)
        comm_size = MPI.COMM_WORLD.Get_size()

        if total_runtime_fail > 0:
            if self.mpi_rank == 0:
                self.log(
                    f"  [timestep failure] Newton solver failed at timestep "
                    f"{timestep}: global consensus "
                    f"{total_runtime_fail}/{comm_size} ranks reported RuntimeError"
                )
            # All ranks raise identically. Message is deterministic across ranks.
            raise RuntimeError(
                f"Newton solver failed at timestep {timestep} "
                f"({total_runtime_fail}/{comm_size} ranks)"
            )

        if total_value_fail > 0:
            if self.mpi_rank == 0:
                self.log(
                    f"  [timestep failure] ValueError at timestep {timestep}: "
                    f"global consensus {total_value_fail}/{comm_size} ranks"
                )
            raise ValueError(
                f"solve_timestep validation error at timestep {timestep} "
                f"({total_value_fail}/{comm_size} ranks)"
            )

        return _local_result

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
            # R2 handoff entry probe (one-shot).
            try:
                from swe4dvar.utils.solver_storage import (
                    _HANDOFF, _jac_handoff_log,
                )
                if not _HANDOFF["cg_entry_fired"]:
                    _jac_handoff_log("cg_implicit_entry", J)
                    _HANDOFF["cg_entry_fired"] = True
            except Exception:
                pass
            self.storage.save_jacobian(J)

    def save_parameter_derivatives(self):
        """Save residual derivatives with respect to active control parameters."""
        provider = getattr(self, "parameter_sensitivity_provider", None)
        if provider is None:
            return
        derivative_vectors = provider.compute_timestep_parameter_jacobian(
            problem=self.problem,
            solver=self,
        )
        if derivative_vectors:
            self.storage.save_parameter_derivatives(derivative_vectors)

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
        plot_every=1,
        plot_name="debug_tide",
        u_0=None,
        save_state=False,
        adjoint_method=False,
        save_bathy=False,
        save_true_bathy=False,
        make_wet=False,
        store_jacobians=False,
        store_parameter_derivatives=False,
        observation_times=None,
        monitor_progress=False,
        newton_diagnostics_config=None,
        enable_video=True,
    ):
        """Time-stepping loop with optional Jacobian storage for 4D-Var.

        Args:
            solver_parameters: Parameters passed to the nonlinear solver.
            stations: Monitoring station coordinates.
            plot_every: Plotting cadence (only used if enable_video=True).
            plot_name: Base name for visualization output.
            u_0: Optional initial condition override.
            save_state: Save every state (legacy mode).
            adjoint_method: Whether to assemble adjoint matrices.
            save_bathy: Save bathymetry samples when applying wetting/drying.
            save_true_bathy: Save true bathymetry and dry node indices.
            make_wet: Apply wetting/drying adjustments before saving.
            store_jacobians: Store Jacobians for 4D-Var adjoint computation.
            observation_times: Optional iterable of timestep indices to save.
            monitor_progress: Use tqdm progress bar instead of logging progress (default: False).
            newton_diagnostics_config: Optional dict with Newton diagnostics configuration.
                Keys: 'print_to_console', 'log_file', 'store_history', 'verbose'.
                Example: {'print_to_console': True, 'log_file': 'newton.log'}
            enable_video: Enable video/animation output (default: True).
        """

        if store_jacobians:
            self.storage.saved_jacobians.clear()
            if self.verbose:
                self.log("4D-Var mode: Jacobians will be stored during forward solve")
        if store_parameter_derivatives:
            self.storage.saved_parameter_derivatives.clear()
            if self.verbose:
                self.log("Augmented-control mode: timestep parameter derivatives will be stored")

        self.points_on_proc = local_points = self.init_stations(stations)
        self.station_data = np.zeros((self.problem.nt + 1, local_points.shape[0], 3))

        # set initial guess for the first time step
        if u_0 is None:
            self.u.x.array[:] = self.u_n.x.array[:]
        else:
            self.u_n.x.array[:] = u_0.x.array[:]
            self.u.x.array[:] = self.u_n.x.array[:]

        # Forward Newton problem lifecycle (Phase C-1 of memory-leak fix).
        # Default: REUSE the same CustomNewtonProblem across cost evals.
        # The Newton problem's A matrix, L vector, and KSP+LU-factor are
        # rebuilt at the C level on every Newton iteration anyway, so the
        # only reason to discard the Python wrapper between evals is
        # cleanliness — and it leaks: the destroy/create cycle pushes
        # ~240 MB/eval into PETSc's allocator pool that destroy() does not
        # reliably return. Reusing one persistent Newton problem closes
        # that channel entirely (in the same way sweep-KSP closed the
        # adjoint channel).
        # Disable with SWE4DVAR_FORWARD_NEWTON_REUSE=0 to fall back to
        # the legacy destroy+recreate path.
        import os as _os_nr
        _reuse_newton = (
            _os_nr.environ.get("SWE4DVAR_FORWARD_NEWTON_REUSE", "1").strip() == "1"
        )
        existing_solver = getattr(self, "solver", None)
        if _reuse_newton and existing_solver is not None:
            solver = existing_solver
            # Update diagnostics if a new config was passed for this run.
            if newton_diagnostics_config is not None:
                from swe4dvar.utils.newton_diagnostics import NewtonDiagnostics
                solver.diagnostics = NewtonDiagnostics(**newton_diagnostics_config)
            # Update solver_parameters reference so any per-call overrides
            # (rtol/atol/max_it/relaxation) take effect on the next solve.
            solver._solver_parameters = solver_parameters
            # Bump the forward-diag eval id (solve_init normally does this on
            # every fresh creation). Keeps per-step CSV correlated with eval id.
            _fwd_diag_bump_eval_id()
        else:
            if existing_solver is not None and hasattr(existing_solver, "destroy"):
                try:
                    existing_solver.destroy()
                except Exception:
                    pass
            self.solver = self.solve_init(
                solver_parameters=solver_parameters,
                newton_diagnostics_config=newton_diagnostics_config,
            )
            solver = self.solver
        # Ensure the persistent solver is recorded on self so subsequent
        # time_loop calls find it.
        self.solver = solver

        # Propagate raise_on_failure flag to Newton solver.
        # During optimization, the cost function sets this attribute so Newton failures
        # trigger exceptions that the line search can handle via backtracking.
        if getattr(self, '_raise_on_newton_failure', False):
            solver.raise_on_failure = True

        if enable_video:
            self.initialize_video(plot_name)
            self.plot_frame()
            if self.verbose:
                self.log(f"Video output enabled (plot every {plot_every} timesteps)")
        elif self.verbose:
            self.log("Video output disabled")

        effective_save_state = save_state or (observation_times is not None)
        data_manager = TimeStepDataManager(
            solver=self,
            save_state=effective_save_state,
            make_wet=make_wet,
            save_bathy=save_bathy,
            save_true_bathy=save_true_bathy,
            store_jacobians=store_jacobians,
            store_parameter_derivatives=store_parameter_derivatives,
            save_adjoints=adjoint_method,
            observation_times=observation_times,
            verbose=self.verbose,
        )

        # Store data manager config for print_config()
        if observation_times is not None:
            self._last_data_manager_config = {
                'mode': 'observation',
                'n_observations': len(observation_times)
            }
        elif effective_save_state or store_jacobians or adjoint_method:
            self._last_data_manager_config = {
                'mode': 'all_timesteps'
            }
        else:
            self._last_data_manager_config = {
                'mode': 'none'
            }

        # Record initial state at stations
        if len(local_points) > 0:
            recorded_data = self.record_stations(self.u_n, local_points)
            if recorded_data.shape[0] > 0:
                self.station_data[0, :, :] = recorded_data

        data_manager.save_timestep(timestep=0, local_points=local_points)

        # Initialize progress bar if requested
        pbar = None
        if (
            monitor_progress
            and TQDM_AVAILABLE
            and (not self.verbose or self.mpi_rank == 0)
        ):
            pbar = tqdm(total=self.problem.nt, desc="Time steps", unit="step")
        elif monitor_progress and not TQDM_AVAILABLE:
            if self.verbose and self.mpi_rank == 0:
                self.log("Warning: tqdm not available, falling back to verbose logging")

        # take first 2 steps with implicit Euler
        self.theta1.value = 0
        for a in range(min(2, self.problem.nt)):
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"method": "Euler"})
            elif self.verbose:
                self.log("Time Step Number", a, "Out of", self.problem.nt)
                self.log(a / self.problem.nt * 100, "% Complete")

            self.update_solution()

            should_get_jacobian = store_jacobians and data_manager.should_save_at(a + 1)
            J = self.solve_timestep(
                solver,
                store_jacobian=should_get_jacobian,
                timestep=a + 1,
                time=(a + 1) * self.problem.dt,
            )

            # Record station data after solving
            if len(local_points) > 0:
                recorded_data = self.record_stations(self.u, local_points)
                if recorded_data.shape[0] > 0:
                    self.station_data[a + 1, :, :] = recorded_data

            if enable_video and a % plot_every == 0:
                self.plot_frame()

            data_manager.save_timestep(
                timestep=a + 1,
                J=J,
                local_points=local_points,
            )

        # switch to high order time stepping (BDF2)
        self.theta1.value = self.theta
        for a in range(2, self.problem.nt):
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"method": "BDF2"})
            elif self.verbose:
                self.log("Time Step Number", a, "Out of", self.problem.nt)
                self.log(a / self.problem.nt * 100, "% Complete")

            self.update_solution()
            should_get_jacobian = store_jacobians and data_manager.should_save_at(a + 1)
            J = self.solve_timestep(
                solver,
                store_jacobian=should_get_jacobian,
                timestep=a + 1,
                time=(a + 1) * self.problem.dt,
            )

            # Record station data after solving
            if len(local_points) > 0:
                recorded_data = self.record_stations(self.u, local_points)
                if recorded_data.shape[0] > 0:
                    self.station_data[a + 1, :, :] = recorded_data

            data_manager.save_timestep(
                timestep=a + 1,
                J=J,
                local_points=local_points,
            )

            if enable_video and a % plot_every == 0:
                self.plot_frame()

        # Close progress bar if it was created
        if pbar is not None:
            pbar.close()

        if enable_video:
            self.finalize_video()

        # Gather station data from all MPI processes
        inds, vals = self.gather_station(0, self.points_on_proc, self.station_data)
        self.vals = vals
        self.inds = inds

        if self.verbose:
            data_manager.print_summary()

        # Optionally evaluate and print L2 error
        if self.problem.check_solution_def is not None:
            print("Checking solution at ", self.problem.t)
            e0 = self.problem.check_solution(self.u, self.V, self.problem.t)
            print("L2 error at t=", str(self.problem.t), " is ", str(e0))

        return (self.u, vals)

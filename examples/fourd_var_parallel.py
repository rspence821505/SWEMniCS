import numpy as np
from dolfinx import fem as fe
from petsc4py import PETSc
from mpi4py import MPI
from tqdm import tqdm
from typing import List, Dict, Tuple, Callable, Any, Literal
import time

from dca_utils import create_problem_solver

from cost_functions import (
    bayes_cost_function,
    dci_cost_function,
    dci_wme_cost_function,
    grad_cost_function,
)


class MPIPETScOptimizationResult:
    """
    Custom result class to mimic scipy.optimize.OptimizeResult for MPI-parallel PETSc TAO optimization.
    """

    def __init__(self):
        self.x = None  # Final solution
        self.fun = None  # Final objective function value
        self.success = False  # Whether optimization succeeded
        self.status = None  # Termination status
        self.message = ""  # Termination message
        self.nit = 0  # Number of iterations
        self.nfev = 0  # Number of function evaluations
        self.njev = 0  # Number of jacobian evaluations
        self.jac = None  # Final gradient


class MPIPETSc4DVarOptimizer:
    """
    MPI-parallel PETSc TAO-based optimizer for 4D-Var data assimilation problems.
    """

    def __init__(
        self,
        cost_function_type: str,
        solver: Callable,
        init_time: Callable,
        comm: MPI.Comm = None,
        **kwargs,
    ):
        """
        Initialize the MPI-parallel PETSc optimizer.

        Parameters
        ----------
        cost_function_type : str
            Type of cost function to use; one of {"bayes", "dci", "dci_wme"}.
        solver : Callable
            Solver object with saved state and adjoint memory.
        init_time : Callable
            Function or float representing the initial model time.
        comm : MPI.Comm, optional
            MPI communicator. If None, uses MPI.COMM_WORLD.
        **kwargs : dict
            Additional keyword arguments for cost and gradient functions.
        """
        self.cost_function_type = cost_function_type
        self.solver = solver
        self.init_time = init_time
        self.kwargs = kwargs

        # MPI setup
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        print(f"[Rank {self.rank}] Reached Inside MPIPETSc4DVarOptimizer", flush=True)

        # Mapping of cost function types
        self.cost_function_map = {
            "bayes": bayes_cost_function,
            "dci": dci_cost_function,
            "dci_wme": dci_wme_cost_function,
        }

        if cost_function_type not in self.cost_function_map:
            raise ValueError(f"Invalid cost_function_type: {cost_function_type}")

        # Initialize counters and storage (only on rank 0 for monitoring)
        self.iteration_count = 0
        self.function_evaluations = 0
        self.gradient_evaluations = 0
        self.cost_history = []

        # PETSc vectors and TAO solver will be initialized in optimize method
        self.tao = None
        self.x_petsc = None
        self.g_petsc = None

    def _objective_function(self, tao, x_petsc, user_context=None):
        """
        Objective function callback for MPI-parallel PETSc TAO.

        Parameters
        ----------
        tao : PETSc TAO solver object
        x_petsc : PETSc Vec
            Current iterate (distributed across MPI processes)
        user_context : optional
            User-defined context (unused)

        Returns
        -------
        float
            Objective function value
        """
        # Get local portion of the vector
        try:
            x_local = x_petsc.getArray(readonly=True)
            # Gather full vector on all processes for cost function evaluation
            x_global = np.zeros(x_petsc.getSize())
            self.comm.Allgather(x_local, x_global)
        except Exception as e:
            if self.rank == 0:
                print(f"Error accessing vector in objective function: {e}", flush=True)
            return np.inf

        # Evaluate cost function (each process can do this independently)
        try:
            cost = self.cost_function_map[self.cost_function_type](
                u0=x_global, solver=self.solver, init_time=self.init_time, **self.kwargs
            )
        except Exception as e:
            if self.rank == 0:
                print(f"Error in objective function evaluation: {e}", flush=True)
            return np.inf

        # Update counters on rank 0
        if self.rank == 0:
            self.function_evaluations += 1
            self.cost_history.append(cost)

        return float(cost)

    def _gradient_function(self, tao, x_petsc, g_petsc, user_context=None):
        """
        Gradient function callback for MPI-parallel PETSc TAO.

        Parameters
        ----------
        tao : PETSc TAO solver object
        x_petsc : PETSc Vec
            Current iterate (distributed)
        g_petsc : PETSc Vec
            Gradient vector to be filled (distributed)
        user_context : optional
            User-defined context (unused)
        """
        # Get local portion of the vector
        try:
            x_local = x_petsc.getArray(readonly=True)
            # Gather full vector on all processes
            x_global = np.zeros(x_petsc.getSize())
            self.comm.Allgather(x_local, x_global)
        except Exception as e:
            if self.rank == 0:
                print(f"Error accessing vector in gradient function: {e}")
            # Set zero gradient on error
            g_petsc.zeroEntries()
            return

        # Evaluate gradient
        try:
            grad_global = grad_cost_function(
                u0=x_global,
                solver=self.solver,
                adjoint_type=self.cost_function_type,
                **self.kwargs,
            )
        except Exception as e:
            if self.rank == 0:

                print(f"Error in gradient function evaluation: {e}")
            grad_global = np.zeros_like(x_global)

        # Ensure gradient is finite
        if not np.all(np.isfinite(grad_global)):
            if self.rank == 0:
                print("Warning: Non-finite gradient detected, replacing with zeros")
            grad_global = np.zeros_like(grad_global)

        # Distribute gradient to local portions
        try:
            # Get local indices for this process
            local_range = g_petsc.getOwnershipRange()
            grad_local = grad_global[local_range[0] : local_range[1]]
            g_petsc.setArray(grad_local)
            g_petsc.assemblyBegin()
            g_petsc.assemblyEnd()
        except Exception as e:
            if self.rank == 0:
                print(f"Error setting gradient vector: {e}")
            g_petsc.zeroEntries()

        # Update counters on rank 0
        if self.rank == 0:
            self.gradient_evaluations += 1

    def _objective_gradient_function(self, tao, x_petsc, g_petsc, user_context=None):
        """
        Combined objective and gradient function callback for MPI-parallel PETSc TAO.

        Parameters
        ----------
        tao : PETSc TAO solver object
        x_petsc : PETSc Vec
            Current iterate (distributed)
        g_petsc : PETSc Vec
            Gradient vector to be filled (distributed)
        user_context : optional
            User-defined context (unused)

        Returns
        -------
        float
            Objective function value
        """
        # Get local portion of the vector
        try:
            x_local = x_petsc.getArray(readonly=True)
            # Gather full vector on all processes
            x_global = np.zeros(x_petsc.getSize())
            self.comm.Allgather(x_local, x_global)
        except Exception as e:
            if self.rank == 0:
                print(
                    f"Error accessing vector in objective-gradient function: {e}",
                    flush=True,
                )
            g_petsc.zeroEntries()
            return np.inf

        # Evaluate cost function
        try:
            cost = self.cost_function_map[self.cost_function_type](
                u0=x_global, solver=self.solver, init_time=self.init_time, **self.kwargs
            )
        except Exception as e:
            if self.rank == 0:
                print(f"Error in objective function evaluation: {e}", flush=True)
            cost = np.inf
            grad_global = np.zeros_like(x_global)
        else:
            # Evaluate gradient only if objective succeeded
            try:
                grad_global = grad_cost_function(
                    u0=x_global,
                    solver=self.solver,
                    adjoint_type=self.cost_function_type,
                    **self.kwargs,
                )
            except Exception as e:
                if self.rank == 0:
                    print(f"Error in gradient function evaluation: {e}", flush=True)
                grad_global = np.zeros_like(x_global)

        # Ensure gradient is finite
        if not np.all(np.isfinite(grad_global)):
            if self.rank == 0:
                print("Warning: Non-finite gradient detected, replacing with zeros")
            grad_global = np.zeros_like(grad_global)

        # Distribute gradient to local portions
        try:
            local_range = g_petsc.getOwnershipRange()
            grad_local = grad_global[local_range[0] : local_range[1]]
            g_petsc.setArray(grad_local)
            g_petsc.assemblyBegin()
            g_petsc.assemblyEnd()
        except Exception as e:
            if self.rank == 0:
                print(f"Error setting gradient vector: {e}", flush=True)
            g_petsc.zeroEntries()

        # Update counters on rank 0
        if self.rank == 0:
            self.function_evaluations += 1
            self.gradient_evaluations += 1
            self.cost_history.append(cost)

        return float(cost)

    def _monitor_function(self, tao, user_context=None):
        """
        Monitor function callback for MPI-parallel PETSc TAO to track optimization progress.

        Parameters
        ----------
        tao : PETSc TAO solver object
        user_context : optional
            User-defined context (unused)
        """
        if self.rank == 0:  # Only print from rank 0
            self.iteration_count += 1
            try:
                obj_value = tao.getFunctionValue()
                print(
                    f"Iteration {self.iteration_count}: Cost = {obj_value:.6f}",
                    flush=True,
                )
            except:
                print(
                    f"Iteration {self.iteration_count}: Cost evaluation failed",
                    flush=True,
                )

    def optimize(
        self,
        u0: np.ndarray,
        method: str = "lmvm",
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> MPIPETScOptimizationResult:
        """
        Perform MPI-parallel optimization using PETSc TAO.

        Parameters
        ----------
        u0 : np.ndarray
            Initial guess for the control variable
        method : str, optional
            TAO method to use. Default is "lmvm".
        max_iterations : int, optional
            Maximum number of iterations. Default is 1000.
        tolerance : float, optional
            Convergence tolerance. Default is 1e-6.

        Returns
        -------
        MPIPETScOptimizationResult
            Optimization result object
        """
        n_global = len(u0)

        # Create distributed PETSc vectors
        self.x_petsc = PETSc.Vec().createMPI(n_global, comm=self.comm)
        self.g_petsc = PETSc.Vec().createMPI(n_global, comm=self.comm)

        # Set initial guess (distributed)
        local_range = self.x_petsc.getOwnershipRange()
        u0_local = u0[local_range[0] : local_range[1]]
        self.x_petsc.setArray(u0_local)
        self.x_petsc.assemblyBegin()
        self.x_petsc.assemblyEnd()

        # Create TAO solver
        self.tao = PETSc.TAO().create(comm=self.comm)
        self.tao.setType(method)

        # Set initial solution
        self.tao.setSolution(self.x_petsc)

        # Set objective and gradient functions
        self.tao.setObjectiveGradient(self._objective_gradient_function)

        # Set monitor function for progress tracking (only on rank 0)
        if self.rank == 0:
            self.tao.setMonitor(self._monitor_function)

        # Set convergence tolerances
        try:
            self.tao.setTolerances(gatol=tolerance, grtol=tolerance, gttol=tolerance)
        except TypeError:
            self.tao.setTolerances(tolerance, tolerance, tolerance)

        # Set maximum iterations
        try:
            self.tao.setMaximumIterations(max_iterations)
        except AttributeError:
            self.tao.setMaximumFunctionEvaluations(max_iterations)

        # Configure TAO from options
        self.tao.setFromOptions()

        # Reset counters
        self.iteration_count = 0
        self.function_evaluations = 0
        self.gradient_evaluations = 0
        self.cost_history = []

        # Synchronize before solving
        self.comm.Barrier()
        start_time = time.time() if self.rank == 0 else None

        # Solve the optimization problem
        try:
            self.tao.solve()
            success = True
            message = "Optimization completed successfully"
        except Exception as e:
            success = False
            message = f"Optimization failed: {str(e)}"
            if self.rank == 0:
                print(f"TAO solve failed: {e}", flush=True)

        # Synchronize after solving
        self.comm.Barrier()
        if self.rank == 0:
            end_time = time.time()
            print(f"Optimization completed in {end_time - start_time:.2f} seconds")

        # Get convergence information
        try:
            converged_reason = self.tao.getConvergedReason()
        except AttributeError:
            converged_reason = 0

        try:
            final_objective = self.tao.getFunctionValue()
        except AttributeError:
            final_objective = self.cost_history[-1] if self.cost_history else np.inf

        try:
            gnorm = self.tao.getGradientNorm()
        except AttributeError:
            gnorm = 0.0

        # Gather solution from all processes
        solution_local = self.x_petsc.getArray()
        solution = np.zeros(n_global)

        # Gather all local portions to get the full solution
        local_range = self.x_petsc.getOwnershipRange()
        self.comm.Allgatherv(
            solution_local,
            [solution, np.diff(self.comm.allgather(local_range)).flatten()],
        )

        # Get final gradient
        try:
            grad_local = self.g_petsc.getArray()
            final_gradient = np.zeros(n_global)
            self.comm.Allgatherv(
                grad_local,
                [final_gradient, np.diff(self.comm.allgather(local_range)).flatten()],
            )
        except Exception:
            if self.rank == 0:
                try:
                    final_gradient = grad_cost_function(
                        u0=solution,
                        solver=self.solver,
                        adjoint_type=self.cost_function_type,
                        **self.kwargs,
                    )
                except:
                    final_gradient = np.zeros_like(solution)
            else:
                final_gradient = np.zeros(n_global)
            # Broadcast final gradient to all processes
            final_gradient = self.comm.bcast(final_gradient, root=0)

        # Create result object
        result = MPIPETScOptimizationResult()
        result.x = solution
        result.fun = final_objective
        result.success = success and (converged_reason > 0)
        result.status = converged_reason
        result.message = message
        result.nit = self.iteration_count
        result.nfev = self.function_evaluations
        result.njev = self.gradient_evaluations
        result.jac = final_gradient

        # Clean up PETSc objects
        self.tao.destroy()
        self.x_petsc.destroy()
        self.g_petsc.destroy()

        return result


def print_optimization_summary(
    result: MPIPETScOptimizationResult, rank: int = 0
) -> None:
    """
    Print a formatted summary of an MPI-parallel PETSc optimization result.

    Parameters
    ----------
    result : MPIPETScOptimizationResult
        The result object returned by MPI-parallel PETSc TAO optimization.
    rank : int, optional
        MPI rank. Only rank 0 prints the summary. Default is 0.
    """
    if rank == 0:
        print("\nOptimization completed:")
        print(f"  Success: {result.success}")
        print(f"  Status: {result.status}")
        print(f"  Message: {result.message}")
        print(f"  Final cost: {result.fun:.6e}")
        print(f"  Iterations: {result.nit}")
        print(f"  Function evaluations: {result.nfev}")
        print(f"  Gradient evaluations: {result.njev}")
        print(f"  Gradient norm at solution: {np.linalg.norm(result.jac):.6e}")
        print("\n" + "-" * 60 + "\n")


def print_state_summary(
    u0: np.ndarray, result: MPIPETScOptimizationResult, step: int = 40, rank: int = 0
) -> None:
    """
    Print a summary of the initial and optimized state vectors.

    Parameters
    ----------
    u0 : np.ndarray
        Initial guess for the state vector.
    result : MPIPETScOptimizationResult
        Result object returned by MPI-parallel PETSc TAO optimization.
    step : int, optional
        Step size for subsampling the state vector when printing. Default is 40.
    rank : int, optional
        MPI rank. Only rank 0 prints the summary. Default is 0.
    """
    if rank == 0:
        print("State comparison (subsampled):")
        print(f"  Initial state (every {step}th entry):   {u0[::step]}\n")
        print(f"  Optimized state (every {step}th entry): {result.x[::step]}\n")


def optimize_4dvar(
    u0: np.ndarray,
    cost_function_type: str,
    solver: Callable,
    init_time: Callable,
    method: str = "lmvm",
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    comm: MPI.Comm = None,
    **kwargs,
) -> Tuple[np.ndarray, MPIPETScOptimizationResult]:
    """
    Perform MPI-parallel 4D-Var optimization using PETSc TAO.

    Parameters
    ----------
    u0 : np.ndarray
        Initial guess for the control variable (e.g., initial state).
    cost_function_type : str
        Type of cost function to use; one of {"bayes", "dci", "dci_wme"}.
    solver : Callable
        Solver object with saved state and adjoint memory.
    init_time : Callable
        Function or float representing the initial model time.
    method : str, optional
        TAO optimization method. Default is "lmvm".
    max_iterations : int, optional
        Maximum number of optimization iterations. Default is 1000.
    tolerance : float, optional
        Convergence tolerance for optimization. Default is 1e-6.
    comm : MPI.Comm, optional
        MPI communicator. If None, uses MPI.COMM_WORLD.
    **kwargs : dict
        Additional keyword arguments for cost and gradient functions.

    Returns
    -------
    Tuple[np.ndarray, MPIPETScOptimizationResult]
        Tuple containing optimal control variable and optimization result.
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    print(f"[Rank {rank}] Reached Inside optimize_4dvar()", flush=True)

    # Create MPI-parallel PETSc optimizer
    optimizer = MPIPETSc4DVarOptimizer(
        cost_function_type=cost_function_type,
        solver=solver,
        init_time=init_time,
        comm=comm,
        **kwargs,
    )

    # Perform optimization
    result = optimizer.optimize(
        u0=u0, method=method, max_iterations=max_iterations, tolerance=tolerance
    )

    # Print results summary (only on rank 0)
    print_optimization_summary(result, rank)
    print_state_summary(u0, result, step=100, rank=rank)

    return result.x, result


def run_assimilation(
    problem_params: Dict[str, Any],
    solver_params: Dict[str, Any],
    stations: np.ndarray,
    y_obs: np.ndarray,
    obs_per_window: int,
    obs_spatial_indices: np.ndarray,
    obs_time_indices: np.ndarray,
    H: np.ndarray,
    covs: Dict[str, np.ndarray],
    hb: np.ndarray,
    problem_type: str,
    cost_function_type: Literal["bayes", "dci", "dci_wme"],
    optimization_method: str = "lmvm",
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    comm: MPI.Comm = None,
) -> np.ndarray:
    """
    Run MPI-parallel 4D-Var assimilation process over multiple assimilation windows.

    Parameters
    ----------
    problem_params : dict
        Dictionary containing model configuration.
    solver_params : dict
        Parameters for the time loop solver.
    stations : np.ndarray
        Spatial locations or indices of observation stations.
    y_obs : np.ndarray
        All observations over all assimilation windows.
    obs_per_window : int
        Number of observations per assimilation window.
    obs_spatial_indices : np.ndarray
        Indices of the observed variables in the state vector.
    obs_time_indices : np.ndarray
        Time indices corresponding to each observation within a window.
    H : np.ndarray
        Observation operator matrix.
    covs : dict
        Dictionary with inverse covariance matrices.
    hb : np.ndarray
        Background height/elevation field.
    problem_type : str
        Identifier for problem setup.
    cost_function_type : {'bayes', 'dci', 'dci_wme'}
        Type of 4D-Var cost function.
    optimization_method : str, optional
        PETSc TAO optimization method. Default is "lmvm".
    max_iterations : int, optional
        Maximum optimization iterations per window. Default is 1000.
    tolerance : float, optional
        Convergence tolerance. Default is 1e-6.
    comm : MPI.Comm, optional
        MPI communicator. If None, uses MPI.COMM_WORLD.

    Returns
    -------
    np.ndarray
        Full spatio-temporal analysis result over all windows.
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"[Rank {rank}] Reached checkpoint inside run_assimilation()", flush=True)
    name = "Hotstart"
    analysis = []
    analysis_state = None
    num_windows = problem_params["num_windows"]
    steps_per_window = problem_params["num_steps"]
    obs_times_current_window = obs_time_indices[:obs_per_window]

    # Use tqdm only on rank 0
    window_iterator = (
        tqdm(range(num_windows), desc="Processing windows", unit="window")
        if rank == 0
        else range(num_windows)
    )

    for idx in window_iterator:
        if rank == 0:
            print(f"\n=== Processing Window {idx + 1}/{num_windows} ===")

        # Extract observations for current window
        indices = np.arange(obs_per_window) + (idx * obs_per_window)
        yobs_current_window = y_obs[indices]

        # Update initial time for model
        initial_time = int(idx * steps_per_window * problem_params["dt"])
        problem_params.update({"t": initial_time})

        # Create problem and solver (each process creates its own)
        _, solver = create_problem_solver(
            problem_params, problem_type, true_signal=False
        )

        solver.problem.t = initial_time

        # Initialize state
        u_0 = fe.Function(solver.V)
        u_0.x.array[:] = (
            solver.u_n.x.array[:] if analysis_state is None else analysis_state
        )

        # Generate background state by running the model forward
        if rank == 0:
            print(f"Solver Time 1: {solver.problem.t}")

        initial_u0 = u_0.copy()
        solver.time_loop(
            solver_parameters=solver_params,
            stations=stations,
            plot_every=60,
            plot_name=name,
            u_0=initial_u0,
            save_states=True,
            adjoint_method=False,
        )

        background = np.array(solver.saved_states)
        observed_background_states = background[obs_times_current_window]
        Q_zb = H @ observed_background_states.T
        solver.saved_states = []  # clear for next window

        # Initial guess and background
        z0 = initial_u0.x.array[:]
        z_b = initial_u0.x.array[:]

        # Synchronize before optimization
        comm.Barrier()

        # MPI-parallel assimilation step using PETSc TAO
        optimized_state, _ = optimize_4dvar(
            u0=z0,
            cost_function_type=cost_function_type,
            solver=solver,
            init_time=initial_time,
            method=optimization_method,
            max_iterations=max_iterations,
            tolerance=tolerance,
            comm=comm,
            u_b=z_b,
            y_obs=yobs_current_window,
            obs_spatial_idxs=obs_spatial_indices,
            obs_time_idxs=obs_times_current_window,
            H=H,
            covs=covs,
            Q_zb=Q_zb,
            stations=stations,
            hb=hb,
            solver_params=solver_params,
        )

        # Update state with analysis
        u_0.x.array[:] = optimized_state

        # Run forward using updated state
        solver.problem.t = initial_time
        if rank == 0:
            print(f"Solver Time 2: {solver.problem.t}")

        solver.time_loop(
            solver_parameters=solver_params,
            stations=stations,
            plot_every=60,
            plot_name=name,
            u_0=u_0,
            adjoint_method=False,
        )

        # Store last state for next window initialization
        analysis_state = solver.u.x.array[:]

        # Collect results
        current_analysis = solver.vals.copy()
        if idx < num_windows - 1:
            current_analysis = current_analysis[:-1, :, :]
        analysis.append(current_analysis)

        if rank == 0:
            print(f"Window {idx + 1} completed successfully")

        # Synchronize after each window
        comm.Barrier()

    return np.concatenate(analysis, axis=0)

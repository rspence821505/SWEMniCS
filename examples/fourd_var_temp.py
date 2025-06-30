import numpy as np
from dolfinx import fem as fe
from petsc4py import PETSc
from tqdm import tqdm
from typing import List, Dict, Tuple, Callable, Any, Literal


from dca_utils import create_problem_solver

from cost_functions import (
    bayes_cost_function,
    dci_cost_function,
    dci_wme_cost_function,
    grad_cost_function,
)


class PETScOptimizationResult:
    """
    Custom result class to mimic scipy.optimize.OptimizeResult for PETSc TAO optimization.
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


class PETSc4DVarOptimizer:
    """
    PETSc TAO-based optimizer for 4D-Var data assimilation problems.
    """

    def __init__(
        self, cost_function_type: str, solver: Callable, init_time: Callable, **kwargs
    ):
        """
        Initialize the PETSc optimizer.

        Parameters
        ----------
        cost_function_type : str
            Type of cost function to use; one of {"bayes", "dci", "dci_wme"}.
        solver : Callable
            Solver object with saved state and adjoint memory.
        init_time : Callable
            Function or float representing the initial model time.
        **kwargs : dict
            Additional keyword arguments for cost and gradient functions.
        """
        self.cost_function_type = cost_function_type
        self.solver = solver
        self.init_time = init_time
        self.kwargs = kwargs

        # Mapping of cost function types
        self.cost_function_map = {
            "bayes": bayes_cost_function,
            "dci": dci_cost_function,
            "dci_wme": dci_wme_cost_function,
        }

        if cost_function_type not in self.cost_function_map:
            raise ValueError(f"Invalid cost_function_type: {cost_function_type}")

        # Initialize counters and storage
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
        Objective function callback for PETSc TAO.

        Parameters
        ----------
        tao : PETSc TAO solver object
        x_petsc : PETSc Vec
            Current iterate
        user_context : optional
            User-defined context (unused)

        Returns
        -------
        float
            Objective function value
        """
        # Convert PETSc Vec to numpy array using getValue to avoid locking
        try:
            # Try to get values element by element to avoid locking issues
            n = x_petsc.getSize()
            x_array = np.zeros(n)
            for i in range(n):
                x_array[i] = x_petsc.getValue(i)
        except Exception:
            # Fallback: try getArray with read-only access
            try:
                x_array = x_petsc.getArray(readonly=True).copy()
            except Exception:
                print("Error: Could not access vector in objective function")
                return np.inf

        # Evaluate cost function
        try:
            cost = self.cost_function_map[self.cost_function_type](
                u0=x_array, solver=self.solver, init_time=self.init_time, **self.kwargs
            )
        except Exception as e:
            print(f"Error in objective function evaluation: {e}")
            return np.inf

        self.function_evaluations += 1
        self.cost_history.append(cost)

        return float(cost)  # Ensure we return a Python float

    def _gradient_function(self, tao, x_petsc, g_petsc, user_context=None):
        """
        Gradient function callback for PETSc TAO.

        Parameters
        ----------
        tao : PETSc TAO solver object
        x_petsc : PETSc Vec
            Current iterate
        g_petsc : PETSc Vec
            Gradient vector to be filled
        user_context : optional
            User-defined context (unused)
        """
        # Convert PETSc Vec to numpy array using getValue to avoid locking
        try:
            # Try to get values element by element to avoid locking issues
            n = x_petsc.getSize()
            x_array = np.zeros(n)
            for i in range(n):
                x_array[i] = x_petsc.getValue(i)
        except Exception:
            # Fallback: try getArray with read-only access
            try:
                x_array = x_petsc.getArray(readonly=True).copy()
            except Exception:
                print("Error: Could not access vector in gradient function")
                x_array = np.zeros(x_petsc.getSize())

        # Evaluate gradient
        try:
            grad = grad_cost_function(
                u0=x_array,
                solver=self.solver,
                adjoint_type=self.cost_function_type,
                **self.kwargs,
            )
        except Exception as e:
            print(f"Error in gradient function evaluation: {e}")
            # Return zero gradient on error
            grad = np.zeros_like(x_array)

        # Ensure gradient is finite
        if not np.all(np.isfinite(grad)):
            print("Warning: Non-finite gradient detected, replacing with zeros")
            grad = np.zeros_like(grad)

        # Set gradient in PETSc Vec using setValue to avoid locking
        try:
            for i in range(len(grad)):
                g_petsc.setValue(i, grad[i])
            g_petsc.assemblyBegin()
            g_petsc.assemblyEnd()
        except Exception:
            # Fallback: try setArray
            try:
                g_petsc.setArray(grad)
            except Exception as e:
                print(f"Error setting gradient vector: {e}")

        self.gradient_evaluations += 1

    def _objective_gradient_function(self, tao, x_petsc, g_petsc, user_context=None):
        """
        Combined objective and gradient function callback for PETSc TAO.

        Parameters
        ----------
        tao : PETSc TAO solver object
        x_petsc : PETSc Vec
            Current iterate
        g_petsc : PETSc Vec
            Gradient vector to be filled
        user_context : optional
            User-defined context (unused)

        Returns
        -------
        float
            Objective function value
        """
        # Convert PETSc Vec to numpy array using getValue to avoid locking
        try:
            # Try to get values element by element to avoid locking issues
            n = x_petsc.getSize()
            x_array = np.zeros(n)
            for i in range(n):
                x_array[i] = x_petsc.getValue(i)
        except Exception:
            # Fallback: try getArray with read-only access
            try:
                x_array = x_petsc.getArray(readonly=True).copy()
            except Exception:
                print("Error: Could not access vector in objective-gradient function")
                return np.inf

        # Evaluate cost function
        try:
            cost = self.cost_function_map[self.cost_function_type](
                u0=x_array, solver=self.solver, init_time=self.init_time, **self.kwargs
            )
        except Exception as e:
            print(f"Error in objective function evaluation: {e}")
            cost = np.inf
            grad = np.zeros_like(x_array)
        else:
            # Evaluate gradient only if objective succeeded
            try:
                grad = grad_cost_function(
                    u0=x_array,
                    solver=self.solver,
                    adjoint_type=self.cost_function_type,
                    **self.kwargs,
                )
            except Exception as e:
                print(f"Error in gradient function evaluation: {e}")
                grad = np.zeros_like(x_array)

        # Ensure gradient is finite
        if not np.all(np.isfinite(grad)):
            print("Warning: Non-finite gradient detected, replacing with zeros")
            grad = np.zeros_like(grad)

        # Set gradient in PETSc Vec using setValue to avoid locking
        try:
            for i in range(len(grad)):
                g_petsc.setValue(i, grad[i])
            g_petsc.assemblyBegin()
            g_petsc.assemblyEnd()
        except Exception:
            # Fallback: try setArray
            try:
                g_petsc.setArray(grad)
            except Exception as e:
                print(f"Error setting gradient vector: {e}")

        self.function_evaluations += 1
        self.gradient_evaluations += 1
        self.cost_history.append(cost)

        return float(cost)  # Ensure we return a Python float

    def _monitor_function(self, tao, user_context=None):
        """
        Monitor function callback for PETSc TAO to track optimization progress.

        Parameters
        ----------
        tao : PETSc TAO solver object
        user_context : optional
            User-defined context (unused)
        """
        self.iteration_count += 1

        # Get current objective value
        obj_value = tao.getSolutionStatus()[0]

        print(f"Iteration {self.iteration_count}: Cost = {obj_value:.6f}")

    def optimize(
        self,
        u0: np.ndarray,
        method: str = "lmvm",
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> PETScOptimizationResult:
        """
        Perform optimization using PETSc TAO.

        Parameters
        ----------
        u0 : np.ndarray
            Initial guess for the control variable
        method : str, optional
            TAO method to use. Options include:
            - "lmvm": Limited-memory variable metric method (similar to L-BFGS)
            - "nls": Newton line search
            - "ntr": Newton trust region
            - "cg": Conjugate gradient
            Default is "lmvm".
        max_iterations : int, optional
            Maximum number of iterations. Default is 1000.
        tolerance : float, optional
            Convergence tolerance. Default is 1e-6.

        Returns
        -------
        PETScOptimizationResult
            Optimization result object
        """
        # Initialize PETSc vectors
        self.x_petsc = PETSc.Vec().createSeq(len(u0))
        self.g_petsc = PETSc.Vec().createSeq(len(u0))

        # Set initial guess
        self.x_petsc.setArray(u0)

        # Create TAO solver
        self.tao = PETSc.TAO().create()
        self.tao.setType(method)

        # Set initial solution
        self.tao.setSolution(self.x_petsc)

        # Set objective and gradient functions
        self.tao.setObjectiveGradient(self._objective_gradient_function)

        # Set monitor function for progress tracking
        self.tao.setMonitor(self._monitor_function)

        # Set convergence tolerances
        # Note: Different versions of PETSc may have slightly different method signatures
        try:
            self.tao.setTolerances(gatol=tolerance, grtol=tolerance, gttol=tolerance)
        except TypeError:
            # Fallback for older PETSc versions
            self.tao.setTolerances(tolerance, tolerance, tolerance)

        # Set maximum iterations
        try:
            self.tao.setMaximumIterations(max_iterations)
        except AttributeError:
            # Fallback method name for older versions
            self.tao.setMaximumFunctionEvaluations(max_iterations)

        # Configure TAO from options (allows runtime configuration)
        self.tao.setFromOptions()

        # Reset counters
        self.iteration_count = 0
        self.function_evaluations = 0
        self.gradient_evaluations = 0
        self.cost_history = []

        # Solve the optimization problem
        try:
            self.tao.solve()
            success = True
            message = "Optimization completed successfully"
        except Exception as e:
            success = False
            message = f"Optimization failed: {str(e)}"
            print(f"TAO solve failed: {e}")

        # Get convergence information before accessing arrays
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

        # Get solution by copying values element by element to avoid locking issues
        solution = np.zeros(len(u0))
        try:
            # Try to get the solution vector from TAO
            solution_vec = self.tao.getSolution()
            # Use a safer method to copy values
            for i in range(len(u0)):
                solution[i] = solution_vec.getValue(i)
        except Exception:
            # Fallback: use our original vector if possible
            try:
                # First try to read from our vector directly
                solution = self.x_petsc.getValues(range(len(u0)))
            except Exception:
                # Last resort: use the last values we know
                print(
                    "Warning: Could not retrieve solution vector, using initial guess"
                )
                solution = u0.copy()

        # Get final gradient
        try:
            # Create a new vector for gradient computation
            grad_vec = PETSc.Vec().createSeq(len(u0))
            self.tao.computeGradient(solution_vec, grad_vec)
            # Copy gradient values element by element
            final_gradient = np.zeros(len(u0))
            for i in range(len(u0)):
                final_gradient[i] = grad_vec.getValue(i)
            grad_vec.destroy()
        except (AttributeError, Exception):
            # If computeGradient is not available, evaluate manually
            try:
                final_gradient = grad_cost_function(
                    u0=solution,
                    solver=self.solver,
                    adjoint_type=self.cost_function_type,
                    **self.kwargs,
                )
            except:
                final_gradient = np.zeros_like(solution)

        # Create result object
        result = PETScOptimizationResult()
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


def print_optimization_summary(result: PETScOptimizationResult) -> None:
    """
    Print a formatted summary of a PETSc optimization result.

    Parameters
    ----------
    result : PETScOptimizationResult
        The result object returned by PETSc TAO optimization.
    """
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
    u0: np.ndarray, result: PETScOptimizationResult, step: int = 40
) -> None:
    """
    Print a summary of the initial and optimized state vectors.

    Parameters
    ----------
    u0 : np.ndarray
        Initial guess for the state vector.
    result : PETScOptimizationResult
        Result object returned by PETSc TAO optimization.
    step : int, optional
        Step size for subsampling the state vector when printing. Default is 40.
    """
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
    **kwargs,
) -> Tuple[np.ndarray, PETScOptimizationResult]:
    """
    Perform 4D-Var optimization using PETSc TAO with a specified cost function and gradient.

    Parameters
    ----------
    u0 : np.ndarray
        Initial guess for the control variable (e.g., initial state).
    cost_function_type : str
        Type of cost function to use; one of {"bayes", "dci", "dci_wme"}.
    solver : Callable
        Solver object with saved state and adjoint memory that is mutated during optimization.
    init_time : Callable
        Function or float representing the initial model time used to reset the solver.
    method : str, optional
        TAO optimization method. Default is "lmvm" (limited-memory variable metric).
    max_iterations : int, optional
        Maximum number of optimization iterations. Default is 1000.
    tolerance : float, optional
        Convergence tolerance for optimization. Default is 1e-6.
    **kwargs : dict
        Additional keyword arguments required by the cost and gradient functions, including:
            - u_b : np.ndarray
            - y_obs : np.ndarray
            - obs_time_idxs : np.ndarray
            - obs_spatial_idxs : np.ndarray
            - H : np.ndarray
            - covs : Dict[str, np.ndarray] with keys "B_inv", "R_inv", and "L_inv"
            - Q_zb : np.ndarray
            - solver_params : dict
            - stations : np.ndarray
            - hb : np.ndarray

    Returns
    -------
    Tuple[np.ndarray, PETScOptimizationResult]
        Tuple containing:
        - Optimal control variable (e.g., initial condition)
        - Full optimization result object from PETSc TAO

    Raises
    ------
    ValueError
        If `cost_function_type` is not one of the recognized values.
    """
    # Create PETSc optimizer
    optimizer = PETSc4DVarOptimizer(
        cost_function_type=cost_function_type,
        solver=solver,
        init_time=init_time,
        **kwargs,
    )

    # Perform optimization
    result = optimizer.optimize(
        u0=u0, method=method, max_iterations=max_iterations, tolerance=tolerance
    )

    # Print results summary
    print_optimization_summary(result)
    print_state_summary(u0, result, step=100)

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
) -> np.ndarray:
    """
    Run a full 4D-Var assimilation process over multiple assimilation windows using PETSc TAO.

    Parameters
    ----------
    problem_params : dict
        Dictionary containing model configuration including time-step size, number of steps,
        number of windows, mesh info, etc.
    solver_params : dict
        Parameters used to control the time loop solver (e.g., Newton tolerances, dt, etc.).
    stations : np.ndarray
        Spatial locations or indices of observation stations.
    y_obs : np.ndarray
        All observations over all assimilation windows. Shape: (n_total_obs, obs_dim).
    obs_per_window : int
        Number of observations per assimilation window.
    obs_spatial_indices : np.ndarray
        Indices of the observed variables in the state vector.
    obs_time_indices : np.ndarray
        Time indices corresponding to each observation within a window.
    H : np.ndarray
        Observation operator matrix mapping full state to observation space.
    covs : dict
        Dictionary with keys "B_inv", "R_inv", "L_inv" for inverse covariances.
    hb : np.ndarray
        Background height/elevation field used in initial/boundary conditions.
    problem_type : str
        Identifier used to select and instantiate a specific problem setup.
    cost_function_type : {'bayes', 'dci', 'dci_wme'}
        Type of 4D-Var cost function to use in optimization.
    optimization_method : str, optional
        PETSc TAO optimization method. Default is "lmvm".
    max_iterations : int, optional
        Maximum number of optimization iterations per window. Default is 1000.
    tolerance : float, optional
        Convergence tolerance for optimization. Default is 1e-6.

    Returns
    -------
    np.ndarray
        Full spatio-temporal analysis result over all assimilation windows.
        Shape: (n_total_steps, n_stations, n_components)
    """
    name = "Hotstart"
    analysis = []
    analysis_state = None
    num_windows = problem_params["num_windows"]
    steps_per_window = problem_params["num_steps"]
    obs_times_current_window = obs_time_indices[:obs_per_window]

    for idx in tqdm(range(num_windows), desc="Processing windows", unit="window"):

        # Extract observations for current window
        indices = np.arange(obs_per_window) + (idx * obs_per_window)
        yobs_current_window = y_obs[indices]

        # Update initial time for model
        initial_time = int(idx * steps_per_window * problem_params["dt"])
        problem_params.update({"t": initial_time})

        # Create problem and solver
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

        # Assimilation step using PETSc TAO
        optimized_state, _ = optimize_4dvar(
            u0=z0,
            cost_function_type=cost_function_type,
            solver=solver,
            init_time=initial_time,
            method=optimization_method,
            max_iterations=max_iterations,
            tolerance=tolerance,
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

        print(
            f"/////////////////////////////////////// Window {idx + 1} Completed ////////////////////////////////////////////////// \n\n"
        )

    return np.concatenate(analysis, axis=0)

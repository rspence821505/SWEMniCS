import numpy as np
from dolfinx import fem as fe
from petsc4py import PETSc
from scipy.optimize import minimize, OptimizeResult
from tqdm import tqdm
from typing import List, Dict, Tuple, Callable
import sys

from dca_utils import create_problem_solver

from cost_functions import (
    bayes_cost_function,
    dci_cost_function,
    dci_wme_cost_function,
    grad_cost_function,
)


def print_optimization_summary(result: OptimizeResult) -> None:
    """
    Print a formatted summary of an optimization result.

    Parameters
    ----------
    result : OptimizeResult
        The result object returned by `scipy.optimize.minimize`.
    """
    print("\nOptimization completed:")
    print(f"  Success: {result.success}")
    print(f"  Status: {result.status}")
    print(f"  Message: {result.message}")
    print(f"  Final cost: {result.fun:.6e}")
    print(f"  Iterations: {result.nit}")
    print(f"  Function evaluations: {result.nfev}")
    print(f"  Gradient norm at solution: {np.linalg.norm(result.jac):.6e}")
    print("\n" + "-" * 60 + "\n")


def print_state_summary(u0: np.ndarray, result: OptimizeResult, step: int = 40) -> None:
    """
    Print a summary of the initial and optimized state vectors.

    Parameters
    ----------
    u0 : np.ndarray
        Initial guess for the state vector.
    result : OptimizeResult
        Result object returned by `scipy.optimize.minimize`.
    step : int, optional
        Step size for subsampling the state vector when printing. Default is 20.
    """
    print("State comparison (subsampled):")
    print(f"  Initial state (every {step}th entry):   {u0[::step]}\n")
    print(f"  Optimized state (every {step}th entry): {result.x[::step]}\n")


def optimize_4dvar(
    u0: np.ndarray,
    cost_function_type: str,
    solver: Callable,
    init_time: Callable,
    **kwargs,
) -> Tuple[np.ndarray, dict]:
    """
    Perform 4D-Var optimization using a specified cost function and its gradient.
    """

    # Mapping of cost function types to their implementations
    cost_function_map = {
        "bayes": bayes_cost_function,
        "dci": dci_cost_function,
        "dci_wme": dci_wme_cost_function,
    }

    cost_function_values = []

    # Cost function wrapper
    def cost_fn(u0):
        return cost_function_map[cost_function_type](
            u0=u0, solver=solver, init_time=init_time, **kwargs
        )

    # Gradient function wrapper
    def grad_fn(u0):
        return grad_cost_function(
            u0=u0, solver=solver, adjoint_type=cost_function_type, **kwargs
        )

    def callback(x):
        cost = cost_fn(x)
        cost_function_values.append(cost)
        print(f"Iteration {len(cost_function_values)}: Cost = {cost:.6f}")

    # options = {"gtol": 1e-6, "ftol": 1e-12, "maxfun": 10, "maxiter": 1000, "disp": True}

    result = minimize(
        fun=cost_fn,
        x0=u0,
        method="L-BFGS-B",
        jac=grad_fn,
        callback=callback,
        # options=options,
    )

    # Print optimization results
    print_optimization_summary(result)

    # Print state comparison
    # print_state_summary(u0, result, step=100)

    return result.x, result


def run_assimilation(
    problem_params,
    solver_params,
    stations,
    y_obs,
    obs_per_window,
    obs_spatial_indices,
    obs_time_indices,
    H,
    covs,
    hb,
    problem_type,
    cost_function_type,
):
    """
    Run 4DVar analysis with over assimilation windows
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
            problem_params, problem_type, true_signal=False, verbose=False
        )

        solver.problem.t = initial_time  # reset time to initial time
        V = solver.V  # get function spaces

        # Initialize state
        u_0 = fe.Function(V)
        u_0.x.array[:] = (
            solver.u_n.x.array[:] if analysis_state is None else analysis_state
        )

        # Generate background z_b
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

        # Process background state
        background = np.array(solver.saved_states)  # shape: (steps, num_stations)
        observed_background_states = background[
            obs_times_current_window
        ]  # shape: (n_obs, state_dim)
        Q_zb = H @ observed_background_states.T  # shape: (n_obs,obs_dim)
        solver.saved_states = []  # reset saved states for next window

        # Get initial state vectors
        z0 = initial_u0.x.array[:]
        z_b = initial_u0.x.array[:]

        # Assimilation Step
        optimized_state, _ = optimize_4dvar(
            u0=z0,
            cost_function_type=cost_function_type,
            solver=solver,
            init_time=initial_time,
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

        # Update state with optimized values
        u_0.x.array[:] = optimized_state

        # Run analysis forward
        solver.problem.t = initial_time
        solver.saved_states = []  # reset saved states for analysis
        solver.saved_adjoints = []  # reset saved adjoints for analysis
        print(f"Solver Time 2: {solver.problem.t}")  # Debugging line
        solver.time_loop(
            solver_parameters=solver_params,
            stations=stations,
            plot_every=60,
            plot_name=name,
            u_0=u_0,
            save_states=True,
            adjoint_method=False,
        )

        # Save analysis state for next window
        analysis_state = solver.u.x.array[:]

        # Collect results
        saved_states = np.array(solver.saved_states)
        current_analysis = saved_states.copy()
        if idx < num_windows - 1:
            current_analysis = current_analysis[:-1, :]
        analysis.append(current_analysis)

    # Combine all windows
    return np.concatenate(analysis, axis=0)

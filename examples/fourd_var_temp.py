import numpy as np
from dolfinx import fem, mesh, io
from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form
import ufl
from ufl import grad, div, dx, ds, inner, dot, TestFunction, TrialFunction
from petsc4py import PETSc
from scipy.optimize import minimize, OptimizeResult
from tqdm import tqdm
from typing import List, Dict, Tuple, Callable
import sys
from mpi4py import MPI

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
    comm: MPI.Comm = None,
    **kwargs,
) -> Tuple[np.ndarray, dict]:
    """
    Perform 4D-Var optimization using a specified cost function and its gradient.
    Updated for DOLFINx with proper MPI handling.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

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
            u0=u0, solver=solver, init_time=init_time, comm=comm, **kwargs
        )

    # Gradient function wrapper
    def grad_fn(u0):
        return grad_cost_function(
            u0=u0, solver=solver, adjoint_type=cost_function_type, comm=comm, **kwargs
        )

    def callback(x):
        cost = cost_fn(x)
        cost_function_values.append(cost)
        if rank == 0:
            print(f"Iteration {len(cost_function_values)}: Cost = {cost:.6f}")

    # Optimization options
    options = {
        "gtol": 1e-6,
        "ftol": 1e-12,
        "maxfun": 50,
        "maxiter": 50,
        "disp": True if rank == 0 else False,
    }

    result = minimize(
        fun=cost_fn,
        x0=u0,
        method="L-BFGS-B",
        jac=grad_fn,
        callback=callback,
        options=options,
    )

    # Print optimization results only on rank 0
    if rank == 0:
        print_optimization_summary(result)
        print_state_summary(u0, result, step=100)

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
    comm=None,
):
    """
    Run 4DVar analysis over assimilation windows using DOLFINx.
    Updated for proper MPI handling and DOLFINx function space operations.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    name = "Hotstart"
    analysis = []
    analysis_state = None
    num_windows = problem_params["num_windows"]
    steps_per_window = problem_params["num_steps"]

    # Calculate observation time indices per window more carefully
    obs_times_current_window = obs_time_indices[:obs_per_window]

    if rank == 0:
        print(
            f"Initial setup - obs_per_window: {obs_per_window}, total y_obs: {len(y_obs)}"
        )
        print(f"obs_time_indices length: {len(obs_time_indices)}")
        print(f"obs_times_current_window: {obs_times_current_window}")

    for idx in tqdm(
        range(num_windows),
        desc="Processing windows",
        unit="window",
        disable=(rank != 0),
    ):

        if rank == 0:
            print(f"\n[Window {idx + 1}/{num_windows}] Starting assimilation window...")

        # Extract observations for current window - ensure we don't exceed bounds
        start_idx = idx * obs_per_window
        end_idx = min(start_idx + obs_per_window, len(y_obs))
        indices = np.arange(start_idx, end_idx)
        yobs_current_window = y_obs[indices]

        # Update obs_per_window for this specific window (may be smaller for last window)
        current_obs_per_window = len(indices)

        if rank == 0:
            print(
                f"[Window {idx + 1}] Observation indices: {start_idx} to {end_idx-1} (total: {current_obs_per_window})"
            )

        # Update initial time for model
        initial_time = int(idx * steps_per_window * problem_params["dt"])
        problem_params_copy = problem_params.copy()
        problem_params_copy.update({"t": initial_time})

        # Create problem and solver
        _, solver = create_problem_solver(
            problem_params_copy, problem_type, true_signal=False
        )

        solver.problem.t = initial_time  # reset time to initial time
        V = solver.V  # get function spaces

        # Initialize state using DOLFINx Function
        u_0 = fem.Function(V)
        if analysis_state is None:
            # Use solver's initial condition
            u_0.x.array[:] = solver.u_n.x.array[:]
        else:
            # Use previous analysis state
            u_0.x.array[:] = analysis_state

        # Synchronize initial state across all processes
        u_0.x.scatter_forward()

        if rank == 0:
            print(f"[Window {idx + 1}] Solver initial time: {solver.problem.t}")

        # Generate background z_b - create a copy for background run
        initial_u0 = fem.Function(V)
        initial_u0.x.array[:] = u_0.x.array[:]
        initial_u0.x.scatter_forward()

        # Run background simulation
        solver.time_loop(
            solver_parameters=solver_params,
            stations=stations,
            plot_every=60,
            plot_name=name,
            u_0=initial_u0,
            save_states=True,
            adjoint_method=True,  # Enable adjoint computation for background
        )

        # Process background state - ensure all ranks have access to background
        if (
            rank == 0
            and hasattr(solver, "saved_states")
            and len(solver.saved_states) > 0
        ):
            background = np.array(solver.saved_states)  # shape: (steps, state_dim)
            print(f"[Window {idx + 1}] Background shape: {background.shape}")

            # Use only the observation time indices that correspond to this window
            current_obs_times = obs_times_current_window[:current_obs_per_window]
            observed_background_states = background[current_obs_times]
            Q_zb = H @ observed_background_states.T  # shape: (obs_dim, n_obs)
        else:
            # Handle case where background states are not available on this rank
            if rank == 0:
                print(
                    f"[Window {idx + 1}] Warning: No background states saved on rank {rank}"
                )
            Q_zb = np.zeros((H.shape[0], current_obs_per_window))
            current_obs_times = obs_times_current_window[:current_obs_per_window]

        # Broadcast background to all ranks
        Q_zb = comm.bcast(Q_zb, root=0)

        # Clear saved states for next iteration
        if hasattr(solver, "saved_states"):
            solver.saved_states = []

        if rank == 0:
            print(f"[Window {idx + 1}] Q_zb shape: {Q_zb.shape}")

        # Get initial state vectors
        z0 = u_0.x.array[:].copy()  # Use copy to avoid reference issues
        z_b = u_0.x.array[:].copy()

        # Get the actual state dimension for this solver instance
        actual_state_dim = len(z0)

        if rank == 0:
            print(
                f"[Window {idx + 1}] z0 shape: {z0.shape}, expected state_dim: {state_dim}, actual: {actual_state_dim}"
            )

        # Adjust covariance matrices if state dimension doesn't match
        if actual_state_dim != state_dim:
            if rank == 0:
                print(
                    f"[Window {idx + 1}] Adjusting covariance matrices from {state_dim} to {actual_state_dim}"
                )

            # Create new appropriately sized covariance matrices
            inflation_factor = 2.0
            B_local = inflation_factor * np.eye(actual_state_dim)
            B_inv_local = np.linalg.inv(B_local)

            # Update covs for this window
            covs_local = {
                "B_inv": B_inv_local,
                "R_inv": covs["R_inv"],
                "L_inv": covs["L_inv"],
            }
        else:
            covs_local = covs

        if rank == 0:
            print(
                f"[Window {idx + 1}] Starting optimization with state dim: {z0.shape}"
            )

        # Assimilation Step - perform optimization
        optimized_state, opt_result = optimize_4dvar(
            u0=z0,
            cost_function_type=cost_function_type,
            solver=solver,
            init_time=initial_time,
            comm=comm,
            u_b=z_b,
            y_obs=yobs_current_window,
            obs_spatial_idxs=obs_spatial_indices,
            obs_time_idxs=current_obs_times,
            H=H,
            covs=covs_local,  # Use locally adjusted covariance matrices
            Q_zb=Q_zb,
            stations=stations,
            hb=hb,
            solver_params=solver_params,
        )

        # Update state with optimized values
        u_0.x.array[:] = optimized_state
        u_0.x.scatter_forward()  # Ensure consistency across processes

        # Reset solver time and run analysis forward
        solver.problem.t = initial_time
        if rank == 0:
            print(
                f"[Window {idx + 1}] Running analysis forward from time: {solver.problem.t}"
            )

        solver.time_loop(
            solver_parameters=solver_params,
            stations=stations,
            plot_every=60,
            plot_name=name,
            u_0=u_0,
            adjoint_method=False,
        )

        # Save analysis state for next window - get final state
        if hasattr(solver, "u") and solver.u is not None:
            analysis_state = solver.u.x.array[:].copy()
        else:
            analysis_state = u_0.x.array[:].copy()

        # Collect results - handle the case where vals might not exist on all ranks
        if hasattr(solver, "vals") and solver.vals is not None:
            current_analysis = solver.vals.copy()
            # Remove last timestep to avoid overlap between windows (except for last window)
            if idx < num_windows - 1:
                current_analysis = current_analysis[:-1, :, :]
            analysis.append(current_analysis)
        else:
            if rank == 0:
                print(f"[Window {idx + 1}] Warning: No analysis results available")

        if rank == 0:
            print(f"[Window {idx + 1}] Completed successfully\n" + "=" * 80)

    # Combine all windows - only on rank 0 or where data exists
    if len(analysis) > 0:
        combined_analysis = np.concatenate(analysis, axis=0)
        if rank == 0:
            print(f"Final combined analysis shape: {combined_analysis.shape}")
        return combined_analysis
    else:
        if rank == 0:
            print("Warning: No analysis data to combine")
        return np.array([])  # Return empty array if no data

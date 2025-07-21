import numpy as np
from dolfinx import fem as fe
from petsc4py import PETSc
from scipy.optimize import minimize, OptimizeResult
from tqdm import tqdm
from typing import List, Dict, Tuple, Callable
import sys
from mpi4py import MPI
import pickle
import os
from dca_utils import (
    create_problem_solver,
    get_true_signal,
    build_observation_matrix,
    setup_observation_indices,
    generate_observations,
    Time,
)

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
    tqdm.write("\nOptimization completed:")
    tqdm.write(f"  Success: {result.success}")
    tqdm.write(f"  Status: {result.status}")
    tqdm.write(f"  Message: {result.message}")
    tqdm.write(f"  Final cost: {result.fun:.6e}")
    tqdm.write(f"  Iterations: {result.nit}")
    tqdm.write(f"  Function evaluations: {result.nfev}")
    tqdm.write(f"  Gradient norm at solution: {np.linalg.norm(result.jac):.6e}")
    tqdm.write("\n" + "-" * 60 + "\n")


# def print_state_summary(u0: np.ndarray, result: OptimizeResult, step: int = 40) -> None:
#     """
#     Print a summary of the initial and optimized state vectors.

#     Parameters
#     ----------
#     u0 : np.ndarray
#         Initial guess for the state vector.
#     result : OptimizeResult
#         Result object returned by `scipy.optimize.minimize`.
#     step : int, optional
#         Step size for subsampling the state vector when printing. Default is 20.
#     """
#     tqdm.write("State comparison (subsampled):")
#     tqdm.write(f"  Initial state (every {step}th entry):   {u0[::step]}\n")
#     tqdm.write(f"  Optimized state (every {step}th entry): {result.x[::step]}\n")


def print_state_summary(u0: np.ndarray, result: OptimizeResult, step: int = 40) -> None:
    """
    Print a comprehensive summary comparing initial and optimized state vectors.

    Parameters
    ----------
    u0 : np.ndarray
        Initial guess for the state vector.
    result : OptimizeResult
        Result object returned by `scipy.optimize.minimize`.
    step : int, optional
        Step size for subsampling when showing individual values. Default is 40.
    """
    u_opt = result.x

    # Compute difference arrays
    abs_diff = u_opt - u0
    rel_diff = np.divide(abs_diff, u0, out=np.zeros_like(abs_diff), where=u0 != 0)

    tqdm.write("=== STATE OPTIMIZATION SUMMARY ===")

    # Overall statistics
    # tqdm.write(f"Array length: {len(u0)}")

    # Difference statistics
    tqdm.write(f"\n--- DIFFERENCE STATISTICS ---")
    tqdm.write(f"Max absolute change: {np.max(np.abs(abs_diff)):.6e}")
    tqdm.write(f"Mean absolute change: {np.mean(np.abs(abs_diff)):.6e}")
    tqdm.write(f"RMS change: {np.sqrt(np.mean(abs_diff**2)):.6e}")
    tqdm.write(f"L2 norm of change: {np.linalg.norm(abs_diff):.6e}")

    # Relative changes (where initial values are non-zero)
    nonzero_mask = u0 != 0
    if np.any(nonzero_mask):
        rel_changes = np.abs(rel_diff[nonzero_mask])
        tqdm.write(
            f"Max relative change: {np.max(rel_changes):.6e} ({np.max(rel_changes)*100:.3f}%)"
        )
        tqdm.write(
            f"Mean relative change: {np.mean(rel_changes):.6e} ({np.mean(rel_changes)*100:.3f}%)"
        )

    # Check if arrays are close
    tqdm.write(f"\n--- CONVERGENCE CHECK ---")
    rtol, atol = 1e-5, 1e-8
    is_close = np.allclose(u0, u_opt, rtol=rtol, atol=atol)
    tqdm.write(f"Arrays close (rtol={rtol}, atol={atol}): {is_close}")

    # Count significant changes
    sig_changes = np.sum(np.abs(rel_diff) > 0.01)  # Changes > 1%
    tqdm.write(
        f"Elements with >1% relative change: {sig_changes}/{len(u0)} ({100*sig_changes/len(u0):.1f}%)"
    )


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
        tqdm.write(f"Iteration {len(cost_function_values)}: Cost = {cost:.6f}")

    # options = {"gtol": 1e-6, "ftol": 1e-12, "maxfun": 10, "maxiter": 1000, "disp": True}

    result = minimize(
        fun=cost_fn,
        x0=u0,
        method="L-BFGS-B",
        jac=grad_fn,
        # callback=callback,
    )

    if not result.success:
        tqdm.write(f"Optimization failed:\n")
        print_optimization_summary(result)
        # sys.exit(1)

    # Print optimization results
    # print_optimization_summary(result)

    # Print state comparison
    # print_state_summary(u0, result, step=200)

    return result.x, result


def run_assimilation(
    problem_params,
    solver_params,
    stations,
    y_obs,
    obs_per_window,
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
        # print(f"Solver Time 1: {solver.problem.t}")
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
        # print(f"Solver Time 2: {solver.problem.t}")  # Debugging line
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


def get_background_covariance(eta_trajectory, sample_freq=1, err2=1.0):
    """
    Estimate background covariance matrix from a trajectory of water surface elevation (η).

    Parameters
    ----------
    eta_trajectory : ndarray of shape (n_timesteps, n_space_points)
        Time series of η values from a shallow water model.
    sample_freq : int
        Temporal sampling frequency to reduce autocorrelation (e.g., every 10 time steps).
    err2 : float
        Target maximum variance for scaling the background covariance.

    Returns
    -------
    B : ndarray (n_space_points x n_space_points)
        Scaled background error covariance matrix.
    Bcorr : ndarray (n_space_points x n_space_points)
        Corresponding correlation matrix.
    """
    # Subsample the time series to reduce temporal correlation
    eta_sampled = eta_trajectory[::sample_freq, :]  # shape: (n_samples, n_space_points)

    # Compute correlation matrix
    Bcorr = np.corrcoef(
        eta_sampled, rowvar=False
    )  # shape: (n_space_points, n_space_points)

    # Compute sample covariance matrix
    B = np.cov(eta_sampled, rowvar=False)  # shape: (n_space_points, n_space_points)

    # Scale covariance matrix so max variance equals err2
    max_var = np.max(np.diag(B))
    if max_var > 0:
        alpha = err2 / max_var
        B *= alpha

    return B


def rescale_background_covariance_to_observation(B_eta, H, R):
    """
    Rescales the background covariance matrix B_eta so that its projection
    onto the observation space matches the scale of the observation error covariance R.

    Parameters
    ----------
    B_eta : ndarray (n_state x n_state)
        Background covariance matrix for η (e.g., from trajectory).
    H : ndarray (n_obs x n_state)
        Observation operator matrix (e.g., row selector from identity).
    R : ndarray (n_obs x n_obs)
        Observation error covariance matrix (usually diagonal).

    Returns
    -------
    B_eta_rescaled : ndarray (n_state x n_state)
        Rescaled background covariance matrix.
    scaling_factor : float
        Factor by which the original B_eta was scaled.
    """
    # Project B_eta into observation space
    B_y = H @ B_eta @ H.T  # shape (n_obs x n_obs)

    # Compute trace of B_y and R
    trace_B_y = np.trace(B_y)
    trace_R = np.trace(R)

    # Compute scaling factor
    scaling_factor = trace_R / trace_B_y if trace_B_y > 0 else 1.0

    # Rescale B_eta
    B_eta_rescaled = scaling_factor * B_eta

    return B_eta_rescaled, scaling_factor


def setup_data_assimilation(
    pickle_path,
    problem_params,
    prob,
    obs_std=1.1,
    obs_space_freq=2,
    obs_time_freq=4,
    final_time=None,
    inflation_factor=5.0,
    print_setup=False,
):
    """
    Set up data assimilation experiment with observations and covariance matrices.

    Parameters:
    -----------
    pickle_path : str
        Path to the pickle file containing true_signal
    problem_params : dict
        Dictionary containing problem parameters
    prob : object
        Problem object with V attribute
    obs_std : float, default=1.1
        Standard deviation for observations
    obs_space_freq : int, default=2
        Frequency of observation in space (every nth cell)
    obs_time_freq : int, default=4
        Frequency of observation in time
    final_time : float, optional
        Final time for the problem (uses problem_params['t_final'] if None)
    inflation_factor : float, default=5.0
        Inflation factor for background covariance matrix
    print_setup : bool, default=False
        If True, prints all the settings for the data assimilation experiment

    Returns:
    --------
    dict : Dictionary containing:
        - 'true_signal': Loaded true signal
        - 'problem_params': Updated problem parameters
        - 'y_obs': Generated observations
        - 'covs': Dictionary with inverse covariance matrices
        - 'H': Observation matrix
        - 'stations': Station locations
        - 'obs_spatial_indices': Spatial observation indices
        - 'obs_indices_per_window': Observation indices per window
        - 'obs_time_indices': Time observation indices
        - 'hb': Computed hb values

    Note:
    -----
    Requires the following functions to be available:
    - build_observation_matrix(prob, prob.V, obs_space_freq)
    - setup_observation_indices(num_steps, obs_time_freq, total_steps)
    - generate_observations(true_signal, H, obs_time_indices, obs_std)
    """
    # Load true signal from pickle
    with open(pickle_path, "rb") as f:
        true_signal = pickle.load(f)

    # Update problem parameters
    problem_params.update({"fric_law": "mannings", "dt": 600})

    # Use final_time parameter or get from problem_params
    if final_time is None:
        final_time = problem_params["t_final"]

    # Calculate time parameters
    total_steps = int((problem_params["t_final"] / problem_params["dt"]) + 1)
    problem_params["num_steps"] = int(
        np.ceil(final_time / problem_params["dt"]) / problem_params["num_windows"]
    )
    obs_per_window = problem_params["num_steps"] // obs_time_freq

    # Build observation matrix
    H, stations, obs_spatial_indices = build_observation_matrix(
        prob, prob.V, obs_space_freq
    )

    # Setup observation indices
    obs_indices_per_window, obs_time_indices = setup_observation_indices(
        problem_params["num_steps"], obs_time_freq, total_steps
    )

    # Create synthetic observations
    y_obs = generate_observations(true_signal, H, obs_time_indices, obs_std)

    # Generate covariance matrices
    state_dim = true_signal.shape[1]
    obs_dim = stations.shape[0]

    # Observation covariance
    R = np.eye(obs_dim) * (obs_std**2)

    # Background covariance
    B = inflation_factor * np.eye(state_dim)
    # B = get_background_covariance(true_signal, sample_freq=12, err2=0.1)

    # B, scaling_factor = rescale_background_covariance_to_observation(B, H, R)

    # Predicted covariance
    L = H @ B @ H.T

    # Get inverse covariance matrices
    R_inv = np.linalg.inv(R)
    B_inv = np.linalg.inv(B)
    L_inv = np.linalg.inv(L)
    covs = {"B_inv": B_inv, "R_inv": R_inv, "L_inv": L_inv}

    # Calculate hb
    hb = 5.0 / 13800 * (13800 - stations[:, 0])

    # Print setup information if requested
    if print_setup:
        print("=" * 60)
        print("DATA ASSIMILATION EXPERIMENT SETUP")
        print("=" * 60)
        print("\nInput Parameters:")
        print(f"  Pickle path: {pickle_path}")
        print(f"  Observation std deviation: {obs_std}")
        print(f"  Observation space frequency: {obs_space_freq}")
        print(f"  Observation time frequency: {obs_time_freq}")
        print(f"  Final time: {final_time}")
        print(f"  Inflation factor: {inflation_factor}")

        print("\nProblem Parameters:")
        for key, value in problem_params.items():
            print(f"  {key}: {value}")

        print("\nCalculated Dimensions:")
        print(f"  State dimension: {state_dim}")
        print(f"  Observation dimension: {obs_dim}")
        print(f"  Total time steps: {total_steps}")
        print(f"  Observations per window: {obs_per_window}")
        print(f"  Number of observation stations: {len(stations)}")

        print("\nMatrix Information:")
        print(f"  Observation matrix H shape: {H.shape}")
        print(f"  Background covariance B shape: {B.shape}")
        print(f"  Observation covariance R shape: {R.shape}")
        print(f"  Predicted covariance L shape: {L.shape}")

        print("\nObservation Setup:")
        print(f"  Number of observation time indices: {len(obs_time_indices)}")
        print(
            f"  Observation spatial indices: {obs_spatial_indices[:10]}..."
            if len(obs_spatial_indices) > 10
            else f"  Observation spatial indices: {obs_spatial_indices}"
        )
        print(
            f"  Station locations (first 5): {stations[:5] if len(stations) > 5 else stations}"
        )

        print("\nGenerated Data:")
        print(f"  True signal shape: {true_signal.shape}")
        print(f"  Observations y_obs shape: {y_obs.shape}")
        print(f"  hb values (first 5): {hb[:5] if len(hb) > 5 else hb}")
        print("=" * 60)

    return {
        "true_signal": true_signal,
        "problem_params": problem_params,
        "y_obs": y_obs,
        "covs": covs,
        "H": H,
        "stations": stations,
        "obs_spatial_indices": obs_spatial_indices,
        "obs_per_window": obs_per_window,
        "obs_time_indices": obs_time_indices,
        "hb": hb,
    }


def run_data_assimilation(
    analysis_types=["dci", "dci_wme", "bayes"],
    run_true_signal=True,
    window_size=None,
    dt=600,
    obs_std=1.1,
    obs_space_freq=2,
    obs_time_freq=1,
    inflation_factor=4.0,
):
    """
    Run data assimilation with specified analysis types.

    Parameters:
    -----------
    analysis_types : list or str
        Which analysis types to run. Can be:
        - A single string: 'dci', 'dci_wme', or 'bayes'
        - A list of strings: ['dci', 'dci_wme'], ['dci', 'bayes'], etc.
        - 'all' to run all three types
    run_true_signal : bool
        Whether to generate the true signal (default: True)
    window_size : float or None
        Window size in seconds. If None, defaults to 1 hour (default: None)
    dt : float
        Time step in seconds (default: 600, i.e., 10 minutes)
    obs_std : float
        Observation standard deviation (default: 1.1)
    obs_space_freq : int
        Observation spatial frequency (default: 2)
    obs_time_freq : int
        Observation temporal frequency (default: 1)
    inflation_factor : float
        Inflation factor for covariance (default: 4.0)

    Returns:
    --------
    dict : Dictionary containing results for each requested analysis type
    """

    # Handle input validation and conversion
    if isinstance(analysis_types, str):
        if analysis_types.lower() == "all":
            analysis_types = ["dci", "dci_wme", "bayes"]
        else:
            analysis_types = [analysis_types]

    # Validate analysis types
    valid_types = {"dci", "dci_wme", "bayes"}
    analysis_types = [atype.lower() for atype in analysis_types]
    invalid_types = set(analysis_types) - valid_types
    if invalid_types:
        raise ValueError(
            f"Invalid analysis types: {invalid_types}. Valid types are: {valid_types}"
        )

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Problem parameters
    final_time = 7 * Time.TWENTY_FOUR_HOURS.seconds  # 7 days in seconds

    # Set default window_size to 1 hour if not provided
    if window_size is None:
        window_size = Time.ONE_HOUR.seconds

    problem_params = {
        "dt": dt,
        "t": 0,
        "t_final": final_time,
        "num_steps": int(np.ceil(final_time / dt)),
        "num_windows": final_time // window_size,  # divide by window_size in seconds
        "fric_law": "linear",  # friction law either quadratic or linear
        "sol_var": "h",  # solution variable either h or hu
    }

    solver_params = {
        "rtol": 1e-5,
        "atol": 1e-6,
        "max_it": 10,
        "relaxation_parameter": 1.0,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_ErrorIfNotConverged": False,
    }

    # Create problem and solver
    prob, solver = create_problem_solver(
        problem_params, "sloped_beach", true_signal=True, verbose=True
    )

    # Generate true signal if requested
    if run_true_signal:
        assert problem_params["num_steps"] == int(
            np.ceil(problem_params["t_final"] / problem_params["dt"])
        )
        true_solver = get_true_signal(solver, "sloped_beach", solver_params, 1)
        true_signal = np.array(true_solver.saved_states)

        # Ensure output directory exists
        os.makedirs("da_output", exist_ok=True)

        with open(os.path.join("da_output", "true_signal.pkl"), "wb") as f:
            pickle.dump(true_signal, f)

    # Setup data assimilation
    result = setup_data_assimilation(
        pickle_path=os.path.join("da_output", "true_signal.pkl"),
        problem_params=problem_params,
        prob=prob,
        obs_std=obs_std,
        obs_space_freq=obs_space_freq,
        obs_time_freq=obs_time_freq,
        final_time=final_time,
        inflation_factor=inflation_factor,
        print_setup=True,
    )

    # save the setup result to a pickle file
    with open(os.path.join("da_output", "setup_result.pkl"), "wb") as f:
        pickle.dump(result, f)

    # Extract common parameters from result
    stations = result["stations"]
    y_obs = result["y_obs"]
    obs_per_window = result["obs_per_window"]
    obs_time_indices = result["obs_time_indices"]
    H = result["H"]
    covs = result["covs"]
    hb = result["hb"]

    # Dictionary to store analysis results
    analyses = {}

    # Run requested analysis types
    if "dci" in analysis_types:
        print("Running DCI analysis...")
        dci_analysis = run_assimilation(
            problem_params,
            solver_params,
            stations,
            y_obs,
            obs_per_window,
            obs_time_indices,
            H,
            covs,
            hb,
            "sloped_beach",
            cost_function_type="dci",
        )

        # Save to pickle file
        with open(os.path.join("da_output", "dci_analysis.pkl"), "wb") as f:
            pickle.dump(dci_analysis, f)

        analyses["dci"] = dci_analysis
        print("DCI analysis completed and saved.")

    if "dci_wme" in analysis_types:
        print("Running DCI-WME analysis...")
        dci_wme_analysis = run_assimilation(
            problem_params,
            solver_params,
            stations,
            y_obs,
            obs_per_window,
            obs_time_indices,
            H,
            covs,
            hb,
            "sloped_beach",
            cost_function_type="dci_wme",
        )

        # Save to pickle file
        with open(os.path.join("da_output", "dci_wme_analysis.pkl"), "wb") as f:
            pickle.dump(dci_wme_analysis, f)

        analyses["dci_wme"] = dci_wme_analysis
        print("DCI-WME analysis completed and saved.")

    if "bayes" in analysis_types:
        print("Running Bayes analysis...")
        bayes_analysis = run_assimilation(
            problem_params,
            solver_params,
            stations,
            y_obs,
            obs_per_window,
            obs_time_indices,
            H,
            covs,
            hb,
            "sloped_beach",
            cost_function_type="bayes",
        )

        # Save to pickle file
        with open(os.path.join("da_output", "bayes_analysis.pkl"), "wb") as f:
            pickle.dump(bayes_analysis, f)

        analyses["bayes"] = bayes_analysis
        print("Bayes analysis completed and saved.")

    return analyses


# Example usage:
# Run all three analyses with default parameters
# results = run_data_assimilation('all')

# Run only DCI
# results = run_data_assimilation('dci')

# Run DCI and Bayes
# results = run_data_assimilation(['dci', 'bayes'])

# Run with custom time parameters
# results = run_data_assimilation(['dci', 'bayes'], dt=300, window_size=7200)  # 5 min timestep, 2 hour windows

# Run with custom observation and time parameters
# results = run_data_assimilation(['dci_wme', 'bayes'],
#                                dt=900,  # 15 minute timesteps
#                                window_size=3600*4,  # 4 hour windows
#                                obs_std=1.5,
#                                inflation_factor=3.0)

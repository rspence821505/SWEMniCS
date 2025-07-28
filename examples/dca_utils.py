import numpy as np
from mpi4py import MPI
from swemnics.problems import SlopedBeachProblem, TidalProblem
from swemnics import solvers as Solvers
import pickle
import os


from mpi4py import MPI

from enum import IntEnum
from pathlib import Path


def save_pickle(filename, data, directory="da_output"):
    with open(Path(directory) / filename, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filename, directory="da_output"):
    filepath = Path(directory) / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")
    with open(filepath, "rb") as f:
        return pickle.load(f)


class Time(IntEnum):
    """
    Enum for common time durations in seconds.

    Using IntEnum allows these values to be used directly in arithmetic
    operations and comparisons while providing better organization and
    type safety compared to plain constants.
    """

    ONE_HOUR = 3600
    TWO_HOURS = 7200
    FOUR_HOURS = 14400
    SIX_HOURS = 21600
    EIGHT_HOURS = 28800
    TWELVE_HOURS = 43200
    TWENTY_FOUR_HOURS = 86400

    def __str__(self) -> str:
        """Return human-readable string representation."""
        hours = self.value // 3600
        if hours == 1:
            return "1 hour"
        else:
            return f"{int(hours)} hours"

    @property
    def hours(self) -> float:
        """Return duration in hours."""
        return self.value // 3600

    @property
    def minutes(self) -> float:
        """Return duration in minutes."""
        return self.value / 60

    @property
    def seconds(self) -> int:
        """Return duration in seconds."""
        return self.value


def create_problem_solver(
    problem_params, problem_type="sloped_beach", true_signal=True, verbose=False
):
    """
    Create a problem and solver based on the problem type and parameters.
    """
    common_solver_kwargs = {
        "theta": 1,
        "p_degree": [1, 1],
        "verbose": verbose,
        # "adjoint_method": False,
        # "save_state": True,
        # "save_true_bathy": True,
        # "save_bathy": False,
    }

    if true_signal:

        sloped_kwargs = {
            "dt": problem_params["dt"],
            "nt": problem_params["num_steps"],
            "friction_law": problem_params["fric_law"],
            "solution_var": problem_params["sol_var"],
            "verbose": verbose,
            "wd_alpha": 0.36,
            "wd": True,
            "alpha": 2.0 * np.pi / Time.TWELVE_HOURS.seconds,  # 2 cycles per 12 hours
            # "alpha": 2.0 * np.pi / (10 * Time.ONE_HOUR.seconds),
            "h_b_val": 5.3,
        }
        prob = SlopedBeachProblem(**sloped_kwargs)
        solver = Solvers.DGImplicit(prob, **common_solver_kwargs)
    else:
        sloped_kwargs = {
            "dt": problem_params["dt"],
            "nt": problem_params["num_steps"],
            "friction_law": problem_params["fric_law"],
            "solution_var": problem_params["sol_var"],
            "verbose": verbose,
            "wd_alpha": 0.36,
            "wd": True,
            "alpha": 2.0 * np.pi / Time.TWELVE_HOURS.seconds,  # 2 cycles per 12 hours
            # "alpha": 2.0 * np.pi / (10 * Time.ONE_HOUR.seconds),
            "h_b_val": 5.0,  # Uncomment if needed
            # "alpha": 0.00024, # Uncomment if needed
        }

        prob = SlopedBeachProblem(**sloped_kwargs)
        solver = Solvers.DGImplicit(prob, **common_solver_kwargs)

    if "t" in problem_params:
        solver.problem.t = problem_params["t"]
    return prob, solver


def get_true_signal(solver, problem_type, solver_params, obs_frequency=1):
    """
    Default values are sea level and 0 velocity
    """

    u_0 = solver.u_n  # full initial condition

    # Doesn't work for DG Case
    V = solver.V  # create full function space
    V_coords = (
        V.sub(0).collapse()[0].tabulate_dof_coordinates()
    )  # collapse to height function space

    stations = V_coords

    solver.time_loop(
        solver_parameters=solver_params,
        stations=stations,
        plot_every=60,
        plot_name="sloped_beach_true_signal",
        u_0=u_0,
        save_state=True,
        adjoint_method=True,
        save_bathy=False,
        save_true_bathy=True,
        make_wet=True,
    )
    # Save the true signal

    save_pickle("true_signal.pkl", np.array(solver.saved_states))
    save_pickle("true_bathy.pkl", np.array(solver.saved_true_bathy))
    return solver


def generate_observations(true_states, H, obs_time_idx, hb, obs_std=0.1):
    # Extract only the states at the observation indices

    np.random.seed(42)  # For reproducibility
    if isinstance(true_states, list):
        true_states = np.array(true_states)  # Ensure true_states is a numpy array

    # bathy = load_pickle("saved_bathy.pkl")

    observed_states = true_states[obs_time_idx]  # shape: (n_obs, state_dim)
    # Select only the first variable (e.g., sea level)
    print(
        f"\n\n Number of dry observations observed_states: {np.sum(observed_states == 0)}"
    )

    # Apply observation operator to all observed states at once
    y_n = observed_states @ H.T  # shape: (n_obs, obs_dim)
    print(f"\n\n Number of dry observations y_n: {np.sum(y_n == 0)}")
    y_n = y_n - hb  # convert to wse eta = H - hb

    # Add Gaussian noise
    noise = obs_std * np.random.randn(*y_n.shape)
    # y_obs = y_n + noise
    y_obs = np.maximum(y_n + noise, -hb)
    print(f"Number of dry observations y_obs: {np.sum(y_obs == 0)}\n\n")

    return y_obs


def setup_observation_indices(window_size, obs_frequency, total_steps):
    """Setup observation indices for windows"""
    obs_indices_per_window = np.arange(0, window_size, obs_frequency)
    obs_indices = np.arange(0, total_steps, obs_frequency)
    return obs_indices_per_window, obs_indices


def build_observation_matrix(prob, V, obs_space_freqs=2):
    num_cells = len(prob.mesh.geometry.dofmap)

    all_cells = np.arange(num_cells)
    obs_space_idx = np.arange(
        0, num_cells, obs_space_freqs
    )  # select every obs_space_freqs-th cell for observation
    station_cells = all_cells[obs_space_idx]  # select cells for observation

    # Create observation matrix
    H = np.zeros((len(station_cells), V.dofmap.index_map.size_local))

    # pick subset of cells for the stations
    station_coords = []

    # collapse the function space to get the coordinates of the dofs
    # in the cells that are selected for observation
    V_collapsed, indices_into_V = V.sub(0).collapse()
    collapsed_dof_coords = V_collapsed.tabulate_dof_coordinates()
    indices_into_V = np.array(indices_into_V)

    for station, i in enumerate(station_cells):
        coords_for_cell = collapsed_dof_coords[V_collapsed.dofmap.cell_dofs(i)]
        dofs_in_orig_V = indices_into_V[V_collapsed.dofmap.cell_dofs(i)]
        H[station, dofs_in_orig_V] = 1 / 3
        station_coord = 1 / 3 * (coords_for_cell.sum(axis=0))
        station_coords.append(station_coord)
        # print(f"Station {station} at {station_coord} corresponds to cell {i} with dofs {dofs_in_orig_V}")

    # print(
    #     f"Total cells in mesh: {num_cells}\n"
    #     f"obs_space_freqs: {obs_space_freqs}\n"
    #     f"obs_space_idx: {obs_space_idx}\n"
    #     f"station_cells: {station_cells}\n"
    #     f"Number of observation stations: {len(station_cells)}"
    # )

    return H, np.array(station_coords), obs_space_idx


# def build_observation_matrix(prob, V, obs_space_freqs=2):
# num_cells = len(prob.mesh.geometry.dofmap)
# all_cells = np.arange(num_cells)
# start = 0
# obs_space_idx = np.arange(
#     start, num_cells, obs_space_freqs
# )  # select every obs_space_freqs-th cell for observation
# station_cells = all_cells[obs_space_idx]  # select cells for observation

# # collapse the function space to get the coordinates of the dofs
# # in the cells that are selected for observation
# V_collapsed, indices_into_V = V.sub(0).collapse()
# collapsed_dof_coords = V_collapsed.tabulate_dof_coordinates()
# indices_into_V = np.array(indices_into_V)

# # Get all DOF indices for station cells at once
# all_dof_indices = np.array([V_collapsed.dofmap.cell_dofs(i) for i in station_cells])

# # Reshape to (num_stations, dofs_per_cell) and index coordinates
# coords_for_cells = collapsed_dof_coords[all_dof_indices]
# sum_coords = coords_for_cells.sum(axis=1)  # Sum over dofs for each cell
# station_coords = (
#     1 / 3 * sum_coords
# )  # Averaged coordinate in middle of triangle for each station

# # col = station_coords[:, 1]
# # col2 = station_coords[:, 0]
# # mask_1 = (
# #     ((600 <= col) & (col <= 900))
# #     | ((1700 <= col) & (col <= 2000))
# #     | ((3000 <= col) & (col <= 3500))
# #     | ((4100 <= col) & (col <= 5000))
# #     | ((5500 <= col) & (col <= 6000))
# #     | ((6100 <= col) & (col <= 6500))
# # )

# # mask_2 = (1000 <= col2) & (col2 <= 12000)
# # mask_2 = col2 <= 11000
# # Apply the mask
# stat_coord = station_coords[:, 0] < 9000

# wet_idxs = np.where(stat_coord)[0]  # Get indices of wet stations
# print(f"wet_idx shape: {wet_idxs.shape}\n\n")
# station_coords = station_coords[
#     wet_idxs
# ]  # Filter coordinates based on x coordinate (avoid stations on dry nodes)
# # print(f"Station coordinates: {station_coords.shape} \n {station_coords}\n\n")

# # update obs_space_idx to only include wet stations
# obs_space_idx = obs_space_idx[
#     wet_idxs
# ]  # Filter observation indices based on wet station indices

# # update station_cells to only include wet stations
# station_cells = station_cells[wet_idxs]  # Filter station cells based on wet station

# # Create observation matrix
# H = np.zeros((len(wet_idxs), V.dofmap.index_map.size_local))
# for station, i in enumerate(station_cells):
#     dofs_in_orig_V = indices_into_V[V_collapsed.dofmap.cell_dofs(i)]
#     H[station, dofs_in_orig_V] = 1 / 3

# # print(f"wet_idxs: {wet_idxs}\n\n")

# print(
#     f"Total cells in mesh: {num_cells}\n\n"
#     # f"obs_space_freqs: {obs_space_freqs}\n\n"
#     f"obs_space_idx: {obs_space_idx}\n\n"
#     f"station_cells: {station_cells}\n\n"
#     f"Number of observation stations: {len(station_cells)}\n\n"
# )

# return H, np.array(station_coords), obs_space_idx


def analyze_parameter_crossval_results(results_dict):
    """
    Analyze the results of a parameter cross validation to identify patterns in success/failure.

    Parameters:
    -----------
    results_dict : dict
        Dictionary returned by run_parameter_crossval

    Returns:
    --------
    dict : Analysis summary
    """
    analysis = {
        "total_experiments": len(results_dict),
        "successful": [],
        "convergence_failures": [],
        "other_failures": [],
        "success_rate": 0,
        "convergence_failure_rate": 0,
    }

    for exp_name, result in results_dict.items():
        if "error" not in result:
            analysis["successful"].append(
                (result["obs_std"], result["inflation_factor"])
            )
        elif result.get("error_type") == "convergence_failure":
            analysis["convergence_failures"].append(
                (result["obs_std"], result["inflation_factor"])
            )
        else:
            analysis["other_failures"].append(
                (result["obs_std"], result["inflation_factor"], result["error"])
            )

    analysis["success_rate"] = (
        len(analysis["successful"]) / analysis["total_experiments"]
    )
    analysis["convergence_failure_rate"] = (
        len(analysis["convergence_failures"]) / analysis["total_experiments"]
    )

    print("Parameter cross validation Analysis:")
    print("=" * 50)
    print(f"Total experiments: {analysis['total_experiments']}")


def load_and_analyze_crossval(summary_pickle_path):
    """
    Load parameter cross validation results from pickle file and analyze them.

    Parameters:
    -----------
    summary_pickle_path : str
        Path to the parameter_crossval_summary.pkl file
    """
    with open(summary_pickle_path, "rb") as f:
        results = pickle.load(f)

    return analyze_parameter_crossval_results(results)


def analyze_error_statistics(
    obs_std_list,
    inflation_list,
    result,
    analysis_type="dci_wme",
    output_dir="examples/da_output",
):
    """
    Calculate and print error statistics for different observation standard deviations and inflation factors.

    Parameters:
    -----------
    obs_std_list : list
        List of observation standard deviation values to analyze
    inflation_list : list
        List of inflation factor values to analyze
    result : dict
        Dictionary containing observation operator (H) and time indices
    analysis_type : str
        Type of analysis ('dci_wme', 'dci', 'bayes') (default: 'dci_wme')
    output_dir : str
        Directory containing the pickle files (default: 'examples/da_output')

    Returns:
    --------
    dict : Dictionary containing RMSE and misfit values for each parameter combination
    """

    def calculate_error_metrics(analysis, true_signal, result):
        """Calculate RMSE and misfit for a given analysis."""
        # Calculate observations
        true_obs = result["H"] @ true_signal[result["obs_time_indices"]].T
        pred_obs = result["H"] @ analysis[result["obs_time_indices"]].T

        # Calculate metrics
        rmse = np.sqrt(np.mean((true_signal - analysis) ** 2))  # time averaged RMSE
        misfit = np.linalg.norm(true_obs - pred_obs, ord=2)  # time averaged misfit

        return rmse, misfit

    # Dictionary to store results
    results_dict = {}

    print(f"Error Statistics for {analysis_type.upper()} Analysis")
    print("=" * 60)
    print(f"{'Obs Std':<10} {'Inflation':<12} {'RMSE':<15} {'Misfit':<15}")
    print("-" * 60)

    true_signal = load_pickle("true_signal.pkl")

    # Loop through all combinations of obs_std and inflation values
    for obs_std in obs_std_list:
        for inflation in inflation_list:
            # Construct filename
            # print(
            #     f"analysis_type: {analysis_type}, obs_std: {obs_std}, inflation: {inflation}"
            # )
            filename = (
                f"{analysis_type}_analysis_obs_std_{obs_std}_inflation_{inflation}.pkl"
            )

            try:
                # Load analysis results

                analysis = load_pickle(filename)

                # Calculate error metrics
                rmse, misfit = calculate_error_metrics(analysis, true_signal, result)

                # Store results
                key = (obs_std, inflation)
                results_dict[key] = {"rmse": rmse, "misfit": misfit}

                # Print results
                print(f"{obs_std:<.3f} {inflation:<.3f} {rmse:<.6f} {misfit:<.6f}")

            except FileNotFoundError:
                print(
                    f"{obs_std:<.3f} {inflation:<.3f} {'FILE NOT FOUND':<15} {'N/A':<15}"
                )
                results_dict[(obs_std, inflation)] = {"rmse": None, "misfit": None}
            except Exception as e:
                print(
                    f"{obs_std:<.3f} {inflation:<.3f} {'ERROR':<15} {str(e)[:10]:<15}"
                )
                results_dict[(obs_std, inflation)] = {"rmse": None, "misfit": None}

    print("-" * 60)
    return results_dict


def calculate_daily_differences(
    true_signal, bayes_analysis, dci_analysis, dci_wme_analysis, num_days=7
):
    """
    Calculate differences between true signal and estimates for each day (explicit version).
    """
    time_steps_per_day = true_signal.shape[0] // num_days
    plot_triplets = []

    # Process days 1 through (num_days-1)
    for day in range(1, num_days):
        time_index = day * time_steps_per_day

        true_val = true_signal[time_index, :]
        bayes_est = bayes_analysis[time_index, :]
        dci_est = dci_analysis[time_index, :]
        dci_wme_est = dci_wme_analysis[time_index, :]

        differences = (
            true_val - bayes_est,  # bayes_diff
            true_val - dci_est,  # dci_diff
            true_val - dci_wme_est,  # dci_wme_diff
        )
        plot_triplets.append(differences)

    # Handle the final day (use last time step)
    final_index = true_signal.shape[0] - 1
    true_val = true_signal[final_index, :]
    bayes_est = bayes_analysis[final_index, :]
    dci_est = dci_analysis[final_index, :]
    dci_wme_est = dci_wme_analysis[final_index, :]

    final_differences = (
        true_val - bayes_est,
        true_val - dci_est,
        true_val - dci_wme_est,
    )
    plot_triplets.append(final_differences)

    return plot_triplets


def calculate_daily_comparison(true_signal, analysis_result, num_days=7):
    """
    Calculate true values, estimates, and differences for each day (explicit version).
    """
    time_steps_per_day = true_signal.shape[0] // num_days
    plot_triplets = []

    # Process days 1 through (num_days-1)
    for day in range(1, num_days):
        time_index = day * time_steps_per_day

        true_val = true_signal[time_index, :]
        estimate = analysis_result[time_index, :]
        diff = true_val - estimate

        plot_triplets.append((true_val, estimate, diff))

    # Handle the final day (use last time step)
    final_index = true_signal.shape[0] - 1
    true_val = true_signal[final_index, :]
    estimate = analysis_result[final_index, :]
    diff = true_val - estimate

    plot_triplets.append((true_val, estimate, diff))

    return plot_triplets

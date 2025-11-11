import os
import numpy as np
import pandas as pd
from mpi4py import MPI
from swemnics.forward.problems import SlopedBeachProblem
from swemnics.forward import solvers as Solvers
from data_io import save_pickle, load_pickle
from metrics import rmse, relative_misfit
from collections import defaultdict

from enum import IntEnum


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
    ONE_DAY = 86400
    TWO_DAYS = 172800
    THREE_DAYS = 259200
    SEVEN_DAYS = 604800

    def __str__(self) -> str:
        """Return human-readable string representation."""
        total_seconds = self.value

        # Check if it's exactly divisible by days
        if total_seconds >= 86400 and total_seconds % 86400 == 0:
            days = total_seconds // 86400
            if days == 1:
                return "1 day"
            else:
                return f"{int(days)} days"

        # Check if it's exactly divisible by hours
        elif total_seconds % 3600 == 0:
            hours = total_seconds // 3600
            if hours == 1:
                return "1 hour"
            else:
                return f"{int(hours)} hours"

        # Fallback to seconds
        else:
            return f"{total_seconds} seconds"

    @property
    def hours(self) -> float:
        """Return duration in hours."""
        return self.value / 3600

    @property
    def minutes(self) -> float:
        """Return duration in minutes."""
        return self.value / 60

    @property
    def seconds(self) -> int:
        """Return duration in seconds."""
        return self.value

    @property
    def days(self) -> float:
        """Return duration in days."""
        return self.value / 86400


def create_problem_solver(
    problem_params, problem_type="sloped_beach", true_signal=True, verbose=False
):
    """
    Create a problem and solver based on the problem type and parameters.
    """
    # Common solver configuration
    common_solver_kwargs = {
        "theta": 1,
        "p_degree": [1, 1],
        "verbose": verbose,
    }

    # Base problem configuration
    sloped_kwargs = {
        "dt": problem_params["dt"],
        "nt": problem_params["num_steps"],
        "friction_law": problem_params["fric_law"],
        "solution_var": problem_params["sol_var"],
        "verbose": verbose,
        "wd_alpha": 0.36,
        "wd": True,
        "mag": 0.75,
        "alpha": 2.0 * np.pi / Time.TWELVE_HOURS.seconds,  # 2 cycles per 12 hours
        "h_b_val": 5.3,
    }

    # Additional parameters for data assimilation case
    if not true_signal:
        sloped_kwargs["h_b_val"] = 5.0

    # Create problem and solver
    prob = SlopedBeachProblem(**sloped_kwargs)
    solver = Solvers.DGImplicit(prob, **common_solver_kwargs)

    # Set time if provided
    if "t" in problem_params:
        solver.problem.t = problem_params["t"]

    return prob, solver


def get_true_signal(solver, problem_type, solver_params, obs_frequency=1):
    """
    Default values are sea level and 0 velocity
    """

    u_0 = solver.u_n  # full initial condition

    V = solver.V  # create full function space
    V_coords = (
        V.sub(0).collapse()[0].tabulate_dof_coordinates()
    )  # collapse to height function space

    # these are dummy stations that allow for the whole state to be saved for adjoints
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

    # Apply observation operator to all observed states at once
    y_n = observed_states @ H.T  # shape: (n_obs, obs_dim)
    y_n = y_n - hb  # convert to wse eta = H - hb

    # Add Gaussian noise
    noise = obs_std * np.random.randn(*y_n.shape)

    y_obs = np.maximum(
        y_n + noise, -hb
    )  # Apply wetting drying bottom boundary condition

    return y_obs


def setup_observation_indices(window_size, obs_frequency, total_steps):
    """Setup observation indices for windows"""
    obs_indices_per_window = np.arange(0, window_size, obs_frequency)
    obs_indices = np.arange(0, total_steps, obs_frequency)
    return obs_indices_per_window, obs_indices


def print_analysis_summary(analysis):
    """Print a formatted summary of the cross-validation analysis."""
    print("Parameter cross validation Analysis:")
    print("=" * 50)
    print(f"Total experiments: {analysis['total_experiments']}")
    print(f"Successful: {len(analysis['successful'])} ({analysis['success_rate']:.1%})")
    print(
        f"Convergence failures: {len(analysis['convergence_failures'])} ({analysis['convergence_failure_rate']:.1%})"
    )
    print(
        f"Other failures: {len(analysis['other_failures'])} ({analysis['other_failure_rate']:.1%})"
    )


def analyze_parameter_crossval_results(results_dict, print_summary=True):
    """
    Alternative implementation using defaultdict for cleaner categorization.
    """

    def get_result_category(result):
        """Determine the category of a result."""
        if "error" not in result:
            return "successful"
        elif result.get("error_type") == "convergence_failure":
            return "convergence_failures"
        else:
            return "other_failures"

    def extract_result_data(result, category):
        """Extract relevant data based on category."""
        base_params = (result["obs_std"], result["inflation_factor"])
        if category == "other_failures":
            return base_params + (result["error"],)
        return base_params

    # Group results by category
    categorized_results = defaultdict(list)

    for result in results_dict.values():
        category = get_result_category(result)
        data = extract_result_data(result, category)
        categorized_results[category].append(data)

    total_experiments = len(results_dict)

    analysis = {
        "total_experiments": total_experiments,
        "successful": categorized_results["successful"],
        "convergence_failures": categorized_results["convergence_failures"],
        "other_failures": categorized_results["other_failures"],
        "success_rate": len(categorized_results["successful"]) / total_experiments,
        "convergence_failure_rate": len(categorized_results["convergence_failures"])
        / total_experiments,
        "other_failure_rate": len(categorized_results["other_failures"])
        / total_experiments,
    }

    if print_summary:
        print_analysis_summary(analysis)

    return analysis


def load_and_analyze_crossval(summary_pickle_path):
    """
    Load parameter cross validation results from pickle file and analyze them.

    Parameters:
    -----------
    summary_pickle_path : str
        Path to the parameter_crossval_summary.pkl file
    """
    results = load_pickle(summary_pickle_path)

    return analyze_parameter_crossval_results(results)


def analyze_error_statistics(
    obs_std_list,
    inflation_list,
    result,
    analysis_type="dci_wme",
    output_dir="da_output",
    print_results=True,
):
    """
    Calculate error statistics for different observation standard deviations and inflation factors.
    """

    def load_and_calculate(obs_std, inflation):
        """Load analysis and calculate metrics, return None on error."""
        filename = (
            f"{analysis_type}_analysis_obs_std_{obs_std}_inflation_{inflation}.pkl"
        )
        try:
            analysis = load_pickle(filename)
            rmse_val = rmse(true_signal, analysis)
            misfit_val = relative_misfit(
                true_signal, analysis, result["H"], result["obs_time_indices"]
            )
            return {"rmse": rmse_val, "misfit": misfit_val}
        except:
            return None

    # Load true signal and calculate all metrics
    true_signal = load_pickle("true_signal.pkl")
    results_dict = {}

    for obs_std in obs_std_list:
        for inflation in inflation_list:
            key = (obs_std, inflation)
            results_dict[key] = load_and_calculate(obs_std, inflation)

    # Print results if requested
    if print_results:
        print(f"Error Statistics for {analysis_type.upper()} Analysis")
        print("=" * 60)
        print(f"{'Obs Std':<10} {'Inflation':<12} {'RMSE':<15} {'Misfit':<15}")
        print("-" * 60)

        for (obs_std, inflation), metrics in results_dict.items():
            if metrics:
                print(
                    f"{obs_std:<10.3f} {inflation:<12.3f} {metrics['rmse']:<15.6f} {metrics['misfit']:<15.6f}"
                )
            else:
                print(
                    f"{obs_std:<10.3f} {inflation:<12.3f} {'ERROR/NOT FOUND':<15} {'N/A':<15}"
                )

        print("-" * 60)

    return results_dict


def create_analysis_table(
    metrics,
    num_days,
    output_file="analysis_table.tex",
    save_csv=False,
    save_pickle=False,
    method_names=None,
    float_format="{:.3f}",
    caption="Analysis Results Summary",
    label="tab:error_results",
):
    """
    Create a formatted DataFrame and LaTeX table from analysis metrics.

    Parameters
    ----------
    metrics : dict
        Dictionary containing daily and total metrics from compute_analysis_metrics()
    num_days : int
        Number of days in the analysis
    output_file : str, default='analysis_table.tex'
        Output filename for LaTeX table
    save_csv : bool, default=False
        Whether to save DataFrame as CSV
    save_pickle : bool, default=False
        Whether to save DataFrame as pickle
    method_names : dict, optional
        Custom names for methods. Default: {'bayes': '4D-Var', 'dci': 'DC-4DVar', 'dci_wme': 'DC-WME 4D-Var'}
    float_format : str, default="{:.3f}"
        Format string for floating point numbers
    caption : str, default='Analysis Results Summary'
        LaTeX table caption
    label : str, default='tab:error_results'
        LaTeX table label

    Returns
    -------
    pd.DataFrame
        Multi-index DataFrame with analysis results
    str
        LaTeX table string
    """

    # Default method names mapping
    if method_names is None:
        method_names = {
            "bayes": "4D-Var",
            "dci": "DC-4DVar",
            "dci_wme": "DC-WME 4D-Var",
        }

    # Create column multi-index
    columns = pd.MultiIndex.from_product(
        [list(method_names.values()), ["RMSE", "Data Misfit"]], names=[None, None]
    )

    # Create day index (1 through num_days)
    day_index = range(1, num_days + 1)

    # Create the data dictionary
    data = {}
    for method_key, method_display in method_names.items():
        data[(method_display, "RMSE")] = metrics["daily_rmses"][method_key]
        data[(method_display, "Data Misfit")] = metrics["daily_misfits"][method_key]

    # Create the DataFrame
    df = pd.DataFrame(data, index=day_index, columns=columns)
    df.index.name = "Day"

    # Add the total row to the DataFrame
    total_row = []
    for method_key in method_names.keys():
        total_row.extend(
            [metrics["total_rmses"][method_key], metrics["total_misfits"][method_key]]
        )

    df.loc["Total"] = total_row

    # Generate LaTeX table
    latex_output = df.to_latex(
        float_format=float_format.format,
        column_format="l|cc|cc|cc",
        multicolumn_format="c",
        escape=False,
        caption=caption,
        label=label,
        position="h!",
        bold_rows=True,
    )

    # Print and save LaTeX
    print("\n=== Customized LaTeX table ===")
    print(latex_output)

    tex_path = os.path.join("da_output", output_file)
    with open(tex_path, "w") as f:
        f.write(latex_output)
    print(f"\nLaTeX table saved to '{tex_path}'")

    # Optional saves
    if save_csv:
        csv_file = output_file.replace(".tex", ".csv")
        csv_path = os.path.join("da_output", csv_file)
        df.to_csv(csv_path)
        print(f"CSV saved to '{csv_path}'")

    if save_pickle:
        pkl_file = output_file.replace(".tex", ".pkl")
        pkl_path = os.path.join("da_output", pkl_file)

        df.to_pickle(pkl_path)
        print(f"Pickle saved to '{pkl_path}'")

    return df, latex_output


def prepare_analysis_data(
    true_signal,
    bayes_analysis,
    dci_analysis,
    dci_wme_analysis,
    saved_bathy,
    saved_true_bathy,
    num_days=7,
):
    """
    Apply bathymetry corrections and prepare plot difference triplets.

    Parameters
    ----------
    true_signal : np.ndarray
        Raw true signal data, shape (time_steps, spatial_points)
    bayes_analysis : np.ndarray
        Raw Bayesian analysis results, same shape as true_signal
    dci_analysis : np.ndarray
        Raw DCI analysis results, same shape as true_signal
    dci_wme_analysis : np.ndarray
        Raw DCI WME analysis results, same shape as true_signal
    saved_bathy : np.ndarray
        Estimated bathymetry data, same shape as signal data
    saved_true_bathy : np.ndarray
        True bathymetry data, same shape as signal data
    num_days : int, default=7
        Number of days in the analysis period

    Returns
    -------
    tuple
        (true_signal_corrected, bayes_corrected, dci_corrected, dci_wme_corrected, plot_triplets)
        where plot_triplets is a list of (bayes_diff, dci_diff, dci_wme_diff) tuples
    """

    # Apply bathymetry corrections
    true_signal = true_signal - saved_true_bathy
    bayes_analysis = bayes_analysis - saved_bathy
    dci_analysis = dci_analysis - saved_bathy
    dci_wme_analysis = dci_wme_analysis - saved_bathy

    # Calculate time step indices
    one_day = int(true_signal.shape[0] / num_days)

    # Create plot triplets for days 1 through num_days-1, plus final day
    plot_triplets = []

    # Days 1 through num_days-1
    for day in range(1, num_days):
        time_idx = day * one_day

        true = true_signal[time_idx, :]
        bayes_estimate = bayes_analysis[time_idx, :]
        dci_estimate = dci_analysis[time_idx, :]
        dci_wme_estimate = dci_wme_analysis[time_idx, :]

        bayes_diff = true - bayes_estimate
        dci_diff = true - dci_estimate
        dci_wme_diff = true - dci_wme_estimate

        plot_triplets.append((bayes_diff, dci_diff, dci_wme_diff))

    # Add the final day using (num_days * one_day - 1)
    final_idx = (num_days * one_day) - 1

    true = true_signal[final_idx, :]
    bayes_estimate = bayes_analysis[final_idx, :]
    dci_estimate = dci_analysis[final_idx, :]
    dci_wme_estimate = dci_wme_analysis[final_idx, :]

    bayes_diff = true - bayes_estimate
    dci_diff = true - dci_estimate
    dci_wme_diff = true - dci_wme_estimate

    plot_triplets.append((bayes_diff, dci_diff, dci_wme_diff))

    return true_signal, bayes_analysis, dci_analysis, dci_wme_analysis, plot_triplets


def create_plot_triplets(true_signal, analysis_signal, num_days=7):
    """
    Create plot triplets of (true, estimate, difference) for selected days.

    Parameters
    ----------
    true_signal : np.ndarray
        True signal data, shape (time_steps, spatial_points)
    analysis_signal : np.ndarray
        Analysis/estimate signal data, same shape as true_signal
    num_days : int, default=7
        Number of days in the analysis period

    Returns
    -------
    list
        List of (true, estimate, diff) tuples for days 1 through num_days-1, plus final day
    """

    one_day = int(true_signal.shape[0] / num_days)
    plot_triplets = []

    # Days 1 through num_days-1
    for day in range(1, num_days):
        time_idx = day * one_day

        true = true_signal[time_idx, :]
        estimate = analysis_signal[time_idx, :]
        diff = true - estimate

        plot_triplets.append((true, estimate, diff))

    # Add the final day using (num_days * one_day - 1)
    final_idx = (num_days * one_day) - 1

    true = true_signal[final_idx, :]
    estimate = analysis_signal[final_idx, :]
    diff = true - estimate

    plot_triplets.append((true, estimate, diff))

    return plot_triplets


def compute_daily_metrics(
    true_signal, bayes_analysis, dci_analysis, dci_wme_analysis, result, num_days=7
):
    """
    Compute daily and total RMSE and misfit metrics for multiple analysis methods.

    Parameters
    ----------
    true_signal : np.ndarray
        True signal data, shape (time_steps, spatial_points)
    bayes_analysis : np.ndarray
        Bayesian analysis results, same shape as true_signal
    dci_analysis : np.ndarray
        DCI analysis results, same shape as true_signal
    dci_wme_analysis : np.ndarray
        DCI WME analysis results, same shape as true_signal

    num_days : int, default=7
        Number of days to analyze

    Returns
    -------
    dict
        Dictionary containing:
            - 'daily_rmses': Dict with keys 'bayes', 'dci', 'dci_wme' containing daily RMSE lists
            - 'daily_misfits': Dict with keys 'bayes', 'dci', 'dci_wme' containing daily misfit lists
            - 'total_rmses': Dict with keys 'bayes', 'dci', 'dci_wme' containing total RMSE values
            - 'total_misfits': Dict with keys 'bayes', 'dci', 'dci_wme' containing total misfit values
    """
    # Calculate time step indices for each day
    one_day = int(true_signal.shape[0] / num_days)

    # Get day indices (days 1 through num_days-1, plus final day)
    day_indices = [day * one_day for day in range(1, num_days)]
    day_indices.append((num_days * one_day) - 1)  # Add final day

    # Initialize metric storage
    analyses = {
        "bayes": bayes_analysis,
        "dci": dci_analysis,
        "dci_wme": dci_wme_analysis,
    }

    daily_metrics = {method: {"rmses": [], "misfits": []} for method in analyses.keys()}

    # Compute daily metrics
    for day_idx in day_indices:
        true_day = true_signal[day_idx, :]

        for method, analysis in analyses.items():
            estimate = analysis[day_idx, :]

            # Compute metrics
            print(
                f"true_signal.shape: {true_signal.shape}, analysis.shape: {estimate.shape}"
            )
            daily_metrics[method]["rmses"].append(rmse(true_day, estimate))
            daily_metrics[method]["misfits"].append(
                relative_misfit(true_day, estimate, H=result["H"])
            )

    # Compute total metrics for entire time series
    total_metrics = {}
    for method, analysis in analyses.items():
        total_metrics[method] = {
            "rmse": rmse(true_signal, analysis),
            "misfit": relative_misfit(
                true_signal, analysis, result["H"], result["obs_time_indices"]
            ),
        }

    # Organize results as dictionaries for consistent access
    return {
        "daily_rmses": {
            "bayes": daily_metrics["bayes"]["rmses"],
            "dci": daily_metrics["dci"]["rmses"],
            "dci_wme": daily_metrics["dci_wme"]["rmses"],
        },
        "daily_misfits": {
            "bayes": daily_metrics["bayes"]["misfits"],
            "dci": daily_metrics["dci"]["misfits"],
            "dci_wme": daily_metrics["dci_wme"]["misfits"],
        },
        "total_rmses": {
            method: metrics["rmse"] for method, metrics in total_metrics.items()
        },
        "total_misfits": {
            method: metrics["misfit"] for method, metrics in total_metrics.items()
        },
    }


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

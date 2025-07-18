import numpy as np
from mpi4py import MPI
from swemnics.problems import SlopedBeachProblem, TidalProblem
from swemnics import solvers as Solvers


from mpi4py import MPI

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
        "adjoint_method": True,
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
            "mag": 2.0,
            "h_b_val": 5.3,  # Uncomment if needed
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
            "mag": 2.0,
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

    stations = V_coords[::obs_frequency, :]

    solver.time_loop(
        solver_parameters=solver_params,
        stations=stations,
        plot_every=60,
        plot_name="sloped_beach_true_signal",
        u_0=u_0,
        save_states=True,
        adjoint_method=True,
    )

    return solver


def generate_observations(true_states, H, obs_time_idx, obs_std=0.1):
    # Extract only the states at the observation indices

    if isinstance(true_states, list):
        true_states = np.array(true_states)  # Ensure true_states is a numpy array

    observed_states = true_states[obs_time_idx]  # shape: (n_obs, state_dim)

    # Apply observation operator to all observed states at once
    y_n = observed_states @ H.T  # shape: (n_obs, obs_dim)

    # Add Gaussian noise
    noise = obs_std * np.random.randn(*y_n.shape)
    y_obs = y_n + noise

    return y_obs


def setup_observation_indices(window_size, obs_frequency, total_steps):
    """Setup observation indices for windows"""
    obs_indices_per_window = np.arange(0, window_size, obs_frequency)
    obs_indices = np.arange(0, total_steps, obs_frequency)
    return obs_indices_per_window, obs_indices

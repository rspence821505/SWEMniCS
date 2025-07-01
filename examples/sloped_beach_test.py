# from swemnics.problems import SlopedBeachProblem
# from swemnics import solvers as Solvers
from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dolfinx import fem as fe
import pickle
from tqdm import tqdm

import pandas as pd
import seaborn as sns
from typing import Callable, Dict, List, Tuple, Any, Union

from plotting import plot_simulation_results, create_comparison_figure
from fourd_var_parallel import run_assimilation
from swemnics.problems import SlopedBeachProblem, TidalProblem
from swemnics import solvers as Solvers


def create_problem_solver(
    problem_params, problem_type="sloped_beach", true_signal=True
):
    """
    Create a problem and solver based on the problem type and parameters.
    """
    common_solver_kwargs = {
        "theta": 1,
        "p_degree": [1, 1],
        "verbose": False,
        "adjoint_method": True,
    }
    optional_solver_kwargs = {
        "mag": 0.11,
        "alpha": 0.00010538918781,
        "h_b": 6.0,
    }
    if problem_type == "tidal":
        tidal_kwargs = {
            "nx": problem_params["nx"],
            "ny": problem_params["ny"],
            "dt": problem_params["dt"],
            "nt": problem_params["num_steps"],
            "solution_var": problem_params["sol_var"],
            "wd": False,
            "adjoint_method": True,
            "verbose": False,
        }
        if true_signal:
            tidal_kwargs["friction_law"] = "linear"
            prob = TidalProblem(**tidal_kwargs)
        else:
            tidal_kwargs["friction_law"] = problem_params["fric_law"]
            prob = TidalProblem(**tidal_kwargs)  # Inverse crime case
            # prob = TidalProblem(**tidal_kwargs, **optional_solver_kwargs)
            solver = Solvers.SUPGImplicit(prob, **common_solver_kwargs)
    else:
        sloped_kwargs = {
            "dt": problem_params["dt"],
            "nt": problem_params["num_steps"],
            "friction_law": problem_params["fric_law"],
            "solution_var": problem_params["sol_var"],
            "wd_alpha": 0.36,
            "wd": True,
        }
        prob = SlopedBeachProblem(**sloped_kwargs)
        solver = Solvers.DGImplicit(prob, **common_solver_kwargs)
    if "t" in problem_params:
        solver.problem.t = problem_params["t"]
    return prob, solver


def get_true_signal(problem_params, problem_type, solver_params, obs_frequency=1):
    """
    Default values are sea level and 0 velocity
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    prob, solver = create_problem_solver(problem_params, problem_type)
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
        plot_name="SUPG_Tide",
        u_0=u_0,
        save_states=True,
        adjoint_method=True,
    )

    return solver, prob, stations, V_coords


def setup_observation_indices(window_size, obs_frequency, total_steps):
    """Setup observation indices for windows"""
    obs_indices_per_window = np.arange(0, window_size, obs_frequency)
    obs_indices = np.arange(0, total_steps - 1, obs_frequency)
    return obs_indices_per_window, obs_indices


def build_observation_matrix(prob, V, obs_time_freq=2):
    num_cells = len(prob.mesh.geometry.dofmap)
    all_cells = np.arange(num_cells)
    obs_space_idx = np.arange(
        0, num_cells, obs_time_freq
    )  # select every obs_time_freq-th cell for observation
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

    return H, np.array(station_coords), obs_space_idx


def generate_observations(true_states, H, obs_time_idx, obs_std=0.1):
    # Extract only the states at the observation indices
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
    obs_indices = np.arange(0, total_steps - 1, obs_frequency)
    return obs_indices_per_window, obs_indices


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

problem_params = {
    "dt": 600,
    "t": 0,
    "t_final": 7 * 24 * 60 * 60,
    "num_steps": int(np.ceil((7 * 24 * 60 * 60) / 600)),
    "num_windows": 4,
    "fric_law": "mannings",  # friction law either quadratic or linear
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
}  # ,"pc_factor_mat_solver_type":"mumps"}


assert problem_params["num_steps"] == int(
    np.ceil(problem_params["t_final"] / problem_params["dt"])
)


true_signal, prob, stations, state_coords = get_true_signal(
    problem_params, "sloped_beach", solver_params, 4
)


obs_std = 0.4
obs_time_freq = 2
obs_space_freq = 2
total_steps = int((problem_params["t_final"] / problem_params["dt"]) + 1)
problem_params["num_steps"] = int(
    np.ceil((7 * 24 * 60 * 60) / 600) / problem_params["num_windows"]
)  # Size of each assimilation window
obs_per_window = problem_params["num_steps"] // obs_time_freq

H, stations, obs_spatial_indices = build_observation_matrix(
    prob, true_signal.V, obs_space_freq
)
obs_indices_per_window, obs_time_indices = setup_observation_indices(
    problem_params["num_steps"], obs_time_freq, total_steps
)

# Create synthetic observations
y_obs = generate_observations(true_signal.saved_states, H, obs_time_indices, obs_std)

print(
    f"Total Steps: {total_steps}\n"
    f"Total Assimilation Windows: {problem_params['num_windows']}\n"
    f"Steps per Window: {problem_params['num_steps']}\n"
    f"Obs Frequency: {obs_time_freq}\n"
    f"Total Obs: {obs_per_window * problem_params['num_windows']}\n"
    f"Number Stations: {stations.shape[0]}\n"
    f"Obs per Window: {obs_per_window}\n"
)


# Generate Background,Observation, and Predicted Error Covariance Matrices
state_dim = true_signal.saved_adjoints[0].shape[0]
obs_dim = stations.shape[0]

# Observation Covariance
R = np.eye(obs_dim) * (obs_std**2)

inflation_factor = 2.0
B = inflation_factor * np.eye(state_dim)

# Predicted Covariance
L = H @ B @ H.T

# Get Inverse Covariance matrices
R_inv = np.linalg.inv(R)
B_inv = np.linalg.inv(B)
L_inv = np.linalg.inv(L)

covs = {"B_inv": B_inv, "R_inv": R_inv, "L_inv": L_inv}

hb = 5.0 / 13800 * (13800 - stations[:, 0])

print(
    f"State Dimension: {state_dim}\n"
    f"Observation Dimension: {obs_dim}\n"
    f"Background Covariance Matrix Shape B: {B.shape}\n"
    f"Observation Covariance Matrix Shape R: {R.shape}\n"
    f"Predicted Error Covariance Matrix shape L: {L.shape}\n"
    f"Observation Matrix Shape H: {H.shape}\n",
    f"y_obs shape: {y_obs.shape}\n",
    f"Stations shape: {stations.shape}\n",
)


bayes_analysis = run_assimilation(
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
    "sloped_beach",
    cost_function_type="bayes",
)

true_states = np.array(true_signal.saved_states)
Hu_true = H @ true_states.T
pred = bayes_analysis[1:, :, 0] + hb
bayes_height_misfit_rmse = np.sqrt(np.mean((Hu_true - pred.T) ** 2))
print(f"Bayes Analysis Height RMSE: {bayes_height_misfit_rmse}")

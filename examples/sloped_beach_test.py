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


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"[Rank {rank}] Reached Initial MPI Call \n\n", flush=True)


with open("simulation_data.pkl", "rb") as f:
    data = pickle.load(f)

# Extract the variables
problem_params = data["problem_params"]
solver_params = data["solver_params"]
stations = data["stations"]
y_obs = data["y_obs"]
obs_per_window = data["obs_per_window"]
obs_spatial_indices = data["obs_spatial_indices"]
obs_time_indices = data["obs_time_indices"]
H = data["H"]
covs = data["covs"]
hb = data["hb"]
saved_states = data["saved_states"]

print(f"[Rank {rank}] Data loaded successfully! \n\n", flush=True)

if comm.size > 1:
    solver_params["ksp_type"] = "preonly"
    solver_params["pc_type"] = "lu"
    solver_params["pc_factor_mat_solver_type"] = "mumps"


print(f"[Rank {rank}] Before run_assimilation\n\n", flush=True)
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
    comm=comm,
)

if rank == 0:
    print(f"[Rank {rank}] Bayes Analysis completed successfully! \n\n", flush=True)
    true_states = np.array(saved_states)
    Hu_true = H @ true_states.T
    pred = bayes_analysis[1:, :, 0] + hb
    bayes_height_misfit_rmse = np.sqrt(np.mean((Hu_true - pred.T) ** 2))
    print(
        f"[Rank {rank}] Bayes Analysis Height RMSE: {bayes_height_misfit_rmse} \n\n",
        flush=True,
    )

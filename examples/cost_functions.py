import numpy as np
from dolfinx import fem as fe
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve
from mpi4py import MPI
from tqdm import tqdm
from typing import List, Dict, Tuple, Callable, Optional, Literal
from dolfinx import fem as fe


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Cost Function Helper Functions \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


def _background_loss(z, z_b, B_inv):
    """Calculate the background loss term."""
    diff_b = z - z_b
    return 0.5 * np.dot(diff_b, np.dot(B_inv, diff_b))


def _observation_loss(Qz, y_obs, R_inv):
    """Calculate the observation loss term."""
    obs_diff = y_obs.T - Qz
    return 0.5 * np.sum(obs_diff * (R_inv @ obs_diff))


def _prediction_loss(Qz, Q_zb, L_inv):
    """Calculate the prediction loss term."""
    pred_diff = Qz - Q_zb
    return 0.5 * np.sum(pred_diff * (L_inv @ pred_diff))


def _adjoint_rhs_bayes(H, Hu, yobs, R_inv, **kwargs):
    """Compute the adjoint right-hand side for the Bayesian cost function."""
    obs_residual = Hu - yobs
    return H.T @ R_inv @ obs_residual


def _adjoint_rhs_dci(H, Hu, yobs, R_inv, L_inv, Q_zb, **kwargs):
    """Compute the adjoint right-hand side for the DCI cost function."""
    obs_residual = Hu - yobs
    pred_residual = Hu - Q_zb
    return H.T @ R_inv @ obs_residual - H.T @ L_inv @ pred_residual


def _adjoint_rhs_dci_wme(H, Hu, yobs, R_inv, L_inv, Q_zb, Q_zb_wme, Q_z_wme, **kwargs):
    """Compute the adjoint right-hand side for the DCI WME cost function."""
    num_obs, obs_var, obs_sum, L_inv_wme = initialize_wme_terms(yobs, R_inv, L_inv)
    # q_wme = wme_map(Hu, yobs, obs_var, num_obs)
    # qzb_wme = wme_map(Q_zb, yobs, obs_var, num_obs)
    gamma = (1 / np.sqrt(num_obs)) * obs_sum
    return gamma * H.T @ (Q_z_wme - L_inv_wme @ (Q_z_wme - Q_zb_wme))


def wme_map(Qz, y_obs, var, num_obs):
    """Calculate Weighted Mean Error terms."""
    # print(f"Qz shape: {Qz.shape}, y_obs shape: {y_obs.shape}")
    residual = (Qz.T - y_obs).T
    # print(f"Residual shape: {residual.shape}")
    wme = (1 / np.sqrt(num_obs)) * np.sum((Qz.T - y_obs).T / np.sqrt(var), axis=1)
    return wme


def initialize_wme_terms(y_obs, R_inv, L_inv):
    """Initialize WME-specific terms."""
    num_obs = y_obs.shape[0]
    diag = np.diag(np.linalg.inv(R_inv))
    obs_sum = np.sum(diag)
    obs_var = diag[0]
    L_inv_wme = (obs_var / num_obs) * L_inv
    return num_obs, obs_var, obs_sum, L_inv_wme


# def get_trajectory(
#     u0: np.ndarray,
#     solver_params: dict,
#     stations: np.ndarray,
#     solver,
#     initial_time: float,
# ) -> any:
#     """
#     Safely reset and run the forward model from a given initial condition `u0`.

#     This version clears mutable solver state to ensure reproducibility between
#     cost function and adjoint gradient computations.
#     """
#     V = solver.V
#     u_0 = fe.Function(V)

#     # Set initial condition
#     u_0.x.array[:] = u0

#     # ⛔️ Clear any stale state
#     if hasattr(solver, "saved_states"):
#         solver.saved_states.clear()

#     if hasattr(solver, "saved_adjoints"):
#         solver.saved_adjoints.clear()

#     if hasattr(solver, "vals"):
#         solver.vals.fill(0.0)

#     # Reset solver time (important for multi-window assimilation)
#     solver.problem.t = initial_time

#     # Run forward model (this must populate saved states and adjoints)
#     _, _ = solver.time_loop(
#         solver_parameters=solver_params, stations=stations, u_0=u_0, adjoint_method=True
#     )

#     return solver


def get_trajectory(u0, solver_params, stations, solver, comm=None):
    """Propagate state through model and get observations."""

    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    print(f"[Rank {rank}] Reached checkpoint D", flush=True)

    # Convert initial state vector in h space to full initial state vector in u space
    V = solver.V
    u_0 = fe.Function(V)

    u_0.x.array[:] = u0  # Set initial state vector

    # ⛔️ Clear any stale state
    if hasattr(solver, "saved_states"):
        solver.saved_states.clear()

    if hasattr(solver, "saved_adjoints"):
        solver.saved_adjoints.clear()

    # # Synchronize before starting the time loop
    # comm.Barrier()

    # Run the time loop to propagate the state through the model

    try:
        _, _ = solver.time_loop(
            solver_parameters=solver_params,
            stations=stations,
            u_0=u_0,
            save_states=True,
            adjoint_method=True,
        )
    except Exception as e:
        if rank == 0:
            print(f"Error in time loop: {e}")
        raise

    # Synchronize after time loop
    # comm.Barrier()

    return solver


def bayes_cost_function(u0, solver, init_time, **kwargs):
    """
    Vectorized cost function for standard 4D-Var using kwargs.
    Required kwargs:
        - u_b
        - y_obs
        - obs_time_indices
        - H
        - B_inv
        - R_inv
        - solver_params
        - stations
        - hb
    """

    # Get MPI communicator
    comm = kwargs.get("comm", MPI.COMM_WORLD)
    rank = comm.Get_rank()
    print(f"[Rank {rank}] Reached checkpoint E", flush=True)
    # Unpack required arguments
    u_b = kwargs["u_b"]
    y_obs = kwargs["y_obs"]
    obs_time_indices = kwargs["obs_time_idxs"]
    H = kwargs["H"]
    B_inv, R_inv, _ = kwargs["covs"].values()
    solver_params = kwargs["solver_params"]
    stations = kwargs["stations"]
    hb = kwargs["hb"]

    # Reset solver time to initial time
    solver.problem.t = init_time

    try:
        # Run model
        solver = get_trajectory(u0, solver_params, stations, solver, comm)
        states = np.array(solver.saved_states)  # shape: (steps, num_stations)
        observed_states = states[obs_time_indices]  # shape: (n_obs, state_dim)
        Qz = H @ observed_states.T  # shape: (n_obs,obs_dim)

    except Exception as e:
        if rank == 0:
            print(f"Error in bayes_cost_function get_trajectory: {e}", flush=True)
        return 1e10

    # print(f"Qz shape: {Qz.shape}, y_obs shape: {y_obs.shape}")
    # Loss terms
    # Compute Background loss term 0.5 * (u0 - u_b).T @ B_inv @ (u0 - u_b)
    J_b = _background_loss(u0, u_b, B_inv)

    # Compute Observation loss term 0.5 * (Qz - y_obs).T @ R_inv @ (Qz - y_obs)
    J_o = _observation_loss(Qz, y_obs, R_inv)

    return J_b + J_o


def dci_cost_function(u0, solver, init_time, **kwargs):
    """
    DCI cost function variant using **kwargs.
    Required kwargs:
        - u_b
        - y_obs
        - obs_time_indices
        - H
        - B_inv
        - R_inv
        - P_inv
        - Q_zb
        - solver_params
        - stations
        - hb
    """
    # Unpack required arguments
    u_b = kwargs["u_b"]
    y_obs = kwargs["y_obs"]
    obs_time_indices = kwargs["obs_time_idxs"]
    H = kwargs["H"]
    B_inv, R_inv, L_inv = kwargs["covs"].values()
    Q_zb = kwargs["Q_zb"]
    solver_params = kwargs["solver_params"]
    stations = kwargs["stations"]
    hb = kwargs["hb"]

    # Reset solver time to initial time
    solver.problem.t = init_time

    # Get model trajectory
    solver = get_trajectory(u0, solver_params, stations, solver)
    states = np.array(solver.saved_states)  # shape: (steps, num_stations)
    # print(f"States shape: {states.shape}")
    # print(f"Observation time indices: {obs_time_indices}")
    observed_states = states[obs_time_indices]  # shape: (n_obs, state_dim)
    Qz = H @ observed_states.T  # shape: (n_obs,obs_dim)

    # Loss terms

    # Compute Background loss term 0.5 * (u0 - u_b).T @ B_inv @ (u0 - u_b)
    J_b = _background_loss(u0, u_b, B_inv)

    # Compute Observation loss term 0.5 * (Qz - y_obs).T @ R_inv @ (Qz - y_obs)
    J_o = _observation_loss(Qz, y_obs, R_inv)

    # Compute Prediction loss term 0.5 * (Qz - Q_zb).T @ P_inv @ (Qz - Q_zb)
    J_p = _prediction_loss(Qz, Q_zb, L_inv)

    return J_b + J_o - J_p


def dci_wme_cost_function(u0, solver, init_time, **kwargs):
    """
    DCI WME (Weighted Mean Error) cost function variant using **kwargs.
    Required kwargs:
        - u_b
        - y_obs
        - obs_time_indices
        - H
        - B_inv
        - R_inv
        - P_inv
        - Q_zb
        - solver_params
        - stations
        - hb
    """
    # Unpack required arguments
    u_b = kwargs["u_b"]
    y_obs = kwargs["y_obs"]
    obs_time_indices = kwargs["obs_time_idxs"]
    H = kwargs["H"]
    B_inv, R_inv, L_inv = kwargs["covs"].values()
    Q_zb = kwargs["Q_zb"]
    solver_params = kwargs["solver_params"]
    stations = kwargs["stations"]
    hb = kwargs["hb"]

    # Reset solver time to initial time
    solver.problem.t = init_time

    # Run model
    solver = get_trajectory(u0, solver_params, stations, solver)
    states = np.array(solver.saved_states)  # shape: (steps, num_stations)
    observed_states = states[obs_time_indices]  # shape: (n_obs, state_dim)
    Qz = H @ observed_states.T  # shape: (n_obs,obs_dim)

    # Initialize WME terms
    num_obs, obs_var, _, L_inv_wme = initialize_wme_terms(y_obs, R_inv, L_inv)

    # Compute Background loss term 0.5 * (u0 - u_b).T @ B_inv @ (u0 - u_b)
    J_b = _background_loss(u0, u_b, B_inv)

    # print(f"Qz shape: {Qz.shape}, y_obs shape: {y_obs.shape}")

    # Observation loss (WME)
    obs_wme = wme_map(Qz, y_obs, obs_var, num_obs)
    J_o = 0.5 * np.sum(obs_wme * (R_inv @ obs_wme))

    # Prediction loss (WME)
    Qz_wme = wme_map(Qz, y_obs, obs_var, num_obs)
    Qzb_wme = wme_map(Q_zb, y_obs, obs_var, num_obs)
    pred_diff = Qz_wme - Qzb_wme
    J_p = 0.5 * np.sum(pred_diff * (L_inv_wme @ pred_diff))

    return J_b + J_o - J_p


def adjoint_rhs(
    H: np.ndarray,
    Hu: np.ndarray,
    yobs: np.ndarray,
    R_inv: np.ndarray,
    L_inv: np.ndarray,
    Q_zb: np.ndarray,
    Q_zb_wme: Optional[np.ndarray] = None,
    Q_z_wme: Optional[np.ndarray] = None,
    adjoint_type: Literal["bayes", "dci", "dci_wme"] = "bayes",
) -> np.ndarray:
    """
    Compute the right-hand side of the adjoint equation for various 4D-Var formulations.
    """
    dispatch = {
        "bayes": _adjoint_rhs_bayes,
        "dci": _adjoint_rhs_dci,
        "dci_wme": _adjoint_rhs_dci_wme,
    }

    if adjoint_type not in dispatch:
        raise ValueError(f"Unknown adjoint type: {adjoint_type}")

    return dispatch[adjoint_type](
        H=H,
        Hu=Hu,
        yobs=yobs,
        R_inv=R_inv,
        L_inv=L_inv,
        Q_zb=Q_zb,
        Q_zb_wme=Q_zb_wme,
        Q_z_wme=Q_z_wme,
    )


def print_adjoint_debug_info(
    nt: int,
    trajectories: list,
    adjoints: list,
    obs_spatial_idxs: np.ndarray,
    obs_time_idxs: np.ndarray,
    λ_shape: tuple,
    obs_data: np.ndarray,
    R_inv: np.ndarray,
    rank: int = 0,
):
    """
    Print detailed debug information about adjoint setup.
    """
    if rank == 0:
        print("\n\n" + "=" * 40 + " Adjoint Debug Info " + "=" * 40)
        print(f"Number of Time Steps: {nt + 1}")
        print(f"Trajectories Length: {len(trajectories)}")
        print(f"Adjoints Length: {len(adjoints)}")
        print(f"Observation Spatial Indices Length: {len(obs_spatial_idxs)}")
        print(f"Observation Time Indices Length: {len(obs_time_idxs)}")
        print(f"Single Trajectory Shape: {trajectories[0].shape}")
        print(f"Single Adjoint Shape: {adjoints[0].shape}")
        print(f"Lambda Shape: {λ_shape}")
        print(f"Observation Spatial Indices: {obs_spatial_idxs}")
        print(f"Observation Time Indices: {obs_time_idxs}")
        print(f"Observation Data Shape: {obs_data.shape}")
        print(f"R_inv Shape: {R_inv.shape}")
        print("=" * 100 + "\n\n")


def print_observation_debug_info(
    Hu: np.ndarray,
    yobs: np.ndarray,
    n: int,
    rank: int = 0,
):
    """
    Print debug information for a single observation time step during the adjoint solve.
    """
    if rank == 0:
        print(f"\n--- Observation Debug Info at Time Step {n} ---")
        print(f"Hu shape: {Hu.shape}", f"Hu: {Hu[::10]}")
        print(f"yobs shape: {yobs.shape}, yobs: {yobs[::10]}")
        print("----------------------------------------------\n")


def swe_adjoint(
    solver,
    H: np.ndarray,
    obs_data: np.ndarray,
    obs_spatial_idxs: np.ndarray,
    obs_time_idxs: np.ndarray,
    R_inv: np.ndarray,
    L_inv: Optional[np.ndarray] = None,
    Q_zb: Optional[np.ndarray] = None,
    adjoint_type: Literal["bayes", "dci", "dci_wme"] = "bayes",
    comm: Optional[MPI.Comm] = None,
) -> np.ndarray:
    """
    Compute the initial adjoint vector λ₀ for a 4D-Var cost functional.

    This function performs a backward-in-time solve of the adjoint equations
    using precomputed adjoint matrices and state trajectories.

    Parameters
    ----------
    solver : Solver
        A solver object with attributes:
            - `saved_adjoints`: List of adjoint matrices (numpy arrays)
            - `saved_states`: List of state vectors at each time step
            - `vals`: Full state array for all time steps
            - `V`: Dolfinx function space
    H : np.ndarray
        Observation operator matrix of shape (m, n).
    obs_data : np.ndarray
        Observation data over the time window, shape (T_obs, m).
    obs_spatial_idxs : np.ndarray
        Indices of spatial locations where observations are taken.
    obs_time_idxs : np.ndarray
        Indices of time steps corresponding to observations.
    R_inv : np.ndarray
        Inverse of observation error covariance matrix, shape (m, m).
    L_inv : Optional[np.ndarray], optional
        Inverse of prior/predictability covariance (used in DCI variants), shape (m, m).
    Q_zb : Optional[np.ndarray], optional
        Prior prediction at observation points (used in DCI variants), shape (m,).
    adjoint_type : {'bayes', 'dci', 'dci_wme'}, default='bayes'
        Type of adjoint calculation to perform.

    Returns
    -------
    np.ndarray
        The initial adjoint vector λ₀, shape (n,).

    Raises
    ------
    ValueError
        If the adjoint matrix at a time step is singular and cannot be pseudo-inverted.
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    print(f"[Rank {rank}] Reached checkpoint F", flush=True)

    adjoints = solver.saved_adjoints  # List of adjoint matrices (NumPy arrays)
    trajectories = solver.saved_states  # List of forward states

    nt = len(adjoints)
    N_dof = adjoints[0].shape[0]  # Number of spatial points
    λ = np.zeros(N_dof)

    num_obs = len(obs_data)
    Qzb = Q_zb.T if Q_zb is not None else None

    # print_adjoint_debug_info(
    #     nt=nt,
    #     trajectories=trajectories,
    #     adjoints=adjoints,
    #     obs_spatial_idxs=obs_spatial_idxs,
    #     obs_time_idxs=obs_time_idxs,
    #     λ_shape=λ.x.array.shape,
    #     obs_data=obs_data,
    #     R_inv=R_inv,
    # )
    # print(f"y_obs shape: {obs_data.shape}, obs_time_idxs: {obs_time_idxs}")

    if adjoint_type == "dci_wme":
        states = np.array(trajectories)  # shape: (steps, num_stations)
        observed_states = states[obs_time_idxs]  # shape: (n_obs, state_dim)
        Qz = H @ observed_states.T  # shape: (n_obs,obs_dim)
        num_obs, obs_var, obs_sum, L_inv_wme = initialize_wme_terms(
            obs_data, R_inv, L_inv
        )
        Qz_wme = wme_map(Qz, obs_data, obs_var, num_obs)
        Qzb_wme = wme_map(Q_zb, obs_data, obs_var, num_obs)
        # print(
        #     f"Qzb_wme shape: {Qzb_wme.shape},Qz_wme shape {Qz_wme.shape},  y_obs shape : {obs_data.shape}"
        # )
    else:
        Qz = None
        Qz_wme = None
        Qzb_wme = None

    for n in reversed(range(nt)):
        if n in obs_time_idxs:
            idx = np.where(obs_time_idxs == n)[0][0]
            u = trajectories[n + 1].copy()  # u:
            Hu = H @ u
            # print(f"Hu shape: {Hu.shape}, obs_data shape: {obs_data.shape}")

            yobs = obs_data[idx].copy()
            q_zb = Qzb[idx].copy() if Q_zb is not None else None

            # print_observation_debug_info(Hu, yobs, n)
            rhs = adjoint_rhs(
                H,
                Hu,
                yobs,
                R_inv,
                L_inv,
                q_zb,
                Q_zb_wme=Qzb_wme,
                Q_z_wme=Qz_wme,
                adjoint_type=adjoint_type,
            )

            # rhs = H.T @ R_inv @ obs_residual
            λ += rhs  # λ_n = λ_n+1 + H^T R_inv (Hu - yobs)

        # Solve A_n^T λ_n = λ
        A_T = adjoints[n]  # Adjoint matrix at time step n

        # Solve the linear system: A_T @ λ_new = λ
        try:
            λ = spsolve(A_T, λ)
        except Exception as e:
            if rank == 0:
                print(f"Error solving adjoint system at time step {n}: {e}", flush=True)
            raise ValueError(
                f"Adjoint matrix at time step {n} is singular or ill-conditioned."
            )

    return λ  # This is λ_0 = ∇J(z_0)


def grad_cost_function(u0, solver, adjoint_type, **kwargs):
    """
    Gradient of the 4D-Var cost function using kwargs for flexibility.

    Required kwargs:
        - z_b
        - y_obs
        - obs_spatial_indices
        - obs_time_indices
        - H
        - B_inv
        - R_inv
        - (optional) P_inv, Q_zb depending on adjoint_type
    """

    # Get MPI communicator
    comm = kwargs.get("comm", MPI.COMM_WORLD)
    rank = comm.Get_rank()

    # Unpack required variables
    u_b = kwargs["u_b"]
    y_obs = kwargs["y_obs"]
    obs_spatial_indices = kwargs["obs_spatial_idxs"]
    obs_time_indices = kwargs["obs_time_idxs"]
    H = kwargs["H"]
    B_inv, R_inv, L_inv = kwargs["covs"].values()
    Q_zb = kwargs["Q_zb"]

    # print(f"Adjoint Time: {solver.problem.t}")  # Debugging line
    # check if saved adjoints are available

    # check of saved adjoints is empty
    if not solver.saved_adjoints:
        if rank == 0:
            raise ValueError(
                "No saved adjoints found. Ensure the model was run with adjoint_method=True."
            )
    if not solver.saved_states:
        if rank == 0:
            raise ValueError(
                "No saved states found. Ensure the model was run with adjoint_method=True."
            )

    # Sychronize before starting the adjoint solve
    # comm.Barrier()

    try:
        # Compute adjoint λ₀
        λ_0 = swe_adjoint(
            solver,
            H,
            y_obs,
            obs_spatial_indices,
            obs_time_indices,
            R_inv,
            L_inv,
            Q_zb,
            adjoint_type,
        )
    except Exception as e:
        if rank == 0:
            print(f"Error in swe_adjoint: {e}", flush=True)
        raise

    return B_inv @ (u0 - u_b) + λ_0

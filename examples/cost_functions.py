import numpy as np
from dolfinx import fem as fe
from scipy.optimize import minimize
from tqdm import tqdm
from typing import List, Dict, Tuple, Callable, Optional, Literal
from dolfinx import fem as fe


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Cost Function Helper Functions \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


def background_loss(z, z_b, B_inv):
    """Calculate the background loss term."""
    diff_b = z - z_b
    return 0.5 * np.dot(diff_b, np.dot(B_inv, diff_b))


def observation_loss(Qz, y_obs, R_inv):
    """Calculate the observation loss term."""
    obs_diff = (y_obs - Qz).T
    return 0.5 * np.sum(obs_diff * (R_inv @ obs_diff))


def prediction_loss(Qz, Q_zb, L_inv):
    """Calculate the prediction loss term."""
    pred_diff = (Qz - Q_zb).T
    return 0.5 * np.sum(pred_diff * (L_inv @ pred_diff))


def wme_map(Qz, y_obs, var, num_obs):
    """Calculate Weighted Mean Error terms."""
    wme = (1 / np.sqrt(num_obs)) * np.sum((Qz - y_obs).T / np.sqrt(var), axis=1)
    return wme


def initialize_wme_terms(y_obs, R_inv, L_inv):
    """Initialize WME-specific terms."""
    num_obs = y_obs.shape[0]
    obs_var = np.diag(np.linalg.inv(R_inv))[0]
    L_inv_wme = (obs_var / num_obs) * L_inv
    return num_obs, obs_var, L_inv_wme


def get_trajectory(u0, solver_params, stations, solver):
    """Propagate state through model and get observations."""

    # Convert initial state vector in h space to full initial state vector in u space
    V = solver.V
    u_0 = fe.Function(V)

    u_0.x.array[:] = u0  # Set initial state vector

    # Run the time loop to propagate the state through the model save adjoints and states
    (
        _,
        _,
    ) = solver.time_loop(
        solver_parameters=solver_params, stations=stations, u_0=u_0, adjoint_method=True
    )

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

    # Run model
    solver = get_trajectory(u0, solver_params, stations, solver)

    # Extract height field (h) and convert to WSE
    trajectory = solver.vals[:, :, 0].copy()
    wse = trajectory - hb

    # Evaluate QoI map (extract obs times)
    Qz = wse[obs_time_indices]

    # Loss terms

    # Compute Background loss term 0.5 * (u0 - u_b).T @ B_inv @ (u0 - u_b)
    J_b = background_loss(u0, u_b, B_inv)

    # Compute Observation loss term 0.5 * (Qz - y_obs).T @ R_inv @ (Qz - y_obs)
    J_o = observation_loss(Qz, y_obs, R_inv)

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

    # Extract height (h) and convert to water surface elevation
    trajectory = solver.vals[:, :, 0].copy()
    wse = trajectory - hb

    # Extract QoI (observation times)
    Qz = wse[obs_time_indices]

    # Loss terms

    # Compute Background loss term 0.5 * (u0 - u_b).T @ B_inv @ (u0 - u_b)
    J_b = background_loss(u0, u_b, B_inv)

    # Compute Observation loss term 0.5 * (Qz - y_obs).T @ R_inv @ (Qz - y_obs)
    J_o = observation_loss(Qz, y_obs, R_inv)

    # Compute Prediction loss term 0.5 * (Qz - Q_zb).T @ P_inv @ (Qz - Q_zb)
    J_p = prediction_loss(Qz, Q_zb, L_inv)

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

    # Initialize WME terms
    num_obs, obs_var, L_inv_wme = initialize_wme_terms(y_obs, R_inv, L_inv)

    # Background loss
    J_b = background_loss(u0, u_b, B_inv)

    # Run model trajectory
    solver = get_trajectory(u0, solver_params, stations, solver)
    trajectory = solver.vals[:, :, 0].copy()
    wse = trajectory - hb
    Qz = wse[obs_time_indices]

    # Observation loss (WME)
    obs_wme = wme_map(Qz, y_obs, obs_var, num_obs)
    J_o = 0.5 * np.sum(obs_wme * (R_inv @ obs_wme))

    # Prediction loss (WME)
    Qz_wme = wme_map(Qz, y_obs, obs_var, num_obs)
    Qzb_wme = wme_map(Q_zb, y_obs, obs_var, num_obs)
    pred_diff = Qz_wme - Qzb_wme
    J_p = 0.5 * np.sum(pred_diff * (L_inv_wme @ pred_diff))

    return J_b + J_o - J_p


def _adjoint_rhs_bayes(H, Hu, yobs, R_inv, **kwargs):
    obs_residual = Hu - yobs
    return H.T @ R_inv @ obs_residual


def _adjoint_rhs_dci(H, Hu, yobs, R_inv, L_inv, Q_zb, **kwargs):
    obs_residual = Hu - yobs
    pred_residual = Hu - Q_zb
    return H.T @ R_inv @ obs_residual - H.T @ L_inv @ pred_residual


def _adjoint_rhs_dci_wme(H, Hu, yobs, R_inv, L_inv, Q_zb, num_obs, **kwargs):
    num_obs, obs_var, L_inv_wme = initialize_wme_terms(yobs, R_inv, L_inv)
    q_wme = wme_map(Hu, yobs, obs_var, num_obs)
    qzb_wme = wme_map(Q_zb, yobs, obs_var, num_obs)
    return H.T @ R_inv @ q_wme - H.T @ L_inv_wme @ (q_wme - qzb_wme)


def adjoint_rhs(
    H: np.ndarray,
    Hu: np.ndarray,
    yobs: np.ndarray,
    R_inv: np.ndarray,
    L_inv: np.ndarray,
    Q_zb: np.ndarray,
    num_obs: int,
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
        num_obs=num_obs,
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
):
    """
    Print detailed debug information about adjoint setup.
    """
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
    z_n: np.ndarray,
    Hz_n: np.ndarray,
    yobs: np.ndarray,
    n: int,
):
    """
    Print debug information for a single observation time step during the adjoint solve.
    """
    print(f"\n--- Observation Debug Info at Time Step {n} ---")
    print(f"z_n shape: {z_n.shape}")
    print(f"Hz_n shape: {Hz_n.shape}")
    print(f"yobs shape: {yobs.shape}")
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
    adjoints = solver.saved_adjoints  # List of adjoint matrices (NumPy arrays)
    trajectories = solver.saved_states  # List of forward states
    nt = solver.vals.shape[0] - 1  # Number of time steps
    V = solver.V  # Function space
    λ = fe.Function(V)  # Adjoint function

    λ_vec = np.zeros((nt + 1, len(λ.x.array)))  # Store adjoint solution over time
    λ.x.array[:] = 0.0
    num_obs = len(obs_data)

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

    for n in reversed(range(nt)):
        rhs = np.zeros(len(λ.x.array))

        if n in obs_time_idxs:
            idx = np.where(obs_time_idxs == n)[0][0]
            u = trajectories[n + 1].copy()
            Hu = u[obs_spatial_idxs]
            yobs = obs_data[idx].copy()

            # print_observation_debug_info(u, Hu, yobs, n)

            obs_contribution = adjoint_rhs(
                H, Hu, yobs, R_inv, L_inv, Q_zb, num_obs, adjoint_type
            )
            rhs += obs_contribution

        try:
            λ_sol = np.linalg.solve(adjoints[n], rhs)
        except np.linalg.LinAlgError:
            print(f"Warning: Using pseudo-inverse for singular matrix at time step {n}")
            λ_sol = np.linalg.pinv(adjoints[n]) @ rhs

        λ.x.array[:] = λ_sol
        λ_vec[n, :] = λ.x.array.copy()

    return λ_vec[0, :]


def grad_cost_function(u0, solver, adjoint_type="bayes", **kwargs):
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

    # Unpack required variables
    u_b = kwargs["u_b"]
    y_obs = kwargs["y_obs"]
    obs_spatial_indices = kwargs["obs_spatial_idxs"]
    obs_time_indices = kwargs["obs_time_idxs"]
    H = kwargs["H"]
    B_inv, R_inv, L_inv = kwargs["covs"].values()
    Q_zb = kwargs["Q_zb"]

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

    return B_inv @ (u0 - u_b) + λ_0

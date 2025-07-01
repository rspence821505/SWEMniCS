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
    gamma = (1 / np.sqrt(num_obs)) * obs_sum
    return gamma * H.T @ (Q_z_wme - L_inv_wme @ (Q_z_wme - Q_zb_wme))


def wme_map(Qz, y_obs, var, num_obs):
    """Calculate Weighted Mean Error terms."""
    residual = (Qz.T - y_obs).T
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


def get_trajectory(u0, solver_params, stations, solver, comm=None):
    """
    MPI-safe trajectory propagation through model.

    Parameters
    ----------
    u0 : np.ndarray
        Initial state vector
    solver_params : dict
        Solver parameters
    stations : np.ndarray
        Station locations
    solver : object
        Solver object
    comm : MPI.Comm, optional
        MPI communicator for synchronization

    Returns
    -------
    solver : object
        Updated solver with trajectory data
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    # Convert initial state vector to full initial state vector
    V = solver.V
    u_0 = fe.Function(V)
    u_0.x.array[:] = u0  # Set initial state vector

    # Clear any existing saved states to ensure clean execution
    if hasattr(solver, "saved_states"):
        solver.saved_states.clear()
    if hasattr(solver, "saved_adjoints"):
        solver.saved_adjoints.clear()

    # Synchronize before time loop
    comm.Barrier()

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
    comm.Barrier()

    return solver


def bayes_cost_function(u0, solver, init_time, **kwargs):
    """
    MPI-compatible vectorized cost function for standard 4D-Var using kwargs.

    Required kwargs:
        - u_b
        - y_obs
        - obs_time_idxs
        - H
        - covs (dict with B_inv, R_inv keys)
        - solver_params
        - stations
        - hb
        - comm (optional MPI communicator)
    """
    # Get MPI communicator
    comm = kwargs.get("comm", MPI.COMM_WORLD)
    rank = comm.Get_rank()

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

    # Run model with MPI synchronization
    try:
        solver = get_trajectory(u0, solver_params, stations, solver, comm)
        states = np.array(solver.saved_states)  # shape: (steps, num_stations)
        observed_states = states[obs_time_indices]  # shape: (n_obs, state_dim)
        Qz = H @ observed_states.T  # shape: (n_obs,obs_dim)
    except Exception as e:
        if rank == 0:
            print(f"Error in bayes_cost_function: {e}")
        # Return a large cost on error
        return 1e10

    # Loss terms
    J_b = _background_loss(u0, u_b, B_inv)
    J_o = _observation_loss(Qz, y_obs, R_inv)

    return J_b + J_o


def dci_cost_function(u0, solver, init_time, **kwargs):
    """
    MPI-compatible DCI cost function variant using **kwargs.

    Required kwargs:
        - u_b
        - y_obs
        - obs_time_idxs
        - H
        - covs (dict with B_inv, R_inv, L_inv keys)
        - Q_zb
        - solver_params
        - stations
        - hb
        - comm (optional MPI communicator)
    """
    # Get MPI communicator
    comm = kwargs.get("comm", MPI.COMM_WORLD)
    rank = comm.Get_rank()

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

    # Get model trajectory with MPI synchronization
    try:
        solver = get_trajectory(u0, solver_params, stations, solver, comm)
        states = np.array(solver.saved_states)  # shape: (steps, num_stations)
        observed_states = states[obs_time_indices]  # shape: (n_obs, state_dim)
        Qz = H @ observed_states.T  # shape: (n_obs,obs_dim)
    except Exception as e:
        if rank == 0:
            print(f"Error in dci_cost_function: {e}")
        return 1e10

    # Loss terms
    J_b = _background_loss(u0, u_b, B_inv)
    J_o = _observation_loss(Qz, y_obs, R_inv)
    J_p = _prediction_loss(Qz, Q_zb, L_inv)

    return J_b + J_o - J_p


def dci_wme_cost_function(u0, solver, init_time, **kwargs):
    """
    MPI-compatible DCI WME (Weighted Mean Error) cost function variant.

    Required kwargs:
        - u_b
        - y_obs
        - obs_time_idxs
        - H
        - covs (dict with B_inv, R_inv, L_inv keys)
        - Q_zb
        - solver_params
        - stations
        - hb
        - comm (optional MPI communicator)
    """
    # Get MPI communicator
    comm = kwargs.get("comm", MPI.COMM_WORLD)
    rank = comm.Get_rank()

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

    # Run model with MPI synchronization
    try:
        solver = get_trajectory(u0, solver_params, stations, solver, comm)
        states = np.array(solver.saved_states)  # shape: (steps, num_stations)
        observed_states = states[obs_time_indices]  # shape: (n_obs, state_dim)
        Qz = H @ observed_states.T  # shape: (n_obs,obs_dim)
    except Exception as e:
        if rank == 0:
            print(f"Error in dci_wme_cost_function: {e}")
        return 1e10

    # Initialize WME terms
    num_obs, obs_var, _, L_inv_wme = initialize_wme_terms(y_obs, R_inv, L_inv)

    # Loss terms
    J_b = _background_loss(u0, u_b, B_inv)

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
    Print detailed debug information about adjoint setup (only on rank 0).
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
    Print debug information for a single observation time step (only on rank 0).
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
    MPI-compatible computation of the initial adjoint vector λ₀ for a 4D-Var cost functional.

    This function performs a backward-in-time solve of the adjoint equations
    using precomputed adjoint matrices and state trajectories.

    Parameters
    ----------
    solver : Solver
        A solver object with saved_adjoints and saved_states
    H : np.ndarray
        Observation operator matrix of shape (m, n)
    obs_data : np.ndarray
        Observation data over the time window, shape (T_obs, m)
    obs_spatial_idxs : np.ndarray
        Indices of spatial locations where observations are taken
    obs_time_idxs : np.ndarray
        Indices of time steps corresponding to observations
    R_inv : np.ndarray
        Inverse of observation error covariance matrix, shape (m, m)
    L_inv : Optional[np.ndarray], optional
        Inverse of prior/predictability covariance (used in DCI variants)
    Q_zb : Optional[np.ndarray], optional
        Prior prediction at observation points (used in DCI variants)
    adjoint_type : {'bayes', 'dci', 'dci_wme'}, default='bayes'
        Type of adjoint calculation to perform
    comm : MPI.Comm, optional
        MPI communicator for synchronization

    Returns
    -------
    np.ndarray
        The initial adjoint vector λ₀, shape (n,)
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    adjoints = solver.saved_adjoints  # List of adjoint matrices (NumPy arrays)
    trajectories = solver.saved_states  # List of forward states

    nt = len(adjoints)
    N_dof = adjoints[0].shape[0]  # Number of spatial points
    λ = np.zeros(N_dof)

    num_obs = len(obs_data)
    Qzb = Q_zb.T if Q_zb is not None else None

    # Initialize WME terms if needed
    if adjoint_type == "dci_wme":
        states = np.array(trajectories)  # shape: (steps, num_stations)
        observed_states = states[obs_time_idxs]  # shape: (n_obs, state_dim)
        Qz = H @ observed_states.T  # shape: (n_obs,obs_dim)
        num_obs, obs_var, obs_sum, L_inv_wme = initialize_wme_terms(
            obs_data, R_inv, L_inv
        )
        Qz_wme = wme_map(Qz, obs_data, obs_var, num_obs)
        Qzb_wme = wme_map(Q_zb, obs_data, obs_var, num_obs)
    else:
        Qz = None
        Qz_wme = None
        Qzb_wme = None

    # Backward adjoint solve
    for n in reversed(range(nt)):
        if n in obs_time_idxs:
            idx = np.where(obs_time_idxs == n)[0][0]
            u = trajectories[n + 1].copy()  # Get state at next time step
            Hu = H @ u
            yobs = obs_data[idx].copy()
            q_zb = Qzb[idx].copy() if Qzb is not None else None

            # Compute adjoint right-hand side
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

            λ += rhs  # Add observation contribution

        # Solve adjoint equation: A_n^T λ_new = λ
        A_T = adjoints[n]  # Adjoint matrix at time step n

        try:
            λ = spsolve(A_T, λ)
        except Exception as e:
            if rank == 0:
                print(f"Error solving adjoint system at step {n}: {e}")
            # Use a more robust solver if spsolve fails
            from scipy.sparse.linalg import lsqr

            λ, _ = lsqr(A_T, λ)[:2]

    return λ  # This is λ_0 = ∇J(z_0)


def grad_cost_function(u0, solver, adjoint_type, **kwargs):
    """
    MPI-compatible gradient of the 4D-Var cost function using kwargs for flexibility.

    Required kwargs:
        - u_b
        - y_obs
        - obs_spatial_idxs
        - obs_time_idxs
        - H
        - covs (dict with B_inv, R_inv, L_inv keys)
        - Q_zb (optional, depends on adjoint_type)
        - comm (optional MPI communicator)
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

    # Check if saved adjoints are available
    if not solver.saved_adjoints:
        if rank == 0:
            print(
                "No saved adjoints found. Ensure the model was run with adjoint_method=True."
            )
        raise ValueError("No saved adjoints found.")

    if not solver.saved_states:
        if rank == 0:
            print(
                "No saved states found. Ensure the model was run with adjoint_method=True."
            )
        raise ValueError("No saved states found.")

    # Synchronize before adjoint computation
    comm.Barrier()

    # Compute adjoint λ₀
    try:
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
            comm,
        )
    except Exception as e:
        if rank == 0:
            print(f"Error in adjoint computation: {e}")
        raise

    # Compute gradient: ∇J = B^{-1}(u₀ - u_b) + λ₀
    gradient = B_inv @ (u0 - u_b) + λ_0

    return gradient

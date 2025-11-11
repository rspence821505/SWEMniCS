import numpy as np
from dolfinx import fem as fe
from scipy.sparse.linalg import spsolve
from mpi4py import MPI
from scipy import linalg as la
from typing import Optional, Literal, Any, Dict, Union, List
from dolfinx import fem as fe
import sys


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Cost Function Loss Functions \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


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


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ WME Functions \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
def wme_map(Qz, y_obs, R_inv, num_obs):
    """Calculate Weighted Mean Error terms."""
    R_inv_sqrt = la.sqrtm(R_inv)
    return (1 / np.sqrt(num_obs)) * np.sum(R_inv_sqrt @ (Qz.T - y_obs).T, axis=1)


def initialize_wme_terms(y_obs, R_inv, L_inv):
    """Initialize WME-specific terms."""
    num_obs = y_obs.shape[0]
    diag = np.diag(np.linalg.inv(R_inv))
    obs_sum = np.sum(diag)
    obs_var = diag[0]
    L_inv_wme = (obs_var / num_obs) * L_inv
    return num_obs, obs_var, obs_sum, L_inv_wme


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Helper Functions \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


def get_trajectory(
    u0: np.ndarray,
    solver_params: Dict[str, Any],
    stations: Union[List[Any], np.ndarray],
    solver: Any,
) -> Any:
    """
    Propagate state through model and get observations.

    Parameters
    ----------
    u0 : np.ndarray
        Initial state vector in h space (height/elevation space).
    solver_params : dict
        Dictionary containing solver configuration parameters for time integration.
    stations : list or np.ndarray
        Station locations or identifiers where observations will be collected.
    solver : Any
        Solver object containing the finite element function space V, time_loop method,
        and storage for saved_states and saved_adjoints.

    Returns
    -------
    Any
        The updated solver object with populated saved_states and saved_adjoints
        from the forward model integration.

    Raises
    ------
    AttributeError
        If solver object is missing required attributes (V, time_loop, etc.).
    ValueError
        If initial state vector u0 cannot be properly assigned to the function space.
    RuntimeError
        If the time integration loop fails to complete successfully.

    Notes
    -----
    This function performs the forward model integration by:
    1. Converting the initial state vector from h space to the full function space
    2. Clearing any previously saved states and adjoints
    3. Running the time integration loop with state saving enabled
    4. Returning the solver with populated trajectory data

    The function requires that the solver object has:
    - V: FEniCS function space
    - time_loop: swemnics method for time integration
    - saved_states: list for storing forward states
    - saved_adjoints: list for storing adjoint states
    """
    try:
        # Convert initial state vector in h space to full initial state vector in u space
        V = solver.V
        u_0 = fe.Function(V)
        u_0.x.array[:] = u0  # Set initial state vector
    except AttributeError as e:
        print(f"Error accessing solver function space: {e}", flush=True)
        raise AttributeError(
            "Solver object missing required attribute 'V' (function space)"
        )
    except (ValueError, IndexError) as e:
        print(f"Error setting initial state vector: {e}", flush=True)
        raise ValueError(
            f"Initial state vector u0 incompatible with function space: {e}"
        )

    try:
        # Clear any stale state
        if hasattr(solver, "saved_states"):
            solver.saved_states.clear()
        if hasattr(solver, "saved_adjoints"):
            solver.saved_adjoints.clear()
        if hasattr(solver, "saved_bathy"):
            solver.saved_bathy.clear()
        if hasattr(solver, "saved_true_bathy"):
            solver.saved_true_bathy.clear()
    except AttributeError as e:
        print(f"Warning: Could not clear solver state arrays: {e}", flush=True)

    try:
        # Run the time loop to propagate the state through the model
        solver.time_loop(
            solver_parameters=solver_params,
            stations=stations,
            u_0=u_0,
            save_state=True,
            adjoint_method=True,
            save_bathy=False,
            save_true_bathy=False,
            make_wet=True,
        )
    except AttributeError as e:
        print(f"Error: Solver missing time_loop method: {e}", flush=True)
        raise AttributeError("Solver object missing required method 'time_loop'")
    except (ValueError, RuntimeError) as e:
        print(f"Error in time loop execution: {e}", flush=True)
        raise RuntimeError(f"Time integration failed: {e}")
    except Exception as e:
        print(f"Unexpected error in time loop: {e}", flush=True)
        raise RuntimeError(f"Unexpected failure during time integration: {e}")

    return solver


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Cost Function Variants \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


def bayes_cost_function(
    u0: np.ndarray, solver: Any, init_time: Union[float, int], **kwargs: Any
) -> float:
    """
    Vectorized cost function for standard 4D-Var using kwargs.

    Parameters
    ----------
    u0 : np.ndarray
        Initial state vector for the optimization.
    solver : Any
        Solver object containing the numerical model and integration methods.
    init_time : float or int
        Initial time for the solver integration.
    **kwargs : dict
        Keyword arguments containing required parameters:

        u_b : np.ndarray
            Background state vector.
        y_obs : np.ndarray
            Observation vector.
        obs_time_idxs : np.ndarray
            Time indices where observations are available.
        H : np.ndarray
            Observation operator matrix.
        covs : dict
            Dictionary containing covariance matrices with keys for B_inv, R_inv.
        solver_params : dict
            Parameters for the solver configuration.
        stations : list or np.ndarray
            Station locations or identifiers.
        hb : list or np.ndarray
            Bathymetry at stations.

    Returns
    -------
    float
        Combined cost function value (J_b + J_o) representing the sum of
        background and observation terms. Returns 1e10 if computation fails.

    Notes
    -----
    The cost function implements the standard 4D-Var formulation:
    J(u0) = 0.5 * (u0 - u_b)^T * B_inv * (u0 - u_b) +
            0.5 * (H*M(u0) - y_obs)^T * R_inv * (H*M(u0) - y_obs)
    where M(u0) represents the model integration from initial state u0.
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

    try:
        # Run model
        solver = get_trajectory(u0, solver_params, stations, solver)
        states = np.array(solver.saved_states)  # shape: (steps, num_stations)
        observed_states = states[obs_time_indices]  # shape: (n_obs, state_dim)
        Qz = H @ observed_states.T  # shape: (n_obs, obs_dim)
        Qz = Qz - hb[:, np.newaxis]  # convert to wse eta = H - hb
    except (ValueError, IndexError, AttributeError) as e:
        print(f"Error in bayes_cost_function get_trajectory: {e}", flush=True)
        return 1e10  # Return a large value to indicate failure
    except Exception as e:
        print(f"Unexpected error in bayes_cost_function: {e}", flush=True)
        return 1e10

    # Compute loss function terms

    # Compute Background loss term 0.5 * (u0 - u_b).T @ B_inv @ (u0 - u_b)
    J_b = _background_loss(u0, u_b, B_inv)

    # Compute Observation loss term 0.5 * (Qz - y_obs).T @ R_inv @ (Qz - y_obs)
    J_o = _observation_loss(Qz, y_obs, R_inv)

    return J_b + J_o


def dci_cost_function(
    u0: np.ndarray, solver: Any, init_time: Union[float, int], **kwargs: Any
) -> float:
    """
    DCI (Data-driven Control and Inference) cost function variant using kwargs.

    Parameters
    ----------
    u0 : np.ndarray
        Initial state vector for the optimization.
    solver : Any
        Solver object containing the numerical model and integration methods.
    init_time : float or int
        Initial time for the solver integration.
    **kwargs : dict
        Keyword arguments containing required parameters:

        u_b : np.ndarray
            Background state vector.
        y_obs : np.ndarray
            Observation vector.
        obs_time_idxs : np.ndarray
            Time indices where observations are available.
        H : np.ndarray
            Observation operator matrix.
        covs : dict
            Dictionary containing covariance matrices with keys for B_inv, R_inv, L_inv.
        Q_zb : np.ndarray
            Background prediction vector.
        solver_params : dict
            Parameters for the solver configuration.
        stations : list or np.ndarray
            Station locations or identifiers.
        hb : Any
            Additional parameter (purpose depends on implementation).

    Returns
    -------
    float
        Combined DCI cost function value (J_b + J_o - J_p) representing the sum of
        background and observation terms minus the prediction term. Returns 1e10 if
        computation fails.

    Notes
    -----
    The DCI cost function implements a modified 4D-Var formulation with prediction term:
    J(u0) = 0.5 * (u0 - u_b)^T * B_inv * (u0 - u_b) +
            0.5 * (H*M(u0) - y_obs)^T * R_inv * (H*M(u0) - y_obs) -
            0.5 * (H*M(u0) - Q_zb)^T * L_inv * (H*M(u0) - Q_zb)
    where M(u0) represents the model integration from initial state u0, and the
    prediction term J_p is subtracted to incorporate prior knowledge.
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

    try:
        # Get model trajectory
        solver = get_trajectory(u0, solver_params, stations, solver)
        states = np.array(solver.saved_states)  # shape: (steps, num_stations)
        observed_states = states[obs_time_indices]  # shape: (n_obs, state_dim)
        Qz = H @ observed_states.T  # shape: (n_obs, obs_dim)
        Qz = Qz - hb[:, np.newaxis]  # convert to wse eta = H - hb
        Q_zb = Q_zb - hb[:, np.newaxis]  # convert to wse eta = H - hb
    except (ValueError, IndexError, AttributeError) as e:
        print(f"Error in dci_cost_function get_trajectory: {e}", flush=True)
        return 1e10  # Return a large value to indicate failure
    except Exception as e:
        print(f"Unexpected error in dci_cost_function: {e}", flush=True)
        return 1e10

    # Loss terms
    # Compute Background loss term 0.5 * (u0 - u_b).T @ B_inv @ (u0 - u_b)
    J_b = _background_loss(u0, u_b, B_inv)
    # Compute Observation loss term 0.5 * (Qz - y_obs).T @ R_inv @ (Qz - y_obs)
    J_o = _observation_loss(Qz, y_obs, R_inv)
    # Compute Prediction loss term 0.5 * (Qz - Q_zb).T @ L_inv @ (Qz - Q_zb)
    J_p = _prediction_loss(Qz, Q_zb, L_inv)

    return J_b + J_o - J_p


def dci_wme_cost_function(
    u0: np.ndarray, solver: Any, init_time: Union[float, int], **kwargs: Any
) -> float:
    """
    DCI WME (Weighted Mean Error) cost function variant using kwargs.

    Parameters
    ----------
    u0 : np.ndarray
        Initial state vector for the optimization.
    solver : Any
        Solver object containing the numerical model and integration methods.
    init_time : float or int
        Initial time for the solver integration.
    **kwargs : dict
        Keyword arguments containing required parameters:

        u_b : np.ndarray
            Background state vector.
        y_obs : np.ndarray
            Observation vector.
        obs_time_idxs : np.ndarray
            Time indices where observations are available.
        H : np.ndarray
            Observation operator matrix.
        covs : dict
            Dictionary containing covariance matrices with keys for B_inv, R_inv, L_inv.
        Q_zb : np.ndarray
            Background prediction vector.
        solver_params : dict
            Parameters for the solver configuration.
        stations : list or np.ndarray
            Station locations or identifiers.
        hb : Any
            Additional parameter (purpose depends on implementation).

    Returns
    -------
    float
        Combined DCI WME cost function value (J_b + J_o - J_p) representing the sum of
        background and weighted observation terms minus the weighted prediction term.
        Returns 1e10 if computation fails.

    Notes
    -----
    The DCI WME cost function implements a modified 4D-Var formulation with weighted
    mean error (WME) terms applied to observations and predictions:
    J(u0) = 0.5 * (u0 - u_b)^T * B_inv * (u0 - u_b) +
            0.5 * WME(H*M(u0), y_obs)^T * R_inv * WME(H*M(u0), y_obs) -
            0.5 * (WME_z - WME_zb)^T * L_inv_wme * (WME_z - WME_zb)
    where M(u0) represents the model integration from initial state u0, and WME
    transforms are applied to reduce the impact of outliers in observations and predictions.
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

    try:
        # Run model
        solver = get_trajectory(u0, solver_params, stations, solver)
        states = np.array(solver.saved_states)  # shape: (steps, num_stations)
        observed_states = states[obs_time_indices]  # shape: (n_obs, state_dim)
        Qz = H @ observed_states.T  # shape: (n_obs, obs_dim)
        Qz = Qz - hb[:, np.newaxis]  # convert to wse eta = H - hb
        Q_zb = Q_zb - hb[:, np.newaxis]  # convert to wse eta = H - hb
    except (ValueError, IndexError, AttributeError) as e:
        print(f"Error in dci_wme_cost_function get_trajectory: {e}", flush=True)
        return 1e10  # Return a large value to indicate failure
    except Exception as e:
        print(f"Unexpected error in dci_wme_cost_function: {e}", flush=True)
        return 1e10

    try:
        # Initialize WME terms
        num_obs, obs_var, obs_sum, L_inv_wme = initialize_wme_terms(y_obs, R_inv, L_inv)

        # Compute Background loss term 0.5 * (u0 - u_b).T @ B_inv @ (u0 - u_b)
        J_b = _background_loss(u0, u_b, B_inv)

        # Observation loss (WME)
        obs_wme = wme_map(Qz, y_obs, R_inv, num_obs)
        J_o = 0.5 * np.sum(obs_wme * obs_wme)

        # Prediction loss (WME)
        Qz_wme = wme_map(Qz, y_obs, R_inv, num_obs)
        Qzb_wme = wme_map(Q_zb, y_obs, R_inv, num_obs)
        pred_diff = Qz_wme - Qzb_wme
        J_p = 0.5 * np.sum(pred_diff * (L_inv_wme @ pred_diff))

    except (ValueError, IndexError, AttributeError) as e:
        print(f"Error in dci_wme_cost_function WME computation: {e}", flush=True)
        return 1e10
    except Exception as e:
        print(
            f"Unexpected error in dci_wme_cost_function WME computation: {e}",
            flush=True,
        )
        return 1e10

    return J_b + J_o - J_p


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Adjoint Function Functions \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


def _adjoint_rhs_bayes(H, Hu, yobs, R_inv):
    """Compute the adjoint right-hand side for the Bayesian cost function."""
    obs_residual = Hu - yobs
    return H.T @ R_inv @ obs_residual


def _adjoint_rhs_dci(H, Hu, yobs, R_inv, L_inv, Q_zb):
    """Compute the adjoint right-hand side for the DCI cost function."""
    obs_residual = Hu - yobs
    pred_residual = Hu - Q_zb
    return H.T @ R_inv @ obs_residual - H.T @ L_inv @ pred_residual


def _adjoint_rhs_dci_wme(H, yobs, R_inv, L_inv, Q_zb_wme, Q_z_wme):
    """Compute the adjoint right-hand side for the DCI WME cost function."""
    num_obs, _, _, L_inv_wme = initialize_wme_terms(yobs, R_inv, L_inv)
    R_inv_sqrt = la.sqrtm(R_inv)
    J = (1 / np.sqrt(num_obs)) * R_inv_sqrt @ H
    return J.T @ (Q_z_wme - L_inv_wme @ (Q_z_wme - Q_zb_wme))


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

    if adjoint_type == "bayes":
        return dispatch[adjoint_type](
            H=H,
            Hu=Hu,
            yobs=yobs,
            R_inv=R_inv,
        )

    elif adjoint_type == "dci":
        return dispatch[adjoint_type](
            H=H,
            Hu=Hu,
            yobs=yobs,
            R_inv=R_inv,
            L_inv=L_inv,
            Q_zb=Q_zb,
        )
    elif adjoint_type == "dci_wme":
        return dispatch[adjoint_type](
            H=H,
            yobs=yobs,
            R_inv=R_inv,
            L_inv=L_inv,
            Q_zb_wme=Q_zb_wme,
            Q_z_wme=Q_z_wme,
        )

    else:
        raise ValueError(f"Unknown adjoint type: {adjoint_type}")


def swe_adjoint(
    solver,
    H: np.ndarray,
    obs_data: np.ndarray,
    obs_time_idxs: np.ndarray,
    R_inv: np.ndarray,
    L_inv: Optional[np.ndarray] = None,
    Q_zb: Optional[np.ndarray] = None,
    hb: Optional[np.ndarray] = None,
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

    adjoints = solver.saved_adjoints  # List of adjoint matrices (NumPy arrays)
    trajectories = solver.saved_states  # List of forward states

    nt = len(adjoints)
    N_dof = adjoints[0].shape[0]  # Number of spatial points
    λ = np.zeros(N_dof)

    num_obs = len(obs_data)
    # print(f"Q_zb shape: {Q_zb.shape}, hb shape: {hb.shape}", flush=True)
    Qzb = (Q_zb - hb[:, np.newaxis]) if Q_zb is not None else None
    Qzb = Q_zb.T if Q_zb is not None else None

    if adjoint_type == "dci_wme":
        states = np.array(trajectories)  # shape: (steps, num_stations)
        observed_states = states[obs_time_idxs]  # shape: (n_obs, state_dim)
        Qz = H @ observed_states.T  # shape: (n_obs,obs_dim)
        Qz = Qz - hb[:, np.newaxis]  # convert to wse eta = H - hb
        num_obs = obs_data.shape[0]
        Qz_wme = wme_map(Qz, obs_data, R_inv, num_obs)
        Qzb_wme = wme_map(Qzb.T, obs_data, R_inv, num_obs)
    else:
        Qz = None
        Qz_wme = None
        Qzb_wme = None

    for n in reversed(range(nt)):
        if n in obs_time_idxs:
            idx = np.where(obs_time_idxs == n)[0][0]
            u = trajectories[n + 1].copy()  # u:
            Hu = H @ u

            yobs = obs_data[idx].copy()
            q_zb = Qzb[idx].copy() if Qzb is not None else None
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


def grad_cost_function(
    u0: np.ndarray, solver: Any, adjoint_type: str, **kwargs: Any
) -> np.ndarray:
    """
    Gradient of the 4D-Var cost function using kwargs for flexibility.

    Parameters
    ----------
    u0 : np.ndarray
        Initial state vector for the optimization.
    solver : Any
        Solver object containing the numerical model, saved states, and saved adjoints.
    adjoint_type : str
        Type of adjoint computation to perform (e.g., 'standard', 'dci', 'wme').
    **kwargs : dict
        Keyword arguments containing required parameters:

        u_b : np.ndarray
            Background state vector.
        y_obs : np.ndarray
            Observation vector.
        obs_spatial_idxs : np.ndarray
            Spatial indices where observations are available.
        obs_time_idxs : np.ndarray
            Time indices where observations are available.
        H : np.ndarray
            Observation operator matrix.
        covs : dict
            Dictionary containing covariance matrices with keys for B_inv, R_inv, L_inv.
        Q_zb : np.ndarray
            Background prediction vector (optional, depending on adjoint_type).
        comm : MPI.Comm, optional
            MPI communicator for parallel processing. Defaults to MPI.COMM_WORLD.

    Returns
    -------
    np.ndarray
        Gradient vector of the cost function with respect to the initial state u0.

    Raises
    ------
    ValueError
        If no saved adjoints or saved states are found in the solver object.

    Notes
    -----
    The gradient computation follows the adjoint method for 4D-Var:
    ∇J(u0) = B_inv * (u0 - u_b) + λ₀
    where λ₀ is computed by solving the adjoint equations backward in time
    using the saved forward states and observation misfits.

    The function requires that the forward model has been run with saved_states=True
    and adjoint_method=True to populate solver.saved_states and solver.saved_adjoints.
    """
    # Get MPI communicator
    comm: MPI.Comm = kwargs.get("comm", MPI.COMM_WORLD)
    rank: int = comm.Get_rank()

    # Unpack required variables
    u_b = kwargs["u_b"]
    y_obs = kwargs["y_obs"]
    obs_time_indices = kwargs["obs_time_idxs"]
    H = kwargs["H"]
    B_inv, R_inv, L_inv = kwargs["covs"].values()
    Q_zb = kwargs["Q_zb"]
    hb = kwargs["hb"]

    try:
        # Check if saved adjoints are available
        if not solver.saved_adjoints:
            if rank == 0:
                raise ValueError(
                    "No saved adjoints found. Ensure the model was run with adjoint_method=True."
                )

        # Check if saved states are available
        if not solver.saved_states:
            if rank == 0:
                raise ValueError(
                    "No saved states found. Ensure the model was run with adjoint_method=True."
                )
    except AttributeError as e:
        if rank == 0:
            print(f"Error accessing solver attributes: {e}", flush=True)
        raise ValueError(
            "Solver object missing required attributes (saved_adjoints or saved_states)"
        )

    try:
        # Compute adjoint λ₀
        λ_0 = swe_adjoint(
            solver,
            H,
            y_obs,
            obs_time_indices,
            R_inv,
            L_inv,
            Q_zb,
            hb,
            adjoint_type,
        )
    except (ValueError, IndexError, AttributeError) as e:
        if rank == 0:
            print(f"Error in swe_adjoint: {e}", flush=True)
        return np.full_like(u0, 1e10)  # Return large gradient to indicate failure
    except Exception as e:
        if rank == 0:
            print(f"Unexpected error in swe_adjoint: {e}", flush=True)
        return np.full_like(u0, 1e10)

    if adjoint_type in {"bayes", "dci"}:
        return B_inv @ (u0 - u_b) + λ_0
    elif adjoint_type == "dci_wme":
        return B_inv @ (u0 - u_b) + λ_0
    else:
        raise ValueError(f"Unknown adjoint_type: {adjoint_type}")

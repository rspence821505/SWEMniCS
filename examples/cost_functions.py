import numpy as np
from dolfinx import fem as fe
from scipy.optimize import minimize
from tqdm import tqdm
from typing import List, Dict, Tuple, Callable


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Cost Function Helper Functions \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


def background_loss(z, z_b, B_inv):
    """Calculate the background loss term."""
    diff_b = z - z_b
    return 0.5 * np.dot(diff_b, np.dot(B_inv, diff_b))


def observation_loss(Qz, y_obs, R_inv):
    """Calculate the observation loss term."""
    obs_diff = (y_obs - Qz).T
    return 0.5 * np.sum(obs_diff * (R_inv @ obs_diff))


def prediction_loss(Qz, Q_zb, P_inv):
    """Calculate the prediction loss term."""
    pred_diff = (Qz - Q_zb).T
    return 0.5 * np.sum(pred_diff * (P_inv @ pred_diff))


def wme_map(Qz, y_obs, var, num_obs):
    """Calculate Weighted Mean Error terms."""
    wme = (1 / np.sqrt(num_obs)) * np.sum((Qz - y_obs).T / np.sqrt(var), axis=1)
    return wme


def initialize_wme_terms(y_obs, R_inv, P_inv):
    """Initialize WME-specific terms."""
    num_obs = y_obs.shape[0]
    obs_var = np.diag(np.linalg.inv(R_inv))[0]
    P_inv_wme = (obs_var / num_obs) * P_inv
    return num_obs, obs_var, P_inv_wme


def _setup_function_spaces(solver):
    """
    Set up common environment for all cost functions
    """
    # solver.problem.t = init_time
    V = solver.V
    h_b = solver.problem.h_b
    V_scalar = h_b._V
    wse_0 = fe.Function(V_scalar)
    u_0 = fe.Function(V)
    h_0 = u_0.sub(0)  # just water depth
    h_b = solver.problem.h_b  # bathymetry

    return u_0, h_0, h_b, wse_0, V


def get_trajectory_observations(z, obs_indices, solver_params, stations, hb, solver):
    """Propagate state through model and get observations."""

    # Convert initial state vector in h space to full initial state vector in u space
    u_0, h_0, h_b, wse_0, V = _setup_function_spaces(solver)

    # print(f"z: {z[::10]}")
    wse_0.x.array[:] = z

    h_0.interpolate(fe.Expression(wse_0 + h_b, V.sub(0).element.interpolation_points()))

    u_0.sub(0).interpolate(h_0)
    _, _, h_jacobian = solver.time_loop(
        solver_parameters=solver_params, stations=stations, u_0=u_0
    )

    trajectory = solver.vals[:, :, 0]  # get h at station locations
    wse = trajectory - hb  # convert h to wse
    wse_obs = wse[obs_indices]  # get wse at observed times

    return wse_obs, h_jacobian


def bayes_cost_function(
    z,
    z_b,
    y_obs,
    obs_indices,
    H,
    B_inv,
    R_inv,
    P_inv,
    Q_zb,
    solver_params,
    stations,
    hb,
    solver,
    init_time,
):
    """
    Vectorized cost function for standard 4D-Var with a generic model.
    """
    # Set up environment
    solver.problem.t = init_time
    # Compute background loss term
    J_b = background_loss(z, z_b, B_inv)

    # Get model trajectory observations
    Qz, _ = get_trajectory_observations(
        z,
        obs_indices,
        solver_params,
        stations,
        hb,
        solver,
    )
    # print("\n")
    # print(f"Qz: {Qz.shape}, y: {y_obs.shape}")
    # print(f"Qz: {Qz[::10]}")
    # print(f"y_obs: {y_obs[::10]}")
    # print("\n")

    # Compute observation loss term
    J_o = observation_loss(Qz, y_obs, R_inv)

    # print(f"J_b: {J_b}, J_o: {J_o}")

    return J_b + J_o


def grad_bayes_cost_function(
    z,
    z_b,
    y_obs,
    obs_indices,
    H,
    B_inv,
    R_inv,
    P_inv,
    Q_zb,
    solver_params,
    stations,
    hb,
    solver,
    init_time,
):
    """
    Computes the gradient of the Bayesian cost function with respect to z.

    Parameters are the same as bayes_cost_function.

    Returns:
        gradient: The gradient vector of the cost function with respect to z.
    """
    # Set up environment
    solver.problem.t = init_time

    # Gradient of background loss term: B_inv @ (z - z_b)
    grad_J_b = np.dot(B_inv, (z - z_b))

    Qz, M = get_trajectory_observations(
        z,
        obs_indices,
        solver_params,
        stations,
        hb,
        solver,
    )
    # print(f"Qz: {Qz.shape}, y: {y_obs.shape}")
    # Compute residual: Qz - y_obs
    residual = (y_obs - Qz).T

    # Compute predicted residual: Qz-Q_zb
    # Calculate gradient of observation term
    # sum over number of observations
    grad_J_o = np.sum(np.dot(M.T @ H.T, np.dot(R_inv, residual)), axis=1)
    # print(f"grad_J_b: {grad_J_b.shape}, grad_J_o: {grad_J_o.shape}")

    # Total gradient is the sum of the background and observation term gradients
    total_gradient = grad_J_b - grad_J_o

    return total_gradient


def dci_cost_function(
    z,
    z_b,
    y_obs,
    obs_indices,
    H,
    B_inv,
    R_inv,
    P_inv,
    Q_zb,
    solver_params,
    stations,
    hb,
    solver,
    init_time,
):
    """
    DCI cost function variant
    """

    # Compute background loss term
    J_b = background_loss(z, z_b, B_inv)

    # Get model trajectory observations
    Qz, _ = get_trajectory_observations(
        z,
        obs_indices,
        solver_params,
        stations,
        hb,
        solver,
    )

    # Compute observation and prediction loss terms
    J_o = observation_loss(Qz, y_obs, R_inv)
    J_p = prediction_loss(Qz, Q_zb, P_inv)

    # print(f"J_b: {J_b}, J_o: {J_o}, J_p: {J_p}")

    return J_b + J_o - J_p


def grad_dci_cost_function(
    z,
    z_b,
    y_obs,
    obs_indices,
    H,
    B_inv,
    R_inv,
    P_inv,
    Q_zb,
    solver_params,
    stations,
    hb,
    solver,
    init_time,
):
    """
    Computes the gradient of the Bayesian cost function with respect to z.

    Parameters are the same as bayes_cost_function.

    Returns:
        gradient: The gradient vector of the cost function with respect to z.
    """
    # Set up environment

    solver.problem.t = init_time

    # Gradient of background loss term: B_inv @ (z - z_b)
    grad_J_b = np.dot(B_inv, (z - z_b))

    Qz, M = get_trajectory_observations(
        z,
        obs_indices,
        solver_params,
        stations,
        hb,
        solver,
    )

    # Compute residual: Qz - y_obs
    obs_residual = (Qz - y_obs).T
    pred_residual = (Qz - Q_zb).T

    # Calculate gradient of observation term
    # For this implementation, we'll use the linearized observation operator H
    # grad_J_o = H^T @ R_inv @ (Qz - y_obs)
    adjoints = H @ M
    grad_J_o = np.sum(np.dot(adjoints.T, np.dot(R_inv, obs_residual)))
    grad_J_p = np.sum(np.dot(adjoints.T, np.dot(P_inv, pred_residual)))

    # print(f"grad_J_o: {grad_J_o.shape}, grad_J_b: {grad_J_b.shape}")

    # Total gradient is the sum of the background and observation term gradients
    total_gradient = grad_J_b - grad_J_o + grad_J_p

    return total_gradient


def dci_wme_cost_function(
    z,
    z_b,
    y_obs,
    obs_indices,
    H,
    B_inv,
    R_inv,
    P_inv,
    Q_zb,
    solver_params,
    stations,
    hb,
    solver,
    init_time,
):
    """
    DCI WME (Weighted Mean Error) cost function variant

    """

    # Initialize WME terms
    num_obs, obs_var, P_inv_wme = initialize_wme_terms(y_obs, R_inv, P_inv)

    # Compute background loss term
    J_b = background_loss(z, z_b, B_inv)

    # Get model trajectory observations
    Qz, _ = get_trajectory_observations(
        z,
        obs_indices,
        solver_params,
        stations,
        hb,
        solver,
    )

    # Compute observation loss with WME
    obs_wme = wme_map(Qz, y_obs, obs_var, num_obs)
    J_o = 0.5 * np.sum(obs_wme * (R_inv @ obs_wme))

    # Compute prediction loss with WME
    Qz_wme = wme_map(Qz, y_obs, obs_var, num_obs)
    Qzb_wme = wme_map(Q_zb, y_obs, obs_var, num_obs)
    pred_diff = Qz_wme - Qzb_wme
    J_p = 0.5 * np.sum(pred_diff * (P_inv_wme @ pred_diff))

    return J_b + J_o - J_p


def grad_dci_wme_cost_function(
    z,
    z_b,
    y_obs,
    obs_indices,
    H,
    B_inv,
    R_inv,
    P_inv,
    Q_zb,
    solver_params,
    stations,
    hb,
    solver,
    init_time,
):
    """
    DCI WME (Weighted Mean Error) cost function variant

    """

    # Initialize WME terms
    num_obs, obs_var, P_inv_wme = initialize_wme_terms(y_obs, R_inv, P_inv)

    # Gradient of background loss term: B_inv @ (z - z_b)
    grad_J_b = np.dot(B_inv, (z - z_b))

    # Get model trajectory observations
    Qz, M = get_trajectory_observations(
        z,
        obs_indices,
        solver_params,
        stations,
        hb,
        solver,
    )
    Qz_wme = wme_map(Qz, y_obs, obs_var, num_obs)
    Qzb_wme = wme_map(Q_zb, y_obs, obs_var, num_obs)

    obs_residual = Qz_wme
    pred_residual = Qz_wme - Qzb_wme

    adjoints = H @ M
    grad_J_o = np.sum(np.dot(adjoints.T, np.dot(R_inv, obs_residual)))
    # get standard deviation of observations
    obs_std = np.sqrt((R_inv[0, 0] ** -1))
    grad_J_p = (np.sqrt(num_obs) / obs_std) * np.sum(
        np.dot(adjoints.T, np.dot(P_inv_wme @ adjoints, pred_residual))
    )

    # Total gradient is the sum of the background and observation term gradients
    total_gradient = grad_J_b - grad_J_o + grad_J_p
    # print(f"grad_J_o: {grad_J_o.shape}, grad_J_b: {grad_J_b.shape}")

    return total_gradient

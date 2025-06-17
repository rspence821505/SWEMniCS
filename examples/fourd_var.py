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


def get_trajectory_observations(z, obs_time_indices, solver_params, stations, hb, solver):
    """Propagate state through model and get observations."""

    # Convert initial state vector in h space to full initial state vector in u space
    u_0, h_0, h_b, wse_0, V = _setup_function_spaces(solver)
    
    wse_0.x.array[:] = z

    h_0.interpolate(fe.Expression(wse_0 + h_b, V.sub(0).element.interpolation_points()))

    u_0.sub(0).interpolate(h_0)
    
    _, _, = solver.time_loop(
        solver_parameters=solver_params, stations=stations, u_0=u_0
    )

    trajectory = solver.vals[:, :, 0]  # get h at station locations
    wse = trajectory - hb  # convert h to wse
    wse_obs = wse[obs_time_indices]  # get wse at observed times

    return wse_obs, solver, V


def bayes_cost_function(
    z,
    z_b,
    y_obs,
    obs_time_indices,
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

    # Get model trajectory at observation times = H(z_k)
    Qz, solver, V = get_trajectory_observations(
        z,
        obs_time_indices,
        solver_params,
        stations,
        hb,
        solver,
    )
    # Compute background loss term = 0.5 * (z - z_b)^T B_inv (z - z_b)
    J_b = background_loss(z, z_b, B_inv)

    # Compute observation loss term = 0.5 * (y_obs - Qz_k)^T R_inv (y_obs - Qz_k)
    J_o = observation_loss(Qz, y_obs, R_inv)

    # print(f"J_b: {J_b}, J_o: {J_o}")

    return J_b + J_o


def grad_bayes_cost_function(
    z,
    z_b,
    y_obs,
    obs_spatial_indices,
    obs_time_indices,
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

    # Get model trajectory at observation times = H(z_k)
    Qz, solver, V = get_trajectory_observations(
        z,
        obs_time_indices,
        solver_params,
        stations,
        hb,
        solver,
    )
    
    # Compute Adjoint 
    λ_0 = swe_adjoint(solver, V, H, y_obs, obs_spatial_indices, obs_time_indices, R_inv)  # Rylan Todo: These inputs are just placeholders, need to be updated

    return B_inv @ (z - z_b) + λ_0


def optimize_4dvar(
    z0: np.ndarray,
    z_b: np.ndarray,
    y_obs: np.ndarray,
    obs_spatial_indices: np.ndarray,
    obs_time_indices: np.ndarray,
    H: np.ndarray,
    B_inv: np.ndarray,
    R_inv: np.ndarray,
    P_inv: np.ndarray,
    Q_zb: np.ndarray,
    stations: List,
    hb: np.ndarray,
    solver_params: Dict,
    cost_function: Callable,
    grad_cost_function: Callable,
    solver: Callable,
    init_time: Callable,
) -> Tuple[np.ndarray, dict]:
    """
    Perform 4D-Var optimization to minimize the cost function using SciPy's L-BFGS-B optimizer.

    Returns
    -------
    Tuple[np.ndarray, dict]
        The optimized state estimate and the optimization result information.
    """

    def cost_fn(z0):
        total_cost = cost_function(
            z0,
            z_b,
            y_obs,
            obs_spatial_indices
            obs_time_indices,
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
        )
        return total_cost

   
    def grad_fn(z0):
        total_gradient = grad_cost_function(
            z0,
            z_b,
            y_obs,
            obs_spatial_indices,
            obs_time_indices,
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
        ) 
        return total_gradient

    cost_function_values = []

    def callback(x):
        cost = cost_fn(x)
        cost_function_values.append(cost)
        print(f"Iteration {len(cost_function_values)}: Cost = {cost:.6f}")

    # options = {"maxfun": 10}
    # options = {"gtol": 1e-6, "ftol": 1e-12, "maxiter": 1000, "disp": True}
    # print("New Expirement")
    # options = {"maxiter": 5, "disp": True}

    result = minimize(
        cost_fn,
        z0,
        method="L-BFGS-B",
        jac=grad_fn,
        callback=callback,
        # options=options,
        tol=1e-8,
    )

    # Print optimization results
    print("\n Optimization completed:")
    print(f"  Success: {result.success}")
    print(f"  Status: {result.status}")
    print(f"  Message: {result.message}")
    print(f"  Final cost: {result.fun}")
    print(f"  Iterations: {result.nit}")
    print(f"  Function evaluations: {result.nfev}")
    print("Gradient at solution:", result.jac)
    print("Gradient norm at solution:", np.linalg.norm(result.jac))
    # print(f"  Gradient evaluations: {result.njev}")
    print("\n\n")

    print(f"  Initial state: {z0[::10]}")  # Print every 10th element for brevity
    print(
        f"  Optimized state: {result.x[::10]}"
    )  # Print every 10th element for brevity
    return result.x, result


def run_assimilation(
    problem_params,
    solver_params,
    stations,
    y_obs,
    obs_per_window,
    obs_time_indices,
    H,
    B_inv,
    R_inv,
    P_inv,
    hb,
    problem_type,
    create_problem_solver,
    cost_function,
    grad_cost_function,
):
    """
    Run 4DVar analysis with over assimilation windows
    """
    name = "Hotstart"
    analysis = []
    analysis_state = None


    for idx in tqdm(
        range(problem_params["num_windows"]), desc="Processing windows", unit="window"
    ):
        # Update initial time parameter
        initial_time = int(idx * problem_params["num_steps"] * problem_params["dt"])
        problem_params.update({"t": initial_time})

        # Create problem and solver
        _, solver = create_problem_solver(
            problem_params, problem_type, true_signal=False
        )
        solver.problem.t = initial_time

        # print(f"Solver Time 1: {solver.problem.t}")

        # Set up function spaces
        V = solver.V
        V_scalar = solver.problem.h_b._V

        # Initialize state
        u_0 = fe.Function(V)
        if analysis_state is None:
            u_0.x.array[:] = solver.u_n.x.array[:]
        else:
            # Use previous window's analysis state
            u_0.x.array[:] = analysis_state

        # Extract components
        h_0 = u_0.sub(0)
        h_b = solver.problem.h_b
        wse_0 = fe.Function(V_scalar)

        initial_u0 = u_0.copy()
        # print(f"Initial u_0 {initial_u0.x.array[:][::10]}")
        # Generate background
        print(f"Solver Time 1: {solver.problem.t}")
        solver.time_loop(
            solver_parameters=solver_params,
            stations=stations,
            plot_every=60,
            plot_name=name,
            u_0=initial_u0,
        )

        # Process background state
        background_h = solver.vals[
            :, :, 0
        ].copy()  # (steps, num_stations, huv) 0 is h index
        # print(f"Backgound shape: {background_h.shape}")
        background_wse = background_h - hb
        start = idx * obs_per_window
        end = start + obs_per_window
        obs_times_current_window = obs_time_indices[start:end]
        Q_zb = background_wse[obs_times_current_window]

        # Compute initial state
        wse_0.interpolate(
            fe.Expression(u_0.sub(0) - h_b, V.sub(0).element.interpolation_points())
        )

        # Extract observations for current window
        indices = np.arange(obs_per_window) + (idx * obs_per_window)
        yobs_current_window = y_obs[indices]
        

        # Get initial state vectors
        z0 = wse_0.x.array[:]
        z_b = wse_0.x.array[:]

        wse_0.x.array[:], solver_state_info = optimize_4dvar(
            z0,
            z_b,
            yobs_current_window,
            obs_spatial_indices_current_window,
            obs_times_current_window,
            H,
            B_inv,
            R_inv,
            P_inv,
            Q_zb,
            stations,
            hb,
            solver_params,
            cost_function,
            grad_cost_function,
            solver,
            initial_time,
        )


        # Update state with optimized values
        h_0.interpolate(
            fe.Expression(wse_0 + h_b, V.sub(0).element.interpolation_points())
        )
        u_0.sub(0).interpolate(h_0)
        # print(f"Optimized u_0 {u_0.x.array[:][::10]}")

        # Run analysis forward
        solver.problem.t = initial_time
        print(f"Solver Time 2: {solver.problem.t}")  # Debugging line
        solver.time_loop(
            solver_parameters=solver_params,
            stations=stations,
            plot_every=60,
            plot_name=name,
            u_0=u_0,
        )

        # Save analysis state for next window
        analysis_state = solver.u.x.array[:]
        
        # Collect results
        current_analysis = solver.vals.copy()
        if idx < problem_params["num_windows"] - 1:
            current_analysis = current_analysis[:-1, :, :]
        analysis.append(current_analysis)

    # Combine all windows
    return np.concatenate(analysis, axis=0)

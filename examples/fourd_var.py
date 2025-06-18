import numpy as np
from dolfinx import fem as fe
from petsc4py import PETSc
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


def get_trajectory_observations(
    z, obs_time_indices, solver_params, stations, hb, solver
):
    """Propagate state through model and get observations."""

    # Convert initial state vector in h space to full initial state vector in u space
    u_0, h_0, h_b, wse_0, V = _setup_function_spaces(solver)

    wse_0.x.array[:] = z

    h_0.interpolate(fe.Expression(wse_0 + h_b, V.sub(0).element.interpolation_points()))

    u_0.sub(0).interpolate(h_0)

    (
        _,
        _,
    ) = solver.time_loop(
        solver_parameters=solver_params, stations=stations, u_0=u_0, adjoint_method=True
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


def swe_adjoint(
    solver, H, obs_data, obs_spatial_idxs, obs_time_idxs, R_inv
) -> np.ndarray:
    """
    Solves the adjoint equation backward in time using UFL's automatic differentiation.

    Parameters:
        solver: Solver object containing the forward problem solution and adjoint forms
        H: Observation operator
        obs_time_idxs: List of observation times (indices)
        obs_data: List of np.ndarrays of observation data
        obs_spatial_idxs: DOF indices of observations
        R_inv: Inverse of observation covariance matrix
    Returns:
        grad_init: NumPy array representing ∇J(z0)
    """
    adjoints = solver.saved_adjoints  # List of adjoint forms for each time step
    trajectories = solver.saved_states  # List of states at each time step

    nt = solver.vals.shape[0] - 1  # Number of time steps
    V = solver.V  # Function space for the problem
    h_space = V.sub(0).collapse()[0]
    λ = fe.Function(h_space)
    λ_vec = np.zeros((nt + 1, len(λ.x.array)))
    λ.x.array[:] = 0.0

    print(
        f"\n\n"
        f"Number of Time Steps: {nt + 1}\n"
        f"Trajectories Length: {len(trajectories)}\n"
        f"Adjoints Length: {len(adjoints)}\n"
        f"Observation Spatial Indices Length: {len(obs_spatial_idxs)}\n"
        f"Observation Time Indices Length: {len(obs_time_idxs)}\n"
        f"Single Trajectory Shape: {trajectories[0].shape}\n"
        f"Single Adjoint Shape: {adjoints[0].shape}\n"
        f"Lambda Shape: {λ.x.array.shape}\n"
        f"Observation Spatial Indices: {obs_spatial_idxs}\n"
        f"Observation Time Indices: {obs_time_idxs}\n"
        f"Observation Data Shape: {obs_data.shape}\n"
        f"R_inv Shape: {R_inv.shape}\n"
        f"\n\n"
    )

    # possible that λ_nt = Initial Misfit
    for n in reversed(range(nt)):

        # build rhs of the adjoint equation H^TRinv(HQ - y)
        # Add the observation term if this is an observation time
        if n in obs_time_idxs:
            print(f"Processing observation at time step {n}")
            idx = np.where(obs_time_idxs == n)[0][0]  # Get index of observation time
            # state at next time step: (126, 1) note 378 = 126 * 3
            z_n = trajectories[n + 1].copy()
            print(f"z_n shape: {z_n.shape}")
            Hz_n = z_n[obs_spatial_idxs]
            print(f"Hz_n shape: {Hz_n.shape}")
            yobs = obs_data[idx, :].copy()
            print(f"yobs shape: {yobs.shape}")
            residual = Hz_n - yobs  # HQ - y
            print(f"Residual shape: {residual.shape}")

            # Create a function to represent the observation term
            obs_func = fe.Function(h_space)
            obs_func.x.array[:] = 0.0
            temp = obs_func.x.array[obs_spatial_idxs]
            obs_func.x.array[:] = H.T @ R_inv @ residual

            # Add contribution to the adjoint right-hand side
            λ.x.array[:] += obs_func.x.array[:]

        # Solve the adjoint equation
        A, b = fe.petsc.assemble_matrix(adjoints[n], bcs=[])
        A.assemble()
        b.assemble()

        # Create solution vector
        λ_sol = fe.Function(h_space)

        # Set up PETSc linear solver
        solver_petsc = PETSc.KSP().create()
        solver_petsc.setOperators(A)
        solver_petsc.setType(PETSc.KSP.Type.PREONLY)  # Direct solver
        pc = solver_petsc.getPC()
        pc.setType(PETSc.PC.Type.LU)

        # Solve the system
        solver_petsc.solve(b, λ_sol.vector)
        λ_sol.x.scatter_forward()

        λ.x.array[:] = λ_sol.x.array[:]
        λ_vec[n, :] = λ.x.array.copy()

    return λ_vec[0, :]


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
    # solver.problem.t = init_time

    # # Get model trajectory at observation times = H(z_k)
    # Qz, solver, V = get_trajectory_observations(
    #     z,
    #     obs_time_indices,
    #     solver_params,
    #     stations,
    #     hb,
    #     solver,
    # )
    _, _, _, _, V = _setup_function_spaces(solver)

    # Compute Adjoint
    λ_0 = swe_adjoint(
        solver, H, y_obs, obs_spatial_indices, obs_time_indices, R_inv
    )  # Rylan Todo: These inputs are just placeholders, need to be updated

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
    obs_spatial_indices,
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

        # Observe the current window's observation time indices
        start = idx * obs_per_window
        end = start + obs_per_window
        obs_times_current_window = obs_time_indices[start:end]

        # Extract observations for current window
        indices = np.arange(obs_per_window) + (idx * obs_per_window)
        yobs_current_window = y_obs[indices]

        # Update initial time for model
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

        # Generate background z_b
        print(f"Solver Time 1: {solver.problem.t}")
        solver.time_loop(
            solver_parameters=solver_params,
            stations=stations,
            plot_every=60,
            plot_name=name,
            u_0=initial_u0,
            adjoint_method=False,
        )

        # Process background state
        background_h = solver.vals[
            :, :, 0
        ].copy()  # (steps, num_stations, huv) 0 is h index
        # print(f"Backgound shape: {background_h.shape}")
        background_wse = background_h - hb

        # Create background QoI map
        Q_zb = background_wse[obs_times_current_window]

        # Compute initial state
        wse_0.interpolate(
            fe.Expression(u_0.sub(0) - h_b, V.sub(0).element.interpolation_points())
        )

        # Get initial state vectors
        z0 = wse_0.x.array[:]
        z_b = wse_0.x.array[:]

        wse_0.x.array[:], solver_state_info = optimize_4dvar(
            z0,
            z_b,
            yobs_current_window,
            obs_spatial_indices,
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

        # Run analysis forward
        solver.problem.t = initial_time
        print(f"Solver Time 2: {solver.problem.t}")  # Debugging line
        solver.time_loop(
            solver_parameters=solver_params,
            stations=stations,
            plot_every=60,
            plot_name=name,
            u_0=u_0,
            adjoint_method=False,
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

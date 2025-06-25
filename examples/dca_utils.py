import numpy as np
from mpi4py import MPI
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

    V = solver.V  # create full function space
    V_coords = (
        V.sub(0).collapse()[0].tabulate_dof_coordinates()
    )  # collapse to height function space
    print(f"V_coords shape: {V_coords.shape}")

    stations = V_coords[::obs_frequency, :]

    solver.time_loop(
        solver_parameters=solver_params,
        stations=stations,
        plot_every=60,
        plot_name="SUPG_Tide",
        u_0=u_0,
        adjoint_method=True,
    )

    return solver, prob, stations, V_coords


def setup_observation_indices(window_size, obs_frequency, total_steps):
    """Setup observation indices for windows"""
    obs_indices_per_window = np.arange(0, window_size, obs_frequency)
    obs_indices = np.arange(0, total_steps, obs_frequency)
    return obs_indices_per_window, obs_indices


def generate_observations(true_signal, h_b, obs_indices, obs_std=0.1):
    """
    Generate water surface elevation observations with noise.
    """

    # Get true water surface elevation
    wse = true_signal.vals[
        :, :, 0
    ]  # Observations are saved as water surface elevation h - h_b

    # Extract observations at specified indices
    y_obs = wse[obs_indices]

    # Extract observations at specified indices: Hu
    # y_obs = true_signal.vals[obs_indices, :, :]

    # Add Gaussian noise to observations Hu + noise
    y_obs += obs_std * np.random.randn(*y_obs.shape)

    return y_obs


def find_obs_indices(array1, array2):
    # Reshape arrays to enable broadcasting
    a1 = array1[:, np.newaxis, :]
    a2 = array2[np.newaxis, :, :]

    # Compare all rows of array1 with all rows of array2
    # This creates a 3D array of shape (len(array1), len(array2), 3)
    # where the last dimension is True/False for each element comparison
    comparison = np.isclose(a1, a2, rtol=1e-10, atol=1e-10)

    # Check if all elements in a row match (along the last dimension)
    # This gives us a 2D array of shape (len(array1), len(array2))
    # where True means the entire row matches
    row_matches = np.all(comparison, axis=2)

    # For each row in array1, get the indices where it appears in array2
    match_indices = [np.where(matches)[0] for matches in row_matches]

    return np.array(match_indices).flatten()


def barycentric_interpolation(triangle, values, point):
    """
    Perform linear interpolation inside a triangle using barycentric coordinates.

    Parameters:
    - triangle: List of three (x, y) vertices [(vert1[0], vert1[1]), (vert2[0], vert2[1]), (vert3[0], vert3[1])]
    - values: List of function values at the vertices [val1, val2, val3]
    - point: (x, y) coordinate inside the triangle

    Returns:
    - Interpolated value at the given point
    """
    # Extract triangle vertices
    vert1, vert2, vert3 = triangle

    val1, val2, val3 = values
    xp, yp = point

    # Compute the area of the triangle using determinant
    detT = (vert2[0] - vert1[0]) * (vert3[1] - vert1[1]) - (vert3[0] - vert1[0]) * (
        vert2[1] - vert1[1]
    )

    # Compute barycentric coordinates
    lambda1 = (
        (vert2[0] - xp) * (vert3[1] - yp) - (vert3[0] - xp) * (vert2[1] - yp)
    ) / detT
    lambda2 = (
        (vert3[0] - xp) * (vert1[1] - yp) - (vert1[0] - xp) * (vert3[1] - yp)
    ) / detT
    lambda3 = 1 - lambda1 - lambda2  # Since they must sum to 1

    # Interpolated value
    interpolated_value = lambda1 * val1 + lambda2 * val2 + lambda3 * val3
    weights = [lambda1, lambda2, lambda3]

    return interpolated_value, weights


def build_observation_matrix(prob, true_signal, stations):
    """
    Build the observation matrix H that maps from FEM solution space to station observations.

    Parameters:
    -----------
    prob : Problem
        The problem object containing the mesh
    true_signal : Signal
        Signal object with station initialization capability
    stations : array-like
        List of station coordinates

    Returns:
    --------
    H : numpy.ndarray
        Observation matrix mapping from FEM solution to station observations
    """
    # Create and get connectivity between cells and vertices
    prob.mesh.topology.create_connectivity(2, 0)
    connectivity = prob.mesh.topology.connectivity(2, 0)
    node_coordinates = prob.mesh.geometry.x

    # Initialize stations
    true_signal.init_stations(stations)
    station_cells = true_signal.cells

    # Get collapsed function space information
    V_collapsed = true_signal.V.sub(0).collapse()[0]
    dofmap = V_collapsed.dofmap

    # Create observation matrix
    H = np.zeros((len(stations), dofmap.index_map.size_local))

    for station_idx, station in enumerate(stations):
        # Get cell nodes for this station
        cell_id = station_cells[station_idx]
        node_indices = connectivity.links(cell_id)

        # Get triangle coordinates and station point
        triangle = node_coordinates[node_indices, :2]  # Drop z-coordinate
        point = station[:2]  # Drop z-coordinate

        # Calculate barycentric weights
        _, weights = barycentric_interpolation(triangle, node_indices, point)

        # Get equation numbers and populate H matrix
        cell_dofs = dofmap.cell_dofs(cell_id)
        H[station_idx, cell_dofs] = weights

    return H

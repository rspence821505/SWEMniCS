import numpy as np
from mpi4py import MPI
from swemnics.problems import SlopedBeachProblem, TidalProblem
from swemnics import solvers as Solvers


from mpi4py import MPI


def create_problem_solver(
    problem_params, problem_type="sloped_beach", true_signal=True, verbose=False
):
    """
    Create a problem and solver based on the problem type and parameters.
    """
    common_solver_kwargs = {
        "theta": 1,
        "p_degree": [1, 1],
        "verbose": verbose,
        "adjoint_method": True,
    }

    if true_signal:

        sloped_kwargs = {
            "dt": problem_params["dt"],
            "nt": problem_params["num_steps"],
            "friction_law": problem_params["fric_law"],
            "solution_var": problem_params["sol_var"],
            "verbose": verbose,
            "wd_alpha": 0.36,
            "wd": True,
            "mag": 2.0,
            "h_b_val": 5.0,  # Uncomment if needed
        }
        prob = SlopedBeachProblem(**sloped_kwargs)
        solver = Solvers.DGImplicit(prob, **common_solver_kwargs)
    else:
        sloped_kwargs = {
            "dt": problem_params["dt"],
            "nt": problem_params["num_steps"],
            "friction_law": problem_params["fric_law"],
            "solution_var": problem_params["sol_var"],
            "verbose": verbose,
            "wd_alpha": 0.36,
            "wd": True,
            "mag": 2.0,
            "h_b_val": 5.0,  # Uncomment if needed
            # "alpha": 0.00024, # Uncomment if needed
        }
        prob = SlopedBeachProblem(**sloped_kwargs)
        solver = Solvers.DGImplicit(prob, **common_solver_kwargs)

    if "t" in problem_params:
        solver.problem.t = problem_params["t"]
    return prob, solver


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


# def build_observation_matrix(prob, true_signal, stations):
#     """
#     Build the observation matrix H that maps from FEM solution space to station observations.

#     Parameters:
#     -----------
#     prob : Problem
#         The problem object containing the mesh
#     true_signal : Signal
#         Signal object with station initialization capability
#     stations : array-like
#         List of station coordinates

#     Returns:
#     --------
#     H : numpy.ndarray
#         Observation matrix mapping from FEM solution to station observations
#     """
#     # Create and get connectivity between cells and vertices
#     prob.mesh.topology.create_connectivity(2, 0)
#     connectivity = prob.mesh.topology.connectivity(2, 0)
#     node_coordinates = prob.mesh.geometry.x

#     # Initialize stations
#     solver.init_stations(stations)
#     station_cells = solver.cells

#     # Get collapsed function space information
#     V_collapsed = solver.V.sub(0).collapse()[0]
#     dofmap = V_collapsed.dofmap

#     # Create observation matrix
#     H = np.zeros((len(stations), dofmap.index_map.size_local))

#     for station_idx, station in enumerate(stations):
#         # Get cell nodes for this station
#         cell_id = station_cells[station_idx]
#         node_indices = connectivity.links(cell_id)

#         # Get triangle coordinates and station point
#         triangle = node_coordinates[node_indices, :2]  # Drop z-coordinate
#         point = station[:2]  # Drop z-coordinate

#         # Calculate barycentric weights
#         _, weights = barycentric_interpolation(triangle, node_indices, point)

#         # Get equation numbers and populate H matrix
#         cell_dofs = dofmap.cell_dofs(cell_id)
#         H[station_idx, cell_dofs] = weights

#     return H

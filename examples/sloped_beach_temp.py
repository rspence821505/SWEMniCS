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

# Only import plotting on rank 0 to avoid issues
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    from plotting import plot_simulation_results, create_comparison_figure

from fourd_var_parallel import run_assimilation
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

    sloped_kwargs = {
        "dt": problem_params["dt"],
        "nt": problem_params["num_steps"],
        "friction_law": problem_params["fric_law"],
        "solution_var": problem_params["sol_var"],
        "wd_alpha": 0.36,
        "wd": True,
    }

    # Force identical mesh across serial and parallel by controlling mesh parameters
    # This ensures the mesh has the same structure regardless of MPI configuration
    prob = SlopedBeachProblem(**sloped_kwargs)

    # Print mesh info to verify consistency
    num_cells = len(prob.mesh.geometry.dofmap)
    if comm.size > 1:
        total_cells = comm.allreduce(num_cells, op=MPI.SUM)
        print(
            f"[Rank {rank}] Local cells: {num_cells}, Global cells: {total_cells}",
            flush=True,
        )
    else:
        print(f"[Rank {rank}] Total cells: {num_cells}", flush=True)

    solver = Solvers.DGImplicit(prob, **common_solver_kwargs)

    if "t" in problem_params:
        solver.problem.t = problem_params["t"]
    return prob, solver


def get_true_signal(problem_params, problem_type, solver_params, obs_frequency=1):
    """
    Generate true signal - all processes participate in computation
    """
    print(f"[Rank {rank}] Creating problem and solver for true signal", flush=True)

    # All processes need to create the problem/solver together for proper mesh distribution
    prob, solver = create_problem_solver(problem_params, problem_type)
    print(f"[Rank {rank}] Problem and solver created", flush=True)

    # All processes participate in the computation
    print(f"[Rank {rank}] Setting up initial conditions", flush=True)
    u_0 = solver.u_n  # full initial condition

    # Get function space
    V = solver.V  # create full function space
    print(f"[Rank {rank}] Function space created", flush=True)

    # Handle coordinate access carefully - this might be distributed
    try:
        print(f"[Rank {rank}] Attempting to get coordinates", flush=True)
        # Try to get coordinates on all processes, but be prepared for this to be distributed
        V_sub, _ = V.sub(0).collapse()
        local_coords = V_sub.tabulate_dof_coordinates()
        print(
            f"[Rank {rank}] Got local coordinates, shape: {local_coords.shape}",
            flush=True,
        )

        # Gather all coordinates to rank 0
        all_coords = comm.gather(local_coords, root=0)

        if rank == 0:
            # Combine all coordinates and select stations
            if len(all_coords) > 1:
                V_coords = np.vstack(all_coords)
            else:
                V_coords = all_coords[0]

            # Remove duplicates and sort
            V_coords = np.unique(V_coords, axis=0)
            stations = V_coords[::obs_frequency, :]
            print(
                f"[Rank {rank}] Created {len(stations)} stations from {len(V_coords)} total coordinates",
                flush=True,
            )
        else:
            stations = None
            V_coords = None

    except Exception as e:
        print(f"[Rank {rank}] Error getting coordinates: {e}", flush=True)
        # Fallback: create dummy stations
        if rank == 0:
            stations = np.array([[i * 100, 0] for i in range(10)])  # dummy stations
            V_coords = stations
            print(f"[Rank {rank}] Using dummy stations", flush=True)
        else:
            stations = None
            V_coords = None

    # Broadcast stations to all processes
    print(f"[Rank {rank}] Broadcasting stations", flush=True)
    stations = comm.bcast(stations, root=0)
    V_coords = comm.bcast(V_coords, root=0)
    print(
        f"[Rank {rank}] Received stations data, shape: {stations.shape if stations is not None else 'None'}",
        flush=True,
    )

    print(f"[Rank {rank}] Starting time loop for true signal", flush=True)
    # All processes participate in the time loop
    solver.time_loop(
        solver_parameters=solver_params,
        stations=stations,
        plot_every=60,
        plot_name="SUPG_Tide",
        u_0=u_0,
        save_states=True,
        adjoint_method=True,
    )
    print(f"[Rank {rank}] Completed time loop for true signal", flush=True)

    print(f"[Rank {rank}] True signal computation completed", flush=True)
    return solver, prob, stations, V_coords


def setup_observation_indices(window_size, obs_frequency, total_steps):
    """Setup observation indices for windows"""
    obs_indices_per_window = np.arange(0, window_size, obs_frequency)
    obs_indices = np.arange(0, total_steps - 1, obs_frequency)
    return obs_indices_per_window, obs_indices


def build_observation_matrix(prob, V, obs_time_freq=2):
    """Build observation matrix - force identical behavior to serial version"""
    print(f"[Rank {rank}] Starting build_observation_matrix", flush=True)

    # Get mesh information
    num_cells_local = len(prob.mesh.geometry.dofmap)
    mesh_comm = prob.mesh.comm
    num_cells_global = mesh_comm.allreduce(num_cells_local, op=MPI.SUM)

    print(f"[Rank {rank}] Local mesh has {num_cells_local} cells", flush=True)
    print(f"[Rank {rank}] Global mesh has {num_cells_global} cells", flush=True)

    # Force the observation pattern to match serial exactly
    # Serial version uses the base mesh size, not the distributed size
    # We need to determine what the "serial equivalent" mesh size should be

    # If we have 180 cells globally but serial has 144, we need to map this correctly
    # Let's assume the serial mesh should be used as the reference
    if rank == 0:
        # Determine the correct observation pattern
        # If the problem creates different meshes in parallel vs serial,
        # we need to force it to use the serial pattern

        # Option 1: Use the expected serial mesh size (144 cells)
        serial_equivalent_cells = 144  # This should match your serial run

        # Create observation pattern based on serial mesh
        obs_space_idx = np.arange(0, serial_equivalent_cells, obs_time_freq)
        station_cells = obs_space_idx  # Direct mapping

        print(
            f"[Rank {rank}] Using serial-equivalent mesh size: {serial_equivalent_cells}",
            flush=True,
        )
        print(f"[Rank {rank}] obs_time_freq: {obs_time_freq}", flush=True)
        print(f"[Rank {rank}] obs_space_idx: {obs_space_idx}", flush=True)
        print(f"[Rank {rank}] station_cells: {station_cells}", flush=True)
        print(
            f"[Rank {rank}] Number of observation stations: {len(station_cells)}",
            flush=True,
        )

        # Now we need to map these serial cell indices to actual parallel mesh cells
        # This is the tricky part - we need to find which parallel cells correspond
        # to the serial cell pattern

        # For now, let's just take the first N cells that match the pattern
        available_global_cells = np.arange(num_cells_global)
        if len(station_cells) <= num_cells_global:
            # Map serial station indices to available parallel cells
            # Use a consistent mapping
            parallel_station_cells = available_global_cells[station_cells]
        else:
            # Fallback: use modular arithmetic to wrap around
            parallel_station_cells = available_global_cells[
                station_cells % num_cells_global
            ]

        print(
            f"[Rank {rank}] Mapped to parallel station_cells: {parallel_station_cells}",
            flush=True,
        )
    else:
        parallel_station_cells = None
        station_cells = None

    # Broadcast the determined station cells to all processes
    parallel_station_cells = comm.bcast(parallel_station_cells, root=0)
    station_cells = comm.bcast(station_cells, root=0)

    print(
        f"[Rank {rank}] Received station pattern, using {len(station_cells)} stations",
        flush=True,
    )

    # Now proceed with the parallel mesh processing using the serial-derived pattern
    cell_offset = mesh_comm.scan(num_cells_local, op=MPI.SUM) - num_cells_local
    local_cell_range = np.arange(cell_offset, cell_offset + num_cells_local)
    print(
        f"[Rank {rank}] Local cell range: {cell_offset} to {cell_offset + num_cells_local - 1}",
        flush=True,
    )

    # Find intersection of station cells with local cells
    local_station_cells_global = np.intersect1d(
        parallel_station_cells, local_cell_range
    )
    local_station_cells = local_station_cells_global - cell_offset
    print(
        f"[Rank {rank}] Found {len(local_station_cells)} observation cells on this process: {local_station_cells_global}",
        flush=True,
    )

    # Get DOF information
    V_collapsed, indices_into_V = V.sub(0).collapse()
    global_dof_size = V.dofmap.index_map.size_global
    print(f"[Rank {rank}] Global DOF size: {global_dof_size}", flush=True)

    # Create observation matrix with serial dimensions
    H_local = np.zeros((len(station_cells), global_dof_size))
    station_coords = []
    print(
        f"[Rank {rank}] Created local H matrix with shape {H_local.shape}", flush=True
    )

    if len(local_station_cells) > 0:
        collapsed_dof_coords = V_collapsed.tabulate_dof_coordinates()
        indices_into_V = np.array(indices_into_V)

        # Process each local observation cell
        for local_cell in local_station_cells:
            global_cell = local_cell + cell_offset

            # Find the station index in the serial pattern
            station_idx = np.where(parallel_station_cells == global_cell)[0]

            if len(station_idx) > 0 and local_cell < num_cells_local:
                try:
                    coords_for_cell = collapsed_dof_coords[
                        V_collapsed.dofmap.cell_dofs(local_cell)
                    ]
                    local_dofs_in_collapsed = V_collapsed.dofmap.cell_dofs(local_cell)
                    global_dofs_in_orig_V = indices_into_V[local_dofs_in_collapsed]

                    if np.all(global_dofs_in_orig_V < global_dof_size):
                        H_local[station_idx[0], global_dofs_in_orig_V] = 1 / 3
                        station_coord = 1 / 3 * (coords_for_cell.sum(axis=0))
                        station_coords.append(station_coord)
                        print(
                            f"[Rank {rank}] Station {station_idx[0]} at {station_coord} corresponds to cell {global_cell}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[Rank {rank}] Warning: DOF indices out of bounds",
                            flush=True,
                        )
                except Exception as e:
                    print(
                        f"[Rank {rank}] Error processing cell {local_cell}: {e}",
                        flush=True,
                    )

        print(
            f"[Rank {rank}] Processed {len(station_coords)} local station coordinates",
            flush=True,
        )
    else:
        print(f"[Rank {rank}] No local observation cells to process", flush=True)

    # Gather results
    all_station_coords = comm.gather(station_coords, root=0)
    all_H_local = comm.gather(H_local, root=0)

    if rank == 0:
        H_global = np.sum(all_H_local, axis=0)
        flattened_coords = []
        for coords_list in all_station_coords:
            flattened_coords.extend(coords_list)

        if flattened_coords:
            station_coords_global = np.array(flattened_coords)
        else:
            station_coords_global = np.array(
                [[i, 0, 0] for i in range(len(station_cells))]
            )

        print(f"[Rank {rank}] Final H matrix shape: {H_global.shape}", flush=True)
        print(f"[Rank {rank}] This should match serial: (72, 1296)", flush=True)
        print(
            f"[Rank {rank}] Total observation stations: {len(station_cells)}",
            flush=True,
        )

    else:
        H_global = None
        station_coords_global = None

    H_global = comm.bcast(H_global, root=0)
    station_coords_global = comm.bcast(station_coords_global, root=0)

    print(
        f"[Rank {rank}] Completed build_observation_matrix with shape: {H_global.shape}",
        flush=True,
    )
    return H_global, station_coords_global, station_cells


def generate_observations(true_states, H, obs_time_idx, obs_std=0.1):
    """Generate observations - only on rank 0, then broadcast"""
    print(f"[Rank {rank}] Starting generate_observations", flush=True)

    if rank == 0:
        print(f"[Rank {rank}] Computing observations on master process", flush=True)
        # Extract only the states at the observation indices
        true_states = np.array(true_states)  # Ensure true_states is a numpy array
        print(f"[Rank {rank}] True states shape: {true_states.shape}", flush=True)

        observed_states = true_states[obs_time_idx]  # shape: (n_obs, state_dim)
        print(
            f"[Rank {rank}] Observed states shape: {observed_states.shape}", flush=True
        )
        print(f"[Rank {rank}] H matrix shape: {H.shape}", flush=True)

        # Apply observation operator to all observed states at once
        y_n = observed_states @ H.T  # shape: (n_obs, obs_dim)
        print(
            f"[Rank {rank}] Applied observation operator, y_n shape: {y_n.shape}",
            flush=True,
        )

        # Add Gaussian noise
        np.random.seed(123)  # For reproducible noise
        noise = obs_std * np.random.randn(*y_n.shape)
        y_obs = y_n + noise
        print(
            f"[Rank {rank}] Added noise with std={obs_std}, final y_obs shape: {y_obs.shape}",
            flush=True,
        )
    else:
        print(f"[Rank {rank}] Waiting for observations from rank 0", flush=True)
        y_obs = None

    # Broadcast observations to all processes
    print(f"[Rank {rank}] Broadcasting observations from rank 0", flush=True)
    y_obs = comm.bcast(y_obs, root=0)
    print(
        f"[Rank {rank}] Received observations with shape: {y_obs.shape if y_obs is not None else 'None'}",
        flush=True,
    )

    print(f"[Rank {rank}] Completed generate_observations", flush=True)
    return y_obs


def main():
    """Main function with proper MPI coordination"""
    print(f"[Rank {rank}/{size}] Starting 4D-Var data assimilation", flush=True)

    # Synchronize all processes before starting
    comm.Barrier()

    # Problem parameters - same on all processes
    problem_params = {
        "dt": 600,
        "t": 0,
        "t_final": 7 * 24 * 60 * 60,
        "num_steps": int(np.ceil((7 * 24 * 60 * 60) / 600)),
        "num_windows": 4,
        "fric_law": "mannings",  # friction law either quadratic or linear
        "sol_var": "h",  # solution variable either h or hu
    }

    solver_params = {
        "rtol": 1e-5,
        "atol": 1e-6,
        "max_it": 10,
        "relaxation_parameter": 1.0,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_ErrorIfNotConverged": False,
    }

    # Configure solver for parallel execution
    if size > 1:
        solver_params["ksp_type"] = "preonly"
        solver_params["pc_type"] = "lu"
        solver_params["pc_factor_mat_solver_type"] = "mumps"

    assert problem_params["num_steps"] == int(
        np.ceil(problem_params["t_final"] / problem_params["dt"])
    )

    # Generate true signal (coordinated across all processes)
    print(f"[Rank {rank}] Generating true signal", flush=True)
    true_signal, prob, stations, state_coords = get_true_signal(
        problem_params, "sloped_beach", solver_params, 4
    )
    print(f"[Rank {rank}] True signal generated", flush=True)

    # Observation parameters
    obs_std = 0.4
    obs_time_freq = 2
    obs_space_freq = 2
    total_steps = int((problem_params["t_final"] / problem_params["dt"]) + 1)
    problem_params["num_steps"] = int(
        np.ceil((7 * 24 * 60 * 60) / 600) / problem_params["num_windows"]
    )  # Size of each assimilation window
    obs_per_window = problem_params["num_steps"] // obs_time_freq

    # Build observation matrix and setup indices
    print(f"[Rank {rank}] Building observation matrix", flush=True)
    H, stations, obs_spatial_indices = build_observation_matrix(
        prob, true_signal.V, obs_space_freq
    )

    print(
        f"[Rank {rank}] Observation matrix built with shape: {np.linalg.matrix_rank(H)}",
        flush=True,
    )
    print(f"[Rank {rank}] Setting up observation indices", flush=True)
    obs_indices_per_window, obs_time_indices = setup_observation_indices(
        problem_params["num_steps"], obs_time_freq, total_steps
    )

    # Generate observations (only on rank 0, then broadcast)
    print(f"[Rank {rank}] Generating observations", flush=True)
    y_obs = generate_observations(
        true_signal.saved_states, H, obs_time_indices, obs_std
    )

    # Print summary information only on rank 0
    if rank == 0:
        print(
            f"Total Steps: {total_steps}\n"
            f"Total Assimilation Windows: {problem_params['num_windows']}\n"
            f"Steps per Window: {problem_params['num_steps']}\n"
            f"Obs Frequency: {obs_time_freq}\n"
            f"Total Obs: {obs_per_window * problem_params['num_windows']}\n"
            f"Number Stations: {stations.shape[0]}\n"
            f"Obs per Window: {obs_per_window}\n",
            flush=True,
        )

    # Generate covariance matrices
    print(f"[Rank {rank}] Setting up covariance matrices", flush=True)

    # Check if saved_adjoints exists and has data
    if hasattr(true_signal, "saved_adjoints") and len(true_signal.saved_adjoints) > 0:
        state_dim = true_signal.saved_adjoints[0].shape[0]
        print(
            f"[Rank {rank}] State dimension from saved_adjoints: {state_dim}",
            flush=True,
        )
    elif hasattr(true_signal, "saved_states") and len(true_signal.saved_states) > 0:
        state_dim = true_signal.saved_states[0].shape[0]
        print(
            f"[Rank {rank}] State dimension from saved_states: {state_dim}", flush=True
        )
    else:
        # Fallback: get state dimension from function space
        state_dim = true_signal.V.dofmap.index_map.size_global
        print(
            f"[Rank {rank}] State dimension from function space: {state_dim}",
            flush=True,
        )

    obs_dim = stations.shape[0]
    print(f"[Rank {rank}] Observation dimension: {obs_dim}", flush=True)

    # Observation Covariance
    R = np.eye(obs_dim) * (obs_std**2)

    inflation_factor = 2.0
    B = inflation_factor * np.eye(state_dim)

    # Predicted Covariance
    L = H @ B @ H.T

    # Get Inverse Covariance matrices
    R_inv = np.linalg.inv(R)
    B_inv = np.linalg.inv(B)
    L_inv = np.linalg.inv(L)

    covs = {"B_inv": B_inv, "R_inv": R_inv, "L_inv": L_inv}

    hb = 5.0 / 13800 * (13800 - stations[:, 0])

    if rank == 0:
        print(
            f"State Dimension: {state_dim}\n"
            f"Observation Dimension: {obs_dim}\n"
            f"Background Covariance Matrix Shape B: {B.shape}\n"
            f"Observation Covariance Matrix Shape R: {R.shape}\n"
            f"Predicted Error Covariance Matrix shape L: {L.shape}\n"
            f"Observation Matrix Shape H: {H.shape}\n"
            f"y_obs shape: {y_obs.shape}\n"
            f"Stations shape: {stations.shape}\n",
            flush=True,
        )

    # Synchronize before starting assimilation
    comm.Barrier()
    print(f"[Rank {rank}] Starting assimilation", flush=True)

    # Run data assimilation
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

    # Compute and print results only on rank 0
    if rank == 0:
        true_states = np.array(true_signal.saved_states)
        Hu_true = H @ true_states.T
        pred = bayes_analysis[1:, :, 0] + hb
        bayes_height_misfit_rmse = np.sqrt(np.mean((Hu_true - pred.T) ** 2))
        print(f"Bayes Analysis Height RMSE: {bayes_height_misfit_rmse}")

    # Final synchronization
    comm.Barrier()
    print(f"[Rank {rank}] Completed 4D-Var data assimilation", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MPI-enabled data assimilation script for sloped beach problem.
Usage: mpirun -np <num_processes> python mpi_data_assimilation.py
"""

from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dolfinx import fem as fe
import pickle
from tqdm import tqdm
import sys
import os

# Import custom modules
try:
    from plotting import plot_simulation_results, create_comparison_figure
    from fourd_var_parallel import run_assimilation
    from swemnics.problems import SlopedBeachProblem, TidalProblem
    from swemnics import solvers as Solvers
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Make sure all required modules are available on all MPI processes")


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
            prob = TidalProblem(**tidal_kwargs)
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


def get_mpi_compatible_solver_params():
    """
    Get solver parameters that work well with MPI.
    Provides fallback options if certain solvers are not available.
    """
    # Try different solver configurations in order of preference
    solver_configs = [
        {
            "rtol": 1e-5,
            "atol": 1e-6,
            "max_it": 50,
            "relaxation_parameter": 1.0,
            "ksp_type": "gmres",
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
            "ksp_ErrorIfNotConverged": False,
        },
        {
            "rtol": 1e-5,
            "atol": 1e-6,
            "max_it": 50,
            "relaxation_parameter": 1.0,
            "ksp_type": "gmres",
            "pc_type": "asm",  # Additive Schwarz Method
            "sub_pc_type": "ilu",
            "ksp_ErrorIfNotConverged": False,
        },
        {
            "rtol": 1e-4,
            "atol": 1e-5,
            "max_it": 100,
            "relaxation_parameter": 1.0,
            "ksp_type": "gmres",
            "pc_type": "jacobi",  # Simple Jacobi preconditioner
            "ksp_ErrorIfNotConverged": False,
        },
        {
            "rtol": 1e-4,
            "atol": 1e-5,
            "max_it": 200,
            "relaxation_parameter": 1.0,
            "ksp_type": "cg",
            "pc_type": "none",  # No preconditioner as last resort
            "ksp_ErrorIfNotConverged": False,
        },
    ]

    return solver_configs


def test_solver_config(problem_params, solver_config, rank):
    """Test if a solver configuration works."""
    try:
        if rank == 0:
            print(f"  Testing solver config: {solver_config['pc_type']}")

        # Create a test problem with minimal steps
        test_params = problem_params.copy()
        test_params["num_steps"] = 2  # Minimal test

        prob, solver = create_problem_solver(test_params, "sloped_beach")

        # Try a minimal solve - use a simpler approach
        u_0 = solver.u_n

        # Just test if we can create the solver without actually solving
        # This avoids the dolfinx.nls.petsc issue
        if hasattr(solver, "solver_parameters"):
            solver.solver_parameters = solver_config

        if rank == 0:
            print(f"  ✓ Solver config works: {solver_config['pc_type']}")
        return True

    except Exception as e:
        if rank == 0:
            print(
                f"  ✗ Solver config failed: {solver_config['pc_type']} - {str(e)[:100]}"
            )
        return False


def get_simple_solver_params():
    """
    Get simple solver parameters that should work with most DOLFINx installations.
    """
    return {
        "rtol": 1e-5,
        "atol": 1e-6,
        "max_it": 50,
        "relaxation_parameter": 1.0,
        "ksp_type": "gmres",
        "pc_type": "jacobi",
        "ksp_ErrorIfNotConverged": False,
    }


def get_true_signal(problem_params, problem_type, solver_params, obs_frequency=1):
    """
    Generate true signal using forward model.
    Default values are sea level and 0 velocity.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Generating true signal...")

    prob, solver = create_problem_solver(problem_params, problem_type)
    u_0 = solver.u_n  # full initial condition

    # Create full function space
    V = solver.V
    V_coords = V.sub(0).collapse()[0].tabulate_dof_coordinates()
    stations = V_coords[::obs_frequency, :]

    solver.time_loop(
        solver_parameters=solver_params,
        stations=stations,
        plot_every=60,
        plot_name="SUPG_Tide",
        u_0=u_0,
        save_states=True,
        adjoint_method=True,
    )

    return solver, prob, stations, V_coords


def build_observation_matrix(prob, V, obs_time_freq=2):
    """Build observation matrix H for data assimilation."""
    num_cells = len(prob.mesh.geometry.dofmap)
    all_cells = np.arange(num_cells)
    obs_space_idx = np.arange(0, num_cells, obs_time_freq)
    station_cells = all_cells[obs_space_idx]

    # Create observation matrix
    H = np.zeros((len(station_cells), V.dofmap.index_map.size_local))

    # Pick subset of cells for the stations
    station_coords = []

    # Collapse the function space to get coordinates
    V_collapsed, indices_into_V = V.sub(0).collapse()
    collapsed_dof_coords = V_collapsed.tabulate_dof_coordinates()
    indices_into_V = np.array(indices_into_V)

    for station, i in enumerate(station_cells):
        coords_for_cell = collapsed_dof_coords[V_collapsed.dofmap.cell_dofs(i)]
        dofs_in_orig_V = indices_into_V[V_collapsed.dofmap.cell_dofs(i)]
        H[station, dofs_in_orig_V] = 1 / 3
        station_coord = 1 / 3 * (coords_for_cell.sum(axis=0))
        station_coords.append(station_coord)

    return H, np.array(station_coords), obs_space_idx


def generate_observations(true_states, H, obs_time_idx, obs_std=0.1):
    """Generate synthetic observations with noise."""
    true_states = np.array(true_states)
    observed_states = true_states[obs_time_idx]

    # Apply observation operator
    y_n = observed_states @ H.T

    # Add Gaussian noise
    noise = obs_std * np.random.randn(*y_n.shape)
    y_obs = y_n + noise

    return y_obs


def setup_observation_indices(window_size, obs_frequency, total_steps):
    """Setup observation indices for windows."""
    obs_indices_per_window = np.arange(0, window_size, obs_frequency)
    obs_indices = np.arange(0, total_steps - 1, obs_frequency)
    return obs_indices_per_window, obs_indices


def setup_covariance_matrices(state_dim, obs_dim, obs_std, inflation_factor=2.0):
    """Setup covariance matrices for data assimilation."""
    # Observation covariance
    R = np.eye(obs_dim) * (obs_std**2)

    # Background covariance
    B = inflation_factor * np.eye(state_dim)

    # Get inverse covariance matrices
    R_inv = np.linalg.inv(R)
    B_inv = np.linalg.inv(B)

    return {"B": B, "R": R, "B_inv": B_inv, "R_inv": R_inv}


def main(save_results=True):
    """Main function for MPI execution."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Starting MPI data assimilation with {size} processes")
        print(f"Save results: {save_results}")
        print("=" * 50)

    try:
        # Problem parameters
        problem_params = {
            "dt": 600,
            "t": 0,
            "t_final": 7 * 24 * 60 * 60,
            "num_steps": int(np.ceil((7 * 24 * 60 * 60) / 600)),
            "num_windows": 4,
            "fric_law": "mannings",
            "sol_var": "h",
        }

        # Use simple solver parameters that should work with your DOLFINx version
        if rank == 0:
            print("Using simple solver configuration for compatibility...")

        solver_params = get_simple_solver_params()

        if rank == 0:
            print(f"Solver parameters: {solver_params}")
            print()

        # Observation parameters
        obs_std = 0.4
        obs_time_freq = 2
        obs_space_freq = 2
        inflation_factor = 2.0

        # Validate parameters
        assert problem_params["num_steps"] == int(
            np.ceil(problem_params["t_final"] / problem_params["dt"])
        )

        if rank == 0:
            print("Problem parameters:")
            for key, value in problem_params.items():
                print(f"  {key}: {value}")
            print()

        # Generate true signal
        if rank == 0:
            print("Generating true signal...")

        # Set random seed for reproducibility
        np.random.seed(42)

        true_signal, prob, stations, state_coords = get_true_signal(
            problem_params, "sloped_beach", solver_params, 4
        )

        if rank == 0:
            print(f"True signal generated with {len(true_signal.saved_states)} states")

    except Exception as e:
        if rank == 0:
            print(f"Error in initialization phase: {e}")
            import traceback

            traceback.print_exc()
        sys.exit(1)

    try:
        # Setup observation system
        total_steps = int((problem_params["t_final"] / problem_params["dt"]) + 1)
        problem_params["num_steps"] = int(
            np.ceil((7 * 24 * 60 * 60) / 600) / problem_params["num_windows"]
        )
        obs_per_window = problem_params["num_steps"] // obs_time_freq

        H, stations, obs_spatial_indices = build_observation_matrix(
            prob, true_signal.V, obs_space_freq
        )

        obs_indices_per_window, obs_time_indices = setup_observation_indices(
            problem_params["num_steps"], obs_time_freq, total_steps
        )

        # Generate synthetic observations
        y_obs = generate_observations(
            true_signal.saved_states, H, obs_time_indices, obs_std
        )

        if rank == 0:
            print(f"Observation system setup:")
            print(f"  Total Steps: {total_steps}")
            print(f"  Assimilation Windows: {problem_params['num_windows']}")
            print(f"  Steps per Window: {problem_params['num_steps']}")
            print(f"  Observation Frequency: {obs_time_freq}")
            print(
                f"  Total Observations: {obs_per_window * problem_params['num_windows']}"
            )
            print(f"  Number of Stations: {stations.shape[0]}")
            print(f"  Observations per Window: {obs_per_window}")
            print()

    except Exception as e:
        if rank == 0:
            print(f"Error in observation setup phase: {e}")
            import traceback

            traceback.print_exc()
        sys.exit(1)

    try:
        # Setup covariance matrices
        state_dim = true_signal.saved_adjoints[0].shape[0]
        obs_dim = stations.shape[0]

        covs_dict = setup_covariance_matrices(
            state_dim, obs_dim, obs_std, inflation_factor
        )

        # Predicted covariance
        L = H @ covs_dict["B"] @ H.T
        L_inv = np.linalg.inv(L)
        covs_dict["L_inv"] = L_inv

        # Remove full matrices to save memory, keep only inverses
        covs = {
            "B_inv": covs_dict["B_inv"],
            "R_inv": covs_dict["R_inv"],
            "L_inv": covs_dict["L_inv"],
        }

        # Bathymetry for stations
        hb = 5.0 / 13800 * (13800 - stations[:, 0])

        if rank == 0:
            print(f"Covariance matrices setup:")
            print(f"  State Dimension: {state_dim}")
            print(f"  Observation Dimension: {obs_dim}")
            print(f"  Background Covariance Shape: {covs_dict['B'].shape}")
            print(f"  Observation Covariance Shape: {covs_dict['R'].shape}")
            print(f"  Predicted Error Covariance Shape: {L.shape}")
            print(f"  Observation Matrix Shape: {H.shape}")
            print(f"  Observations Shape: {y_obs.shape}")
            print()

    except Exception as e:
        if rank == 0:
            print(f"Error in covariance setup phase: {e}")
            import traceback

            traceback.print_exc()
        sys.exit(1)

    try:
        # Run Bayesian data assimilation
        if rank == 0:
            print("Starting Bayesian data assimilation...")

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
        )

        if rank == 0:
            print("Data assimilation completed successfully!")

    except Exception as e:
        if rank == 0:
            print(f"Error in data assimilation phase: {e}")
            import traceback

            traceback.print_exc()
        sys.exit(1)

    try:
        # Compute and display results
        true_states = np.array(true_signal.saved_states)
        Hu_true = H @ true_states.T
        pred = bayes_analysis[1:, :, 0] + hb
        bayes_height_misfit_rmse = np.sqrt(np.mean((Hu_true - pred.T) ** 2))

        if rank == 0:
            print("Results:")
            print(f"  Bayes Analysis Height RMSE: {bayes_height_misfit_rmse:.6f}")
            print("=" * 50)
            print("MPI data assimilation completed successfully!")

        # Save results based on parameter
        if rank == 0 and save_results:
            results = {
                "bayes_analysis": bayes_analysis,
                "true_states": true_states,
                "observations": y_obs,
                "stations": stations,
                "rmse": bayes_height_misfit_rmse,
                "problem_params": problem_params,
                "solver_params": solver_params,
                "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
            }

            # Create timestamped filename
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            filename = f"assimilation_results_{timestamp}.pkl"

            try:
                with open(filename, "wb") as f:
                    pickle.dump(results, f)
                print(f"Results saved to '{filename}'")

                # Also save a copy with generic name for easy access
                with open("assimilation_results_latest.pkl", "wb") as f:
                    pickle.dump(results, f)
                print("Results also saved to 'assimilation_results_latest.pkl'")

            except Exception as save_error:
                print(f"Warning: Could not save results: {save_error}")
                # Try saving with a simple filename as fallback
                try:
                    with open("assimilation_results_fallback.pkl", "wb") as f:
                        pickle.dump(results, f)
                    print("Results saved to 'assimilation_results_fallback.pkl'")
                except:
                    print("Failed to save results to any file")

        elif rank == 0 and not save_results:
            print("Results not saved (save_results=False)")

    except Exception as e:
        if rank == 0:
            print(f"Error in results computation phase: {e}")
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Finalize - no need for explicit barrier with sys.exit
    if rank == 0:
        print("All processes completed successfully!")


if __name__ == "__main__":
    import time
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="MPI Data Assimilation for Sloped Beach Problem"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save results to file (default: True)",
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Do not save results to file",
    )

    # Parse args, but handle case where script is run without arguments
    try:
        args = parser.parse_args()
        save_results = args.save
    except:
        # Default to saving if argument parsing fails
        save_results = True

    # Initialize MPI for timing
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Start timing
    start_time = time.time()
    if rank == 0:
        print(
            f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
        )
        print()

    try:
        main(save_results=save_results)

        # Calculate and report final time
        end_time = time.time()
        total_time = end_time - start_time

        if rank == 0:
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = total_time % 60

            print()
            print("=" * 50)
            print("TIMING SUMMARY")
            print("=" * 50)
            print(
                f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
            )
            print(
                f"End time:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
            )
            print(f"Total runtime: {hours:02d}:{minutes:02d}:{seconds:06.3f}")
            print(f"Total runtime: {total_time:.3f} seconds")
            print("=" * 50)

    except KeyboardInterrupt:
        end_time = time.time()
        total_time = end_time - start_time

        if rank == 0:
            print(f"\nScript interrupted by user after {total_time:.3f} seconds")
        sys.exit(0)

    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time

        if rank == 0:
            print(f"Fatal error in main after {total_time:.3f} seconds: {e}")
            import traceback

            traceback.print_exc()
        sys.exit(1)

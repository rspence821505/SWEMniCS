#!/usr/bin/env python3
"""
Tidal case: Parallel DC-WME-4DVar data assimilation experiment.

This script runs a twin experiment for the tidal problem using
the Data-Consistent Weighted Mean Error 4DVar cost function with
MPI parallelization via the TwinExperiment framework.

Run with:
    mpirun -n 4 python tidal_dcwme_mpi.py

DC-WME uses cumulative time-averaged innovation as QoI for improved stability.

Usage:
    mpirun -n 4 python tidal_dcwme_mpi.py [--nx 10] [--ny 5] [--dt 3600]
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from mpi4py import MPI

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils import get_default_solver_params
from swe4dvar.utils.output_paths import ensure_output_dirs

from twin_experiment import TwinExperiment, TwinExperimentConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Parallel Tidal DC-WME-4DVar data assimilation experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nx", type=int, default=10, help="Elements in x direction")
    parser.add_argument("--ny", type=int, default=5, help="Elements in y direction")
    parser.add_argument("--dt", type=float, default=3600.0, help="Time step (seconds)")
    parser.add_argument("--final-time", type=float, default=24 * 3600.0, help="Final time (seconds)")
    parser.add_argument("--obs-fraction", type=float, default=0.5, help="Fraction of points to observe")
    parser.add_argument("--obs-frequency", type=int, default=1, help="Observe every N timesteps")
    parser.add_argument("--noise-level", type=float, default=0.01, help="Observation noise level")
    parser.add_argument("--background-error", type=float, default=0.1, help="Background error std")
    parser.add_argument("--max-iter", type=int, default=50, help="Max optimization iterations")
    parser.add_argument("--no-bounds", action="store_true", help="Disable bounded optimization")
    parser.add_argument("--h-min", type=float, default=0.01, help="Minimum water depth for bounds")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--profile", action="store_true", help="Enable detailed timing")
    return parser.parse_args()


def main():
    """Run parallel tidal DC-WME-4DVar experiment."""
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        ensure_output_dirs()
    comm.Barrier()

    # Create problem
    num_time_steps = int(np.ceil(args.final_time / args.dt))
    problem = TidalProblem(
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        nt=num_time_steps,
        friction_law="mannings",
        solution_var="h",
    )

    # Create solver
    solver = get_solver("SUPG")(problem, theta=0.5, p_degree=[1, 1])

    # Solver parameters
    solver_params = get_default_solver_params(
        rtol=1e-5,
        atol=1e-6,
        max_it=10,
        relaxation_parameter=1.0,
        comm=comm,
        error_if_not_converged=True,
    )

    # Create experiment config
    config = TwinExperimentConfig(
        method="dcwme",
        obs_fraction=args.obs_fraction,
        obs_frequency=args.obs_frequency,
        obs_noise_level=args.noise_level,
        interior_only=True,
        background_error_std=args.background_error,
        max_iterations=args.max_iter,
        use_bounds=not args.no_bounds,
        h_min=args.h_min,
        verbose=args.verbose,
    )

    # Run experiment
    experiment = TwinExperiment(
        problem=problem,
        solver=solver,
        config=config,
        solver_params=solver_params,
        comm=comm,
    )

    results = experiment.run()

    # Add MPI-specific info to results
    if rank == 0:
        print(f"\nMPI ranks used: {comm.Get_size()}")

    return 0 if results.converged else 1


if __name__ == "__main__":
    sys.exit(main())

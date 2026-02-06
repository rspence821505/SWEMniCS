#!/usr/bin/env python3
"""
Command-line runner for generalized twin experiments.

This script provides a unified interface for running twin experiments
with different problems (Tidal, ADCIRC/Shinnecock, etc.) and DA methods
(4D-Var, DC-WME).

Usage:
    # Tidal problem with 4D-Var
    python run_twin_experiment.py --problem tidal --method 4dvar

    # Tidal problem with DC-WME
    python run_twin_experiment.py --problem tidal --method dcwme

    # Shinnecock with 4D-Var
    python run_twin_experiment.py --problem shinnecock --method 4dvar \\
        --adios-file data/shinnecock_inlet

    # Custom grid size and time parameters
    python run_twin_experiment.py --problem tidal --nx 20 --ny 10 \\
        --dt 1800 --final-time 172800

    # MPI parallel run
    mpirun -n 4 python run_twin_experiment.py --problem tidal --method 4dvar

Examples:
    # Quick test with small grid
    python run_twin_experiment.py --problem tidal --nx 5 --ny 3 \\
        --final-time 7200 --max-iter 10

    # Full experiment with default settings
    python run_twin_experiment.py --problem tidal --method 4dvar

    # Shinnecock inlet experiment
    python run_twin_experiment.py --problem shinnecock --method dcwme \\
        --adios-file data/shinnecock_inlet --dt 600 --T 24
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from mpi4py import MPI

from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils import get_default_solver_params
from swe4dvar.utils.output_paths import ensure_output_dirs

from twin_experiment import (
    TwinExperiment,
    TwinExperimentConfig,
    create_problem,
    register_problem,
    PROBLEM_REGISTRY,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run twin experiment for data assimilation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Problem selection
    parser.add_argument(
        "--problem",
        type=str,
        default="tidal",
        choices=["tidal", "shinnecock", "adcirc"],
        help="Problem type to run",
    )

    # DA method
    parser.add_argument(
        "--method",
        type=str,
        default="4dvar",
        choices=["4dvar", "dcwme"],
        help="Data assimilation method",
    )

    # Problem parameters (Tidal)
    parser.add_argument(
        "--nx", type=int, default=10, help="Elements in x direction (tidal)"
    )
    parser.add_argument(
        "--ny", type=int, default=5, help="Elements in y direction (tidal)"
    )
    parser.add_argument(
        "--dt", type=float, default=3600.0, help="Time step in seconds"
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=24 * 3600.0,
        help="Final time in seconds (tidal)",
    )
    parser.add_argument(
        "--T",
        type=float,
        default=None,
        help="Final time in hours (overrides --final-time)",
    )

    # Problem parameters (ADCIRC/Shinnecock)
    parser.add_argument(
        "--adios-file",
        type=str,
        default="data/shinnecock_inlet",
        help="Path to ADCIRC ADIOS files (without extension)",
    )
    parser.add_argument(
        "--dramp",
        type=float,
        default=2.0,
        help="Tidal forcing ramp-up time in days",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Wetting/drying alpha parameter (None disables WD)",
    )

    # Solver parameters
    parser.add_argument(
        "--solver",
        type=str,
        default="SUPG",
        choices=["CG", "SUPG", "DG", "DGNC"],
        help="Solver type",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.5,
        help="Time-stepping scheme (0=IE, 0.5=BDF2, 1=CN)",
    )

    # Observation parameters
    parser.add_argument(
        "--obs-fraction",
        type=float,
        default=0.5,
        help="Fraction of points to observe",
    )
    parser.add_argument(
        "--obs-frequency",
        type=int,
        default=1,
        help="Observe every N timesteps",
    )
    parser.add_argument(
        "--obs-noise",
        type=float,
        default=0.01,
        help="Observation noise level",
    )
    parser.add_argument(
        "--obs-points-file",
        type=str,
        default=None,
        help="JSON file with pre-selected observation points",
    )
    parser.add_argument(
        "--all-nodes",
        action="store_true",
        help="Observe all nodes (not just interior)",
    )

    # Background error
    parser.add_argument(
        "--background-error",
        type=float,
        default=0.1,
        help="Background error std deviation",
    )

    # Optimization parameters
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="Maximum optimization iterations",
    )
    parser.add_argument(
        "--no-bounds",
        action="store_true",
        help="Disable bounded optimization",
    )
    parser.add_argument(
        "--h-min",
        type=float,
        default=0.01,
        help="Minimum water depth for bounded optimization",
    )
    parser.add_argument(
        "--component-aware-cov",
        action="store_true",
        help="Use component-aware covariance",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/data",
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    return parser.parse_args()


def create_tidal_problem(args):
    """Create a TidalProblem instance from arguments."""
    from swe4dvar.forward.problems import TidalProblem

    final_time = args.T * 3600 if args.T is not None else args.final_time
    num_time_steps = int(np.ceil(final_time / args.dt))

    return TidalProblem(
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        nt=num_time_steps,
        friction_law="mannings",
        solution_var="h",
    )


def create_adcirc_problem(args):
    """Create an ADCIRCProblem instance from arguments."""
    from swe4dvar.forward.adcirc_problem import ADCIRCProblem

    final_time = args.T * 3600 if args.T is not None else args.final_time
    num_time_steps = int(np.ceil(final_time / args.dt))

    wd = args.alpha is not None
    bath_adjust = 0 if wd else 4.0

    return ADCIRCProblem(
        adios_file=args.adios_file,
        spherical=True,
        solution_var="h",
        friction_law="mannings",
        wd=wd,
        wd_alpha=args.alpha,
        dt=args.dt,
        bathy_adjustment=bath_adjust,
        nt=num_time_steps,
        dramp=args.dramp,
    )


def main():
    """Main entry point."""
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Ensure output directories exist
    if rank == 0:
        ensure_output_dirs()
    comm.Barrier()

    # Create problem
    if args.problem == "tidal":
        problem = create_tidal_problem(args)
    elif args.problem in ["shinnecock", "adcirc"]:
        problem = create_adcirc_problem(args)
    else:
        raise ValueError(f"Unknown problem: {args.problem}")

    # Create solver
    solver = get_solver(args.solver)(
        problem, theta=args.theta, p_degree=[1, 1], verbose=not args.quiet
    )

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
        method=args.method,
        obs_fraction=args.obs_fraction,
        obs_frequency=args.obs_frequency,
        obs_noise_level=args.obs_noise,
        obs_points_file=args.obs_points_file,
        interior_only=not args.all_nodes,
        background_error_std=args.background_error,
        max_iterations=args.max_iter,
        use_bounds=not args.no_bounds,
        h_min=args.h_min,
        component_aware_cov=args.component_aware_cov,
        output_dir=args.output_dir,
        verbose=args.verbose and not args.quiet,
    )

    # Create and run experiment
    experiment = TwinExperiment(
        problem=problem,
        solver=solver,
        config=config,
        solver_params=solver_params,
        comm=comm,
    )

    results = experiment.run()

    # Return exit code based on convergence
    return 0 if results.converged else 1


if __name__ == "__main__":
    sys.exit(main())

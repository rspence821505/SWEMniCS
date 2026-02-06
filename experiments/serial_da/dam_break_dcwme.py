#!/usr/bin/env python3
"""
Dam break case: DC-WME-4DVar data assimilation experiment.

This script runs a twin experiment for the dam break problem using
the Data-Consistent Weighted Mean Error 4DVar cost function via
the TwinExperiment framework.

The dam break problem features:
- Discontinuous initial water height
- Rapid flow dynamics with shock formation
- No friction for analytical comparison

Usage:
    python dam_break_dcwme.py [--nx 30] [--ny 30] [--dt 0.5] [--final-time 20]
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from mpi4py import MPI

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from swe4dvar.forward.problems import DamProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar import FrictionLaw
from swe4dvar.utils import get_default_solver_params
from swe4dvar.utils.output_paths import ensure_output_dirs

from twin_experiment import TwinExperiment, TwinExperimentConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dam break DC-WME-4DVar data assimilation experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nx", type=int, default=30, help="Elements in x direction")
    parser.add_argument("--ny", type=int, default=30, help="Elements in y direction")
    parser.add_argument("--dt", type=float, default=0.5, help="Time step (seconds)")
    parser.add_argument("--final-time", type=float, default=20.0, help="Final time (seconds)")
    parser.add_argument("--obs-fraction", type=float, default=0.5, help="Fraction of points to observe")
    parser.add_argument("--obs-frequency", type=int, default=4, help="Observe every N timesteps")
    parser.add_argument("--noise-level", type=float, default=0.01, help="Observation noise level")
    parser.add_argument("--background-error", type=float, default=0.1, help="Background error std")
    parser.add_argument("--max-iter", type=int, default=50, help="Max optimization iterations")
    parser.add_argument("--solver", type=str, default="DG", choices=["CG", "DG", "SUPG"],
                        help="Solver type")
    parser.add_argument("--no-bounds", action="store_true", help="Disable bounded optimization")
    parser.add_argument("--h-min", type=float, default=0.01, help="Minimum water depth for bounds")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main():
    """Run dam break DC-WME-4DVar experiment."""
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        ensure_output_dirs()
    comm.Barrier()

    # Create problem
    num_time_steps = int(np.ceil(args.final_time / args.dt))
    problem = DamProblem(
        dt=args.dt,
        nt=num_time_steps,
        nx=args.nx,
        ny=args.ny,
        friction_law=FrictionLaw.none,
        solution_var="h",
        spherical=False,
    )

    # Create solver
    solver = get_solver(args.solver)(problem, theta=0.5, p_degree=[1, 1])

    # Solver parameters
    solver_params = get_default_solver_params(
        rtol=1e-5,
        atol=1e-6,
        max_it=10,
        relaxation_parameter=1.0,
        ksp_type="gmres",
        pc_type="ilu",
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
    return 0 if results.converged else 1


if __name__ == "__main__":
    sys.exit(main())

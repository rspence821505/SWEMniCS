# # Shinnecock Inlet Simulation
#
# This example demonstrates how to run a real-world tidal simulation for
# Shinnecock Inlet on Long Island, NY. It uses real bathymetry data and
# tidal forcing from ADCIRC input files.
#
# The simulation supports optional data assimilation (4D-Var or DC-WME)
# for parameter estimation and state reconstruction.
#
# Command-line Options:
#   --dt SECONDS          : Time step in seconds (default: 600)
#   --T HOURS             : Total simulation time in hours (default: 119)
#   --solver TYPE         : Solver type: DG, SUPG, DGNC (default: DG)
#   --alpha VALUE         : Wetting/drying alpha parameter (default: 1.5)
#   --theta VALUE         : Time-stepping scheme (default: 1.0)
#   --output-prefix NAME  : Prefix for output files (default: shinnecock)
#   --verbose             : Enable verbose output
#   --profile             : Enable detailed profiling
#   --no-video            : Disable video/animation output
#
# Newton Diagnostics Options:
#   --newton-log FILE     : Log detailed Newton iteration info to FILE
#   --newton-quiet        : Suppress Newton console output
#   --newton-store        : Store convergence history for analysis
#   --newton-verbose      : Show detailed iteration info (not just summaries)
#
# Data Assimilation Options:
#   --da-mode MODE        : DA mode: none, 4dvar, dcwme (default: none)
#   --obs-fraction FRAC   : Fraction of spatial points to observe (default: 0.5)
#   --obs-frequency N     : Observe every N timesteps (default: 6)
#   --obs-noise LEVEL     : Observation noise level (default: 0.01)
#   --background-error STD: Background error std deviation (default: 0.1)
#   --max-da-iter N       : Max DA optimization iterations (default: 50)
#
# Examples:
#   # Default forward simulation:
#   python shinnecock.py
#
#   # High-resolution simulation with profiling:
#   python shinnecock.py --dt 300 --T 48 --profile
#
#   # Quiet mode with file logging:
#   python shinnecock.py --newton-quiet --newton-log newton.log
#
#   # Run 4D-Var data assimilation:
#   python shinnecock.py --da-mode 4dvar --obs-fraction 0.5
#
#   # MPI parallel run:
#   mpirun -n 4 python shinnecock.py --verbose

import argparse
import time
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc

from swe4dvar.forward import solvers as Solvers
from swe4dvar.forward.adcirc_problem import ADCIRCProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils.timing import Timer
from swe4dvar.utils import get_default_solver_params
from swe4dvar.utils.output_paths import (
    FIGURES_DIR,
    DATA_DIR,
    LOGS_DIR,
    CHECKPOINTS_DIR,
    ensure_output_dirs,
)
from swe4dvar.utils.parallel_ops import ParallelTimer
from swe4dvar.physics.constants import R

# Optional DA imports (only loaded if DA mode is enabled)
DA_AVAILABLE = True
try:
    # Use generalized twin experiment framework
    sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
    from twin_experiment import TwinExperiment, TwinExperimentConfig
except ImportError:
    DA_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Shinnecock Inlet tidal simulation with optional data assimilation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Simulation parameters
    parser.add_argument(
        "--dt", type=float, default=600, help="Time step in seconds"
    )
    parser.add_argument(
        "--T",
        type=float,
        default=119,
        help="Total simulation time in hours (default: ~5 days minus 1 hour)",
    )
    parser.add_argument(
        "--solver",
        choices=["dg", "supg", "dgnc", "DG", "SUPG", "DGNC"],
        default="dg",
        help="Solver type",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.5,
        help="Wetting/drying alpha parameter (None disables WD)",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=1.0,
        help="Time-stepping scheme: 0=Implicit Euler, 0.5=BDF2, 1=Crank-Nicholson",
    )
    parser.add_argument(
        "--dramp",
        type=float,
        default=2.0,
        help="Tidal forcing ramp-up time in days",
    )

    # Data file configuration
    parser.add_argument(
        "--adios-file",
        type=str,
        default="data/shinnecock_inlet",
        help="Path to ADCIRC ADIOS files (without extension)",
    )

    # Output configuration
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="shinnecock",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable detailed profiling"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        default=True,
        help="Disable video/animation output",
    )

    # Newton diagnostics
    parser.add_argument(
        "--newton-log",
        type=str,
        default=None,
        help="Path to Newton diagnostics log file",
    )
    parser.add_argument(
        "--newton-quiet",
        action="store_true",
        help="Suppress Newton solver console output",
    )
    parser.add_argument(
        "--newton-store",
        action="store_true",
        help="Store Newton convergence history for analysis",
    )
    parser.add_argument(
        "--newton-verbose",
        action="store_true",
        help="Show detailed Newton iteration info",
    )

    # Data assimilation options
    parser.add_argument(
        "--da-mode",
        choices=["none", "4dvar", "dcwme"],
        default="none",
        help="Data assimilation mode",
    )
    parser.add_argument(
        "--obs-points-file",
        type=str,
        default=None,
        help="JSON file with pre-selected observation points (from select_observation_points.py)",
    )
    parser.add_argument(
        "--obs-fraction",
        type=float,
        default=0.5,
        help="Fraction of spatial points to observe (ignored if --obs-points-file is set)",
    )
    parser.add_argument(
        "--obs-frequency",
        type=int,
        default=6,
        help="Observe every N timesteps (6 = ~1 hour with dt=600)",
    )
    parser.add_argument(
        "--obs-noise",
        type=float,
        default=0.01,
        help="Observation noise level (fraction of signal)",
    )
    parser.add_argument(
        "--background-error",
        type=float,
        default=0.1,
        help="Background error standard deviation",
    )
    parser.add_argument(
        "--max-da-iter",
        type=int,
        default=50,
        help="Maximum DA optimization iterations",
    )

    return parser.parse_args()


def setup_observation_stations():
    """
    Set up observation stations for Shinnecock Inlet.

    Returns coordinates in projected (meters) format for DOLFINx.
    The default station is in the middle of the channel.
    """
    # Station in middle of channel (Lon/Lat)
    stations = np.array([[-72.476519, 40.840969, 0.0]])

    # Transform to projected coordinates (radians, then meters)
    stations_rad = np.deg2rad(stations)
    lat0 = 35  # Reference latitude for projection
    stations_rad[:, 0] *= R * np.cos(np.deg2rad(lat0))
    stations_rad[:, 1] *= R

    return stations_rad


def get_tidal_verification_data(nt, t_f, dt):
    """
    Get expected tidal signal for verification.

    Returns the theoretical tidal elevation at a boundary node
    based on the tidal constituents.
    """
    nbfr = 5  # Number of tidal constituents
    t = np.linspace(0, t_f, nt + 1)

    # Tidal constituent attributes
    nodal_factors = np.array([1.021, 1.021, 1.000, 0.947, 0.913])
    rad_freq = np.array([
        0.000140518902509,
        0.000137879699487,
        0.000145444104333,
        0.000072921158358,
        0.000067597744151,
    ])
    equil_args = np.array([98.846, 285.394, 360.000, 32.493, 70.357])

    # Amplitudes and phases at verification location
    amplitudes = np.array([0.44836049, 0.11585067, 0.07134235, 0.06428241, 0.05777383])
    phases = np.array([343.380, 335.853, 18.367, 180.254, 189.278])

    equil_args = np.deg2rad(equil_args)
    phases = np.deg2rad(phases)

    # Compute tidal signal
    eta_input = np.zeros(nt + 1)
    for i in range(nbfr):
        eta_input += (
            nodal_factors[i]
            * amplitudes[i]
            * np.cos(rad_freq[i] * t + equil_args[i] - phases[i])
        )

    return t, eta_input


def run_forward_simulation(args, comm, rank):
    """
    Run forward simulation without data assimilation.

    This is the original functionality of shinnecock.py, enhanced with
    timing instrumentation and output directory management.
    """
    # Initialize parallel timer
    timer = ParallelTimer(comm) if args.profile else None
    if timer:
        timer.start("total")

    # Ensure output directories exist
    if rank == 0:
        ensure_output_dirs()
    comm.Barrier()

    # Simulation parameters
    dt = args.dt
    t_f = args.T * 3600  # Convert hours to seconds
    nt = int(t_f / dt)
    is_spherical = True
    wd = args.alpha is not None
    bath_adjust = 0 if wd else 4.0

    if rank == 0 and args.verbose:
        print("=" * 70)
        print("Shinnecock Inlet Forward Simulation")
        print("=" * 70)
        print(f"Time step: {dt} s")
        print(f"Final time: {t_f} s ({args.T} hours)")
        print(f"Number of time steps: {nt}")
        print(f"Wetting/drying: {wd} (alpha={args.alpha})")
        print(f"Solver: {args.solver.upper()}")
        print("=" * 70)

    # Create problem
    with Timer("Problem Setup", verbose=(rank == 0 and args.verbose), track_key="setup"):
        if timer:
            timer.start("setup")

        prob = ADCIRCProblem(
            adios_file=args.adios_file,
            spherical=is_spherical,
            solution_var="h",
            friction_law="mannings",
            wd=wd,
            wd_alpha=args.alpha,
            dt=dt,
            bathy_adjustment=bath_adjust,
            nt=nt,
            dramp=args.dramp,
        )

        # Set up observation stations
        stations = setup_observation_stations()
        if rank == 0 and args.verbose:
            print(f"Observation stations: {stations}")

        # Create solver
        theta = args.theta
        p_degree = [1, 1]
        solver_name = args.solver.upper()

        if solver_name == "SUPG":
            solver = Solvers.SUPGImplicit(prob, theta, p_degree=p_degree)
        elif solver_name == "DG":
            solver = Solvers.DGImplicit(prob, theta, p_degree=p_degree)
        elif solver_name == "DGNC":
            solver = Solvers.DGImplicitNonConservative(prob, theta, p_degree=p_degree)
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

        # Solver parameters
        params = get_default_solver_params(
            rtol=1e-5,
            atol=1e-6,
            max_it=10,
            relaxation_parameter=1.0,
            ksp_type="gmres",
            pc_type="bjacobi",
            comm=comm,
            error_if_not_converged=True,
        )

        if rank == 0 and args.verbose:
            solver.print_config()

        if timer:
            timer.stop("setup")

    # Configure Newton diagnostics
    newton_config = None
    if args.newton_log or args.newton_quiet or args.newton_store or args.newton_verbose:
        newton_config = {
            "print_to_console": not args.newton_quiet,
            "log_file": args.newton_log,
            "store_history": args.newton_store,
            "verbose": args.newton_verbose,
        }

    # Run time loop
    with Timer(
        "Time Loop (Simulation)", verbose=(rank == 0 and args.verbose), track_key="simulation"
    ):
        if timer:
            timer.start("simulation")

        start_time = time.time()
        plot_name = (
            f"{args.output_prefix}"
            if not wd
            else f"{args.output_prefix}-wd-{args.alpha}"
        )

        solver.time_loop(
            solver_parameters=params,
            plot_every=1,
            plot_name=plot_name,
            stations=stations,
            save_state=True,
            adjoint_method=True,
            monitor_progress=(rank == 0 and args.verbose),
            newton_diagnostics_config=newton_config,
            enable_video=not args.no_video,
        )

        if timer:
            timer.stop("simulation")

    # Post-processing
    with Timer(
        "Post-Processing", verbose=(rank == 0 and args.verbose), track_key="post_processing"
    ):
        if timer:
            timer.start("post_processing")

        # Print Newton diagnostics summary if stored
        if args.newton_store:
            diagnostics = solver.solver.diagnostics
            diagnostics.print_summary()

            # Save to JSON
            json_file = str(DATA_DIR / f"{args.output_prefix}_newton_convergence.json")
            diagnostics.save_json(json_file)

        if rank == 0:
            elapsed = time.time() - start_time
            print(
                f"---------Simulation finished with run time {elapsed:.2f} seconds -------------"
            )

            # Save results to output directory
            outdir = str(DATA_DIR) + "/"
            np.savetxt(
                f"{outdir}{solver_name}_p1_{args.output_prefix}_h.csv",
                solver.vals[:, :, 0],
                delimiter=",",
            )
            np.savetxt(
                f"{outdir}{solver_name}_p1_{args.output_prefix}_xvel.csv",
                solver.vals[:, :, 1],
                delimiter=",",
            )
            np.savetxt(
                f"{outdir}{solver_name}_p1_{args.output_prefix}_yvel.csv",
                solver.vals[:, :, 2],
                delimiter=",",
            )

            # Generate verification plots
            t, eta_input = get_tidal_verification_data(nt, t_f, dt)
            plot_results(
                solver,
                nt,
                t_f,
                t,
                eta_input,
                solver_name,
                args.output_prefix,
                str(FIGURES_DIR),
            )

        if timer:
            timer.stop("post_processing")

    if timer:
        timer.stop("total")
        timer.report(root=0)

    # Print timing summary
    if rank == 0:
        Timer.print_summary(per_item_key="simulation", num_items=nt)


def run_da_experiment(args, comm, rank):
    """
    Run data assimilation experiment (4D-Var or DC-WME).

    Uses the generalized TwinExperiment framework.
    """
    if not DA_AVAILABLE:
        if rank == 0:
            print("ERROR: Data assimilation modules not available.")
            print("Please ensure twin_experiment module is available.")
        return

    # Ensure output directories
    if rank == 0:
        ensure_output_dirs()
    comm.Barrier()

    # Simulation parameters
    dt = args.dt
    t_f = args.T * 3600
    nt = int(t_f / dt)
    is_spherical = True
    wd = args.alpha is not None
    bath_adjust = 0 if wd else 4.0

    # Create problem
    prob = ADCIRCProblem(
        adios_file=args.adios_file,
        spherical=is_spherical,
        solution_var="h",
        friction_law="mannings",
        wd=wd,
        wd_alpha=args.alpha,
        dt=dt,
        bathy_adjustment=bath_adjust,
        nt=nt,
        dramp=args.dramp,
    )

    # Create solver
    theta = args.theta
    p_degree = [1, 1]
    solver_name = args.solver.upper()

    if solver_name == "SUPG":
        solver = Solvers.SUPGImplicit(prob, theta, p_degree=p_degree, verbose=not args.newton_quiet)
    elif solver_name == "DG":
        solver = Solvers.DGImplicit(prob, theta, p_degree=p_degree, verbose=not args.newton_quiet)
    elif solver_name == "DGNC":
        solver = Solvers.DGImplicitNonConservative(
            prob, theta, p_degree=p_degree, verbose=not args.newton_quiet
        )
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    # Solver parameters
    params = get_default_solver_params(
        rtol=1e-5,
        atol=1e-6,
        max_it=10,
        relaxation_parameter=1.0,
        ksp_type="gmres",
        pc_type="bjacobi",
        comm=comm,
        error_if_not_converged=True,
    )

    # Create experiment config
    config = TwinExperimentConfig(
        method=args.da_mode,
        obs_fraction=args.obs_fraction,
        obs_frequency=args.obs_frequency,
        obs_noise_level=args.obs_noise,
        obs_points_file=args.obs_points_file,
        interior_only=True,
        background_error_std=args.background_error,
        max_iterations=args.max_da_iter,
        use_bounds=True,
        h_min=0.01,
        output_dir=str(DATA_DIR),
        verbose=args.verbose,
    )

    # Create and run experiment
    experiment = TwinExperiment(
        problem=prob,
        solver=solver,
        config=config,
        solver_params=params,
        comm=comm,
    )

    results = experiment.run()

    # Timing report
    if args.profile and rank == 0:
        print(f"\nTotal wall time: {results.wall_time:.2f} seconds")


def plot_results(solver, nt, t_f, t, eta_input, solver_name, prefix, output_dir):
    """Generate verification plots comparing simulation to expected tidal signal."""
    f_extension = "spherical.png"

    # Height plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.linspace(0, t_f, nt + 1),
        solver.vals[: nt + 1, 0, 0],
        "k",
        linewidth=2,
        label=f"{solver_name} solver",
    )
    plt.plot(
        np.linspace(0, t_f, nt + 1),
        eta_input,
        "bo",
        linewidth=2,
        label="Expected tidal signal",
        markersize=2,
    )
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Water Height (m)")
    plt.title("Shinnecock Inlet: Water Height at Channel Station")
    plt.legend()
    plt.savefig(f"{output_dir}/{prefix}_height_{f_extension}")
    plt.close()

    # X-velocity plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.linspace(0, t_f, nt + 1),
        solver.vals[: nt + 1, 0, 1],
        "k",
        linewidth=2,
        label=f"{solver_name} solver",
    )
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("X Velocity (m/s)")
    plt.title("Shinnecock Inlet: X Velocity at Channel Station")
    plt.legend()
    plt.savefig(f"{output_dir}/{prefix}_xvel_{f_extension}")
    plt.close()

    # Y-velocity plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.linspace(0, t_f, nt + 1),
        solver.vals[: nt + 1, 0, 2],
        "k",
        linewidth=2,
        label=f"{solver_name} solver",
    )
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Y Velocity (m/s)")
    plt.title("Shinnecock Inlet: Y Velocity at Channel Station")
    plt.legend()
    plt.savefig(f"{output_dir}/{prefix}_yvel_{f_extension}")
    plt.close()

    print(f"\nPlots saved to {output_dir}:")
    print(f"  - {prefix}_height_{f_extension}")
    print(f"  - {prefix}_xvel_{f_extension}")
    print(f"  - {prefix}_yvel_{f_extension}")


if __name__ == "__main__":
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Parse arguments
    args = parse_args()

    if rank == 0:
        print("Running Shinnecock Inlet simulation")

    # Run appropriate mode
    if args.da_mode == "none":
        run_forward_simulation(args, comm, rank)
    else:
        run_da_experiment(args, comm, rank)

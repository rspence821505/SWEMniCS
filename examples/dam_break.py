# # Dam Break Simulation
#
# This example demonstrates how to run a dam break simulation with optional
# Newton solver diagnostics for monitoring convergence behavior.
#
# Command-line Options:
#   --nx NUM              : Number of elements in x direction (default: 100)
#   --ny NUM              : Number of elements in y direction (default: 100)
#   --dt SECONDS          : Time step in seconds (default: 0.5)
#   --solver TYPE         : Solver type: CG, SUPG, DG, DGCG (default: CG)
#   --theta VALUE         : Time-stepping: 0=Implicit Euler, 0.5=BDF2, 1=Crank-Nicholson
#   --no-video            : Disable video/animation output during simulation
#   --outdir PATH         : Output directory for CSV files (default: current directory)
#
# Newton Diagnostics Options:
#   --newton-log FILE     : Log detailed Newton iteration info to FILE
#   --newton-quiet        : Suppress all solver console output (Newton, residuals, etc.)
#   --newton-store        : Store convergence history for analysis
#   --newton-verbose      : Show detailed iteration info (not just summaries)
#
# Examples:
#   # Default behavior:
#   python dam_break.py --solver CG
#
#   # Disable video for faster simulation:
#   python dam_break.py --solver SUPG --no-video
#
#   # Log Newton diagnostics to file:
#   python dam_break.py --solver CG --newton-log newton.log
#
#   # Store history and analyze convergence:
#   python dam_break.py --solver DG --newton-store
#
#   # Quiet mode with file logging and no video:
#   python dam_break.py --solver CG --newton-quiet --newton-log newton.log --no-video
#


import argparse
from swe4dvar.forward.problems import DamProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils.visualization import SolverVisualizer
from swe4dvar.utils.timing import Timer
from swe4dvar.utils import get_default_solver_params
from swe4dvar.utils.output_paths import FIGURES_DIR, DATA_DIR, ensure_output_dirs
from swe4dvar import FrictionLaw
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import os


parser = argparse.ArgumentParser(
    description="Dam break simulation",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--nx", dest="nx", type=int, default=100, help="number of elements in x direction"
)
parser.add_argument(
    "--ny", dest="ny", type=int, default=100, help="number of elements in y direction"
)
parser.add_argument(
    "--dt", type=float, dest="dt", default=0.5, help="time step in seconds"
)
parser.add_argument(
    "--final_time",
    type=float,
    dest="final_time",
    default=40,
    help="final time in seconds",
)
parser.add_argument(
    "--solver",
    dest="solver",
    type=str,
    default="CG",
    help="solver type",
    choices=["CG", "SUPG", "DG", "DGCG"],
)
parser.add_argument(
    "--theta",
    dest="theta",
    type=float,
    default=1.0,
    choices=[0, 0.5, 1],
    help="Time-stepping scheme: 0: Implicit Euler, 0.5: BDF2, 1: Crank-Nicholson",
)
parser.add_argument(
    "--newton-log",
    dest="newton_log",
    type=str,
    default=None,
    help="Path to Newton diagnostics log file (default: no file logging)",
)
parser.add_argument(
    "--newton-quiet",
    dest="newton_quiet",
    action="store_true",
    help="Suppress all solver console output (Newton iterations, residuals, etc.)",
)
parser.add_argument(
    "--newton-store",
    dest="newton_store",
    action="store_true",
    help="Store Newton convergence history for analysis",
)
parser.add_argument(
    "--newton-verbose",
    dest="newton_verbose",
    action="store_true",
    help="Show detailed Newton iteration info (default: summaries only)",
)
parser.add_argument(
    "--no-video",
    dest="no_video",
    action="store_true",
    default=False,
    help="Disable video/animation output during simulation",
)
parser.add_argument(
    "--outdir",
    dest="outdir",
    type=str,
    default=None,
    help="Output directory for CSV files",
)

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Ensure output directories exist
ensure_output_dirs()

with Timer("Total Runtime", verbose=False, track_key="total"):
    with Timer("Problem Setup", verbose=False, track_key="setup"):
        # Used in plotting
        Lx = 1000
        dam_height = 2.0

        num_time_steps = int(np.ceil(args.final_time / args.dt))

        # Friction law either quadratic or linear
        fric_law = FrictionLaw.none
        # Choose solution variable, either h or eta or flux
        sol_var = "h"

        problem = DamProblem(
            dt=args.dt,
            nt=num_time_steps,
            nx=args.nx,
            ny=args.ny,
            friction_law=fric_law,
            solution_var=sol_var,
            spherical=False,
        )

        # Polynomial degree for each variable
        p_degree = [1, 1]

        # Time series output stations
        nx_stations = 100
        stations = np.zeros((nx_stations, 3))
        stations[:, 0] = np.linspace(0, 1000, nx_stations)
        stations[:, 1] = 450

        # Create solver object (verbose is controlled by --newton-quiet flag)
        solver = get_solver(args.solver)(
            problem, args.theta, p_degree=p_degree, verbose=not args.newton_quiet
        )

        # Get solver parameters with MPI-aware defaults
        params = get_default_solver_params(
            rtol=1e-5,
            atol=1e-6,
            max_it=10,
            relaxation_parameter=1.0,
            ksp_type="gmres",
            pc_type="ilu",
            comm=comm,
            error_if_not_converged=True,
        )

        # Print solver configuration
        solver.print_config()

        # Configure Newton diagnostics based on command-line arguments
        newton_config = None  # Default: uses solver.verbose
        if (
            args.newton_log
            or args.newton_quiet
            or args.newton_store
            or args.newton_verbose
        ):
            newton_config = {
                "print_to_console": not args.newton_quiet,
                "log_file": args.newton_log,
                "store_history": args.newton_store,
                "verbose": args.newton_verbose,
            }

    with Timer("Time Loop (Simulation)", verbose=False, track_key="simulation"):
        solver.time_loop(
            solver_parameters=params,
            stations=stations,
            plot_every=1,
            plot_name="dam_test_" + args.solver.upper(),
            save_state=True,
            adjoint_method=True,
            monitor_progress=True,
            newton_diagnostics_config=newton_config,
            enable_video=not args.no_video,
        )

    with Timer("Post-Processing", verbose=False, track_key="post_processing"):
        # Print Newton diagnostics summary if history was stored (MPI-aware)
        if args.newton_store:
            diagnostics = solver.solver.diagnostics
            diagnostics.print_summary()

            # Save to JSON for offline analysis (MPI-aware)
            json_file = str(DATA_DIR / "newton_convergence.json")
            diagnostics.save_json(json_file)

        # Save array for post processing
        # Use centralized DATA_DIR unless user explicitly specifies --outdir
        if args.outdir is not None:
            os.makedirs(args.outdir, exist_ok=True)
            outdir = args.outdir + "/"
        else:
            outdir = str(DATA_DIR) + "/"

        np.savetxt(
            f"{outdir}{args.solver.upper()}_p1_stations_h.csv",
            solver.vals[:, :, 0],
            delimiter=",",
        )
        np.savetxt(
            f"{outdir}{args.solver.upper()}_p1_stations_xvel.csv",
            solver.vals[:, :, 1],
            delimiter=",",
        )
        np.savetxt(
            f"{outdir}{args.solver.upper()}_p1_stations_yvel.csv",
            solver.vals[:, :, 2],
            delimiter=",",
        )

        # Plot results using SolverVisualizer (MPI-aware, no rank check needed)
        visualizer = SolverVisualizer(
            domain=solver.domain,
            V_scalar=solver.V_scalar,
            V_vel=solver.V_vel,
            problem=problem,
            verbose=False,
        )

        plt_nums = [0, 40, num_time_steps]
        # Use FIGURES_DIR for plots unless user explicitly specifies --outdir
        figures_outdir = args.outdir if args.outdir is not None else str(FIGURES_DIR)
        visualizer.plot_dam_break(
            solver_vals=solver.vals,
            dt=args.dt,
            nt=num_time_steps,
            Lx=Lx,
            dam_height=dam_height,
            timesteps=plt_nums,
            scheme_name=args.solver.upper(),
            output_dir=figures_outdir,
            analytical_solution_func=problem.get_analytic_solution,
        )

        visualizer.print_saved_files(
            f"\nPlots saved:",
            f"  - {figures_outdir}/dam_height_{args.solver.upper()}_order1_dt.png",
            f"  - {figures_outdir}/dam_velocity_{args.solver.upper()}_order1_dt.png",
        )

# Print timing summary
Timer.print_summary(per_item_key="simulation", num_items=num_time_steps)

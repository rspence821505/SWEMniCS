# # Simple tidal problem
#
# This example demonstrates how to run a tidal simulation with optional
# Newton solver diagnostics for monitoring convergence behavior.
#
# Command-line Options:
#   --nx NUM              : Number of elements in x direction (default: 20)
#   --ny NUM              : Number of elements in y direction (default: 5)
#   --dt SECONDS          : Time step in seconds (default: 3600)
#   --solver TYPE         : Solver type: CG, SUPG, DG, DGCG, DGNC (default: SUPG)
#   --theta VALUE         : Time-stepping: 0=Implicit Euler, 0.5=BDF2, 1=Crank-Nicholson
#   --no-video            : Disable video/animation output during simulation
#
# Newton Diagnostics Options:
#   --newton-log FILE     : Log detailed Newton iteration info to FILE
#   --newton-quiet        : Suppress Newton console output
#   --newton-store        : Store convergence history for analysis
#   --newton-verbose      : Show detailed iteration info (not just summaries)
#
# Examples:
#   # Default behavior (uses solver.verbose):
#   python tidal.py --nx 20 --ny 5
#
#   # Disable video for faster simulation:
#   python tidal.py --no-video
#
#   # Log Newton diagnostics to file:
#   python tidal.py --newton-log newton.log
#
#   # Store history and analyze convergence:
#   python tidal.py --newton-store
#
#   # Quiet mode with file logging and no video:
#   python tidal.py --newton-quiet --newton-log newton.log --no-video
#


import argparse
from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.forward import solvers as Solvers
from swe4dvar.utils.visualization import SolverVisualizer
from swe4dvar.utils.timing import Timer
from swe4dvar.utils import get_default_solver_params
from swe4dvar.utils.output_paths import FIGURES_DIR, DATA_DIR, LOGS_DIR, ensure_output_dirs

# from swe4dvar import FrictionLaw
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI


parser = argparse.ArgumentParser(
    description="Simple tidal problem",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--nx", dest="nx", type=int, default=20, help="number of elements in x direction"
)
parser.add_argument(
    "--ny", dest="ny", type=int, default=5, help="number of elements in y direction"
)
parser.add_argument(
    "--dt", type=float, dest="dt", default=3600, help="time step in seconds"
)
parser.add_argument(
    "--final_time",
    type=float,
    dest="final_time",
    default=7 * 24 * 60 * 60,
    help="final time in seconds",
)
# parser.add_argument(
#     "--friction",
#     dest="friction_law",
#     type=FrictionLaw,
#     choices=[FrictionLaw.linear, FrictionLaw.quadratic, FrictionLaw.mannings],
#     help="Choice of friction law",
#     default=FrictionLaw.mannings,
# )
parser.add_argument(
    "--solver",
    dest="solver",
    type=str,
    default="SUPG",
    help="solver type",
    choices=["CG", "SUPG", "DG", "DGCG", "DGNC"],
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
    help="Suppress Newton solver console output",
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
    default=True,
    help="Disable video/animation output during simulation",
)

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Ensure output directories exist
ensure_output_dirs()

with Timer("Total Runtime", verbose=False, track_key="total"):
    with Timer("Problem Setup", verbose=False, track_key="setup"):
        initial_time = 0
        final_time = 7 * 24 * 60 * 60  # 24*7
        num_time_steps = int(np.ceil(final_time / args.dt))

        # choose solution variable, either h or eta or flux
        sol_var = "h"

        # polynomial degree for each variable
        p_degree = [1, 1]

        problem = TidalProblem(
            nx=args.nx,
            ny=args.ny,
            dt=args.dt,
            nt=num_time_steps,
            friction_law="mannings",
            solution_var=sol_var,
        )

        # time series output locations
        stations = np.array([[800.5, 1000.5, 0.0]])

        # create solver object
        solver = get_solver(args.solver)(problem, args.theta, p_degree=p_degree)

        # Get solver parameters with MPI-aware defaults
        params = get_default_solver_params(
            rtol=1e-5,
            atol=1e-6,
            max_it=10,
            relaxation_parameter=1.0,
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
            plot_name="SUPG_tide",
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

        visualizer = SolverVisualizer(
            domain=solver.domain,
            V_scalar=solver.V_scalar,
            V_vel=solver.V_vel,
            problem=problem,
            verbose=False,
        )

        visualizer.plot_tidal_timeseries(
            solver_vals=solver.vals,
            dt=args.dt,
            nt=num_time_steps,
            station_idx=0,
            scheme_name=args.solver.upper(),
            output_dir=str(FIGURES_DIR),
        )

        visualizer.print_saved_files(
            f"\nPlots saved:",
            f"  - {FIGURES_DIR}/tidal_height_{args.solver.upper()}.png (water depth)",
            f"  - {FIGURES_DIR}/tidal_elevation_{args.solver.upper()}.png (water surface elevation)",
            f"  - {FIGURES_DIR}/tidal_velocity_u_{args.solver.upper()}.png (x-velocity)",
            f"  - {FIGURES_DIR}/tidal_velocity_v_{args.solver.upper()}.png (y-velocity)",
        )

# Print timing summary
Timer.print_summary(per_item_key="simulation", num_items=num_time_steps)

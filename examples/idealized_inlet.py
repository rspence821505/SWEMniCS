# # Idealized Inlet Problem
#
# This example demonstrates how to run an idealized inlet simulation with optional
# Newton solver diagnostics for monitoring convergence behavior.
#
# Command-line Options:
#   --dt SECONDS          : Time step in seconds (default: 1200)
#   --final_time SECONDS  : Final time in seconds (default: 345600, 4 days)
#   --solver TYPE         : Solver type: CG, SUPG, DG, DGCG, DGNC (default: DG)
#   --theta VALUE         : Time-stepping: 0=Implicit Euler, 0.5=BDF2, 1=Crank-Nicholson
#   --friction TYPE       : Friction law: linear, quadratic, mannings (default: quadratic)
#   --dramp DAYS          : Duration of ramp function in days (default: 2.0)
#   --no-video            : Disable video/animation output during simulation
#
# Newton Diagnostics Options:
#   --newton-log FILE     : Log detailed Newton iteration info to FILE
#   --newton-quiet        : Suppress Newton console output
#   --newton-store        : Store convergence history for analysis
#   --newton-verbose      : Show detailed iteration info (not just summaries)
#
# Examples:
#   # Default behavior:
#   python idealized_inlet.py
#
#   # Disable video for faster simulation:
#   python idealized_inlet.py --no-video
#
#   # Log Newton diagnostics to file:
#   python idealized_inlet.py --newton-log newton.log
#
#   # Store history and analyze convergence:
#   python idealized_inlet.py --newton-store
#
#   # Quiet mode with file logging and no video:
#   python idealized_inlet.py --newton-quiet --newton-log newton.log --no-video
#


import argparse
from swe4dvar.forward.problems import IdealizedInlet
from swe4dvar.forward.solvers import get_solver
from swe4dvar.forward import solvers as Solvers
from swe4dvar.utils.visualization import SolverVisualizer
from swe4dvar.utils.timing import Timer
from swe4dvar.utils import get_default_solver_params
from swe4dvar.utils.output_paths import FIGURES_DIR, DATA_DIR, ensure_output_dirs

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI


parser = argparse.ArgumentParser(
    description="Idealized Inlet Problem",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--dt", type=float, dest="dt", default=1200, help="time step in seconds"
)
parser.add_argument(
    "--final_time",
    type=float,
    dest="final_time",
    default=24 * 60 * 60 * 4,
    help="final time in seconds",
)
parser.add_argument(
    "--solver",
    dest="solver",
    type=str,
    default="DG",
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
    "--friction",
    dest="friction_law",
    type=str,
    default="quadratic",
    choices=["linear", "quadratic", "mannings"],
    help="friction law",
)
parser.add_argument(
    "--dramp",
    dest="dramp",
    type=float,
    default=2.0,
    help="duration of ramp function in days",
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
        num_time_steps = int(np.ceil(args.final_time / args.dt))

        # choose solution variable, either h or eta or flux
        sol_var = "h"

        # polynomial degree for each variable
        p_degree = [1, 1]

        problem = IdealizedInlet(
            dt=args.dt,
            nt=num_time_steps,
            xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
            friction_law=args.friction_law,
            solution_var=sol_var,
            dramp=args.dramp,
        )

        # time series output locations
        stations = np.array([[25000.5, 15000.5, 0.0], [25000.5, 0.0, 0.0]])
        h_b_offset = 14.0 - (9 / 20000) * stations[:, 1]

        # create solver object
        solver = get_solver(args.solver)(problem, args.theta, p_degree=p_degree)

        # Get solver parameters with MPI-aware defaults
        params = get_default_solver_params(
            rtol=1e-9,
            atol=1e-10,
            max_it=15,
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
            plot_name="Ideal_Inlet",
            save_state=False,
            adjoint_method=False,
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
            json_file = str(DATA_DIR / "newton_convergence_inlet.json")
            diagnostics.save_json(json_file)

        visualizer = SolverVisualizer(
            domain=solver.domain,
            V_scalar=solver.V_scalar,
            V_vel=solver.V_vel,
            problem=problem,
            verbose=False,
        )

        visualizer.plot_inlet_comparison(
            solver_vals=solver.vals,
            dt=args.dt,
            nt=num_time_steps,
            station_idx=0,
            adcirc_file="data/Ideal_inlet_adcirc_openboundary.csv",
            dgswem_file="data/DGSWEM_Ideal_inlet_adcirc.csv",
            scheme_name=args.solver.upper(),
            output_dir=str(FIGURES_DIR),
        )

        visualizer.print_saved_files(
            f"\nPlots saved:",
            f"  - {FIGURES_DIR}/inlet_height_{args.solver.upper()}.png",
            f"  - {FIGURES_DIR}/inlet_velocity_{args.solver.upper()}.png",
        )

# Print timing summary
Timer.print_summary(per_item_key="simulation", num_items=num_time_steps)

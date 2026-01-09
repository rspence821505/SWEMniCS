from swemnics.forward.problems import DamProblem
from swemnics.forward.solvers import get_solver
from swemnics import FrictionLaw
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import timeit
import argparse as ap
import os


def run_experiment(name, outdir=None, **kwargs):
    start = timeit.default_timer()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nx = ny = 100

    dt = 0.5
    # dt = 2
    t = 0
    t_f = 40
    # used in plotting
    Lx = 1000
    dam_height = 2.0

    nt = int(np.ceil(t_f / dt))
    print("Number of time steps", nt)
    # friction law either quadratic or linear
    fric_law = FrictionLaw.none
    # choose solution variable, either h or eta or flux
    sol_var = "h"

    prob = DamProblem(
        dt=dt,
        nt=nt,
        nx=nx,
        ny=ny,
        friction_law=fric_law,
        solution_var=sol_var,
        spherical=False,
    )
    p_degree = [1, 1]
    rel_toleran = 1e-5
    abs_toleran = 1e-6
    max_iter = 10
    relax_param = 1.0
    # time series output
    # time series output
    nx = 100
    stations = np.zeros((nx, 3))
    stations[:, 0] = np.linspace(0, 1000, nx)
    stations[:, 1] = 450
    # create solver object

    # cg
    theta = 1
    solver = get_solver(name)(prob, theta, p_degree=p_degree, **kwargs)

    name = name.upper()
    params = {
        "rtol": rel_toleran,
        "atol": abs_toleran,
        "max_it": max_iter,
        "relaxation_parameter": relax_param,
        "ksp_type": "gmres",
        "pc_type": "ilu",
    }  # ,"pc_factor_mat_solver_type":"mumps"}
    solver.time_loop(
        solver_parameters=params,
        stations=stations,
        plot_every=1,
        plot_name="dam_test_" + name,
    )

    # Save array for post processing
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
    outdir = "" if outdir is None else outdir + "/"
    np.savetxt(f"{outdir}{name}_p1_stations_h.csv", solver.vals[:, :, 0], delimiter=",")
    np.savetxt(
        f"{outdir}{name}_p1_stations_xvel.csv", solver.vals[:, :, 1], delimiter=","
    )
    np.savetxt(
        f"{outdir}{name}_p1_stations_yvel.csv", solver.vals[:, :, 2], delimiter=","
    )

    # Plot results using SolverVisualizer (MPI-aware, no rank check needed)
    from swemnics.utils.visualization import SolverVisualizer

    visualizer = SolverVisualizer(
        domain=solver.domain,
        V_scalar=solver.V_scalar,
        V_vel=solver.V_vel,
        problem=prob,
        verbose=False,
    )

    plt_nums = [0, 40, nt]
    visualizer.plot_dam_break(
        solver_vals=solver.vals,
        dt=dt,
        nt=nt,
        Lx=Lx,
        dam_height=dam_height,
        timesteps=plt_nums,
        scheme_name=name.upper(),
        output_dir=outdir.rstrip("/") if outdir else ".",
        analytical_solution_func=prob.get_analytic_solution,
    )

    visualizer.print_saved_files(
        f"\nPlots saved: {outdir}dam_height_{name.upper()}_order1_dt.png, {outdir}dam_velocity_{name.upper()}_order1_dt.png"
    )

    # Your statements here

    stop = timeit.default_timer()

    print("Time: ", stop - start)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("solver", choices=["cg", "supg", "dg", "dgcg"])
    args = parser.parse_args()
    run_experiment(args.solver)

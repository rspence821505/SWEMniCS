from swemnics.forward.problems import IdealizedInlet
from swemnics.forward import solvers as Solvers
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import timeit

start = timeit.default_timer()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# not used
nx = 20
ny = 5

# time step in seconds
dt = 1200  # 2400
# start time
t = 0
# final time in seconds
t_f = 24 * 60 * 60 * 4  # 24*11*60*60#*24*11*60*60
nt = int(np.ceil(t_f / dt))
print("nmber of time steps", nt)

# friction law either quadratic or linear
fric_law = "quadratic"
# choose solution variable, either h or eta or flux
sol_var = "h"
# duration of ramp function in days, same as in adcirc
dramp = 2.0

prob = IdealizedInlet(
    dt=dt,
    nt=nt,
    xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
    friction_law=fric_law,
    solution_var=sol_var,
    dramp=dramp,
)
rel_toleran = 1e-9
abs_toleran = 1e-10  # 1e-6 reccomended for SUPG, 1e-10 for DG
max_iter = 15
relax_param = 1.0
p_degree = [1, 1]
plot_int = 1  # np.ceil(3600/dt)
# time series output
stations = np.array([[25000.5, 15000.5, 0.0], [25000.5, 0.0, 0.0]])
h_b_offset = 14.0 - (9 / 20000) * stations[:, 1]
# stations = np.array([[25000.5,0.5,0.0]])
# create solver object
# cg
theta = 1
# solver = Solvers.CGImplicit(prob,theta)
# supg
# solver = Solvers.SUPGImplicit(prob,theta,p_degree=p_degree)
# dg
solver = Solvers.DGImplicit(prob, theta)
# dgcg
# solver = Solvers.DGCGImplicit(prob)
params = {
    "rtol": rel_toleran,
    "atol": abs_toleran,
    "max_it": max_iter,
    "relaxation_parameter": relax_param,
}  # , "ksp_type": "preonly", "pc_type": "lu","pc_factor_mat_solver_type":"mumps"}
solver.time_loop(
    solver_parameters=params,
    stations=stations,
    plot_every=plot_int,
    plot_name="Ideal_Inlet",
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

visualizer.print_saved_files(
    f"\n\nSolver DOF: {solver.u.x.array.shape}\n",
    f"Solver vals shape: {solver.vals.shape}\n",
)

visualizer.plot_inlet_comparison(
    solver_vals=solver.vals,
    dt=dt,
    nt=nt,
    station_idx=0,
    adcirc_file="data/Ideal_inlet_adcirc_openboundary.csv",
    dgswem_file="data/DGSWEM_Ideal_inlet_adcirc.csv",
    scheme_name="DG",
    output_dir=".",
)

visualizer.print_saved_files(
    f"\nPlots saved: inlet_height_DG.png, inlet_velocity_DG.png"
)

# Your statements here

stop = timeit.default_timer()

print("Time: ", stop - start)

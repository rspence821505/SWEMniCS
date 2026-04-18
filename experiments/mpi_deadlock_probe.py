#!/usr/bin/env python3
"""
Minimal MPI deadlock probe for idealized inlet 4D-Var.

Reproduces the exact Step 8 code path with minimal timesteps
and heavy instrumentation to localize where MPI hangs.

Usage:
  mpirun -np 2 python experiments/mpi_deadlock_probe.py
"""
from __future__ import annotations
import sys, os, time, gc
import numpy as np

os.environ.setdefault("CC", "/usr/bin/clang")

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def log(msg):
    sys.stdout.write(f"  [rank {rank}] {msg}\n")
    sys.stdout.flush()

log(f"START — MPI size={size}")

# ============================================================
# Phase 1: Build problem (same as idealized_inlet_da.py)
# ============================================================
from petsc4py import PETSc
from dolfinx import la

from swe4dvar.forward.problems import IdealizedInlet
from swe4dvar.forward.solvers import get_solver
from swe4dvar.physics.forcing import GriddedForcing
from swe4dvar.utils import get_default_solver_params
from swe4dvar.data_assimilation import (
    DiagonalCovariance,
    PointObservationOperator,
    create_cost_function,
)
from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper
from experiments.twin_experiment import (
    TwinExperiment, TwinExperimentConfig, ForwardModelWrapper,
)
from experiments.idealized_inlet_twin import (
    CartesianVortexConfig, generate_cartesian_vortex,
    write_cartesian_wind_hdf5, create_perturbed_config,
)

dt = 600.0
nt_ramp = 2   # MINIMAL ramp
nt_da = 2     # MINIMAL DA window
nt_total = nt_ramp + nt_da
times = np.arange(0, (nt_total + 1) * dt, dt)

log("building wind...")
vortex_cfg = CartesianVortexConfig(Vmax=20.0, Rmax=15000.0, ramp_time_s=nt_ramp * dt)
wind_dir = PROJECT_ROOT / "results" / "mpi_probe_wind"
if rank == 0:
    wind_dir.mkdir(parents=True, exist_ok=True)
comm.Barrier()

truth_file = wind_dir / "truth.h5"
pert_file = wind_dir / "perturbed.h5"
x_grid = np.linspace(-10000, 60000, 71)
y_grid = np.linspace(-30000, 50000, 81)

if not truth_file.exists() and rank == 0:
    wx, wy, p = generate_cartesian_vortex(vortex_cfg, x_grid, y_grid, times)
    write_cartesian_wind_hdf5(str(truth_file), x_grid, y_grid, times, wx, wy, p)
pert_cfg = create_perturbed_config(vortex_cfg, 10.0)
if not pert_file.exists() and rank == 0:
    wx, wy, p = generate_cartesian_vortex(pert_cfg, x_grid, y_grid, times)
    write_cartesian_wind_hdf5(str(pert_file), x_grid, y_grid, times, wx, wy, p)
comm.Barrier()
log("wind done")

# ============================================================
# Phase 2: Truth problem + ramp + trajectory
# ============================================================
solver_params = get_default_solver_params(
    rtol=1e-5, atol=1e-6, max_it=30, relaxation_parameter=0.7,
    comm=comm, error_if_not_converged=False, ksp_max_it=2000,
)
log(f"solver_params: ksp_type={solver_params.get('ksp_type')}, pc_type={solver_params.get('pc_type')}")

forcing_truth = GriddedForcing(str(truth_file), cartesian=True)
prob_truth = IdealizedInlet(
    dt=dt, nt=nt_total,
    xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
    friction_law="mannings", solution_var="h",
    dramp=nt_ramp * dt / 86400.0, forcing=forcing_truth,
)
solver_truth = get_solver("DG")(prob_truth, theta=1.0, p_degree=[1, 1])
state_size = solver_truth.V.dofmap.index_map.size_local * solver_truth.V.dofmap.index_map_bs
log(f"state_size (local) = {state_size}")

# Ramp
prob_truth.nt = nt_ramp
solver_truth.time_loop(
    solver_parameters=solver_params, stations=[], plot_every=9999,
    save_state=False, store_jacobians=False, enable_video=False,
)
t_da_start = prob_truth.t
ramp_end_state = solver_truth.u_n.x.array[:state_size].copy()
log(f"ramp done, t={t_da_start}")

# Truth trajectory
solver_truth.storage.clear()
prob_truth.nt = nt_da
solver_truth.time_loop(
    solver_parameters=solver_params, stations=[], plot_every=9999,
    save_state=True, store_jacobians=False, enable_video=False,
)
m_true_arr = ramp_end_state
truth_trajectory = []
for s in solver_truth.storage.saved_states:
    vec = la.create_petsc_vector(solver_truth.V.dofmap.index_map, solver_truth.V.dofmap.index_map_bs)
    vec.setArray(s[:state_size])
    vec.assemble()
    truth_trajectory.append(vec)
log(f"truth traj: {len(truth_trajectory)} states")

solver_truth.storage.clear()
del solver_truth, prob_truth, forcing_truth
gc.collect()
log("truth solver freed")

# ============================================================
# Phase 3: DA problem
# ============================================================
forcing_da = GriddedForcing(str(pert_file), cartesian=True)
prob_da = IdealizedInlet(
    dt=dt, nt=nt_da,
    xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
    friction_law="mannings", solution_var="h",
    dramp=nt_ramp * dt / 86400.0, forcing=forcing_da,
)
solver_da = get_solver("DG")(prob_da, theta=1.0, p_degree=[1, 1])
state_size_da = solver_da.V.dofmap.index_map.size_local * solver_da.V.dofmap.index_map_bs
log(f"DA state_size (local) = {state_size_da}")

solver_da.u_n.x.array[:state_size_da] = ramp_end_state[:state_size_da]
solver_da.u_n_old.x.array[:state_size_da] = ramp_end_state[:state_size_da]
solver_da.u.x.array[:state_size_da] = ramp_end_state[:state_size_da]
solver_da.u_n.x.scatter_forward()
solver_da.u_n_old.x.scatter_forward()
solver_da.u.x.scatter_forward()
prob_da.t = t_da_start
log("DA solver initialized")

# ============================================================
# Phase 4: Twin experiment setup
# ============================================================
config = TwinExperimentConfig(
    method="4dvar", obs_fraction=0.1, obs_frequency=1,
    obs_noise_level=0.01, interior_only=True,
    background_error_std=0.02, background_correlation_length=500.0,
    component_aware_cov=True, max_iterations=2, max_funcs=5,
    gradient_tolerance=1e-3, cost_tolerance=1e-4,
    use_bounds=True, h_min=0.01,
    verbose=(rank == 0), obs_seed=42, background_seed=123,
)
exp = TwinExperiment(
    problem=prob_da, solver=solver_da, config=config,
    solver_params=solver_params, comm=comm,
)
exp.truth_trajectory = truth_trajectory
m_true = la.create_petsc_vector(solver_da.V.dofmap.index_map, solver_da.V.dofmap.index_map_bs)
m_true.setArray(m_true_arr[:state_size_da])
m_true.assemble()
exp.m_true = m_true
exp.t_da_start = t_da_start
log("TwinExperiment created")

# Observations + background
log("calling _setup_observations...")
obs_points, obs_operator, obs_times = exp._setup_observations()
log(f"obs setup done: {obs_operator.get_num_observations()} points, times={obs_times}")

log("calling _generate_observations...")
exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
log(f"observations generated: {len(exp.observations)} times")

log("calling _setup_background...")
background_error = exp._setup_background()
log(f"background setup done: RMSE={background_error:.6f}")

log("calling _setup_covariances...")
B, R, B_lwme = exp._setup_covariances(obs_operator, obs_noise_stds)
log("covariances done")

# ============================================================
# Phase 5: Gradient smoother
# ============================================================
log("building gradient smoother...")
h_indices, u_indices, v_indices = exp._get_component_dof_indices(owned_only=True)
log(f"DOF indices: h={len(h_indices)}, u={len(u_indices)}, v={len(v_indices)}, total local={state_size_da}")
log(f"h_indices range: [{h_indices.min()}, {h_indices.max()}]")

smoothing_matrix = exp._build_smoothing_matrix(h_indices, 500.0)
log(f"smoothing matrix shape: {smoothing_matrix.shape}, nnz={smoothing_matrix.nnz}")

# Verify dimensions match
test_arr = np.zeros(state_size_da)
test_h = test_arr[h_indices]
log(f"test_h shape: {test_h.shape}, smoothing_matrix shape: {smoothing_matrix.shape}")
try:
    result = smoothing_matrix @ test_h
    log(f"smoother matmul OK: input {test_h.shape} -> output {result.shape}")
except Exception as e:
    log(f"SMOOTHER MATMUL FAILED: {e}")

def gradient_smoother(grad_array):
    smoothed = grad_array.copy()
    smoothed[h_indices] = smoothing_matrix @ grad_array[h_indices]
    smoothed[u_indices] = smoothing_matrix @ grad_array[u_indices]
    smoothed[v_indices] = smoothing_matrix @ grad_array[v_indices]
    return smoothed

# ============================================================
# Phase 6: Cost function
# ============================================================
log("building cost function...")
forward_model = ForwardModelWrapper(
    solver=solver_da, problem=prob_da,
    solver_params=solver_params, t_start=t_da_start,
)
cost_fn = create_cost_function(
    "4dvar",
    forward_model=forward_model,
    observation_operator=obs_operator,
    background_cov=B,
    observation_cov=R,
    m_background=exp.m_background,
    observations=exp.observations,
    obs_times=obs_times,
)
inner = cost_fn
while hasattr(inner, 'base_cost'):
    inner = inner.base_cost
inner.gradient_smoother = gradient_smoother
log("cost function ready")

# ============================================================
# Phase 7: DIRECT cost function test (bypass TAO)
# ============================================================
log("=== DIRECT VALUE_GRADIENT TEST (no TAO) ===")
comm.Barrier()
log("barrier passed, calling value_gradient...")

t0 = time.time()
try:
    cost_val, grad_vec = cost_fn.value_gradient(exp.m_background)
    gnorm = grad_vec.norm()
    log(f"value_gradient SUCCEEDED: cost={cost_val:.4f}, ||grad||={gnorm:.4e}, time={time.time()-t0:.1f}s")
    grad_vec.destroy()
except Exception as e:
    log(f"value_gradient EXCEPTION: {e}")

comm.Barrier()
log("post-value_gradient barrier passed")

# ============================================================
# Phase 8: TAO optimizer test
# ============================================================
log("=== TAO OPTIMIZER TEST ===")
lower = exp.m_background.duplicate()
upper = exp.m_background.duplicate()
lower_arr = lower.getArray()
upper_arr = upper.getArray()
lower_arr[h_indices] = 0.01
upper_arr[h_indices] = 1e10
lower_arr[u_indices] = -1e10
upper_arr[u_indices] = 1e10
lower_arr[v_indices] = -1e10
upper_arr[v_indices] = 1e10
lower.setArray(lower_arr)
lower.assemble()
upper.setArray(upper_arr)
upper.assemble()
log("bounds built")

optimizer = PETScTAOWrapper(
    cost_fn, tao_type="blmvm",
    lower_bounds=lower, upper_bounds=upper,
    options={
        "max_iterations": 2,
        "max_funcs": 5,
        "gradient_tolerance": 1e-3,
        "cost_tolerance": 1e-4,
        "verbose": True,
    },
)
log("calling optimizer.solve()...")
comm.Barrier()

t0 = time.time()
try:
    m_analysis = optimizer.solve(exp.m_background)
    log(f"optimizer.solve() SUCCEEDED in {time.time()-t0:.1f}s, evals={optimizer.n_func_evals}")
    m_analysis.destroy()
except Exception as e:
    log(f"optimizer.solve() EXCEPTION: {e}")

comm.Barrier()
log("=== ALL DONE ===")
lower.destroy()
upper.destroy()

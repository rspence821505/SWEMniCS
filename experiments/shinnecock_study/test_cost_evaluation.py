#!/usr/bin/env python3
"""
Diagnostic script to test if the cost function can be evaluated.

This helps debug the Phase 1 issue where the optimizer exits with 0 iterations
because the forward model fails when evaluating the cost at the perturbed background.
"""

import os
os.environ.setdefault("CC", "/usr/bin/clang")

import sys
from pathlib import Path
import numpy as np
import time

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mpi4py import MPI
from swe4dvar.forward.adcirc_problem import ADCIRCProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils import get_default_solver_params
from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ================================================================
# Use SHORT TEST configuration
# ================================================================
dt = 600.0
nt_ramp = 24  # 2h warmup
nt_da = 12    # 2h DA window
nt_total = nt_ramp + nt_da

obs_fraction = 0.5
obs_frequency = 2
obs_noise_level = 0.01
background_error_std = 0.02  # Test at 2% first
cov_inflation_factor = 2000.0
max_iterations = 10

if rank == 0:
    print("=" * 70)
    print("DIAGNOSTIC: Test Cost Function Evaluation")
    print("=" * 70)
    print(f"  Background perturbation: {background_error_std*100}%")
    print(f"  Inflation factor: {cov_inflation_factor}x")
    print("=" * 70)

# ================================================================
# Setup problem and solver
# ================================================================
prob = ADCIRCProblem(
    adios_file="data/shinnecock_inlet",
    spherical=True,
    solution_var="h",
    friction_law="mannings",
    wd=True,
    wd_alpha=1.5,
    dt=dt,
    bathy_adjustment=0,
    nt=nt_total,
    dramp=2.0,
)
solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])

warmup_params = get_default_solver_params(
    rtol=1e-5, atol=1e-6, max_it=10,
    relaxation_parameter=1.0,
    comm=comm, error_if_not_converged=True,
)

da_solver_params = get_default_solver_params(
    rtol=1e-5, atol=1e-6, max_it=15,
    relaxation_parameter=1.0,
    ksp_type="preonly", pc_type="lu",
    comm=comm, error_if_not_converged=True,
)

# ================================================================
# Warmup
# ================================================================
if rank == 0:
    print(f"\n[1/6] Running {nt_ramp}-step warmup...")
prob.nt = nt_ramp
solver.time_loop(
    solver_parameters=warmup_params,
    stations=[],
    plot_every=9999,
    save_state=False,
    store_jacobians=False,
    enable_video=False,
    monitor_progress=False,
)
t_da_start = prob.t

if rank == 0:
    print(f"  ✓ Warmup complete (t = {t_da_start:.0f}s)")

# ================================================================
# Setup TwinExperiment
# ================================================================
prob.nt = nt_da

config = TwinExperimentConfig(
    method="4dvar",
    obs_fraction=obs_fraction,
    obs_frequency=obs_frequency,
    obs_noise_level=obs_noise_level,
    background_error_std=background_error_std,
    max_iterations=max_iterations,
    gradient_tolerance=1e-6,
    cost_tolerance=1e-8,
    n_windows=1,
    perturb_friction=False,
    friction_scale_factor=1.0,
    use_bounds=True,
    h_min=0.01,
    interior_only=True,
    verbose=(rank == 0),
)

exp = TwinExperiment(
    problem=prob, solver=solver, config=config,
    solver_params=da_solver_params, comm=comm,
)

# ================================================================
# Generate truth
# ================================================================
if rank == 0:
    print(f"\n[2/6] Generating truth trajectory ({nt_da} steps)...")
exp._generate_truth()
if rank == 0:
    print(f"  ✓ Truth generated")

# ================================================================
# Setup observations and background
# ================================================================
if rank == 0:
    print(f"\n[3/6] Setting up observations and background...")
obs_points, obs_operator, obs_times = exp._setup_observations()
exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
background_error = exp._setup_background()
B, R, B_lwme = exp._setup_covariances(obs_operator, obs_noise_stds)

# Inflate B
if cov_inflation_factor != 1.0:
    B.diagonal.scale(cov_inflation_factor)
    B.inv_diagonal.scale(1.0 / cov_inflation_factor)

n_obs = obs_operator.get_num_observations()
if rank == 0:
    print(f"  ✓ Observations: {n_obs} points at times {obs_times}")
    print(f"  ✓ Background RMS error: {background_error:.6f}")

# ================================================================
# Create forward model and cost function
# ================================================================
if rank == 0:
    print(f"\n[4/6] Creating forward model and cost function...")
forward_model = exp._create_forward_model(t_start=t_da_start)
cost_function = exp._setup_cost_function(
    forward_model, obs_operator, B, R, obs_times, B_lwme=B_lwme
)
if rank == 0:
    print(f"  ✓ Cost function created")

# ================================================================
# TEST 1: Evaluate cost at TRUE state
# ================================================================
if rank == 0:
    print(f"\n[5/6] TEST 1: Evaluate cost at TRUE state...")
    print(f"  (Should be very small, ~0)")
try:
    cost_at_true = cost_function.value(exp.m_true)
    if rank == 0:
        if np.isfinite(cost_at_true):
            print(f"  ✓ Cost at true: {cost_at_true:.6e}")
        else:
            print(f"  ✗ Cost at true: {cost_at_true} (INFINITE/NAN!)")
except Exception as e:
    if rank == 0:
        print(f"  ✗ EXCEPTION: {e}")

# ================================================================
# TEST 2: Evaluate cost at BACKGROUND state
# ================================================================
if rank == 0:
    print(f"\n[6/6] TEST 2: Evaluate cost at BACKGROUND state...")
    print(f"  (Should be finite, J_b + J_o)")
try:
    t_start = time.time()
    cost_at_background = cost_function.value(exp.m_background)
    t_eval = time.time() - t_start

    if rank == 0:
        if np.isfinite(cost_at_background):
            print(f"  ✓ Cost at background: {cost_at_background:.6e} (eval time: {t_eval:.1f}s)")
        else:
            print(f"  ✗ Cost at background: {cost_at_background} (INFINITE/NAN!)")
            print(f"  → This is the bug! Forward model fails at perturbed background.")
            print(f"  → TAO sees cost=inf, sets grad=0, exits immediately.")
except Exception as e:
    if rank == 0:
        print(f"  ✗ EXCEPTION during cost evaluation: {e}")
        import traceback
        traceback.print_exc()

# ================================================================
# TEST 3: Try smaller perturbation
# ================================================================
if rank == 0:
    print(f"\n[BONUS] Testing smaller perturbation (1% instead of 2%)...")

# Recreate background with smaller perturbation
config_small = TwinExperimentConfig(
    method="4dvar",
    obs_fraction=obs_fraction,
    obs_frequency=obs_frequency,
    obs_noise_level=obs_noise_level,
    background_error_std=0.01,  # 1% instead of 2%
    max_iterations=max_iterations,
    gradient_tolerance=1e-6,
    cost_tolerance=1e-8,
    n_windows=1,
    perturb_friction=False,
    use_bounds=True,
    h_min=0.01,
    interior_only=True,
    verbose=False,
)

exp_small = TwinExperiment(
    problem=prob, solver=solver, config=config_small,
    solver_params=da_solver_params, comm=comm,
)
exp_small.truth_trajectory = exp.truth_trajectory
exp_small.m_true = exp.m_true.copy()
background_error_small = exp_small._setup_background()

try:
    cost_at_background_small = cost_function.value(exp_small.m_background)
    if rank == 0:
        if np.isfinite(cost_at_background_small):
            print(f"  ✓ Cost at 1% background: {cost_at_background_small:.6e}")
            print(f"  → Smaller perturbation works! Reduce background_error_std.")
        else:
            print(f"  ✗ Cost at 1% background: {cost_at_background_small} (still fails)")
except Exception as e:
    if rank == 0:
        print(f"  ✗ EXCEPTION with 1% perturbation: {e}")

if rank == 0:
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")

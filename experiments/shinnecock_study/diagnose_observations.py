#!/usr/bin/env python3
"""
Diagnose why observations don't match model predictions.

In inverse crime setup, observations come from truth run.
Running truth forward again should reproduce observations almost exactly.
Large mismatches indicate a problem.
"""

import os
os.environ.setdefault("CC", "/usr/bin/clang")

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mpi4py import MPI
from swe4dvar.forward.adcirc_problem import ADCIRCProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils import get_default_solver_params
from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Use same config as Phase 1
dt = 600.0
nt_ramp = 24
nt_da = 12
nt_total = nt_ramp + nt_da

if rank == 0:
    print("="*70)
    print("OBSERVATION DIAGNOSTIC")
    print("="*70)

# Setup problem
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

# Warmup
if rank == 0:
    print("\n[1] Running warmup...")
prob.nt = nt_ramp
solver.time_loop(
    solver_parameters=warmup_params,
    stations=[], plot_every=9999, save_state=False,
    store_jacobians=False, enable_video=False, monitor_progress=False,
)
t_da_start = prob.t

# Setup TwinExperiment
prob.nt = nt_da
config = TwinExperimentConfig(
    method="4dvar",
    obs_fraction=0.5,
    obs_frequency=2,
    obs_noise_level=0.01,
    background_error_std=0.01,
    max_iterations=10,
    gradient_tolerance=1e-6,
    cost_tolerance=1e-8,
    n_windows=1,
    perturb_friction=False,
    use_bounds=True,
    h_min=0.01,
    interior_only=True,
    verbose=(rank == 0),
)

exp = TwinExperiment(
    problem=prob, solver=solver, config=config,
    solver_params=da_solver_params, comm=comm,
)

# Generate truth RUN 1
if rank == 0:
    print("\n[2] Generating truth trajectory (RUN 1)...")
exp._generate_truth()

# Setup observations from truth RUN 1
if rank == 0:
    print("\n[3] Generating observations from truth...")
obs_points, obs_operator, obs_times = exp._setup_observations()
exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)

n_obs = obs_operator.get_num_observations()
if rank == 0:
    print(f"  Generated {len(exp.observations)} observation vectors")
    print(f"  Obs times: {obs_times}")
    print(f"  Points per time: {n_obs}")

# Now run forward model AGAIN from m_true (RUN 2)
if rank == 0:
    print("\n[4] Running forward model from m_true (RUN 2)...")
    print("  This should reproduce truth exactly (deterministic model)")

forward_model = exp._create_forward_model(t_start=t_da_start)
trajectory_run2, _ = forward_model.solve(exp.m_true, store_jacobians=True)

if rank == 0:
    print(f"  Generated {len(trajectory_run2)} states")

# Compare observations from RUN 1 vs predictions from RUN 2
if rank == 0:
    print("\n[5] Comparing observations (RUN 1) vs model predictions (RUN 2)...")
    print("  These should match almost exactly!")

mismatches = []
for i, k in enumerate(obs_times):
    # Observation from RUN 1
    y_obs = exp.observations[i]

    # Prediction from RUN 2
    u_k = trajectory_run2[k]
    H_u_k = obs_operator.forward(u_k)

    # Innovation
    innovation = H_u_k.duplicate()
    innovation.waxpy(-1.0, y_obs, H_u_k)  # H(u) - y

    innov_array = innovation.getArray()
    mismatch_rms = np.sqrt(np.mean(innov_array**2))
    mismatch_mean = np.mean(innov_array)
    mismatch_max = np.max(np.abs(innov_array))

    mismatches.append(mismatch_rms)

    if rank == 0:
        print(f"\n  Time {k} (t={k*dt/3600:.1f}h):")
        print(f"    Innovation RMS:  {mismatch_rms:.6f}")
        print(f"    Innovation mean: {mismatch_mean:.6f}")
        print(f"    Innovation max:  {mismatch_max:.6f}")

if rank == 0:
    avg_mismatch = np.mean(mismatches)
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC RESULT:")
    print(f"{'='*70}")
    print(f"  Average innovation RMS: {avg_mismatch:.6f}")
    print()
    if avg_mismatch < 0.001:
        print("  ✓ PASS: Observations match model predictions")
        print("  → Problem is elsewhere (gradient, cost function, optimizer)")
    elif avg_mismatch < 0.1:
        print("  ⚠ WARN: Small mismatch, possibly numerical")
        print("  → Check solver tolerances, random seeds")
    else:
        print("  ✗ FAIL: LARGE mismatch!")
        print("  → Observations don't match model predictions")
        print("  → Possible causes:")
        print("    1. Different solver settings (LU vs GMRES)")
        print("    2. Different random seeds")
        print("    3. Observation operator extracting wrong variable")
        print("    4. Truth trajectory corrupted")
    print(f"{'='*70}")

"""
Gradient direction diagnostic for Phase 1.

Evaluates J(m_b + alpha * d) for d = -gradient (descent) and d = +gradient (ascent)
at various step sizes to determine:
1. Whether the gradient is actually a descent direction
2. If there's a sign error in the adjoint
3. At what scale numerical noise dominates

Usage:
    python experiments/shinnecock_study/diagnose_gradient.py
"""

import sys
import os
import time
import numpy as np

os.environ.setdefault("CC", "/usr/bin/clang")

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "experiments"))

from mpi4py import MPI
from petsc4py import PETSc

from experiments.twin_experiment import (
    TwinExperiment, TwinExperimentConfig, ForwardModelWrapper,
)
from swe4dvar.forward.adcirc_problem import ADCIRCProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils import get_default_solver_params
from swe4dvar.data_assimilation.cost_functions import FourDVarCost

comm = MPI.COMM_WORLD
rank = comm.rank


def main():
    # ================================================================
    # Setup (same as run_phase_1)
    # ================================================================
    dt = 600.0
    nt_ramp = 288
    nt_da = 72
    nt_total = nt_ramp + nt_da
    obs_fraction = 0.05
    obs_frequency = 6
    obs_noise_level = 0.01
    background_error_std = 0.02

    adios_file = os.path.join(project_root, "data/shinnecock_inlet")

    if rank == 0:
        print("=" * 70)
        print("GRADIENT DIRECTION DIAGNOSTIC")
        print("=" * 70)

    prob = ADCIRCProblem(
        adios_file=adios_file,
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

    if rank == 0:
        print(f"\n--- Warm-up ({nt_ramp} steps) ---")

    prob.nt = nt_ramp
    solver.time_loop(
        solver_parameters=warmup_params,
        stations=[], plot_every=9999,
        save_state=False, store_jacobians=False,
        enable_video=False, monitor_progress=(rank == 0),
    )
    t_da_start = prob.t

    if rank == 0:
        print(f"  Warm-up complete, t = {t_da_start:.0f}s")

    # ================================================================
    # Setup twin experiment (truth, obs, background)
    # ================================================================
    prob.nt = nt_da

    config = TwinExperimentConfig(
        method="4dvar",
        obs_fraction=obs_fraction,
        obs_frequency=obs_frequency,
        obs_noise_level=obs_noise_level,
        background_error_std=background_error_std,
        max_iterations=1,
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

    if rank == 0:
        print("\n--- Generating truth ---")
    exp._generate_truth()

    if rank == 0:
        print("\n--- Setting up observations ---")
    obs_points, obs_operator, obs_times = exp._setup_observations()
    exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
    background_error = exp._setup_background()
    B, R, B_lwme = exp._setup_covariances(obs_operator, obs_noise_stds)

    if rank == 0:
        print(f"  Background RMS error: {background_error:.6f}")

    forward_model = exp._create_forward_model(t_start=t_da_start)

    # ================================================================
    # Create RAW cost function (no M^{-1} preconditioning, no bounds)
    # ================================================================
    if rank == 0:
        print("\n--- Creating RAW cost function (no preconditioning) ---")

    raw_cost = FourDVarCost(
        forward_model=forward_model,
        observation_operator=obs_operator,
        background_cov=B,
        observation_cov=R,
        m_background=exp.m_background,
        observations=exp.observations,
        obs_times=obs_times,
        comm=comm,
    )

    m_b = exp.m_background.copy()

    # ================================================================
    # Step 1: Evaluate cost at m_background
    # ================================================================
    if rank == 0:
        print("\n--- Evaluating J(m_b) ---")

    t0 = time.time()
    J_b, grad_b = raw_cost.value_gradient(m_b)
    t1 = time.time()

    gnorm = grad_b.norm()
    if rank == 0:
        print(f"  J(m_b) = {J_b:.10f}")
        print(f"  ||∇J(m_b)|| = {gnorm:.6e}")
        print(f"  Time: {t1-t0:.1f}s")

    # ================================================================
    # Step 2: Evaluate cost along gradient direction at tiny steps
    # ================================================================
    # Normalize gradient direction
    d = grad_b.copy()
    d.scale(1.0 / gnorm)  # Unit direction

    if rank == 0:
        print("\n" + "=" * 70)
        print("LINE SEARCH DIAGNOSTIC: J(m_b + alpha * d)")
        print("d = -∇J/||∇J|| (normalized descent direction)")
        print("=" * 70)
        print(f"{'alpha':>14} {'J(m_b - a*d)':>18} {'ΔJ (descent)':>15} {'J(m_b + a*d)':>18} {'ΔJ (ascent)':>15}")
        print("-" * 85)

    # Test various step sizes
    alphas = [1e-14, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

    for alpha in alphas:
        # Descent direction: m_b - alpha * d (should decrease cost)
        m_descent = m_b.copy()
        m_descent.axpy(-alpha, d)

        # Clear cache so forward model re-runs
        raw_cost.clear_cache()
        J_descent = raw_cost.value(m_descent)

        # Ascent direction: m_b + alpha * d (should increase cost)
        m_ascent = m_b.copy()
        m_ascent.axpy(+alpha, d)

        raw_cost.clear_cache()
        J_ascent = raw_cost.value(m_ascent)

        dJ_descent = J_descent - J_b
        dJ_ascent = J_ascent - J_b

        if rank == 0:
            print(f"{alpha:14.2e} {J_descent:18.10f} {dJ_descent:+15.8e} {J_ascent:18.10f} {dJ_ascent:+15.8e}")

        m_descent.destroy()
        m_ascent.destroy()

    # ================================================================
    # Step 3: Also test with UN-normalized gradient (raw scale)
    # ================================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("LINE SEARCH WITH RAW GRADIENT (unnormalized)")
        print("d_raw = ∇J(m_b)")
        print("=" * 70)
        print(f"{'alpha':>14} {'J(m_b - a*g)':>18} {'ΔJ':>15}")
        print("-" * 50)

    raw_alphas = [1e-20, 1e-18, 1e-16, 1e-14, 1e-12, 1e-10]

    for alpha in raw_alphas:
        m_test = m_b.copy()
        m_test.axpy(-alpha, grad_b)

        raw_cost.clear_cache()
        J_test = raw_cost.value(m_test)
        dJ = J_test - J_b

        if rank == 0:
            print(f"{alpha:14.2e} {J_test:18.10f} {dJ:+15.8e}")

        m_test.destroy()

    # ================================================================
    # Step 4: Check gradient components
    # ================================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("GRADIENT STATISTICS")
        print("=" * 70)

    grad_arr = grad_b.getArray()
    mb_arr = m_b.getArray()

    if rank == 0:
        print(f"  ||∇J|| = {gnorm:.6e}")
        print(f"  ∇J min = {grad_arr.min():.6e}")
        print(f"  ∇J max = {grad_arr.max():.6e}")
        print(f"  ∇J mean = {grad_arr.mean():.6e}")
        print(f"  ∇J std = {grad_arr.std():.6e}")
        print(f"  # positive components: {np.sum(grad_arr > 0)}")
        print(f"  # negative components: {np.sum(grad_arr < 0)}")
        print(f"  # zero components: {np.sum(grad_arr == 0)}")
        print(f"  m_b min = {mb_arr.min():.6e}")
        print(f"  m_b max = {mb_arr.max():.6e}")
        print(f"  m_b mean = {mb_arr.mean():.6e}")

    # ================================================================
    # Step 5: Check background term separately
    # ================================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("COST FUNCTION DECOMPOSITION AT m_b")
        print("=" * 70)

    # Background term should be 0 at m_b (since delta_m = 0)
    delta_m = m_b.duplicate()
    delta_m.waxpy(-1.0, exp.m_background, m_b)
    dm_norm = delta_m.norm()

    if rank == 0:
        print(f"  ||m_b - m_background|| = {dm_norm:.6e} (should be ~0)")
        print(f"  J_total(m_b) = {J_b:.10f}")
        print(f"  Background term should be 0 at m_b (it's the reference)")
        print(f"  So J_obs(m_b) ≈ {J_b:.10f}")

    # ================================================================
    # Step 6: Check gradient at m_true (where we verified it works)
    # ================================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("GRADIENT AT m_true (reference)")
        print("=" * 70)

    m_true = exp.m_truth.copy()
    raw_cost.clear_cache()
    J_true, grad_true = raw_cost.value_gradient(m_true)
    gnorm_true = grad_true.norm()

    if rank == 0:
        print(f"  J(m_true) = {J_true:.10f}")
        print(f"  ||∇J(m_true)|| = {gnorm_true:.6e}")

    # Direction from m_b to m_true should be descent
    dm_truth = m_true.copy()
    dm_truth.axpy(-1.0, m_b)  # dm = m_true - m_b
    dm_truth_norm = dm_truth.norm()

    # Directional derivative: <∇J(m_b), m_true - m_b>
    dir_deriv = grad_b.dot(dm_truth)

    if rank == 0:
        print(f"\n  ||m_true - m_b|| = {dm_truth_norm:.6e}")
        print(f"  <∇J(m_b), m_true - m_b> = {dir_deriv:.6e}")
        print(f"  (negative = direction toward m_true is descent)")
        print(f"  Cosine angle = {dir_deriv / (gnorm * dm_truth_norm):.6f}")

    # Test cost along m_b → m_true direction
    if rank == 0:
        print(f"\n  J along m_b → m_true:")
        print(f"  {'alpha':>8} {'J':>18}")
        print(f"  {'-'*30}")

    for alpha in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        m_test = m_b.copy()
        m_test.axpy(alpha, dm_truth)  # m_test = m_b + alpha*(m_true - m_b)

        raw_cost.clear_cache()
        J_test = raw_cost.value(m_test)

        if rank == 0:
            print(f"  {alpha:8.3f} {J_test:18.10f}")

        m_test.destroy()

    if rank == 0:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()

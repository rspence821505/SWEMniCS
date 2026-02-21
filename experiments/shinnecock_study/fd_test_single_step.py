"""
Single-step FD gradient test at m_background.

This tests with nt_da=1 to isolate the initial gradient computation
from the backward sweep. If this passes but multi-step fails, the
bug is in the backward sweep.

Test hierarchy:
  nt=1: Only terminal condition + initial gradient (no backward sweep)
  nt=2: One backward sweep step (backward Euler only)
  nt=3: BDF2 transition enters
"""

import os
os.environ.setdefault("CC", "/usr/bin/clang")

import sys
import time
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mpi4py import MPI
from petsc4py import PETSc

from swe4dvar.forward.adcirc_problem import ADCIRCProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils import get_default_solver_params
from swe4dvar.data_assimilation import FourDVarCost
from experiments.twin_experiment import (
    TwinExperiment, TwinExperimentConfig, ForwardModelWrapper,
)

comm = MPI.COMM_WORLD
rank = comm.rank


def run_fd_test(nt_da, obs_frequency=1):
    """Run FD gradient test with given DA window length."""
    dt = 600.0
    nt_ramp = 288
    nt_total = nt_ramp + nt_da
    obs_fraction = 0.05
    obs_noise_level = 0.01
    background_error_std = 0.02

    adios_file = str(project_root / "data" / "shinnecock_inlet")

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"FD TEST: nt_da={nt_da}, obs_frequency={obs_frequency}")
        print(f"{'='*70}", flush=True)

    prob = ADCIRCProblem(
        adios_file=adios_file, spherical=True, solution_var="h",
        friction_law="mannings", wd=True, wd_alpha=1.5,
        dt=dt, bathy_adjustment=0, nt=nt_total, dramp=2.0,
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])

    warmup_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=10,
        relaxation_parameter=1.0, comm=comm, error_if_not_converged=True,
    )
    da_solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=15, relaxation_parameter=1.0,
        ksp_type="preonly", pc_type="lu",
        comm=comm, error_if_not_converged=True,
    )

    n_dofs = len(solver.u.x.array)
    if rank == 0:
        print(f"  {n_dofs} DOFs", flush=True)

    # Warm-up
    if rank == 0:
        print(f"  Warm-up ({nt_ramp} steps)...", flush=True)
    prob.nt = nt_ramp
    solver.time_loop(
        solver_parameters=warmup_params, stations=[], plot_every=9999,
        save_state=False, store_jacobians=False, enable_video=False,
        monitor_progress=(rank == 0),
    )
    t_da_start = prob.t
    if rank == 0:
        print(f"  t_start = {t_da_start:.0f}s", flush=True)

    # Twin experiment setup
    prob.nt = nt_da
    config = TwinExperimentConfig(
        method="4dvar", obs_fraction=obs_fraction,
        obs_frequency=obs_frequency,
        obs_noise_level=obs_noise_level,
        background_error_std=background_error_std,
        max_iterations=1, gradient_tolerance=1e-6, cost_tolerance=1e-8,
        n_windows=1, perturb_friction=False, friction_scale_factor=1.0,
        use_bounds=True, h_min=0.01, interior_only=True, verbose=False,
    )
    exp = TwinExperiment(
        problem=prob, solver=solver, config=config,
        solver_params=da_solver_params, comm=comm,
    )

    if rank == 0:
        print("  Generating truth...", flush=True)
    exp._generate_truth()
    obs_points, obs_operator, obs_times = exp._setup_observations()
    exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
    background_error = exp._setup_background()
    B, R, _ = exp._setup_covariances(obs_operator, obs_noise_stds)

    forward_model = exp._create_forward_model(t_start=t_da_start)

    cost_fn = FourDVarCost(
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

    if rank == 0:
        print(f"  Obs times: {obs_times}")
        print(f"  Background RMS error: {background_error:.6f}", flush=True)

    # Compute gradient
    if rank == 0:
        print("  Computing J(m_b) and grad...", flush=True)

    t0 = time.time()
    J_b, grad_b = cost_fn.value_gradient(m_b)
    t_grad = time.time() - t0
    gnorm = grad_b.norm()
    grad_arr = grad_b.getArray().copy()

    if rank == 0:
        print(f"  J(m_b) = {J_b:.10f}")
        print(f"  ||grad|| = {gnorm:.6e}")
        print(f"  Time: {t_grad:.1f}s")
        print(f"  grad min={grad_arr.min():.4e}, max={grad_arr.max():.4e}", flush=True)

    # Select test DOFs (h-component)
    V = solver.V
    _, h_to_parent = V.sub(0).collapse()
    h_indices = sorted(set(h_to_parent))

    abs_grad = np.abs(grad_arr)
    h_grads = [(i, abs_grad[i]) for i in h_indices if abs_grad[i] > 0]
    h_grads.sort(key=lambda x: x[1], reverse=True)

    test_dofs = []
    if len(h_grads) > 0:
        test_dofs.append(h_grads[0][0])  # Largest
        test_dofs.append(h_grads[len(h_grads) // 2][0])  # Median
        test_dofs.append(h_grads[-1][0])  # Smallest non-zero

    if rank == 0:
        print(f"  Test DOFs (h component):")
        for d in test_dofs:
            print(f"    DOF {d}: grad = {grad_arr[d]:.6e}", flush=True)

    # FD gradient test
    epsilons = [1e-4, 1e-5, 1e-6]

    if rank == 0:
        print(f"\n{'DOF':>6} {'eps':>10} {'g_adj':>14} {'g_fd':>14} {'ratio':>10} {'status':>8}")
        print("-" * 70, flush=True)

    pass_count = 0
    fail_count = 0

    for dof_idx in test_dofs:
        for eps in epsilons:
            cost_fn.clear_cache()
            m_plus = m_b.copy()
            arr = m_plus.getArray()
            arr[dof_idx] += eps
            m_plus.setArray(arr)
            J_plus = cost_fn.value(m_plus)
            m_plus.destroy()

            cost_fn.clear_cache()
            m_minus = m_b.copy()
            arr = m_minus.getArray()
            arr[dof_idx] -= eps
            m_minus.setArray(arr)
            J_minus = cost_fn.value(m_minus)
            m_minus.destroy()

            g_adj = grad_arr[dof_idx]
            if np.isinf(J_plus) or np.isinf(J_minus):
                g_fd = float('inf')
                ratio = float('inf')
                status = "INF"
                fail_count += 1
            else:
                g_fd = (J_plus - J_minus) / (2 * eps)
                if abs(g_fd) > 1e-30:
                    ratio = g_adj / g_fd
                    if abs(ratio - 1.0) < 0.1:
                        status = "PASS"
                        pass_count += 1
                    else:
                        status = "FAIL"
                        fail_count += 1
                else:
                    ratio = float('inf')
                    status = "ZERO"
                    fail_count += 1

            if rank == 0:
                print(f"{dof_idx:>6} {eps:>10.0e} {g_adj:>14.6e} {g_fd:>14.6e} {ratio:>10.4f} {status:>8}", flush=True)

    total = pass_count + fail_count
    if rank == 0:
        print(f"\n  FD Test (nt={nt_da}): {pass_count}/{total} PASS\n", flush=True)

    return pass_count, total


def main():
    # Test hierarchy: single step, 2 steps, 3 steps (BDF2), then original 18 steps
    for nt_da in [1, 2, 3]:
        # obs_frequency=1 ensures observations every step
        run_fd_test(nt_da, obs_frequency=1)

    # Also test with the original 18 steps and obs_frequency=6
    run_fd_test(18, obs_frequency=6)


if __name__ == "__main__":
    main()

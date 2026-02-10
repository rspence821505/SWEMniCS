"""
Diagnose: check if the adjoint/FD ratio depends on perturbation size alpha.
If ratio is constant across alpha values → coefficient bug.
If ratio approaches 1.0 as alpha → 0 → nonlinearity effect.
"""
import os
os.environ.setdefault("CC", "/usr/bin/clang")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from petsc4py import PETSc


def run_diagnostic(nt=4, eps=1e-5):
    from swe4dvar.forward.problems import TidalProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params
    from experiments.twin_experiment import (
        TwinExperiment, TwinExperimentConfig, ForwardModelWrapper
    )

    problem = TidalProblem(nx=20, ny=10, dt=1800, nt=nt)
    solver = get_solver('DG')(problem, theta=0.5, p_degree=[1, 1])
    solver_params = get_default_solver_params()

    config = TwinExperimentConfig(
        method='4dvar',
        obs_fraction=0.5,
        obs_frequency=max(1, nt // 4),
        obs_noise_level=0.01,
        background_error_std=0.1,
        max_iterations=1,
        gradient_tolerance=1e-6,
        verbose=False
    )

    exp = TwinExperiment(
        problem=problem, solver=solver, config=config,
        solver_params=solver_params
    )

    exp._generate_truth()
    obs_points, obs_operator, obs_times = exp._setup_observations()
    observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
    exp._setup_background()
    B, R = exp._setup_covariances(obs_operator, obs_noise_stds)
    forward_model = exp._create_forward_model()

    from swe4dvar.data_assimilation.cost_functions import FourDVarCost

    cost_fn = FourDVarCost(
        forward_model=forward_model,
        observation_operator=obs_operator,
        background_cov=B,
        observation_cov=R,
        m_background=exp.m_background.copy(),
        observations=observations,
        obs_times=obs_times,
    )

    m_b = exp.m_background.copy()

    # Get adjoint gradient at m_b for direction
    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    _, grad_b = cost_fn.value_gradient(m_b)
    grad_b_arr = grad_b.getArray().copy()
    grad_b_norm = grad_b.norm()

    # DOF structure
    V = solver.V
    _, h_to_parent = V.sub(0).collapse()
    h_set = set(h_to_parent)
    h_list = sorted(h_set)
    test_dof = h_list[0]  # test with first h DOF

    print(f"Testing DOF {test_dof} (h), eps={eps}")
    print(f"||grad(m_b)|| = {grad_b_norm:.6e}")
    print()

    # Test various alpha values
    alphas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1]

    print(f"{'alpha':>10} {'||m-m_b||':>12} {'cost':>14} {'adj_grad':>14} {'fd_grad':>14} {'ratio':>10}")
    print(f"{'-' * 80}")

    direction = grad_b_arr / grad_b_norm if grad_b_norm > 0 else np.zeros_like(grad_b_arr)

    for alpha in alphas:
        m1 = m_b.copy()
        if alpha > 0:
            m1_arr = m1.getArray()
            m1_arr[:] = m1_arr - alpha * direction
            m1.setArray(m1_arr)

        delta_norm = 0.0
        if alpha > 0:
            diff = m1.copy()
            diff.axpy(-1.0, m_b)
            delta_norm = diff.norm()
            diff.destroy()

        # Adjoint gradient at m1
        cost_fn._control_hash = None
        cost_fn._trajectory = None
        cost_fn._jacobians = None
        cost_val, grad_m1 = cost_fn.value_gradient(m1)
        g_adj = grad_m1.getArray()[test_dof]

        # FD gradient at m1
        cost_fn._control_hash = None
        cost_fn._trajectory = None
        cost_fn._jacobians = None
        m_plus = m1.copy()
        arr = m_plus.getArray()
        arr[test_dof] += eps
        m_plus.setArray(arr)
        J_plus = cost_fn.value(m_plus)
        m_plus.destroy()

        cost_fn._control_hash = None
        cost_fn._trajectory = None
        cost_fn._jacobians = None
        m_minus = m1.copy()
        arr = m_minus.getArray()
        arr[test_dof] -= eps
        m_minus.setArray(arr)
        J_minus = cost_fn.value(m_minus)
        m_minus.destroy()

        g_fd = (J_plus - J_minus) / (2 * eps)
        ratio = g_adj / g_fd if abs(g_fd) > 1e-30 else float('inf')

        print(f"{alpha:>10.1e} {delta_norm:>12.6e} {cost_val:>14.6f} {g_adj:>14.6e} {g_fd:>14.6e} {ratio:>10.4f}")

        m1.destroy()

    # Also test: background-only gradient (no observation term)
    print(f"\n--- Background term only check ---")
    m1 = m_b.copy()
    m1_arr = m1.getArray()
    m1_arr[:] -= 0.01 * direction
    m1.setArray(m1_arr)

    delta_m = m1.copy()
    delta_m.axpy(-1.0, m_b)
    grad_back = B.apply_inverse(delta_m)
    g_back_dof = grad_back.getArray()[test_dof]

    # FD of background only
    def background_cost(m):
        dm = m.duplicate()
        dm.waxpy(-1.0, m_b, m)
        Binv_dm = B.apply_inverse(dm)
        val = 0.5 * dm.dot(Binv_dm)
        dm.destroy()
        Binv_dm.destroy()
        return val

    m_plus = m1.copy()
    arr = m_plus.getArray()
    arr[test_dof] += eps
    m_plus.setArray(arr)
    Jb_plus = background_cost(m_plus)
    m_plus.destroy()

    m_minus = m1.copy()
    arr = m_minus.getArray()
    arr[test_dof] -= eps
    m_minus.setArray(arr)
    Jb_minus = background_cost(m_minus)
    m_minus.destroy()

    g_back_fd = (Jb_plus - Jb_minus) / (2 * eps)
    ratio_back = g_back_dof / g_back_fd if abs(g_back_fd) > 1e-30 else float('inf')
    print(f"  Background term: adj={g_back_dof:.6e}, fd={g_back_fd:.6e}, ratio={ratio_back:.6f}")

    # Observation-only gradient (subtract background term from full gradient)
    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    _, grad_full = cost_fn.value_gradient(m1)
    g_obs_adj = grad_full.getArray()[test_dof] - g_back_dof

    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    m_plus = m1.copy()
    arr = m_plus.getArray()
    arr[test_dof] += eps
    m_plus.setArray(arr)
    J_full_plus = cost_fn.value(m_plus)
    m_plus.destroy()

    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    m_minus = m1.copy()
    arr = m_minus.getArray()
    arr[test_dof] -= eps
    m_minus.setArray(arr)
    J_full_minus = cost_fn.value(m_minus)
    m_minus.destroy()

    g_full_fd = (J_full_plus - J_full_minus) / (2 * eps)
    g_obs_fd = g_full_fd - g_back_fd
    ratio_obs = g_obs_adj / g_obs_fd if abs(g_obs_fd) > 1e-30 else float('inf')
    print(f"  Observation term: adj={g_obs_adj:.6e}, fd={g_obs_fd:.6e}, ratio={ratio_obs:.6f}")
    print(f"  Full gradient:    adj={grad_full.getArray()[test_dof]:.6e}, fd={g_full_fd:.6e}, ratio={grad_full.getArray()[test_dof]/g_full_fd:.6f}")

    m1.destroy()
    delta_m.destroy()

    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nt", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()
    run_diagnostic(nt=args.nt, eps=args.eps)

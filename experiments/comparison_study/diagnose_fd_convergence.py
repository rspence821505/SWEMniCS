"""
Diagnose: check FD convergence at a non-background point.
If adj/FD ratio changes with epsilon → FD inaccuracy.
If constant → adjoint error.
"""
import os
os.environ.setdefault("CC", "/usr/bin/clang")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from petsc4py import PETSc


def run_diagnostic(nt=4, alpha=0.01):
    from swe4dvar.forward.problems import TidalProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params
    from experiments.twin_experiment import (
        TwinExperiment, TwinExperimentConfig,
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

    # Get gradient at m_b for direction
    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    _, grad_b = cost_fn.value_gradient(m_b)
    grad_b_arr = grad_b.getArray().copy()
    grad_b_norm = grad_b.norm()

    # Create m_1 = m_b - alpha * g_b / ||g_b||
    m1 = m_b.copy()
    m1_arr = m1.getArray()
    m1_arr[:] -= alpha * (grad_b_arr / grad_b_norm)
    m1.setArray(m1_arr)

    print(f"Testing at m_1 = m_b - {alpha} * g/||g||")
    print(f"||m_1 - m_b|| = {alpha:.6e}")

    # Adjoint gradient at m_1
    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    cost_val, grad_m1 = cost_fn.value_gradient(m1)
    g_adj = grad_m1.getArray().copy()
    print(f"Cost(m_1) = {cost_val:.10f}")
    print(f"||grad(m_1)|| = {grad_m1.norm():.6e}")

    # DOF structure
    V = solver.V
    _, h_to_parent = V.sub(0).collapse()
    h_list = sorted(h_to_parent)
    test_dof = h_list[0]
    print(f"Test DOF: {test_dof} (h)")

    # Test FD with multiple epsilon values
    epsilons = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    print(f"\n{'epsilon':>12} {'FD_grad':>14} {'adj_grad':>14} {'ratio':>10} {'FD_2nd_order':>14}")
    print(f"{'-' * 70}")

    prev_fd = None
    for eps in epsilons:
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

        fd = (J_plus - J_minus) / (2 * eps)
        ratio = g_adj[test_dof] / fd if abs(fd) > 1e-30 else float('inf')

        # Richardson extrapolation check
        rich = ""
        if prev_fd is not None:
            rich = f"{(4*fd - prev_fd)/3:.6e}"

        prev_fd = fd

        print(f"{eps:>12.1e} {fd:>14.6e} {g_adj[test_dof]:>14.6e} {ratio:>10.4f} {rich:>14}")

    # Also test at m_b for comparison
    print(f"\nSame test at m_b:")
    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    _, grad_mb = cost_fn.value_gradient(m_b)
    g_adj_mb = grad_mb.getArray()[test_dof]

    print(f"{'epsilon':>12} {'FD_grad':>14} {'adj_grad':>14} {'ratio':>10}")
    print(f"{'-' * 56}")

    for eps in [1e-4, 1e-5, 1e-6]:
        cost_fn._control_hash = None
        cost_fn._trajectory = None
        cost_fn._jacobians = None
        m_plus = m_b.copy()
        arr = m_plus.getArray()
        arr[test_dof] += eps
        m_plus.setArray(arr)
        J_plus = cost_fn.value(m_plus)
        m_plus.destroy()

        cost_fn._control_hash = None
        cost_fn._trajectory = None
        cost_fn._jacobians = None
        m_minus = m_b.copy()
        arr = m_minus.getArray()
        arr[test_dof] -= eps
        m_minus.setArray(arr)
        J_minus = cost_fn.value(m_minus)
        m_minus.destroy()

        fd = (J_plus - J_minus) / (2 * eps)
        ratio = g_adj_mb / fd if abs(fd) > 1e-30 else float('inf')
        print(f"{eps:>12.1e} {fd:>14.6e} {g_adj_mb:>14.6e} {ratio:>10.4f}")

    m1.destroy()
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nt", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args()
    run_diagnostic(nt=args.nt, alpha=args.alpha)

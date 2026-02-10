"""
Debug: test 4D-Var with simple steepest descent + Armijo backtracking.
Bypasses L-BFGS to verify the gradient produces convergence.
"""
import os
os.environ.setdefault("CC", "/usr/bin/clang")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from petsc4py import PETSc


def run_debug(nt=4, max_iter=30):
    from swe4dvar.forward.problems import TidalProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params
    from experiments.twin_experiment import (
        TwinExperiment, TwinExperimentConfig,
        ZeroBoundaryGradientCost,
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
        verbose=False,
        interior_only=True,
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

    # NO M^{-1} preconditioning, just boundary zeroing
    boundary_dofs = exp._get_boundary_dofs()
    cost_fn_wrapped = ZeroBoundaryGradientCost(cost_fn, boundary_dofs)

    m_truth = exp.m_true.copy()
    m = exp.m_background.copy()

    # Steepest descent with Armijo backtracking
    c1 = 1e-4
    alpha_init = 1.0
    beta = 0.5  # backtracking factor

    print(f"Steepest descent with Armijo (c1={c1}, beta={beta})")
    print(f"{'iter':>4} {'cost':>14} {'||grad||':>12} {'alpha':>12} {'||m-m_t||':>12} {'reduction%':>10} {'LS_evals':>8}")
    print(f"{'-' * 80}")

    diff = m.copy()
    diff.axpy(-1.0, m_truth)
    err_bg = diff.norm()

    for it in range(max_iter):
        cost_fn_wrapped.clear_cache()
        cost_val, grad = cost_fn_wrapped.value_gradient(m)
        gnorm = grad.norm()
        g_arr = grad.getArray()

        # Compute directional derivative
        dir_deriv = -gnorm**2  # g^T (-g) = -||g||^2

        diff.waxpy(-1.0, m_truth, m)
        err = diff.norm()
        reduction = (1 - err / err_bg) * 100

        # Armijo backtracking
        alpha = alpha_init
        ls_evals = 0
        accepted = False

        for ls in range(30):
            m_trial = m.copy()
            m_trial_arr = m_trial.getArray()
            m_trial_arr[:] -= alpha * g_arr
            m_trial.setArray(m_trial_arr)

            cost_fn_wrapped.clear_cache()
            try:
                cost_trial = cost_fn_wrapped.value(m_trial)
            except Exception:
                cost_trial = float('inf')
            ls_evals += 1

            # Check Armijo condition
            if np.isfinite(cost_trial) and cost_trial <= cost_val + c1 * alpha * dir_deriv:
                # Accept step
                m.destroy()
                m = m_trial
                accepted = True
                break
            else:
                m_trial.destroy()
                alpha *= beta

        if not accepted:
            print(f"{it:>4} {cost_val:>14.6f} {gnorm:>12.6e} {'FAILED':>12} {err:>12.6e} {reduction:>10.2f}")
            print("Line search failed!")
            break

        print(f"{it:>4} {cost_val:>14.6f} {gnorm:>12.6e} {alpha:>12.6e} {err:>12.6e} {reduction:>10.2f} {ls_evals:>8}")

        # Check convergence
        if gnorm < 1e-6:
            print(f"\nConverged: ||grad|| = {gnorm:.6e} < 1e-6")
            break

        # Adaptive initial step: increase alpha for next iteration if step was accepted easily
        if ls_evals <= 2:
            alpha_init = min(alpha_init * 2.0, 100.0)
        elif ls_evals > 5:
            alpha_init = max(alpha * 2.0, 1e-6)

    # Final evaluation
    cost_fn_wrapped.clear_cache()
    cost_final, grad_final = cost_fn_wrapped.value_gradient(m)
    diff.waxpy(-1.0, m_truth, m)
    err_final = diff.norm()

    print(f"\nFinal: cost={cost_final:.6f}, ||grad||={grad_final.norm():.6e}, ||m-m_t||={err_final:.6e}")
    print(f"Error reduction: {(1 - err_final / err_bg) * 100:.2f}%")

    m.destroy()
    diff.destroy()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nt", type=int, default=4)
    parser.add_argument("--max_iter", type=int, default=30)
    args = parser.parse_args()
    run_debug(nt=args.nt, max_iter=args.max_iter)

"""
Debug TAO line search: run 4D-Var optimization with detailed TAO output.
"""
import os
os.environ.setdefault("CC", "/usr/bin/clang")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from petsc4py import PETSc

def run_debug(nt=4):
    from swe4dvar.forward.problems import TidalProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params
    from experiments.twin_experiment import (
        TwinExperiment, TwinExperimentConfig, ForwardModelWrapper,
        MassMatrixPreconditionedCost, ZeroBoundaryGradientCost,
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
        max_iterations=50,
        gradient_tolerance=1e-6,
        verbose=False,
        interior_only=True,
        use_bounds=True,
        h_min=0.01,
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

    # Apply wrappers
    M = forward_model.get_mass_matrix()
    cost_fn = MassMatrixPreconditionedCost(cost_fn, M)

    boundary_dofs = exp._get_boundary_dofs()
    cost_fn = ZeroBoundaryGradientCost(cost_fn, boundary_dofs)

    # Get initial gradient info
    m0 = exp.m_background.copy()
    cost_val, grad0 = cost_fn.value_gradient(m0)
    print(f"Initial cost = {cost_val:.6f}")
    print(f"||grad|| = {grad0.norm():.6e}")
    print(f"grad min/max = {grad0.getArray().min():.6e} / {grad0.getArray().max():.6e}")
    print()

    # Test 1: Simple gradient descent to verify cost decreases
    print("=" * 70)
    print("TEST 1: Manual gradient descent")
    print("=" * 70)
    alphas_test = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    for alpha in alphas_test:
        m_trial = m0.copy()
        m_trial_arr = m_trial.getArray()
        g_arr = grad0.getArray()
        m_trial_arr[:] -= alpha * g_arr
        m_trial.setArray(m_trial_arr)

        cost_fn.clear_cache()
        try:
            J_trial = cost_fn.value(m_trial)
        except Exception as e:
            J_trial = float('inf')
        dJ = J_trial - cost_val
        print(f"  alpha={alpha:<10.1e}  ||step||={alpha*grad0.norm():<12.6e}  J={J_trial:<14.6f}  dJ={dJ:<14.6e}")
        m_trial.destroy()

    # Test 2: TAO with verbose output using PETSc options
    print()
    print("=" * 70)
    print("TEST 2: TAO BLMVM with Armijo line search")
    print("=" * 70)

    from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper

    # Set PETSc options for detailed monitoring
    opts = PETSc.Options()
    opts.setValue("-tao_monitor", None)
    opts.setValue("-tao_ls_monitor", None)

    cost_fn.clear_cache()

    lower, upper = exp._create_physical_bounds()
    optimizer = PETScTAOWrapper(
        cost_fn,
        tao_type="blmvm",
        lower_bounds=lower,
        upper_bounds=upper,
        options={
            "max_iterations": 50,
            "gradient_tolerance": 1e-6,
            "cost_tolerance": 1e-8,
            "verbose": True,
            "line_search_type": "armijo",
            "line_search_max_funcs": 30,
        },
    )

    m_opt = optimizer.solve(m0.copy())
    print(f"\nResult: iterations={optimizer.iteration}, converged={optimizer.converged}")
    print(f"  func_evals={optimizer.n_func_evals}, grad_evals={optimizer.n_grad_evals}")

    # Test 3: TAO with lmvm (unbounded) to see if bounds cause issue
    print()
    print("=" * 70)
    print("TEST 3: TAO LMVM (unbounded) with Armijo")
    print("=" * 70)

    cost_fn.clear_cache()
    optimizer2 = PETScTAOWrapper(
        cost_fn,
        tao_type="lmvm",
        options={
            "max_iterations": 50,
            "gradient_tolerance": 1e-6,
            "cost_tolerance": 1e-8,
            "verbose": True,
            "line_search_type": "armijo",
            "line_search_max_funcs": 30,
        },
    )

    m_opt2 = optimizer2.solve(m0.copy())
    print(f"\nResult: iterations={optimizer2.iteration}, converged={optimizer2.converged}")
    print(f"  func_evals={optimizer2.n_func_evals}, grad_evals={optimizer2.n_grad_evals}")

    m0.destroy()
    m_opt.destroy()
    m_opt2.destroy()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nt", type=int, default=4)
    args = parser.parse_args()
    run_debug(nt=args.nt)

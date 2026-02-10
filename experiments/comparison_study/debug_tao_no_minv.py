"""
Debug TAO: test 4D-Var WITHOUT M^{-1} preconditioning.
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

    # NO M^{-1} preconditioning, just boundary zeroing
    boundary_dofs = exp._get_boundary_dofs()
    cost_fn_wrapped = ZeroBoundaryGradientCost(cost_fn, boundary_dofs)

    m0 = exp.m_background.copy()
    cost_val, grad0 = cost_fn_wrapped.value_gradient(m0)
    print(f"WITHOUT M^-1 preconditioning:")
    print(f"  cost = {cost_val:.6f}")
    print(f"  ||grad|| = {grad0.norm():.6e}")
    print()

    # Set PETSc options
    opts = PETSc.Options()
    opts.setValue("-tao_monitor", None)

    from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper

    # Test BLMVM
    print("=" * 70)
    print("BLMVM (bounded) without M^{-1}")
    print("=" * 70)

    cost_fn_wrapped.clear_cache()
    lower, upper = exp._create_physical_bounds()
    optimizer = PETScTAOWrapper(
        cost_fn_wrapped,
        tao_type="blmvm",
        lower_bounds=lower,
        upper_bounds=upper,
        options={
            "max_iterations": 50,
            "gradient_tolerance": 1e-4,  # Adjusted for un-preconditioned gradient
            "verbose": True,
            "line_search_type": "armijo",
            "line_search_max_funcs": 30,
        },
    )

    m_opt = optimizer.solve(m0.copy())
    print(f"\nResult: iterations={optimizer.iteration}, converged={optimizer.converged}")
    print(f"  func_evals={optimizer.n_func_evals}")

    # Compute error
    m_truth = exp.m_true.copy()
    m_bg = exp.m_background.copy()

    diff_opt = m_opt.copy()
    diff_opt.axpy(-1.0, m_truth)
    err_opt = diff_opt.norm()

    diff_bg = m_bg.copy()
    diff_bg.axpy(-1.0, m_truth)
    err_bg = diff_bg.norm()

    print(f"  ||m_bg - m_true|| = {err_bg:.6e}")
    print(f"  ||m_opt - m_true|| = {err_opt:.6e}")
    print(f"  Error reduction: {(1 - err_opt/err_bg)*100:.2f}%")

    m0.destroy()
    m_opt.destroy()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nt", type=int, default=4)
    args = parser.parse_args()
    run_debug(nt=args.nt)

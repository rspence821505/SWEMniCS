"""
Diagnose: test adjoint gradient with flux_formulation=False.

If ratio becomes 1.0, the bug is in how flux_formulation interacts with
the rest of the adjoint. If ratio stays wrong, the bug is elsewhere.
"""
import os
os.environ.setdefault("CC", "/usr/bin/clang")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from petsc4py import PETSc


def run_diagnostic(nt=2, alpha=0.01, eps=1e-5):
    from swe4dvar.forward.problems import TidalProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params
    from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig

    problem = TidalProblem(nx=20, ny=10, dt=1800, nt=nt)
    solver = get_solver('DG')(problem, theta=0.5, p_degree=[1, 1])
    solver_params = get_default_solver_params()

    config = TwinExperimentConfig(
        method='4dvar', obs_fraction=0.5,
        obs_frequency=max(1, nt // 4),
        obs_noise_level=0.01, background_error_std=0.1,
        max_iterations=1, gradient_tolerance=1e-6, verbose=False
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

    # Get gradient at m_b
    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    _, grad_b = cost_fn.value_gradient(m_b)
    grad_b_arr = grad_b.getArray().copy()
    grad_b_norm = grad_b.norm()

    # Create m_1
    m1 = m_b.copy()
    m1_arr = m1.getArray()
    m1_arr[:] -= alpha * (grad_b_arr / grad_b_norm)
    m1.setArray(m1_arr)

    V = solver.V
    _, h_to_parent = V.sub(0).collapse()
    h_list = sorted(h_to_parent)
    test_dof = h_list[0]

    print("=" * 80)
    print("Flux vs No-Flux Adjoint Diagnostic")
    print(f"nt={nt}, alpha={alpha}, eps={eps}, test DOF={test_dof}")
    print("=" * 80)

    # Test 1: Default adjoint (flux_formulation=True, auto-detected)
    print("\n[1] Default adjoint (flux_formulation=True, auto)")
    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    _, grad_flux = cost_fn.value_gradient(m1)
    g_flux = grad_flux.getArray()[test_dof]

    # Test 2: Adjoint with flux_formulation=False
    print("\n[2] Adjoint with flux_formulation=False")
    original_solve_adjoint = cost_fn._solve_adjoint

    def patched_solve_adjoint_noflux(trajectory, jacobians):
        from swe4dvar.adjoint.implicit_adjoint import ImplicitAdjointSolver
        from swe4dvar.utils import get_boundary_dofs
        obs_forcings = cost_fn._compute_observation_forcings(trajectory)
        vf = getattr(cost_fn.forward_model, 'var_form', None)
        if vf is None and hasattr(cost_fn.forward_model, 'solver'):
            vf = getattr(cost_fn.forward_model.solver, 'var_form', None)
        bc_dofs = None
        if hasattr(cost_fn.forward_model, 'solver') and hasattr(cost_fn.forward_model, 'problem'):
            bd = get_boundary_dofs(cost_fn.forward_model.solver.V,
                                   cost_fn.forward_model.problem.mesh)
            bc_dofs = set(bd.tolist())
        adj = ImplicitAdjointSolver(
            cost_fn.forward_model, trajectory, jacobians,
            cost_fn.forward_model.dt,
            variational_form=vf, bc_dof_indices=bc_dofs,
            flux_formulation=False,
        )
        terminal = trajectory[-1].duplicate()
        terminal.zeroEntries()
        return adj.solve(terminal, obs_forcings)

    cost_fn._solve_adjoint = patched_solve_adjoint_noflux
    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    _, grad_noflux = cost_fn.value_gradient(m1)
    g_noflux = grad_noflux.getArray()[test_dof]

    cost_fn._solve_adjoint = original_solve_adjoint

    # Test 3: FD gradient
    print("\n[3] FD gradient")

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

    print(f"\n{'Method':>20}  {'grad[{test_dof}]':>14}  {'ratio vs FD':>12}")
    print(f"{'-' * 50}")
    print(f"{'FD':>20}  {g_fd:>14.6e}  {1.0:>12.6f}")
    print(f"{'Adj (flux=True)':>20}  {g_flux:>14.6e}  {g_flux/g_fd:>12.6f}")
    print(f"{'Adj (flux=False)':>20}  {g_noflux:>14.6e}  {g_noflux/g_fd:>12.6f}")

    # Also test at m_b for comparison
    print(f"\nSame test at m_b:")
    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    _, grad_mb = cost_fn.value_gradient(m_b)
    g_mb = grad_mb.getArray()[test_dof]

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

    g_fd_mb = (J_plus - J_minus) / (2 * eps)
    print(f"{'FD':>20}  {g_fd_mb:>14.6e}  {1.0:>12.6f}")
    print(f"{'Adj (flux=True)':>20}  {g_mb:>14.6e}  {g_mb/g_fd_mb:>12.6f}")

    m1.destroy()
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nt", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()
    run_diagnostic(nt=args.nt, alpha=args.alpha, eps=args.eps)

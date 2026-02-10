"""
Diagnose adjoint gradient error: compare adjoint vs finite-difference gradient.
Tests with flux_formulation=True (default) and flux_formulation=False.
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

    # Replicate the setup from run()
    exp._generate_truth()
    obs_points, obs_operator, obs_times = exp._setup_observations()
    observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
    exp._setup_background()
    B, R = exp._setup_covariances(obs_operator, obs_noise_stds)
    forward_model = exp._create_forward_model()

    # Create BASE cost function (no M^{-1} wrapper)
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

    m = exp.m_background.copy()
    n_dofs = m.getSize()

    # Determine DOF structure
    V = solver.V
    _, h_to_parent = V.sub(0).collapse()
    h_set = set(h_to_parent)
    try:
        _, ux_to_parent = V.sub(1).sub(0).collapse()
        _, uy_to_parent = V.sub(1).sub(1).collapse()
        ux_set = set(ux_to_parent)
        uy_set = set(uy_to_parent)
    except Exception:
        ux_set = set()
        uy_set = set()

    h_list = sorted(h_set)
    ux_list = sorted(ux_set)
    uy_list = sorted(uy_set)

    # Pick test DOFs: 2 from h, 2 from ux, 2 from uy
    test_dofs = [h_list[0], h_list[len(h_list)//2],
                 ux_list[0], ux_list[len(ux_list)//2],
                 uy_list[0], uy_list[len(uy_list)//2]]

    print(f"=" * 80)
    print(f"Adjoint vs FD Gradient Diagnostic (nt={nt}, n_dofs={n_dofs}, eps={eps})")
    print(f"h DOFs: {len(h_set)}, range [{min(h_set)}, {max(h_set)}]")
    print(f"ux DOFs: {len(ux_set)}, range [{min(ux_set)}, {max(ux_set)}]")
    print(f"uy DOFs: {len(uy_set)}, range [{min(uy_set)}, {max(uy_set)}]")
    print(f"Test DOFs: {test_dofs}")
    print(f"obs_times: {obs_times}")
    print(f"=" * 80)

    # Step 1: Adjoint gradient (flux_formulation=True, default)
    print("\n[1] Adjoint gradient (flux_formulation=True)")
    cost_val, grad_flux = cost_fn.value_gradient(m)
    grad_flux_arr = grad_flux.getArray().copy()
    print(f"  Cost = {cost_val:.6f}, ||grad|| = {grad_flux.norm():.6e}")

    # Step 2: Adjoint gradient (flux_formulation=False)
    print("\n[2] Adjoint gradient (flux_formulation=False)")
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

    cost_val2, grad_noflux = cost_fn.value_gradient(m)
    grad_noflux_arr = grad_noflux.getArray().copy()
    print(f"  Cost = {cost_val2:.6f}, ||grad|| = {grad_noflux.norm():.6e}")

    cost_fn._solve_adjoint = original_solve_adjoint

    # Step 3: FD gradient for test DOFs
    print(f"\n[3] FD gradient for {len(test_dofs)} DOFs")
    fd_grad = np.zeros(len(test_dofs))

    for idx, dof in enumerate(test_dofs):
        cost_fn._control_hash = None
        cost_fn._trajectory = None
        cost_fn._jacobians = None
        m_plus = m.copy()
        arr = m_plus.getArray()
        arr[dof] += eps
        m_plus.setArray(arr)
        J_plus = cost_fn.value(m_plus)
        m_plus.destroy()

        cost_fn._control_hash = None
        cost_fn._trajectory = None
        cost_fn._jacobians = None
        m_minus = m.copy()
        arr = m_minus.getArray()
        arr[dof] -= eps
        m_minus.setArray(arr)
        J_minus = cost_fn.value(m_minus)
        m_minus.destroy()

        fd_grad[idx] = (J_plus - J_minus) / (2 * eps)
        print(f"  DOF {dof}: FD = {fd_grad[idx]:.6e}  (J+={J_plus:.10f}, J-={J_minus:.10f})")

    # Step 4: Comparison table at m_b
    print(f"\n{'=' * 100}")
    print(f"  AT m_b (background point)")
    print(f"{'DOF':>6} {'comp':>4}  {'adj(flux)':>14}  {'adj(noflux)':>14}  {'FD':>14}  {'flux/FD':>14}  {'noflux/FD':>14}")
    print(f"{'-' * 100}")

    for idx, dof in enumerate(test_dofs):
        comp = "h" if dof in h_set else ("ux" if dof in ux_set else "uy")
        g_f = grad_flux_arr[dof]
        g_nf = grad_noflux_arr[dof]
        g_fd = fd_grad[idx]

        r_f = g_f / g_fd if abs(g_fd) > 1e-30 else float('inf')
        r_nf = g_nf / g_fd if abs(g_fd) > 1e-30 else float('inf')

        print(f"{dof:>6} {comp:>4}  {g_f:>14.6e}  {g_nf:>14.6e}  {g_fd:>14.6e}  {r_f:>14.2f}  {r_nf:>14.2f}")

    # Step 5: Test at non-background point m_1
    print(f"\n{'=' * 100}")
    print(f"  TESTING AT NON-BACKGROUND POINT m_1")
    print(f"{'=' * 100}")

    # Create m_1 by perturbing m_b along the negative gradient direction
    alpha = 0.01  # small step
    m1 = m.copy()
    m1_arr = m1.getArray()
    grad_norm = grad_flux.norm()
    if grad_norm > 0:
        direction = grad_flux_arr / grad_norm
        m1_arr[:] = m1_arr - alpha * direction
    else:
        # Just add a small random perturbation
        rng = np.random.default_rng(42)
        m1_arr[:] = m1_arr + 0.001 * rng.standard_normal(len(m1_arr))
    m1.setArray(m1_arr)

    print(f"  alpha={alpha}, ||m1 - m_b|| = {np.linalg.norm(m1_arr - m.getArray()):.6e}")

    # Adjoint gradient at m_1
    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    cost_fn._solve_adjoint = original_solve_adjoint  # ensure default adjoint

    cost_val_m1, grad_m1 = cost_fn.value_gradient(m1)
    grad_m1_arr = grad_m1.getArray().copy()
    print(f"  Cost(m_1) = {cost_val_m1:.6f}, ||grad(m_1)|| = {grad_m1.norm():.6e}")

    # FD gradient at m_1
    print(f"\n  FD gradient at m_1 for {len(test_dofs)} DOFs")
    fd_grad_m1 = np.zeros(len(test_dofs))

    for idx, dof in enumerate(test_dofs):
        cost_fn._control_hash = None
        cost_fn._trajectory = None
        cost_fn._jacobians = None
        m_plus = m1.copy()
        arr = m_plus.getArray()
        arr[dof] += eps
        m_plus.setArray(arr)
        J_plus = cost_fn.value(m_plus)
        m_plus.destroy()

        cost_fn._control_hash = None
        cost_fn._trajectory = None
        cost_fn._jacobians = None
        m_minus = m1.copy()
        arr = m_minus.getArray()
        arr[dof] -= eps
        m_minus.setArray(arr)
        J_minus = cost_fn.value(m_minus)
        m_minus.destroy()

        fd_grad_m1[idx] = (J_plus - J_minus) / (2 * eps)

    # Comparison table at m_1
    print(f"\n{'DOF':>6} {'comp':>4}  {'adj':>14}  {'FD':>14}  {'ratio':>14}")
    print(f"{'-' * 70}")

    for idx, dof in enumerate(test_dofs):
        comp = "h" if dof in h_set else ("ux" if dof in ux_set else "uy")
        g_adj = grad_m1_arr[dof]
        g_fd = fd_grad_m1[idx]
        r = g_adj / g_fd if abs(g_fd) > 1e-30 else float('inf')
        print(f"{dof:>6} {comp:>4}  {g_adj:>14.6e}  {g_fd:>14.6e}  {r:>14.2f}")

    m1.destroy()

    print(f"\n{'=' * 100}")
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nt", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()
    run_diagnostic(nt=args.nt, eps=args.eps)

"""
Diagnose: compare adjoint coupling transpose action.

Properly test C^T * lambda vs c_n * M_Q^T * lambda,
which is the actual computation used in the adjoint.
"""
import os
os.environ.setdefault("CC", "/usr/bin/clang")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from petsc4py import PETSc
import ufl
from dolfinx import fem


def run_diagnostic(nt=2):
    from swe4dvar.forward.problems import TidalProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params

    problem = TidalProblem(nx=20, ny=10, dt=1800, nt=nt)
    solver = get_solver('DG')(problem, theta=0.5, p_degree=[1, 1])
    solver_params = get_default_solver_params()

    solver.init_fields()
    solver.init_weak_form()

    F = solver.F
    u = solver.u
    u_n = solver.u_n
    u_n_old = solver.u_n_old

    print("=" * 80)
    print("Adjoint Coupling Transpose Diagnostic")
    print("=" * 80)

    # Compute dF/du_n as bilinear form
    delta = ufl.TrialFunction(solver.V)
    dF_du_n = ufl.derivative(F, u_n, delta)

    # Set up twin experiment for trajectory
    from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig

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

    # Get gradient at m_b for perturbation direction
    m_b = exp.m_background.copy()
    _, grad_b = cost_fn.value_gradient(m_b)
    grad_b_arr = grad_b.getArray().copy()
    grad_b_norm = grad_b.norm()

    # Create m_1
    alpha = 0.01
    m1 = m_b.copy()
    m1_arr = m1.getArray()
    m1_arr[:] -= alpha * (grad_b_arr / grad_b_norm)
    m1.setArray(m1_arr)

    # Get trajectory at m_1
    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    trajectory, jacobians = cost_fn._run_forward_model(m1, store_jacobians=True)

    V = solver.V
    n_dofs = trajectory[0].getSize()
    u_owned = V.dofmap.index_map.size_local

    # Create a random lambda vector for testing transpose action
    rng = np.random.default_rng(42)
    lambda_vec = trajectory[0].duplicate()
    lambda_arr = rng.standard_normal(n_dofs) * 0.01
    lambda_vec.setArray(lambda_arr)

    print(f"\nTesting with random lambda vector (||lambda|| = {lambda_vec.norm():.6e})")

    for step in range(1, len(trajectory)):
        state_n = trajectory[step - 1]
        state_curr = trajectory[step]

        state_n_arr = state_n.getArray()
        state_curr_arr = state_curr.getArray()

        # Set solver state
        u_n.x.array[:u_owned] = state_n_arr[:u_owned]
        u_n.x.scatter_forward()
        u.x.array[:u_owned] = state_curr_arr[:u_owned]
        u.x.scatter_forward()

        if step >= 2:
            state_n_old = trajectory[step - 2]
            u_n_old.x.array[:u_owned] = state_n_old.getArray()[:u_owned]
            u_n_old.x.scatter_forward()

        # Set theta for this step
        if step <= 2:
            solver.theta1.value = 0
        else:
            solver.theta1.value = solver.theta

        try:
            # 1. Assemble UFL coupling matrix C = dF/du_n
            coupling_form = fem.form(dF_du_n)
            C = fem.petsc.assemble_matrix(coupling_form)
            C.assemble()

            # 2. Compute C^T * lambda (what the adjoint actually needs)
            ct_lambda = C.createVecLeft()
            C.multTranspose(lambda_vec, ct_lambda)

            # 3. Compute analytical: c_n * M_Q^T * lambda
            from swe4dvar.adjoint.implicit_adjoint import ImplicitAdjointSolver
            adj = ImplicitAdjointSolver(
                forward_model, trajectory, jacobians,
                forward_model.dt,
            )

            dt_val = problem.dt
            if step <= 2:
                c_n = -1.0 / dt_val  # BE coefficient
            else:
                c_n = -(solver.theta + 1.0) / dt_val  # BDF2 coefficient

            mq_t_lambda = adj._compute_flux_mass_transpose_action(state_n, lambda_vec)
            analytical = mq_t_lambda.duplicate()
            mq_t_lambda.copy(analytical)
            analytical.scale(c_n)

            # Compare
            diff = ct_lambda.duplicate()
            ct_lambda.copy(diff)
            diff.axpy(-1.0, analytical)

            ct_norm = ct_lambda.norm()
            ana_norm = analytical.norm()
            diff_norm = diff.norm()

            print(f"\n  Step {step}: theta1={solver.theta1.value}")
            print(f"    ||C^T * lambda|| = {ct_norm:.6e}")
            print(f"    ||c_n * M_Q^T * lambda|| = {ana_norm:.6e}")
            print(f"    ||difference|| = {diff_norm:.6e}")
            if ct_norm > 1e-30:
                print(f"    Relative diff: {diff_norm/ct_norm:.6e}")

            # Check component-wise for a few DOFs
            ct_arr = ct_lambda.getArray()
            ana_arr = analytical.getArray()
            diff_arr = diff.getArray()

            # Find DOFs with largest differences
            abs_diff = np.abs(diff_arr)
            top_dofs = np.argsort(abs_diff)[-5:][::-1]

            _, h_to_parent = V.sub(0).collapse()
            h_set = set(h_to_parent)
            try:
                _, ux_to_parent = V.sub(1).sub(0).collapse()
                _, uy_to_parent = V.sub(1).sub(1).collapse()
                ux_set = set(ux_to_parent)
                uy_set = set(uy_to_parent)
            except:
                ux_set = set()
                uy_set = set()

            print(f"    Top 5 DOFs with largest |difference|:")
            print(f"    {'DOF':>6} {'comp':>4} {'C^T*λ':>14} {'c*MQ^T*λ':>14} {'diff':>14} {'ratio':>10}")
            for d in top_dofs:
                comp = "h" if d in h_set else ("ux" if d in ux_set else "uy")
                r = ct_arr[d] / ana_arr[d] if abs(ana_arr[d]) > 1e-30 else float('inf')
                print(f"    {d:>6} {comp:>4} {ct_arr[d]:>14.6e} {ana_arr[d]:>14.6e} {diff_arr[d]:>14.6e} {r:>10.4f}")

            # Clean up
            ct_lambda.destroy()
            analytical.destroy()
            diff.destroy()
            C.destroy()
            mq_t_lambda.destroy()

        except Exception as e:
            print(f"    Error at step {step}: {e}")
            import traceback
            traceback.print_exc()

    lambda_vec.destroy()
    m1.destroy()
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nt", type=int, default=2)
    args = parser.parse_args()
    run_diagnostic(nt=args.nt)

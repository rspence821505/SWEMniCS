"""
Diagnose: compare analytical M_Q coupling vs UFL-derived coupling.

Compute ∂F/∂u_n using UFL (exact) and compare with the analytical formula
used in the adjoint: -(theta+1)/dt * M_Q(u_n).

If they disagree, the adjoint coupling formula is incomplete/wrong.
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

    # Initialize the solver (sets up weak form, etc.)
    solver.init_fields()
    solver.init_weak_form()

    # The residual form F
    F = solver.F
    u = solver.u
    u_n = solver.u_n
    u_n_old = solver.u_n_old

    print("=" * 80)
    print("Coupling Matrix Diagnostic")
    print("=" * 80)

    # Check if F depends on u_n by taking UFL derivative
    print("\n[1] Computing dF/du_n using UFL...")

    # Create a direction vector (as a TestFunction wouldn't work here,
    # we use ufl.derivative which gives a bilinear form in (v, delta_u_n))
    # Actually, derivative(F, u_n, delta) gives a linear form in v
    # But F is already a linear form in v (test function), so derivative gives a bilinear form

    # Let's try using action to get a matrix
    from ufl import TrialFunction
    delta = TrialFunction(solver.V)

    # dF/du_n is a bilinear form: derivative of the residual w.r.t. u_n
    dF_du_n = ufl.derivative(F, u_n, delta)
    print(f"  dF/du_n form computed (type: {type(dF_du_n)})")

    # Similarly, dF/du_n_old
    dF_du_n_old = ufl.derivative(F, u_n_old, delta)
    print(f"  dF/du_n_old form computed")

    # Run a forward solve to get a trajectory
    from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig

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

    # Run forward model at a perturbed point m_1
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

    # Create m_1 = m_b - alpha * g/||g||
    alpha = 0.01
    m1 = m_b.copy()
    m1_arr = m1.getArray()
    m1_arr[:] -= alpha * (grad_b_arr / grad_b_norm)
    m1.setArray(m1_arr)

    # Run forward model from m_1 to get trajectory
    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    trajectory, jacobians = cost_fn._run_forward_model(m1, store_jacobians=True)
    print(f"\n  Trajectory length: {len(trajectory)}, Jacobians: {len(jacobians)}")

    # Now test the coupling matrix at each timestep
    # We need to set self.u, self.u_n, self.u_n_old to the trajectory states
    # and then assemble dF/du_n

    V = solver.V
    n_dofs = trajectory[0].getSize()

    # Test DOFs
    _, h_to_parent = V.sub(0).collapse()
    h_list = sorted(h_to_parent)
    test_dof = h_list[0]

    print(f"\n[2] Comparing coupling matrices at each timestep")
    print(f"  Test DOF: {test_dof}")

    for step in range(1, len(trajectory)):
        # Set the solver state to match what was used at this timestep
        state_n = trajectory[step - 1]  # u^{n-1} (previous state)
        state_curr = trajectory[step]    # u^n (current state, converged)

        u_owned = V.dofmap.index_map.size_local
        state_n_arr = state_n.getArray()
        state_curr_arr = state_curr.getArray()

        # For step >= 2, we also need u_n_old
        if step >= 2:
            state_n_old = trajectory[step - 2]
            state_n_old_arr = state_n_old.getArray()
            u_n_old.x.array[:u_owned] = state_n_old_arr[:u_owned]
            u_n_old.x.scatter_forward()

        # Set u_n to the previous state
        u_n.x.array[:u_owned] = state_n_arr[:u_owned]
        u_n.x.scatter_forward()

        # Set u to the current (converged) state
        u.x.array[:u_owned] = state_curr_arr[:u_owned]
        u.x.scatter_forward()

        # Set theta1 to match forward solver behavior
        if step <= 2:
            solver.theta1.value = 0  # backward Euler for steps 1,2
        else:
            solver.theta1.value = solver.theta  # BDF2 for steps 3+

        # Assemble the EXACT coupling matrix dF/du_n using UFL
        try:
            coupling_form = fem.form(dF_du_n)
            coupling_matrix = fem.petsc.assemble_matrix(coupling_form)
            coupling_matrix.assemble()

            # Also get the coupling w.r.t. u_n_old
            if step >= 2:
                coupling_old_form = fem.form(dF_du_n_old)
                coupling_old_matrix = fem.petsc.assemble_matrix(coupling_old_form)
                coupling_old_matrix.assemble()
            else:
                coupling_old_matrix = None

            print(f"\n  Step {step}: theta1={solver.theta1.value}")
            print(f"    UFL coupling matrix ∂F/∂u_n assembled: {coupling_matrix.getSize()}")

            # Now compute the ANALYTICAL coupling using M_Q
            from swe4dvar.adjoint.implicit_adjoint import ImplicitAdjointSolver

            # Create adjoint solver just to use its M_Q computation
            adj = ImplicitAdjointSolver(
                forward_model, trajectory, jacobians,
                forward_model.dt,
            )

            # Create a test vector e_j (unit vector at test_dof)
            e_j = trajectory[0].duplicate()
            e_j.zeroEntries()
            e_j_arr = e_j.getArray()
            e_j_arr[test_dof] = 1.0
            e_j.setArray(e_j_arr)

            # UFL coupling action: (dF/du_n) * e_j
            ufl_result = coupling_matrix.createVecRight()
            coupling_matrix.mult(e_j, ufl_result)

            # M_Q^T action: M_Q^T(u_n) * e_j
            mq_result = adj._compute_flux_mass_transpose_action(state_n, e_j)

            # Expected coupling coefficient
            dt_val = problem.dt
            if step <= 2:
                # Backward Euler: coefficient of Qn is -1/dt
                c_n = -1.0 / dt_val
            else:
                # Theta-blended BDF2: coefficient of Qn is -(theta+1)/dt
                c_n = -(solver.theta + 1.0) / dt_val

            # Analytical coupling: c_n * M_Q(u_n) * e_j
            analytical_result = mq_result.duplicate()
            mq_result.copy(analytical_result)
            analytical_result.scale(c_n)

            # Compare UFL vs analytical
            ufl_arr = ufl_result.getArray()
            ana_arr = analytical_result.getArray()

            # Check a few DOFs
            print(f"    Coefficient c_n = {c_n:.6e}")
            print(f"    {'DOF':>6}  {'UFL(dF/du_n*e_j)':>18}  {'Analytical(-c*MQ*e_j)':>22}  {'Ratio':>10}  {'Diff':>14}")
            print(f"    {'-' * 80}")

            check_dofs = [h_list[0], h_list[len(h_list)//2]]
            # Also check some velocity DOFs
            try:
                _, ux_to_parent = V.sub(1).sub(0).collapse()
                ux_list = sorted(ux_to_parent)
                check_dofs.extend([ux_list[0], ux_list[len(ux_list)//2]])
            except:
                pass

            any_mismatch = False
            for d in check_dofs:
                u_val = ufl_arr[d]
                a_val = ana_arr[d]
                ratio = u_val / a_val if abs(a_val) > 1e-30 else float('inf')
                diff = abs(u_val - a_val)
                if abs(ratio - 1.0) > 0.001:
                    any_mismatch = True
                print(f"    {d:>6}  {u_val:>18.10e}  {a_val:>22.10e}  {ratio:>10.6f}  {diff:>14.6e}")

            # Norm comparison
            ufl_norm = ufl_result.norm()
            ana_norm = analytical_result.norm()
            diff_vec = ufl_result.duplicate()
            ufl_result.copy(diff_vec)
            diff_vec.axpy(-1.0, analytical_result)
            diff_norm = diff_vec.norm()
            print(f"    ||UFL|| = {ufl_norm:.6e}, ||Analytical|| = {ana_norm:.6e}, ||diff|| = {diff_norm:.6e}")
            if ufl_norm > 1e-30:
                print(f"    Relative diff: {diff_norm/ufl_norm:.6e}")

            if any_mismatch:
                print(f"    *** MISMATCH DETECTED ***")

            # Also check coupling w.r.t u_n_old
            if coupling_old_matrix is not None:
                ufl_old_result = coupling_old_matrix.createVecRight()
                coupling_old_matrix.mult(e_j, ufl_old_result)
                ufl_old_norm = ufl_old_result.norm()
                if ufl_old_norm > 1e-20:
                    print(f"\n    Coupling dF/du_n_old is NON-ZERO (norm = {ufl_old_norm:.6e})")
                else:
                    print(f"\n    Coupling dF/du_n_old is zero (norm = {ufl_old_norm:.6e})")
                ufl_old_result.destroy()

            # Clean up
            e_j.destroy()
            ufl_result.destroy()
            analytical_result.destroy()
            diff_vec.destroy()
            coupling_matrix.destroy()
            if coupling_old_matrix is not None:
                coupling_old_matrix.destroy()
            mq_result.destroy()

        except Exception as e:
            print(f"    Error at step {step}: {e}")
            import traceback
            traceback.print_exc()

    m1.destroy()
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nt", type=int, default=2)
    args = parser.parse_args()
    run_diagnostic(nt=args.nt)

"""
Diagnose: compute gradient using FORWARD SENSITIVITY method and compare with adjoint.

Forward sensitivity for DOF j:
  δu^1 = -J_1^{-1} * C_{10} * e_j
  δu^2 = -J_2^{-1} * C_{21} * δu^1
  ∂J/∂m_j = ∂J_b/∂m_j + obs_forcing_0[j] + obs_forcing_1 . δu^1 + obs_forcing_2 . δu^2

This is mathematically equivalent to the adjoint, so any difference reveals an implementation bug.
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
    B, R_cov = exp._setup_covariances(obs_operator, obs_noise_stds)
    forward_model = exp._create_forward_model()

    from swe4dvar.data_assimilation.cost_functions import FourDVarCost

    cost_fn = FourDVarCost(
        forward_model=forward_model,
        observation_operator=obs_operator,
        background_cov=B,
        observation_cov=R_cov,
        m_background=exp.m_background.copy(),
        observations=observations,
        obs_times=obs_times,
    )

    m_b = exp.m_background.copy()

    # Get gradient at m_b for perturbation direction
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

    # Step 1: Get adjoint gradient at m_1
    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    cost_val, grad_adj = cost_fn.value_gradient(m1)
    adj_arr = grad_adj.getArray().copy()

    # Step 2: Now run forward model again to get fresh trajectory/jacobians
    # (the value_gradient call above stored them, but let's get fresh ones)
    cost_fn._control_hash = None
    cost_fn._trajectory = None
    cost_fn._jacobians = None
    trajectory, jacobians = cost_fn._run_forward_model(m1, store_jacobians=True)

    # Deep copy trajectory and jacobians to avoid dangling references
    traj_copies = []
    for v in trajectory:
        vc = v.copy()
        traj_copies.append(vc)
    trajectory = traj_copies

    jac_copies = []
    for j in jacobians:
        jc = j.copy()
        jac_copies.append(jc)
    jacobians = jac_copies

    # Compute observation forcings at each time
    obs_forcings = cost_fn._compute_observation_forcings(trajectory)

    # Deep copy obs_forcings
    obs_forcing_copies = []
    for f in obs_forcings:
        if f is not None:
            obs_forcing_copies.append(f.copy())
        else:
            obs_forcing_copies.append(None)
    obs_forcings = obs_forcing_copies

    # Compute background gradient
    delta_m = m1.duplicate()
    delta_m.waxpy(-1.0, exp.m_background, m1)
    grad_bg = B.apply_inverse(delta_m)
    bg_arr = grad_bg.getArray().copy()

    V = solver.V
    _, h_to_parent = V.sub(0).collapse()
    h_list = sorted(h_to_parent)
    try:
        _, ux_to_parent = V.sub(1).sub(0).collapse()
        _, uy_to_parent = V.sub(1).sub(1).collapse()
        ux_list = sorted(ux_to_parent)
        uy_list = sorted(uy_to_parent)
    except:
        ux_list = []
        uy_list = []

    # Pick test DOFs
    test_dofs = [h_list[0], h_list[len(h_list)//2]]
    if ux_list:
        test_dofs.append(ux_list[0])
    if uy_list:
        test_dofs.append(uy_list[0])

    n_dofs = trajectory[0].getSize()
    dt_val = problem.dt

    print("=" * 100)
    print(f"Forward Sensitivity vs Adjoint Diagnostic (nt={nt}, alpha={alpha})")
    print(f"obs_times = {obs_times}")
    print(f"Number of obs_forcings: {sum(1 for f in obs_forcings if f is not None)}")
    print(f"Trajectory length: {len(trajectory)}, Jacobians: {len(jacobians)}")
    print("=" * 100)

    # Build coupling matrices using UFL dF/du_n
    import ufl
    from dolfinx import fem
    delta_trial = ufl.TrialFunction(solver.V)
    dF_du_n_form = ufl.derivative(solver.F, solver.u_n, delta_trial)
    coupling_form = fem.form(dF_du_n_form)

    u_owned = V.dofmap.index_map.size_local

    for test_dof in test_dofs:
        comp = "h" if test_dof in set(h_to_parent) else ("ux" if test_dof in set(ux_to_parent) else "uy")

        # Create perturbation direction e_j
        e_j = trajectory[0].duplicate()
        e_j.zeroEntries()
        e_j_arr = e_j.getArray()
        e_j_arr[test_dof] = 1.0
        e_j.setArray(e_j_arr)

        # FORWARD SENSITIVITY METHOD
        # For each step n, C_{n,n-1} = dF/du_n evaluated at (u^n, u^{n-1})
        # δu^n = -J_n^{-1} * C_{n,n-1} * δu^{n-1}  (where δu^0 = e_j)

        delta_u_prev = e_j.copy()  # δu^0 = e_j
        delta_u_list = [e_j.copy()]  # Store all δu^k

        for step in range(1, len(trajectory)):
            # Set solver state to match this timestep
            solver.u_n.x.array[:u_owned] = trajectory[step - 1].getArray()[:u_owned]
            solver.u_n.x.scatter_forward()
            solver.u.x.array[:u_owned] = trajectory[step].getArray()[:u_owned]
            solver.u.x.scatter_forward()

            # Set theta1 to match what forward solver used
            if step <= 2:
                solver.theta1.value = 0  # backward Euler for steps 1,2
            else:
                solver.theta1.value = solver.theta  # BDF2 for steps 3+

            # Assemble coupling matrix C_{step, step-1} = dF/du_n
            C = fem.petsc.assemble_matrix(coupling_form)
            C.assemble()

            # C * δu^{n-1}
            C_du = C.createVecRight()
            C.mult(delta_u_prev, C_du)

            # Solve J_n * δu^n = -C * δu^{n-1}
            J_n = jacobians[step - 1]  # jacobians[0] = J_1, jacobians[1] = J_2, etc.
            ksp = PETSc.KSP().create(J_n.getComm())
            ksp.setOperators(J_n)
            ksp.setType(PETSc.KSP.Type.PREONLY)
            ksp.getPC().setType(PETSc.PC.Type.LU)

            rhs = C_du.copy()
            rhs.scale(-1.0)

            delta_u = rhs.duplicate()
            ksp.solve(rhs, delta_u)

            delta_u_list.append(delta_u.copy())

            # Clean up
            delta_u_prev.destroy()
            delta_u_prev = delta_u.copy()
            ksp.destroy()
            C_du.destroy()
            rhs.destroy()
            delta_u.destroy()
            C.destroy()

        delta_u_prev.destroy()

        # Now compute gradient using forward sensitivities
        # dJ_o/dm_j = sum_k obs_forcing_k . δu^k
        fwd_obs_grad = 0.0
        for k in range(len(trajectory)):
            if obs_forcings[k] is not None:
                fwd_obs_grad += obs_forcings[k].dot(delta_u_list[k])

        fwd_total_grad = bg_arr[test_dof] + fwd_obs_grad

        # Also compute adjoint observation gradient
        adj_obs_grad = adj_arr[test_dof] - bg_arr[test_dof]

        # Also compute FD gradient
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

        fd_total_grad = (J_plus - J_minus) / (2 * eps)
        fd_obs_grad = fd_total_grad - bg_arr[test_dof]

        print(f"\nDOF {test_dof} ({comp}):")
        print(f"  Background gradient:  {bg_arr[test_dof]:>14.6e}")
        print(f"  Obs grad (FD):        {fd_obs_grad:>14.6e}")
        print(f"  Obs grad (FwdSens):   {fwd_obs_grad:>14.6e}  ratio to FD: {fwd_obs_grad/fd_obs_grad if abs(fd_obs_grad) > 1e-30 else float('inf'):.6f}")
        print(f"  Obs grad (Adjoint):   {adj_obs_grad:>14.6e}  ratio to FD: {adj_obs_grad/fd_obs_grad if abs(fd_obs_grad) > 1e-30 else float('inf'):.6f}")
        print(f"  Total grad (FD):      {fd_total_grad:>14.6e}")
        print(f"  Total grad (FwdSens): {fwd_total_grad:>14.6e}  ratio to FD: {fwd_total_grad/fd_total_grad if abs(fd_total_grad) > 1e-30 else float('inf'):.6f}")
        print(f"  Total grad (Adjoint): {adj_arr[test_dof]:>14.6e}  ratio to FD: {adj_arr[test_dof]/fd_total_grad if abs(fd_total_grad) > 1e-30 else float('inf'):.6f}")

        # δu norms
        for k, du in enumerate(delta_u_list):
            print(f"  ||δu^{k}|| = {du.norm():.6e}")

        # Check: is FwdSens == Adjoint?
        fwd_adj_diff = abs(fwd_obs_grad - adj_obs_grad)
        print(f"\n  |FwdSens - Adjoint| obs grad diff: {fwd_adj_diff:.6e}")

        # Clean up delta_u_list
        for du in delta_u_list:
            du.destroy()
        e_j.destroy()

    m1.destroy()
    delta_m.destroy()
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nt", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()
    run_diagnostic(nt=args.nt, alpha=args.alpha, eps=args.eps)

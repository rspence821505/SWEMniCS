#!/usr/bin/env python3
"""Debug the adjoint computation step by step."""

import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils import get_default_solver_params
from swe4dvar.adjoint.implicit_adjoint import ImplicitAdjointSolver
from swe4dvar.forward.variational_forms import SWEVariationalForm

from da_experiment_utils import (
    ForwardModelWrapper,
    generate_observation_points,
    generate_observations,
)
from swe4dvar.data_assimilation import (
    DiagonalCovariance,
    PointObservationOperator,
)


def debug_adjoint():
    """Debug adjoint computation with detailed output."""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Small problem
    nx, ny = 4, 2
    dt = 3600.0
    num_time_steps = 2
    obs_fraction = 0.5
    obs_noise_level = 0.01
    background_error_std = 0.1

    print("=" * 70)
    print("ADJOINT DEBUG")
    print("=" * 70)

    # Create problem and solver
    problem = TidalProblem(nx=nx, ny=ny, dt=dt, nt=num_time_steps)
    solver = get_solver("CG")(problem, theta=0.5, p_degree=[1, 1])
    solver_params = get_default_solver_params()

    # Generate truth trajectory
    print("\n[1] Running forward model for truth...")
    solver.time_loop(
        solver_parameters=solver_params,
        store_jacobians=True,
        save_state=True,
        enable_video=False,
    )

    truth_trajectory = [
        PETSc.Vec().createWithArray(s.copy(), comm=comm)
        for s in solver.storage.saved_states
    ]
    print(f"    States: {len(truth_trajectory)}")
    print(f"    State sizes: {[s.getSize() for s in truth_trajectory]}")

    # Setup observations
    obs_points = generate_observation_points(problem.mesh, fraction=obs_fraction)
    obs_op = PointObservationOperator(solver.V, obs_points, comm=comm)
    obs_times = list(range(1, len(truth_trajectory)))
    observations, _ = generate_observations(
        truth_trajectory, obs_op, obs_times,
        noise_level=obs_noise_level, seed=42
    )
    print(f"\n[2] Observations at times: {obs_times}")
    print(f"    Num obs points: {obs_op.get_num_observations()}")

    # Create perturbed background
    m_truth = truth_trajectory[0]
    m_background = m_truth.copy()
    np.random.seed(42)
    perturbation = background_error_std * np.random.randn(m_truth.getSize())
    m_background.setArray(m_background.getArray() + perturbation)

    # Create forward wrapper
    forward_wrapper = ForwardModelWrapper(solver, problem, solver_params)

    # Run forward with perturbed initial condition
    print("\n[3] Running forward model with perturbed IC...")
    m_test = m_background.copy()
    trajectory, jacobians = forward_wrapper.solve(m_test, store_jacobians=True)
    print(f"    Trajectory length: {len(trajectory)}")
    print(f"    Jacobians: {len(jacobians)}")
    for i, J in enumerate(jacobians):
        print(f"    Jacobian[{i}] size: {J.getSize()}, norm: {J.norm():.6e}")

    # Compute observation forcings manually
    print("\n[4] Computing observation forcings...")
    obs_forcings = [None] * len(trajectory)
    R = DiagonalCovariance(comm, obs_op.get_num_observations(), variance=obs_noise_level**2)

    for i, k in enumerate(obs_times):
        u_k = trajectory[k]
        Hu_k = obs_op.forward(u_k)
        d_k = Hu_k.duplicate()
        d_k.waxpy(-1.0, observations[i], Hu_k)  # d_k = Hu_k - y_k
        R_inv_d = R.apply_inverse(d_k)
        obs_forcings[k] = obs_op.adjoint(R_inv_d)
        print(f"    f_{k} = H^T R^{{-1}} (H u_{k} - y_{k})")
        print(f"        ||d_k|| = {d_k.norm():.6e}")
        print(f"        ||R^{{-1}} d_k|| = {R_inv_d.norm():.6e}")
        print(f"        ||f_k|| = {obs_forcings[k].norm():.6e}")

    # Create variational form for mass matrix
    var_form = SWEVariationalForm(solver.V, dt)

    # Create adjoint solver
    print("\n[5] Creating adjoint solver...")
    adjoint_solver = ImplicitAdjointSolver(
        forward_wrapper,
        trajectory,
        jacobians,
        dt,
        variational_form=var_form
    )

    # Get mass matrix
    M = adjoint_solver._get_mass_matrix()
    print(f"    Mass matrix norm: {M.norm():.6e}")

    # Terminal forcing (zero)
    terminal = trajectory[-1].duplicate()
    terminal.zeroEntries()

    # === MANUAL ADJOINT SOLVE ===
    print("\n[6] Manual adjoint solve...")
    num_steps = len(trajectory) - 1  # = 2

    # Step 1: Solve for λ_N (n=2)
    print(f"\n    Step 1: Solve for λ_{num_steps} (final time)")
    final_rhs = terminal.copy()
    if obs_forcings[-1] is not None:
        print(f"        RHS = terminal - f_{num_steps}")
        print(f"        ||terminal|| = {terminal.norm():.6e}")
        print(f"        ||f_{num_steps}|| = {obs_forcings[-1].norm():.6e}")
        final_rhs.axpy(-1.0, obs_forcings[-1])
    print(f"        ||RHS|| = {final_rhs.norm():.6e}")

    # Solve J_N^T λ_N = RHS
    J_N = jacobians[num_steps - 1]  # jacobians[1]
    print(f"        Using J_{num_steps} = jacobians[{num_steps-1}], norm = {J_N.norm():.6e}")

    ksp = PETSc.KSP().create(J_N.getComm())
    ksp.setOperators(J_N)
    ksp.setType(PETSc.KSP.Type.GMRES)
    ksp.getPC().setType(PETSc.PC.Type.NONE)
    ksp.setTolerances(rtol=1e-10, atol=1e-12)

    lambda_2 = final_rhs.duplicate()
    ksp.solveTranspose(final_rhs, lambda_2)
    print(f"        ||λ_{num_steps}|| = {lambda_2.norm():.6e}")

    # Check: J_N^T λ_N should equal RHS
    check = final_rhs.duplicate()
    J_N.multTranspose(lambda_2, check)
    check.axpy(-1.0, final_rhs)
    print(f"        ||J^T λ - RHS|| = {check.norm():.6e} (should be ~0)")

    # Step 2: Solve for λ_1 (n=1)
    print(f"\n    Step 2: Solve for λ_1")
    # For n=1: RHS = (4/(2dt)) M λ_2 - f_1
    c_next = 4.0 / (2.0 * dt)
    print(f"        Time coupling coeff c_{{n+1}} = {c_next:.6e}")

    forcing = lambda_2.duplicate()
    M.mult(lambda_2, forcing)
    forcing.scale(c_next)
    print(f"        ||(4/(2dt)) M λ_2|| = {forcing.norm():.6e}")

    if obs_forcings[1] is not None:
        print(f"        ||f_1|| = {obs_forcings[1].norm():.6e}")
        forcing.axpy(-1.0, obs_forcings[1])
    print(f"        ||RHS|| = {forcing.norm():.6e}")

    # Solve J_1^T λ_1 = RHS
    J_1 = jacobians[0]
    print(f"        Using J_1 = jacobians[0], norm = {J_1.norm():.6e}")

    ksp2 = PETSc.KSP().create(J_1.getComm())
    ksp2.setOperators(J_1)
    ksp2.setType(PETSc.KSP.Type.GMRES)
    ksp2.getPC().setType(PETSc.PC.Type.NONE)
    ksp2.setTolerances(rtol=1e-10, atol=1e-12)

    lambda_1 = forcing.duplicate()
    ksp2.solveTranspose(forcing, lambda_1)
    print(f"        ||λ_1|| = {lambda_1.norm():.6e}")

    # Step 3: Compute initial gradient
    print(f"\n    Step 3: Compute initial gradient ∂L/∂u_0")
    # ∂L/∂u_0 = -(1/dt) M λ_1 + (1/(2dt)) M λ_2 + f_0
    c_1 = -1.0 / dt
    c_2 = 1.0 / (2.0 * dt)
    print(f"        Coefficients: c_1 = {c_1:.6e}, c_2 = {c_2:.6e}")

    gradient_u0 = lambda_1.duplicate()
    M.mult(lambda_1, gradient_u0)
    gradient_u0.scale(c_1)
    print(f"        ||c_1 M λ_1|| = {gradient_u0.norm():.6e}")

    temp = lambda_2.duplicate()
    M.mult(lambda_2, temp)
    gradient_u0.axpy(c_2, temp)
    print(f"        After adding c_2 M λ_2: ||grad_u0|| = {gradient_u0.norm():.6e}")

    if obs_forcings[0] is not None:
        print(f"        Adding f_0...")
        gradient_u0.axpy(1.0, obs_forcings[0])
    print(f"        Final ||grad_u0|| = {gradient_u0.norm():.6e}")

    # Add background term
    print("\n[7] Adding background gradient...")
    B = DiagonalCovariance(comm, m_test.getSize(), variance=background_error_std**2)
    delta_m = m_test.duplicate()
    delta_m.waxpy(-1.0, m_background, m_test)  # delta_m = m - m_b
    grad_background = B.apply_inverse(delta_m)
    print(f"    ||m - m_b|| = {delta_m.norm():.6e}")
    print(f"    ||B^{{-1}}(m - m_b)|| = {grad_background.norm():.6e}")

    full_gradient = grad_background.duplicate()
    full_gradient.axpy(1.0, gradient_u0)
    print(f"    ||full gradient|| = {full_gradient.norm():.6e}")

    # Finite difference check
    print("\n[8] Finite difference check...")
    np.random.seed(123)
    direction = np.random.randn(m_test.getSize())
    direction /= np.linalg.norm(direction)
    d = PETSc.Vec().createWithArray(direction, comm=comm)

    analytic_dd = full_gradient.dot(d)
    print(f"    Analytic directional derivative: {analytic_dd:.6e}")

    # We need to compute J(m + eps*d) - J(m - eps*d)
    # First compute J(m_test)
    def compute_cost(m_vec):
        """Compute 4D-Var cost at m_vec."""
        traj, _ = forward_wrapper.solve(m_vec, store_jacobians=False)

        # Background term
        delta = m_vec.duplicate()
        delta.waxpy(-1.0, m_background, m_vec)
        B_inv_delta = B.apply_inverse(delta)
        J_b = 0.5 * delta.dot(B_inv_delta)

        # Observation term
        J_obs = 0.0
        for i, k in enumerate(obs_times):
            Hu = obs_op.forward(traj[k])
            d_k = Hu.duplicate()
            d_k.waxpy(-1.0, observations[i], Hu)
            R_inv_d = R.apply_inverse(d_k)
            J_obs += 0.5 * d_k.dot(R_inv_d)

        return J_b + J_obs

    J0 = compute_cost(m_test)
    print(f"    J(m) = {J0:.6e}")

    for eps in [1e-3, 1e-4, 1e-5]:
        m_plus = m_test.copy()
        m_plus.axpy(eps, d)
        m_minus = m_test.copy()
        m_minus.axpy(-eps, d)

        J_plus = compute_cost(m_plus)
        J_minus = compute_cost(m_minus)

        fd_dd = (J_plus - J_minus) / (2 * eps)
        error = abs(fd_dd - analytic_dd)
        rel_error = error / (abs(analytic_dd) + 1e-14)

        print(f"    eps={eps:.0e}: J+ = {J_plus:.6e}, J- = {J_minus:.6e}")
        print(f"               FD = {fd_dd:.6e}, error = {rel_error:.2%}")

    # Check if removing background term helps
    print("\n[9] Checking observation gradient only...")
    analytic_dd_obs = gradient_u0.dot(d)
    print(f"    Analytic observation-only directional derivative: {analytic_dd_obs:.6e}")

    def compute_obs_cost(m_vec):
        """Compute observation term only."""
        traj, _ = forward_wrapper.solve(m_vec, store_jacobians=False)
        J_obs = 0.0
        for i, k in enumerate(obs_times):
            Hu = obs_op.forward(traj[k])
            d_k = Hu.duplicate()
            d_k.waxpy(-1.0, observations[i], Hu)
            R_inv_d = R.apply_inverse(d_k)
            J_obs += 0.5 * d_k.dot(R_inv_d)
        return J_obs

    J0_obs = compute_obs_cost(m_test)
    print(f"    J_obs(m) = {J0_obs:.6e}")

    for eps in [1e-3, 1e-4]:
        m_plus = m_test.copy()
        m_plus.axpy(eps, d)
        m_minus = m_test.copy()
        m_minus.axpy(-eps, d)

        J_plus_obs = compute_obs_cost(m_plus)
        J_minus_obs = compute_obs_cost(m_minus)

        fd_dd_obs = (J_plus_obs - J_minus_obs) / (2 * eps)
        error = abs(fd_dd_obs - analytic_dd_obs)
        rel_error = error / (abs(analytic_dd_obs) + 1e-14)

        print(f"    eps={eps:.0e}: FD = {fd_dd_obs:.6e}, analytic = {analytic_dd_obs:.6e}, error = {rel_error:.2%}")


if __name__ == "__main__":
    debug_adjoint()

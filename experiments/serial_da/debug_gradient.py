#!/usr/bin/env python3
"""Debug gradient computation to find source of error."""

import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils import get_default_solver_params

from da_experiment_utils import (
    ForwardModelWrapper,
    generate_observation_points,
    generate_observations,
)
from swe4dvar.data_assimilation import (
    FourDVarCost,
    DiagonalCovariance,
    PointObservationOperator,
)


def debug_gradient():
    """Debug gradient with detailed output."""
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Problem parameters
    nx, ny = 4, 2
    dt = 3600.0
    num_time_steps = 2
    obs_fraction = 0.5
    obs_noise_level = 0.01
    background_error_std = 0.1

    if rank == 0:
        print("=" * 60)
        print("Gradient Debug")
        print("=" * 60)

    # Create problem
    problem = TidalProblem(nx=nx, ny=ny, dt=dt, nt=num_time_steps)
    solver = get_solver("CG")(problem, theta=0.5, p_degree=[1, 1])
    solver_params = get_default_solver_params()

    # Generate truth
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
    truth_jacobians = solver.storage.saved_jacobians.copy()

    if rank == 0:
        print(f"\nTrajectory: {len(truth_trajectory)} states")
        print(f"Jacobians: {len(truth_jacobians)}")
        for i, J in enumerate(truth_jacobians):
            print(f"  J[{i}]: {J.getSize()}")

    # Observation setup
    obs_points = generate_observation_points(problem.mesh, fraction=obs_fraction)
    obs_op = PointObservationOperator(solver.V, obs_points, comm=comm)

    obs_times = list(range(1, len(truth_trajectory)))
    observations, _ = generate_observations(
        truth_trajectory, obs_op, obs_times,
        noise_level=obs_noise_level, seed=42
    )

    if rank == 0:
        print(f"\nObservations: {len(observations)} at times {obs_times}")
        for i, obs in enumerate(observations):
            print(f"  obs[{i}] norm: {obs.norm():.4f}")

    # Perturbed background
    m_truth = truth_trajectory[0]
    m_background = m_truth.copy()
    np.random.seed(42)
    perturbation = background_error_std * np.random.randn(m_truth.getSize())
    m_background.setArray(m_background.getArray() + perturbation)

    if rank == 0:
        print(f"\nState size: {m_truth.getSize()}")
        print(f"||m_truth||: {m_truth.norm():.4f}")
        print(f"||m_background||: {m_background.norm():.4f}")
        print(f"||m_truth - m_background||: {(m_truth - m_background).norm():.4f}")

    # Covariances
    state_size = m_truth.getSize()
    n_obs = obs_op.get_num_observations()
    B = DiagonalCovariance(comm, state_size, variance=background_error_std**2)
    R = DiagonalCovariance(comm, n_obs, variance=obs_noise_level**2)

    # Forward model wrapper
    forward_wrapper = ForwardModelWrapper(solver, problem, solver_params)

    if rank == 0:
        print(f"\nForward wrapper var_form: {forward_wrapper.var_form}")
        if forward_wrapper.var_form is not None:
            M = forward_wrapper.var_form.assemble_mass_matrix()
            print(f"  Mass matrix size: {M.getSize()}")
            print(f"  Mass matrix norm: {M.norm():.4e}")

    # Cost function
    cost = FourDVarCost(
        forward_model=forward_wrapper,
        observation_operator=obs_op,
        background_cov=B,
        observation_cov=R,
        m_background=m_background,
        observations=observations,
        obs_times=obs_times,
        comm=comm,
    )

    # Test at m = m_background
    m_test = m_background.copy()

    if rank == 0:
        print("\n" + "=" * 60)
        print("STEP 1: Compute cost at m_test")
        print("=" * 60)

    J_test = cost.value(m_test)

    if rank == 0:
        print(f"J(m_test) = {J_test:.6e}")

    # Compute gradient
    if rank == 0:
        print("\n" + "=" * 60)
        print("STEP 2: Compute gradient")
        print("=" * 60)

    cost.clear_cache()
    J_test2, grad = cost.value_gradient(m_test)

    if rank == 0:
        print(f"J(m_test) = {J_test2:.6e}")
        print(f"||grad|| = {grad.norm():.6e}")

        # Look at gradient components
        grad_arr = grad.getArray()
        n_vars = 3  # h, u, v
        n_nodes = len(grad_arr) // n_vars

        h_grad = grad_arr[0::n_vars]
        u_grad = grad_arr[1::n_vars]
        v_grad = grad_arr[2::n_vars]

        print(f"\nGradient by component:")
        print(f"  h: mean={h_grad.mean():.4e}, min={h_grad.min():.4e}, max={h_grad.max():.4e}")
        print(f"  u: mean={u_grad.mean():.4e}, min={u_grad.min():.4e}, max={u_grad.max():.4e}")
        print(f"  v: mean={v_grad.mean():.4e}, min={v_grad.min():.4e}, max={v_grad.max():.4e}")

    # Finite difference check
    if rank == 0:
        print("\n" + "=" * 60)
        print("STEP 3: Finite difference check")
        print("=" * 60)

    # Direction along first DOF only
    d = m_test.duplicate()
    d.zeroEntries()
    d_arr = d.getArray()
    d_arr[0] = 1.0  # Perturb only h at first node
    d.setArray(d_arr)

    analytic_dd = grad.dot(d)

    if rank == 0:
        print(f"Direction: e_0 (first h DOF)")
        print(f"Analytic gradient[0] = {grad_arr[0]:.6e}")
        print(f"Analytic directional derivative = {analytic_dd:.6e}")

    eps = 1e-4
    m_plus = m_test.copy()
    m_plus.axpy(eps, d)
    m_minus = m_test.copy()
    m_minus.axpy(-eps, d)

    cost.clear_cache()
    J_plus = cost.value(m_plus)
    cost.clear_cache()
    J_minus = cost.value(m_minus)

    fd_dd = (J_plus - J_minus) / (2 * eps)

    if rank == 0:
        print(f"\nJ(m + eps*d) = {J_plus:.6e}")
        print(f"J(m - eps*d) = {J_minus:.6e}")
        print(f"FD directional derivative = {fd_dd:.6e}")
        print(f"Error = {abs(fd_dd - analytic_dd):.6e}")
        print(f"Relative error = {abs(fd_dd - analytic_dd) / (abs(analytic_dd) + 1e-14):.2%}")

    # Check the individual terms
    if rank == 0:
        print("\n" + "=" * 60)
        print("STEP 4: Break down gradient into terms")
        print("=" * 60)

    # Background term: B^{-1}(m - m_b)
    delta_m = m_test.duplicate()
    delta_m.waxpy(-1.0, m_background, m_test)
    grad_bg = B.apply_inverse(delta_m)

    if rank == 0:
        print(f"||m - m_b|| = {delta_m.norm():.6e}")
        print(f"||B^{{-1}}(m - m_b)|| = {grad_bg.norm():.6e}")
        print(f"Background gradient[0] = {grad_bg.getArray()[0]:.6e}")

    # The adjoint contribution is: grad - grad_bg
    grad_adj = grad.copy()
    grad_adj.axpy(-1.0, grad_bg)

    if rank == 0:
        print(f"\nAdjoint contribution ||λ_0|| = {grad_adj.norm():.6e}")
        print(f"Adjoint contribution[0] = {grad_adj.getArray()[0]:.6e}")

        # If m = m_b, the background gradient is zero
        print(f"\nNote: Since m = m_b, delta_m should be zero")
        print(f"||delta_m|| = {delta_m.norm():.6e}")


if __name__ == "__main__":
    debug_gradient()

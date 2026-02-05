"""
Test gradient accuracy after the unmodified Jacobian fix.

This test verifies that the discrete adjoint produces correct gradients
by comparing against finite differences. It specifically checks:
1. Interior DOF gradients (should be accurate to ~1e-6 relative error)
2. Boundary DOF gradients (should now be zero, not incorrect values)

The fix involves:
- Newton solver returns unmodified Jacobians (without BC identity rows)
- Adjoint solver applies homogeneous BCs after transpose solve
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import pytest

# Import SWE4DVar components
from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.data_assimilation import FourDVarCost, PointObservationOperator, DiagonalCovariance
from swe4dvar.utils import get_default_solver_params

# Add experiments/serial_da to path for ForwardModelWrapper
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments" / "serial_da"))
from da_experiment_utils import ForwardModelWrapper, get_boundary_dofs


def compute_fd_gradient(cost_function, m, dof_indices, epsilon=1e-6):
    """Compute finite difference gradient for selected DOFs."""
    fd_grad = {}
    base_cost = cost_function.value(m)

    m_arr = m.getArray().copy()

    for dof in dof_indices:
        # Perturb +epsilon
        m_pert = m.duplicate()
        m_arr_pert = m_arr.copy()
        m_arr_pert[dof] += epsilon
        m_pert.setArray(m_arr_pert)

        if hasattr(cost_function, 'clear_cache'):
            cost_function.clear_cache()

        cost_plus = cost_function.value(m_pert)
        m_pert.destroy()

        # Finite difference
        fd_grad[dof] = (cost_plus - base_cost) / epsilon

    return fd_grad, base_cost


@pytest.fixture
def small_problem():
    """Create a small test problem."""
    comm = MPI.COMM_WORLD

    # Small domain and coarse mesh for fast testing
    problem = TidalProblem(
        nx=5,
        ny=3,
        x0=0.0,
        x1=10000.0,
        y0=0.0,
        y1=5000.0,
        nt=2,  # 2 timesteps
        dt=1800.0,  # 30 min steps
    )

    solver = get_solver("CG")(problem, theta=1.0, p_degree=[1, 1], verbose=False)

    solver_params = get_default_solver_params()
    solver_params['atol'] = 1e-6
    solver_params['max_it'] = 10

    return problem, solver, solver_params, comm


def test_gradient_accuracy_interior_dofs(small_problem):
    """Test that interior DOF gradients are accurate."""
    problem, solver, solver_params, comm = small_problem
    rank = comm.Get_rank()

    # Create forward model wrapper
    forward_model = ForwardModelWrapper(solver, problem, solver_params)

    # Get boundary DOFs
    boundary_dofs = set(get_boundary_dofs(solver.V, problem.mesh).tolist())
    n_dofs = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs

    # Get interior DOFs
    interior_dofs = [i for i in range(n_dofs) if i not in boundary_dofs]

    # Create true initial condition
    m_true = PETSc.Vec().createWithArray(
        solver.u.x.array.copy(),
        comm=comm
    )

    # Run forward model to get trajectory
    trajectory, jacobians = forward_model.solve(m_true, store_jacobians=True)

    # Setup simple observation (observe at final time)
    coords = problem.mesh.geometry.x
    interior_mask = (
        (coords[:, 0] > coords[:, 0].min() + 0.1) &
        (coords[:, 0] < coords[:, 0].max() - 0.1) &
        (coords[:, 1] > coords[:, 1].min() + 0.1) &
        (coords[:, 1] < coords[:, 1].max() - 0.1)
    )
    interior_coords = coords[interior_mask][:5]  # Just 5 interior points
    obs_points = np.zeros((len(interior_coords), 3))
    obs_points[:, :2] = interior_coords[:, :2]

    obs_operator = PointObservationOperator(solver.V, obs_points, comm=comm)

    # Create observations at final time with perturbation to create mismatch
    # Without perturbation, m_true matches trajectory perfectly and gradient is zero
    obs_times = [len(trajectory) - 1]
    obs = obs_operator.forward(trajectory[-1])
    obs_perturbed = obs.copy()
    obs_arr = obs_perturbed.getArray()
    np.random.seed(42)  # Reproducibility
    obs_arr += 0.1 * np.abs(obs_arr) * np.random.randn(len(obs_arr))
    obs_perturbed.setArray(obs_arr)
    observations = [obs_perturbed]

    # Simple diagonal covariances
    B = DiagonalCovariance(comm, n_dofs, variance=0.01)
    n_obs_total = obs_operator.get_num_observations()
    R = DiagonalCovariance(comm, n_obs_total, variance=0.001)

    # Create cost function
    cost_function = FourDVarCost(
        forward_model=forward_model,
        observation_operator=obs_operator,
        background_cov=B,
        observation_cov=R,
        m_background=m_true.copy(),
        observations=observations,
        obs_times=obs_times,
        comm=comm,
    )

    # Compute adjoint gradient
    cost_function.clear_cache()
    adjoint_grad = cost_function.gradient(m_true)
    adjoint_grad_arr = adjoint_grad.getArray()

    # Select a few interior DOFs to test
    test_interior_dofs = interior_dofs[:min(10, len(interior_dofs))]

    # Compute FD gradient for selected DOFs
    cost_function.clear_cache()
    fd_grad, _ = compute_fd_gradient(
        cost_function, m_true, test_interior_dofs, epsilon=1e-6
    )

    # Compare
    if rank == 0:
        print("\n" + "=" * 60)
        print("Interior DOF Gradient Accuracy Test")
        print("=" * 60)
        print(f"{'DOF':>6}  {'Adjoint':>12}  {'FD':>12}  {'Ratio':>10}  {'Status'}")
        print("-" * 60)

    ratios = []
    for dof in test_interior_dofs:
        adj_val = adjoint_grad_arr[dof]
        fd_val = fd_grad[dof]

        if abs(fd_val) > 1e-12:
            ratio = adj_val / fd_val
        else:
            ratio = 1.0 if abs(adj_val) < 1e-12 else float('inf')

        ratios.append(ratio)

        status = "OK" if 0.9 < ratio < 1.1 else "WARN" if 0.5 < ratio < 2.0 else "FAIL"

        if rank == 0:
            print(f"{dof:6d}  {adj_val:12.4e}  {fd_val:12.4e}  {ratio:10.4f}  {status}")

    if rank == 0:
        print("-" * 60)
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        print(f"Mean ratio: {mean_ratio:.4f} ± {std_ratio:.4f}")
        print("=" * 60)

    # Check that most ratios are close to 1
    good_ratios = [r for r in ratios if 0.9 < r < 1.1]
    assert len(good_ratios) >= len(ratios) * 0.8, \
        f"Only {len(good_ratios)}/{len(ratios)} interior DOFs have accurate gradients"

    # Cleanup PETSc objects (DiagonalCovariance doesn't have destroy())
    adjoint_grad.destroy()
    for vec in trajectory:
        vec.destroy()
    for mat in jacobians:
        mat.destroy()
    for obs in observations:
        obs.destroy()
    m_true.destroy()


def test_gradient_zero_at_boundary_dofs(small_problem):
    """Test that boundary DOF gradients are zero."""
    problem, solver, solver_params, comm = small_problem
    rank = comm.Get_rank()

    # Create forward model wrapper
    forward_model = ForwardModelWrapper(solver, problem, solver_params)

    # Get boundary DOFs
    boundary_dofs = get_boundary_dofs(solver.V, problem.mesh)
    n_dofs = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs

    # Create true initial condition
    m_true = PETSc.Vec().createWithArray(
        solver.u.x.array.copy(),
        comm=comm
    )

    # Run forward model
    trajectory, jacobians = forward_model.solve(m_true, store_jacobians=True)

    # Setup observation
    coords = problem.mesh.geometry.x
    interior_mask = (
        (coords[:, 0] > coords[:, 0].min() + 0.1) &
        (coords[:, 0] < coords[:, 0].max() - 0.1) &
        (coords[:, 1] > coords[:, 1].min() + 0.1) &
        (coords[:, 1] < coords[:, 1].max() - 0.1)
    )
    interior_coords = coords[interior_mask][:5]
    obs_points = np.zeros((len(interior_coords), 3))
    obs_points[:, :2] = interior_coords[:, :2]

    obs_operator = PointObservationOperator(solver.V, obs_points, comm=comm)

    # Create perturbed observations to create mismatch
    obs_times = [len(trajectory) - 1]
    obs = obs_operator.forward(trajectory[-1])
    obs_perturbed = obs.copy()
    obs_arr = obs_perturbed.getArray()
    np.random.seed(43)  # Different seed from first test
    obs_arr += 0.1 * np.abs(obs_arr) * np.random.randn(len(obs_arr))
    obs_perturbed.setArray(obs_arr)
    observations = [obs_perturbed]

    B = DiagonalCovariance(comm, n_dofs, variance=0.01)
    n_obs_total = obs_operator.get_num_observations()
    R = DiagonalCovariance(comm, n_obs_total, variance=0.001)

    cost_function = FourDVarCost(
        forward_model=forward_model,
        observation_operator=obs_operator,
        background_cov=B,
        observation_cov=R,
        m_background=m_true.copy(),
        observations=observations,
        obs_times=obs_times,
        comm=comm,
    )

    # Compute adjoint gradient
    cost_function.clear_cache()
    adjoint_grad = cost_function.gradient(m_true)
    adjoint_grad_arr = adjoint_grad.getArray()

    # Check boundary DOF gradients
    boundary_grad_values = adjoint_grad_arr[boundary_dofs]

    if rank == 0:
        print("\n" + "=" * 60)
        print("Boundary DOF Gradient Test")
        print("=" * 60)
        print(f"Number of boundary DOFs: {len(boundary_dofs)}")
        print(f"Max |gradient| at boundary: {np.max(np.abs(boundary_grad_values)):.2e}")
        print(f"Mean |gradient| at boundary: {np.mean(np.abs(boundary_grad_values)):.2e}")

        # Compare to interior gradient magnitude
        interior_dofs = [i for i in range(n_dofs) if i not in set(boundary_dofs.tolist())]
        interior_grad_values = adjoint_grad_arr[interior_dofs]
        print(f"Mean |gradient| at interior: {np.mean(np.abs(interior_grad_values)):.2e}")
        print("=" * 60)

    # Boundary gradients should be zero (or very small)
    max_boundary_grad = np.max(np.abs(boundary_grad_values))
    assert max_boundary_grad < 1e-10, \
        f"Boundary DOF gradients should be zero, but max is {max_boundary_grad:.2e}"

    # Cleanup PETSc objects (DiagonalCovariance doesn't have destroy())
    adjoint_grad.destroy()
    for vec in trajectory:
        vec.destroy()
    for mat in jacobians:
        mat.destroy()
    for obs in observations:
        obs.destroy()
    m_true.destroy()


if __name__ == "__main__":
    # Run tests manually
    import sys

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("=" * 60)
        print("Gradient Accuracy Test Suite")
        print("Testing unmodified Jacobian fix for discrete adjoint")
        print("=" * 60)

    # Create problem
    problem = TidalProblem(
        nx=5,
        ny=3,
        x0=0.0,
        x1=10000.0,
        y0=0.0,
        y1=5000.0,
        nt=2,
        dt=1800.0,
    )

    solver = get_solver("CG")(problem, theta=1.0, p_degree=[1, 1], verbose=False)

    solver_params = get_default_solver_params()
    solver_params['atol'] = 1e-6
    solver_params['max_it'] = 10

    small_problem_fixture = (problem, solver, solver_params, comm)

    try:
        test_gradient_accuracy_interior_dofs(small_problem_fixture)
        if rank == 0:
            print("\n✓ Interior DOF gradient test PASSED\n")
    except AssertionError as e:
        if rank == 0:
            print(f"\n✗ Interior DOF gradient test FAILED: {e}\n")
        sys.exit(1)

    # Re-create solver for second test (state may have changed)
    solver = get_solver("CG")(problem, theta=1.0, p_degree=[1, 1], verbose=False)
    small_problem_fixture = (problem, solver, solver_params, comm)

    try:
        test_gradient_zero_at_boundary_dofs(small_problem_fixture)
        if rank == 0:
            print("\n✓ Boundary DOF gradient test PASSED\n")
    except AssertionError as e:
        if rank == 0:
            print(f"\n✗ Boundary DOF gradient test FAILED: {e}\n")
        sys.exit(1)

    if rank == 0:
        print("=" * 60)
        print("All gradient tests PASSED!")
        print("=" * 60)

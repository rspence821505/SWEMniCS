#!/usr/bin/env python3
"""
Diagnostic script to trace why gradient is ~0 at background state.

Traces the full chain:
1. Observation generation
2. Forward model evaluation at m_b
3. Innovation computation
4. Observation forcing (H^T R^{-1} d_k)
5. Adjoint solve
6. Final gradient
"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mpi4py import MPI
from petsc4py import PETSc

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.data_assimilation import (
    FourDVarCost,
    DiagonalCovariance,
    PointObservationOperator,
)
from swe4dvar.utils import get_default_solver_params
from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig, ForwardModelWrapper


def diagnose():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    print("=" * 70)
    print("GRADIENT DIAGNOSTIC")
    print("=" * 70)

    # Step 1: Create problem and solver
    print("\n--- Step 1: Setup ---")
    problem = TidalProblem(nx=20, ny=10, dt=1800, nt=96)
    solver = get_solver("DG")(problem, theta=0.5, p_degree=[1, 1])
    solver_params = get_default_solver_params()

    V = solver.V
    n_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    print(f"  DOFs: {n_dofs}")
    print(f"  V element: {V.ufl_element()}")

    # Step 2: Generate truth trajectory
    print("\n--- Step 2: Truth trajectory ---")
    problem.t = 0.0
    if hasattr(problem, 'update_boundary'):
        problem.update_boundary()

    # Get truth IC
    m_true = PETSc.Vec().createSeq(n_dofs, comm=PETSc.COMM_SELF)
    m_true.setArray(solver.u.x.array[:n_dofs].copy())
    m_true.assemble()

    true_array = m_true.getArray()
    print(f"  m_true size: {m_true.getSize()}")
    print(f"  m_true norm: {m_true.norm():.6f}")
    print(f"  m_true range: [{true_array.min():.4f}, {true_array.max():.4f}]")

    # Run forward model to get truth trajectory
    fm = ForwardModelWrapper(solver, problem, solver_params)
    truth_traj, _ = fm.solve(m_true, store_jacobians=False)
    print(f"  Truth trajectory length: {len(truth_traj)}")
    print(f"  Truth state[0] norm: {truth_traj[0].norm():.6f}")
    print(f"  Truth state[-1] norm: {truth_traj[-1].norm():.6f}")

    # Step 3: Setup observations
    print("\n--- Step 3: Observations ---")
    obs_frequency = 4
    obs_times = list(range(obs_frequency, len(truth_traj), obs_frequency))
    print(f"  Observation times: {obs_times[:5]}... ({len(obs_times)} total)")

    # Generate interior observation points
    from experiments.twin_experiment import TwinExperiment
    config = TwinExperimentConfig(method="4dvar", obs_fraction=0.5, obs_frequency=4,
                                  obs_noise_level=0.01, interior_only=True,
                                  background_error_std=0.1,
                                  obs_seed=42, background_seed=123)

    # Get mesh coordinates for observation points
    mesh = problem.mesh
    coords = mesh.geometry.x[:, :2]
    n_total = len(coords)

    # Select interior points
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    tol = 1e-10
    interior_mask = (
        (coords[:, 0] > x_min + tol) & (coords[:, 0] < x_max - tol) &
        (coords[:, 1] > y_min + tol) & (coords[:, 1] < y_max - tol)
    )
    interior_indices = np.where(interior_mask)[0]
    print(f"  Total mesh nodes: {n_total}")
    print(f"  Interior nodes: {len(interior_indices)}")

    rng = np.random.default_rng(42)
    n_obs = max(1, int(0.5 * len(interior_indices)))
    selected = rng.choice(interior_indices, size=n_obs, replace=False)
    obs_points = np.zeros((n_obs, 3))
    obs_points[:, :2] = coords[selected]
    print(f"  Observation points: {n_obs}")

    # Create observation operator
    obs_op = PointObservationOperator(V, obs_points, comm=comm)
    print(f"  Obs operator n_obs: {obs_op.get_num_observations()}")
    print(f"  Obs operator is_mixed: {obs_op.is_mixed}")
    print(f"  Obs operator is_dg: {obs_op.is_dg}")

    # Generate observations from truth
    observations = []
    noise_stds = []
    for k in obs_times:
        H_u = obs_op.forward(truth_traj[k])
        H_u_array = H_u.getArray()
        signal_mag = np.abs(H_u_array).mean() + 1e-10
        noise_std = 0.01 * signal_mag
        noise_stds.append(noise_std)
        noise = rng.normal(0, noise_std, size=H_u_array.shape)
        noisy_obs = H_u_array + noise

        obs_vec = PETSc.Vec().createSeq(len(noisy_obs), comm=PETSc.COMM_SELF)
        obs_vec.setArray(noisy_obs)
        obs_vec.assemble()
        observations.append(obs_vec)

    print(f"  Mean noise std: {np.mean(noise_stds):.6f}")
    print(f"  Obs[0] norm: {observations[0].norm():.6f}")
    print(f"  Obs[0] range: [{observations[0].getArray().min():.4f}, {observations[0].getArray().max():.4f}]")

    # Step 4: Create background
    print("\n--- Step 4: Background ---")
    rng_bg = np.random.default_rng(123)

    # Get component DOF indices
    h_dofs = V.sub(0).dofmap.list.flatten()
    uv_dofs = V.sub(1).dofmap.list.flatten()
    h_indices = np.unique(h_dofs)
    uv_indices = np.unique(uv_dofs)
    u_indices = uv_indices[0::2]
    v_indices = uv_indices[1::2]

    h_mag = np.abs(true_array[h_indices]).mean() + 1e-10
    uv_mag = max(np.abs(true_array[u_indices]).max(), np.abs(true_array[v_indices]).max(), 0.1)
    h_error = 0.1 * h_mag
    uv_error = 0.1 * uv_mag

    perturbation = np.zeros_like(true_array)
    perturbation[h_indices] = rng_bg.normal(0, h_error, len(h_indices))
    perturbation[u_indices] = rng_bg.normal(0, uv_error, len(u_indices))
    perturbation[v_indices] = rng_bg.normal(0, uv_error, len(v_indices))

    bg_array = true_array + perturbation
    bg_array[h_indices] = np.maximum(bg_array[h_indices], 0.01)

    m_bg = m_true.duplicate()
    m_bg.setArray(bg_array)
    m_bg.assemble()

    diff = m_bg.copy()
    diff.axpy(-1.0, m_true)
    bg_error = np.sqrt(diff.dot(diff) / diff.getSize())
    print(f"  Background RMS error: {bg_error:.6f}")
    print(f"  m_bg norm: {m_bg.norm():.6f}")
    print(f"  m_bg - m_true norm: {diff.norm():.6f}")
    diff.destroy()

    # Step 5: Run forward from background
    print("\n--- Step 5: Forward from background ---")
    problem.t = 0.0
    if hasattr(problem, 'update_boundary'):
        problem.update_boundary()
    solver.storage.clear()

    fm2 = ForwardModelWrapper(solver, problem, solver_params)
    bg_traj, bg_jacs = fm2.solve(m_bg, store_jacobians=True)
    print(f"  Background trajectory length: {len(bg_traj)}")
    print(f"  bg_traj[0] norm: {bg_traj[0].norm():.6f}")
    print(f"  bg_traj[-1] norm: {bg_traj[-1].norm():.6f}")

    # Check trajectory difference
    for k_idx in [0, 1, len(truth_traj)//2, -1]:
        diff_k = bg_traj[k_idx].copy()
        diff_k.axpy(-1.0, truth_traj[k_idx])
        print(f"  ||bg_traj[{k_idx}] - truth_traj[{k_idx}]|| = {diff_k.norm():.6e}")
        diff_k.destroy()

    # Step 6: Compute innovations
    print("\n--- Step 6: Innovations ---")
    for i, k in enumerate(obs_times[:5]):
        Hu_k = obs_op.forward(bg_traj[k])
        d_k = Hu_k.duplicate()
        d_k.waxpy(-1.0, observations[i], Hu_k)  # d_k = Hu_k - y_k
        print(f"  Time {k}: ||H(u_k(m_b))|| = {Hu_k.norm():.4f}, "
              f"||y_k|| = {observations[i].norm():.4f}, "
              f"||d_k|| = {d_k.norm():.6e}, "
              f"mean(d_k) = {d_k.getArray().mean():.6e}")
        Hu_k.destroy()
        d_k.destroy()

    # Step 7: Compute observation forcings
    print("\n--- Step 7: Observation forcings ---")
    truth_mag = np.abs(m_true.getArray()).mean()
    bg_variance = (0.1 * truth_mag) ** 2
    obs_variance = np.mean(noise_stds) ** 2

    B = DiagonalCovariance(comm, n_dofs, variance=bg_variance)
    R = DiagonalCovariance(comm, n_obs, variance=obs_variance)

    print(f"  Background variance: {bg_variance:.6e}")
    print(f"  Observation variance: {obs_variance:.6e}")
    print(f"  B^{-1} scale: {1.0/bg_variance:.6e}")
    print(f"  R^{-1} scale: {1.0/obs_variance:.6e}")

    N = len(bg_traj)
    forcings = [None] * N

    for i, k in enumerate(obs_times):
        Hu_k = obs_op.forward(bg_traj[k])
        d_k = Hu_k.duplicate()
        d_k.waxpy(-1.0, observations[i], Hu_k)

        R_inv_d = R.apply_inverse(d_k)

        forcing_k = obs_op.adjoint(R_inv_d)
        forcings[k] = forcing_k

        if i < 5:
            print(f"  Time {k}: ||d_k|| = {d_k.norm():.6e}, "
                  f"||R^-1 d_k|| = {R_inv_d.norm():.6e}, "
                  f"||H^T R^-1 d_k|| = {forcing_k.norm():.6e}")
            # Check forcing size
            print(f"    forcing size: {forcing_k.getSize()}, "
                  f"state size: {bg_traj[k].getSize()}")
            # Check if forcing is a ghosted vector
            local_size = forcing_k.getLocalSize()
            global_size = forcing_k.getSize()
            print(f"    forcing local_size: {local_size}, global_size: {global_size}")

        Hu_k.destroy()
        d_k.destroy()
        R_inv_d.destroy()

    # Count non-None forcings
    n_forcings = sum(1 for f in forcings if f is not None)
    print(f"  Non-None forcings: {n_forcings}")

    # Step 8: Adjoint solve
    print("\n--- Step 8: Adjoint solve ---")
    from swe4dvar.adjoint.implicit_adjoint import ImplicitAdjointSolver
    from swe4dvar.utils import get_boundary_dofs

    V_solver = fm2.solver.V
    mesh_solver = fm2.problem.mesh
    boundary_dofs = get_boundary_dofs(V_solver, mesh_solver)
    bc_dof_indices = set(boundary_dofs.tolist())
    print(f"  Boundary DOFs: {len(bc_dof_indices)} out of {n_dofs}")
    print(f"  Boundary DOF fraction: {len(bc_dof_indices)/n_dofs:.1%}")

    var_form = getattr(fm2, 'var_form', None)
    if var_form is None and hasattr(fm2, 'solver'):
        var_form = getattr(fm2.solver, 'var_form', None)
    print(f"  Variational form: {'found' if var_form else 'not found'}")

    # Check if Jacobians are available
    if bg_jacs:
        print(f"  Jacobians: {len(bg_jacs)} available")
        if bg_jacs[0] is not None:
            print(f"  Jacobian[0] size: {bg_jacs[0].getSize()}")
    else:
        print(f"  Jacobians: None")

    adjoint_solver = ImplicitAdjointSolver(
        fm2,
        bg_traj,
        bg_jacs,
        fm2.dt,
        variational_form=var_form,
        bc_dof_indices=bc_dof_indices,
    )

    terminal = bg_traj[-1].duplicate()
    terminal.zeroEntries()

    # Check forcing vector compatibility
    print("\n  Checking forcing/trajectory vector compatibility:")
    for i, k in enumerate(obs_times[:3]):
        if forcings[k] is not None:
            f_size = forcings[k].getSize()
            t_size = bg_traj[k].getSize()
            print(f"    Time {k}: forcing size={f_size}, trajectory size={t_size}, "
                  f"match={f_size == t_size}")

    # Check mass matrix
    M = adjoint_solver._get_mass_matrix()
    print(f"\n  Mass matrix: type={type(M)}, size={M.getSize()}")
    # Check if mass matrix is identity
    test_vec = bg_traj[0].duplicate()
    test_vec.set(1.0)
    result_vec = test_vec.duplicate()
    M.mult(test_vec, result_vec)
    print(f"  M*ones norm: {result_vec.norm():.6e} (should be ~{test_vec.norm():.6e} for identity)")
    print(f"  M*ones == ones? {np.allclose(result_vec.getArray(), test_vec.getArray(), atol=1e-10)}")
    # Check diagonal
    diag = M.getDiagonal()
    diag_arr = diag.getArray()
    print(f"  M diagonal: min={diag_arr.min():.6e}, max={diag_arr.max():.6e}, mean={diag_arr.mean():.6e}")
    test_vec.destroy()
    result_vec.destroy()
    diag.destroy()

    # Check BDF2 settings
    print(f"  use_bdf2: {adjoint_solver.use_bdf2}")
    print(f"  bdf2_start_step: {adjoint_solver.bdf2_start_step}")
    print(f"  flux_formulation: {adjoint_solver.flux_formulation}")
    print(f"  h_dof_indices: {'set' if adjoint_solver.h_dof_indices else 'None'}")
    print(f"  dt: {adjoint_solver.dt}")

    # Manual step-by-step adjoint to trace the issue
    print("\n  --- Manual adjoint backward sweep (key steps) ---")

    # Final step: J_N^T λ_N = -f_N
    N = adjoint_solver.num_steps  # 96
    final_rhs = terminal.copy()
    if forcings[-1] is not None:
        final_rhs.axpy(-1.0, forcings[-1])
        print(f"  Step {N} (final): ||f_N|| = {forcings[-1].norm():.6e}, ||rhs|| = {final_rhs.norm():.6e}")
    else:
        print(f"  Step {N} (final): no observation forcing, ||rhs|| = {final_rhs.norm():.6e}")

    # Solve J_N^T λ_N = rhs
    J_N = bg_jacs[N - 1]
    ksp = PETSc.KSP().create(J_N.getComm())
    ksp.setOperators(J_N)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    lambda_N = final_rhs.duplicate()
    ksp.solveTranspose(final_rhs, lambda_N)
    reason = ksp.getConvergedReason()
    print(f"  Step {N}: J^T λ = rhs solved, reason={reason}")
    print(f"  λ_{N} norm: {lambda_N.norm():.6e}")
    ksp.destroy()
    final_rhs.destroy()

    # Step N-1
    c_next = 2.0 / adjoint_solver.dt if adjoint_solver.use_bdf2 else 1.0 / adjoint_solver.dt
    rhs_Nm1 = lambda_N.duplicate()
    M.mult(lambda_N, rhs_Nm1)
    print(f"  Step {N-1}: M*λ_{N} norm = {rhs_Nm1.norm():.6e}")
    rhs_Nm1.scale(c_next)
    print(f"  Step {N-1}: c*M*λ_{N} norm = {rhs_Nm1.norm():.6e} (c={c_next:.6e})")
    if forcings[N-1] is not None:
        rhs_Nm1.axpy(-1.0, forcings[N-1])
        print(f"  Step {N-1}: after obs forcing, ||rhs|| = {rhs_Nm1.norm():.6e}")

    J_Nm1 = bg_jacs[N - 2]
    ksp2 = PETSc.KSP().create(J_Nm1.getComm())
    ksp2.setOperators(J_Nm1)
    ksp2.setType(PETSc.KSP.Type.PREONLY)
    ksp2.getPC().setType(PETSc.PC.Type.LU)
    lambda_Nm1 = rhs_Nm1.duplicate()
    ksp2.solveTranspose(rhs_Nm1, lambda_Nm1)
    reason2 = ksp2.getConvergedReason()
    print(f"  λ_{N-1} norm: {lambda_Nm1.norm():.6e}, solve reason={reason2}")
    ksp2.destroy()
    rhs_Nm1.destroy()

    # Step N-2
    rhs_Nm2 = lambda_N.duplicate()
    M.mult(lambda_Nm1, rhs_Nm2)
    rhs_Nm2.scale(c_next)
    # Add BDF2 coupling from lambda_N
    if adjoint_solver.use_bdf2:
        c_next_next = -1.0 / (2.0 * adjoint_solver.dt)
        temp = lambda_N.duplicate()
        M.mult(lambda_N, temp)
        rhs_Nm2.axpy(c_next_next, temp)
        temp.destroy()
    if forcings[N-2] is not None:
        rhs_Nm2.axpy(-1.0, forcings[N-2])
        print(f"  Step {N-2}: ||rhs|| = {rhs_Nm2.norm():.6e} (has obs forcing)")
    else:
        print(f"  Step {N-2}: ||rhs|| = {rhs_Nm2.norm():.6e} (no obs)")

    J_Nm2 = bg_jacs[N - 3]
    ksp3 = PETSc.KSP().create(J_Nm2.getComm())
    ksp3.setOperators(J_Nm2)
    ksp3.setType(PETSc.KSP.Type.PREONLY)
    ksp3.getPC().setType(PETSc.PC.Type.LU)
    lambda_Nm2 = rhs_Nm2.duplicate()
    ksp3.solveTranspose(rhs_Nm2, lambda_Nm2)
    print(f"  λ_{N-2} norm: {lambda_Nm2.norm():.6e}")
    ksp3.destroy()
    rhs_Nm2.destroy()

    # Clean up manual solve vars
    lambda_N.destroy()
    lambda_Nm1.destroy()
    lambda_Nm2.destroy()

    # Now run full solve
    lambda_0 = adjoint_solver.solve(terminal, forcings)
    print(f"\n  Full adjoint solve:")
    print(f"  λ_0 norm: {lambda_0.norm():.6e}")
    print(f"  λ_0 size: {lambda_0.getSize()}")
    lambda_arr = lambda_0.getArray()
    print(f"  λ_0 range: [{lambda_arr.min():.6e}, {lambda_arr.max():.6e}]")
    print(f"  λ_0 nonzeros: {np.count_nonzero(np.abs(lambda_arr) > 1e-15)}")

    # Step 9: Full gradient
    print("\n--- Step 9: Full gradient ---")
    delta_m = m_bg.copy()
    delta_m.axpy(-1.0, m_bg)  # = 0 (since we evaluate at m_b)
    grad_bg = B.apply_inverse(delta_m)
    print(f"  ||B^-1(m-m_b)|| = {grad_bg.norm():.6e} (should be 0 at m_b)")

    grad = grad_bg.duplicate()
    grad.axpy(1.0, lambda_0)
    print(f"  ||∇J(m_b)|| = {grad.norm():.6e}")

    # Step 10: Check ZeroBoundaryGradientCost effect
    print("\n--- Step 10: Boundary gradient zeroing effect ---")
    # Using the _get_boundary_dofs from TwinExperiment
    from dolfinx.mesh import locate_entities_boundary
    import dolfinx

    tdim = mesh.topology.dim
    fdim = tdim - 1
    def on_boundary(x):
        return np.full(x.shape[1], True)
    boundary_facets = locate_entities_boundary(mesh, fdim, on_boundary)
    twin_boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
    print(f"  TwinExperiment boundary DOFs: {len(twin_boundary_dofs)} out of {n_dofs}")
    print(f"  Boundary DOF fraction: {len(twin_boundary_dofs)/n_dofs:.1%}")

    # Apply boundary zeroing
    grad_arr = grad.getArray().copy()
    grad_before_zeroing = np.linalg.norm(grad_arr)
    grad_arr[twin_boundary_dofs] = 0.0
    grad_after_zeroing = np.linalg.norm(grad_arr)
    print(f"  ||grad|| before boundary zeroing: {grad_before_zeroing:.6e}")
    print(f"  ||grad|| after boundary zeroing: {grad_after_zeroing:.6e}")
    print(f"  Fraction of gradient zeroed: {1 - grad_after_zeroing/(grad_before_zeroing+1e-30):.1%}")

    # Cleanup
    terminal.destroy()
    m_true.destroy()
    m_bg.destroy()

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    diagnose()

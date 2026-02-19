#!/usr/bin/env python3
"""
Phase 0.5: Gradient verification for Shinnecock Inlet.

Finite-difference gradient check on a short Shinnecock window to verify
the adjoint is correct before trusting any DA results.

Tests ~10 DOFs across component types and physical regions:
- Interior h DOF in deep water (should be easy)
- Interior h DOF in shallow/WD-active region (tests WD Jacobian)
- Velocity DOF near open boundary (tests BC handling)
- Velocity DOF in WD region (tests coupling)

Uses central FD with Richardson extrapolation across multiple epsilon values.

Usage:
    CC=/usr/bin/clang python experiments/shinnecock_study/verify_gradient.py
    CC=/usr/bin/clang python experiments/shinnecock_study/verify_gradient.py --nt 6 --n-obs 10
"""

import os
os.environ.setdefault("CC", "/usr/bin/clang")

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def classify_dof(dof_idx, h_set, ux_set, uy_set):
    """Return component name for a DOF index."""
    if dof_idx in h_set:
        return "h"
    elif dof_idx in ux_set:
        return "ux"
    elif dof_idx in uy_set:
        return "uy"
    return "?"


def select_test_dofs(solver, problem, h_indices, ux_indices, uy_indices):
    """Select ~10 test DOFs spanning different physical regions.

    Returns list of (dof_idx, component, region_description) tuples.
    """
    # Get mesh coordinates for DOF classification
    mesh = problem.mesh
    V = solver.V

    # Get DOF coordinates via cell centroids (DG elements)
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    geom = mesh.geometry.x

    cell_centroids = np.zeros((num_cells, 2))
    for cell_idx in range(num_cells):
        cell_verts = mesh.geometry.dofmap[cell_idx]
        cell_centroids[cell_idx] = geom[cell_verts, :2].mean(axis=0)

    n_total_dofs = len(solver.u.x.array)
    dof_coords = np.zeros((n_total_dofs, 2))
    dofmap = V.dofmap
    for cell_idx in range(num_cells):
        for dof in dofmap.cell_dofs(cell_idx):
            if dof < n_total_dofs:
                dof_coords[dof] = cell_centroids[cell_idx]

    # Get bathymetry at DOF locations for depth classification
    depth_values = problem.depth.x.array.copy()

    # For DG, we approximate DOF depth by cell centroid depth
    # depth is defined on P1 CG, so use mean over cell vertices
    V_depth = problem.depth.function_space
    dof_depth = np.zeros(n_total_dofs)
    for cell_idx in range(num_cells):
        cell_verts = mesh.geometry.dofmap[cell_idx]
        # Average depth over cell vertices
        cell_depth = depth_values[cell_verts].mean() if len(cell_verts) <= len(depth_values) else 10.0
        for dof in dofmap.cell_dofs(cell_idx):
            if dof < n_total_dofs:
                dof_depth[dof] = cell_depth

    # Get boundary DOFs
    open_dofs = set(problem.dof_open) if hasattr(problem, 'dof_open') else set()

    test_dofs = []

    # 1. h DOF in deep water (depth > 20m, far from boundary)
    deep_h = [d for d in h_indices if dof_depth[d] > 20.0 and d not in open_dofs]
    if deep_h:
        idx = deep_h[len(deep_h) // 4]
        test_dofs.append((idx, "h", f"deep water (depth={dof_depth[idx]:.1f}m)"))
        idx2 = deep_h[len(deep_h) // 2]
        test_dofs.append((idx2, "h", f"deep water mid (depth={dof_depth[idx2]:.1f}m)"))

    # 2. h DOF in shallow/WD-active region (depth < alpha=1.5m)
    shallow_h = [d for d in h_indices if dof_depth[d] < 1.5 and d not in open_dofs]
    if shallow_h:
        idx = shallow_h[len(shallow_h) // 2]
        test_dofs.append((idx, "h", f"shallow/WD (depth={dof_depth[idx]:.1f}m)"))
    elif len(h_indices) > 0:
        # Fallback: shallowest h DOF
        shallowest = sorted(h_indices, key=lambda d: dof_depth[d])
        idx = shallowest[0]
        test_dofs.append((idx, "h", f"shallowest (depth={dof_depth[idx]:.1f}m)"))

    # 3. h DOF near open boundary
    boundary_h = [d for d in h_indices if d in open_dofs]
    if boundary_h:
        idx = boundary_h[len(boundary_h) // 2]
        test_dofs.append((idx, "h", f"open boundary (depth={dof_depth[idx]:.1f}m)"))

    # 4. ux DOF in deep water
    deep_ux = [d for d in ux_indices if dof_depth[d] > 20.0 and d not in open_dofs]
    if deep_ux:
        idx = deep_ux[len(deep_ux) // 3]
        test_dofs.append((idx, "ux", f"deep water (depth={dof_depth[idx]:.1f}m)"))

    # 5. ux DOF in shallow region
    shallow_ux = [d for d in ux_indices if dof_depth[d] < 5.0 and d not in open_dofs]
    if shallow_ux:
        idx = shallow_ux[len(shallow_ux) // 2]
        test_dofs.append((idx, "ux", f"shallow (depth={dof_depth[idx]:.1f}m)"))

    # 6. uy DOF in deep water
    deep_uy = [d for d in uy_indices if dof_depth[d] > 20.0 and d not in open_dofs]
    if deep_uy:
        idx = deep_uy[len(deep_uy) // 3]
        test_dofs.append((idx, "uy", f"deep water (depth={dof_depth[idx]:.1f}m)"))

    # 7. uy DOF in WD region
    shallow_uy = [d for d in uy_indices if dof_depth[d] < 1.5 and d not in open_dofs]
    if shallow_uy:
        idx = shallow_uy[len(shallow_uy) // 2]
        test_dofs.append((idx, "uy", f"shallow/WD (depth={dof_depth[idx]:.1f}m)"))

    # 8. Random DOF for good measure
    rng = np.random.default_rng(42)
    all_interior = [d for d in range(n_total_dofs) if d not in open_dofs]
    if all_interior:
        random_dof = rng.choice(all_interior)
        comp = classify_dof(random_dof, set(h_indices), set(ux_indices), set(uy_indices))
        test_dofs.append((random_dof, comp, f"random (depth={dof_depth[random_dof]:.1f}m)"))

    return test_dofs


def run_gradient_verification(nt=6, n_obs=20, epsilons=None, adios_file="data/shinnecock_inlet"):
    """Run finite-difference gradient verification on Shinnecock.

    Parameters
    ----------
    nt : int
        Number of timesteps for the short verification window.
    n_obs : int
        Number of observation points.
    epsilons : list of float
        Epsilon values for central FD.
    adios_file : str
        Path to ADCIRC data files.
    """
    from mpi4py import MPI
    from petsc4py import PETSc
    from swe4dvar.forward.adcirc_problem import ADCIRCProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params
    from swe4dvar.data_assimilation import (
        FourDVarCost, DiagonalCovariance, PointObservationOperator,
    )
    from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig, ForwardModelWrapper

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if epsilons is None:
        epsilons = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    output_dir = Path("outputs/shinnecock_study/data")
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    dt = 600.0
    T_hours = nt * dt / 3600.0

    if rank == 0:
        print("=" * 80)
        print("PHASE 0.5: GRADIENT VERIFICATION (Shinnecock)")
        print("=" * 80)
        print(f"  Timesteps: {nt} ({T_hours:.1f} hours)")
        print(f"  Observation points: {n_obs}")
        print(f"  Epsilons: {epsilons}")
        print("=" * 80)

    # ----------------------------------------------------------------
    # Step 1: Create problem and solver
    # ----------------------------------------------------------------
    if rank == 0:
        print("\n--- Step 1: Creating problem and solver ---")

    prob = ADCIRCProblem(
        adios_file=adios_file,
        spherical=True,
        solution_var="h",
        friction_law="mannings",
        wd=True,
        wd_alpha=1.5,
        dt=dt,
        bathy_adjustment=0,
        nt=nt,
        dramp=2.0,
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])

    # Use default solver params (ILU for serial, as found in Phase 0)
    solver_params = get_default_solver_params(
        rtol=1e-5,
        atol=1e-6,
        max_it=10,
        relaxation_parameter=1.0,
        comm=comm,
        error_if_not_converged=True,
    )

    if rank == 0:
        print(f"  Problem: ADCIRCProblem, {prob.mesh.topology.index_map(0).size_local} nodes")
        print(f"  Solver: DG p=1, {len(solver.u.x.array)} DOFs")

    # ----------------------------------------------------------------
    # Step 2: Set up twin experiment for observations/background
    # ----------------------------------------------------------------
    if rank == 0:
        print("\n--- Step 2: Setting up twin experiment ---")

    # Compute obs_fraction from desired n_obs
    n_mesh_nodes = prob.mesh.topology.index_map(0).size_local
    obs_fraction = min(n_obs / n_mesh_nodes, 0.5)

    config = TwinExperimentConfig(
        method="4dvar",
        obs_fraction=obs_fraction,
        obs_frequency=max(1, nt // 4),
        obs_noise_level=0.01,
        background_error_std=0.01,  # Small perturbation to keep background near truth
        max_iterations=1,
        gradient_tolerance=1e-6,
        interior_only=True,
        verbose=False,
    )

    exp = TwinExperiment(
        problem=prob, solver=solver, config=config,
        solver_params=solver_params, comm=comm,
    )

    # Replicate setup from run()
    exp._generate_truth()
    obs_points, obs_operator, obs_times = exp._setup_observations()
    observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
    exp._setup_background()
    B, R, _ = exp._setup_covariances(obs_operator, obs_noise_stds)
    forward_model = exp._create_forward_model()

    n_actual_obs = obs_operator.get_num_observations()

    if rank == 0:
        print(f"  Truth trajectory: {len(exp.truth_trajectory)} states")
        print(f"  Observation points: {n_actual_obs}")
        print(f"  Observation times: {obs_times}")

    # ----------------------------------------------------------------
    # Step 3: Create BASE cost function (no M^{-1} wrapper)
    # ----------------------------------------------------------------
    if rank == 0:
        print("\n--- Step 3: Creating cost function ---")

    cost_fn = FourDVarCost(
        forward_model=forward_model,
        observation_operator=obs_operator,
        background_cov=B,
        observation_cov=R,
        m_background=exp.m_background.copy(),
        observations=observations,
        obs_times=obs_times,
        comm=comm,
    )

    # Evaluate gradient at m_true (truth IC) to guarantee forward model convergence.
    # The gradient will be non-zero because:
    #   - Background term: B^{-1}(m_true - m_background) != 0
    #   - Observation term: small (noise-level mismatch)
    # Using m_background would cause solver divergence on Shinnecock's complex
    # bathymetry because white-noise DG perturbations create cell-to-cell jumps
    # that produce enormous numerical fluxes in shallow regions.
    m = exp.m_true.copy()
    n_dofs = m.getSize()

    if rank == 0:
        m_bg_diff = exp.m_true.copy()
        m_bg_diff.axpy(-1.0, exp.m_background)
        print(f"  Control vector size: {n_dofs}")
        print(f"  Evaluating at: m_true (||m_true - m_bg|| = {m_bg_diff.norm():.6e})")
        m_bg_diff.destroy()

    # ----------------------------------------------------------------
    # Step 4: Identify DOF structure and select test DOFs
    # ----------------------------------------------------------------
    if rank == 0:
        print("\n--- Step 4: Selecting test DOFs ---")

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

    h_indices = sorted(h_set)
    ux_indices = sorted(ux_set)
    uy_indices = sorted(uy_set)

    if rank == 0:
        print(f"  h DOFs: {len(h_indices)}, ux DOFs: {len(ux_indices)}, uy DOFs: {len(uy_indices)}")

    test_dofs = select_test_dofs(solver, prob, h_indices, ux_indices, uy_indices)

    if rank == 0:
        print(f"  Selected {len(test_dofs)} test DOFs:")
        for dof_idx, comp, desc in test_dofs:
            print(f"    DOF {dof_idx:>6} ({comp:>2}): {desc}")

    # ----------------------------------------------------------------
    # Step 5: Compute adjoint gradient
    # ----------------------------------------------------------------
    if rank == 0:
        print("\n--- Step 5: Computing adjoint gradient ---")

    t_start = time.time()
    cost_val, grad_adj = cost_fn.value_gradient(m)
    t_adjoint = time.time() - t_start
    grad_adj_arr = grad_adj.getArray().copy()

    if rank == 0:
        print(f"  Cost = {cost_val:.10f}")
        print(f"  ||grad|| = {grad_adj.norm():.6e}")
        print(f"  Adjoint time: {t_adjoint:.2f} s")

    # ----------------------------------------------------------------
    # Step 6: Central FD gradient with Richardson extrapolation
    # ----------------------------------------------------------------
    if rank == 0:
        print(f"\n--- Step 6: Finite-difference gradient ({len(epsilons)} epsilon values) ---")

    # Results: fd_results[dof_idx][eps] = fd_gradient
    fd_results = {dof_info[0]: {} for dof_info in test_dofs}
    t_fd_start = time.time()

    for eps in epsilons:
        if rank == 0:
            print(f"  eps = {eps:.0e}: ", end="", flush=True)

        for dof_idx, comp, desc in test_dofs:
            # J(m + eps * e_i)
            cost_fn._control_hash = None
            cost_fn._trajectory = None
            cost_fn._jacobians = None
            m_plus = m.copy()
            arr = m_plus.getArray()
            arr[dof_idx] += eps
            m_plus.setArray(arr)
            J_plus = cost_fn.value(m_plus)
            m_plus.destroy()

            # J(m - eps * e_i)
            cost_fn._control_hash = None
            cost_fn._trajectory = None
            cost_fn._jacobians = None
            m_minus = m.copy()
            arr = m_minus.getArray()
            arr[dof_idx] -= eps
            m_minus.setArray(arr)
            J_minus = cost_fn.value(m_minus)
            m_minus.destroy()

            if np.isinf(J_plus) or np.isinf(J_minus):
                fd_results[dof_idx][eps] = float('inf')
                if rank == 0:
                    print(f"\n    WARNING: inf cost for DOF {dof_idx} at eps={eps:.0e} "
                          f"(J+={J_plus}, J-={J_minus})", end="", flush=True)
            else:
                fd_results[dof_idx][eps] = (J_plus - J_minus) / (2 * eps)

        if rank == 0:
            print("done")

    t_fd = time.time() - t_fd_start

    if rank == 0:
        print(f"  FD total time: {t_fd:.1f} s")

    # ----------------------------------------------------------------
    # Step 7: Results and convergence analysis
    # ----------------------------------------------------------------
    if rank == 0:
        print(f"\n{'=' * 110}")
        print("GRADIENT VERIFICATION RESULTS")
        print(f"{'=' * 110}")

        # Table header
        print(f"\n{'DOF':>6} {'comp':>3} {'region':<35}", end="")
        for eps in epsilons:
            print(f" {'eps='+f'{eps:.0e}':>12}", end="")
        print(f" {'adj':>14}")
        print("-" * 110)

        all_results = []
        for dof_idx, comp, desc in test_dofs:
            g_adj = grad_adj_arr[dof_idx]
            print(f"{dof_idx:>6} {comp:>3} {desc:<35}", end="")

            best_ratio = None
            for eps in epsilons:
                g_fd = fd_results[dof_idx][eps]
                ratio = g_adj / g_fd if abs(g_fd) > 1e-30 else float('inf')
                print(f" {ratio:>12.4f}", end="")
                # Use eps=1e-5 or 1e-6 as the "best" reference
                if eps in (1e-5, 1e-6) and best_ratio is None:
                    best_ratio = ratio
            print(f" {g_adj:>14.6e}")

            all_results.append({
                "dof": int(dof_idx),
                "component": comp,
                "region": desc,
                "adjoint_gradient": float(g_adj),
                "fd_gradients": {str(eps): float(fd_results[dof_idx][eps]) for eps in epsilons},
                "ratios": {str(eps): float(g_adj / fd_results[dof_idx][eps])
                           if abs(fd_results[dof_idx][eps]) > 1e-30 else float('inf')
                           for eps in epsilons},
            })

        # ----------------------------------------------------------------
        # Step 8: Pass/fail assessment
        # ----------------------------------------------------------------
        print(f"\n{'=' * 110}")
        print("PASS/FAIL ASSESSMENT")
        print(f"{'=' * 110}")

        # Use eps=1e-5 as the reference epsilon for assessment
        ref_eps = 1e-5
        pass_count = 0
        fail_count = 0

        for dof_idx, comp, desc in test_dofs:
            g_adj = grad_adj_arr[dof_idx]
            g_fd = fd_results[dof_idx][ref_eps]

            if abs(g_fd) < 1e-30 and abs(g_adj) < 1e-30:
                status = "SKIP (both ~0)"
                pass_count += 1
            elif abs(g_fd) < 1e-30:
                status = "FAIL (FD ~0)"
                fail_count += 1
            else:
                ratio = g_adj / g_fd
                is_wd = "shallow" in desc.lower() or "wd" in desc.lower()
                is_boundary = "boundary" in desc.lower()

                if is_wd:
                    threshold = 0.2  # Within [0.8, 1.2] for WD
                elif is_boundary:
                    threshold = 0.2  # Boundary also relaxed
                else:
                    threshold = 0.05  # Within [0.95, 1.05] for deep water

                if abs(ratio - 1.0) <= threshold:
                    status = f"PASS (ratio={ratio:.4f})"
                    pass_count += 1
                else:
                    status = f"FAIL (ratio={ratio:.4f}, threshold={threshold})"
                    fail_count += 1

            print(f"  DOF {dof_idx:>6} ({comp:>2}) {desc:<35}: {status}")

        total = pass_count + fail_count
        print(f"\n  Summary: {pass_count}/{total} PASS, {fail_count}/{total} FAIL")

        overall_pass = fail_count == 0

        # ----------------------------------------------------------------
        # Step 9: Richardson extrapolation convergence check
        # ----------------------------------------------------------------
        print(f"\n{'=' * 110}")
        print("RICHARDSON EXTRAPOLATION (convergence rate)")
        print(f"{'=' * 110}")
        print(f"  Expected: |ratio - 1| should decrease at rate ~eps^2 for central FD")
        print()

        for dof_idx, comp, desc in test_dofs:
            g_adj = grad_adj_arr[dof_idx]
            if abs(g_adj) < 1e-30:
                continue
            print(f"  DOF {dof_idx} ({comp}, {desc}):")
            prev_err = None
            for eps in epsilons:
                g_fd = fd_results[dof_idx][eps]
                if abs(g_fd) < 1e-30:
                    continue
                ratio = g_adj / g_fd
                err = abs(ratio - 1.0)
                rate_str = ""
                if prev_err is not None and err > 0:
                    rate = np.log10(prev_err / err) / np.log10(10)  # per decade
                    rate_str = f"  (rate: {rate:.1f}/decade)"
                print(f"    eps={eps:.0e}: ratio={ratio:.8f}, |ratio-1|={err:.2e}{rate_str}")
                prev_err = err

        # ----------------------------------------------------------------
        # Step 10: Save results
        # ----------------------------------------------------------------
        results = {
            "phase": 0.5,
            "status": "success" if overall_pass else "partial_fail",
            "overall_pass": overall_pass,
            "pass_count": int(pass_count),
            "fail_count": int(fail_count),
            "total_dofs_tested": int(total),
            "config": {
                "nt": int(nt),
                "dt": float(dt),
                "n_obs": int(n_actual_obs),
                "obs_times": [int(t) for t in obs_times],
                "epsilons": [float(e) for e in epsilons],
                "reference_epsilon": float(ref_eps),
            },
            "timing": {
                "adjoint_s": float(t_adjoint),
                "fd_total_s": float(t_fd),
            },
            "cost_at_truth": float(cost_val),
            "gradient_norm": float(grad_adj.norm()),
            "dof_results": all_results,
        }

        results_file = output_dir / "phase05_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 110}")
        print(f"PHASE 0.5: GRADIENT VERIFICATION {'PASS' if overall_pass else 'FAIL'}")
        print(f"{'=' * 110}")
        print(f"  {pass_count}/{total} DOFs passed")
        print(f"  Results saved to: {results_file}")
        print(f"{'=' * 110}")

        return results

    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 0.5: Gradient verification for Shinnecock Inlet"
    )
    parser.add_argument("--nt", type=int, default=6,
                        help="Number of timesteps (default: 6, = 1 hour)")
    parser.add_argument("--n-obs", type=int, default=20,
                        help="Number of observation points")
    parser.add_argument("--adios-file", type=str, default="data/shinnecock_inlet",
                        help="Path to ADCIRC ADIOS files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gradient_verification(
        nt=args.nt,
        n_obs=args.n_obs,
        adios_file=args.adios_file,
    )

#!/usr/bin/env python3
"""
State-Only 4D-Var and DC-WME Validation
========================================

Authoritative validation script for the state-estimation pipeline:
  1. Classical 4D-Var gradient check
  2. DC-WME (static L_wme) gradient check
  3. Recovery comparison: 4D-Var vs DC-WME under wind ramp-up

Usage:
  python experiments/state_estimation_validation.py --experiment gradient    # gradient checks only
  python experiments/state_estimation_validation.py --experiment recovery    # recovery comparison
  python experiments/state_estimation_validation.py --experiment all         # both
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Memory safety
# ---------------------------------------------------------------------------

MEM_LIMIT_MB = 8 * 1024  # 8 GB default limit


def _get_process_memory_mb():
    """Return current process RSS in MB, or None if unavailable."""
    try:
        import resource
        import platform
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return rss / (1024 * 1024)
        else:
            return rss / 1024
    except Exception:
        return None


def _check_memory(context: str = "", limit_mb: float = MEM_LIMIT_MB):
    """Log current memory and abort if over limit."""
    rss = _get_process_memory_mb()
    if rss is not None:
        print(f"  [mem] {context}: {rss:.0f} MB", flush=True)
        if rss > limit_mb:
            raise MemoryError(
                f"Process RSS {rss:.0f} MB exceeds limit {limit_mb:.0f} MB at '{context}'. "
                f"Aborting to prevent OOM."
            )


def _estimate_and_guard(description: str, bytes_needed: float, limit_mb: float = MEM_LIMIT_MB):
    """Estimate memory for an allocation and abort if it would exceed limits."""
    mb_needed = bytes_needed / 1e6
    current = _get_process_memory_mb() or 0
    projected = current + mb_needed
    print(f"  [mem] {description}: est. {mb_needed:.0f} MB (current {current:.0f} MB, projected {projected:.0f} MB)")
    if projected > limit_mb:
        raise MemoryError(
            f"Allocation '{description}' would use {mb_needed:.0f} MB, "
            f"pushing RSS to ~{projected:.0f} MB (limit {limit_mb:.0f} MB). "
            f"Aborting."
        )


def _cleanup():
    """Aggressive memory cleanup."""
    gc.collect()
    try:
        from petsc4py import PETSc
        PETSc.garbage_cleanup()
    except Exception:
        pass
    gc.collect()


def build_shinnecock_system(
    *,
    dt: float = 600.0,
    nt_ramp: int = 144,
    nt_da: int = 12,
    truth_vmax: float = 30.0,
    obs_fraction: float = 0.1,
    obs_frequency: int = 6,
    obs_noise_level: float = 0.01,
    background_error_std: float = 0.02,
):
    """Build the complete Shinnecock problem, solver, truth, and observations.

    Returns a dict with everything needed for state-only 4D-Var and DC-WME.
    """
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import la

    from swe4dvar.forward import ADCIRCProblem, get_solver
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params
    from swe4dvar.data_assimilation import (
        DiagonalCovariance,
        PointObservationOperator,
        create_cost_function,
    )

    rank = MPI.COMM_WORLD.Get_rank()
    nt_total = nt_ramp + nt_da

    # --- Wind file ---
    wind_dir = PROJECT_ROOT / "results" / "state_validation" / "wind"
    if rank == 0:
        wind_dir.mkdir(parents=True, exist_ok=True)
    MPI.COMM_WORLD.Barrier()

    wind_file = wind_dir / f"vmax{truth_vmax:.0f}_ramp{nt_ramp * dt:.0f}.h5"
    if not wind_file.exists() and rank == 0:
        from experiments.shinnecock_study.wind_models import (
            DEFAULT_TRACK,
            HollandHurricaneConfig,
            WindGridConfig,
            generate_holland_wind_field,
            write_wind_hdf5,
        )
        times = np.arange(0.0, (nt_total + 1) * dt, dt)
        grid = WindGridConfig()
        wind_cfg = HollandHurricaneConfig(
            track_waypoints=DEFAULT_TRACK,
            Vmax=truth_vmax,
        )
        windx, windy, pressure = generate_holland_wind_field(wind_cfg, grid, times)
        write_wind_hdf5(str(wind_file), grid, times, windx, windy, pressure)
        print(f"  Wind file: {wind_file}")
    MPI.COMM_WORLD.Barrier()

    # --- Problem and solver ---
    forcing = GriddedForcing(str(wind_file), lat0=35)
    problem = ADCIRCProblem(
        adios_file="data/shinnecock_inlet",
        spherical=True,
        solution_var="h",
        friction_law="mannings",
        wd=True,
        wd_alpha=1.5,
        dt=dt,
        bathy_adjustment=0,
        nt=nt_total,
        dramp=2.0,
        forcing=forcing,
    )
    solver = get_solver("DG")(problem, theta=1.0, p_degree=[1, 1])
    solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=20, relaxation_parameter=1.0,
        comm=MPI.COMM_WORLD, error_if_not_converged=False,
    )

    _check_memory("after problem+solver construction")

    state_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs

    # --- Truth trajectory ---
    # Memory estimate: (nt_total+1) saved states × state_size × 8 bytes
    traj_bytes = (nt_total + 1) * state_size * 8
    _estimate_and_guard(f"truth trajectory ({nt_total+1} × {state_size} DOFs)", traj_bytes)

    if rank == 0:
        print(f"  Running truth trajectory ({nt_total} steps)...")
    solver.time_loop(
        solver_parameters=solver_params,
        stations=[],
        plot_every=9999,
        save_state=True,
        store_jacobians=False,
        enable_video=False,
        monitor_progress=False,
    )
    truth_state = solver.u_n.x.array[:state_size].copy()
    truth_trajectory = [s[:state_size].copy() for s in solver.storage.saved_states]
    _check_memory("after truth trajectory")

    # --- Observation points ---
    coords = problem.mesh.geometry.x
    all_coords = MPI.COMM_WORLD.gather(coords, root=0)
    if rank == 0:
        coords_all = np.vstack(all_coords)
        _, unique_idx = np.unique(np.round(coords_all[:, :2], decimals=10), axis=0, return_index=True)
        unique_coords = coords_all[unique_idx]
        x_min, x_max = unique_coords[:, 0].min(), unique_coords[:, 0].max()
        y_min, y_max = unique_coords[:, 1].min(), unique_coords[:, 1].max()
        interior = unique_coords[
            (unique_coords[:, 0] > x_min + 1e-10) &
            (unique_coords[:, 0] < x_max - 1e-10) &
            (unique_coords[:, 1] > y_min + 1e-10) &
            (unique_coords[:, 1] < y_max - 1e-10)
        ]
        rng = np.random.default_rng(42)
        n_obs = max(1, int(len(interior) * obs_fraction))
        chosen = rng.choice(len(interior), size=min(n_obs, len(interior)), replace=False)
        obs_points = np.zeros((len(chosen), 3))
        obs_points[:, :2] = interior[chosen, :2]
    else:
        obs_points = None
    obs_points = MPI.COMM_WORLD.bcast(obs_points, root=0)
    obs_operator = PointObservationOperator(solver.V, obs_points, comm=MPI.COMM_WORLD)

    # --- Observations ---
    obs_times = list(range(nt_ramp, nt_total + 1, obs_frequency))
    n_obs_per_t = obs_operator.get_num_observations()
    truth_obs_matrix = np.zeros((len(obs_times), n_obs_per_t))
    for i, t in enumerate(obs_times):
        vec = la.create_petsc_vector(solver.V.dofmap.index_map, solver.V.dofmap.index_map_bs)
        vec.setArray(truth_trajectory[t])
        vec.assemble()
        obs_vec = obs_operator.forward(vec)
        truth_obs_matrix[i] = obs_vec.getArray().copy()
        obs_vec.destroy()
        vec.destroy()

    rng_noise = np.random.default_rng(1042)
    signal = np.mean(np.abs(truth_obs_matrix), axis=1) + 1e-10
    noise_stds = obs_noise_level * signal
    noise = rng_noise.normal(0.0, noise_stds[:, None], size=truth_obs_matrix.shape)
    noisy_obs = truth_obs_matrix + noise

    obs_vecs = []
    for row in noisy_obs:
        vec = PETSc.Vec().createSeq(row.size, comm=PETSc.COMM_SELF)
        vec.setArray(row.copy())
        vec.assemble()
        obs_vecs.append(vec)

    obs_covs = {}
    for t, sigma in zip(obs_times, noise_stds):
        obs_covs[t] = DiagonalCovariance(PETSc.COMM_SELF, size=n_obs_per_t, variance=float(sigma**2))

    # --- Background state (perturbed truth) ---
    rng_bg = np.random.default_rng(123)
    scale = np.maximum(np.abs(truth_state), 1e-3)
    background_state = truth_state + rng_bg.normal(0.0, background_error_std * scale)
    state_variances = (background_error_std * np.maximum(np.abs(truth_state), 1e-3)) ** 2
    B = DiagonalCovariance(PETSc.COMM_SELF, size=state_size, diagonal=state_variances)

    m_background = PETSc.Vec().createSeq(state_size, comm=PETSc.COMM_SELF)
    m_background.setArray(background_state.copy())
    m_background.assemble()

    m_truth = PETSc.Vec().createSeq(state_size, comm=PETSc.COMM_SELF)
    m_truth.setArray(truth_state.copy())
    m_truth.assemble()

    # --- Forward model wrapper ---
    from experiments.twin_experiment import ForwardModelWrapper
    forward_model = ForwardModelWrapper(
        solver, problem, solver_params,
        t_start=0.0,
    )

    # --- Gradient smoother (critical for L-BFGS convergence) ---
    # The adjoint gradient with diagonal B has high-frequency noise that
    # prevents L-BFGS line search from finding good step sizes. Spatial
    # smoothing filters this noise, matching the perturbation correlation.
    from experiments.twin_experiment import TwinExperiment
    _dummy_exp = object.__new__(TwinExperiment)
    _dummy_exp.solver = solver
    _dummy_exp.problem = problem
    _dummy_exp.mpi_rank = rank
    _dummy_exp.rank = rank
    # log() needs config.verbose
    class _DummyConfig:
        verbose = True
    _dummy_exp.config = _DummyConfig()
    h_indices, u_indices, v_indices = _dummy_exp._get_component_dof_indices(owned_only=True)
    gradient_smoothing_L = 500.0  # meters — matches background perturbation correlation
    smoothing_matrix = _dummy_exp._build_smoothing_matrix(h_indices, gradient_smoothing_L)
    _check_memory("after smoothing matrix construction")

    def gradient_smoother(grad_array):
        smoothed = grad_array.copy()
        smoothed[h_indices] = smoothing_matrix @ grad_array[h_indices]
        smoothed[u_indices] = smoothing_matrix @ grad_array[u_indices]
        smoothed[v_indices] = smoothing_matrix @ grad_array[v_indices]
        return smoothed

    if rank == 0:
        print(f"  Gradient smoother: L={gradient_smoothing_L}m, nnz={smoothing_matrix.nnz}")
        print(f"  State size: {state_size}")
        print(f"  Obs points: {n_obs_per_t}")
        print(f"  Obs times: {obs_times} ({len(obs_times)} snapshots)")
        bg_rmse = np.sqrt(np.mean((background_state - truth_state)**2))
        print(f"  Background RMSE: {bg_rmse:.6e}")

    return {
        "solver": solver,
        "problem": problem,
        "solver_params": solver_params,
        "forward_model": forward_model,
        "obs_operator": obs_operator,
        "obs_vecs": obs_vecs,
        "obs_covs": obs_covs,
        "obs_times": obs_times,
        "B": B,
        "m_background": m_background,
        "m_truth": m_truth,
        "truth_state": truth_state,
        "background_state": background_state,
        "state_size": state_size,
        "noise_stds": noise_stds,
        "n_obs_per_t": n_obs_per_t,
        "obs_variance": float(noise_stds[0]**2),
        "gradient_smoother": gradient_smoother,
    }


# =========================================================================
# Gradient checks
# =========================================================================

def run_gradient_checks(system: dict):
    """FD gradient check for both 4D-Var and DC-WME (static)."""
    from petsc4py import PETSc
    from swe4dvar.data_assimilation import create_cost_function

    print("\n" + "="*70)
    print("GRADIENT VALIDATION")
    print("="*70)

    results = {}

    for method_name, method_key, use_static_lwme in [
        ("Classical 4D-Var", "4dvar", False),
        ("DC-WME (static L_wme)", "dc_wme", True),
    ]:
        print(f"\n--- {method_name} ---")

        extra_kwargs = {}
        if use_static_lwme:
            # Build static L_wme using the same function as the experiment runner.
            # Memory estimate: H matrix is (d_obs × n_state) dense float64
            d_obs = system["n_obs_per_t"]
            n_state = system["state_size"]
            h_matrix_bytes = d_obs * n_state * 8
            hbht_bytes = d_obs * d_obs * 8
            _estimate_and_guard(
                f"static L_wme (H: {d_obs}×{n_state}, H B H^T: {d_obs}×{d_obs})",
                h_matrix_bytes + hbht_bytes * 3,  # H + HB + L_dense + eigvecs
            )

            from experiments.shinnecock_study.run_comparison import _compute_static_L_wme
            from mpi4py import MPI
            print(f"  Computing static L_wme...")
            static_lwme, lwme_diag = _compute_static_L_wme(
                system["obs_operator"],
                system["B"],
                n_obs_times=len(system["obs_times"]),
                obs_variance=system["obs_variance"],
                m_template=system["m_background"],
                predictability_gamma=0.1,
                adaptive_gamma=True,
                comm=MPI.COMM_WORLD,
                rank=MPI.COMM_WORLD.Get_rank(),
            )
            extra_kwargs = {
                "predicted_cov_wme": static_lwme,
                "n_l_wme_samples": 0,  # Skip dynamic computation
            }
            _check_memory("after static L_wme construction")
            print(f"  Static L_wme built (inflation={lwme_diag.get('inflation_factor', '?')})")

        cost_fn = create_cost_function(
            method_key,
            forward_model=system["forward_model"],
            observation_operator=system["obs_operator"],
            background_cov=system["B"],
            observation_cov=system["obs_covs"],
            m_background=system["m_background"],
            observations=system["obs_vecs"],
            obs_times=system["obs_times"],
            **extra_kwargs,
        )
        # Attach gradient smoother (critical for L-BFGS convergence)
        inner = cost_fn
        while hasattr(inner, 'base_cost'):
            inner = inner.base_cost
        inner.gradient_smoother = system["gradient_smoother"]

        m = system["m_background"]
        print(f"  Computing value_gradient at background...")
        t0 = time.time()
        J0, grad = cost_fn.value_gradient(m)
        print(f"  J(m_bg) = {J0:.6f}, ||grad|| = {grad.norm():.6e} ({time.time()-t0:.1f}s)")

        # FD check on 5 random directions
        rng = np.random.default_rng(42)
        epsilons = [1e-5, 1e-4, 1e-3]
        n_dirs = 3
        grad_arr = grad.getArray(readonly=True)
        bg_arr = m.getArray(readonly=True).copy()

        fd_results = []
        for d in range(n_dirs):
            direction = rng.standard_normal(system["state_size"])
            direction /= np.linalg.norm(direction)
            analytic_dd = np.dot(grad_arr, direction)

            for eps in epsilons:
                m_plus = m.copy()
                m_minus = m.copy()
                m_plus.setArray(bg_arr + eps * direction)
                m_plus.assemble()
                m_minus.setArray(bg_arr - eps * direction)
                m_minus.assemble()

                J_plus = cost_fn.value(m_plus)
                J_minus = cost_fn.value(m_minus)
                fd_dd = (J_plus - J_minus) / (2 * eps)

                rel_err = abs(fd_dd - analytic_dd) / max(abs(analytic_dd), 1e-30)
                fd_results.append({
                    "direction": d, "epsilon": eps,
                    "analytic": float(analytic_dd), "fd": float(fd_dd),
                    "rel_error": float(rel_err),
                })
                print(f"  dir={d} ε={eps:.0e}: analytic={analytic_dd:.6e} FD={fd_dd:.6e} rel_err={rel_err:.2e}")

                m_plus.destroy()
                m_minus.destroy()

        # Descent check: does stepping in -grad direction reduce J?
        # Use step size that produces a meaningful cost change:
        # α = ε_target / ||∇J||, aiming for ~0.1% relative cost change
        grad_norm = grad.norm()
        step_size = 0.001 * J0 / max(grad_norm**2, 1e-30)
        step_size = min(step_size, 1e-4)  # conservative cap — PDE solver can diverge for large steps
        m_stepped = m.copy()
        stepped_arr = bg_arr - step_size * grad_arr
        m_stepped.setArray(stepped_arr)
        m_stepped.assemble()
        J_stepped = cost_fn.value(m_stepped)
        descent_ok = J_stepped < J0
        predicted_decrease = step_size * grad_norm**2
        print(f"  Descent check (α={step_size:.2e}): J(m-α∇J) = {J_stepped:.6f} {'<' if descent_ok else '>='} J(m) = {J0:.6f}")
        print(f"    Predicted decrease: {predicted_decrease:.6e}, Actual: {J0 - J_stepped:.6e} → {'PASS' if descent_ok else 'FAIL'}")
        m_stepped.destroy()

        # Summary
        max_rel_err = max(r["rel_error"] for r in fd_results)
        median_rel_err = np.median([r["rel_error"] for r in fd_results])
        # Pass if FD errors < 1% AND descent works (or descent step was too small to matter)
        gradient_ok = max_rel_err < 0.01
        passed = gradient_ok and descent_ok
        print(f"  RESULT: {'PASS' if passed else 'FAIL'} (max rel_err={max_rel_err:.2e}, median={median_rel_err:.2e})")

        results[method_name] = {
            "J_at_background": float(J0),
            "grad_norm": float(grad.norm()),
            "fd_results": fd_results,
            "max_rel_error": float(max_rel_err),
            "descent_ok": descent_ok,
            "passed": passed,
        }

        grad.destroy()
        del cost_fn
        _cleanup()
        _check_memory(f"after {method_name} gradient check")

    return results


# =========================================================================
# Recovery comparison
# =========================================================================

def run_recovery_comparison(system: dict, max_iterations: int = 15):
    """Run 4D-Var and DC-WME recovery and compare state improvement."""
    from petsc4py import PETSc
    from swe4dvar.data_assimilation import create_cost_function
    from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper

    print("\n" + "="*70)
    print("RECOVERY COMPARISON: 4D-Var vs DC-WME (static L_wme)")
    print("="*70)

    truth_arr = system["m_truth"].getArray(readonly=True)
    bg_arr = system["m_background"].getArray(readonly=True)
    bg_rmse = np.sqrt(np.mean((bg_arr - truth_arr)**2))
    print(f"  Background RMSE: {bg_rmse:.6e}")

    results = {}

    for method_name, method_key, use_static_lwme in [
        ("4D-Var", "4dvar", False),
        ("DC-WME (static)", "dc_wme", True),
    ]:
        print(f"\n--- {method_name} ---")

        extra_kwargs = {}
        if use_static_lwme:
            from experiments.shinnecock_study.run_comparison import _compute_static_L_wme
            from mpi4py import MPI
            print(f"  Computing static L_wme...")
            static_lwme, lwme_diag = _compute_static_L_wme(
                system["obs_operator"],
                system["B"],
                n_obs_times=len(system["obs_times"]),
                obs_variance=system["obs_variance"],
                m_template=system["m_background"],
                predictability_gamma=0.1,
                adaptive_gamma=True,
                comm=MPI.COMM_WORLD,
                rank=MPI.COMM_WORLD.Get_rank(),
            )
            extra_kwargs = {
                "predicted_cov_wme": static_lwme,
                "n_l_wme_samples": 0,
            }

        cost_fn = create_cost_function(
            method_key,
            forward_model=system["forward_model"],
            observation_operator=system["obs_operator"],
            background_cov=system["B"],
            observation_cov=system["obs_covs"],
            m_background=system["m_background"],
            observations=system["obs_vecs"],
            obs_times=system["obs_times"],
            **extra_kwargs,
        )
        # Attach gradient smoother
        inner = cost_fn
        while hasattr(inner, 'base_cost'):
            inner = inner.base_cost
        inner.gradient_smoother = system["gradient_smoother"]

        optimizer = PETScTAOWrapper(
            cost_fn,
            tao_type="lmvm",
            options={
                "max_iterations": max_iterations,
                "max_funcs": 30,          # Cap evals to prevent infinite line search backtracking
                "gradient_tolerance": 1e-3,  # Relaxed — smoother introduces O(1e-3) error
                "cost_tolerance": 1e-4,
                "verbose": True,
            },
        )

        print(f"  Running optimizer (max {max_iterations} iterations)...")
        _check_memory(f"before {method_name} optimization")
        t0 = time.time()
        m_analysis = optimizer.solve(system["m_background"])
        wall_time = time.time() - t0

        analysis_arr = m_analysis.getArray(readonly=True)
        analysis_rmse = np.sqrt(np.mean((analysis_arr - truth_arr)**2))
        improvement = (bg_rmse - analysis_rmse) / bg_rmse * 100

        print(f"  Wall time: {wall_time:.1f}s")
        print(f"  Iterations: {optimizer.n_func_evals} func evals")
        print(f"  Background RMSE: {bg_rmse:.6e}")
        print(f"  Analysis RMSE:   {analysis_rmse:.6e}")
        print(f"  Improvement:     {improvement:.1f}%")

        results[method_name] = {
            "bg_rmse": float(bg_rmse),
            "analysis_rmse": float(analysis_rmse),
            "improvement_pct": float(improvement),
            "n_func_evals": optimizer.n_func_evals,
            "wall_time_s": float(wall_time),
        }

        m_analysis.destroy()
        del cost_fn, optimizer
        _cleanup()
        _check_memory(f"after {method_name} recovery")

    return results


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="State-Only 4D-Var / DC-WME Validation")
    parser.add_argument("--experiment", default="gradient",
                        choices=["gradient", "recovery", "all"])
    parser.add_argument("--vmax", type=float, default=30.0, help="Wind Vmax (m/s)")
    parser.add_argument("--nt-ramp", type=int, default=144, help="Ramp timesteps")
    parser.add_argument("--nt-da", type=int, default=12, help="DA window timesteps")
    parser.add_argument("--max-iter", type=int, default=15, help="Max optimizer iterations")
    parser.add_argument("--mem-limit-gb", type=float, default=8.0,
                        help="Memory limit in GB (default 8). Aborts if RSS exceeds this.")
    args = parser.parse_args()

    global MEM_LIMIT_MB
    MEM_LIMIT_MB = args.mem_limit_gb * 1024

    output_dir = PROJECT_ROOT / "results" / "state_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("STATE-ONLY ESTIMATION VALIDATION")
    print("="*70)
    print(f"  Vmax: {args.vmax} m/s")
    print(f"  Ramp: {args.nt_ramp} steps ({args.nt_ramp * 600 / 3600:.1f}h)")
    print(f"  DA window: {args.nt_da} steps ({args.nt_da * 600 / 3600:.1f}h)")

    t0 = time.time()
    system = build_shinnecock_system(
        truth_vmax=args.vmax,
        nt_ramp=args.nt_ramp,
        nt_da=args.nt_da,
    )
    print(f"  System setup: {time.time() - t0:.1f}s")

    all_results = {}

    if args.experiment in ("gradient", "all"):
        all_results["gradients"] = run_gradient_checks(system)

    if args.experiment in ("recovery", "all"):
        all_results["recovery"] = run_recovery_comparison(system, max_iterations=args.max_iter)

    # Save
    all_results["config"] = {
        "vmax": args.vmax,
        "nt_ramp": args.nt_ramp,
        "nt_da": args.nt_da,
        "dt": 600.0,
    }
    all_results["total_time_s"] = time.time() - t0

    results_file = output_dir / "validation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    raise SystemExit(main())

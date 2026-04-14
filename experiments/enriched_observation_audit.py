#!/usr/bin/env python3
"""
Enriched Observation Identifiability Audit
==========================================

Tests whether richer observation functionals can increase the effective
rank of the whitened parameter sensitivity matrix for Manning's-n estimation.

Observation families tested:
  A. Cross-section flux: integrated transport Q(t) = ∫_Γ h·u·n ds
  B. (reserved)
  C. Harmonic projections: sine/cosine coefficients of h(t,x) over the window
  D. Multi-station differentials: h(t,x_i) - h(t,x_j)

Each family is compared against the baseline (pointwise elevation)
using the same SVD identifiability framework.

Usage:
  python experiments/enriched_observation_audit.py --kl-modes 3
  python experiments/enriched_observation_audit.py --kl-modes 3 --families flux harmonic differential
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

from experiments.identifiability_audit import (
    AuditConfig,
    _to_ladder_config,
    whiten_sensitivity,
)
from experiments.validation_ladder import (
    build_problem_and_solver,
    create_observations,
    generate_truth_trajectory,
    LadderConfig,
)


# =========================================================================
# Core: Run forward model and return full trajectory
# =========================================================================

def run_forward_and_store(
    theta: np.ndarray,
    *,
    solver, problem, controller, solver_params,
    truth_state: np.ndarray,
) -> list[np.ndarray]:
    """Run forward model with given θ and return the full saved trajectory."""
    state_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs

    controller.apply(problem, solver, theta)
    solver.storage.clear()
    solver.u_n.x.array[:state_size] = truth_state
    solver.u_n_old.x.array[:state_size] = truth_state
    solver.u.x.array[:state_size] = truth_state
    solver.u_n.x.scatter_forward()
    solver.u_n_old.x.scatter_forward()
    solver.u.x.scatter_forward()
    problem.t = 0.0
    if hasattr(problem, "update_boundary"):
        problem.update_boundary()

    solver.time_loop(
        solver_parameters=solver_params,
        stations=[],
        plot_every=9999,
        save_state=True,
        store_jacobians=False,
        enable_video=False,
        monitor_progress=False,
    )

    trajectory = [s.copy() for s in solver.storage.saved_states]
    gc.collect()
    return trajectory


def extract_h_at_points(
    trajectory: list[np.ndarray],
    obs_times: list[int],
    point_obs_operator,
    solver,
) -> np.ndarray:
    """Extract h at observation points for given times. Returns (n_times, n_points)."""
    from dolfinx import la

    state_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs
    n_points = point_obs_operator.get_num_observations()
    result = np.zeros((len(obs_times), n_points))

    for i, t_idx in enumerate(obs_times):
        vec = la.create_petsc_vector(
            solver.V.dofmap.index_map, solver.V.dofmap.index_map_bs,
        )
        vec.setArray(trajectory[t_idx][:state_size])
        vec.assemble()
        obs_vec = point_obs_operator.forward(vec)
        result[i] = obs_vec.getArray().copy()
        obs_vec.destroy()
        vec.destroy()

    return result


# =========================================================================
# Family C: Harmonic / tidal projection observations
# =========================================================================

def harmonic_observations(
    h_time_series: np.ndarray,
    dt: float,
    nt_ramp: int,
    periods: list[float] = None,
) -> np.ndarray:
    """Project elevation time series onto sine/cosine basis at specified periods.

    Given h(t, x_j) at observation points over the DA window, compute:
      a_k(x_j) = (2/T) ∫ h(t, x_j) cos(2π t / T_k) dt
      b_k(x_j) = (2/T) ∫ h(t, x_j) sin(2π t / T_k) dt

    for each period T_k. This extracts the amplitude and phase of
    oscillatory components that friction may attenuate or phase-shift.

    Args:
        h_time_series: (n_times, n_points) elevation at obs points
        dt: timestep in seconds
        nt_ramp: number of ramp timesteps (to compute physical times)
        periods: list of periods in seconds to project onto.
                 Default: [M2 ≈ 44712s, M4 ≈ 22356s, and window-length]

    Returns:
        Flat observation vector: [a_0(x_0), ..., a_0(x_m), b_0(x_0), ..., b_0(x_m),
                                  a_1(x_0), ..., b_K(x_m)]
    """
    n_times, n_points = h_time_series.shape

    if periods is None:
        T_window = n_times * dt
        periods = [
            44712.0,       # M2 tidal period (~12.42 hours)
            22356.0,       # M4 (quarter-diurnal)
            T_window,      # Full window period
        ]
        # Only keep periods that fit at least half a cycle in the window
        periods = [T for T in periods if T <= 2 * T_window]
        if not periods:
            periods = [T_window]

    # Physical times for the DA window (relative to ramp end)
    times = np.arange(n_times) * dt

    obs_list = []
    for T in periods:
        omega = 2 * np.pi / T
        cos_basis = np.cos(omega * times)  # (n_times,)
        sin_basis = np.sin(omega * times)  # (n_times,)

        # Trapezoidal integration: (2/T_window) ∫ h(t) cos(ωt) dt
        T_window = (n_times - 1) * dt
        scale = 2.0 / max(T_window, 1e-30)

        a_coeffs = scale * np.trapz(h_time_series * cos_basis[:, np.newaxis], dx=dt, axis=0)
        b_coeffs = scale * np.trapz(h_time_series * sin_basis[:, np.newaxis], dx=dt, axis=0)

        obs_list.append(a_coeffs)  # (n_points,)
        obs_list.append(b_coeffs)  # (n_points,)

    return np.concatenate(obs_list)


# =========================================================================
# Family D: Multi-station differential observations
# =========================================================================

def differential_observations(
    h_time_series: np.ndarray,
    n_pairs: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Compute elevation differences between random station pairs.

    δh_{ij}(t) = h(t, x_i) - h(t, x_j)

    This cancels the global mode (which affects all stations similarly)
    and may expose spatial structure.

    Args:
        h_time_series: (n_times, n_points)
        n_pairs: number of random pairs to generate
        seed: random seed for pair selection

    Returns:
        Flat vector: [δh_{i1,j1}(t_0), ..., δh_{i1,j1}(t_K), δh_{i2,j2}(t_0), ...]
    """
    n_times, n_points = h_time_series.shape
    rng = np.random.default_rng(seed)

    # Generate random pairs (no duplicates, no self-pairs)
    pairs = set()
    while len(pairs) < min(n_pairs, n_points * (n_points - 1) // 2):
        i, j = rng.choice(n_points, size=2, replace=False)
        pair = (min(i, j), max(i, j))
        pairs.add(pair)
    pairs = sorted(pairs)

    obs_list = []
    for i, j in pairs:
        diff = h_time_series[:, i] - h_time_series[:, j]  # (n_times,)
        obs_list.append(diff)

    return np.concatenate(obs_list)


# =========================================================================
# Family A: Cross-section flux (simplified as station-pair transport proxy)
# =========================================================================

def flux_proxy_observations(
    trajectory: list[np.ndarray],
    obs_times: list[int],
    solver,
    obs_points: np.ndarray,
    n_transects: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Approximate cross-section flux using pairs of nearby observation points.

    For each "transect" (pair of nearby points), compute a transport proxy:
      Q_ij(t) ≈ h_avg(t) · (u_i(t) - u_j(t)) · Δx

    where h_avg is the average depth and (u_i - u_j) approximates the
    velocity gradient, giving a proxy for transport divergence.

    This avoids mesh-level integration (which requires FEM assembly) while
    still capturing transport-related information that differs from
    pointwise state values.

    Actually, simpler and more physical: at each point, compute h·u and h·v
    (discharge per unit width), then take differences between points.
    This gives spatial gradients of transport, which are more sensitive
    to friction distribution than raw h or u alone.

    Args:
        trajectory: full forward trajectory
        obs_times: timestep indices to observe
        solver: FEM solver (for function space)
        obs_points: observation point coordinates
        n_transects: number of point pairs
        seed: random seed

    Returns:
        Flat observation vector of transport differences across times.
    """
    from dolfinx import la
    from swe4dvar.data_assimilation import PointObservationOperator
    from mpi4py import MPI

    state_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs
    n_points = len(obs_points)

    # Get h, u, v at observation points for each obs time
    op_h = PointObservationOperator(solver.V, obs_points, component_indices=[0], comm=MPI.COMM_WORLD)
    op_u = PointObservationOperator(solver.V, obs_points, component_indices=[1], sub_component=0, comm=MPI.COMM_WORLD)
    op_v = PointObservationOperator(solver.V, obs_points, component_indices=[1], sub_component=1, comm=MPI.COMM_WORLD)

    h_series = np.zeros((len(obs_times), n_points))
    u_series = np.zeros((len(obs_times), n_points))
    v_series = np.zeros((len(obs_times), n_points))

    for i, t_idx in enumerate(obs_times):
        vec = la.create_petsc_vector(solver.V.dofmap.index_map, solver.V.dofmap.index_map_bs)
        vec.setArray(trajectory[t_idx][:state_size])
        vec.assemble()

        h_obs = op_h.forward(vec); h_series[i] = h_obs.getArray().copy(); h_obs.destroy()
        u_obs = op_u.forward(vec); u_series[i] = u_obs.getArray().copy(); u_obs.destroy()
        v_obs = op_v.forward(vec); v_series[i] = v_obs.getArray().copy(); v_obs.destroy()
        vec.destroy()

    # Compute discharge: qx = h*u, qy = h*v at each point
    qx = h_series * u_series  # (n_times, n_points)
    qy = h_series * v_series

    # Generate random pairs of nearby points
    from scipy.spatial import cKDTree
    tree = cKDTree(obs_points[:, :2])
    rng = np.random.default_rng(seed)

    pairs = []
    anchors = rng.choice(n_points, size=min(n_transects, n_points), replace=False)
    for anchor in anchors:
        # Find nearest neighbor that isn't self
        dists, idxs = tree.query(obs_points[anchor, :2], k=2)
        neighbor = idxs[1]
        if neighbor != anchor:
            pairs.append((anchor, neighbor))

    # Compute flux differences: Δqx_{ij}(t) = qx(t, x_i) - qx(t, x_j)
    obs_list = []
    for i, j in pairs:
        obs_list.append(qx[:, i] - qx[:, j])
        obs_list.append(qy[:, i] - qy[:, j])

    return np.concatenate(obs_list)


# =========================================================================
# Generic sensitivity builder for trajectory-level observations
# =========================================================================

def build_trajectory_sensitivity(
    obs_func,
    *,
    n_params: int,
    eps: float,
    solver, problem, controller, solver_params,
    truth_state, obs_times, cfg,
    obs_func_kwargs: dict = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build J_θ via FD for a trajectory-level observation functional.

    obs_func(trajectory, obs_times, ...) -> flat observation vector

    Returns (J_theta, G_ref).
    """
    if obs_func_kwargs is None:
        obs_func_kwargs = {}

    fwd_kwargs = dict(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params, truth_state=truth_state,
    )

    print(f"  Computing reference trajectory at θ=0...")
    traj_ref = run_forward_and_store(np.zeros(n_params), **fwd_kwargs)
    G_ref = obs_func(traj_ref, obs_times, **obs_func_kwargs)
    n_obs = G_ref.size
    print(f"  Observation dimension: {n_obs}")

    J_theta = np.zeros((n_obs, n_params))

    for p in range(n_params):
        print(f"  Computing J_θ column {p} (±ε={eps})...")
        theta_plus = np.zeros(n_params); theta_plus[p] = +eps
        theta_minus = np.zeros(n_params); theta_minus[p] = -eps

        traj_plus = run_forward_and_store(theta_plus, **fwd_kwargs)
        traj_minus = run_forward_and_store(theta_minus, **fwd_kwargs)

        G_plus = obs_func(traj_plus, obs_times, **obs_func_kwargs)
        G_minus = obs_func(traj_minus, obs_times, **obs_func_kwargs)

        J_theta[:, p] = (G_plus - G_minus) / (2 * eps)
        print(f"    ||J_θ[:, {p}]|| = {np.linalg.norm(J_theta[:, p]):.6e}")

    return J_theta, G_ref


def analyze_and_report(J_theta, label, n_params):
    """SVD analysis and collinearity report for a sensitivity matrix."""
    U, sigma, Vt = np.linalg.svd(J_theta, full_matrices=False)
    V = Vt.T
    cond = sigma[0] / max(sigma[-1], 1e-30)
    energy = np.cumsum(sigma**2) / max(np.sum(sigma**2), 1e-30)

    print(f"\n  [{label}]")
    print(f"  Singular values: {sigma}")
    print(f"  Condition number: {cond:.2e}")
    print(f"  Energy: {energy}")

    for tol in [0.05, 0.10]:
        rank = int(np.sum(sigma > tol * sigma[0]))
        print(f"  Effective rank (tol={tol}): {rank}")

    # Pairwise collinearity of columns
    col_matrix = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            ni = np.linalg.norm(J_theta[:, i])
            nj = np.linalg.norm(J_theta[:, j])
            if ni > 1e-30 and nj > 1e-30:
                col_matrix[i, j] = np.dot(J_theta[:, i], J_theta[:, j]) / (ni * nj)
    print(f"  Collinearity matrix:")
    for i in range(n_params):
        print(f"    " + "  ".join(f"{col_matrix[i, j]:+.4f}" for j in range(n_params)))

    return {
        "sigma": sigma.tolist(),
        "condition": float(cond),
        "energy": energy.tolist(),
        "collinearity": col_matrix.tolist(),
        "right_singular_vectors": V.tolist(),
    }


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Enriched Observation Identifiability Audit")
    parser.add_argument("--kl-modes", type=int, default=3)
    parser.add_argument("--kl-length", type=float, default=None)
    parser.add_argument("--families", nargs="+",
                        default=["baseline", "harmonic", "differential", "flux"],
                        choices=["baseline", "harmonic", "differential", "flux", "all"],
                        help="Observation families to test")
    args = parser.parse_args()

    if "all" in args.families:
        args.families = ["baseline", "harmonic", "differential", "flux"]

    cfg = AuditConfig(
        kl_n_modes=args.kl_modes,
        kl_correlation_length=args.kl_length,
        fd_epsilon=0.01,
    )

    output_dir = PROJECT_ROOT / "results" / "enriched_obs_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("ENRICHED OBSERVATION IDENTIFIABILITY AUDIT")
    print("="*70)
    print(f"  KL modes: {cfg.kl_n_modes}")
    print(f"  Families: {args.families}")

    t0 = time.time()
    ladder_cfg = _to_ladder_config(cfg)
    problem, solver, controller, solver_params, forcing = build_problem_and_solver(ladder_cfg)
    truth_state, theta_truth = generate_truth_trajectory(
        ladder_cfg, problem, solver, controller, solver_params,
    )
    obs_bundle = create_observations(
        ladder_cfg, solver, truth_state, controller, solver_params, problem,
    )
    print(f"  Setup time: {time.time() - t0:.1f}s")
    print(f"  Obs points: {obs_bundle['obs_points'].shape[0]}")
    print(f"  Obs times: {obs_bundle['obs_times']}")

    all_results = {"config": {"kl_n_modes": cfg.kl_n_modes, "families": args.families}}

    # Shared forward kwargs
    fwd_kwargs = dict(
        solver=solver, problem=problem, controller=controller,
        solver_params=solver_params, truth_state=truth_state,
    )
    n_params = cfg.n_params
    eps = cfg.fd_epsilon
    obs_times = obs_bundle["obs_times"]

    # ===== BASELINE: pointwise elevation =====
    if "baseline" in args.families:
        print("\n" + "#"*70)
        print("# BASELINE: Pointwise elevation")
        print("#"*70)

        def baseline_obs(traj, times, **kw):
            return np.concatenate([
                extract_h_at_points([traj[t] for t in range(len(traj))], [t], obs_bundle["obs_operator"], solver).ravel()
                for t in times
            ])

        # Actually simpler: use the existing extract_h_at_points
        def baseline_obs_func(traj, times):
            return extract_h_at_points(traj, times, obs_bundle["obs_operator"], solver).ravel()

        J_base, G_base = build_trajectory_sensitivity(
            baseline_obs_func,
            n_params=n_params, eps=eps,
            obs_times=obs_times, cfg=cfg,
            **fwd_kwargs,
        )
        # Whiten
        n_obs_per_t = obs_bundle["obs_points"].shape[0]
        J_base_w = whiten_sensitivity(J_base, obs_bundle["noise_stds"], n_obs_per_t)
        all_results["baseline"] = analyze_and_report(J_base_w, "BASELINE (whitened h)", n_params)

    # ===== FAMILY C: Harmonic projections =====
    if "harmonic" in args.families:
        print("\n" + "#"*70)
        print("# FAMILY C: Harmonic / tidal projections")
        print("#"*70)

        def harmonic_obs_func(traj, times):
            h_ts = extract_h_at_points(traj, times, obs_bundle["obs_operator"], solver)
            return harmonic_observations(h_ts, cfg.dt, cfg.nt_ramp)

        J_harm, G_harm = build_trajectory_sensitivity(
            harmonic_obs_func,
            n_params=n_params, eps=eps,
            obs_times=obs_times, cfg=cfg,
            **fwd_kwargs,
        )
        # No standard whitening for harmonic obs — use raw SVD
        # (harmonic coefficients have mixed units; normalize columns instead)
        col_norms = np.linalg.norm(J_harm, axis=0)
        J_harm_norm = J_harm / np.maximum(col_norms[np.newaxis, :], 1e-30)
        all_results["harmonic_raw"] = analyze_and_report(J_harm, "HARMONIC (raw)", n_params)
        all_results["harmonic_normalized"] = analyze_and_report(J_harm_norm, "HARMONIC (col-normalized)", n_params)

    # ===== FAMILY D: Differential observations =====
    if "differential" in args.families:
        print("\n" + "#"*70)
        print("# FAMILY D: Multi-station differentials")
        print("#"*70)

        def differential_obs_func(traj, times):
            h_ts = extract_h_at_points(traj, times, obs_bundle["obs_operator"], solver)
            return differential_observations(h_ts, n_pairs=50, seed=42)

        J_diff, G_diff = build_trajectory_sensitivity(
            differential_obs_func,
            n_params=n_params, eps=eps,
            obs_times=obs_times, cfg=cfg,
            **fwd_kwargs,
        )
        all_results["differential_raw"] = analyze_and_report(J_diff, "DIFFERENTIAL (raw)", n_params)
        # Normalize
        col_norms = np.linalg.norm(J_diff, axis=0)
        J_diff_norm = J_diff / np.maximum(col_norms[np.newaxis, :], 1e-30)
        all_results["differential_normalized"] = analyze_and_report(J_diff_norm, "DIFFERENTIAL (col-normalized)", n_params)

    # ===== FAMILY A: Flux proxy =====
    if "flux" in args.families:
        print("\n" + "#"*70)
        print("# FAMILY A: Flux proxy (transport differences)")
        print("#"*70)

        def flux_obs_func(traj, times):
            return flux_proxy_observations(
                traj, times, solver, obs_bundle["obs_points"],
                n_transects=20, seed=42,
            )

        J_flux, G_flux = build_trajectory_sensitivity(
            flux_obs_func,
            n_params=n_params, eps=eps,
            obs_times=obs_times, cfg=cfg,
            **fwd_kwargs,
        )
        all_results["flux_raw"] = analyze_and_report(J_flux, "FLUX PROXY (raw)", n_params)
        col_norms = np.linalg.norm(J_flux, axis=0)
        J_flux_norm = J_flux / np.maximum(col_norms[np.newaxis, :], 1e-30)
        all_results["flux_normalized"] = analyze_and_report(J_flux_norm, "FLUX PROXY (col-normalized)", n_params)

    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("COMPARATIVE SUMMARY")
    print("="*70)

    for key in sorted(all_results.keys()):
        if key == "config":
            continue
        r = all_results[key]
        sigma = np.array(r["sigma"])
        col = np.array(r["collinearity"])
        min_offdiag = min(abs(col[i, j]) for i in range(n_params) for j in range(i+1, n_params))
        cond = r["condition"]
        rank5 = int(np.sum(sigma > 0.05 * sigma[0]))
        rank10 = int(np.sum(sigma > 0.10 * sigma[0]))
        print(f"  {key:30s}: σ=[{sigma[0]:.4e}, {sigma[1]:.4e}, {sigma[2]:.4e}]  "
              f"cond={cond:.0e}  rank@5%={rank5}  rank@10%={rank10}  "
              f"min_collin={min_offdiag:.4f}")

    # Save
    all_results["total_time_s"] = time.time() - t0
    results_file = output_dir / "enriched_obs_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    raise SystemExit(main())

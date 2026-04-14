#!/usr/bin/env python3
"""
DC-WME with manually specified component-aware B.

Skips TLM Eq 38 inflation — uses the provided h and uv variances directly.
Runs only DC-WME static, not 4D-Var.

Usage:
  python experiments/run_dcwme_manual_B.py --h-var 0.327 --uv-var 0.0213
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CC", "/usr/bin/clang")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h-var", type=float, required=True, help="h component variance")
    parser.add_argument("--uv-var", type=float, required=True, help="u/v component variance")
    parser.add_argument("--vmax", type=float, default=30.0)
    parser.add_argument("--nt-ramp", type=int, default=144)
    parser.add_argument("--nt-da", type=int, default=12)
    parser.add_argument("--predictability-gamma", type=float, default=0.1)
    args = parser.parse_args()

    import gc
    import time
    import numpy as np
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx import la

    from swe4dvar.forward import ADCIRCProblem, get_solver
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params
    from swe4dvar.data_assimilation import (
        DiagonalCovariance,
        PointObservationOperator,
    )
    from swe4dvar.data_assimilation.cost_functions import DCWMEFourDVarCost
    from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper
    from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig, ForwardModelWrapper
    from experiments.shinnecock_study.run_comparison import _compute_static_L_wme

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    dt = 600.0
    nt_total = args.nt_ramp + args.nt_da
    obs_fraction = 0.1
    obs_frequency = 6
    obs_noise_level = 0.01
    background_error_std = 0.02

    output_dir = PROJECT_ROOT / "results" / "dcwme_manual_B"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    print(f"DC-WME with manual B: h_var={args.h_var}, uv_var={args.uv_var}")

    # --- Wind files (reuse from previous run if available) ---
    wind_dir = PROJECT_ROOT / "results" / "state_dcwme_validation" / "wind"
    truth_wind = wind_dir / f"truth_vmax{args.vmax:.0f}.h5"
    perturbed_wind = wind_dir / f"perturbed_vmax{args.vmax:.0f}.h5"

    if not truth_wind.exists() or not perturbed_wind.exists():
        print("ERROR: Wind files not found. Run run_state_dcwme_validation.py first.")
        return 1

    # --- Build problem + solver using TwinExperiment for proper setup ---
    truth_forcing = GriddedForcing(str(truth_wind), lat0=35)
    perturbed_forcing = GriddedForcing(str(perturbed_wind), lat0=35)

    # Truth problem (for warm-up and truth trajectory)
    prob_truth = ADCIRCProblem(
        adios_file="data/shinnecock_inlet", spherical=True, solution_var="h",
        friction_law="mannings", wd=True, wd_alpha=1.5, dt=dt,
        bathy_adjustment=0, nt=nt_total, dramp=2.0, forcing=truth_forcing,
    )
    solver_truth = get_solver("DG")(prob_truth, theta=1.0, p_degree=[1, 1])
    solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=20, relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=False,
    )

    state_size = solver_truth.V.dofmap.index_map.size_local * solver_truth.V.dofmap.index_map_bs

    # --- Step 1: Warm-up ---
    t0 = time.time()
    print(f"  Step 1: Warm-up ({args.nt_ramp} steps)...")
    prob_truth.nt = args.nt_ramp
    solver_truth.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=False, store_jacobians=False, enable_video=False,
    )
    t_da_start = prob_truth.t
    print(f"  Warm-up done: {time.time()-t0:.0f}s, t={t_da_start:.0f}s")

    # --- Step 2: Truth trajectory (DA window) ---
    print(f"  Step 2: Truth trajectory ({args.nt_da} steps)...")
    solver_truth.storage.clear()
    prob_truth.nt = args.nt_da
    solver_truth.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=True, store_jacobians=True, enable_video=False,
    )
    m_true_arr = solver_truth.u_n.x.array[:state_size].copy()
    m_true = la.create_petsc_vector(solver_truth.V.dofmap.index_map, solver_truth.V.dofmap.index_map_bs)
    m_true.setArray(m_true_arr)
    m_true.assemble()

    truth_trajectory = []
    for s in solver_truth.storage.saved_states:
        vec = la.create_petsc_vector(solver_truth.V.dofmap.index_map, solver_truth.V.dofmap.index_map_bs)
        vec.setArray(s[:state_size])
        vec.assemble()
        truth_trajectory.append(vec)
    truth_jacobians = solver_truth.storage.saved_jacobians.copy()
    print(f"  Truth: {len(truth_trajectory)} states, {len(truth_jacobians)} Jacobians")

    # --- Step 3: DA problem (perturbed wind) ---
    print(f"  Step 3: DA problem setup...")
    prob_da = ADCIRCProblem(
        adios_file="data/shinnecock_inlet", spherical=True, solution_var="h",
        friction_law="mannings", wd=True, wd_alpha=1.5, dt=dt,
        bathy_adjustment=0, nt=args.nt_da, dramp=2.0, forcing=perturbed_forcing,
    )
    solver_da = get_solver("DG")(prob_da, theta=1.0, p_degree=[1, 1])

    # Copy truth state to DA solver as starting point
    solver_da.u_n.x.array[:state_size] = solver_truth.u_n.x.array[:state_size]
    solver_da.u_n_old.x.array[:state_size] = solver_truth.u_n.x.array[:state_size]
    solver_da.u.x.array[:state_size] = solver_truth.u_n.x.array[:state_size]
    solver_da.u_n.x.scatter_forward()
    solver_da.u_n_old.x.scatter_forward()
    solver_da.u.x.scatter_forward()
    prob_da.t = t_da_start

    # --- Step 4: Observations ---
    print(f"  Step 4: Observations...")
    obs_times = list(range(0, args.nt_da + 1, obs_frequency))
    coords = prob_truth.mesh.geometry.x
    all_coords = comm.gather(coords, root=0)
    if rank == 0:
        coords_all = np.vstack(all_coords)
        _, unique_idx = np.unique(np.round(coords_all[:, :2], decimals=10), axis=0, return_index=True)
        unique_coords = coords_all[unique_idx]
        x_min, x_max = unique_coords[:, 0].min(), unique_coords[:, 0].max()
        y_min, y_max = unique_coords[:, 1].min(), unique_coords[:, 1].max()
        interior = unique_coords[
            (unique_coords[:, 0] > x_min + 1e-10) & (unique_coords[:, 0] < x_max - 1e-10) &
            (unique_coords[:, 1] > y_min + 1e-10) & (unique_coords[:, 1] < y_max - 1e-10)
        ]
        rng = np.random.default_rng(42)
        n_obs = max(1, int(len(interior) * obs_fraction))
        chosen = rng.choice(len(interior), size=min(n_obs, len(interior)), replace=False)
        obs_points = np.zeros((len(chosen), 3))
        obs_points[:, :2] = interior[chosen, :2]
    else:
        obs_points = None
    obs_points = comm.bcast(obs_points, root=0)
    obs_operator = PointObservationOperator(solver_da.V, obs_points, comm=comm)
    n_obs_pts = obs_operator.get_num_observations()

    # Generate observations from truth
    obs_vecs = []
    for t_idx in obs_times:
        obs_vec = obs_operator.forward(truth_trajectory[t_idx])
        # Add noise
        arr = obs_vec.getArray()
        rng_noise = np.random.default_rng(42 + 1000 + t_idx)
        noise_std = obs_noise_level * (np.abs(arr).mean() + 1e-10)
        arr += rng_noise.normal(0, noise_std, arr.shape)
        obs_vec.setArray(arr)
        obs_vec.assemble()
        obs_vecs.append(obs_vec)

    obs_variance = obs_noise_level ** 2 * (np.abs(m_true_arr).mean() + 1e-10) ** 2
    R = DiagonalCovariance(comm, size=n_obs_pts, variance=obs_variance)
    print(f"  Obs: {n_obs_pts} points, times={obs_times}, R variance={obs_variance:.6e}")

    # --- Step 5: Background with MANUAL B ---
    print(f"  Step 5: Manual B (h_var={args.h_var}, uv_var={args.uv_var})...")

    # Get component DOF indices
    from experiments.twin_experiment import TwinExperiment
    dummy = object.__new__(TwinExperiment)
    dummy.solver = solver_da
    dummy.problem = prob_da
    dummy.mpi_rank = rank
    dummy.rank = rank
    class _C: verbose = True
    dummy.config = _C()
    h_indices, u_indices, v_indices = dummy._get_component_dof_indices(owned_only=True)

    variances = np.zeros(state_size)
    variances[h_indices] = args.h_var
    variances[u_indices] = args.uv_var
    variances[v_indices] = args.uv_var
    B = DiagonalCovariance(comm, size=state_size, diagonal=variances)
    print(f"  B: min={B.min_eigenvalue():.6e}, h_var={args.h_var}, uv_var={args.uv_var}")

    # Background state (perturbed from truth)
    rng_bg = np.random.default_rng(123)
    scale = np.maximum(np.abs(m_true_arr), 1e-3)
    bg_arr = m_true_arr + rng_bg.normal(0.0, background_error_std * scale)
    m_background = la.create_petsc_vector(solver_da.V.dofmap.index_map, solver_da.V.dofmap.index_map_bs)
    m_background.setArray(bg_arr)
    m_background.assemble()

    bg_rmse = np.sqrt(np.mean((bg_arr - m_true_arr) ** 2))
    print(f"  Background RMSE: {bg_rmse:.6f}")

    # --- Step 6: Static L_wme (NO Eq 38 inflation — B already set correctly) ---
    print(f"  Step 6: Static L_wme (skip_eq38_inflation=True)...")
    static_L_wme, static_diag = _compute_static_L_wme(
        obs_operator, B, len(obs_times), obs_variance,
        m_true, predictability_gamma=args.predictability_gamma,
        adaptive_gamma=True, comm=comm, rank=rank,
        skip_eq38_inflation=True,
    )
    if 'eigvals_regularized' in static_diag:
        eigvals = np.array(static_diag['eigvals_regularized'])
        print(f"  L_wme eigenvalues: [{eigvals.min():.4e}, {eigvals.max():.4e}]")
        print(f"  L_wme natural: {static_diag.get('n_natural', '?')}/{len(eigvals)}")

    # --- Step 7: Build cost function ---
    print(f"  Step 7: DC-WME cost function...")
    forward_model = ForwardModelWrapper(
        solver=solver_da, problem=prob_da,
        solver_params=solver_params, t_start=t_da_start,
    )

    obs_covs = {}
    for t in obs_times:
        obs_covs[t] = DiagonalCovariance(comm, size=n_obs_pts, variance=obs_variance)

    cost_fn = DCWMEFourDVarCost(
        forward_model=forward_model,
        observation_operator=obs_operator,
        background_cov=B,
        observation_cov=obs_covs,
        m_background=m_background,
        observations=obs_vecs,
        obs_times=obs_times,
        predicted_cov_wme=static_L_wme,
        n_l_wme_samples=0,
        auto_inflate_B=False,
        predictability_gamma=args.predictability_gamma,
        adaptive_gamma=True,
        comm=comm,
    )

    # Gradient smoother
    smoothing_matrix = dummy._build_smoothing_matrix(h_indices, 500.0)
    def gradient_smoother(grad_array):
        smoothed = grad_array.copy()
        smoothed[h_indices] = smoothing_matrix @ grad_array[h_indices]
        smoothed[u_indices] = smoothing_matrix @ grad_array[u_indices]
        smoothed[v_indices] = smoothing_matrix @ grad_array[v_indices]
        return smoothed
    cost_fn.gradient_smoother = gradient_smoother

    # --- Step 8: Optimize ---
    print(f"  Step 8: L-BFGS optimization...")
    optimizer = PETScTAOWrapper(
        cost_fn, tao_type="lmvm",
        options={"max_iterations": 15, "max_funcs": 30,
                 "gradient_tolerance": 1e-3, "cost_tolerance": 1e-4,
                 "verbose": True},
    )
    t_opt = time.time()
    m_analysis = optimizer.solve(m_background)
    opt_time = time.time() - t_opt

    analysis_arr = m_analysis.getArray(readonly=True)
    analysis_rmse = np.sqrt(np.mean((analysis_arr - m_true_arr) ** 2))
    improvement = (bg_rmse - analysis_rmse) / bg_rmse * 100

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Background RMSE:  {bg_rmse:.6f}")
    print(f"  Analysis RMSE:    {analysis_rmse:.6f}")
    print(f"  Improvement:      {improvement:.1f}%")
    print(f"  Func evals:       {optimizer.n_func_evals}")
    print(f"  Optimization:     {opt_time:.0f}s")
    print(f"  Total:            {time.time()-t0:.0f}s")

    results = {
        "h_var": args.h_var, "uv_var": args.uv_var,
        "bg_rmse": float(bg_rmse), "analysis_rmse": float(analysis_rmse),
        "improvement_pct": float(improvement),
        "n_func_evals": optimizer.n_func_evals,
        "opt_time_s": float(opt_time),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    raise SystemExit(main())

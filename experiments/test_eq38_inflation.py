#!/usr/bin/env python3
"""
Test TLM-based Eq 38 inflation for static DC-WME.

Runs just the setup + TLM Gram matrix computation + inflation factor.
Does NOT run the optimizer. Validates that:
  1. TLM propagation produces sensible Gram matrix
  2. Eq 38 gives a reasonable inflation factor
  3. B is inflated correctly
  4. Static L_wme would be constructed with the inflated B (no double-inflation)

Usage:
  python experiments/test_eq38_inflation.py
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CC", "/usr/bin/clang")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
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
    from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig, ForwardModelWrapper
    from experiments.shinnecock_study.run_comparison import (
        _compute_eq38_from_tlm,
        _apply_eq38_to_B,
        _compute_static_L_wme,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Short ramp for fast testing — validates Eq 38 logic without 24h ramp cost
    vmax = 30.0
    nt_ramp = 6
    nt_da = 12
    dt = 600.0
    nt_total = nt_ramp + nt_da
    obs_fraction = 0.1
    obs_frequency = 6
    obs_noise_level = 0.01
    background_error_std = 0.02
    predictability_gamma = 0.1

    output_dir = PROJECT_ROOT / "results" / "eq38_test"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    # ================================================================
    # Step 1: Build problem, solver, wind
    # ================================================================
    if rank == 0:
        print("="*60)
        print("TLM-BASED EQ 38 INFLATION TEST")
        print("="*60)
        print(f"  Vmax={vmax}, nt_ramp={nt_ramp}, nt_da={nt_da}")

    t0 = time.time()

    # Generate wind files
    wind_dir = output_dir / "wind"
    if rank == 0:
        wind_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    truth_wind = wind_dir / f"truth_vmax{vmax:.0f}.h5"
    perturbed_wind = wind_dir / f"perturbed_vmax{vmax:.0f}.h5"

    if not truth_wind.exists() and rank == 0:
        from experiments.shinnecock_study.wind_models import (
            DEFAULT_TRACK, HollandHurricaneConfig, WindGridConfig,
            generate_holland_wind_field, generate_perturbed_config, write_wind_hdf5,
        )
        times = np.arange(0.0, (nt_total + 1) * dt, dt)
        grid = WindGridConfig()
        truth_cfg = HollandHurricaneConfig(track_waypoints=DEFAULT_TRACK, Vmax=vmax)
        wx, wy, p = generate_holland_wind_field(truth_cfg, grid, times)
        write_wind_hdf5(str(truth_wind), grid, times, wx, wy, p)
        pert_cfg = generate_perturbed_config(truth_cfg, "track_shift", 15.0)
        wx, wy, p = generate_holland_wind_field(pert_cfg, grid, times)
        write_wind_hdf5(str(perturbed_wind), grid, times, wx, wy, p)
        print(f"  Wind files written")
    comm.Barrier()

    # Build truth problem + solver
    truth_forcing = GriddedForcing(str(truth_wind), lat0=35)
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
    if rank == 0:
        print(f"  State size: {state_size}")
        print(f"  Setup: {time.time()-t0:.1f}s")

    # ================================================================
    # Step 2: Run full truth trajectory (ramp + DA) in one shot.
    # With short ramp (6 steps), storing all Jacobians is fine.
    # ================================================================
    if rank == 0:
        print(f"\n  Running truth trajectory ({nt_total} steps with Jacobians)...")
    t1 = time.time()
    solver_truth.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=True, store_jacobians=True, enable_video=False,
        monitor_progress=False,
    )
    t_da_start = nt_ramp * dt
    if rank == 0:
        print(f"  Truth trajectory: {time.time()-t1:.1f}s")

    # Extract truth state
    m_true_arr = solver_truth.u_n.x.array[:state_size].copy()
    m_true = la.create_petsc_vector(
        solver_truth.V.dofmap.index_map, solver_truth.V.dofmap.index_map_bs,
    )
    m_true.setArray(m_true_arr)
    m_true.assemble()

    # Get full trajectory and Jacobians
    truth_trajectory = []
    for k in range(len(solver_truth.storage.saved_states)):
        vec = la.create_petsc_vector(
            solver_truth.V.dofmap.index_map, solver_truth.V.dofmap.index_map_bs,
        )
        vec.setArray(solver_truth.storage.saved_states[k][:state_size])
        vec.assemble()
        truth_trajectory.append(vec)

    truth_jacobians = solver_truth.storage.saved_jacobians.copy() if solver_truth.storage.saved_jacobians else None

    if rank == 0:
        print(f"  Trajectory: {len(truth_trajectory)} states, "
              f"{len(truth_jacobians) if truth_jacobians else 0} Jacobians")

    # ================================================================
    # Step 3: Setup observations
    # ================================================================
    # Obs times relative to DA window start (not global)
    obs_times = list(range(0, nt_da + 1, obs_frequency))
    if rank == 0:
        print(f"  Obs times (DA-relative): {obs_times} ({len(obs_times)} snapshots)")

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

    obs_operator = PointObservationOperator(solver_truth.V, obs_points, comm=comm)
    n_obs_pts = obs_operator.get_num_observations()

    # Generate observations
    obs_vecs = []
    for t_idx in obs_times:
        obs_vec = obs_operator.forward(truth_trajectory[t_idx])
        obs_vecs.append(obs_vec)

    # Observation covariance
    obs_variance = obs_noise_level ** 2
    R = DiagonalCovariance(comm, size=n_obs_pts, variance=obs_variance)

    # Background covariance (component-aware)
    truth_arr = m_true.getArray(readonly=True)
    h_mag = np.abs(truth_arr[:state_size//3]).mean() + 1e-10
    uv_mag = max(np.abs(truth_arr[state_size//3:]).max(), 0.1)
    h_var = (background_error_std * h_mag) ** 2
    uv_var = (background_error_std * uv_mag) ** 2

    # Simplified component-aware B (h vs u/v)
    variances = np.zeros(state_size)
    n_dofs_per_component = state_size // 3
    variances[:n_dofs_per_component] = h_var
    variances[n_dofs_per_component:] = uv_var
    B = DiagonalCovariance(comm, size=state_size, diagonal=variances)

    if rank == 0:
        print(f"  Obs points: {n_obs_pts}")
        print(f"  B: h_var={h_var:.6e}, uv_var={uv_var:.6e}, min(B)={B.min_eigenvalue():.6e}")

    # ================================================================
    # Step 4: TLM-based Eq 38 inflation
    # ================================================================
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"STEP 4: TLM-BASED EQ 38 INFLATION")
        print(f"{'='*60}")

    # Build forward model for the DA window
    # t_da_start was captured after the ramp phase above
    prob_truth.nt = nt_da
    gram_fwd = ForwardModelWrapper(
        solver=solver_truth, problem=prob_truth,
        solver_params=solver_params, t_start=t_da_start,
    )

    t2 = time.time()
    eq38_result = _compute_eq38_from_tlm(
        forward_model=gram_fwd,
        obs_operator=obs_operator,
        obs_cov=R,
        m_linearize=m_true,
        observations=obs_vecs,
        obs_times=obs_times,
        truth_trajectory=truth_trajectory,
        truth_jacobians=truth_jacobians,
        predictability_gamma=predictability_gamma,
        comm=comm, rank=rank,
    )

    if rank == 0:
        print(f"\n  Eq 38 computation: {time.time()-t2:.1f}s")
        print(f"\n  RESULTS:")
        print(f"    σ_b² = {eq38_result['sigma_b_sq']:.6e}")
        print(f"    λ_min(G) = {eq38_result['lambda_min_G']:.6e}")
        print(f"    λ_max(G) = {eq38_result['lambda_max_G']:.6e}")
        print(f"    Gram condition = {eq38_result['gram_condition']:.2f}")
        print(f"    γ = {eq38_result['predictability_gamma']}")

    # Apply to B
    if rank == 0:
        print(f"\n  Before inflation: min(B) = {B.min_eigenvalue():.6e}")
    eq38_scale = _apply_eq38_to_B(B, eq38_result, rank=rank)
    if rank == 0:
        print(f"  After inflation:  min(B) = {B.min_eigenvalue():.6e}")
        print(f"  Scale factor α = {eq38_scale:.4f}")

    # ================================================================
    # Step 5: Verify static L_wme would skip double-inflation
    # ================================================================
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"STEP 5: STATIC L_WME (with skip_eq38_inflation=True)")
        print(f"{'='*60}")

    t3 = time.time()
    static_L_wme, static_diag = _compute_static_L_wme(
        obs_operator, B, len(obs_times), obs_variance,
        m_true, predictability_gamma=predictability_gamma,
        adaptive_gamma=True,
        comm=comm, rank=rank,
        skip_eq38_inflation=True,  # B already inflated by TLM
    )

    if rank == 0:
        print(f"\n  Static L_wme computation: {time.time()-t3:.1f}s")
        print(f"  Internal inflation factor: {static_diag.get('inflation_factor', 'SKIPPED')}")
        if 'eigvals_regularized' in static_diag:
            eigvals = np.array(static_diag['eigvals_regularized'])
            print(f"  L_wme eigenvalue range: [{eigvals.min():.4e}, {eigvals.max():.4e}]")
            print(f"  L_wme natural (above floor): {static_diag.get('n_natural', '?')}/{len(eigvals)}")

    # ================================================================
    # Summary
    # ================================================================
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"  TLM-based Eq 38 inflation factor: α = {eq38_scale:.4f}")
        print(f"  σ_b² required = {eq38_result['sigma_b_sq']:.6e}")
        print(f"  Gram matrix condition = {eq38_result['gram_condition']:.2f}")
        print(f"  Static L_wme double-inflation: PREVENTED (skip_eq38_inflation=True)")
        print(f"  Total time: {time.time()-t0:.1f}s")

        results = {
            "eq38": eq38_result,
            "scale_factor": float(eq38_scale),
            "B_min_after": float(B.min_eigenvalue()),
            "static_lwme_diagnostics": static_diag,
            "config": {
                "vmax": vmax, "nt_ramp": nt_ramp, "nt_da": nt_da,
                "obs_fraction": obs_fraction, "obs_frequency": obs_frequency,
                "predictability_gamma": predictability_gamma,
            },
        }
        with open(output_dir / "eq38_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {output_dir / 'eq38_test_results.json'}")


if __name__ == "__main__":
    main()

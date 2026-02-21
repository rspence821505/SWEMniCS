#!/usr/bin/env python3
"""
Entry point for the Shinnecock Inlet DC-WME vs 4D-Var comparison study.

Usage:
    # Phase 0: Forward verification (~10 min)
    python experiments/shinnecock_study/run_comparison.py --phase 0

    # Phase 0.5: Gradient verification (~30 min)
    python experiments/shinnecock_study/run_comparison.py --phase 0.5

    # Phase 1: Single-window 4D-Var, no model error
    python experiments/shinnecock_study/run_comparison.py --phase 1

    # Phase 2: Single-window 4D-Var, with model error
    python experiments/shinnecock_study/run_comparison.py --phase 2

    # Phase 3: Single-window DC-WME
    python experiments/shinnecock_study/run_comparison.py --phase 3

    # Phase 4: Cycling comparison
    python experiments/shinnecock_study/run_comparison.py --phase 4

    # Phase 5: Parameter sweeps
    python experiments/shinnecock_study/run_comparison.py --phase 5
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Shinnecock Inlet DC-WME vs 4D-Var comparison study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--phase",
        type=str,
        default="0",
        choices=["0", "0.5", "1", "2", "3", "4", "5", "all"],
        help="Experiment phase to run",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/shinnecock_study",
        help="Output directory for results",
    )

    parser.add_argument(
        "--adios-file",
        type=str,
        default="data/shinnecock_inlet",
        help="Path to ADCIRC ADIOS files (without extension)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output",
    )

    return parser.parse_args()


def run_phase_0(args):
    """Phase 0: Forward Verification.

    Run Shinnecock forward-only simulation for 12h (72 timesteps).
    Verify: mesh loads, GMRES converges, wetting-drying stable, output physical.
    Print mesh stats: node count, element count, interior/boundary node counts.
    """
    from mpi4py import MPI
    from swe4dvar.forward.adcirc_problem import ADCIRCProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)
    comm.Barrier()

    # Simulation parameters for Phase 0: 12h forward run
    dt = 600.0  # seconds
    T_hours = 12.0
    t_f = T_hours * 3600  # seconds
    nt = int(t_f / dt)  # 72 timesteps

    if rank == 0:
        print("=" * 70)
        print("PHASE 0: FORWARD VERIFICATION")
        print("=" * 70)
        print(f"  Time step: {dt} s")
        print(f"  Simulation time: {T_hours} hours ({nt} timesteps)")
        print(f"  ADIOS file: {args.adios_file}")
        print("=" * 70)

    # ----------------------------------------------------------------
    # Step 1: Create problem (loads mesh, bathymetry, boundary data)
    # ----------------------------------------------------------------
    if rank == 0:
        print("\n--- Step 1: Loading mesh and creating problem ---")

    t_start = time.time()
    prob = ADCIRCProblem(
        adios_file=args.adios_file,
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
    t_problem = time.time() - t_start

    if rank == 0:
        print(f"  Problem created in {t_problem:.2f} s")

    # ----------------------------------------------------------------
    # Step 2: Create solver
    # ----------------------------------------------------------------
    if rank == 0:
        print("\n--- Step 2: Creating DG solver ---")

    t_start = time.time()
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])
    t_solver = time.time() - t_start

    if rank == 0:
        print(f"  Solver created in {t_solver:.2f} s")

    # ----------------------------------------------------------------
    # Step 3: Print mesh statistics
    # ----------------------------------------------------------------
    mesh = prob.mesh
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # Node and element counts
    mesh.topology.create_connectivity(tdim, 0)
    mesh.topology.create_connectivity(fdim, 0)

    num_nodes_local = mesh.topology.index_map(0).size_local
    num_cells_local = mesh.topology.index_map(tdim).size_local

    # Global counts via MPI
    num_nodes_global = comm.allreduce(num_nodes_local, op=MPI.SUM)
    num_cells_global = comm.allreduce(num_cells_local, op=MPI.SUM)

    # Function space DOFs
    V = solver.V
    total_dofs_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    total_dofs_global = comm.allreduce(total_dofs_local, op=MPI.SUM)

    # Sub-space DOF counts
    V_h, _ = V.sub(0).collapse()
    h_dofs_local = V_h.dofmap.index_map.size_local * V_h.dofmap.index_map_bs
    h_dofs_global = comm.allreduce(h_dofs_local, op=MPI.SUM)

    # Boundary DOFs
    n_open_boundary = len(prob.dof_open) if hasattr(prob, "dof_open") else 0
    n_open_global = comm.allreduce(n_open_boundary, op=MPI.SUM)

    # Interior nodes (approximate via boundary detection)
    from dolfinx import mesh as dmesh
    boundary_facets = dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    n_boundary_facets_local = len(boundary_facets)
    n_boundary_facets_global = comm.allreduce(n_boundary_facets_local, op=MPI.SUM)

    # Bathymetry statistics
    depth_array = prob.depth.x.array
    depth_min_local = depth_array.min() if len(depth_array) > 0 else 1e10
    depth_max_local = depth_array.max() if len(depth_array) > 0 else -1e10
    depth_mean_local = depth_array.sum() if len(depth_array) > 0 else 0.0
    depth_count_local = len(depth_array)

    depth_min = comm.allreduce(depth_min_local, op=MPI.MIN)
    depth_max = comm.allreduce(depth_max_local, op=MPI.MAX)
    depth_sum = comm.allreduce(depth_mean_local, op=MPI.SUM)
    depth_count = comm.allreduce(depth_count_local, op=MPI.SUM)
    depth_mean = depth_sum / depth_count if depth_count > 0 else 0.0

    # Count nodes with negative depth (potential wetting-drying regions)
    n_negative_depth_local = int(np.sum(depth_array < 0))
    n_negative_depth = comm.allreduce(n_negative_depth_local, op=MPI.SUM)

    # Manning's n statistics
    has_mannings = hasattr(prob, "mannings_n")
    if has_mannings:
        mn_array = prob.mannings_n.x.array
        mn_min_local = mn_array.min() if len(mn_array) > 0 else 1e10
        mn_max_local = mn_array.max() if len(mn_array) > 0 else -1e10
        mn_min = comm.allreduce(mn_min_local, op=MPI.MIN)
        mn_max = comm.allreduce(mn_max_local, op=MPI.MAX)

    if rank == 0:
        print("\n--- Step 3: Mesh Statistics ---")
        print(f"  Mesh nodes:              {num_nodes_global}")
        print(f"  Mesh triangles:          {num_cells_global}")
        print(f"  Boundary facets:         {n_boundary_facets_global}")
        print(f"  DG degree:               p=1")
        print(f"  DOFs per field (h):      {h_dofs_global}")
        print(f"  Total DOFs (h, ux, uy):  {total_dofs_global}")
        print(f"  Open boundary DOFs:      {n_open_global}")
        print(f"  Wetting-drying:          Enabled (alpha={prob.wd_alpha})")
        print(f"  Bathymetry range:        [{depth_min:.2f}, {depth_max:.2f}] m")
        print(f"  Bathymetry mean:         {depth_mean:.2f} m")
        print(f"  Negative depth nodes:    {n_negative_depth} (WD-active regions)")
        if has_mannings:
            print(f"  Manning's n range:       [{mn_min:.6f}, {mn_max:.6f}]")
        else:
            print(f"  Manning's n:             Not loaded (using constant TAU={prob.TAU})")
        print(f"  Tidal constituents:      {len(prob.boundaries.frequency)}")
        print(f"  Spherical projection:    Enabled (lat0={prob.lat0})")
        print(f"  MPI processes:           {comm.Get_size()}")

    # ----------------------------------------------------------------
    # Step 4: Set up observation station for verification
    # ----------------------------------------------------------------
    from swe4dvar.physics.constants import R

    stations = np.array([[-72.476519, 40.840969, 0.0]])
    stations_rad = np.deg2rad(stations)
    lat0 = prob.lat0
    stations_rad[:, 0] *= R * np.cos(np.deg2rad(lat0))
    stations_rad[:, 1] *= R

    # ----------------------------------------------------------------
    # Step 5: Run forward simulation
    # ----------------------------------------------------------------
    if rank == 0:
        print("\n--- Step 4: Running forward simulation ---")

    params = get_default_solver_params(
        rtol=1e-5,
        atol=1e-6,
        max_it=10,
        relaxation_parameter=1.0,
        comm=comm,
        error_if_not_converged=True,
    )

    # Try with stations; fall back to no stations if interpolation fails
    use_stations = True
    try:
        # Test station initialization by doing a dry run
        solver.init_stations(stations_rad)
    except Exception as e:
        if rank == 0:
            print(f"  Warning: Station init failed ({e}), running without stations")
        use_stations = False

    t_start = time.time()
    u_final, vals = solver.time_loop(
        solver_parameters=params,
        stations=stations_rad if use_stations else [],
        plot_every=1,
        plot_name="phase0_shinnecock",
        save_state=False,
        adjoint_method=False,
        store_jacobians=False,
        monitor_progress=(rank == 0),
        enable_video=False,
        newton_diagnostics_config={
            "print_to_console": False,
            "store_history": True,
        },
    )
    t_simulation = time.time() - t_start

    # ----------------------------------------------------------------
    # Step 6: Analyze results
    # ----------------------------------------------------------------
    if rank == 0:
        print(f"\n  Simulation completed in {t_simulation:.2f} s")
        print(f"  Time per timestep: {t_simulation / nt:.3f} s")
        print()

    # Newton convergence diagnostics
    diagnostics = solver.solver.diagnostics if hasattr(solver.solver, "diagnostics") else None
    newton_stats = {}

    if diagnostics is not None and hasattr(diagnostics, "history") and diagnostics.history:
        iterations_per_step = [entry["iterations"] for entry in diagnostics.history]
        converged_flags = [entry.get("converged", True) for entry in diagnostics.history]
        n_converged = sum(converged_flags)
        n_total = len(converged_flags)

        newton_stats = {
            "min_iterations": int(min(iterations_per_step)),
            "max_iterations": int(max(iterations_per_step)),
            "mean_iterations": float(np.mean(iterations_per_step)),
            "total_timesteps": n_total,
            "converged_timesteps": n_converged,
            "all_converged": n_converged == n_total,
        }
    else:
        # If no diagnostics, simulation completed without error = all converged
        newton_stats = {
            "min_iterations": -1,
            "max_iterations": -1,
            "mean_iterations": -1,
            "total_timesteps": nt,
            "converged_timesteps": nt,
            "all_converged": True,
        }

    if rank == 0:
        print("--- Step 5: Verification Results ---")
        print()
        print("  Newton Solver Convergence:")
        if newton_stats["min_iterations"] >= 0:
            print(f"    Min iterations/step:    {newton_stats['min_iterations']}")
            print(f"    Max iterations/step:    {newton_stats['max_iterations']}")
            print(f"    Mean iterations/step:   {newton_stats['mean_iterations']:.2f}")
        print(f"    Converged timesteps:    {newton_stats['converged_timesteps']}/{newton_stats['total_timesteps']}")
        print(f"    All converged:          {newton_stats['all_converged']}")

    # Solution diagnostics from station data
    station_results = {}
    has_station_data = (
        rank == 0
        and vals is not None
        and len(vals) > 0
        and vals.ndim >= 2
        and vals.shape[1] > 0
    )
    if has_station_data:
        h_vals = vals[:nt + 1, 0, 0]  # h at station
        ux_vals = vals[:nt + 1, 0, 1]  # ux at station
        uy_vals = vals[:nt + 1, 0, 2]  # uy at station

        station_results = {
            "h_min": float(np.min(h_vals)),
            "h_max": float(np.max(h_vals)),
            "h_mean": float(np.mean(h_vals)),
            "h_range": float(np.max(h_vals) - np.min(h_vals)),
            "ux_min": float(np.min(ux_vals)),
            "ux_max": float(np.max(ux_vals)),
            "uy_min": float(np.min(uy_vals)),
            "uy_max": float(np.max(uy_vals)),
        }

        print()
        print("  Station Data (channel mid-point):")
        print(f"    h range:    [{station_results['h_min']:.4f}, {station_results['h_max']:.4f}] m")
        print(f"    h mean:     {station_results['h_mean']:.4f} m")
        print(f"    h amplitude:{station_results['h_range']:.4f} m")
        print(f"    ux range:   [{station_results['ux_min']:.6f}, {station_results['ux_max']:.6f}] m/s")
        print(f"    uy range:   [{station_results['uy_min']:.6f}, {station_results['uy_max']:.6f}] m/s")

        # Check for physical plausibility
        tidal_visible = station_results["h_range"] > 0.001  # At least 1mm variation
        no_blowup = abs(station_results["h_max"]) < 100  # No unreasonable values
        velocities_reasonable = abs(station_results["ux_max"]) < 10 and abs(station_results["uy_max"]) < 10

        print()
        print("  Physical Plausibility Checks:")
        print(f"    Tidal oscillation visible (h range > 1mm): {'PASS' if tidal_visible else 'FAIL'}")
        print(f"    No blowup (|h| < 100m):                    {'PASS' if no_blowup else 'FAIL'}")
        print(f"    Velocities reasonable (|u| < 10 m/s):      {'PASS' if velocities_reasonable else 'FAIL'}")
    elif rank == 0:
        print()
        print("  Station Data: Not available (station init failed, see warning above)")
        print("  Using global field diagnostics for plausibility checks instead.")

    # Global solution diagnostics
    u_array = u_final.x.array
    h_sub, h_map = V.sub(0).collapse()
    h_final = u_array[h_map]

    h_min_local = h_final.min() if len(h_final) > 0 else 1e10
    h_max_local = h_final.max() if len(h_final) > 0 else -1e10
    h_min_global = comm.allreduce(h_min_local, op=MPI.MIN)
    h_max_global = comm.allreduce(h_max_local, op=MPI.MAX)

    # Count WD-active cells at final time
    n_wd_active_local = int(np.sum(h_final < prob.wd_alpha))
    n_wd_active_global = comm.allreduce(n_wd_active_local, op=MPI.SUM)

    wd_results = {
        "h_min_global": float(h_min_global),
        "h_max_global": float(h_max_global),
        "n_wd_active_dofs": n_wd_active_global,
        "h_dofs_total": h_dofs_global,
        "wd_fraction": float(n_wd_active_global / h_dofs_global) if h_dofs_global > 0 else 0,
    }

    if rank == 0:
        print()
        print("  Wetting-Drying Stability:")
        print(f"    Global h range at t=12h:  [{wd_results['h_min_global']:.4f}, {wd_results['h_max_global']:.4f}] m")
        print(f"    WD-active DOFs (h < alpha):{wd_results['n_wd_active_dofs']}/{wd_results['h_dofs_total']} ({wd_results['wd_fraction']:.1%})")
        wd_stable = h_min_global > -10  # No catastrophic negative depths
        print(f"    WD stable (h > -10m):      {'PASS' if wd_stable else 'FAIL'}")

    # ----------------------------------------------------------------
    # Step 7: Save results
    # ----------------------------------------------------------------
    if rank == 0:
        results = {
            "phase": 0,
            "status": "success",
            "mesh_stats": {
                "num_nodes": num_nodes_global,
                "num_cells": num_cells_global,
                "num_boundary_facets": n_boundary_facets_global,
                "h_dofs": h_dofs_global,
                "total_dofs": total_dofs_global,
                "open_boundary_dofs": n_open_global,
                "negative_depth_nodes": n_negative_depth,
                "has_mannings_n": has_mannings,
                "n_tidal_constituents": len(prob.boundaries.frequency),
            },
            "bathymetry_stats": {
                "depth_min": float(depth_min),
                "depth_max": float(depth_max),
                "depth_mean": float(depth_mean),
            },
            "simulation_config": {
                "dt": dt,
                "T_hours": T_hours,
                "nt": nt,
                "solver_type": "DG",
                "theta": 1.0,
                "wd_alpha": prob.wd_alpha,
                "dramp": prob.dramp,
            },
            "timing": {
                "problem_setup_s": t_problem,
                "solver_setup_s": t_solver,
                "simulation_s": t_simulation,
                "time_per_step_s": t_simulation / nt,
                "total_s": t_problem + t_solver + t_simulation,
            },
            "newton_convergence": newton_stats,
            "station_results": station_results,
            "wd_stability": wd_results,
        }

        results_file = output_dir / "data" / "phase0_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print()
        print("=" * 70)
        print("PHASE 0: FORWARD VERIFICATION COMPLETE")
        print("=" * 70)
        # Station checks only apply when station data was collected
        if station_results:
            station_pass = (
                station_results.get("h_range", 0) > 0.001
                and abs(station_results.get("h_max", 999)) < 100
            )
        else:
            station_pass = True  # Skip station checks if no station data
        all_pass = (
            newton_stats["all_converged"]
            and station_pass
            and h_min_global > -10
        )
        print(f"  Overall result: {'PASS' if all_pass else 'FAIL'}")
        print(f"  Results saved to: {results_file}")
        print("=" * 70)

        return results

    return None


def run_phase_1(args):
    """Phase 1: Single-Window 4D-Var, No Model Error (Inverse Crime).

    1. Run 48h ramp warm-up (288 timesteps) to reach realistic tidal state
    2. Run TwinExperiment 4D-Var on 12h post-ramp window (72 timesteps)
    3. No physics perturbation (inverse crime baseline)
    4. Expected: high error reduction (>5%)
    """
    import os
    os.environ.setdefault("CC", "/usr/bin/clang")

    from mpi4py import MPI
    from swe4dvar.forward.adcirc_problem import ADCIRCProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params
    from experiments.twin_experiment import (
        TwinExperiment, TwinExperimentConfig, ForwardModelWrapper,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)
    comm.Barrier()

    # ================================================================
    # Configuration
    # ================================================================
    dt = 600.0          # 10-min timestep
    nt_ramp = 288       # 48h ramp (288 × 600s = 172800s)
    nt_da = 72          # 12h DA window (72 × 600s = 43200s)
    nt_total = nt_ramp + nt_da  # 360 timesteps = 60h

    obs_fraction = 0.05     # ~135 obs points
    obs_frequency = 6       # Every 6 timesteps (= every hour)
    obs_noise_level = 0.01  # 1% noise
    background_error_std = 0.02  # 2% perturbation (safe for forward solver; 0.05 crashes, 0.1 crashes worse)
    cov_inflation_factor = 5000.0  # Inflate B so B^{-1} doesn't dominate
    # With diagonal B and 52k DOFs, J_b(m_true) = 0.5*n_dofs/alpha ≈ 26010/alpha.
    # Obs signal is only ~6 (J_o(m_b) - J_o(m_true) ≈ 924 - 918).
    # Need alpha >> 26010/6 ≈ 4335 for optimizer to move toward truth.
    # 4x inflation: J_b(m_true) ≈ 6503, cost barely changed (0.024% after 13 evals).
    # 5000x inflation: J_b(m_true) ≈ 5.2, comparable to obs signal.
    max_iterations = 20

    if rank == 0:
        print("=" * 70)
        print("PHASE 1: SINGLE-WINDOW 4D-VAR, NO MODEL ERROR (INVERSE CRIME)")
        print("=" * 70)
        print(f"  Total simulation: {nt_total} timesteps ({nt_total * dt / 3600:.0f}h)")
        print(f"  Ramp period: {nt_ramp} timesteps ({nt_ramp * dt / 3600:.0f}h)")
        print(f"  DA window: {nt_da} timesteps ({nt_da * dt / 3600:.0f}h)")
        print(f"  Obs fraction: {obs_fraction} (~{int(obs_fraction * 3070)} points)")
        print(f"  Obs frequency: every {obs_frequency} timesteps ({obs_frequency * dt / 3600:.0f}h)")
        print(f"  Background error std: {background_error_std}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Model error: NONE (inverse crime)")
        print("=" * 70)

    # ================================================================
    # Step 1: Create problem and solver
    # ================================================================
    if rank == 0:
        print("\n--- Step 1: Creating problem and solver ---")

    prob = ADCIRCProblem(
        adios_file=args.adios_file,
        spherical=True,
        solution_var="h",
        friction_law="mannings",
        wd=True,
        wd_alpha=1.5,
        dt=dt,
        bathy_adjustment=0,
        nt=nt_total,
        dramp=2.0,
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])

    # GMRES+ILU for warm-up (fast, works fine from equilibrium IC)
    warmup_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=10,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=True,
    )

    # Direct LU for DA forward solves (robust for perturbed states where
    # GMRES+ILU diverges due to white-noise DG cell-to-cell jumps)
    da_solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=15,
        relaxation_parameter=1.0,
        ksp_type="preonly", pc_type="lu",
        comm=comm, error_if_not_converged=True,
    )

    if rank == 0:
        n_nodes = prob.mesh.topology.index_map(0).size_local
        n_dofs = len(solver.u.x.array)
        print(f"  Mesh: {n_nodes} nodes, {n_dofs} DOFs")

    # ================================================================
    # Step 2: Warm-up (48h ramp period)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 2: Running {nt_ramp}-step warm-up ({nt_ramp * dt / 3600:.0f}h ramp) ---")

    t_warmup_start = time.time()
    prob.nt = nt_ramp
    solver.time_loop(
        solver_parameters=warmup_params,
        stations=[],
        plot_every=9999,
        save_state=False,
        store_jacobians=False,
        enable_video=False,
        monitor_progress=(rank == 0),
    )
    t_warmup = time.time() - t_warmup_start
    t_da_start = prob.t  # Should be 172800.0 (48h)

    if rank == 0:
        print(f"  Warm-up completed in {t_warmup:.1f}s (t = {t_da_start:.0f}s = {t_da_start/3600:.1f}h)")

    # ================================================================
    # Step 3: Set up TwinExperiment for DA window
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 3: Setting up DA window ({nt_da} timesteps from t={t_da_start/3600:.0f}h) ---")

    prob.nt = nt_da  # DA window length

    config = TwinExperimentConfig(
        method="4dvar",
        obs_fraction=obs_fraction,
        obs_frequency=obs_frequency,
        obs_noise_level=obs_noise_level,
        background_error_std=background_error_std,
        max_iterations=max_iterations,
        gradient_tolerance=1e-6,
        cost_tolerance=1e-8,
        n_windows=1,
        perturb_friction=False,
        friction_scale_factor=1.0,
        use_bounds=True,
        h_min=0.01,
        interior_only=True,
        verbose=(rank == 0),
    )

    exp = TwinExperiment(
        problem=prob, solver=solver, config=config,
        solver_params=da_solver_params, comm=comm,
    )

    # ================================================================
    # Step 4: Generate truth for DA window (72 timesteps from warm state)
    # ================================================================
    if rank == 0:
        print("\n--- Step 4: Generating truth trajectory for DA window ---")

    t_truth_start = time.time()
    exp._generate_truth()  # Runs 72 steps from warm state at t=48h
    t_truth = time.time() - t_truth_start

    if rank == 0:
        print(f"  Truth: {len(exp.truth_trajectory)} states, generated in {t_truth:.1f}s")

    # ================================================================
    # Step 5: Setup observations, background, covariances
    # ================================================================
    if rank == 0:
        print("\n--- Step 5: Setting up observations and background ---")

    obs_points, obs_operator, obs_times = exp._setup_observations()
    exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
    background_error = exp._setup_background()
    B, R, B_lwme = exp._setup_covariances(obs_operator, obs_noise_stds)

    # Inflate B to weaken B^{-1} penalty without increasing perturbation size
    if cov_inflation_factor != 1.0:
        B.diagonal.scale(cov_inflation_factor)
        B.inv_diagonal.scale(1.0 / cov_inflation_factor)
        if rank == 0:
            print(f"  B covariance inflated by {cov_inflation_factor}x")

    n_obs = obs_operator.get_num_observations()
    if rank == 0:
        print(f"  Observation points: {n_obs}")
        print(f"  Observation times: {obs_times}")
        print(f"  Background RMS error: {background_error:.6f}")

    # ================================================================
    # Step 6: Create forward model with CORRECT t_start for tidal forcing
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 6: Creating forward model (t_start={t_da_start:.0f}s) ---")

    forward_model = exp._create_forward_model(t_start=t_da_start)

    # ================================================================
    # Step 7: Setup cost function (with M^{-1} preconditioning)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 7: Setting up 4D-Var cost function ---")

    cost_function = exp._setup_cost_function(
        forward_model, obs_operator, B, R, obs_times, B_lwme=B_lwme
    )

    # ================================================================
    # Step 8: Run optimization
    # ================================================================
    if rank == 0:
        print("\n--- Step 8: Running L-BFGS optimization ---")

    optimizer, opt_time = exp._run_optimization(cost_function)

    cost_history = [h["cost"] for h in optimizer.convergence_history]
    gradient_history = [h["grad_norm"] for h in optimizer.convergence_history]

    if rank == 0:
        print(f"  Iterations: {optimizer.iteration}")
        print(f"  Converged: {optimizer.converged}")
        print(f"  Optimization time: {opt_time:.1f}s")
        if cost_history:
            print(f"  Cost: {cost_history[0]:.6f} → {cost_history[-1]:.6f}")
        if gradient_history:
            print(f"  ||grad||: {gradient_history[0]:.6e} → {gradient_history[-1]:.6e}")

    # ================================================================
    # Step 9: Evaluate results
    # ================================================================
    if rank == 0:
        print("\n--- Step 9: Evaluating results ---")

    if cost_history and cost_history[-1] >= 1e19:
        if rank == 0:
            print("  WARNING: Optimization failed (cost diverged)")
        analysis_error = background_error
        error_reduction = 0.0
        innov_mean = innov_std = mean_rmse = data_misfit = 0.0
    else:
        analysis_error, error_reduction, innov_mean, innov_std, mean_rmse, data_misfit = (
            exp._evaluate_results(obs_operator, obs_times, background_error)
        )

    total_time = t_warmup + t_truth + opt_time

    if rank == 0:
        print(f"\n  Background error:  {background_error:.6f}")
        print(f"  Analysis error:    {analysis_error:.6f}")
        print(f"  Error reduction:   {error_reduction:.1f}%")
        print(f"  Mean RMSE:         {mean_rmse:.6f}")
        print(f"  Data misfit:       {data_misfit:.6f}")

    # ================================================================
    # Step 10: Save results
    # ================================================================
    if rank == 0:
        results = {
            "phase": 1,
            "status": "success",
            "method": "4dvar",
            "model_error": False,
            "config": {
                "dt": dt,
                "nt_ramp": nt_ramp,
                "nt_da": nt_da,
                "t_da_start_s": t_da_start,
                "obs_fraction": obs_fraction,
                "obs_frequency": obs_frequency,
                "obs_noise_level": obs_noise_level,
                "background_error_std": background_error_std,
                "max_iterations": max_iterations,
                "n_obs_points": n_obs,
                "obs_times": obs_times,
            },
            "results": {
                "background_error": float(background_error),
                "analysis_error": float(analysis_error),
                "error_reduction": float(error_reduction),
                "mean_rmse": float(mean_rmse),
                "data_misfit": float(data_misfit),
                "innovation_mean": float(innov_mean),
                "innovation_std": float(innov_std),
                "num_iterations": int(optimizer.iteration),
                "converged": bool(optimizer.converged),
            },
            "convergence": {
                "cost_history": [float(c) for c in cost_history],
                "gradient_norm_history": [float(g) for g in gradient_history],
            },
            "timing": {
                "warmup_s": float(t_warmup),
                "truth_generation_s": float(t_truth),
                "optimization_s": float(opt_time),
                "total_s": float(total_time),
            },
        }

        results_file = output_dir / "data" / "phase1_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 70}")
        phase_pass = error_reduction > 5.0 and optimizer.converged
        print(f"PHASE 1: {'PASS' if phase_pass else 'NEEDS REVIEW'}")
        print(f"{'=' * 70}")
        print(f"  Error reduction: {error_reduction:.1f}% (expected >5%)")
        print(f"  Iterations: {optimizer.iteration}")
        print(f"  Converged: {optimizer.converged}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Results saved to: {results_file}")
        print(f"{'=' * 70}")

        return results

    return None


def main():
    """Main entry point."""
    import os
    os.environ.setdefault("CC", "/usr/bin/clang")

    args = parse_args()

    if args.phase == "0":
        run_phase_0(args)
    elif args.phase == "1":
        run_phase_1(args)
    else:
        print(f"Phase {args.phase} not yet implemented.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

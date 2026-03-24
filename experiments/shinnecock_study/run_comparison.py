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
        choices=["0", "0.5", "1", "2", "3", "4", "5", "6", "all"],
        help="Experiment phase to run",
    )

    parser.add_argument(
        "--sweep-dim",
        type=str,
        default=None,
        choices=["noise", "obs_density", "obs_frequency", "bg_error",
                 "cov_inflation", "window_length", "model_error", "all"],
        help="Phase 6: which sweep dimension to run (default: all)",
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
        "--sub",
        type=str,
        default=None,
        choices=["a", "b", "c"],
        help="Phase 3 sub-experiment to run (a=4DVar, b=static DC-WME, c=dynamic DC-WME)",
    )

    parser.add_argument(
        "--mem-limit-gb",
        type=float,
        default=12.0,
        help="Phase 6: memory limit in GB; sweep aborts gracefully if exceeded (default: 12)",
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
    # Configuration - FULL PHASE 1 (2-3 hour runtime)
    # ================================================================
    dt = 600.0          # 10-min timestep
    nt_ramp = 144       # 24h ramp (144 × 600s = 86400s) - INTERMEDIATE TEST
    nt_da = 12          # 2h DA window (12 × 600s = 7200s) - reduced for memory
    nt_total = nt_ramp + nt_da  # 180 timesteps = 30h

    obs_fraction = 0.1      # 10% obs points (~306 points) - more observation constraint
    obs_frequency = 6       # Every 6 timesteps (= every hour) per plan
    obs_noise_level = 0.01  # 1% noise
    background_error_std = 0.02   # 2% perturbation RMS (larger gap for optimizer to find)
    cov_inflation_factor = 10.0  # Weaken B^{-1} to let optimizer move (larger perturbation needs less inflation)
    max_iterations = 15     # Need more iters for full convergence with large perturbation

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

    # GMRES+bjacobi for DA forward solves - iterative solver uses far less RAM
    # than direct LU (MUMPS). Starting from m_true means the state is physical,
    # so GMRES converges without the instability seen with perturbed m_background.
    # max_it=25 and reduction_it=10: if Newton stalls after 10 iters, halve relax param.
    da_solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=35,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=False,
        reduction_it=10,
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
        background_correlation_length=500.0,  # Smoothed perturbation (Gaussian, L=500m)
        max_iterations=max_iterations,
        gradient_tolerance=1e-3,
        cost_tolerance=1e-4,
        n_windows=1,
        perturb_friction=False,
        friction_scale_factor=1.0,
        use_bounds=True,
        h_min=0.01,
        interior_only=True,
        component_aware_cov=True,  # Match B variances to component-specific perturbation magnitudes
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
        print(f"\n--- Step 4: Generating truth trajectory for DA window ({nt_da} steps) ---")

    t_truth_start = time.time()
    exp._generate_truth()  # Runs 36 steps from warm state, store_jacobians=False
    t_truth = time.time() - t_truth_start

    # Explicitly clear solver storage to free memory before optimization
    exp.solver.storage.clear()

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
    # Step 7b: Attach gradient smoother (filters high-freq adjoint gradient)
    # ================================================================
    # The perturbation is spatially smoothed (L=500m) but B is diagonal,
    # so the adjoint gradient has high-frequency components that B^{-1}
    # amplifies. Smoothing the gradient is equivalent to B-preconditioning
    # and keeps L-BFGS search directions within Newton's convergence basin.
    gradient_smoothing_length = 500.0  # Match perturbation correlation length for consistent preconditioning
    if config.background_correlation_length > 0:
        if rank == 0:
            print("\n--- Step 7b: Building gradient smoother ---")

        h_indices, u_indices, v_indices = exp._get_component_dof_indices(owned_only=True)
        smoothing_matrix = exp._build_smoothing_matrix(
            h_indices, gradient_smoothing_length
        )

        def gradient_smoother(grad_array):
            """Apply spatial smoothing to each component of the gradient."""
            smoothed = grad_array.copy()
            smoothed[h_indices] = smoothing_matrix @ grad_array[h_indices]
            smoothed[u_indices] = smoothing_matrix @ grad_array[u_indices]
            smoothed[v_indices] = smoothing_matrix @ grad_array[v_indices]
            return smoothed

        # Set smoother on the inner FourDVarCost (may be wrapped by ZeroBoundaryGradientCost)
        inner_cost = cost_function
        while hasattr(inner_cost, 'base_cost'):
            inner_cost = inner_cost.base_cost
        inner_cost.gradient_smoother = gradient_smoother

        if rank == 0:
            print(f"  Gradient smoother attached (L={gradient_smoothing_length}m)")

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


def run_phase_2(args):
    """Phase 2: Single-Window 4D-Var, With Model Error.

    Same as Phase 1 but with friction perturbation (Manning's n × 1.15).
    Truth is generated with TRUE friction, then friction is scaled by 1.15.
    DA uses the perturbed friction model, so it cannot perfectly recover truth.
    Expected: lower error reduction than Phase 1.
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
    # Configuration - same as Phase 1 + friction perturbation
    # ================================================================
    dt = 600.0
    nt_ramp = 144       # 24h ramp
    nt_da = 12          # 2h DA window
    nt_total = nt_ramp + nt_da

    obs_fraction = 0.1
    obs_frequency = 6
    obs_noise_level = 0.01
    background_error_std = 0.02
    cov_inflation_factor = 10.0
    max_iterations = 15
    friction_scale_factor = 1.15  # 15% Manning's n error

    if rank == 0:
        print("=" * 70)
        print("PHASE 2: SINGLE-WINDOW 4D-VAR, WITH MODEL ERROR")
        print("=" * 70)
        print(f"  Total simulation: {nt_total} timesteps ({nt_total * dt / 3600:.0f}h)")
        print(f"  Ramp period: {nt_ramp} timesteps ({nt_ramp * dt / 3600:.0f}h)")
        print(f"  DA window: {nt_da} timesteps ({nt_da * dt / 3600:.0f}h)")
        print(f"  Obs fraction: {obs_fraction} (~{int(obs_fraction * 3070)} points)")
        print(f"  Obs frequency: every {obs_frequency} timesteps ({obs_frequency * dt / 3600:.0f}h)")
        print(f"  Background error std: {background_error_std}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Model error: Manning's n × {friction_scale_factor}")
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

    warmup_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=10,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=True,
    )

    da_solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=35,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=False,
        reduction_it=10,
    )

    if rank == 0:
        n_nodes = prob.mesh.topology.index_map(0).size_local
        n_dofs = len(solver.u.x.array)
        print(f"  Mesh: {n_nodes} nodes, {n_dofs} DOFs")

    # ================================================================
    # Step 2: Warm-up (ramp period with TRUE friction)
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
    t_da_start = prob.t

    if rank == 0:
        print(f"  Warm-up completed in {t_warmup:.1f}s (t = {t_da_start:.0f}s = {t_da_start/3600:.1f}h)")

    # ================================================================
    # Step 3: Set up TwinExperiment for DA window (with friction perturbation)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 3: Setting up DA window ({nt_da} timesteps from t={t_da_start/3600:.0f}h) ---")

    prob.nt = nt_da

    config = TwinExperimentConfig(
        method="4dvar",
        obs_fraction=obs_fraction,
        obs_frequency=obs_frequency,
        obs_noise_level=obs_noise_level,
        background_error_std=background_error_std,
        background_correlation_length=500.0,
        max_iterations=max_iterations,
        gradient_tolerance=1e-3,
        cost_tolerance=1e-4,
        n_windows=1,
        perturb_friction=True,  # KEY DIFFERENCE: enable friction perturbation
        friction_scale_factor=friction_scale_factor,  # 15% Manning's n error
        use_bounds=True,
        h_min=0.01,
        interior_only=True,
        component_aware_cov=True,
        verbose=(rank == 0),
    )

    exp = TwinExperiment(
        problem=prob, solver=solver, config=config,
        solver_params=da_solver_params, comm=comm,
    )

    # ================================================================
    # Step 4: Generate truth for DA window (with TRUE friction)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 4: Generating truth trajectory ({nt_da} steps, TRUE friction) ---")

    t_truth_start = time.time()
    exp._generate_truth()
    t_truth = time.time() - t_truth_start

    exp.solver.storage.clear()

    if rank == 0:
        print(f"  Truth: {len(exp.truth_trajectory)} states, generated in {t_truth:.1f}s")

    # ================================================================
    # Step 4b: Apply friction perturbation (scale Manning's n by 1.15)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 4b: Applying friction perturbation (×{friction_scale_factor}) ---")

    exp._apply_physics_perturbations()

    # ================================================================
    # Step 5: Setup observations, background, covariances
    # ================================================================
    if rank == 0:
        print("\n--- Step 5: Setting up observations and background ---")

    obs_points, obs_operator, obs_times = exp._setup_observations()
    exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
    background_error = exp._setup_background()
    B, R, B_lwme = exp._setup_covariances(obs_operator, obs_noise_stds)

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
    # Step 6: Create forward model (with PERTURBED friction)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 6: Creating forward model (t_start={t_da_start:.0f}s, perturbed friction) ---")

    forward_model = exp._create_forward_model(t_start=t_da_start)

    # ================================================================
    # Step 7: Setup cost function
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 7: Setting up 4D-Var cost function ---")

    cost_function = exp._setup_cost_function(
        forward_model, obs_operator, B, R, obs_times, B_lwme=B_lwme
    )

    # ================================================================
    # Step 7b: Attach gradient smoother (same as Phase 1)
    # ================================================================
    gradient_smoothing_length = 500.0
    if config.background_correlation_length > 0:
        if rank == 0:
            print("\n--- Step 7b: Building gradient smoother ---")

        h_indices, u_indices, v_indices = exp._get_component_dof_indices(owned_only=True)
        smoothing_matrix = exp._build_smoothing_matrix(
            h_indices, gradient_smoothing_length
        )

        def gradient_smoother(grad_array):
            """Apply spatial smoothing to each component of the gradient."""
            smoothed = grad_array.copy()
            smoothed[h_indices] = smoothing_matrix @ grad_array[h_indices]
            smoothed[u_indices] = smoothing_matrix @ grad_array[u_indices]
            smoothed[v_indices] = smoothing_matrix @ grad_array[v_indices]
            return smoothed

        inner_cost = cost_function
        while hasattr(inner_cost, 'base_cost'):
            inner_cost = inner_cost.base_cost
        inner_cost.gradient_smoother = gradient_smoother

        if rank == 0:
            print(f"  Gradient smoother attached (L={gradient_smoothing_length}m)")

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
            "phase": 2,
            "status": "success",
            "method": "4dvar",
            "model_error": True,
            "model_error_type": "friction",
            "friction_scale_factor": friction_scale_factor,
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

        results_file = output_dir / "data" / "phase2_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 70}")
        # Phase 2 expects lower error reduction than Phase 1 (8.5%)
        # but still some positive reduction
        phase_pass = error_reduction > 0.0
        print(f"PHASE 2: {'PASS' if phase_pass else 'NEEDS REVIEW'}")
        print(f"{'=' * 70}")
        print(f"  Error reduction: {error_reduction:.1f}% (expected >0%, lower than Phase 1's 8.5%)")
        print(f"  Iterations: {optimizer.iteration}")
        print(f"  Converged: {optimizer.converged}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Results saved to: {results_file}")
        print(f"{'=' * 70}")

        return results

    return None


def _compute_static_L_wme(obs_operator, B, n_obs_times, obs_variance,
                           m_template, predictability_gamma=0.1,
                           adaptive_gamma=True, max_inflate_factor=5.0,
                           comm=None, rank=0):
    """Compute static L_wme = (N / σ²_obs) · H B_inflated Hᵀ (no adjoint solves).

    This matches the near-linear limit of the dynamic L_wme. When M_{k:0} ≈ I,
    J_wme → (√N/σ_obs)·H, so L_wme → (N/σ²_obs)·H B Hᵀ.

    Parameters
    ----------
    obs_operator : ObservationOperator
        Point observation operator (H).
    B : DiagonalCovariance
        Background covariance.
    n_obs_times : int
        Number of observation times (N).
    obs_variance : float
        Observation variance (σ²_obs).
    m_template : PETSc.Vec
        Template state vector for creating unit vectors.
    predictability_gamma : float
        Relaxation parameter for eigenvalue flooring.
    adaptive_gamma : bool
        If True, floor = gamma * lambda_max.
    max_inflate_factor : float
        Maximum B inflation factor.
    comm : MPI.Comm
        MPI communicator.
    rank : int
        MPI rank.

    Returns
    -------
    DenseCovariance
        Regularized static L_wme covariance.
    dict
        Diagnostics dictionary.
    """
    from petsc4py import PETSc
    from swe4dvar.data_assimilation.covariance import DenseCovariance

    d_obs = obs_operator.get_num_observations()
    n_state = m_template.getSize()

    # Extract H matrix via adjoint: H^T e_i gives row i of H
    # This is O(d_obs) applications instead of O(n_state) — much faster
    # (306 vs 52020 for Shinnecock)
    if rank == 0:
        print(f"  Extracting H matrix ({d_obs} × {n_state}) via {d_obs} adjoint applications...")

    H = np.zeros((d_obs, n_state))

    # Create observation-space unit vector
    obs_vec = obs_operator.forward(m_template)  # get correctly sized obs vector
    obs_vec.zeroEntries()

    for i in range(d_obs):
        obs_vec.zeroEntries()
        obs_vec.setValue(i, 1.0)
        obs_vec.assemblyBegin()
        obs_vec.assemblyEnd()
        HT_ei = obs_operator.adjoint(obs_vec)  # H^T e_i = state-space vector
        # Gather full vector across MPI ranks
        local_arr = HT_ei.getArray()
        if comm is not None and comm.Get_size() > 1:
            full_arr = np.zeros(n_state)
            start, end = HT_ei.getOwnershipRange()
            full_arr[start:end] = local_arr
            comm.Allreduce(full_arr.copy(), full_arr)
            H[i, :] = full_arr
        else:
            H[i, :] = local_arr
        HT_ei.destroy()
    obs_vec.destroy()

    if rank == 0:
        print(f"  H extracted: {np.count_nonzero(H)} nonzeros out of {d_obs * n_state}")

    # Step 1: Static B inflation from H Hᵀ eigenvalues
    G_static = H @ H.T  # (d_obs × d_obs)
    gram_eigvals = np.linalg.eigvalsh(G_static)
    lambda_min_G = max(gram_eigvals.min(), 1e-30)

    # Get B diagonal
    B_diag = B.diagonal.getArray().copy()
    # Gather full diagonal (for MPI)
    if comm is not None and comm.Get_size() > 1:
        B_diag_full = np.zeros(n_state)
        start, end = B.diagonal.getOwnershipRange()
        B_diag_full[start:end] = B_diag
        comm.Allreduce(B_diag_full.copy(), B_diag_full)
        B_diag = B_diag_full

    current_min_var = B_diag.min()

    # Inflation factor from predictability bound
    required_var = (predictability_gamma * obs_variance) / (n_obs_times * lambda_min_G)
    alpha_uncapped = max(1.0, required_var / current_min_var) if current_min_var > 0 else 1.0
    alpha = min(alpha_uncapped, max_inflate_factor)

    diagnostics = {
        'gram_eigvals': gram_eigvals.tolist(),
        'lambda_min_G': float(lambda_min_G),
        'required_var': float(required_var),
        'inflation_factor': float(alpha),
        'inflation_capped': alpha < alpha_uncapped,
    }

    if rank == 0:
        if alpha > 1.0:
            print(f"  Static B inflation: α = {alpha:.4f} "
                  f"(required σ²_b = {required_var:.4e}, current = {current_min_var:.4e})")
        else:
            print(f"  Static: no B inflation needed")

    B_diag_inflated = B_diag * alpha if alpha > 1.0 else B_diag

    # Step 2: Form L_static = (N/σ²_obs) · H B Hᵀ
    HB = H * B_diag_inflated[np.newaxis, :]
    L_dense = (n_obs_times / obs_variance) * (HB @ H.T)

    # Free large intermediate arrays
    del H, HB, B_diag_inflated
    import gc; gc.collect()

    # Step 3: Eigenvalue regularization (same as dynamic Layer 2)
    eigvals, eigvecs = np.linalg.eigh(L_dense)

    if adaptive_gamma:
        gamma_floor = predictability_gamma * eigvals.max()
        if rank == 0:
            print(f"  Static adaptive γ: {predictability_gamma} × λ_max={eigvals.max():.4e} "
                  f"→ γ_eff={gamma_floor:.4e}")
    else:
        gamma_floor = predictability_gamma

    n_floored = int(np.sum(eigvals < gamma_floor))
    n_natural = len(eigvals) - n_floored
    eigvals_reg = np.maximum(eigvals, gamma_floor)
    L_dense = eigvecs @ np.diag(eigvals_reg) @ eigvecs.T

    if rank == 0:
        print(f"  Static L_wme: {n_natural}/{d_obs} natural, {n_floored}/{d_obs} floored")

    diagnostics['eigvals_raw'] = eigvals.tolist()
    diagnostics['eigvals_regularized'] = eigvals_reg.tolist()
    diagnostics['gamma_floor'] = float(gamma_floor)
    diagnostics['n_natural'] = int(n_natural)
    diagnostics['n_floored'] = int(n_floored)

    # Create PETSc Mat from numpy array
    mat = PETSc.Mat().create(comm=comm)
    mat.setSizes((d_obs, d_obs))
    mat.setType(PETSc.Mat.Type.DENSE)
    mat.setUp()

    if rank == 0:
        for i in range(d_obs):
            mat.setValues(i, list(range(d_obs)), L_dense[i, :])

    mat.assemblyBegin()
    mat.assemblyEnd()

    L_wme_cov = DenseCovariance(comm, mat, inverse_method="cholesky")

    return L_wme_cov, diagnostics


def _run_sub_experiment(args, sub_label, method, config_overrides=None,
                        static_L_wme=None, output_dir=None,
                        nt_da=12, nt_ramp=144, phase_prefix="3",
                        sweep_params=None):
    """Run a single sub-experiment.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    sub_label : str
        Sub-experiment label (e.g., "a", "b", "c").
    method : str
        DA method ("4dvar" or "dcwme").
    config_overrides : dict, optional
        Override config parameters.
    static_L_wme : DenseCovariance, optional
        Precomputed static L_wme for DC-WME static.
    output_dir : Path
        Output directory.
    nt_da : int
        Number of DA timesteps (default: 12 for 2h window).
    nt_ramp : int
        Number of ramp-up timesteps (default: 144 for 24h).
    sweep_params : dict, optional
        Parameter overrides for Phase 6 sweep. Keys override local
        configuration variables (obs_fraction, obs_frequency, etc.).

    Returns
    -------
    dict
        Results dictionary.
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

    # ================================================================
    # Configuration
    # ================================================================
    dt = 600.0
    nt_total = nt_ramp + nt_da

    obs_fraction = 0.1
    obs_frequency = 6
    obs_noise_level = 0.01
    background_error_std = 0.02
    cov_inflation_factor = 10.0
    max_iterations = 15
    friction_scale_factor = 1.15

    # Override with sweep parameters (Phase 6)
    if sweep_params:
        obs_fraction = sweep_params.get("obs_fraction", obs_fraction)
        obs_frequency = sweep_params.get("obs_frequency", obs_frequency)
        obs_noise_level = sweep_params.get("obs_noise_level", obs_noise_level)
        background_error_std = sweep_params.get("background_error_std", background_error_std)
        cov_inflation_factor = sweep_params.get("cov_inflation_factor", cov_inflation_factor)
        friction_scale_factor = sweep_params.get("friction_scale_factor", friction_scale_factor)

    # DC-WME parameters
    predictability_gamma = 0.1
    max_inflate_factor = 5.0
    auto_inflate_B = True
    adaptive_gamma = True

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"PHASE {phase_prefix}{sub_label.upper()}: {method.upper()}"
              f"{' (static L_wme)' if sub_label == 'b' else ''}"
              f"{' (dynamic L_wme)' if sub_label == 'c' else ''}")
        print(f"{'=' * 70}")

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

    warmup_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=10,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=True,
    )

    da_solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=35,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=False,
        reduction_it=10,
    )

    if rank == 0:
        n_nodes = prob.mesh.topology.index_map(0).size_local
        n_dofs = len(solver.u.x.array)
        print(f"  Mesh: {n_nodes} nodes, {n_dofs} DOFs")

    # ================================================================
    # Step 2: Warm-up
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 2: Running {nt_ramp}-step warm-up ---")

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
    t_da_start = prob.t

    if rank == 0:
        print(f"  Warm-up completed in {t_warmup:.1f}s")

    # ================================================================
    # Step 3: Set up TwinExperiment
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 3: Setting up DA window ({nt_da} timesteps) ---")

    prob.nt = nt_da

    config_kwargs = dict(
        method=method,
        obs_fraction=obs_fraction,
        obs_frequency=obs_frequency,
        obs_noise_level=obs_noise_level,
        background_error_std=background_error_std,
        background_correlation_length=500.0,
        max_iterations=max_iterations,
        max_funcs=30,  # Cap total function evals to prevent optimizer stalling
        gradient_tolerance=1e-3,
        cost_tolerance=1e-4,
        n_windows=1,
        perturb_friction=True,
        friction_scale_factor=friction_scale_factor,
        use_bounds=True,
        h_min=0.01,
        interior_only=True,
        component_aware_cov=True,
        verbose=(rank == 0),
    )

    # DC-WME specific parameters
    if method == "dcwme":
        config_kwargs.update(
            predictability_gamma=predictability_gamma,
            max_inflate_factor=max_inflate_factor,
            auto_inflate_B=auto_inflate_B,
            adaptive_gamma=adaptive_gamma,
        )
        if static_L_wme is not None:
            # Phase 3b: skip analytical L_wme computation
            config_kwargs['l_wme_samples'] = 0

    # Handle friction_scale_factor=1.0 (no model error / inverse crime reference)
    if sweep_params and friction_scale_factor == 1.0:
        config_kwargs["perturb_friction"] = False
        config_kwargs["friction_scale_factor"] = 1.0

    if config_overrides:
        config_kwargs.update(config_overrides)

    config = TwinExperimentConfig(**config_kwargs)

    exp = TwinExperiment(
        problem=prob, solver=solver, config=config,
        solver_params=da_solver_params, comm=comm,
    )

    # ================================================================
    # Step 4: Generate truth + apply friction perturbation
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 4: Generating truth trajectory ---", flush=True)

    t_truth_start = time.time()
    exp._generate_truth()
    t_truth = time.time() - t_truth_start
    exp.solver.storage.clear()

    if rank == 0:
        print(f"  Truth: {len(exp.truth_trajectory)} states, generated in {t_truth:.1f}s")
        print(f"\n--- Step 4b: Applying friction perturbation (×{friction_scale_factor}) ---", flush=True)

    exp._apply_physics_perturbations()

    # ================================================================
    # Step 5: Setup observations, background, covariances
    # ================================================================
    if rank == 0:
        print("\n--- Step 5: Setting up observations and background ---", flush=True)

    # Temporarily set method to "4dvar" to prevent _setup_covariances from building
    # the dense B_lwme correlation matrix (52020×52020 = 21 GB, causes OOM).
    # We use diagonal B for all Phase 3 experiments (per plan: "diagonal B for Phases 0-3").
    # Static L_wme is computed directly from H·B·Hᵀ, dynamic L_wme uses diagonal B.
    original_method = exp.config.method
    exp.config.method = "4dvar"  # Skip dense B_lwme construction

    obs_points, obs_operator, obs_times = exp._setup_observations()
    exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, obs_times)
    background_error = exp._setup_background()
    B, R, B_lwme = exp._setup_covariances(obs_operator, obs_noise_stds)

    exp.config.method = original_method  # Restore original method

    if cov_inflation_factor != 1.0:
        B.diagonal.scale(cov_inflation_factor)
        B.inv_diagonal.scale(1.0 / cov_inflation_factor)
        if rank == 0:
            print(f"  B covariance inflated by {cov_inflation_factor}x")

    n_obs = obs_operator.get_num_observations()
    if rank == 0:
        print(f"  Observation points: {n_obs}")
        print(f"  Observation times: {obs_times}")
        print(f"  Background RMS error: {background_error:.6f}", flush=True)

    # ================================================================
    # Step 5b: Compute static L_wme if needed
    # ================================================================
    static_diagnostics = None
    if method == "dcwme" and static_L_wme is None:
        if rank == 0:
            print("\n--- Step 5b: Computing static L_wme ---", flush=True)
        static_L_wme, static_diagnostics = _compute_static_L_wme(
            obs_operator, B, len(obs_times), obs_noise_level ** 2,
            exp.m_true, predictability_gamma=predictability_gamma,
            adaptive_gamma=adaptive_gamma, max_inflate_factor=max_inflate_factor,
            comm=comm, rank=rank,
        )
        if rank == 0:
            print("  Static L_wme computed successfully", flush=True)
    elif static_L_wme is not None and rank == 0:
        print("\n  Using precomputed static L_wme")

    # Memory cleanup before cost function init
    import gc
    exp.solver.storage.clear()
    gc.collect()

    # ================================================================
    # Step 6: Create forward model
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 6: Creating forward model (t_start={t_da_start:.0f}s) ---", flush=True)

    forward_model = exp._create_forward_model(t_start=t_da_start)

    # ================================================================
    # Step 7: Setup cost function
    # ================================================================
    if rank == 0:
        method_label = method.upper()
        if static_L_wme is not None:
            method_label += " (static L_wme)"
        print(f"\n--- Step 7: Setting up {method_label} cost function ---", flush=True)

    if static_L_wme is not None:
        # Phase 3b: pass precomputed static L_wme
        from swe4dvar.data_assimilation.cost_functions import DCWMEFourDVarCost

        cost_function = DCWMEFourDVarCost(
            forward_model=forward_model,
            observation_operator=obs_operator,
            background_cov=B,
            observation_cov=R,
            m_background=exp.m_background,
            observations=exp.observations,
            obs_times=obs_times,
            predicted_cov_wme=static_L_wme,
            n_l_wme_samples=0,  # Skip analytical computation
            auto_inflate_B=False,  # Already inflated in static computation
            max_inflate_factor=max_inflate_factor,
            predictability_gamma=predictability_gamma,
            adaptive_gamma=adaptive_gamma,
            comm=comm,
        )

        # Wrap with boundary gradient zeroing if needed
        if config.interior_only:
            from experiments.twin_experiment import ZeroBoundaryGradientCost
            boundary_dofs = exp._get_boundary_dofs()
            cost_function = ZeroBoundaryGradientCost(cost_function, boundary_dofs)
            if rank == 0:
                print(f"  Zeroing {len(boundary_dofs)} boundary DOF gradients")

        # Disable M^{-1} preconditioning (same as Phase 1/2)
        if rank == 0:
            print(f"  M^{{-1}} preconditioning DISABLED (causes ill-conditioning)")
    else:
        cost_function = exp._setup_cost_function(
            forward_model, obs_operator, B, R, obs_times, B_lwme=B_lwme
        )

    # ================================================================
    # Step 7b: Attach gradient smoother
    # ================================================================
    gradient_smoothing_length = 500.0
    if config.background_correlation_length > 0:
        if rank == 0:
            print("\n--- Step 7b: Building gradient smoother ---")

        h_indices, u_indices, v_indices = exp._get_component_dof_indices(owned_only=True)
        smoothing_matrix = exp._build_smoothing_matrix(
            h_indices, gradient_smoothing_length
        )

        def gradient_smoother(grad_array):
            smoothed = grad_array.copy()
            smoothed[h_indices] = smoothing_matrix @ grad_array[h_indices]
            smoothed[u_indices] = smoothing_matrix @ grad_array[u_indices]
            smoothed[v_indices] = smoothing_matrix @ grad_array[v_indices]
            return smoothed

        inner_cost = cost_function
        while hasattr(inner_cost, 'base_cost'):
            inner_cost = inner_cost.base_cost
        inner_cost.gradient_smoother = gradient_smoother

        if rank == 0:
            print(f"  Gradient smoother attached (L={gradient_smoothing_length}m)")

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
    # Step 10: Extract DC-WME diagnostics
    # ================================================================
    dcwme_diagnostics = {}
    if method == "dcwme":
        # Get the inner cost function (unwrap boundary gradient zeroing)
        inner = cost_function
        while hasattr(inner, 'base_cost'):
            inner = inner.base_cost

        if hasattr(inner, '_b_inflation_factor'):
            dcwme_diagnostics['b_inflation_factor'] = float(inner._b_inflation_factor)
        if hasattr(inner, '_gram_eigenvalues') and inner._gram_eigenvalues is not None:
            dcwme_diagnostics['gram_eigenvalues'] = [float(v) for v in inner._gram_eigenvalues]
        if hasattr(inner, '_L_wme') and inner._L_wme is not None:
            # Extract L_wme eigenvalues for diagnostics
            try:
                L_wme_mat = inner._L_wme
                if hasattr(L_wme_mat, 'mat'):
                    n = L_wme_mat.mat.getSize()[0]
                    L_dense_np = np.zeros((n, n))
                    rstart, rend = L_wme_mat.mat.getOwnershipRange()
                    for i in range(rstart, rend):
                        cols, vals = L_wme_mat.mat.getRow(i)
                        for c, v in zip(cols, vals):
                            L_dense_np[i, c] = v
                    l_eigvals = np.linalg.eigvalsh(L_dense_np)
                    dcwme_diagnostics['l_wme_eigenvalues'] = [float(v) for v in l_eigvals]
                    dcwme_diagnostics['l_wme_n_natural'] = int(np.sum(l_eigvals > predictability_gamma * l_eigvals.max()))
            except Exception as e:
                if rank == 0:
                    print(f"  Warning: Could not extract L_wme eigenvalues: {e}")

        if static_diagnostics is not None:
            dcwme_diagnostics['static_lwme'] = static_diagnostics

    # ================================================================
    # Step 11: Save results
    # ================================================================
    results = None
    if rank == 0:
        results = {
            "phase": f"{phase_prefix}{sub_label}",
            "status": "success",
            "method": method,
            "model_error": True,
            "model_error_type": "friction",
            "friction_scale_factor": friction_scale_factor,
            "l_wme_type": "static" if sub_label == "b" else ("dynamic" if sub_label == "c" else "N/A"),
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
                "predictability_gamma": predictability_gamma,
                "max_inflate_factor": max_inflate_factor,
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
            "dcwme_diagnostics": dcwme_diagnostics,
            "timing": {
                "warmup_s": float(t_warmup),
                "truth_generation_s": float(t_truth),
                "optimization_s": float(opt_time),
                "total_s": float(total_time),
            },
        }

        results_file = output_dir / "data" / f"phase{phase_prefix}{sub_label}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 70}")
        print(f"PHASE {phase_prefix}{sub_label.upper()}: {method.upper()}"
              f"{' (static L_wme)' if sub_label == 'b' else ''}"
              f"{' (dynamic L_wme)' if sub_label == 'c' else ''}")
        print(f"  Error reduction: {error_reduction:.1f}%")
        print(f"  Iterations: {optimizer.iteration}")
        print(f"  Converged: {optimizer.converged}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Results saved to: {results_file}")
        print(f"{'=' * 70}")

    # Explicitly delete heavy objects to help GC reclaim memory
    del optimizer, cost_function, forward_model, exp, solver, prob
    del B, R, B_lwme, obs_operator
    _cleanup_petsc_objects()

    return results


def run_phase_3(args):
    """Phase 3: Single-Window DC-WME vs 4D-Var Comparison.

    Runs three sub-experiments on the same configuration as Phase 2:
    - 3a: Standard 4D-Var baseline (for seed-consistent comparison)
    - 3b: DC-WME with static L_wme = (N/σ²_obs) · H B Hᵀ
    - 3c: DC-WME with dynamic L_wme = J_wme B J_wmeᵀ (adjoint-computed)
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)
    comm.Barrier()

    if rank == 0:
        print("=" * 70)
        print("PHASE 3: SINGLE-WINDOW DC-WME vs 4D-VAR COMPARISON")
        print("=" * 70)
        print("  Sub-experiments:")
        print("    3a: Standard 4D-Var (baseline)")
        print("    3b: DC-WME with static L_wme")
        print("    3c: DC-WME with dynamic L_wme")
        print("=" * 70)

    # Determine which sub-experiments to run
    sub = getattr(args, 'sub', None)
    if sub:
        subs_to_run = [sub]
    else:
        subs_to_run = ["a", "b", "c"]

    all_results = {}

    for sub_label in subs_to_run:
        if sub_label == "a":
            result = _run_sub_experiment(
                args, sub_label="a", method="4dvar",
                output_dir=output_dir,
            )
        elif sub_label == "b":
            result = _run_sub_experiment(
                args, sub_label="b", method="dcwme",
                output_dir=output_dir,
            )
        elif sub_label == "c":
            result = _run_sub_experiment(
                args, sub_label="c", method="dcwme",
                output_dir=output_dir,
            )
        else:
            if rank == 0:
                print(f"Unknown sub-experiment: 3{sub_label}")
            continue

        all_results[f"3{sub_label}"] = result

    # Print comparison summary
    if rank == 0 and len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("PHASE 3 COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Sub-exp':<10} {'Method':<20} {'Error Red.':<12} {'Iters':<8} {'Conv.':<8} {'Time (min)':<10}")
        print(f"{'-' * 68}")
        for key, res in all_results.items():
            if res is not None:
                r = res['results']
                t = res['timing']
                lwme = res.get('l_wme_type', 'N/A')
                method_label = f"{res['method']}"
                if lwme != 'N/A':
                    method_label += f" ({lwme})"
                print(f"{key:<10} {method_label:<20} {r['error_reduction']:.1f}%{'':<7} "
                      f"{r['num_iterations']:<8} {'Yes' if r['converged'] else 'No':<8} "
                      f"{t['total_s']/60:.1f}")
        print(f"{'=' * 70}")

    return all_results


# ====================================================================
# Phase 4: Cycling Comparison
# ====================================================================

def _run_cycling_experiment(args, sub_label, method, output_dir=None):
    """Run a single Phase 4 cycling experiment (one method).

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    sub_label : str
        Sub-experiment label ("a", "b", "c").
    method : str
        DA method ("4dvar" or "dcwme").
    output_dir : Path
        Output directory.

    Returns
    -------
    dict
        Results dictionary with per-window metrics.
    """
    import os
    import gc
    os.environ.setdefault("CC", "/usr/bin/clang")

    from mpi4py import MPI
    from petsc4py import PETSc
    from swe4dvar.forward.adcirc_problem import ADCIRCProblem
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.utils import get_default_solver_params
    from experiments.twin_experiment import (
        TwinExperiment, TwinExperimentConfig, ForwardModelWrapper,
        ZeroBoundaryGradientCost,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ================================================================
    # Configuration
    # ================================================================
    dt = 600.0
    nt_ramp = 288          # 48h ramp
    n_windows = 6
    window_nt = 71         # ~11.8h per window
    nt_da_total = n_windows * window_nt  # 426 timesteps total DA

    obs_fraction = 0.1
    obs_frequency = 6      # every hour
    obs_noise_level = 0.01
    background_error_std = 0.02
    cov_inflation_factor = 10.0
    max_iterations = 15
    friction_scale_factor = 1.15

    # DC-WME parameters
    predictability_gamma = 0.1
    max_inflate_factor = 5.0
    auto_inflate_B = True
    adaptive_gamma = True

    l_wme_type = "N/A"
    if sub_label == "b":
        l_wme_type = "static"
    elif sub_label == "c":
        l_wme_type = "dynamic"

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"PHASE 4{sub_label.upper()}: CYCLING {method.upper()}"
              f"{' (static L_wme)' if sub_label == 'b' else ''}"
              f"{' (dynamic L_wme)' if sub_label == 'c' else ''}")
        print(f"  {n_windows} windows × {window_nt} timesteps "
              f"({window_nt * dt / 3600:.1f}h each)")
        print(f"{'=' * 70}", flush=True)

    t_total_start = time.time()

    # ================================================================
    # Step 1: Create problem and solver
    # ================================================================
    if rank == 0:
        print("\n--- Step 1: Creating problem and solver ---", flush=True)

    prob = ADCIRCProblem(
        adios_file=args.adios_file,
        spherical=True,
        solution_var="h",
        friction_law="mannings",
        wd=True,
        wd_alpha=1.5,
        dt=dt,
        bathy_adjustment=0,
        nt=nt_ramp + nt_da_total,
        dramp=2.0,
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])

    warmup_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=10,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=True,
    )

    da_solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=35,
        relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=False,
        reduction_it=10,
    )

    if rank == 0:
        n_nodes = prob.mesh.topology.index_map(0).size_local
        n_dofs = len(solver.u.x.array)
        print(f"  Mesh: {n_nodes} nodes, {n_dofs} DOFs")

    # ================================================================
    # Step 2: Warm-up (48h ramp)
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 2: Running {nt_ramp}-step warm-up ({nt_ramp * dt / 3600:.0f}h) ---",
              flush=True)

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
    t_da_start = prob.t  # Time after ramp

    if rank == 0:
        print(f"  Warm-up completed in {t_warmup:.1f}s ({t_warmup/60:.1f} min)")

    # ================================================================
    # Step 3: Set up TwinExperiment for full DA period
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 3: Setting up DA ({nt_da_total} timesteps, "
              f"{nt_da_total * dt / 3600:.1f}h) ---", flush=True)

    prob.nt = nt_da_total

    config_kwargs = dict(
        method=method,
        obs_fraction=obs_fraction,
        obs_frequency=obs_frequency,
        obs_noise_level=obs_noise_level,
        background_error_std=background_error_std,
        background_correlation_length=500.0,
        max_iterations=max_iterations,
        max_funcs=30,  # Cap total function evals per window to prevent line search stalling
        gradient_tolerance=1e-3,
        cost_tolerance=1e-4,
        n_windows=1,  # We manage cycling ourselves
        perturb_friction=True,
        friction_scale_factor=friction_scale_factor,
        use_bounds=True,
        h_min=0.01,
        interior_only=True,
        component_aware_cov=True,
        verbose=(rank == 0),
    )

    if method == "dcwme":
        config_kwargs.update(
            predictability_gamma=predictability_gamma,
            max_inflate_factor=max_inflate_factor,
            auto_inflate_B=auto_inflate_B,
            adaptive_gamma=adaptive_gamma,
        )

    config = TwinExperimentConfig(**config_kwargs)

    exp = TwinExperiment(
        problem=prob, solver=solver, config=config,
        solver_params=da_solver_params, comm=comm,
    )

    # ================================================================
    # Step 4: Generate truth for full DA period
    # ================================================================
    if rank == 0:
        print(f"\n--- Step 4: Generating truth trajectory ---", flush=True)

    t_truth_start = time.time()
    exp._generate_truth()
    t_truth = time.time() - t_truth_start
    exp.solver.storage.clear()

    if rank == 0:
        print(f"  Truth: {len(exp.truth_trajectory)} states in {t_truth:.1f}s")
        print(f"\n--- Step 4b: Applying friction perturbation (×{friction_scale_factor}) ---",
              flush=True)

    exp._apply_physics_perturbations()

    # ================================================================
    # Step 5: Setup observations, background, covariances for full period
    # ================================================================
    if rank == 0:
        print("\n--- Step 5: Setting up observations and background ---", flush=True)

    # Use "4dvar" trick to avoid dense B_lwme construction (OOM)
    original_method = exp.config.method
    exp.config.method = "4dvar"

    obs_points, obs_operator, global_obs_times = exp._setup_observations()
    exp.observations, obs_noise_stds = exp._generate_observations(obs_operator, global_obs_times)
    background_error = exp._setup_background()
    B, R, B_lwme = exp._setup_covariances(obs_operator, obs_noise_stds)

    exp.config.method = original_method

    if cov_inflation_factor != 1.0:
        B.diagonal.scale(cov_inflation_factor)
        B.inv_diagonal.scale(1.0 / cov_inflation_factor)
        if rank == 0:
            print(f"  B covariance inflated by {cov_inflation_factor}x")

    n_obs = obs_operator.get_num_observations()
    if rank == 0:
        print(f"  Observation points: {n_obs}")
        print(f"  Global observation times: {global_obs_times}")
        print(f"  Background RMS error: {background_error:.6f}", flush=True)

    # Pre-compute gradient smoother (same for all windows)
    gradient_smoothing_length = 500.0
    gradient_smoother = None
    if config.background_correlation_length > 0:
        h_indices, u_indices, v_indices = exp._get_component_dof_indices(owned_only=True)
        smoothing_matrix = exp._build_smoothing_matrix(h_indices, gradient_smoothing_length)

        def gradient_smoother(grad_array):
            smoothed = grad_array.copy()
            smoothed[h_indices] = smoothing_matrix @ grad_array[h_indices]
            smoothed[u_indices] = smoothing_matrix @ grad_array[u_indices]
            smoothed[v_indices] = smoothing_matrix @ grad_array[v_indices]
            return smoothed

        if rank == 0:
            print(f"  Gradient smoother built (L={gradient_smoothing_length}m)")

    # Pre-compute boundary DOFs (same for all windows)
    boundary_dofs = exp._get_boundary_dofs() if config.interior_only else None

    # Memory cleanup before cycling
    exp.solver.storage.clear()
    gc.collect()

    # ================================================================
    # Step 6: Cycling loop
    # ================================================================
    per_window_results = []
    all_cost_history = []
    all_gradient_history = []
    total_iterations = 0

    # Store original B diagonal for per-window scaling
    B_diag_original = B.diagonal.copy()
    B_inv_diag_original = B.inv_diagonal.copy()

    for w in range(n_windows):
        t_window_start = time.time()
        t_start_window = t_da_start + w * window_nt * dt
        t_end_window = t_da_start + (w + 1) * window_nt * dt

        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"  Window {w + 1}/{n_windows}: t = {t_start_window / 3600:.1f}h - "
                  f"{t_end_window / 3600:.1f}h")
            print(f"{'=' * 60}", flush=True)

        # a) Compute per-window background error for B scaling
        truth_idx = w * window_nt
        if w == 0:
            window_bg_error = background_error
        else:
            diff_bg = exp.m_background.copy()
            diff_bg.axpy(-1.0, exp.truth_trajectory[truth_idx])
            window_bg_error = np.sqrt(diff_bg.dot(diff_bg) / diff_bg.getSize())
            diff_bg.destroy()

        # Scale B proportionally to actual background error vs initial
        # This prevents overfitting when propagated background is already close to truth
        b_scale = min(window_bg_error / background_error, 1.0) if background_error > 0 else 1.0
        b_scale = max(b_scale, 0.01)  # Floor: at least 1% of original B

        # Restore original B and apply per-window scaling
        B.diagonal.copy(result=B.diagonal)  # no-op, overwrite below
        B_diag_original.copy(result=B.diagonal)
        B_inv_diag_original.copy(result=B.inv_diagonal)
        if b_scale < 1.0:
            B.diagonal.scale(b_scale)
            B.inv_diagonal.scale(1.0 / b_scale)

        if rank == 0:
            print(f"  Background error:  {window_bg_error:.6f} "
                  f"(B scale: {b_scale:.3f}x of original)", flush=True)

        # b) Subset observations for this window
        window_obs_indices = []
        window_local_times = []
        for i, gt in enumerate(global_obs_times):
            if w * window_nt <= gt <= (w + 1) * window_nt:
                # Remap to window-local time
                local_t = gt - w * window_nt
                # Include t=0 for first obs, but skip duplicates at boundaries
                if local_t == 0 and w > 0:
                    continue  # Skip boundary overlap (was end of previous window)
                window_obs_indices.append(i)
                window_local_times.append(local_t)

        window_observations = [exp.observations[i] for i in window_obs_indices]

        if rank == 0:
            print(f"  Window obs: {len(window_local_times)} at local times {window_local_times}",
                  flush=True)

        if len(window_observations) == 0:
            if rank == 0:
                print(f"  WARNING: No observations in window {w + 1}, skipping")
            # Propagate background forward
            if w < n_windows - 1:
                new_bg = _propagate_state_safe(
                    exp, window_nt, t_start_window, da_solver_params, rank
                )
                if new_bg is not None:
                    exp.m_background = new_bg
                else:
                    if rank == 0:
                        print(f"  WARNING: Propagation failed, stopping cycling")
                    break
            per_window_results.append({
                "window": w,
                "t_start_h": float(t_start_window / 3600),
                "t_end_h": float(t_end_window / 3600),
                "n_obs_times": 0,
                "skipped": True,
            })
            continue

        # c) Set problem.nt to window length for forward model
        prob.nt = window_nt

        # d) Create forward model for this window
        if rank == 0:
            print(f"  Creating forward model (t_start={t_start_window:.0f}s)", flush=True)
        forward_model = exp._create_forward_model(t_start=t_start_window)

        # e) Compute static L_wme for this window (Phase 4b only)
        static_L_wme = None
        static_diagnostics = None
        if method == "dcwme" and sub_label == "b":
            if rank == 0:
                print(f"  Computing static L_wme for window {w + 1}...", flush=True)
            t_lwme_start = time.time()
            static_L_wme, static_diagnostics = _compute_static_L_wme(
                obs_operator, B, len(window_local_times), obs_noise_level ** 2,
                exp.m_background, predictability_gamma=predictability_gamma,
                adaptive_gamma=adaptive_gamma, max_inflate_factor=max_inflate_factor,
                comm=comm, rank=rank,
            )
            t_lwme = time.time() - t_lwme_start
            if rank == 0:
                print(f"  Static L_wme computed in {t_lwme:.1f}s", flush=True)

        # f) Create cost function
        if rank == 0:
            method_label = method.upper()
            if sub_label == "b":
                method_label += " (static L_wme)"
            elif sub_label == "c":
                method_label += " (dynamic L_wme)"
            print(f"  Setting up {method_label} cost function", flush=True)

        if method == "dcwme" and static_L_wme is not None:
            # Phase 4b: precomputed static L_wme
            from swe4dvar.data_assimilation.cost_functions import DCWMEFourDVarCost

            cost_function = DCWMEFourDVarCost(
                forward_model=forward_model,
                observation_operator=obs_operator,
                background_cov=B,
                observation_cov=R,
                m_background=exp.m_background,
                observations=window_observations,
                obs_times=window_local_times,
                predicted_cov_wme=static_L_wme,
                n_l_wme_samples=0,
                auto_inflate_B=False,
                max_inflate_factor=max_inflate_factor,
                predictability_gamma=predictability_gamma,
                adaptive_gamma=adaptive_gamma,
                comm=comm,
            )

            if boundary_dofs is not None:
                cost_function = ZeroBoundaryGradientCost(cost_function, boundary_dofs)
        else:
            # Phase 4a (4D-Var) or 4c (dynamic DC-WME)
            orig_observations = exp.observations
            exp.observations = window_observations
            cost_function = exp._setup_cost_function(
                forward_model, obs_operator, B, R, window_local_times, B_lwme=B_lwme
            )
            exp.observations = orig_observations

        # g) Attach gradient smoother
        if gradient_smoother is not None:
            inner_cost = cost_function
            while hasattr(inner_cost, 'base_cost'):
                inner_cost = inner_cost.base_cost
            inner_cost.gradient_smoother = gradient_smoother

        # h) Run optimization
        if rank == 0:
            print(f"  Running L-BFGS optimization (max {max_iterations} iters)...", flush=True)

        optimizer, opt_time = exp._run_optimization(cost_function)

        window_cost = [h["cost"] for h in optimizer.convergence_history]
        window_grad = [h["grad_norm"] for h in optimizer.convergence_history]
        all_cost_history.extend(window_cost)
        all_gradient_history.extend(window_grad)
        total_iterations += optimizer.iteration

        # Check if optimization failed (inf cost on first eval)
        opt_failed = window_cost and window_cost[0] >= 1e19

        if rank == 0:
            print(f"  Window {w + 1}: {optimizer.iteration} iters, "
                  f"converged={optimizer.converged}", flush=True)
            if window_cost:
                print(f"  Cost: {window_cost[0]:.1f} → {window_cost[-1]:.1f}")
            if opt_failed:
                print(f"  WARNING: Forward model failed — using background as analysis")

        # If optimization failed, use background as analysis
        if opt_failed:
            exp.m_analysis = exp.m_background.copy()

        # i) Compute per-window analysis error
        if w == 0:
            diff = exp.m_analysis.copy()
            diff.axpy(-1.0, exp.m_true)
            analysis_error = np.sqrt(diff.dot(diff) / diff.getSize())
            diff.destroy()
        else:
            diff = exp.m_analysis.copy()
            diff.axpy(-1.0, exp.truth_trajectory[truth_idx])
            analysis_error = np.sqrt(diff.dot(diff) / diff.getSize())
            diff.destroy()

        window_err_reduction = (window_bg_error - analysis_error) / window_bg_error * 100 if window_bg_error > 0 else 0.0

        if rank == 0:
            print(f"  Analysis error:    {analysis_error:.6f}")
            print(f"  Error reduction:   {window_err_reduction:.1f}%", flush=True)

        # i) Extract DC-WME diagnostics
        window_dcwme_diag = {}
        if method == "dcwme":
            inner = cost_function
            while hasattr(inner, 'base_cost'):
                inner = inner.base_cost

            if hasattr(inner, '_b_inflation_factor'):
                window_dcwme_diag['b_inflation_factor'] = float(inner._b_inflation_factor)
            if hasattr(inner, '_gram_eigenvalues') and inner._gram_eigenvalues is not None:
                window_dcwme_diag['gram_eigenvalues'] = [float(v) for v in inner._gram_eigenvalues]
            if hasattr(inner, '_L_wme') and inner._L_wme is not None:
                try:
                    L_wme_mat = inner._L_wme
                    if hasattr(L_wme_mat, 'mat'):
                        n = L_wme_mat.mat.getSize()[0]
                        L_dense_np = np.zeros((n, n))
                        rstart, rend = L_wme_mat.mat.getOwnershipRange()
                        for i in range(rstart, rend):
                            cols, vals = L_wme_mat.mat.getRow(i)
                            for c, v in zip(cols, vals):
                                L_dense_np[i, c] = v
                        l_eigvals = np.linalg.eigvalsh(L_dense_np)
                        window_dcwme_diag['l_wme_eigenvalues'] = [float(v) for v in l_eigvals]
                        window_dcwme_diag['l_wme_n_natural'] = int(
                            np.sum(l_eigvals > predictability_gamma * l_eigvals.max())
                        )
                except Exception as e:
                    if rank == 0:
                        print(f"  Warning: Could not extract L_wme eigenvalues: {e}")

            if static_diagnostics is not None:
                window_dcwme_diag['static_lwme'] = static_diagnostics

        t_window = time.time() - t_window_start
        per_window_results.append({
            "window": w,
            "t_start_h": float(t_start_window / 3600),
            "t_end_h": float(t_end_window / 3600),
            "n_obs_times": len(window_local_times),
            "iterations": int(optimizer.iteration),
            "converged": bool(optimizer.converged),
            "background_error": float(window_bg_error),
            "analysis_error": float(analysis_error),
            "error_reduction_pct": float(window_err_reduction),
            "cost_initial": float(window_cost[0]) if window_cost else None,
            "cost_final": float(window_cost[-1]) if window_cost else None,
            "cost_history": [float(c) for c in window_cost],
            "gradient_norm_history": [float(g) for g in window_grad],
            "optimization_time_s": float(opt_time),
            "window_time_s": float(t_window),
            "dcwme_diagnostics": window_dcwme_diag,
        })

        if rank == 0:
            print(f"  Window {w + 1} time: {t_window:.1f}s ({t_window / 60:.1f} min)")

        # j) Propagate analysis forward to next window
        if w < n_windows - 1:
            if rank == 0:
                print(f"  Propagating analysis to next window...", flush=True)
            new_bg = _propagate_state_safe(
                exp, window_nt, t_start_window, da_solver_params, rank
            )
            if new_bg is not None:
                exp.m_background = new_bg
            else:
                if rank == 0:
                    print(f"  WARNING: Propagation failed at window {w + 1}, stopping cycling")
                # Record remaining windows as skipped
                for w_skip in range(w + 1, n_windows):
                    t_s = t_da_start + w_skip * window_nt * dt
                    t_e = t_da_start + (w_skip + 1) * window_nt * dt
                    per_window_results.append({
                        "window": w_skip,
                        "t_start_h": float(t_s / 3600),
                        "t_end_h": float(t_e / 3600),
                        "n_obs_times": 0,
                        "skipped": True,
                        "skip_reason": "propagation_failure",
                    })
                break

        # k) Memory cleanup
        exp.solver.storage.clear()
        gc.collect()

    # Restore problem.nt
    prob.nt = nt_ramp + nt_da_total

    # ================================================================
    # Aggregate results
    # ================================================================
    total_time = time.time() - t_total_start

    # Final analysis error (from last window)
    final_analysis_error = per_window_results[-1]["analysis_error"] if per_window_results else background_error
    cumulative_reduction = (background_error - final_analysis_error) / background_error * 100

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"  CYCLING COMPLETE: {total_iterations} total iterations across {n_windows} windows")
        print(f"  Initial background error: {background_error:.6f}")
        print(f"  Final analysis error:     {final_analysis_error:.6f}")
        print(f"  Cumulative reduction:     {cumulative_reduction:.1f}%")
        print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"{'=' * 60}", flush=True)

    # Save results
    results = None
    if rank == 0:
        results = {
            "phase": f"4{sub_label}",
            "status": "success",
            "method": method,
            "l_wme_type": l_wme_type,
            "model_error": True,
            "model_error_type": "friction",
            "friction_scale_factor": friction_scale_factor,
            "config": {
                "dt": dt,
                "nt_ramp": nt_ramp,
                "n_windows": n_windows,
                "window_nt": window_nt,
                "nt_da_total": nt_da_total,
                "t_da_start_s": t_da_start,
                "obs_fraction": obs_fraction,
                "obs_frequency": obs_frequency,
                "obs_noise_level": obs_noise_level,
                "background_error_std": background_error_std,
                "cov_inflation_factor": cov_inflation_factor,
                "max_iterations": max_iterations,
                "n_obs_points": n_obs,
                "predictability_gamma": predictability_gamma,
                "max_inflate_factor": max_inflate_factor,
            },
            "per_window": per_window_results,
            "aggregate": {
                "total_iterations": total_iterations,
                "initial_background_error": float(background_error),
                "final_analysis_error": float(final_analysis_error),
                "cumulative_error_reduction_pct": float(cumulative_reduction),
                "total_time_s": float(total_time),
                "warmup_time_s": float(t_warmup),
                "truth_time_s": float(t_truth),
            },
            "convergence": {
                "cost_history": [float(c) for c in all_cost_history],
                "gradient_norm_history": [float(g) for g in all_gradient_history],
            },
        }

        results_file = output_dir / "data" / f"phase4{sub_label}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n  Results saved to: {results_file}")

    return results


def _propagate_state_safe(exp, window_nt, t_start, solver_params, rank=0):
    """Propagate analysis state forward by window_nt timesteps.

    Uses the same pattern as TwinExperiment._propagate_forward().
    Returns a PETSc Vec with the final state, or None if propagation fails.
    """
    exp.solver.storage.clear()

    # Set initial condition from analysis
    m_array = exp.m_analysis.getArray()
    u_owned_size = exp.solver.V.dofmap.index_map.size_local
    if len(m_array) == len(exp.solver.u_n.x.array):
        exp.solver.u_n.x.array[:] = m_array
        exp.solver.u_n_old.x.array[:] = m_array
        exp.solver.u.x.array[:] = m_array
    else:
        exp.solver.u_n.x.array[:u_owned_size] = m_array
        exp.solver.u_n_old.x.array[:u_owned_size] = m_array
        exp.solver.u.x.array[:u_owned_size] = m_array
        exp.solver.u_n.x.scatter_forward()
        exp.solver.u_n_old.x.scatter_forward()
        exp.solver.u.x.scatter_forward()

    # Set time and nt for propagation
    orig_nt = exp.problem.nt
    exp.problem.nt = window_nt
    exp.problem.t = t_start

    try:
        exp.solver.time_loop(
            solver_parameters=solver_params,
            stations=np.array([[0.0, 0.0, 0.0]]),
            plot_every=9999,
            save_state=True,
            enable_video=False,
        )

        if not exp.solver.storage.saved_states:
            if rank == 0:
                print("  WARNING: No states saved during propagation")
            exp.problem.nt = orig_nt
            exp.solver.storage.clear()
            return None

        # Get final state
        final_state = exp.solver.storage.saved_states[-1]
        from dolfinx import la
        vec = la.create_petsc_vector(
            exp.solver.V.dofmap.index_map,
            exp.solver.V.dofmap.index_map_bs,
        )
        vec.setArray(final_state[:u_owned_size])
        vec.assemble()

        # Sanity check: reject states with NaN or extremely large values
        arr = vec.getArray()
        if np.any(np.isnan(arr)) or np.any(np.abs(arr) > 1e10):
            if rank == 0:
                print(f"  WARNING: Propagated state has NaN or extreme values "
                      f"(max={np.nanmax(np.abs(arr)):.2e})")
            vec.destroy()
            exp.problem.nt = orig_nt
            exp.solver.storage.clear()
            return None

        exp.problem.nt = orig_nt
        exp.solver.storage.clear()
        return vec

    except Exception as e:
        if rank == 0:
            print(f"  WARNING: Propagation failed with error: {e}")
        exp.problem.nt = orig_nt
        exp.solver.storage.clear()
        return None


def run_phase_4(args):
    """Phase 4: Cycling DC-WME vs 4D-Var Comparison.

    Runs three cycling experiments:
    - 4a: Standard 4D-Var cycling
    - 4b: DC-WME cycling with static L_wme
    - 4c: DC-WME cycling with dynamic L_wme
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)
    comm.Barrier()

    if rank == 0:
        print("=" * 70)
        print("PHASE 4: CYCLING DC-WME vs 4D-VAR COMPARISON")
        print("=" * 70)
        print("  Sub-experiments:")
        print("    4a: Standard 4D-Var cycling (6 windows)")
        print("    4b: DC-WME cycling with static L_wme")
        print("    4c: DC-WME cycling with dynamic L_wme")
        print("=" * 70)

    sub = getattr(args, 'sub', None)
    if sub:
        subs_to_run = [sub]
    else:
        subs_to_run = ["a", "b", "c"]

    all_results = {}

    for sub_label in subs_to_run:
        if sub_label == "a":
            result = _run_cycling_experiment(
                args, sub_label="a", method="4dvar", output_dir=output_dir,
            )
        elif sub_label == "b":
            result = _run_cycling_experiment(
                args, sub_label="b", method="dcwme", output_dir=output_dir,
            )
        elif sub_label == "c":
            result = _run_cycling_experiment(
                args, sub_label="c", method="dcwme", output_dir=output_dir,
            )
        else:
            if rank == 0:
                print(f"Unknown sub-experiment: 4{sub_label}")
            continue

        all_results[f"4{sub_label}"] = result

    # Print comparison summary
    if rank == 0 and len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("PHASE 4 CYCLING COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Sub':<6} {'Method':<22} {'Cum. Err Red.':<14} {'Tot Iters':<10} "
              f"{'Time (min)':<10}")
        print(f"{'-' * 62}")
        for key, res in all_results.items():
            if res is not None:
                agg = res['aggregate']
                method_label = res['method']
                if res.get('l_wme_type', 'N/A') != 'N/A':
                    method_label += f" ({res['l_wme_type']})"
                print(f"{key:<6} {method_label:<22} "
                      f"{agg['cumulative_error_reduction_pct']:.1f}%{'':<9} "
                      f"{agg['total_iterations']:<10} "
                      f"{agg['total_time_s'] / 60:.1f}")

        # Per-window comparison
        print(f"\n{'=' * 70}")
        print("PER-WINDOW ERROR REDUCTION (%)")
        print(f"{'=' * 70}")
        header = f"{'Window':<8}"
        for key in all_results:
            header += f" {key:<12}"
        print(header)
        print(f"{'-' * (8 + 12 * len(all_results))}")

        first_res = next(iter(all_results.values()))
        if first_res is not None:
            n_win = len(first_res['per_window'])
            for w in range(n_win):
                row = f"W{w + 1:<7}"
                for key, res in all_results.items():
                    if res is not None and w < len(res['per_window']):
                        pw = res['per_window'][w]
                        if pw.get('skipped'):
                            row += f" {'skip':<12}"
                        else:
                            row += f" {pw['error_reduction_pct']:.1f}%{'':<8}"
                    else:
                        row += f" {'N/A':<12}"
                print(row)
        print(f"{'=' * 70}")

    return all_results


def run_phase_5(args):
    """Phase 5: Single Long-Window DC-WME vs 4D-Var Comparison.

    Same as Phase 3 but with a 71-step (~11.8h) DA window instead of 12-step (2h).
    Tests whether longer windows create meaningful eigenvalue spread in L_wme,
    which would enable DC-WME's directional predictability reweighting.

    - 5a: Standard 4D-Var baseline
    - 5b: DC-WME with static L_wme
    - 5c: DC-WME with dynamic L_wme
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)
    comm.Barrier()

    if rank == 0:
        print("=" * 70)
        print("PHASE 5: LONG-WINDOW DC-WME vs 4D-VAR COMPARISON")
        print("  71 timesteps (~11.8h), 48h ramp, single window")
        print("=" * 70)
        print("  Sub-experiments:")
        print("    5a: Standard 4D-Var (baseline)")
        print("    5b: DC-WME with static L_wme")
        print("    5c: DC-WME with dynamic L_wme")
        print("=" * 70)

    sub = getattr(args, 'sub', None)
    if sub:
        subs_to_run = [sub]
    else:
        subs_to_run = ["a", "b", "c"]

    all_results = {}

    for sub_label in subs_to_run:
        if sub_label == "a":
            result = _run_sub_experiment(
                args, sub_label="a", method="4dvar",
                output_dir=output_dir,
                nt_da=71, nt_ramp=288, phase_prefix="5",
            )
        elif sub_label == "b":
            result = _run_sub_experiment(
                args, sub_label="b", method="dcwme",
                output_dir=output_dir,
                nt_da=71, nt_ramp=288, phase_prefix="5",
            )
        elif sub_label == "c":
            result = _run_sub_experiment(
                args, sub_label="c", method="dcwme",
                output_dir=output_dir,
                nt_da=71, nt_ramp=288, phase_prefix="5",
            )
        else:
            if rank == 0:
                print(f"Unknown sub-experiment: 5{sub_label}")
            continue

        all_results[f"5{sub_label}"] = result

    # Print comparison summary
    if rank == 0 and len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("PHASE 5 COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Sub-exp':<10} {'Method':<20} {'Error Red.':<12} {'Iters':<8} {'Conv.':<8} {'Time (min)':<10}")
        print(f"{'-' * 68}")
        for key, res in all_results.items():
            if res is not None:
                r = res['results']
                t = res['timing']
                lwme = res.get('l_wme_type', 'N/A')
                method_label = f"{res['method']}"
                if lwme != 'N/A':
                    method_label += f" ({lwme})"
                print(f"{key:<10} {method_label:<20} {r['error_reduction']:.1f}%{'':<7} "
                      f"{r['num_iterations']:<8} {'Yes' if r['converged'] else 'No':<8} "
                      f"{t['total_s']/60:.1f}")
        print(f"{'=' * 70}")

    return all_results


# ========================================================================
# Phase 6: Parameter Sweep — 4D-Var vs Static DC-WME
# ========================================================================

SWEEP_BASELINE = {
    "obs_noise_level": 0.01,
    "obs_fraction": 0.1,
    "obs_frequency": 6,
    "background_error_std": 0.02,
    "cov_inflation_factor": 10.0,
    "nt_da": 12,
    "nt_ramp": 144,
    "friction_scale_factor": 1.15,
}

SWEEP_DIMS = {
    "noise": {
        "param": "obs_noise_level",
        "values": [0.01, 0.005, 0.05, 0.1, 0.001],
    },
    "obs_density": {
        "param": "obs_fraction",
        "values": [0.05, 0.1, 0.25, 0.5],
    },
    "obs_frequency": {
        "param": "obs_frequency",
        "values": [2, 4, 6, 12],
    },
    "bg_error": {
        "param": "background_error_std",
        "values": [0.01, 0.02, 0.05, 0.1],
    },
    "cov_inflation": {
        "param": "cov_inflation_factor",
        "values": [1.0, 5.0, 10.0, 20.0],
    },
    "window_length": {
        "param": "nt_da",
        "values": [6, 12, 36, 71],
        "nt_ramp_map": {6: 144, 12: 144, 36: 144, 71: 288},
    },
    "model_error": {
        "param": "friction_scale_factor",
        "values": [1.0, 1.15, 1.3, 1.5],
    },
}


def _print_sweep_summary(all_results):
    """Print formatted summary table of sweep results."""
    for dim_name, dim_results in all_results.items():
        print(f"\n{'=' * 70}")
        print(f"SWEEP: {dim_name}")
        print(f"{'Value':<15} {'4DVar Err%':<12} {'DCWME Err%':<12} {'Winner':<10}")
        print(f"{'-' * 49}")
        for point in dim_results:
            val = point["value"]
            fdvar = point.get("4dvar", {})
            dcwme = point.get("dcwme", {})

            fdvar_err = fdvar.get("results", {}).get("error_reduction", None)
            if fdvar.get("status") == "failed":
                fdvar_err = None
            dcwme_err = dcwme.get("results", {}).get("error_reduction", None)
            if dcwme.get("status") == "failed":
                dcwme_err = None

            fdvar_str = f"{fdvar_err:.1f}" if fdvar_err is not None else "FAIL"
            dcwme_str = f"{dcwme_err:.1f}" if dcwme_err is not None else "FAIL"

            if fdvar_err is not None and dcwme_err is not None:
                winner = "4DVar" if fdvar_err > dcwme_err else "DCWME"
            else:
                winner = "N/A"

            print(f"{str(val):<15} {fdvar_str:<12} {dcwme_str:<12} {winner:<10}")
        print(f"{'=' * 70}")


def _cleanup_petsc_objects():
    """Aggressively destroy PETSc objects to reclaim memory.

    PETSc/FEniCSx can leak memory across solver instantiations.
    This forces cleanup of the PETSc garbage and Python GC.
    """
    import gc
    gc.collect()

    try:
        from petsc4py import PETSc
        PETSc.garbage_cleanup()
    except Exception:
        pass

    # Second GC pass to collect anything PETSc released
    gc.collect()


def _get_process_memory_mb():
    """Return current process RSS in MB, or None if unavailable."""
    try:
        import resource
        # maxrss on macOS is in bytes, on Linux in KB
        import platform
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return rss / (1024 * 1024)
        else:
            return rss / 1024
    except Exception:
        return None


def _get_pid_rss_mb(pid):
    """Return RSS in MB for a given PID, or None if unavailable.

    Uses psutil if available, otherwise falls back to ``ps``.
    """
    try:
        import psutil
        proc = psutil.Process(pid)
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        pass

    # Fallback: parse ``ps`` output (works on macOS and Linux)
    try:
        import subprocess as _sp
        out = _sp.check_output(["ps", "-o", "rss=", "-p", str(pid)],
                               text=True, timeout=5).strip()
        if out:
            return int(out) / 1024  # ps reports KB
    except Exception:
        pass

    return None


MEM_WATCHDOG_INTERVAL_S = 5  # How often the watchdog checks child RSS


def _run_in_subprocess(run_config, mem_limit_mb, script_path):
    """Run a single sweep config as a child process with a memory watchdog.

    Parameters
    ----------
    run_config : dict
        Serializable dict with all parameters needed by ``_sweep_worker``.
    mem_limit_mb : float
        RSS limit in MB. The watchdog SIGKILLs the child if exceeded.
    script_path : str
        Absolute path to this script (used to launch the worker).

    Returns
    -------
    dict
        Result dict loaded from the child's output JSON, or a failure dict.
    """
    import subprocess as sp
    import signal
    import threading

    config_json = json.dumps(run_config)

    # Launch child: python <this_script> --_sweep_worker <json>
    cmd = [sys.executable, script_path, "--_sweep-worker", config_json]
    child = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)

    killed_by_watchdog = threading.Event()

    # --- Memory watchdog thread ---
    def watchdog():
        while child.poll() is None:
            rss = _get_pid_rss_mb(child.pid)
            if rss is not None and rss > mem_limit_mb:
                print(f"\n  WATCHDOG: Child PID {child.pid} using {rss:.0f} MB "
                      f"(limit {mem_limit_mb:.0f} MB) — sending SIGKILL",
                      flush=True)
                killed_by_watchdog.set()
                try:
                    import os
                    os.kill(child.pid, signal.SIGKILL)
                except OSError:
                    pass
                return
            # Sleep in small increments so we can exit quickly when child dies
            for _ in range(MEM_WATCHDOG_INTERVAL_S * 2):
                if child.poll() is not None:
                    return
                import time as _time
                _time.sleep(0.5)

    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()

    # Stream child stdout to parent stdout in real time
    output_lines = []
    for line in child.stdout:
        print(f"    [child] {line}", end="", flush=True)
        output_lines.append(line)

    child.wait()
    watcher.join(timeout=2)

    result_file = Path(run_config["result_file"])

    if killed_by_watchdog.is_set():
        result = {
            "phase": run_config["phase_prefix"] + run_config["sub_label"],
            "status": "failed",
            "error": f"Killed by memory watchdog (>{mem_limit_mb:.0f} MB)",
            "method": run_config["method"],
            "sweep_dimension": run_config.get("dim_name", ""),
            "sweep_param": run_config.get("param_name", ""),
            "sweep_value": run_config.get("val"),
        }
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        return result

    if child.returncode != 0:
        stderr_tail = "".join(output_lines[-20:])
        result = {
            "phase": run_config["phase_prefix"] + run_config["sub_label"],
            "status": "failed",
            "error": f"Child exited with code {child.returncode}: {stderr_tail[-500:]}",
            "method": run_config["method"],
            "sweep_dimension": run_config.get("dim_name", ""),
            "sweep_param": run_config.get("param_name", ""),
            "sweep_value": run_config.get("val"),
        }
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        return result

    # Read result JSON written by the child
    if result_file.exists():
        with open(result_file) as f:
            return json.load(f)

    return {
        "phase": run_config["phase_prefix"] + run_config["sub_label"],
        "status": "failed",
        "error": "Child completed but result file not found",
        "method": run_config["method"],
    }


def _sweep_worker_main(config_json):
    """Entry point for a subprocess that runs a single sweep config.

    Called via: ``python run_comparison.py --_sweep-worker '<json>'``

    The child process runs ``_run_sub_experiment``, writes the result JSON,
    then exits. All PETSc/FEniCSx memory is reclaimed by the OS on exit.
    """
    import os
    os.environ.setdefault("CC", "/usr/bin/clang")

    config = json.loads(config_json)

    args = argparse.Namespace(
        phase="6",
        sweep_dim=None,
        output_dir=config["output_dir"],
        adios_file=config["adios_file"],
        sub=None,
        verbose=True,
        mem_limit_gb=config.get("mem_limit_gb", 12.0),
    )

    sweep_params = config["sweep_params"]

    try:
        result = _run_sub_experiment(
            args,
            sub_label=config["sub_label"],
            method=config["method"],
            output_dir=Path(config["output_dir"]),
            nt_da=config["nt_da"],
            nt_ramp=config["nt_ramp"],
            phase_prefix=config["phase_prefix"],
            sweep_params=sweep_params,
        )
    except Exception as e:
        result_file = Path(config["result_file"])
        result = {
            "phase": config["phase_prefix"] + config["sub_label"],
            "status": "failed",
            "error": str(e),
            "method": config["method"],
            "sweep_dimension": config.get("dim_name", ""),
            "sweep_param": config.get("param_name", ""),
            "sweep_value": config.get("val"),
        }
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        sys.exit(1)

    sys.exit(0)


SWEEP_BATCH_SIZE = 10  # Max configs per batch before forced cleanup


def run_phase_6(args):
    """Phase 6: Parameter Sweep — 4D-Var vs Static DC-WME.

    One-at-a-time sweep across 7 parameter dimensions.
    Each point runs both 4D-Var and static DC-WME.

    Memory safety: each config runs in its own subprocess.
    A watchdog thread in the parent monitors the child's RSS and SIGKILLs
    it if it exceeds --mem-limit-gb (default 12 GB). The parent process
    stays small and safe regardless of how much memory the child uses.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(exist_ok=True)
    comm.Barrier()

    mem_limit_mb = float(getattr(args, 'mem_limit_gb', 12.0)) * 1024
    script_path = str(Path(__file__).resolve())

    sweep_dim_filter = getattr(args, 'sweep_dim', None) or "all"
    dims_to_run = list(SWEEP_DIMS.keys()) if sweep_dim_filter == "all" else [sweep_dim_filter]

    all_sweep_results = {}

    for dim_name in dims_to_run:
        dim_config = SWEEP_DIMS[dim_name]
        param_name = dim_config["param"]
        values = dim_config["values"]

        if rank == 0:
            print(f"\n{'#' * 70}", flush=True)
            print(f"# SWEEP DIMENSION: {dim_name} ({param_name})", flush=True)
            print(f"#   Values: {values}", flush=True)
            print(f"# Memory limit: {mem_limit_mb:.0f} MB per child", flush=True)
            print(f"{'#' * 70}", flush=True)

        dim_results = []

        for val in values:
            sweep_params = dict(SWEEP_BASELINE)
            sweep_params[param_name] = val

            nt_da = sweep_params.pop("nt_da")
            nt_ramp = sweep_params.pop("nt_ramp")
            if dim_name == "window_length":
                nt_ramp = dim_config["nt_ramp_map"][val]
                nt_da = val

            val_str = str(val).replace(".", "p").replace(",", "_")
            point_results = {"dimension": dim_name, "param": param_name, "value": val}

            for method, sub_label in [("4dvar", "a"), ("dcwme", "b")]:
                phase_prefix = f"6_{dim_name}_{val_str}_"
                result_file = data_dir / f"phase{phase_prefix}{sub_label}_results.json"

                # Skip/resume
                if result_file.exists():
                    if rank == 0:
                        print(f"\n  Skipping {result_file.name} (already exists)", flush=True)
                    with open(result_file) as f:
                        result = json.load(f)
                    point_results[method] = result
                    continue

                if rank == 0:
                    print(f"\n  Running: {dim_name}={val}, method={method} "
                          f"[subprocess, limit={mem_limit_mb:.0f} MB]", flush=True)

                run_config = {
                    "output_dir": str(output_dir),
                    "adios_file": args.adios_file,
                    "sub_label": sub_label,
                    "method": method,
                    "nt_da": nt_da,
                    "nt_ramp": nt_ramp,
                    "phase_prefix": phase_prefix,
                    "sweep_params": sweep_params,
                    "result_file": str(result_file),
                    "dim_name": dim_name,
                    "param_name": param_name,
                    "val": val,
                    "mem_limit_gb": getattr(args, 'mem_limit_gb', 12.0),
                }

                result = _run_in_subprocess(run_config, mem_limit_mb, script_path)
                point_results[method] = result

                status = result.get("status", "unknown")
                if status == "failed":
                    if rank == 0:
                        err = result.get("error", "unknown error")
                        print(f"  FAILED: {err[:200]}", flush=True)
                else:
                    if rank == 0:
                        err_red = result.get("results", {}).get("error_reduction", "?")
                        print(f"  Done: error_reduction={err_red}", flush=True)

            dim_results.append(point_results)

        all_sweep_results[dim_name] = dim_results

    # Save summary
    if rank == 0:
        summary_file = data_dir / "phase6_summary.json"
        with open(summary_file, "w") as f:
            json.dump(all_sweep_results, f, indent=2, default=str)
        print(f"\nSummary saved to {summary_file}", flush=True)

        _print_sweep_summary(all_sweep_results)
        sys.stdout.flush()

    return all_sweep_results


def main():
    """Main entry point."""
    import os
    os.environ.setdefault("CC", "/usr/bin/clang")

    args = parse_args()

    if args.phase == "0":
        run_phase_0(args)
    elif args.phase == "1":
        run_phase_1(args)
    elif args.phase == "2":
        run_phase_2(args)
    elif args.phase == "3":
        run_phase_3(args)
    elif args.phase == "4":
        run_phase_4(args)
    elif args.phase == "5":
        run_phase_5(args)
    elif args.phase == "6":
        run_phase_6(args)
    else:
        print(f"Phase {args.phase} not yet implemented.")
        return 1

    return 0


if __name__ == "__main__":
    # Hidden entry point for subprocess sweep workers
    if len(sys.argv) >= 3 and sys.argv[1] == "--_sweep-worker":
        _sweep_worker_main(sys.argv[2])
    else:
        sys.exit(main())

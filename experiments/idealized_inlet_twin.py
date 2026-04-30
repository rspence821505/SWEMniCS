#!/usr/bin/env python3
"""
Idealized Inlet Wind-Driven Twin Experiment
============================================

Creates a storm-surge twin experiment on the idealized inlet domain:
  - Truth: wind vortex translating across the domain
  - Forecast: same vortex with a track shift (model error)
  - Observations: synthetic elevation from the truth trajectory
  - Diagnostics: state mismatch, forcing comparison, DA readiness

The wind forcing uses a Cartesian Rankine-like vortex (no lat/lon).
Model error comes from a perpendicular track shift, analogous to
the Holland hurricane track shift used in the Shinnecock experiments.

Usage:
  python experiments/idealized_inlet_twin.py                    # forward validation
  python experiments/idealized_inlet_twin.py --track-shift 10   # 10 km shift
  python experiments/idealized_inlet_twin.py --vmax 40          # stronger storm
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

os.environ.setdefault("CC", "/usr/bin/clang")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =========================================================================
# Memory safety
# =========================================================================

MEM_LIMIT_MB = 8 * 1024  # 8 GB default


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
                f"Process RSS {rss:.0f} MB exceeds limit {limit_mb:.0f} MB "
                f"at '{context}'. Aborting to prevent OOM."
            )


def _estimate_and_guard(description: str, bytes_needed: float, limit_mb: float = MEM_LIMIT_MB):
    """Estimate memory for an allocation and abort if it would exceed limits."""
    mb_needed = bytes_needed / 1e6
    current = _get_process_memory_mb() or 0
    projected = current + mb_needed
    print(f"  [mem] {description}: est. {mb_needed:.0f} MB "
          f"(current {current:.0f} MB, projected {projected:.0f} MB)")
    if projected > limit_mb:
        raise MemoryError(
            f"Allocation '{description}' would use {mb_needed:.0f} MB, "
            f"pushing RSS to ~{projected:.0f} MB (limit {limit_mb:.0f} MB). Aborting."
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


# =========================================================================
# Wind vortex generation (Cartesian coordinates)
# =========================================================================

@dataclass
class CartesianVortexConfig:
    """Translating wind vortex in Cartesian (x, y) coordinates.

    The vortex follows a linear track from start to end over the
    simulation time. Wind speed follows a modified Rankine profile:
      |W(r)| = Vmax * (r / Rmax)         for r < Rmax
      |W(r)| = Vmax * (Rmax / r)^decay   for r >= Rmax
    with cyclonic (counterclockwise) rotation + inflow angle.
    """
    # Track: linear from start to end
    track_start_x: float = 40000.0   # meters
    track_start_y: float = -20000.0  # meters (south of domain)
    track_end_x: float = 25000.0     # meters
    track_end_y: float = 50000.0     # meters (north of domain)

    # Storm parameters
    Vmax: float = 30.0               # max sustained wind (m/s)
    Rmax: float = 15000.0            # radius of max wind (meters)
    decay: float = 0.5               # decay exponent outside Rmax
    inflow_angle_deg: float = 20.0   # boundary layer inflow angle

    # Ambient pressure
    p_ambient_Pa: float = 101325.0   # 1013.25 mbar in Pa
    p_deficit_Pa: float = 3000.0     # central pressure deficit (~30 mbar)

    # Temporal ramp
    ramp_time_s: float = 7200.0      # 2-hour ramp to full strength

    # Track duration (decoupled from simulation length).
    # If None, falls back to legacy "track spans full simulation length"
    # behavior — kept for backward compatibility with experiments that
    # rely on the old wind. For new cycling runs and any case where the
    # SAME physical wind must reproduce across different ``n_windows``,
    # set this to a fixed absolute duration (in seconds). When ``t``
    # exceeds ``track_duration_s`` the storm is held at the end position.
    #
    # WHY: the legacy code linearly interpolated (start→end) across
    # ``times[-1] - times[0]`` so the storm crossed the domain faster
    # in shorter runs. Different ``n_windows`` therefore produced
    # genuinely different truth winds at the same absolute time t,
    # which destabilized the truth Newton at the first DA-window
    # timestep when n_windows was bumped (observed: r24n24 OK,
    # r24n36 truth Newton diverged).
    track_duration_s: Optional[float] = None


def generate_cartesian_vortex(
    config: CartesianVortexConfig,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate wind and pressure fields for a translating Cartesian vortex.

    Parameters
    ----------
    config : CartesianVortexConfig
    x_grid : 1D array of x coordinates (meters)
    y_grid : 1D array of y coordinates (meters)
    times : 1D array of times (seconds)

    Returns
    -------
    windx : (nt, ny, nx) eastward wind (m/s)
    windy : (nt, ny, nx) northward wind (m/s)
    pressure : (nt, ny, nx) atmospheric pressure (mbar, for HDF5 compat)
    """
    nt = len(times)
    ny = len(y_grid)
    nx = len(x_grid)

    # Memory guard: 3 arrays of (nt, ny, nx) float64 + workspace
    wind_bytes = 4 * nt * ny * nx * 8
    _estimate_and_guard(f"wind field arrays ({nt}x{ny}x{nx})", wind_bytes)

    windx = np.zeros((nt, ny, nx))
    windy = np.zeros((nt, ny, nx))
    pressure = np.full((nt, ny, nx), config.p_ambient_Pa / 100.0)  # mbar

    # Track interpolation. Prefer absolute track_duration_s (independent of
    # simulation length) so the same physical wind reproduces across runs
    # with different n_windows / nt_total. Falls back to legacy behavior
    # (track spans the times array) when track_duration_s is None.
    inflow_rad = np.deg2rad(config.inflow_angle_deg)
    if config.track_duration_s is not None:
        t_total = float(config.track_duration_s)
    else:
        t_total = times[-1] - times[0]

    xx, yy = np.meshgrid(x_grid, y_grid)  # (ny, nx)

    for k, t in enumerate(times):
        # Storm center position (linear interpolation along track)
        frac = (t - times[0]) / max(t_total, 1e-30)
        frac = np.clip(frac, 0, 1)
        cx = config.track_start_x + frac * (config.track_end_x - config.track_start_x)
        cy = config.track_start_y + frac * (config.track_end_y - config.track_start_y)

        # Distance from center
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx**2 + dy**2)
        r = np.maximum(r, 100.0)  # avoid division by zero

        # Rankine wind profile
        speed = np.where(
            r < config.Rmax,
            config.Vmax * (r / config.Rmax),
            config.Vmax * (config.Rmax / r) ** config.decay,
        )

        # Cyclonic rotation (counterclockwise) + inflow
        # Tangential unit vector (CCW): (-dy/r, dx/r)
        # Inflow unit vector (toward center): (-dx/r, -dy/r)
        # Combined: cos(inflow) * tangential + sin(inflow) * inflow
        tang_x = -dy / r
        tang_y = dx / r
        inflow_x = -dx / r
        inflow_y = -dy / r

        wx = speed * (np.cos(inflow_rad) * tang_x + np.sin(inflow_rad) * inflow_x)
        wy = speed * (np.cos(inflow_rad) * tang_y + np.sin(inflow_rad) * inflow_y)

        # Temporal ramp
        ramp = np.tanh(2.0 * t / max(config.ramp_time_s, 1e-30))

        windx[k] = ramp * wx
        windy[k] = ramp * wy

        # Pressure deficit (Gaussian profile)
        p_deficit = config.p_deficit_Pa * np.exp(-0.5 * (r / config.Rmax) ** 2)
        pressure[k] = (config.p_ambient_Pa - ramp * p_deficit) / 100.0  # Pa → mbar

    return windx, windy, pressure


def write_cartesian_wind_hdf5(
    filepath: str,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    times: np.ndarray,
    windx: np.ndarray,
    windy: np.ndarray,
    pressure: np.ndarray,
):
    """Write wind field to HDF5 in the same format as Shinnecock wind files.

    The 'longitude' and 'latitude' datasets store x and y coordinates
    in meters (not degrees). GriddedForcing with cartesian=True reads them
    correctly.
    """
    import h5py
    with h5py.File(filepath, "w") as f:
        f.create_dataset("longitude", data=x_grid)   # x-axis (meters)
        f.create_dataset("latitude", data=y_grid)     # y-axis (meters)
        f.create_dataset("time", data=times)
        f.create_dataset("windx", data=windx)
        f.create_dataset("windy", data=windy)
        f.create_dataset("pressure", data=pressure)

    nt, ny, nx = windx.shape
    print(f"  Wind HDF5: {filepath} ({nt} times, {ny}x{nx} grid)")


def create_perturbed_config(
    config: CartesianVortexConfig,
    track_shift_km: float = 10.0,
) -> CartesianVortexConfig:
    """Create a perturbed vortex config with a perpendicular track shift.

    The shift is perpendicular to the track direction, analogous to the
    Holland hurricane track shift used in Shinnecock experiments.
    """
    import copy
    perturbed = copy.deepcopy(config)

    # Track direction vector
    track_dx = config.track_end_x - config.track_start_x
    track_dy = config.track_end_y - config.track_start_y
    track_len = np.sqrt(track_dx**2 + track_dy**2)

    # Perpendicular direction (rotate 90 degrees CCW)
    perp_x = -track_dy / track_len
    perp_y = track_dx / track_len

    # Apply shift
    shift_m = track_shift_km * 1000.0
    perturbed.track_start_x += shift_m * perp_x
    perturbed.track_start_y += shift_m * perp_y
    perturbed.track_end_x += shift_m * perp_x
    perturbed.track_end_y += shift_m * perp_y

    return perturbed


# =========================================================================
# Experiment driver
# =========================================================================

def build_wind_files(cfg, output_dir, track_shift_km):
    """Generate truth and perturbed wind HDF5 files."""
    # Wind grid covering the domain with margin
    x_grid = np.linspace(-10000, 60000, 71)   # 1km spacing, extends beyond domain
    y_grid = np.linspace(-30000, 50000, 81)

    truth_file = output_dir / "wind_truth.h5"
    perturbed_file = output_dir / "wind_perturbed.h5"

    if not truth_file.exists():
        print("Generating truth wind...")
        wx, wy, p = generate_cartesian_vortex(cfg, x_grid, y_grid, cfg._times)
        write_cartesian_wind_hdf5(str(truth_file), x_grid, y_grid, cfg._times, wx, wy, p)

    pert_cfg = create_perturbed_config(cfg, track_shift_km)
    if not perturbed_file.exists():
        print(f"Generating perturbed wind (track shift {track_shift_km} km)...")
        wx, wy, p = generate_cartesian_vortex(pert_cfg, x_grid, y_grid, cfg._times)
        write_cartesian_wind_hdf5(str(perturbed_file), x_grid, y_grid, cfg._times, wx, wy, p)

    return truth_file, perturbed_file, pert_cfg


def run_forward(problem, solver, solver_params, nt, label=""):
    """Run forward model and return final state + trajectory."""
    print(f"  Running {label} ({nt} steps)...")
    t0 = time.time()
    solver.time_loop(
        solver_parameters=solver_params,
        stations=[],
        plot_every=9999,
        save_state=True,
        store_jacobians=False,
        enable_video=False,
        monitor_progress=False,
    )
    state_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs
    final_state = solver.u_n.x.array[:state_size].copy()
    trajectory = [s[:state_size].copy() for s in solver.storage.saved_states]
    print(f"  {label}: {time.time()-t0:.1f}s, {len(trajectory)} states saved")
    return final_state, trajectory


def compute_diagnostics(truth_traj, pert_traj, solver, obs_times, obs_fraction=0.1):
    """Compute state mismatch and observation-space diagnostics."""
    from dolfinx import la
    from swe4dvar.data_assimilation import PointObservationOperator
    from mpi4py import MPI

    state_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs
    comm = MPI.COMM_WORLD

    # State-space RMSE over time
    n_steps = min(len(truth_traj), len(pert_traj))
    rmse_time = []
    for k in range(n_steps):
        diff = truth_traj[k][:state_size] - pert_traj[k][:state_size]
        rmse_time.append(float(np.sqrt(np.mean(diff**2))))

    # Observation-space mismatch at obs times
    mesh = solver.V.mesh
    coords = mesh.geometry.x
    all_coords = comm.gather(coords, root=0)
    if comm.Get_rank() == 0:
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
    obs_operator = PointObservationOperator(solver.V, obs_points, comm=comm)

    obs_mismatch = []
    for t_idx in obs_times:
        if t_idx >= n_steps:
            continue
        vec_truth = la.create_petsc_vector(solver.V.dofmap.index_map, solver.V.dofmap.index_map_bs)
        vec_truth.setArray(truth_traj[t_idx][:state_size])
        vec_truth.assemble()
        vec_pert = la.create_petsc_vector(solver.V.dofmap.index_map, solver.V.dofmap.index_map_bs)
        vec_pert.setArray(pert_traj[t_idx][:state_size])
        vec_pert.assemble()

        obs_t = obs_operator.forward(vec_truth).getArray()
        obs_p = obs_operator.forward(vec_pert).getArray()
        obs_mismatch.append({
            "time_index": int(t_idx),
            "obs_rmse": float(np.sqrt(np.mean((obs_t - obs_p)**2))),
            "obs_max_diff": float(np.max(np.abs(obs_t - obs_p))),
            "obs_mean_truth": float(np.mean(obs_t)),
        })
        vec_truth.destroy()
        vec_pert.destroy()

    return {
        "state_rmse_time": rmse_time,
        "max_state_rmse": float(max(rmse_time)) if rmse_time else 0.0,
        "final_state_rmse": rmse_time[-1] if rmse_time else 0.0,
        "obs_mismatch": obs_mismatch,
        "n_obs_points": len(obs_points),
    }


def main():
    parser = argparse.ArgumentParser(description="Idealized Inlet Wind Twin Experiment")
    parser.add_argument("--vmax", type=float, default=30.0, help="Max wind speed (m/s)")
    parser.add_argument("--rmax-km", type=float, default=15.0, help="Radius of max wind (km)")
    parser.add_argument("--track-shift", type=float, default=10.0, help="Track shift (km)")
    parser.add_argument("--dt", type=float, default=600.0, help="Timestep (s)")
    parser.add_argument("--nt-ramp", type=int, default=24, help="Ramp timesteps (24 × 600s = 4h)")
    parser.add_argument("--nt-da", type=int, default=12, help="DA window timesteps")
    parser.add_argument("--obs-fraction", type=float, default=0.1, help="Observation fraction")
    parser.add_argument("--obs-frequency", type=int, default=4, help="Obs every N steps")
    parser.add_argument("--mem-limit-gb", type=float, default=8.0,
                        help="Memory limit in GB (default 8). Aborts if RSS exceeds this.")
    args = parser.parse_args()

    global MEM_LIMIT_MB
    MEM_LIMIT_MB = args.mem_limit_gb * 1024

    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()

    output_dir = PROJECT_ROOT / "results" / "idealized_inlet_twin"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    MPI.COMM_WORLD.Barrier()

    nt_total = args.nt_ramp + args.nt_da
    times = np.arange(0, (nt_total + 1) * args.dt, args.dt)

    # Vortex config
    vortex_cfg = CartesianVortexConfig(
        Vmax=args.vmax,
        Rmax=args.rmax_km * 1000,
        ramp_time_s=args.nt_ramp * args.dt,
    )
    vortex_cfg._times = times  # attach for wind generation

    print("="*60)
    print("IDEALIZED INLET WIND-DRIVEN TWIN EXPERIMENT")
    print("="*60)
    print(f"  Vmax={args.vmax} m/s, Rmax={args.rmax_km} km")
    print(f"  Track shift: {args.track_shift} km")
    print(f"  Ramp: {args.nt_ramp} steps ({args.nt_ramp * args.dt / 3600:.1f}h)")
    print(f"  DA window: {args.nt_da} steps ({args.nt_da * args.dt / 3600:.1f}h)")
    print(f"  Total: {nt_total} steps ({nt_total * args.dt / 3600:.1f}h)")
    print(f"  Memory limit: {args.mem_limit_gb:.1f} GB")

    # --- Step 1: Generate wind files ---
    if rank == 0:
        truth_file, pert_file, pert_cfg = build_wind_files(
            vortex_cfg, output_dir, args.track_shift)
    else:
        truth_file = pert_file = None
    truth_file = MPI.COMM_WORLD.bcast(truth_file, root=0)
    pert_file = MPI.COMM_WORLD.bcast(pert_file, root=0)

    # --- Step 2: Build problems + solvers ---
    from swe4dvar.forward.problems import IdealizedInlet
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params

    truth_forcing = GriddedForcing(str(truth_file), cartesian=True)
    pert_forcing = GriddedForcing(str(pert_file), cartesian=True)

    prob_truth = IdealizedInlet(
        dt=args.dt, nt=nt_total,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings",
        solution_var="h",
        dramp=args.nt_ramp * args.dt / 86400.0,  # days
        forcing=truth_forcing,
    )
    solver_truth = get_solver("DG")(prob_truth, theta=1.0, p_degree=[1, 1])

    prob_pert = IdealizedInlet(
        dt=args.dt, nt=nt_total,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings",
        solution_var="h",
        dramp=args.nt_ramp * args.dt / 86400.0,
        forcing=pert_forcing,
    )
    solver_pert = get_solver("DG")(prob_pert, theta=1.0, p_degree=[1, 1])

    solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=20, relaxation_parameter=1.0,
        comm=MPI.COMM_WORLD, error_if_not_converged=False,
    )

    state_size = solver_truth.V.dofmap.index_map.size_local * solver_truth.V.dofmap.index_map_bs
    print(f"  State size: {state_size} DOFs")
    _check_memory("after problem+solver construction")

    # Memory estimate for trajectory storage
    traj_bytes = (nt_total + 1) * state_size * 8
    _estimate_and_guard(
        f"truth trajectory ({nt_total+1} x {state_size} DOFs)", traj_bytes)

    # --- Step 3: Run truth and perturbed forward models ---
    t0 = time.time()
    truth_state, truth_traj = run_forward(
        prob_truth, solver_truth, solver_params, nt_total, "Truth")
    _check_memory("after truth trajectory")

    _cleanup()

    _estimate_and_guard(
        f"perturbed trajectory ({nt_total+1} x {state_size} DOFs)", traj_bytes)
    pert_state, pert_traj = run_forward(
        prob_pert, solver_pert, solver_params, nt_total, "Perturbed")
    _check_memory("after perturbed trajectory")

    print(f"  Total forward time: {time.time()-t0:.1f}s")

    # --- Step 4: Diagnostics ---
    _check_memory("before diagnostics")
    obs_times = list(range(args.nt_ramp, nt_total + 1, args.obs_frequency))
    print(f"\n  Obs times: {obs_times} ({len(obs_times)} snapshots)")

    diag = compute_diagnostics(
        truth_traj, pert_traj, solver_truth, obs_times, args.obs_fraction)
    _cleanup()

    print(f"\n{'='*60}")
    print("DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"  State RMSE (final): {diag['final_state_rmse']:.6e}")
    print(f"  State RMSE (max):   {diag['max_state_rmse']:.6e}")
    print(f"  Obs points:         {diag['n_obs_points']}")
    for om in diag["obs_mismatch"]:
        print(f"  Obs t={om['time_index']:3d}: RMSE={om['obs_rmse']:.6e}, "
              f"max_diff={om['obs_max_diff']:.6e}, mean_h={om['obs_mean_truth']:.4f}")

    # Assess DA readiness
    nontrivial = diag["max_state_rmse"] > 1e-4
    obs_nontrivial = any(om["obs_rmse"] > 1e-5 for om in diag["obs_mismatch"])

    print(f"\n  State mismatch nontrivial: {'YES' if nontrivial else 'NO'}")
    print(f"  Obs-space mismatch nontrivial: {'YES' if obs_nontrivial else 'NO'}")

    if nontrivial and obs_nontrivial:
        print(f"  DA READINESS: READY — meaningful model error for assimilation")
    else:
        print(f"  DA READINESS: NOT READY — perturbation too weak or forcing not entering model")

    # Save results
    results = {
        "config": {
            "vmax": args.vmax, "rmax_km": args.rmax_km,
            "track_shift_km": args.track_shift,
            "dt": args.dt, "nt_ramp": args.nt_ramp, "nt_da": args.nt_da,
            "obs_fraction": args.obs_fraction, "obs_frequency": args.obs_frequency,
        },
        "diagnostics": diag,
        "da_ready": nontrivial and obs_nontrivial,
    }
    results_file = output_dir / "experiment_results.json"
    if rank == 0:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
    print(f"\n  Results: {results_file}")
    print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    raise SystemExit(main())

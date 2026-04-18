#!/usr/bin/env python3
"""
Quick stability test for idealized inlet DA configurations.

Tests whether the perturbed-wind forward model remains viable.
Uses iterative solver for speed (we only need to know if it crashes).

Usage:
  python experiments/idealized_inlet_quick_stability.py
"""

import gc, os, sys, time, json
import numpy as np

os.environ.setdefault("CC", "/usr/bin/clang")
PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.idealized_inlet_twin import (
    CartesianVortexConfig, generate_cartesian_vortex,
    write_cartesian_wind_hdf5, create_perturbed_config,
)


def test_config(vmax, track_shift_km, nt_ramp=24, nt_da=12, dt=600.0):
    """Quick stability test. Returns min_h and state_rmse."""
    from mpi4py import MPI
    from swe4dvar.forward.problems import IdealizedInlet
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params

    comm = MPI.COMM_WORLD
    nt_total = nt_ramp + nt_da
    times = np.arange(0, (nt_total + 1) * dt, dt)

    wind_dir = PROJECT_ROOT / "results" / "stability_sweep" / "wind"
    wind_dir.mkdir(parents=True, exist_ok=True)

    x_grid = np.linspace(-10000, 60000, 71)
    y_grid = np.linspace(-30000, 50000, 81)

    cfg = CartesianVortexConfig(Vmax=vmax, Rmax=15000, ramp_time_s=nt_ramp * dt)

    truth_file = wind_dir / f"truth_v{vmax:.0f}.h5"
    if not truth_file.exists():
        wx, wy, p = generate_cartesian_vortex(cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(truth_file), x_grid, y_grid, times, wx, wy, p)

    pert_cfg = create_perturbed_config(cfg, track_shift_km)
    pert_file = wind_dir / f"pert_v{vmax:.0f}_s{track_shift_km:.0f}.h5"
    if not pert_file.exists():
        wx, wy, p = generate_cartesian_vortex(pert_cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(pert_file), x_grid, y_grid, times, wx, wy, p)

    # Use fast iterative solver for ramp (always stable)
    solver_params_fast = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=20, relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=False,
    )

    # Build + ramp
    forcing_truth = GriddedForcing(str(truth_file), cartesian=True)
    prob = IdealizedInlet(
        dt=dt, nt=nt_total, xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
        dramp=nt_ramp * dt / 86400.0, forcing=forcing_truth,
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])
    state_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs

    prob.nt = nt_ramp
    solver.time_loop(
        solver_parameters=solver_params_fast, stations=[], plot_every=9999,
        save_state=False, store_jacobians=False, enable_video=False,
    )
    ramp_end = solver.u_n.x.array[:state_size].copy()
    t_da = prob.t

    # DA window with PERTURBED wind
    forcing_pert = GriddedForcing(str(pert_file), cartesian=True)
    prob_p = IdealizedInlet(
        dt=dt, nt=nt_da, xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
        dramp=nt_ramp * dt / 86400.0, forcing=forcing_pert,
    )
    solver_p = get_solver("DG")(prob_p, theta=1.0, p_degree=[1, 1])
    solver_p.u_n.x.array[:state_size] = ramp_end
    solver_p.u_n_old.x.array[:state_size] = ramp_end
    solver_p.u.x.array[:state_size] = ramp_end
    solver_p.u_n.x.scatter_forward()
    solver_p.u_n_old.x.scatter_forward()
    solver_p.u.x.scatter_forward()
    prob_p.t = t_da

    solver_p.storage.clear()
    prob_p.nt = nt_da
    solver_p.time_loop(
        solver_parameters=solver_params_fast, stations=[], plot_every=9999,
        save_state=True, store_jacobians=False, enable_video=False,
    )

    # Also run truth DA window for RMSE comparison
    solver.storage.clear()
    solver.u_n.x.array[:state_size] = ramp_end
    solver.u_n_old.x.array[:state_size] = ramp_end
    solver.u.x.array[:state_size] = ramp_end
    solver.u_n.x.scatter_forward()
    solver.u_n_old.x.scatter_forward()
    solver.u.x.scatter_forward()
    prob.t = t_da
    prob.nt = nt_da
    solver.time_loop(
        solver_parameters=solver_params_fast, stations=[], plot_every=9999,
        save_state=True, store_jacobians=False, enable_video=False,
    )

    # Analyze
    n_h = state_size // 3  # approximate h DOF count
    min_h = float("inf")
    has_nan = False
    for s in solver_p.storage.saved_states:
        arr = s[:state_size]
        if np.any(np.isnan(arr)):
            has_nan = True
            break
        min_h = min(min_h, float(np.min(arr[:n_h])))

    if not has_nan and len(solver.storage.saved_states) > 0 and len(solver_p.storage.saved_states) > 0:
        truth_f = solver.storage.saved_states[-1][:state_size]
        pert_f = solver_p.storage.saved_states[-1][:state_size]
        rmse = float(np.sqrt(np.mean((truth_f - pert_f)**2)))
    else:
        rmse = float("nan")

    stable = min_h > -0.1 and not has_nan
    gc.collect()
    return stable, min_h, rmse, has_nan


def main():
    configs = [
        # (Vmax, track_shift_km)
        (5, 2), (5, 5), (5, 10),
        (10, 2), (10, 5), (10, 10),
        (15, 2), (15, 5), (15, 10),
        (20, 2), (20, 5), (20, 10),
        (25, 5), (30, 5),
    ]

    print("="*70)
    print("QUICK STABILITY SWEEP")
    print("="*70)

    results = []
    for vmax, shift in configs:
        t0 = time.time()
        stable, min_h, rmse, has_nan = test_config(vmax, shift)
        elapsed = time.time() - t0
        status = "OK" if stable else "FAIL"
        rmse_s = f"{rmse:.4e}" if not np.isnan(rmse) else "NaN"
        print(f"  Vmax={vmax:3d} shift={shift:2d}km: {status:4s}  min_h={min_h:+8.3f}  "
              f"rmse={rmse_s:>12s}  nan={has_nan}  {elapsed:.0f}s", flush=True)
        results.append({
            "vmax": vmax, "shift": shift, "stable": stable,
            "min_h": min_h, "rmse": rmse, "has_nan": has_nan,
        })

    print(f"\n{'='*70}")
    print("VIABLE FOR DA (stable + RMSE > 0.01):")
    for r in results:
        if r["stable"] and r["rmse"] > 0.01:
            print(f"  Vmax={r['vmax']}, shift={r['shift']}km: "
                  f"RMSE={r['rmse']:.4e}, min_h={r['min_h']:+.3f}")

    output_dir = PROJECT_ROOT / "results" / "stability_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "quick_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_dir / 'quick_sweep.json'}")


if __name__ == "__main__":
    raise SystemExit(main())

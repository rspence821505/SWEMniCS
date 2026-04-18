#!/usr/bin/env python3
"""
Idealized Inlet Stability Sweep
================================

Systematically tests whether the perturbed-wind forward model remains
numerically viable across (Vmax, track_shift) combinations.

For each configuration:
  1. Run ramp with truth wind
  2. Run DA window with PERTURBED wind from the ramp-end state
  3. Record: min(h), max(|u|), Newton failures, NaN occurrence

This identifies the stability boundary for DA-viable configurations.

Usage:
  python experiments/idealized_inlet_stability_sweep.py
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


def test_configuration(vmax, track_shift_km, nt_ramp=24, nt_da=12, dt=600.0):
    """Test a single (Vmax, track_shift) configuration.

    Returns dict with stability diagnostics.
    """
    from mpi4py import MPI
    from swe4dvar.forward.problems import IdealizedInlet
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params

    comm = MPI.COMM_WORLD
    nt_total = nt_ramp + nt_da
    times = np.arange(0, (nt_total + 1) * dt, dt)

    # Wind files
    wind_dir = PROJECT_ROOT / "results" / "stability_sweep" / "wind"
    wind_dir.mkdir(parents=True, exist_ok=True)

    truth_file = wind_dir / f"truth_v{vmax:.0f}.h5"
    pert_file = wind_dir / f"pert_v{vmax:.0f}_s{track_shift_km:.0f}.h5"

    x_grid = np.linspace(-10000, 60000, 71)
    y_grid = np.linspace(-30000, 50000, 81)

    cfg = CartesianVortexConfig(Vmax=vmax, Rmax=15000, ramp_time_s=nt_ramp * dt)
    if not truth_file.exists():
        wx, wy, p = generate_cartesian_vortex(cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(truth_file), x_grid, y_grid, times, wx, wy, p)

    pert_cfg = create_perturbed_config(cfg, track_shift_km)
    if not pert_file.exists():
        wx, wy, p = generate_cartesian_vortex(pert_cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(pert_file), x_grid, y_grid, times, wx, wy, p)

    # Build truth problem + run ramp
    forcing_truth = GriddedForcing(str(truth_file), cartesian=True)
    prob = IdealizedInlet(
        dt=dt, nt=nt_total,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
        dramp=nt_ramp * dt / 86400.0,
        forcing=forcing_truth,
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])
    # Use LU for robustness diagnostics
    solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=20, relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=False,
        ksp_type="preonly", pc_type="lu",
    )

    state_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs

    # Ramp
    prob.nt = nt_ramp
    solver.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=False, store_jacobians=False, enable_video=False,
    )
    ramp_end_state = solver.u_n.x.array[:state_size].copy()

    # Now run DA window with PERTURBED wind from ramp-end state
    forcing_pert = GriddedForcing(str(pert_file), cartesian=True)
    prob_pert = IdealizedInlet(
        dt=dt, nt=nt_da,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
        dramp=nt_ramp * dt / 86400.0,
        forcing=forcing_pert,
    )
    solver_pert = get_solver("DG")(prob_pert, theta=1.0, p_degree=[1, 1])

    solver_pert.u_n.x.array[:state_size] = ramp_end_state
    solver_pert.u_n_old.x.array[:state_size] = ramp_end_state
    solver_pert.u.x.array[:state_size] = ramp_end_state
    solver_pert.u_n.x.scatter_forward()
    solver_pert.u_n_old.x.scatter_forward()
    solver_pert.u.x.scatter_forward()
    prob_pert.t = prob.t

    # Run DA window and track stability
    solver_pert.storage.clear()
    prob_pert.nt = nt_da

    try:
        solver_pert.time_loop(
            solver_parameters=solver_params, stations=[], plot_every=9999,
            save_state=True, store_jacobians=False, enable_video=False,
        )
    except Exception as e:
        return {
            "vmax": vmax, "track_shift_km": track_shift_km,
            "stable": False, "error": str(e),
            "min_h": float("nan"), "max_vel": float("nan"),
        }

    # Analyze stability
    min_h = float("inf")
    max_vel = 0.0
    n_newton_failed = 0
    has_nan = False

    # Get component indices
    V = solver_pert.V
    n_sub = V.ufl_element().num_sub_elements
    if n_sub == 2:
        h_sub = V.sub(0).collapse()[0]
        n_h_dofs = h_sub.dofmap.index_map.size_local
        h_slice = slice(0, n_h_dofs)  # approximate
    else:
        n_h_dofs = state_size // 3
        h_slice = slice(0, n_h_dofs)

    for s in solver_pert.storage.saved_states:
        arr = s[:state_size]
        if np.any(np.isnan(arr)):
            has_nan = True
            break
        # h values (first component)
        h_vals = arr[:n_h_dofs]
        min_h = min(min_h, float(np.min(h_vals)))
        # velocity magnitude
        if n_sub == 2:
            # (h, (u,v)) mixed space — velocity is in second block
            uv_vals = arr[n_h_dofs:]
            vel_mag = np.sqrt(uv_vals[::2]**2 + uv_vals[1::2]**2)
        else:
            u_vals = arr[n_h_dofs:2*n_h_dofs]
            v_vals = arr[2*n_h_dofs:3*n_h_dofs]
            vel_mag = np.sqrt(u_vals**2 + v_vals**2)
        max_vel = max(max_vel, float(np.max(vel_mag)))

    stable = min_h > -0.1 and not has_nan

    # Also run truth DA window for comparison (state mismatch)
    solver.storage.clear()
    solver.u_n.x.array[:state_size] = ramp_end_state
    solver.u_n_old.x.array[:state_size] = ramp_end_state
    solver.u.x.array[:state_size] = ramp_end_state
    solver.u_n.x.scatter_forward()
    solver.u_n_old.x.scatter_forward()
    solver.u.x.scatter_forward()
    prob.t = prob_pert.t - nt_da * dt  # reset to DA start
    prob.nt = nt_da
    solver.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=True, store_jacobians=False, enable_video=False,
    )

    # State RMSE between truth and perturbed at final step
    if stable and not has_nan:
        truth_final = solver.storage.saved_states[-1][:state_size]
        pert_final = solver_pert.storage.saved_states[-1][:state_size]
        state_rmse = float(np.sqrt(np.mean((truth_final - pert_final)**2)))
    else:
        state_rmse = float("nan")

    gc.collect()

    return {
        "vmax": vmax,
        "track_shift_km": track_shift_km,
        "stable": stable,
        "min_h": float(min_h),
        "max_vel": float(max_vel),
        "has_nan": has_nan,
        "state_rmse": state_rmse,
    }


def main():
    output_dir = PROJECT_ROOT / "results" / "stability_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sweep grid
    vmax_values = [5, 10, 15, 20, 25, 30]
    shift_values = [2, 5, 10, 15]

    print("="*70)
    print("IDEALIZED INLET STABILITY SWEEP")
    print("="*70)
    print(f"  Vmax: {vmax_values}")
    print(f"  Track shifts: {shift_values} km")
    print(f"  Total configs: {len(vmax_values) * len(shift_values)}")

    results = []
    for vmax in vmax_values:
        for shift in shift_values:
            print(f"\n--- Vmax={vmax}, shift={shift}km ---")
            t0 = time.time()
            result = test_configuration(vmax, shift)
            result["time_s"] = time.time() - t0
            results.append(result)

            status = "STABLE" if result["stable"] else "UNSTABLE"
            print(f"  {status}: min_h={result['min_h']:.4f}, "
                  f"max_vel={result['max_vel']:.2f}, "
                  f"rmse={result['state_rmse']:.4e}, "
                  f"time={result['time_s']:.0f}s")

    # Summary table
    print(f"\n{'='*70}")
    print("STABILITY BOUNDARY")
    print(f"{'='*70}")
    print(f"{'Vmax':>6} {'Shift':>6} {'Status':>10} {'min_h':>10} {'max_vel':>10} {'RMSE':>12}")
    print("-" * 70)
    for r in results:
        status = "OK" if r["stable"] else "FAIL"
        rmse = f"{r['state_rmse']:.4e}" if not np.isnan(r['state_rmse']) else "NaN"
        print(f"{r['vmax']:6.0f} {r['track_shift_km']:6.0f} {status:>10} "
              f"{r['min_h']:10.4f} {r['max_vel']:10.2f} {rmse:>12}")

    # Identify viable configs (stable AND meaningful RMSE)
    viable = [r for r in results if r["stable"] and r["state_rmse"] > 0.01]
    print(f"\nViable for DA ({len(viable)} configs):")
    for r in viable:
        print(f"  Vmax={r['vmax']}, shift={r['track_shift_km']}km: "
              f"RMSE={r['state_rmse']:.4e}, min_h={r['min_h']:.4f}")

    # Save
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {output_dir / 'sweep_results.json'}")


if __name__ == "__main__":
    raise SystemExit(main())

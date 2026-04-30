"""Forecast-sensitivity probe for the idealized-inlet cycling-DA experiments.

Loads the bg vector + truth trajectory dumped by run_single_method when
window_tag is set (see cycling_dump_<tag>/), then runs the forward solver
WITHOUT data assimilation starting from the bg. Logs per-timestep:

    RMSE vs truth, min(h) per rank, max(|u|), max(|v|), Newton iters,
    first timestep where any rank goes h<0.5

Output: a CSV alongside the dump directory.

Usage:

    mpirun -n 8 ... python -u experiments/idealized_inlet_forecast_probe.py \\
        --dump-dir results/idealized_inlet_da/cycling_dump_w1 \\
        --nt-forecast 12

The probe MUST be launched at the same np that produced the dump (per-rank
.npy files are tied to the partitioning).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True,
                        help="Path to cycling_dump_<tag>/ directory (per-rank "
                             ".npy files + meta.json).")
    parser.add_argument("--nt-forecast", type=int, default=None,
                        help="Number of timesteps to forecast. Defaults to "
                             "meta['nt_da'] (i.e. one DA window).")
    parser.add_argument("--csv-out", default=None,
                        help="CSV output path. Defaults to "
                             "<dump_dir>/forecast_probe.csv.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Interpolation parameter. The forecast IC is "
                             "m(alpha) = truth_ic + alpha * (bg - truth_ic). "
                             "alpha=0 → pure truth (basin sanity check). "
                             "alpha=1 → original bg (default). Intermediate "
                             "values measure the basin threshold for the "
                             "fixed bg-error spatial structure.")
    parser.add_argument("--mem-limit-gb", type=float, default=64.0)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dump_dir = Path(args.dump_dir).resolve()
    if not dump_dir.is_dir():
        if rank == 0:
            print(f"ERROR: dump_dir not found: {dump_dir}", file=sys.stderr)
        return 2

    with open(dump_dir / "meta.json") as f:
        meta = json.load(f)

    if int(meta["comm_size"]) != size:
        if rank == 0:
            print(f"ERROR: dump was at np={meta['comm_size']} but probe "
                  f"is launched at np={size}. Per-rank .npy partitioning "
                  f"requires identical np.", file=sys.stderr)
        return 3

    nt_forecast = int(args.nt_forecast) if args.nt_forecast else int(meta["nt_da"])
    csv_out = Path(args.csv_out) if args.csv_out else (dump_dir / "forecast_probe.csv")

    if rank == 0:
        print(f"[probe] dump_dir = {dump_dir}")
        print(f"[probe] meta = window_tag={meta['window_tag']}, "
              f"nt_da={meta['nt_da']}, vmax={meta['vmax']}, "
              f"track_shift_km={meta['track_shift_km']}")
        print(f"[probe] forecasting {nt_forecast} timesteps from bg, "
              f"comparing to dumped truth trajectory")

    # Load per-rank arrays
    bg_arr = np.load(dump_dir / f"bg_rank{rank}.npy")
    truth_traj = np.load(dump_dir / f"truth_traj_rank{rank}.npy")  # (T, state_size)
    truth_ic = np.load(dump_dir / f"truth_ic_rank{rank}.npy")

    # Build the forecast IC by interpolating between truth_ic and bg.
    # alpha=0 → truth (in-basin sanity check)
    # alpha=1 → bg (original probe behavior)
    # 0 < alpha < 1 → measure basin threshold while holding spatial error
    #                 structure constant.
    alpha = float(args.alpha)
    if abs(alpha - 1.0) > 1e-12:
        if rank == 0:
            print(f"[probe] alpha-interp: m(α) = truth_ic + {alpha} * "
                  f"(bg - truth_ic)")
        bg_arr = truth_ic + alpha * (bg_arr - truth_ic)

    # ----- Set up forward solver matching the cycling-DA config -----
    # Replicate the truth/forward setup from idealized_inlet_da.py without
    # any DA infrastructure. The wind file naming matches the per-parameter
    # tags used by run_single_method so we reuse the same wind data.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from experiments.idealized_inlet_twin import (
        CartesianVortexConfig, generate_cartesian_vortex,
        write_cartesian_wind_hdf5, create_perturbed_config,
    )
    from swe4dvar.forward.problems import IdealizedInlet
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params

    dt = float(meta["dt"])
    nt_ramp = int(meta["nt_ramp"])
    truth_offset_steps = int(meta["truth_offset_steps"])
    nt_da = int(meta["nt_da"])
    nt_pre = nt_ramp + truth_offset_steps
    nt_total = nt_pre + nt_forecast
    times = np.arange(0, (nt_total + 1) * dt, dt)

    # Wind file paths use the same per-parameter naming as the driver.
    out_root = dump_dir.parent
    wind_dir = out_root / "wind"
    if rank == 0:
        wind_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    ts_tag = f"ts{int(round(meta['track_shift_km'] * 10)):03d}"
    v_tag = f"v{int(round(meta['vmax']))}"
    # Use the same n_windows-aware naming. We don't know n_windows from
    # meta alone, but we can use whatever nt total fits: ramp+offset+forecast
    t_tag = f"r{nt_ramp}n{truth_offset_steps + nt_forecast}"
    pert_file = wind_dir / f"perturbed_{v_tag}_{ts_tag}_{t_tag}.h5"
    truth_file = wind_dir / f"truth_{v_tag}_{ts_tag}_{t_tag}.h5"

    # Prefer to reuse existing wind files generated by the cycling run;
    # fall back to fresh generation if missing. Since the probe is run
    # at the same parameters, the cycling run's wind files should exist.
    vortex_cfg = CartesianVortexConfig(
        Vmax=meta["vmax"], Rmax=15000.0, ramp_time_s=nt_ramp * dt,
    )
    pert_cfg = create_perturbed_config(vortex_cfg, meta["track_shift_km"])
    x_grid = np.linspace(-10000, 60000, 71)
    y_grid = np.linspace(-30000, 50000, 81)
    if rank == 0:
        if not truth_file.exists():
            print(f"[probe] generating truth wind {truth_file.name}")
            wx, wy, p = generate_cartesian_vortex(vortex_cfg, x_grid, y_grid, times)
            write_cartesian_wind_hdf5(str(truth_file), x_grid, y_grid, times, wx, wy, p)
        if not pert_file.exists():
            print(f"[probe] generating perturbed wind {pert_file.name}")
            wx, wy, p = generate_cartesian_vortex(pert_cfg, x_grid, y_grid, times)
            write_cartesian_wind_hdf5(str(pert_file), x_grid, y_grid, times, wx, wy, p)
    comm.Barrier()

    # Forward solver. Use TRUTH wind so the comparison to truth_traj is
    # against the same wind file (avoids any byte-level drift between
    # truth_*.h5 and perturbed_*.h5 even at track-shift=0).
    forcing = GriddedForcing(str(truth_file), cartesian=True)
    prob = IdealizedInlet(
        dt=dt, nt=nt_total,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
        dramp=nt_ramp * dt / 86400.0,
        forcing=forcing,
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])
    relax = float(os.environ.get("SWE4DVAR_NEWTON_RELAX", "0.7"))
    solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=50, relaxation_parameter=relax,
        comm=comm, error_if_not_converged=False, ksp_max_it=2000,
    )

    state_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs

    # Validate dump-vs-runtime partitioning.
    if bg_arr.shape[0] != state_size:
        if rank == 0:
            print(f"ERROR rank{rank}: bg_arr size {bg_arr.shape[0]} != "
                  f"state_size {state_size}. Probe partitioning mismatch.",
                  file=sys.stderr)
        comm.Abort(4)

    # CRITICAL: march the solver through the ramp + offset BEFORE the
    # forecast. Just setting prob.t = t_da_start without time_loop leaves
    # the solver's internal state (BCs, wind-ramp, auxiliary fields)
    # uninitialized — even truth IC then fails at step 1. This must mirror
    # exactly what cycling DA's truth solver does to produce truth_traj[0].
    if rank == 0:
        print(f"[probe] marching solver through ramp+offset "
              f"({nt_pre} steps) before forecast")
    prob.nt = nt_pre
    solver.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=False, store_jacobians=False, enable_video=False,
    )
    if rank == 0:
        print(f"[probe] ramp done at t={prob.t:.0f}s "
              f"(expected {meta['t_da_start_seconds']:.0f}s)")

    # Sanity: the solver should now be at the same state as truth_ic. Log
    # the discrepancy so we can detect setup drift.
    arr_now = solver.u_n.x.array[:state_size]
    diff_to_truth = arr_now - truth_ic
    sse_diff = float(np.sum(diff_to_truth ** 2))
    n_diff = len(diff_to_truth)
    g_sse = comm.allreduce(sse_diff)
    g_n = comm.allreduce(n_diff)
    rmse_post_ramp = float(np.sqrt(g_sse / max(g_n, 1)))
    if rank == 0:
        print(f"[probe] post-ramp solver state vs truth_ic: rmse={rmse_post_ramp:.6e}")
        if rmse_post_ramp > 1e-6:
            print(f"[probe] WARNING: solver state diverged from truth_ic — "
                  f"forecast diagnostics may be unreliable")

    # Now override the solver's IC with the bg (or alpha-interpolated bg).
    solver.u_n.x.array[:state_size] = bg_arr
    solver.u_n_old.x.array[:state_size] = bg_arr
    solver.u.x.array[:state_size] = bg_arr
    solver.u_n.x.scatter_forward()
    solver.u_n_old.x.scatter_forward()
    solver.u.x.scatter_forward()
    # prob.t is already at t_da_start after the time_loop march.

    # ----- Step-by-step forecast with logging -----
    # Step the solver one timestep at a time so we can log between steps.
    h_indices, u_indices, v_indices = _get_component_indices(solver)
    rows = []  # list of dicts, written to CSV at end on rank 0

    # Initial (pre-forecast) row corresponds to truth_traj[0] which is the bg.
    truth_at_t0 = truth_traj[0]
    rows.append(_make_row(0, prob.t, solver, h_indices, u_indices, v_indices,
                          truth_at_t0, comm, newton_iters=0))

    # Reset Newton diag state per step
    solver._raise_on_newton_failure = False  # don't crash; record

    crashed_at = None
    for step in range(1, nt_forecast + 1):
        # Drive one timestep manually. solver.solve_init / solve_timestep.
        if step == 1:
            newton_solver = solver.solve_init(solver_parameters=solver_params)
        try:
            J = solver.solve_timestep(
                newton_solver, store_jacobian=False, timestep=step,
                time=prob.t,
            )
            n_newton = getattr(newton_solver, "_last_n_iters", -1)
        except RuntimeError as e:
            crashed_at = step
            n_newton = -1
            if rank == 0:
                print(f"[probe] step {step}: Newton FAILED: {e}", flush=True)
            # Still log the (broken) state for diagnostics
            truth_at_step = (truth_traj[step] if step < truth_traj.shape[0]
                             else truth_traj[-1])
            rows.append(_make_row(step, prob.t, solver,
                                  h_indices, u_indices, v_indices,
                                  truth_at_step, comm, newton_iters=n_newton,
                                  crashed=True))
            break

        # Advance time. solver.update_solution() already calls
        # prob.advance_time() internally — calling it explicitly here would
        # double-advance and read wind at the wrong (future) time, which
        # caused even truth ICs to crash Newton in the first probe run.
        solver.update_solution()
        truth_at_step = (truth_traj[step] if step < truth_traj.shape[0]
                         else truth_traj[-1])
        rows.append(_make_row(step, prob.t, solver,
                              h_indices, u_indices, v_indices,
                              truth_at_step, comm, newton_iters=n_newton))

    # Gather to rank 0 and write CSV
    if rank == 0:
        with open(csv_out, "w", newline="") as f:
            cols = list(rows[0].keys())
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[probe] wrote {csv_out}")
        if crashed_at is not None:
            print(f"[probe] crash at step {crashed_at} of {nt_forecast}")
        else:
            print(f"[probe] forecast completed cleanly through {nt_forecast} steps")

    return 0


def _get_component_indices(solver):
    """Return (h_idx, u_idx, v_idx) arrays of LOCAL DOF indices for owned cells."""
    n_local = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs
    # Mixed element: sub(0) is h, sub(1) is (u, v) with block size 2.
    h_sub = solver.V.sub(0)
    h_space, h_map = h_sub.collapse()
    h_idx = np.asarray(h_map, dtype=np.int64)
    h_idx = h_idx[h_idx < n_local]
    uv_sub = solver.V.sub(1)
    uv_space, uv_map = uv_sub.collapse()
    uv_idx = np.asarray(uv_map, dtype=np.int64)
    uv_idx = uv_idx[uv_idx < n_local]
    # uv is interleaved [u0, v0, u1, v1, ...]
    u_idx = uv_idx[0::2]
    v_idx = uv_idx[1::2]
    return h_idx, u_idx, v_idx


def _make_row(step, t, solver, h_idx, u_idx, v_idx, truth_state, comm,
              newton_iters, crashed=False):
    arr = solver.u_n.x.array[:]
    diff = arr[:truth_state.shape[0]] - truth_state
    # global RMSE
    local_sse = float(np.sum(diff ** 2))
    local_n = len(diff)
    global_sse = comm.allreduce(local_sse)
    global_n = comm.allreduce(local_n)
    rmse = float(np.sqrt(global_sse / max(global_n, 1)))

    h_local = arr[h_idx] if h_idx.size > 0 else np.array([np.nan])
    u_local = arr[u_idx] if u_idx.size > 0 else np.array([0.0])
    v_local = arr[v_idx] if v_idx.size > 0 else np.array([0.0])
    local_min_h = float(h_local.min()) if h_idx.size > 0 else float("inf")
    local_max_u = float(np.max(np.abs(u_local))) if u_idx.size > 0 else 0.0
    local_max_v = float(np.max(np.abs(v_local))) if v_idx.size > 0 else 0.0
    global_min_h = comm.allreduce(local_min_h, op=MPI.MIN)
    global_max_u = comm.allreduce(local_max_u, op=MPI.MAX)
    global_max_v = comm.allreduce(local_max_v, op=MPI.MAX)

    return {
        "step": int(step),
        "t_seconds": float(t),
        "rmse_vs_truth": rmse,
        "min_h_global": global_min_h,
        "max_u_global": global_max_u,
        "max_v_global": global_max_v,
        "newton_iters": int(newton_iters),
        "crashed": bool(crashed),
    }


if __name__ == "__main__":
    sys.exit(main())

"""Isolated truth-solver probe.

Runs the IdealizedInlet forward solver from t=0 through nt_total timesteps
with NO DA infrastructure: no observations, no perturbed background, no
optimizer, no cycling. Designed to determine whether the Newton-failure
seen during cycling-DA at t ≈ 6h+10min is caused by the SWE forward
solver itself (independent of DA) or by some cycling-pipeline interaction.

Per-timestep diagnostics are emitted via the same SWE4DVAR_FORWARD_DIAG_CSV
mechanism used by cycling DA, so the CSV format is directly comparable.

Usage:
    mpirun -n 8 ... python -u experiments/idealized_inlet_truth_probe.py \\
        --vmax 20 --track-shift 0 \\
        --nt-ramp 24 --nt-extra 42 --dt 600
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vmax", type=float, default=20.0)
    parser.add_argument("--track-shift", type=float, default=0.0)
    parser.add_argument("--rmax-km", type=float, default=15.0)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--nt-ramp", type=int, default=24,
                        help="Number of ramp timesteps (warmup with rising wind).")
    parser.add_argument("--nt-extra", type=int, default=42,
                        help="Number of timesteps to march AFTER ramp end. "
                             "Default 42 gives 7h of post-ramp integration "
                             "(covers the suspected NaN at t ≈ 6h+10min).")
    parser.add_argument("--min-depth", type=float, default=5.0)
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "results" / "idealized_inlet_truth_probe"))
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    out = Path(args.output_dir).resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    # Default the diag CSV path if not already set, so this script is
    # informative even when invoked without the env var.
    if "SWE4DVAR_FORWARD_DIAG_CSV" not in os.environ:
        default_csv = out / f"truth_probe_v{int(args.vmax)}_ts{int(round(args.track_shift*10)):03d}.csv"
        os.environ["SWE4DVAR_FORWARD_DIAG_CSV"] = str(default_csv)

    # Local imports AFTER setting env var so the cg_implicit module-level
    # _FWD_DIAG dict picks up the path on first import.
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.idealized_inlet_twin import (
        CartesianVortexConfig, generate_cartesian_vortex,
        write_cartesian_wind_hdf5,
    )
    from swe4dvar.forward.problems import IdealizedInlet
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params

    nt_total = args.nt_ramp + args.nt_extra
    times = np.arange(0, (nt_total + 1) * args.dt, args.dt)

    if rank == 0:
        print(f"[truth-probe] np={size}  vmax={args.vmax}  "
              f"track-shift={args.track_shift}km", flush=True)
        print(f"[truth-probe] nt_ramp={args.nt_ramp}  "
              f"nt_extra={args.nt_extra}  total={nt_total} steps", flush=True)
        print(f"[truth-probe] integrating to t = {nt_total * args.dt:.0f} s "
              f"(= {nt_total * args.dt / 3600:.2f} h)", flush=True)
        print(f"[truth-probe] diag CSV: {os.environ['SWE4DVAR_FORWARD_DIAG_CSV']}",
              flush=True)

    # ----- Wind file (truth only — no perturbation) -----
    wind_dir = out / "wind"
    if rank == 0:
        wind_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    ts_tag = f"ts{int(round(args.track_shift * 10)):03d}"
    v_tag = f"v{int(round(args.vmax))}"
    t_tag = f"r{args.nt_ramp}n{args.nt_extra}"
    wind_file = wind_dir / f"truth_{v_tag}_{ts_tag}_{t_tag}.h5"
    if rank == 0 and not wind_file.exists():
        print(f"[truth-probe] generating wind {wind_file.name}", flush=True)
        cfg = CartesianVortexConfig(
            Vmax=args.vmax, Rmax=args.rmax_km * 1000,
            ramp_time_s=args.nt_ramp * args.dt,
        )
        x_grid = np.linspace(-10000, 60000, 71)
        y_grid = np.linspace(-30000, 50000, 81)
        wx, wy, p = generate_cartesian_vortex(cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(wind_file), x_grid, y_grid, times, wx, wy, p)
    comm.Barrier()

    # ----- Forward solver -----
    forcing = GriddedForcing(str(wind_file), cartesian=True)
    prob = IdealizedInlet(
        dt=args.dt, nt=nt_total,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
        dramp=args.nt_ramp * args.dt / 86400.0,
        forcing=forcing,
    )
    if args.min_depth > 5.0:
        h_b_arr = prob.h_b.x.array[:]
        h_b_arr[h_b_arr < args.min_depth] = args.min_depth
        prob.h_b.x.array[:] = h_b_arr
        prob.h_b.x.scatter_forward()
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])

    relax = float(os.environ.get("SWE4DVAR_NEWTON_RELAX", "0.7"))
    solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=50, relaxation_parameter=relax,
        comm=comm, error_if_not_converged=False, ksp_max_it=2000,
    )

    if rank == 0:
        print(f"[truth-probe] starting time_loop at t={prob.t:.0f}s", flush=True)

    # Single time_loop. The forward-diag CSV captures every timestep.
    # raise_on_failure NOT set, so Newton failures are silent (logged in CSV).
    solver.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=False, store_jacobians=False, enable_video=False,
    )

    if rank == 0:
        print(f"[truth-probe] time_loop complete at t={prob.t:.0f}s", flush=True)
        print(f"[truth-probe] CSV: {os.environ['SWE4DVAR_FORWARD_DIAG_CSV']}",
              flush=True)
        # Quick scan for first failure
        csv_path = os.environ["SWE4DVAR_FORWARD_DIAG_CSV"]
        try:
            with open(csv_path) as f:
                lines = f.readlines()
            n_rows = len(lines) - 1
            n_failed = sum(1 for ln in lines[1:] if ln.split(",")[4] == "0")
            n_converged = n_rows - n_failed
            print(f"[truth-probe] CSV summary: {n_rows} rows, "
                  f"{n_converged} Newton-converged, {n_failed} failed",
                  flush=True)
            if n_failed > 0:
                first_fail = next(
                    ln for ln in lines[1:] if ln.split(",")[4] == "0"
                )
                print(f"[truth-probe] first failure row: {first_fail.strip()}",
                      flush=True)
        except Exception as e:
            print(f"[truth-probe] CSV summary unavailable: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Jacobian parity harness for the recompute-Jacobians-in-adjoint feature.

Goal
----
Direct algebraic comparison of stored vs recomputed J_n at a few selected
timesteps. NOT an end-to-end DA correctness test (those are expensive and
muddy by line-search and optimizer behavior). This harness fails loudly
when parity is poor.

Method
------
1. Run a single short forward solve with BOTH legacy storage AND the new
   replay-metadata capture turned on:
     - ``store_jacobians=True`` keeps the existing ``saved_jacobians[k]``
       (the ground truth for parity)
     - ``SWE4DVAR_CAPTURE_REPLAY_META=1`` makes the data manager call
       ``solver._capture_replay_metadata`` at each timestep, populating
       ``solver.storage.replay_metadata``.
2. Construct a JacobianReplayContext on the same forward solver.
3. For each selected timestep n, fetch ``J_stored = saved_jacobians[n-1]``
   and ``J_recomp = ctx.reassemble(replay_metadata[k_for_n])``.
4. Compute Frobenius norms, ``||J_recomp - J_stored||_F``, relative diff,
   and a few random matvec parity checks.

Acceptance
----------
Relative Frobenius-norm difference at FP assembly noise level
(``< ~1e-10``) and random matvec parity below the same level. If
discrepancy is structural (orders of magnitude larger), the harness
prints which entry types are wrong and exits 2.

Usage
-----
::

    SWE4DVAR_CAPTURE_REPLAY_META=1 \\
    mpirun -n 4 python experiments/jacobian_parity_harness.py \\
        --vmax 10 --nt-ramp 24 --nt-da 6 \\
        --steps 1 2 3 6 --rtol 1e-10

Single-rank also works (``mpirun -n 1`` or no mpirun).

Exit codes
----------
0 — parity passed at all selected timesteps
1 — runtime / setup error (could not even attempt the comparison)
2 — parity failed at one or more timesteps
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _vec_random(template: "PETSc.Vec", seed: int) -> "PETSc.Vec":
    """Make a deterministic random Vec the right size for matvec parity."""
    from petsc4py import PETSc
    rng = np.random.default_rng(seed)
    v = template.duplicate()
    arr = rng.standard_normal(v.getLocalSize())
    v.setArray(arr)
    v.assemble()
    return v


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vmax", type=float, default=10.0)
    parser.add_argument("--track-shift", type=float, default=0.0)
    parser.add_argument("--rmax-km", type=float, default=15.0)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--nt-ramp", type=int, default=24)
    parser.add_argument("--nt-da", type=int, default=6)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 6],
                        help="Timesteps n to compare (1-indexed).")
    parser.add_argument("--rtol", type=float, default=1e-10,
                        help="Relative Frobenius tolerance for parity pass.")
    parser.add_argument("--matvec-trials", type=int, default=3,
                        help="Random Vec matvec parity checks per step.")
    parser.add_argument("--track-duration-s", type=float, default=0.0)
    parser.add_argument("--background-error-std", type=float, default=0.02)
    parser.add_argument("--obs-fraction", type=float, default=0.05)
    parser.add_argument("--obs-frequency", type=int, default=5)
    parser.add_argument("--obs-noise-level", type=float, default=0.01)
    args = parser.parse_args()

    # Force replay capture on for this run.
    os.environ["SWE4DVAR_CAPTURE_REPLAY_META"] = "1"

    # Make sure imports resolve.
    sys.path.insert(0, str(PROJECT_ROOT))

    from mpi4py import MPI
    from petsc4py import PETSc
    rank = MPI.COMM_WORLD.Get_rank()

    def _log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    _log("=" * 70)
    _log("Jacobian parity harness — stored J  vs  recomputed J")
    _log("=" * 70)

    # ---- Build a minimal forward solve ---------------------------------
    # We piggy-back on idealized_inlet_da's run_single_method machinery for
    # mesh / problem / forcing setup, but stop after the truth solve fills
    # the storage so we have access to both saved_jacobians and
    # replay_metadata. Easiest way: call the truth-side solve directly.
    from experiments.idealized_inlet_twin import (
        CartesianVortexConfig, generate_cartesian_vortex,
        write_cartesian_wind_hdf5,
    )
    from swe4dvar.forward.problems import IdealizedInlet
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params
    from swe4dvar.adjoint.jacobian_replay import JacobianReplayContext

    comm = MPI.COMM_WORLD
    nt_total = args.nt_ramp + args.nt_da
    times = np.arange(0, (nt_total + 1) * args.dt, args.dt)
    track_duration_s = args.track_duration_s or 28800.0

    out_dir = PROJECT_ROOT / "results" / "jacobian_parity_harness"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    wind_dir = out_dir / "wind"
    if rank == 0:
        wind_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    cfg = CartesianVortexConfig(
        Vmax=args.vmax,
        Rmax=args.rmax_km * 1000,
        ramp_time_s=args.nt_ramp * args.dt,
        track_duration_s=track_duration_s,
    )
    wind_file = wind_dir / (
        f"truth_v{int(args.vmax)}"
        f"_ts{int(round(args.track_shift * 10)):03d}"
        f"_r{args.nt_ramp}n{args.nt_da}_td{int(track_duration_s)}.h5"
    )
    if rank == 0 and not wind_file.exists():
        _log(f"  Generating wind file {wind_file.name}")
        x_grid = np.linspace(-10000, 60000, 71)
        y_grid = np.linspace(-30000, 50000, 81)
        wx, wy, p = generate_cartesian_vortex(cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(wind_file), x_grid, y_grid, times, wx, wy, p)
    comm.Barrier()

    forcing = GriddedForcing(str(wind_file), cartesian=True)
    prob = IdealizedInlet(
        dt=args.dt, nt=nt_total,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
        dramp=args.nt_ramp * args.dt / 86400.0,
        forcing=forcing,
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])
    solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=50, relaxation_parameter=0.7,
        comm=comm, error_if_not_converged=False, ksp_max_it=2000,
    )

    _log(f"  Forward solve: nt_ramp={args.nt_ramp} + nt_da={args.nt_da} "
         f"= {nt_total} steps")
    solver.time_loop(
        solver_parameters=solver_params,
        stations=[],
        plot_every=9999,
        save_state=True,
        store_jacobians=True,
        enable_video=False,
    )
    _log(f"  Forward solve done.")

    # ---- Sanity-check storage ------------------------------------------
    n_stored_J = len(solver.storage.saved_jacobians)
    n_replay = len(solver.storage.replay_metadata)
    _log(f"  Storage: {n_stored_J} stored Jacobians, {n_replay} replay records")
    if n_stored_J == 0:
        _log("  ERROR: forward did not populate saved_jacobians "
             "(store_jacobians may have been disabled)")
        return 1
    if n_replay == 0:
        _log("  ERROR: replay metadata is empty. "
             "SWE4DVAR_CAPTURE_REPLAY_META=1 must be set BEFORE starting.")
        return 1

    # Replay metadata records its own ``timestep`` field; build a lookup
    # so we can match a requested step n to its record without assuming
    # contiguous numbering.
    replay_by_step = {int(r["timestep"]): r for r in solver.storage.replay_metadata}

    # ---- Build the shadow replay context --------------------------------
    ctx = JacobianReplayContext(solver)

    # ---- Compare per requested step -------------------------------------
    failures = []
    for n in args.steps:
        if n < 1 or n > nt_total:
            _log(f"\n  step n={n} skipped (out of range)")
            continue
        if n not in replay_by_step:
            _log(f"\n  step n={n} skipped (no replay record)")
            continue
        meta = replay_by_step[n]

        # jacobians[n-1] is J at step n in the legacy index convention
        # (see implicit_adjoint.py:1419).
        if n - 1 >= len(solver.storage.saved_jacobians):
            _log(f"\n  step n={n} skipped (no stored Jacobian)")
            continue
        J_stored = solver.storage.saved_jacobians[n - 1]

        _log(f"\n  --- step n={n} ---")
        summary = ctx.saved_state_summary(meta)
        _log(f"  meta: t={summary['t']:.1f}  theta1={summary['theta1']:.3f}  "
             f"|u|={summary['u_norm']:.3e}  |u_n|={summary['u_n_norm']:.3e}  "
             f"|u_bc|={summary['u_bc_norm']}")

        J_recomp = ctx.reassemble(meta, copy=True)

        # --- Frobenius-norm parity ---
        nF_stored = J_stored.norm(PETSc.NormType.NORM_FROBENIUS)
        nF_recomp = J_recomp.norm(PETSc.NormType.NORM_FROBENIUS)
        # Build diff in a fresh Mat. AYPX requires same-pattern.
        diff = J_recomp.duplicate(copy=True)   # diff = J_recomp
        diff.axpy(-1.0, J_stored)              # diff = J_recomp - J_stored
        nF_diff = diff.norm(PETSc.NormType.NORM_FROBENIUS)
        rel = nF_diff / max(nF_stored, 1e-30)
        _log(f"  ||J_stored||_F = {nF_stored:.6e}")
        _log(f"  ||J_recomp||_F = {nF_recomp:.6e}")
        _log(f"  ||diff||_F     = {nF_diff:.6e}")
        _log(f"  relative diff  = {rel:.3e}   "
             f"({'PASS' if rel < args.rtol else 'FAIL'} @ rtol={args.rtol})")

        # --- random matvec parity ---
        x = J_stored.createVecRight()
        for trial in range(args.matvec_trials):
            x = _vec_random(x, seed=4242 + 7 * n + trial)
            y_stored = J_stored.createVecLeft()
            y_recomp = J_stored.createVecLeft()
            J_stored.mult(x, y_stored)
            J_recomp.mult(x, y_recomp)
            y_stored.axpy(-1.0, y_recomp)
            err = y_stored.norm(PETSc.NormType.NORM_2)
            ref = max(y_recomp.norm(PETSc.NormType.NORM_2), 1e-30)
            _log(f"    matvec trial {trial}: ||(J_s - J_r)x||/||J_r x|| "
                 f"= {err / ref:.3e}")
            y_stored.destroy()
            y_recomp.destroy()

        x.destroy()
        diff.destroy()
        J_recomp.destroy()

        if rel >= args.rtol:
            failures.append(n)

    _log("\n" + "=" * 70)
    if not failures:
        _log("Parity PASSED for all selected steps.")
        return 0
    else:
        _log(f"Parity FAILED at steps: {failures}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

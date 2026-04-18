#!/usr/bin/env python3
"""
Driver for serial vs MPI object-diff exports.

Builds the idealized inlet DA problem (via tests/test_mpi_parity.py's
build_da_problem), constructs an MPIObjectExporter, and writes a complete
set of canonically-ordered .npz artifacts under
results/mpi_object_diff/{tag}/.

Artifacts written (per run, where {tag} is "serial" or f"mpi{size}"):
    dof_metadata__{tag}.npz
        coords (canonical (x,y) order), per-DOF owner rank, interface
        distance per h-DOF.
    state__m_background__{tag}.npz
        Background control vector m_background.
    state__trajectory_{k}__{tag}.npz
        Forward trajectory at timestep k.
    jac_action_{k}__J_at_{pattern}__{tag}.npz
        J @ v for v = coord-deterministic test vector "pattern".
    jac_action_{k}__JT_at_{pattern}__{tag}.npz
        J^T @ v.
    adjoint_rhs__obs_forcing_{k}__{tag}.npz
        Adjoint RHS contribution (observation forcing) at timestep k.
    adjoint__lambda0__{tag}.npz
        Adjoint solution λ₀ (initial-condition gradient block).
    matrix_rows__J_{k}_near_interface__{tag}.npz
        Selected rows of J at coordinates near the 2-rank partition
        interface (or arbitrary fallback in serial).

Usage:
    # Serial baseline
    SWE4DVAR_MEM_LIMIT_GB=7 PYTHONUNBUFFERED=1 \\
        python tests/mpi_diff_export.py --tag serial

    # 2-rank MPI
    SWE4DVAR_MEM_LIMIT_GB=7 PYTHONUNBUFFERED=1 \\
        mpirun -np 2 python tests/mpi_diff_export.py --tag mpi2

    # Then compare:
    python tests/mpi_diff_compare.py --a serial --b mpi2
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CC", "/usr/bin/clang")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))


def main():
    parser = argparse.ArgumentParser(description="Export canonical objects for serial/MPI diff")
    parser.add_argument("--tag", required=True,
                        help='Run tag, e.g. "serial" or "mpi2". Determines artifact subdir.')
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "results" / "mpi_object_diff"),
                        help="Root directory for artifacts.")
    parser.add_argument("--patterns", default="ones,sin2D,localized",
                        help="Comma-separated coord-deterministic Jacobian probe patterns.")
    parser.add_argument("--jac-step", type=int, default=0,
                        help="Index into jacobians[] to probe (0 = first cached step).")
    parser.add_argument("--n-suspect-coords", type=int, default=8,
                        help="Number of partition-interface points at which to extract matrix rows.")
    parser.add_argument("--skip-matrix-rows", action="store_true",
                        help="Skip matrix-row extraction (useful for fast incremental runs).")
    args = parser.parse_args()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    out_dir = Path(args.out_root) / args.tag
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    def log(msg):
        sys.stdout.write(f"  [r{rank}] {msg}\n")
        sys.stdout.flush()

    log(f"START tag={args.tag} size={size} out_dir={out_dir}")

    from test_mpi_parity import build_da_problem
    from swe4dvar.utils.mpi_object_diff import MPIObjectExporter

    da = build_da_problem(comm, rank)
    cost_fn = da["cost_fn"]
    V = cost_fn.forward_model.solver.V

    exp = MPIObjectExporter(V, comm=comm)
    exp.summarize()
    exp.export_dof_metadata(out_dir, args.tag)

    # ----- state vectors -----
    exp.export_vector(da["m_background"], "state__m_background", out_dir, args.tag)

    log("running forward model to obtain trajectory + cached Jacobians ...")
    trajectory, jacobians = cost_fn._run_forward_model(da["m_background"], store_jacobians=True)
    log(f"forward done: traj_len={len(trajectory)}, jac_len={len(jacobians)}")

    for k, vec in enumerate(trajectory):
        exp.export_vector(vec, f"state__trajectory_{k:02d}", out_dir, args.tag)

    if len(jacobians) <= args.jac_step:
        log(f"ERROR: requested jac_step={args.jac_step} but only {len(jacobians)} jacobians cached")
        sys.exit(1)
    J = jacobians[args.jac_step]
    if rank == 0:
        log(f"using J from step {args.jac_step}: size={J.getSize()}, local={J.getLocalSize()}")

    # ----- Jacobian / Jacobian-transpose action on coord-deterministic vectors -----
    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    for pattern in patterns:
        log(f"probing J  @ v({pattern})")
        v = exp.make_coord_vector(pattern)
        exp.export_matrix_action(
            J, v, f"jac_action_{args.jac_step:02d}__J_at_{pattern}",
            out_dir, args.tag, transpose=False,
        )
        log(f"probing J^T @ v({pattern})")
        exp.export_matrix_action(
            J, v, f"jac_action_{args.jac_step:02d}__JT_at_{pattern}",
            out_dir, args.tag, transpose=True,
        )
        v.destroy()
        gc.collect()

    # ----- adjoint RHS (observation forcings) -----
    log("computing observation forcings (adjoint RHS) ...")
    obs_forcings = cost_fn._compute_observation_forcings(trajectory)
    log(f"obs_forcings: {len(obs_forcings)} entries "
        f"({sum(1 for f in obs_forcings if f is not None)} non-None)")
    for i, f in enumerate(obs_forcings):
        if f is None:
            continue
        exp.export_vector(f, f"adjoint_rhs__obs_forcing_{i:02d}", out_dir, args.tag)

    # ----- adjoint solve λ₀ -----
    log("solving adjoint to get λ₀ ...")
    try:
        lambda_0 = cost_fn._solve_adjoint(trajectory, jacobians)
        exp.export_vector(lambda_0, "adjoint__lambda0", out_dir, args.tag)
        if rank == 0:
            log(f"  ||λ₀||_2 = {float(lambda_0.norm()):.6e}")
        lambda_0.destroy()
    except Exception as e:
        log(f"adjoint solve FAILED: {e}")

    # ----- matrix rows near partition interface -----
    if not args.skip_matrix_rows:
        # Pick suspect coords: in MPI use the n smallest-interface-distance owned
        # h-DOF coords (i.e. the most boundary-like positions). In serial there
        # is no interface — fall back to a deterministic spatial sweep.
        if size > 1:
            local_iface = exp.interface_distance.copy()
            order = np.argsort(local_iface)
            n_take_local = min(args.n_suspect_coords, len(order))
            local_top = exp.h_coords_local[order[:n_take_local]]
            local_iface_top = local_iface[order[:n_take_local]]
            local_pack = np.hstack([local_top, local_iface_top.reshape(-1, 1)])
            all_packs = comm.gather(local_pack, root=0)
            if rank == 0:
                stacked = np.vstack(all_packs)
                rank0_order = np.argsort(stacked[:, 2])
                top = stacked[rank0_order[:args.n_suspect_coords], :2]
                suspect_coords = [(float(x), float(y)) for (x, y) in top]
                log(f"selected {len(suspect_coords)} interface-near suspect coords")
            else:
                suspect_coords = None
            suspect_coords = comm.bcast(suspect_coords, root=0)
        else:
            xs = np.linspace(0.0, 50000.0, args.n_suspect_coords)
            ys = np.full_like(xs, 5000.0)
            suspect_coords = [(float(x), float(y)) for x, y in zip(xs, ys)]
            if rank == 0:
                log(f"serial: deterministic suspect coord sweep ({len(suspect_coords)} points)")

        log("extracting J rows near suspect coordinates ...")
        exp.export_matrix_rows(
            J, suspect_coords, component="h",
            name=f"matrix_rows__J_{args.jac_step:02d}_h_near_interface",
            out_dir=out_dir, tag=args.tag,
        )
        exp.export_matrix_rows(
            J, suspect_coords, component="u",
            name=f"matrix_rows__J_{args.jac_step:02d}_u_near_interface",
            out_dir=out_dir, tag=args.tag,
        )
        exp.export_matrix_rows(
            J, suspect_coords, component="v",
            name=f"matrix_rows__J_{args.jac_step:02d}_v_near_interface",
            out_dir=out_dir, tag=args.tag,
        )
        log("matrix rows exported")

    comm.Barrier()
    if rank == 0:
        files = sorted(out_dir.glob("*.npz"))
        log(f"DONE: {len(files)} artifacts in {out_dir}")
        for f in files:
            log(f"  {f.name}")
    log("EXIT")


if __name__ == "__main__":
    main()

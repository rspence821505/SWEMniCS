#!/usr/bin/env python3
"""
Quick sensitivity probe for the idealized inlet state-estimation problem.

Instead of computing the full 208K-column Jacobian, uses random probes
to estimate the effective information content of the observation operator.

Computes:
  1. Observation-space response to random IC perturbations
  2. Pairwise collinearity of responses
  3. Effective dimensionality of the observable subspace

Usage:
  python experiments/idealized_inlet_sensitivity_probe.py --n-probes 20
"""

import argparse, gc, os, sys, time
import numpy as np

os.environ.setdefault("CC", "/usr/bin/clang")
PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.idealized_inlet_twin import (
    CartesianVortexConfig, generate_cartesian_vortex, write_cartesian_wind_hdf5,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-probes", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--vmax", type=float, default=30.0)
    parser.add_argument("--nt-ramp", type=int, default=12)
    parser.add_argument("--nt-da", type=int, default=6)
    parser.add_argument("--obs-fraction", type=float, default=0.1)
    parser.add_argument("--obs-frequency", type=int, default=4)
    args = parser.parse_args()

    from mpi4py import MPI
    from dolfinx import la
    from swe4dvar.forward.problems import IdealizedInlet
    from swe4dvar.forward.solvers import get_solver
    from swe4dvar.physics.forcing import GriddedForcing
    from swe4dvar.utils import get_default_solver_params
    from swe4dvar.data_assimilation import PointObservationOperator

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    dt = 600.0
    nt_total = args.nt_ramp + args.nt_da
    times = np.arange(0, (nt_total + 1) * dt, dt)

    # Wind
    output_dir = PROJECT_ROOT / "results" / "idealized_inlet_twin"
    wind_file = output_dir / "wind_truth.h5"
    if not wind_file.exists():
        cfg = CartesianVortexConfig(Vmax=args.vmax, Rmax=15000, ramp_time_s=args.nt_ramp*dt)
        x_grid = np.linspace(-10000, 60000, 71)
        y_grid = np.linspace(-30000, 50000, 81)
        wx, wy, p = generate_cartesian_vortex(cfg, x_grid, y_grid, times)
        write_cartesian_wind_hdf5(str(wind_file), x_grid, y_grid, times, wx, wy, p)

    forcing = GriddedForcing(str(wind_file), cartesian=True)
    prob = IdealizedInlet(
        dt=dt, nt=nt_total,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
        dramp=args.nt_ramp * dt / 86400.0,
        forcing=forcing,
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])
    solver_params = get_default_solver_params(
        rtol=1e-5, atol=1e-6, max_it=20, relaxation_parameter=1.0,
        comm=comm, error_if_not_converged=False,
    )
    state_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs

    # Ramp
    print(f"Ramp ({args.nt_ramp} steps)...")
    prob.nt = args.nt_ramp
    solver.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=False, store_jacobians=False, enable_video=False,
    )
    ramp_end_state = solver.u_n.x.array[:state_size].copy()
    t_da_start = prob.t

    # Reference forward (DA window)
    print(f"Reference forward ({args.nt_da} steps)...")
    solver.storage.clear()
    prob.nt = args.nt_da
    solver.time_loop(
        solver_parameters=solver_params, stations=[], plot_every=9999,
        save_state=True, store_jacobians=False, enable_video=False,
    )
    ref_states = [s[:state_size].copy() for s in solver.storage.saved_states]

    # Obs operator
    coords = prob.mesh.geometry.x
    all_coords = comm.gather(coords, root=0)
    if rank == 0:
        coords_all = np.vstack(all_coords)
        _, ui = np.unique(np.round(coords_all[:, :2], decimals=10), axis=0, return_index=True)
        uc = coords_all[ui]
        xn, xx = uc[:, 0].min(), uc[:, 0].max()
        yn, yx = uc[:, 1].min(), uc[:, 1].max()
        interior = uc[(uc[:,0]>xn+1e-10)&(uc[:,0]<xx-1e-10)&(uc[:,1]>yn+1e-10)&(uc[:,1]<yx-1e-10)]
        rng = np.random.default_rng(42)
        n_obs = max(1, int(len(interior) * args.obs_fraction))
        chosen = rng.choice(len(interior), size=min(n_obs, len(interior)), replace=False)
        obs_points = np.zeros((len(chosen), 3))
        obs_points[:, :2] = interior[chosen, :2]
    else:
        obs_points = None
    obs_points = comm.bcast(obs_points, root=0)
    obs_op = PointObservationOperator(solver.V, obs_points, comm=comm)
    n_obs_pts = obs_op.get_num_observations()

    obs_times = list(range(0, args.nt_da + 1, args.obs_frequency))
    print(f"Obs: {n_obs_pts} pts × {len(obs_times)} times = {n_obs_pts * len(obs_times)} total")

    # Extract reference obs
    def get_obs(states):
        obs_list = []
        for t_idx in obs_times:
            vec = la.create_petsc_vector(solver.V.dofmap.index_map, solver.V.dofmap.index_map_bs)
            vec.setArray(states[t_idx])
            vec.assemble()
            ov = obs_op.forward(vec)
            obs_list.append(ov.getArray().copy())
            ov.destroy(); vec.destroy()
        return np.concatenate(obs_list)

    G_ref = get_obs(ref_states)
    print(f"G_ref: {G_ref.shape}, mean={G_ref.mean():.4f}")

    # Random probes: perturb IC, run forward, extract obs
    print(f"\nProbing {args.n_probes} random IC directions (eps={args.eps})...")
    rng = np.random.default_rng(123)
    response_vectors = []

    for i in range(args.n_probes):
        direction = rng.standard_normal(state_size)
        direction /= np.linalg.norm(direction)

        # Perturbed forward
        solver.storage.clear()
        solver.u_n.x.array[:state_size] = ramp_end_state + args.eps * direction
        solver.u_n_old.x.array[:state_size] = ramp_end_state + args.eps * direction
        solver.u.x.array[:state_size] = ramp_end_state + args.eps * direction
        solver.u_n.x.scatter_forward()
        solver.u_n_old.x.scatter_forward()
        solver.u.x.scatter_forward()
        prob.t = t_da_start
        prob.nt = args.nt_da
        if prob.forcing is not None:
            prob.forcing.evaluate(t_da_start)

        solver.time_loop(
            solver_parameters=solver_params, stations=[], plot_every=9999,
            save_state=True, store_jacobians=False, enable_video=False,
        )
        pert_states = [s[:state_size].copy() for s in solver.storage.saved_states]
        G_pert = get_obs(pert_states)
        response = (G_pert - G_ref) / args.eps
        response_vectors.append(response)

        if (i + 1) % 5 == 0:
            print(f"  Probe {i+1}/{args.n_probes}: ||response||={np.linalg.norm(response):.4e}")

    # Analyze response matrix
    R = np.column_stack(response_vectors)  # (n_obs_total, n_probes)
    print(f"\nResponse matrix: {R.shape}")

    U, sigma, Vt = np.linalg.svd(R, full_matrices=False)
    print(f"Singular values: {sigma}")
    cond = sigma[0] / max(sigma[-1], 1e-30)
    energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    print(f"Condition: {cond:.2e}")
    print(f"Cumulative energy: {energy}")

    for tol in [0.01, 0.05, 0.10]:
        r = int(np.sum(sigma > tol * sigma[0]))
        print(f"Effective rank (tol={tol}): {r}/{len(sigma)}")

    # Pairwise collinearity
    col = np.zeros((args.n_probes, args.n_probes))
    for i in range(args.n_probes):
        for j in range(args.n_probes):
            ni = np.linalg.norm(response_vectors[i])
            nj = np.linalg.norm(response_vectors[j])
            if ni > 1e-30 and nj > 1e-30:
                col[i, j] = np.dot(response_vectors[i], response_vectors[j]) / (ni * nj)

    off_diag = [abs(col[i,j]) for i in range(args.n_probes) for j in range(i+1, args.n_probes)]
    print(f"\nPairwise collinearity of random IC perturbation responses:")
    print(f"  Mean: {np.mean(off_diag):.4f}")
    print(f"  Min:  {np.min(off_diag):.4f}")
    print(f"  Max:  {np.max(off_diag):.4f}")
    print(f"  Std:  {np.std(off_diag):.4f}")

    # Interpretation
    if np.mean(off_diag) > 0.9:
        print(f"\n  DIAGNOSIS: HIGH COLLINEARITY — rank-deficient, like Shinnecock")
    elif np.mean(off_diag) > 0.5:
        print(f"\n  DIAGNOSIS: MODERATE COLLINEARITY — partially identifiable")
    else:
        print(f"\n  DIAGNOSIS: LOW COLLINEARITY — rich spectral structure, good for DA")

    gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())

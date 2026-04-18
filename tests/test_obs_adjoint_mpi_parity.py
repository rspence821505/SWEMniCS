#!/usr/bin/env python3
"""
Regression test: DG PointObservationOperator.adjoint() MPI parity.

Before the setValueLocal fix (and the upstream is_discontinuous_space fix),
the DG adjoint in MPI splattered observation residuals to wrong DG DOFs
because:
  1. is_discontinuous_space misclassified mixed DG as CG (fixed first)
  2. setValue(dof, ..., ADD) used GLOBAL DOF indices but the DOF passed in
     was a LOCAL index — rank 1's writes went to rank 0's DOFs

This test applies obs_operator.adjoint to an all-ones innovation and
verifies that the aggregated h-component values:
  * preserve total mass (sum = n_obs)
  * concentrate at exactly n_obs unique coordinates (the obs vertices)
  * have no nonzero contribution at coords that aren't observation points

Run:
  # Serial:
  python tests/test_obs_adjoint_mpi_parity.py
  # MPI (should match):
  PYTHONUNBUFFERED=1 mpirun -np 2 python tests/test_obs_adjoint_mpi_parity.py
"""
import os, sys
os.environ.setdefault("CC", "/usr/bin/clang")
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from swe4dvar.forward.problems import IdealizedInlet
from swe4dvar.forward.solvers import get_solver
from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig


def test_dg_point_adjoint_mpi_parity():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    prob = IdealizedInlet(
        dt=600.0, nt=2,
        xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        friction_law="mannings", solution_var="h",
    )
    solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])
    config = TwinExperimentConfig(
        method="4dvar", obs_fraction=0.1, obs_frequency=1,
        interior_only=True, verbose=(rank == 0), obs_seed=42,
    )
    exp = TwinExperiment(problem=prob, solver=solver, config=config, comm=comm)
    obs_points, obs_operator, _ = exp._setup_observations()

    # 1. Obs operator must detect as DG (regression for is_discontinuous_space)
    assert obs_operator.is_dg, "PointObservationOperator must detect mixed DG space"
    n_obs = obs_operator.get_num_observations()

    # 2. Apply adjoint to all-ones innovation
    innov = PETSc.Vec().createSeq(n_obs, comm=PETSc.COMM_SELF)
    innov.set(1.0)
    innov.assemble()
    adj_state = obs_operator.adjoint(innov)

    # 3. Gather owned h-component globally and aggregate by coord
    V = solver.V
    n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    h_sub = V.sub(0)
    h_space, h_map = h_sub.collapse()
    h_map_arr = np.asarray(h_map, dtype=np.int64)
    all_hcoords = h_space.tabulate_dof_coordinates()[:, :2]
    v_to_h = {int(v): i for i, v in enumerate(h_map_arr)}

    h_indices = np.unique(V.sub(0).dofmap.list.flatten())
    h_indices = h_indices[h_indices < n_owned]

    owned_arr = adj_state.getArray()
    h_values = owned_arr[h_indices]
    h_coords = np.array([all_hcoords[v_to_h[int(v)]] for v in h_indices])

    # Gather to rank 0
    if size == 1:
        all_coords = h_coords
        all_values = h_values
    else:
        sizes = comm.allgather(len(h_indices))
        counts = np.asarray(sizes, dtype=np.int64)
        displs = np.concatenate([[0], np.cumsum(counts)[:-1]])
        total = int(counts.sum())
        all_values = np.zeros(total, dtype=np.float64)
        all_coords_flat = np.zeros(2 * total, dtype=np.float64)
        comm.Allgatherv(
            [np.ascontiguousarray(h_coords.ravel()), 2*len(h_indices), MPI.DOUBLE],
            [all_coords_flat, counts * 2, displs * 2, MPI.DOUBLE],
        )
        comm.Allgatherv(
            [np.ascontiguousarray(h_values), len(h_indices), MPI.DOUBLE],
            [all_values, counts, displs, MPI.DOUBLE],
        )
        all_coords = all_coords_flat.reshape(-1, 2)

    if rank == 0:
        # Build obs-point coord set
        obs_set = set((float(x), float(y)) for x, y in obs_points[:, :2])

        # Aggregate by unique coord
        rounded = np.round(all_coords, 6)
        unique_coords, inv = np.unique(rounded, axis=0, return_inverse=True)
        n_unique = inv.max() + 1
        v_agg = np.zeros(n_unique)
        np.add.at(v_agg, inv, all_values)

        # Checks
        nonzero_mask = np.abs(v_agg) > 1e-10
        nonzero_coords = unique_coords[nonzero_mask]

        # Total mass
        total_mass = float(np.sum(all_values))
        mass_rel_err = abs(total_mass - n_obs) / n_obs
        assert mass_rel_err < 1e-6, (
            f"Total mass must equal n_obs={n_obs}, got {total_mass}, "
            f"rel_err={mass_rel_err}"
        )

        # Count of nonzero unique coords must equal n_obs
        assert int(np.sum(nonzero_mask)) == n_obs, (
            f"DG adjoint should write to exactly n_obs={n_obs} unique coords, "
            f"got {int(np.sum(nonzero_mask))} — splatter means MPI parity is broken"
        )

        # All nonzero coords must be observation points
        n_at_obs = sum(1 for x, y in nonzero_coords if (float(x), float(y)) in obs_set)
        assert n_at_obs == n_obs, (
            f"Every nonzero coord must be an obs point: "
            f"{n_at_obs}/{len(nonzero_coords)} at obs, "
            f"{len(nonzero_coords) - n_at_obs} are splatter"
        )

        print(f"PASS [size={size}]: total_mass={total_mass:.4f} (expected {n_obs}), "
              f"n_nonzero_unique={int(np.sum(nonzero_mask))} == n_obs")


if __name__ == "__main__":
    test_dg_point_adjoint_mpi_parity()

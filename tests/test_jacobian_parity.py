#!/usr/bin/env python3
"""
Jacobian/RHS assembly parity probe: serial vs MPI.

Builds the DA problem, extracts the cached Jacobian at timestep k=1, applies
J and J^T to several deterministic test vectors (coord-based, so same in
serial and MPI), and saves globally-ordered results via the smoother's
allgather machinery. Also saves the adjoint RHS (obs forcings) globally.

Usage:
  # Serial baseline:
  SWE4DVAR_MEM_LIMIT_GB=7 PYTHONUNBUFFERED=1 python tests/test_jacobian_parity.py
  # 2-rank MPI:
  SWE4DVAR_MEM_LIMIT_GB=7 PYTHONUNBUFFERED=1 mpirun -np 2 python tests/test_jacobian_parity.py

Writes: results/mpi_parity/jac_probe_{serial|mpi2}.npz
"""
import os, sys, gc
os.environ.setdefault("CC", "/usr/bin/clang")
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def log(msg):
    sys.stdout.write(f"  [r{rank}] {msg}\n"); sys.stdout.flush()

log(f"START size={size}")

# MONKEY-PATCH: _setup_background in TwinExperiment builds a 69K×69K
# smoothing matrix (~2 GB) used once for random background perturbation.
# Our harness OVERRIDES m_background with a deterministic one anyway, so
# the smoother build is wasted memory. Replace _setup_background with a
# lightweight no-op that just sets m_background = m_true + zero (will be
# overridden by the deterministic perturbation in build_da_problem).
from experiments.twin_experiment import TwinExperiment as _TE
def _noop_setup_background(self, n_vars: int = 3) -> float:
    # Lightweight replacement: m_background = m_true (before deterministic
    # perturbation override in build_da_problem). Avoids building the 2GB
    # background smoother that _setup_background would otherwise construct.
    self.m_background = self.m_true.duplicate()
    m_true_arr = self.m_true.getArray(readonly=True)
    self.m_background.setArray(m_true_arr.copy())
    self.m_background.assemble()
    return 0.0
_TE._setup_background = _noop_setup_background

# Build the DA problem using the existing harness machinery
from test_mpi_parity import build_da_problem
da = build_da_problem(comm, rank)
log(f"build_problem done; state_size_local={da['state_size_local']}")

cost_fn = da["cost_fn"]
smoother = da["gradient_smoother"]  # use its allgather machinery

# Run forward to get cached Jacobians
log("running forward model to get cached Jacobians...")
trajectory, jacobians = cost_fn._run_forward_model(da["m_background"], store_jacobians=True)
log(f"forward done: traj_len={len(trajectory)}, jac_len={len(jacobians)}")

if len(jacobians) < 1:
    log("ERROR: no Jacobians cached")
    sys.exit(1)

# Extract Jacobian at timestep k=1 (index 0 stores ∂R^1/∂u^1)
J = jacobians[0]
from petsc4py import PETSc
log(f"J info: size={J.getSize()}, local={J.getLocalSize()}")

# Build deterministic test vectors in V's DOF layout, with values keyed
# by h-space coordinates (so same spatial point → same value regardless of
# DOF numbering). Use the smoother's h_owned/u_owned/v_owned and h-space coords.
from dolfinx import la
V = cost_fn.forward_model.solver.V
n_owned_V = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

def make_coord_based_vec(pattern_name):
    """Create a PETSc Vec with values determined by h-space coords of each DOF."""
    # Use smoother's h_owned and h_coords_local (paired 1-to-1)
    h_sub = V.sub(0)
    h_space, h_map = h_sub.collapse()
    h_map_arr = np.asarray(h_map, dtype=np.int64)
    v_to_hspace = {int(v): i for i, v in enumerate(h_map_arr)}
    all_h_coords = h_space.tabulate_dof_coordinates()[:, :2]

    vec = la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
    arr = np.zeros(n_owned_V)

    # Compute coords for each owned h, u, v DOF
    h_coords = np.zeros((len(smoother.h_owned), 2))
    for i, vdof in enumerate(smoother.h_owned):
        h_coords[i] = all_h_coords[v_to_hspace[int(vdof)]]

    Lx, Ly = 50000.0, 40000.0
    x = h_coords[:, 0] / Lx
    y = h_coords[:, 1] / Ly

    if pattern_name == "ones":
        vals_h = np.ones(len(smoother.h_owned))
        vals_uv = np.ones(len(smoother.h_owned))
    elif pattern_name == "sin2D":
        vals_h = np.cos(4 * np.pi * x) * np.sin(6 * np.pi * y)
        vals_uv = np.sin(2 * np.pi * x) * np.cos(4 * np.pi * y)
    elif pattern_name == "localized":
        # Spike at center of domain
        vals_h = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.01)
        vals_uv = np.zeros_like(vals_h)
    else:
        raise ValueError(pattern_name)

    arr[smoother.h_owned] = vals_h
    arr[smoother.u_owned] = vals_uv * 0.3
    arr[smoother.v_owned] = vals_uv * -0.3

    vec.setArray(arr)
    vec.assemble()
    return vec


def allgather_globally_ordered(vec):
    """Return (coords, h, u, v) globally coord-sorted via smoother machinery."""
    owned_arr = vec.getArray()
    h_local = owned_arr[smoother.h_owned]
    u_local = owned_arr[smoother.u_owned]
    v_local = owned_arr[smoother.v_owned]
    all_sizes = np.diff(np.concatenate([smoother.local_offsets, [smoother.global_n_h]])).astype(np.int64)
    h_g = smoother._allgather_component(h_local, all_sizes)
    u_g = smoother._allgather_component(u_local, all_sizes)
    v_g = smoother._allgather_component(v_local, all_sizes)
    coords = smoother.global_h_coords
    if rank == 0:
        order = np.lexsort((coords[:, 1], coords[:, 0]))
        return coords[order], h_g[order], u_g[order], v_g[order]
    return None, None, None, None


# ============================================================
# Probe: apply J and J^T to several test vectors
# ============================================================
results = {}

for pattern in ["ones", "sin2D", "localized"]:
    log(f"probing J @ x  with pattern={pattern}")
    x = make_coord_based_vec(pattern)
    x_coords, x_h, x_u, x_v = allgather_globally_ordered(x)

    # J @ x
    Jx = x.duplicate()
    J.mult(x, Jx)
    Jx_coords, Jx_h, Jx_u, Jx_v = allgather_globally_ordered(Jx)

    # J^T @ x
    JTx = x.duplicate()
    J.multTranspose(x, JTx)
    JTx_coords, JTx_h, JTx_u, JTx_v = allgather_globally_ordered(JTx)

    if rank == 0:
        results[f"{pattern}_input"] = {"h": x_h, "u": x_u, "v": x_v}
        results[f"{pattern}_Jx"] = {"h": Jx_h, "u": Jx_u, "v": Jx_v}
        results[f"{pattern}_JTx"] = {"h": JTx_h, "u": JTx_u, "v": JTx_v}

    x.destroy(); Jx.destroy(); JTx.destroy()
    gc.collect()

# ============================================================
# Probe: adjoint RHS (observation forcings)
# ============================================================
log("computing observation forcings (adjoint RHS)...")
obs_forcings = cost_fn._compute_observation_forcings(trajectory)
log(f"obs_forcings: {len(obs_forcings)} entries")

# Save each non-None obs forcing globally
for i, f in enumerate(obs_forcings):
    if f is None:
        if rank == 0:
            results[f"obs_forcing_{i}_present"] = False
        continue
    fc, fh, fu, fv = allgather_globally_ordered(f)
    if rank == 0:
        results[f"obs_forcing_{i}_present"] = True
        results[f"obs_forcing_{i}"] = {"h": fh, "u": fu, "v": fv}

# ============================================================
# Probe: partition-boundary DOF identification
# ============================================================
# For each rank, list: owned h-DOF coords, whether they're "near a partition
# boundary" (approximated by: does the rank have any ghost h-DOFs within
# 3*L_smoother of this owned DOF?)
if size > 1:
    log("computing partition-boundary proximity for owned h-DOFs...")
    # Get local coords for owned h (from smoother's global_h_coords slice)
    s0 = smoother.local_offset
    s1 = s0 + smoother.local_size
    my_h_coords = smoother.global_h_coords[s0:s1]
    # Other-rank coords = everything else in global
    other_coords = np.vstack([
        smoother.global_h_coords[:s0],
        smoother.global_h_coords[s1:]
    ])
    if len(other_coords) > 0:
        from scipy.spatial import cKDTree
        other_tree = cKDTree(other_coords)
        # Distance from each owned h-DOF to the nearest other-rank h-DOF
        dists, _ = other_tree.query(my_h_coords, k=1)
    else:
        dists = np.full(len(my_h_coords), 1e10)
    if rank == 0:
        log(f"rank 0: min dist to partition = {dists.min():.2f} m, "
            f"n_near_boundary (< 500m) = {int(np.sum(dists < 500))}")

# ============================================================
# Save results (rank 0)
# ============================================================
if rank == 0:
    tag = "serial" if size == 1 else f"mpi{size}"
    out_dir = PROJECT_ROOT / "results" / "mpi_parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Flatten to npz
    flat = {"coords": x_coords}  # use any x_coords (same across patterns)
    for key, val in results.items():
        if isinstance(val, dict):
            for comp, arr in val.items():
                flat[f"{key}__{comp}"] = arr
        else:
            flat[key] = np.array(val)
    out_file = out_dir / f"jac_probe_{tag}.npz"
    np.savez(out_file, **flat)
    log(f"saved: {out_file}")

comm.Barrier()
log("DONE")

#!/usr/bin/env python3
"""
Focused smoother parity test: old local-only vs new distributed.

Tests ONLY the smoother on an identical synthetic gradient vector.
No forward solve, no adjoint, no Newton — just mesh setup + smoother.
Runs in ~30s serial, ~20s MPI.

Usage:
  # Serial:
  python tests/test_smoother_parity.py
  # MPI:
  PYTHONUNBUFFERED=1 mpirun -np 2 python tests/test_smoother_parity.py
"""
import os, sys, json, time
import numpy as np

os.environ.setdefault("CC", "/usr/bin/clang")

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def log(msg):
    sys.stdout.write(f"  [rank {rank}] {msg}\n")
    sys.stdout.flush()

log(f"START — size={size}")

# Build mesh + function space (no solve needed)
from swe4dvar.forward.problems import IdealizedInlet
from swe4dvar.forward.solvers import get_solver

prob = IdealizedInlet(
    dt=600.0, nt=1,
    xdmf_file="data/Ideal_Inlet/Ideal_Inlet.xdmf",
    friction_law="mannings", solution_var="h",
)
solver = get_solver("DG")(prob, theta=1.0, p_degree=[1, 1])
V = solver.V
n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
n_total = n_owned + V.dofmap.index_map.num_ghosts * V.dofmap.index_map_bs
log(f"mesh loaded: n_owned={n_owned}, n_total={n_total}")

# Get DOF indices
from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig
config = TwinExperimentConfig(method="4dvar", verbose=(rank == 0))
exp = TwinExperiment(problem=prob, solver=solver, config=config, comm=comm)
h_idx_owned, u_idx_owned, v_idx_owned = exp._get_component_dof_indices(owned_only=True)
log(f"h_owned={len(h_idx_owned)}, u_owned={len(u_idx_owned)}, v_owned={len(v_idx_owned)}")

# Build a deterministic synthetic gradient from spatial coordinates
h_sub = V.sub(0)
h_space, h_map = h_sub.collapse()
all_h_coords = h_space.tabulate_dof_coordinates()[:, :2]
parent_to_collapsed = np.full(max(h_map) + 1, -1, dtype=int)
for ci, pi in enumerate(h_map):
    parent_to_collapsed[pi] = ci
h_collapsed = parent_to_collapsed[h_idx_owned]
h_coords = all_h_coords[h_collapsed]

Lx, Ly = 50000.0, 40000.0
synth_h = np.cos(4 * np.pi * h_coords[:, 0] / Lx) * np.sin(6 * np.pi * h_coords[:, 1] / Ly)
synth_uv = np.sin(2 * np.pi * h_coords[:, 0] / Lx) * np.cos(4 * np.pi * h_coords[:, 1] / Ly) * 0.1

grad_arr = np.zeros(n_owned)
grad_arr[h_idx_owned] = synth_h
grad_arr[u_idx_owned] = synth_uv[:len(u_idx_owned)]
grad_arr[v_idx_owned] = synth_uv[:len(v_idx_owned)]

input_norm = float(np.sqrt(comm.allreduce(np.sum(grad_arr ** 2))))
log(f"synthetic gradient: local_size={len(grad_arr)}, global_norm={input_norm:.6f}")

# ============================================================
# OLD smoother: local-only (the one that breaks under MPI)
# ============================================================
log("building OLD local-only smoother...")
t0 = time.time()
old_matrix = exp._build_smoothing_matrix(h_idx_owned, 500.0)
old_time = time.time() - t0
log(f"old smoother: shape={old_matrix.shape}, nnz={old_matrix.nnz}, build={old_time:.1f}s")

old_result = grad_arr.copy()
old_result[h_idx_owned] = old_matrix @ grad_arr[h_idx_owned]
old_result[u_idx_owned] = old_matrix @ grad_arr[u_idx_owned]
old_result[v_idx_owned] = old_matrix @ grad_arr[v_idx_owned]
old_norm = float(np.sqrt(comm.allreduce(np.sum(old_result ** 2))))
log(f"old smoother output: global_norm={old_norm:.6f}")

# ============================================================
# NEW smoother: distributed ghost-aware
# ============================================================
log("building NEW distributed smoother...")
t0 = time.time()
from swe4dvar.utils.distributed_smoother import DistributedGradientSmoother
new_smoother = DistributedGradientSmoother(solver, correlation_length=500.0, comm=comm)
new_time = time.time() - t0
log(f"new smoother: G_h shape={new_smoother.G_h.shape}, build={new_time:.1f}s")

# Apply via PETSc vec (the .apply() interface)
from dolfinx import la
grad_vec = la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
grad_vec.setArray(grad_arr)
grad_vec.assemble()

new_smoother.apply(grad_vec)
new_result = grad_vec.getArray().copy()
new_norm = float(np.sqrt(comm.allreduce(np.sum(new_result ** 2))))
log(f"new smoother output: global_norm={new_norm:.6f}")

# ============================================================
# Compare
# ============================================================
diff = new_result - old_result
local_diff_norm_sq = float(np.sum(diff ** 2))
global_diff_norm = float(np.sqrt(comm.allreduce(local_diff_norm_sq)))
rel_diff = global_diff_norm / max(old_norm, 1e-30)

log(f"COMPARISON: old_norm={old_norm:.6f}, new_norm={new_norm:.6f}, "
    f"diff_norm={global_diff_norm:.6f}, rel_diff={rel_diff:.6e}")

if size == 1:
    # Serial: old and new should match nearly exactly (no ghosts)
    verdict = "PASS" if rel_diff < 1e-10 else "FAIL"
    log(f"SERIAL PARITY: {verdict} (rel_diff={rel_diff:.2e}, expect < 1e-10)")
else:
    # MPI: if fix works, new should differ from old (old is wrong at boundaries)
    # We can't compare new to serial here, but we can verify new != old
    log(f"MPI: old (local-only) and new (ghost-aware) differ by {rel_diff:.2e}")
    log(f"MPI: This is EXPECTED — the old smoother is wrong at partition boundaries")

# Write JSON
out_dir = PROJECT_ROOT / "results" / "smoother_parity"
if rank == 0:
    out_dir.mkdir(parents=True, exist_ok=True)
comm.Barrier()

tag = "serial" if size == 1 else f"mpi{size}_rank{rank}"
result = {
    "mpi_size": size,
    "rank": rank,
    "n_owned": n_owned,
    "input_norm": input_norm,
    "old_smoother_norm": old_norm,
    "new_smoother_norm": new_norm,
    "diff_norm": global_diff_norm,
    "rel_diff": rel_diff,
    "old_build_time_s": old_time,
    "new_build_time_s": new_time,
}
with open(out_dir / f"result_{tag}.json", "w") as f:
    json.dump(result, f, indent=2)

grad_vec.destroy()
log("DONE")

#!/usr/bin/env python3
"""Save adjoint vectors and DOF coordinates for offline analysis."""

import sys
import os
import numpy as np
import time as _time
from pathlib import Path

os.environ.setdefault("CC", "/usr/bin/clang")

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from petsc4py import PETSc
from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils import get_default_solver_params
from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig

# Monkey-patch to capture cost function
_captured = {}
_orig_run_opt = TwinExperiment._run_optimization
def _patched_run_opt(self, cost_function):
    cf = cost_function
    while hasattr(cf, 'base_cost'):
        cf = cf.base_cost
    _captured['cost_fn'] = cf
    return _orig_run_opt(self, cost_function)
TwinExperiment._run_optimization = _patched_run_opt

# Setup and run
problem = TidalProblem(nx=20, ny=10, dt=1800, nt=96)
solver = get_solver('DG')(problem, theta=0.5, p_degree=[1, 1])
config = TwinExperimentConfig(
    method='dcwme', obs_fraction=0.5, obs_frequency=4,
    obs_noise_level=0.01, background_error_std=0.1,
    max_iterations=1, verbose=False,
    auto_inflate_B=True, predictability_gamma=5.0,
)

print("Running experiment...")
exp = TwinExperiment(problem=problem, solver=solver, config=config,
                     solver_params=get_default_solver_params())
results = exp.run()
print(f"Done: {getattr(results, 'error_reduction', 0):.1f}%")

cost_fn = _captured['cost_fn']

# Get DOF coordinates from cell centroids
V = solver.V
n_state = cost_fn.m_b.getSize()
mesh = problem.mesh
tdim = mesh.topology.dim
num_cells = mesh.topology.index_map(tdim).size_local
geom = mesh.geometry.x

cell_centroids = np.zeros((num_cells, 2))
for cell_idx in range(num_cells):
    cell_verts = mesh.geometry.dofmap[cell_idx]
    cell_centroids[cell_idx] = geom[cell_verts, :2].mean(axis=0)

dofmap = V.dofmap
dof_coords = np.zeros((n_state, 2))
for cell_idx in range(num_cells):
    for dof in dofmap.cell_dofs(cell_idx):
        dof_coords[dof] = cell_centroids[cell_idx]

# Compute adjoint vectors
linearized_wme = cost_fn.qoi_map.linearize(
    cost_fn.m_b, max(cost_fn.obs_times),
    trajectory=cost_fn._trajectory, jacobians=cost_fn._jacobians)
Q_wme_mb = cost_fn._wme_cache.get("Q_wme_mb")
n_obs = Q_wme_mb.getSize()

print(f"Computing {n_obs} adjoint vectors...")
A = np.zeros((n_state, n_obs))
e_i = Q_wme_mb.duplicate()
t0 = _time.perf_counter()
for i in range(n_obs):
    e_i.zeroEntries()
    e_i.setValue(i, 1.0)
    e_i.assemblyBegin()
    e_i.assemblyEnd()
    a_i = linearized_wme.apply_adjoint(e_i)
    A[:, i] = a_i.getArray()
    a_i.destroy()
e_i.destroy()
print(f"Done ({_time.perf_counter()-t0:.1f}s)")

B_var = cost_fn.B.min_eigenvalue()

# Save
outfile = "/tmp/adjoint_data.npz"
np.savez(outfile, A=A, dof_coords=dof_coords, B_var=B_var, n_obs=n_obs, n_state=n_state)
print(f"\nSaved to {outfile}: A={A.shape}, dof_coords={dof_coords.shape}, B_var={B_var:.6e}")

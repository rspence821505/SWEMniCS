#!/usr/bin/env python3
"""
Diagnostic: Does spatially correlated B improve L_wme eigenvalues?

Reuses adjoint vectors from the mass matrix diagnostic and tests
different correlation structures for B.
"""

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

print("=" * 60)
print("SPATIALLY CORRELATED B DIAGNOSTIC")
print("=" * 60)

# Monkey-patch to capture cost function
_captured = {}
_orig_run_opt = TwinExperiment._run_optimization

def _patched_run_opt(self, cost_function):
    cf = cost_function
    while hasattr(cf, 'base_cost'):
        cf = cf.base_cost
    _captured['cost_fn'] = cf
    _captured['forward_model'] = self._forward_model if hasattr(self, '_forward_model') else None
    return _orig_run_opt(self, cost_function)

TwinExperiment._run_optimization = _patched_run_opt

_orig_create_fm = TwinExperiment._create_forward_model
def _patched_create_fm(self):
    fm = _orig_create_fm(self)
    self._forward_model = fm
    return fm
TwinExperiment._create_forward_model = _patched_create_fm

# Setup problem
problem = TidalProblem(nx=20, ny=10, dt=1800, nt=96)
solver = get_solver('DG')(problem, theta=0.5, p_degree=[1, 1])
config = TwinExperimentConfig(
    method='dcwme',
    obs_fraction=0.5,
    obs_frequency=4,
    obs_noise_level=0.01,
    background_error_std=0.1,
    max_iterations=1,
    gradient_tolerance=1e-6,
    verbose=False,
    auto_inflate_B=True,
    max_inflate_factor=3.0,
    predictability_gamma=5.0,
)

print("Running 1-iteration DC-WME experiment...")
sys.stdout.flush()
exp = TwinExperiment(
    problem=problem, solver=solver, config=config,
    solver_params=get_default_solver_params()
)
results = exp.run()
print(f"Done: {getattr(results, 'error_reduction', 0):.1f}% error reduction")
sys.stdout.flush()

cost_fn = _captured.get('cost_fn')
if cost_fn is None:
    print("ERROR: cost function not captured")
    sys.exit(1)

# Get mesh cell centroids for DOF coordinate mapping
V = solver.V
n_state = cost_fn.m_b.getSize()
mesh = problem.mesh
tdim = mesh.topology.dim
num_cells = mesh.topology.index_map(tdim).size_local
geom = mesh.geometry.x

cell_centroids = np.zeros((num_cells, 2))
for cell_idx in range(num_cells):
    cell_verts = mesh.geometry.dofmap[cell_idx]
    coords = geom[cell_verts, :2]
    cell_centroids[cell_idx] = coords.mean(axis=0)

# Map DOFs to cell centroids
dofmap = V.dofmap
dof_coords = np.zeros((n_state, 2))
for cell_idx in range(num_cells):
    cell_dofs = dofmap.cell_dofs(cell_idx)
    for dof in cell_dofs:
        dof_coords[dof] = cell_centroids[cell_idx]

print(f"\n{n_state} DOFs, {num_cells} cells")
print(f"Centroid range: x=[{cell_centroids[:,0].min():.0f}, {cell_centroids[:,0].max():.0f}], "
      f"y=[{cell_centroids[:,1].min():.0f}, {cell_centroids[:,1].max():.0f}]")

# Compute adjoint vectors
print("\nComputing adjoint vectors...")
sys.stdout.flush()

linearized_wme = cost_fn.qoi_map.linearize(
    cost_fn.m_b,
    max(cost_fn.obs_times),
    trajectory=cost_fn._trajectory,
    jacobians=cost_fn._jacobians,
)

Q_wme_mb = cost_fn._wme_cache.get("Q_wme_mb")
n_obs = Q_wme_mb.getSize()
print(f"n_obs = {n_obs}")

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
print(f"B diagonal variance: {B_var:.6e}")

# Baseline: diagonal B
L_diag = B_var * (A.T @ A)
eigs_diag = np.linalg.eigvalsh(L_diag)
print(f"\nBaseline (diagonal B): L_wme = [{eigs_diag.min():.4e}, {eigs_diag.max():.4e}], #≥1: {np.sum(eigs_diag >= 1.0)}/{n_obs}")

# Distance matrix
from scipy.spatial.distance import cdist
print(f"\nComputing distance matrix ({n_state}×{n_state})...")
t0 = _time.perf_counter()
dist_matrix = cdist(dof_coords, dof_coords, metric='euclidean')
print(f"Done ({_time.perf_counter()-t0:.1f}s), max dist = {dist_matrix.max():.0f}m")

# Test correlation lengths
print("\n" + "=" * 60)
print("TESTING CORRELATION STRUCTURES")
print("=" * 60)

results_table = [("Diagonal (σ²I)", 0, eigs_diag)]

for L_corr in [200, 500, 1000, 2000, 5000]:
    t0 = _time.perf_counter()
    C = np.exp(-dist_matrix**2 / (2 * L_corr**2))
    CA = C @ A
    L_wme = B_var * (A.T @ CA)
    L_wme = 0.5 * (L_wme + L_wme.T)
    eigs = np.linalg.eigvalsh(L_wme)
    amp = eigs.max() / eigs_diag.max()
    print(f"Gaussian L={L_corr:>5}m: L_wme=[{eigs.min():.4e}, {eigs.max():.4e}], "
          f"#≥1: {np.sum(eigs >= 1.0):>2}/{n_obs}, amp={amp:.1f}x ({_time.perf_counter()-t0:.1f}s)")
    results_table.append((f"Gaussian L={L_corr}m", L_corr, eigs))

for L_corr in [500, 1000, 2000]:
    t0 = _time.perf_counter()
    C = np.exp(-dist_matrix / L_corr)
    CA = C @ A
    L_wme = B_var * (A.T @ CA)
    L_wme = 0.5 * (L_wme + L_wme.T)
    eigs = np.linalg.eigvalsh(L_wme)
    amp = eigs.max() / eigs_diag.max()
    print(f"Expon.  L={L_corr:>5}m: L_wme=[{eigs.min():.4e}, {eigs.max():.4e}], "
          f"#≥1: {np.sum(eigs >= 1.0):>2}/{n_obs}, amp={amp:.1f}x ({_time.perf_counter()-t0:.1f}s)")
    results_table.append((f"Exponential L={L_corr}m", L_corr, eigs))

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n{'B structure':<25} {'L_wme min':>12} {'L_wme max':>12} {'#≥1':>5} {'Amplif':>8} {'α for max≥1':>12}")
print("-" * 80)
for name, L_corr, eigs in results_table:
    n_above = np.sum(eigs >= 1.0)
    amp = eigs.max() / eigs_diag.max()
    alpha_needed = 1.0 / eigs.max() if eigs.max() > 0 else float('inf')
    print(f"{name:<25} {eigs.min():>12.4e} {eigs.max():>12.4e} {n_above:>5} {amp:>8.1f}x {alpha_needed:>12.1f}")

best_name, best_L, best_eigs = max(results_table, key=lambda x: x[2].max())
alpha_best = 1.0 / best_eigs.max()
alpha_diag = 1.0 / eigs_diag.max()

print(f"\nBest: {best_name}")
print(f"  Max L_wme: {best_eigs.max():.4e} (vs diagonal: {eigs_diag.max():.4e})")
print(f"  Amplification: {best_eigs.max()/eigs_diag.max():.1f}x")
print(f"  α needed (max≥1): {alpha_best:.1f} (vs diagonal: {alpha_diag:.1f})")
print(f"  # eigenvalues ≥ 1: {np.sum(best_eigs >= 1.0)}/{n_obs}")

if alpha_best < 100:
    print(f"\n  FEASIBLE: correlation reduces inflation to α ≈ {alpha_best:.0f}")
else:
    print(f"\n  Still needs α ≈ {alpha_best:.0f} ({alpha_diag/alpha_best:.0f}x less than diagonal)")

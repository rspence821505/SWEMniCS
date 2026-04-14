#!/usr/bin/env python3
"""
Diagnostic: Does M⁻¹ correction change L_wme eigenvalues?

Tests whether adjoint vectors' embedded mass matrix M affects the
Gram/L_wme eigenvalue spectrum that controls DC-WME effectiveness.
"""

import sys
import os
import numpy as np
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
print("MASS MATRIX SCALING DIAGNOSTIC")
print("=" * 60)

# Monkey-patch TwinExperiment to capture the cost function
_captured = {}
_orig_run_opt = TwinExperiment._run_optimization

def _patched_run_opt(self, cost_function):
    # Unwrap wrappers (ZeroBoundaryGradientCost, MassMatrixPreconditionedCost)
    cf = cost_function
    while hasattr(cf, 'base_cost'):
        cf = cf.base_cost
    _captured['cost_fn'] = cf
    _captured['forward_model'] = self._forward_model if hasattr(self, '_forward_model') else None
    return _orig_run_opt(self, cost_function)

TwinExperiment._run_optimization = _patched_run_opt

# Also capture the forward model
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
    predictability_gamma=5.0,
)

print("Running 1-iteration DC-WME experiment...")
exp = TwinExperiment(
    problem=problem, solver=solver, config=config,
    solver_params=get_default_solver_params()
)
results = exp.run()
print(f"Experiment done: {getattr(results, 'error_reduction', 0):.1f}% error reduction")

# Get captured objects
cost_fn = _captured.get('cost_fn')
forward_model = _captured.get('forward_model')

if cost_fn is None:
    print("ERROR: cost function not captured")
    sys.exit(1)

print(f"Cost function type: {type(cost_fn).__name__}")

# Get mass matrix
if hasattr(forward_model, 'get_mass_matrix'):
    M = forward_model.get_mass_matrix()
    M_diag = M.getDiagonal()
    M_arr = M_diag.getArray()
    print(f"\nMass matrix diagonal: min={M_arr.min():.4f}, max={M_arr.max():.4f}, mean={M_arr.mean():.4f}")
    M_diag.destroy()

# Get stored Gram eigenvalues
if hasattr(cost_fn, '_gram_eigenvalues') and cost_fn._gram_eigenvalues is not None:
    gram_eigs_stored = cost_fn._gram_eigenvalues
    print(f"\nStored Gram eigenvalues: [{gram_eigs_stored.min():.6e}, {gram_eigs_stored.max():.6e}]")
    print(f"  Count above 1.0: {np.sum(gram_eigs_stored >= 1.0)}/{len(gram_eigs_stored)}")

# Recompute adjoint vectors for the diagnostic
print("\n" + "=" * 60)
print("RECOMPUTING ADJOINT VECTORS FOR COMPARISON")
print("=" * 60)

linearized_wme = cost_fn.qoi_map.linearize(
    cost_fn.m_b,
    max(cost_fn.obs_times),
    trajectory=cost_fn._trajectory,
    jacobians=cost_fn._jacobians,
)

Q_wme_mb = cost_fn._wme_cache.get("Q_wme_mb")
n_obs = Q_wme_mb.getSize()
print(f"n_obs = {n_obs}")

# Compute raw adjoint vectors
adjoint_vectors = []
e_i = Q_wme_mb.duplicate()
for i in range(n_obs):
    e_i.zeroEntries()
    e_i.setValue(i, 1.0)
    e_i.assemblyBegin()
    e_i.assemblyEnd()
    a_i = linearized_wme.apply_adjoint(e_i)
    adjoint_vectors.append(a_i)
e_i.destroy()

norms_raw = [a.norm() for a in adjoint_vectors]
print(f"Raw adjoint vector norms: min={min(norms_raw):.4e}, max={max(norms_raw):.4e}")

# Raw Gram matrix
G_raw = np.zeros((n_obs, n_obs))
for i in range(n_obs):
    for j in range(i, n_obs):
        val = adjoint_vectors[i].dot(adjoint_vectors[j])
        G_raw[i, j] = val
        G_raw[j, i] = val

eigs_raw = np.linalg.eigvalsh(G_raw)

# M⁻¹ corrected: solve M x = a_i
from dolfinx import fem
from ufl import TrialFunction, TestFunction, inner, dx

V = solver.V
u_trial, v_test = TrialFunction(V), TestFunction(V)
M_mat = fem.petsc.assemble_matrix(fem.form(inner(u_trial, v_test) * dx))
M_mat.assemble()

ksp = PETSc.KSP().create(M_mat.getComm())
ksp.setOperators(M_mat)
ksp.setType(PETSc.KSP.Type.CG)
ksp.getPC().setType(PETSc.PC.Type.JACOBI)
ksp.setTolerances(rtol=1e-12, atol=1e-14)
ksp.setUp()

adjoint_corrected = []
for i in range(n_obs):
    x = adjoint_vectors[i].duplicate()
    ksp.solve(adjoint_vectors[i], x)
    adjoint_corrected.append(x)

norms_corr = [a.norm() for a in adjoint_corrected]
print(f"M⁻¹ corrected vector norms: min={min(norms_corr):.4e}, max={max(norms_corr):.4e}")
print(f"Norm ratio raw/corrected: {np.mean(norms_raw)/np.mean(norms_corr):.2f}")

# Corrected Gram matrix
G_corrected = np.zeros((n_obs, n_obs))
for i in range(n_obs):
    for j in range(i, n_obs):
        val = adjoint_corrected[i].dot(adjoint_corrected[j])
        G_corrected[i, j] = val
        G_corrected[j, i] = val

eigs_corr = np.linalg.eigvalsh(G_corrected)

# Also try M-weighted inner product: G_M[i,j] = a_i^T M⁻¹ a_j
# This is the proper dual-space inner product
G_M = np.zeros((n_obs, n_obs))
for i in range(n_obs):
    # a_i^T M⁻¹ a_j = a_i^T x_j where M x_j = a_j (already computed)
    for j in range(i, n_obs):
        val = adjoint_vectors[i].dot(adjoint_corrected[j])
        G_M[i, j] = val
        G_M[j, i] = val

eigs_M = np.linalg.eigvalsh(G_M)

# Results
B_var = cost_fn.B.min_eigenvalue()
print(f"\n" + "=" * 60)
print(f"RESULTS")
print(f"=" * 60)
print(f"B min eigenvalue: {B_var:.6e}")
print(f"\n{'Variant':<30} {'Gram min':>12} {'Gram max':>12} {'L_wme min':>12} {'L_wme max':>12} {'#>1':>5}")
print("-" * 85)

variants = [
    ("Raw (a_i^T a_j)", eigs_raw),
    ("M⁻¹ corrected (λ_i^T λ_j)", eigs_corr),
    ("M-weighted (a_i^T M⁻¹ a_j)", eigs_M),
]

for name, eigs in variants:
    L = B_var * eigs
    n_above = np.sum(L >= 1.0)
    print(f"{name:<30} {eigs.min():>12.4e} {eigs.max():>12.4e} {L.min():>12.4e} {L.max():>12.4e} {n_above:>5}")

print(f"\nDetailed eigenvalue distributions:")
for name, eigs in variants:
    L = B_var * eigs
    print(f"\n  {name}:")
    print(f"    Gram: top5={eigs[-5:]}")
    print(f"    Gram: bot5={eigs[:5]}")
    print(f"    L_wme: top5={L[-5:]}")
    print(f"    L_wme: bot5={L[:5]}")

# What inflation is needed?
print(f"\n" + "=" * 60)
print(f"INFLATION REQUIREMENTS")
print(f"=" * 60)
for name, eigs in variants:
    L = B_var * eigs
    alpha_max1 = 1.0 / L.max() if L.max() > 0 else float('inf')
    alpha_all1 = 1.0 / L.min() if L.min() > 0 else float('inf')
    print(f"\n  {name}:")
    print(f"    For max(L) ≥ 1: α ≥ {alpha_max1:.2f}")
    print(f"    For ALL L ≥ 1:  α ≥ {alpha_all1:.2f}")

# Conclusion
print(f"\n" + "=" * 60)
print(f"CONCLUSION")
print(f"=" * 60)
L_raw_max = B_var * eigs_raw.max()
L_M_max = B_var * eigs_M.max()

if eigs_M.max() > eigs_raw.max() * 1.5:
    print(f"M-weighted Gram has {eigs_M.max()/eigs_raw.max():.1f}x LARGER max eigenvalue")
    print(f"→ Using M⁻¹ inner product would HELP DC-WME")
elif eigs_M.max() < eigs_raw.max() / 1.5:
    print(f"M-weighted Gram has {eigs_raw.max()/eigs_M.max():.1f}x SMALLER max eigenvalue")
    print(f"→ M⁻¹ inner product would HURT DC-WME")
else:
    print(f"M-weighted and raw Gram have similar max eigenvalues (ratio: {eigs_M.max()/eigs_raw.max():.2f})")
    print(f"→ Mass matrix is NOT the root cause of small L_wme eigenvalues")

print(f"\nThe real issue: max L_wme = {L_raw_max:.4e}, need ≥ 1.0")
print(f"Required inflation: α ≥ {1.0/L_raw_max:.1f} (current cap: 3.0)")

# Save adjoint vectors for offline analysis (minimal addition)
try:
    A_np = np.zeros((adjoint_vectors[0].getSize(), n_obs))
    for i in range(n_obs):
        A_np[:, i] = adjoint_vectors[i].getArray()
    # Get DOF coords from cell centroids
    mesh = exp.problem.mesh
    nc = mesh.topology.index_map(mesh.topology.dim).size_local
    geom = mesh.geometry.x
    cc = np.zeros((nc, 2))
    for ci in range(nc):
        cv = mesh.geometry.dofmap[ci]
        cc[ci] = geom[cv, :2].mean(axis=0)
    dm = solver.V.dofmap
    dc = np.zeros((A_np.shape[0], 2))
    for ci in range(nc):
        for d in dm.cell_dofs(ci):
            dc[d] = cc[ci]
    np.savez("/tmp/adjoint_data.npz", A=A_np, dof_coords=dc, B_var=B_var)
    print(f"Saved adjoint data to /tmp/adjoint_data.npz")
except Exception as e:
    print(f"Warning: could not save adjoint data: {e}")

# Cleanup
for v in adjoint_vectors:
    v.destroy()
for v in adjoint_corrected:
    v.destroy()
ksp.destroy()
M_mat.destroy()

#!/usr/bin/env python3
"""
Analyze L_wme eigenvalues with spatially correlated B.
Uses pre-saved adjoint vectors from diagnose_mass_matrix.py.
Pure numpy — no PETSc/DOLFINx needed.
"""

import numpy as np
from scipy.spatial.distance import cdist

data = np.load("/tmp/adjoint_data.npz")
A = data['A']
dof_coords = data['dof_coords']
B_var = float(data['B_var'])
n_state, n_obs = A.shape

print(f"Loaded: A={A.shape}, n_obs={n_obs}, B_var={B_var:.6e}")

# Baseline: diagonal B
eigs_diag = np.linalg.eigvalsh(B_var * (A.T @ A))
print(f"\nBaseline (diagonal B): L_wme = [{eigs_diag.min():.4e}, {eigs_diag.max():.4e}], "
      f"#≥1: {np.sum(eigs_diag >= 1.0)}/{n_obs}")

# Distance matrix
print(f"\nComputing distance matrix ({n_state}x{n_state})...")
dist = cdist(dof_coords, dof_coords, metric='euclidean')
print(f"Max distance: {dist.max():.0f}m")

results = [("Diagonal (σ²I)", 0, eigs_diag)]

# Gaussian correlations
print("\n--- Gaussian correlations ---")
for L in [200, 500, 1000, 2000, 5000]:
    C = np.exp(-dist**2 / (2 * L**2))
    L_wme = B_var * (A.T @ (C @ A))
    L_wme = 0.5 * (L_wme + L_wme.T)
    eigs = np.linalg.eigvalsh(L_wme)
    amp = eigs.max() / eigs_diag.max()
    print(f"  L={L:>5}m: [{eigs.min():.4e}, {eigs.max():.4e}], "
          f"#≥1: {np.sum(eigs >= 1.0):>2}/{n_obs}, amp={amp:.1f}x")
    results.append((f"Gaussian L={L}m", L, eigs))

# Exponential correlations
print("\n--- Exponential correlations ---")
for L in [500, 1000, 2000]:
    C = np.exp(-dist / L)
    L_wme = B_var * (A.T @ (C @ A))
    L_wme = 0.5 * (L_wme + L_wme.T)
    eigs = np.linalg.eigvalsh(L_wme)
    amp = eigs.max() / eigs_diag.max()
    print(f"  L={L:>5}m: [{eigs.min():.4e}, {eigs.max():.4e}], "
          f"#≥1: {np.sum(eigs >= 1.0):>2}/{n_obs}, amp={amp:.1f}x")
    results.append((f"Exponential L={L}m", L, eigs))

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"\n{'B structure':<25} {'L_wme max':>12} {'#≥1':>5} {'Amplif':>8} {'α for max≥1':>12}")
print("-" * 70)
for name, L, eigs in results:
    amp = eigs.max() / eigs_diag.max()
    alpha = 1.0 / eigs.max() if eigs.max() > 0 else float('inf')
    print(f"{name:<25} {eigs.max():>12.4e} {np.sum(eigs >= 1.0):>5} {amp:>8.1f}x {alpha:>12.1f}")

best_name, _, best_eigs = max(results, key=lambda x: x[2].max())
alpha_best = 1.0 / best_eigs.max()
alpha_diag = 1.0 / eigs_diag.max()
print(f"\nBest: {best_name}")
print(f"  α needed: {alpha_best:.1f} (vs diagonal: {alpha_diag:.1f})")
print(f"  Reduction: {alpha_diag/alpha_best:.0f}x less inflation needed")

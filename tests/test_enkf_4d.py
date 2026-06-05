"""
Smoke tests for the stochastic 4D-EnKF baseline filter ``EnKF4D``.

Companion file to ``test_qpca_endcf.py`` — verifies shape, finiteness,
seed reproducibility, and that the optional localization argument
changes the result (so it's actually being used).
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_problem(seed: int = 0, n: int = 40, m: int = 20, N: int = 30, L: int = 3):
    rng = np.random.default_rng(seed)
    H = np.zeros((m, n))
    for i in range(m):
        H[i, 2 * i] = 1.0
    R = (0.1 ** 2) * np.eye(m)

    truth = rng.standard_normal(n)
    X_path = [truth[:, None] + 0.1 * rng.standard_normal((n, N)) for _ in range(L)]
    z_stack = np.concatenate(
        [H @ truth + 0.1 * rng.standard_normal(m) for _ in range(L)]
    )
    return H, R, X_path, z_stack


def test_update_shape_and_finiteness():
    from swe4dvar.data_assimilation.enkf_4d import EnKF4D

    H, R, X_path, z_stack = _make_problem(seed=1)
    f = EnKF4D(apply_H=lambda X: H @ X, R=R, window_len=len(X_path), seed=0)
    X_a = f.update(X_path, z_stack)
    assert X_a.shape == X_path[-1].shape
    assert np.all(np.isfinite(X_a))


def test_seed_reproducibility():
    """Two filters with the same seed must produce identical X_a."""
    from swe4dvar.data_assimilation.enkf_4d import EnKF4D

    H, R, X_path, z_stack = _make_problem(seed=2)
    f1 = EnKF4D(apply_H=lambda X: H @ X, R=R, window_len=len(X_path), seed=42)
    f2 = EnKF4D(apply_H=lambda X: H @ X, R=R, window_len=len(X_path), seed=42)
    X_a1 = f1.update(X_path, z_stack)
    X_a2 = f2.update(X_path, z_stack)
    np.testing.assert_allclose(X_a1, X_a2, rtol=1e-12, atol=1e-12)


def test_different_seeds_differ():
    """Different seeds → different perturbations → different X_a."""
    from swe4dvar.data_assimilation.enkf_4d import EnKF4D

    H, R, X_path, z_stack = _make_problem(seed=3)
    f1 = EnKF4D(apply_H=lambda X: H @ X, R=R, window_len=len(X_path), seed=1)
    f2 = EnKF4D(apply_H=lambda X: H @ X, R=R, window_len=len(X_path), seed=2)
    X_a1 = f1.update(X_path, z_stack)
    X_a2 = f2.update(X_path, z_stack)
    diff = np.linalg.norm(X_a1 - X_a2)
    assert diff > 1e-6, "different RNG seeds should give different analyses"


def test_localization_changes_result():
    """Passing a non-trivial GC taper should change the analysis."""
    from scipy.sparse import csr_matrix
    from swe4dvar.data_assimilation.enkf_4d import EnKF4D

    H, R, X_path, z_stack = _make_problem(seed=4, n=30, m=10, N=20, L=2)
    n, _ = X_path[-1].shape
    L = len(X_path)
    m = R.shape[0]
    # Build a band-limited taper: each state DOF i sees obs j with |i - 2j| < 4
    rows, cols, vals = [], [], []
    for i in range(n):
        for t in range(L):
            for p in range(m):
                if abs(i - 2 * p) < 4:
                    rows.append(i)
                    cols.append(t * m + p)
                    vals.append(max(0.0, 1.0 - abs(i - 2 * p) / 4.0))
    rho = csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
        shape=(n, m * L),
    )

    f1 = EnKF4D(apply_H=lambda X: H @ X, R=R, window_len=L, seed=7)
    f2 = EnKF4D(apply_H=lambda X: H @ X, R=R, window_len=L, seed=7)
    X_no_loc = f1.update(X_path, z_stack)
    X_loc = f2.update(X_path, z_stack, rho=rho)
    diff = np.linalg.norm(X_no_loc - X_loc) / np.linalg.norm(X_no_loc)
    assert diff > 1e-3, "localization should change the analysis meaningfully"


def test_update_rejects_wrong_window_length():
    from swe4dvar.data_assimilation.enkf_4d import EnKF4D

    H, R, X_path, z_stack = _make_problem(seed=5, L=3)
    f = EnKF4D(apply_H=lambda X: H @ X, R=R, window_len=4, seed=0)
    with pytest.raises(ValueError, match="window_len"):
        f.update(X_path, z_stack)


def test_update_rejects_single_member():
    from swe4dvar.data_assimilation.enkf_4d import EnKF4D

    H, R, X_path, z_stack = _make_problem(seed=6, N=1)
    f = EnKF4D(apply_H=lambda X: H @ X, R=R, window_len=len(X_path), seed=0)
    with pytest.raises(ValueError, match="ensemble"):
        f.update(X_path, z_stack)

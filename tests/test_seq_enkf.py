"""
Smoke tests for the sequential stochastic EnKF baseline filter ``SeqEnKF``.

Companion file to ``test_enkf_4d.py`` — verifies shape, finiteness,
seed reproducibility, and that the optional localization argument
changes the result (so it is actually being used).
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_problem(seed: int = 0, n: int = 40, m: int = 20, N: int = 30):
    rng = np.random.default_rng(seed)
    H = np.zeros((m, n))
    for i in range(m):
        H[i, 2 * i] = 1.0
    R = (0.1 ** 2) * np.eye(m)

    truth = rng.standard_normal(n)
    X = truth[:, None] + 0.1 * rng.standard_normal((n, N))
    z = H @ truth + 0.1 * rng.standard_normal(m)
    return H, R, X, z


def test_update_shape_and_finiteness():
    from swe4dvar.data_assimilation.seq_enkf import SeqEnKF

    H, R, X, z = _make_problem(seed=1)
    f = SeqEnKF(apply_H=lambda M: H @ M, R=R, seed=0)
    X_a = f.update(X, z)
    assert X_a.shape == X.shape
    assert np.all(np.isfinite(X_a))


def test_seed_reproducibility():
    """Two filters with the same seed must produce identical X_a."""
    from swe4dvar.data_assimilation.seq_enkf import SeqEnKF

    H, R, X, z = _make_problem(seed=2)
    f1 = SeqEnKF(apply_H=lambda M: H @ M, R=R, seed=42)
    f2 = SeqEnKF(apply_H=lambda M: H @ M, R=R, seed=42)
    X_a1 = f1.update(X, z)
    X_a2 = f2.update(X, z)
    np.testing.assert_allclose(X_a1, X_a2, rtol=1e-12, atol=1e-12)


def test_different_seeds_differ():
    """Different seeds → different perturbations → different X_a."""
    from swe4dvar.data_assimilation.seq_enkf import SeqEnKF

    H, R, X, z = _make_problem(seed=3)
    f1 = SeqEnKF(apply_H=lambda M: H @ M, R=R, seed=1)
    f2 = SeqEnKF(apply_H=lambda M: H @ M, R=R, seed=2)
    X_a1 = f1.update(X, z)
    X_a2 = f2.update(X, z)
    diff = np.linalg.norm(X_a1 - X_a2)
    assert diff > 1e-6, "different RNG seeds should give different analyses"


def test_localization_changes_result():
    """Passing a non-trivial GC taper should change the analysis."""
    from scipy.sparse import csr_matrix
    from swe4dvar.data_assimilation.seq_enkf import SeqEnKF

    H, R, X, z = _make_problem(seed=4, n=30, m=10, N=20)
    n = X.shape[0]
    m = R.shape[0]
    rows, cols, vals = [], [], []
    for i in range(n):
        for p in range(m):
            if abs(i - 2 * p) < 4:
                rows.append(i)
                cols.append(p)
                vals.append(max(0.0, 1.0 - abs(i - 2 * p) / 4.0))
    rho = csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
        shape=(n, m),
    )

    f1 = SeqEnKF(apply_H=lambda M: H @ M, R=R, seed=7)
    f2 = SeqEnKF(apply_H=lambda M: H @ M, R=R, seed=7)
    X_no_loc = f1.update(X, z)
    X_loc = f2.update(X, z, rho=rho)
    diff = np.linalg.norm(X_no_loc - X_loc) / np.linalg.norm(X_no_loc)
    assert diff > 1e-3, "localization should change the analysis meaningfully"


def test_update_rejects_wrong_z_size():
    from swe4dvar.data_assimilation.seq_enkf import SeqEnKF

    H, R, X, z = _make_problem(seed=5)
    f = SeqEnKF(apply_H=lambda M: H @ M, R=R, seed=0)
    bad_z = np.zeros(z.size + 1)
    with pytest.raises(ValueError, match="size"):
        f.update(X, bad_z)


def test_update_rejects_single_member():
    from swe4dvar.data_assimilation.seq_enkf import SeqEnKF

    H, R, X, z = _make_problem(seed=6, N=1)
    f = SeqEnKF(apply_H=lambda M: H @ M, R=R, seed=0)
    with pytest.raises(ValueError, match="ensemble"):
        f.update(X, z)

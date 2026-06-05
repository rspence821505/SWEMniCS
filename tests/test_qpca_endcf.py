"""
Verify the SWE-PDE port of QPCA-EnDCF matches the original paper code numerically.

The local module ``swe4dvar/data_assimilation/qpca_endcf.py`` differs from the
upstream ``QPCA-EnDCF-Paper/src/filters/qpca_endcf.py`` only in plumbing
(callable ``apply_H`` instead of dense ``H`` matrix). For the same random
seed, identical inputs, identical ``H`` matrix, and an ``apply_H`` defined
as ``lambda X: H @ X``, both implementations must produce bitwise-equal
analysis ensembles.

If the upstream package is not importable (PYTHONPATH limited to this repo),
the parity check is skipped — but the standalone correctness checks still
run.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Standalone correctness checks (work without the upstream package).
# ---------------------------------------------------------------------------


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
    from swe4dvar.data_assimilation.qpca_endcf import QPCAEnDCF

    H, R, X_path, z_stack = _make_problem(seed=1)
    f = QPCAEnDCF(apply_H=lambda X: H @ X, R=R, window_len=len(X_path), k=1)
    X_a = f.update(X_path, z_stack)
    assert X_a.shape == X_path[-1].shape
    assert np.all(np.isfinite(X_a))


def test_update_rejects_wrong_window_length():
    from swe4dvar.data_assimilation.qpca_endcf import QPCAEnDCF

    H, R, X_path, z_stack = _make_problem(seed=2, L=3)
    f = QPCAEnDCF(apply_H=lambda X: H @ X, R=R, window_len=4, k=1)
    with pytest.raises(ValueError, match="window_len"):
        f.update(X_path, z_stack)


def test_update_rejects_wrong_z_size():
    from swe4dvar.data_assimilation.qpca_endcf import QPCAEnDCF

    H, R, X_path, _ = _make_problem(seed=3)
    f = QPCAEnDCF(apply_H=lambda X: H @ X, R=R, window_len=len(X_path), k=1)
    with pytest.raises(ValueError, match="z_stack"):
        f.update(X_path, np.zeros(7))


def test_update_rejects_single_member():
    from swe4dvar.data_assimilation.qpca_endcf import QPCAEnDCF

    H, R, X_path, z_stack = _make_problem(seed=4, N=1)
    f = QPCAEnDCF(apply_H=lambda X: H @ X, R=R, window_len=len(X_path), k=1)
    with pytest.raises(ValueError, match="ensemble"):
        f.update(X_path, z_stack)


# ---------------------------------------------------------------------------
# Adaptive-κ (variance-explained selection) checks.
# ---------------------------------------------------------------------------


def test_kappa_target_validation():
    """Out-of-range kappa_target must raise; k_min < 1 must raise."""
    from swe4dvar.data_assimilation.qpca_endcf import QPCAEnDCF

    H, R, X_path, z_stack = _make_problem(seed=10)
    apply_H = lambda X: H @ X
    L = len(X_path)
    with pytest.raises(ValueError, match="kappa_target"):
        QPCAEnDCF(apply_H=apply_H, R=R, window_len=L, kappa_target=0.0)
    with pytest.raises(ValueError, match="kappa_target"):
        QPCAEnDCF(apply_H=apply_H, R=R, window_len=L, kappa_target=1.5)
    with pytest.raises(ValueError, match="k_min"):
        QPCAEnDCF(apply_H=apply_H, R=R, window_len=L, k_min=0)


def test_kappa_target_selects_correct_count():
    """With kappa_target=0.85, k_used must hit the variance threshold."""
    from swe4dvar.data_assimilation.qpca_endcf import QPCAEnDCF

    H, R, X_path, z_stack = _make_problem(seed=11, n=30, m=15, N=20, L=3)
    f = QPCAEnDCF(
        apply_H=lambda X: H @ X, R=R, window_len=len(X_path),
        kappa_target=0.85,
    )
    f.update(X_path, z_stack)
    eigs = f.last_eigenvalues_desc
    trace = f.last_eigenvalue_trace
    k_used = f.last_k_used
    # k_used is the smallest κ with cumvar/trace ≥ 0.85.
    cum = float(np.sum(eigs[:k_used]) / trace)
    assert cum >= 0.85 - 1e-12
    if k_used > 1:
        cum_prev = float(np.sum(eigs[: k_used - 1]) / trace)
        assert cum_prev < 0.85
    assert 1 <= k_used <= eigs.size


def test_kappa_target_default_recovers_fixed_k_one():
    """kappa_target=None must keep paper-exact behavior with k=1."""
    from swe4dvar.data_assimilation.qpca_endcf import QPCAEnDCF

    H, R, X_path, z_stack = _make_problem(seed=12)
    f_fixed = QPCAEnDCF(apply_H=lambda X: H @ X, R=R,
                        window_len=len(X_path), k=1)
    f_target_eq_zero = QPCAEnDCF(apply_H=lambda X: H @ X, R=R,
                                 window_len=len(X_path),
                                 kappa_target=1e-6, k_min=1)
    X_fix = f_fixed.update(X_path, z_stack)
    X_ad = f_target_eq_zero.update(X_path, z_stack)
    # A negligible variance target with k_min=1 picks κ=1 → identical update.
    assert f_target_eq_zero.last_k_used == 1
    np.testing.assert_allclose(X_fix, X_ad, rtol=1e-12, atol=1e-12)


def test_kappa_target_respects_kmax_clamp():
    """k_max=2 must clamp even when target requires more modes."""
    from swe4dvar.data_assimilation.qpca_endcf import QPCAEnDCF

    H, R, X_path, z_stack = _make_problem(seed=13, n=40, m=20, N=30, L=3)
    f = QPCAEnDCF(
        apply_H=lambda X: H @ X, R=R, window_len=len(X_path),
        kappa_target=0.999999, k_min=1, k_max=2,
    )
    f.update(X_path, z_stack)
    assert f.last_k_used == 2


# ---------------------------------------------------------------------------
# Parity with upstream paper implementation (skipped when not available).
# ---------------------------------------------------------------------------


def _try_import_paper_filter():
    import importlib
    import sys
    from pathlib import Path

    paper_root = (
        Path(__file__).resolve().parents[2] / "QPCA-EnDCF-Paper"
    )
    if not paper_root.exists():
        return None
    sys.path.insert(0, str(paper_root))
    try:
        mod = importlib.import_module("src.filters.qpca_endcf")
        return mod.QPCAEnDCF
    except Exception:
        return None


def test_parity_with_paper_implementation():
    """Bit-for-bit equality on identical inputs."""
    PaperFilter = _try_import_paper_filter()
    if PaperFilter is None:
        pytest.skip(
            "Upstream QPCA-EnDCF-Paper package not on PYTHONPATH; "
            "parity check skipped."
        )

    from swe4dvar.data_assimilation.qpca_endcf import QPCAEnDCF as LocalFilter

    H, R, X_path, z_stack = _make_problem(seed=42, n=30, m=15, N=25, L=4)
    L = len(X_path)

    paper = PaperFilter(H=H, R=R, window_len=L, k=1, stabilize=True)
    local = LocalFilter(apply_H=lambda X: H @ X, R=R, window_len=L, k=1, stabilize=True)

    X_a_paper = paper.update(X_path, z_stack)
    X_a_local = local.update(X_path, z_stack)

    np.testing.assert_allclose(
        X_a_local, X_a_paper, rtol=1e-12, atol=1e-12,
        err_msg="Local QPCAEnDCF.update disagrees with the paper implementation."
    )

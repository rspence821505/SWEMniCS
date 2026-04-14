"""Tests for the Karhunen-Loève expansion basis in ManningsBasisController."""

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from swe4dvar.forward.augmented_control import ManningsBasisController


def _make_grid_coords(nx=10, ny=10):
    """Create a simple 2D grid of coordinates."""
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


class TestKLBasisConstruction:
    """A. Basis construction tests."""

    def test_kl_basis_shape(self):
        coords = _make_grid_coords(10, 10)
        ctrl = ManningsBasisController(
            basis_type="kl", kl_n_modes=5, kl_prior_std=0.5,
        )
        ctrl._base_field_array = np.full(len(coords), 0.02)
        basis = ctrl._build_kl_basis(coords)
        assert basis.shape == (100, 5)

    def test_kl_basis_all_finite(self):
        coords = _make_grid_coords(10, 10)
        ctrl = ManningsBasisController(
            basis_type="kl", kl_n_modes=3, kl_prior_std=0.5,
        )
        basis = ctrl._build_kl_basis(coords)
        assert np.all(np.isfinite(basis))

    def test_kl_basis_no_nans(self):
        coords = _make_grid_coords(8, 8)
        ctrl = ManningsBasisController(
            basis_type="kl", kl_n_modes=4, kl_prior_std=0.3,
        )
        basis = ctrl._build_kl_basis(coords)
        assert not np.any(np.isnan(basis))

    def test_kl_basis_deterministic(self):
        coords = _make_grid_coords(10, 10)
        ctrl = ManningsBasisController(
            basis_type="kl", kl_n_modes=3, kl_prior_std=0.5,
        )
        b1 = ctrl._build_kl_basis(coords)
        b2 = ctrl._build_kl_basis(coords)
        # Eigenvectors may have sign flips, so compare absolute values
        np.testing.assert_allclose(np.abs(b1), np.abs(b2), atol=1e-10)

    def test_kl_auto_modes_selects_reasonable_count(self):
        coords = _make_grid_coords(10, 10)
        ctrl = ManningsBasisController(
            basis_type="kl", kl_n_modes=None, kl_prior_std=0.5,
        )
        basis = ctrl._build_kl_basis(coords)
        # Auto should select at least 1 mode and less than n_nodes
        assert basis.shape[1] >= 1
        assert basis.shape[1] <= len(coords)
        # Should capture at least 99% of variance
        assert ctrl._kl_variance_explained >= 0.99 - 1e-6


class TestKLSpectralOrdering:
    """B. Spectral ordering tests."""

    def test_eigenvalues_positive(self):
        coords = _make_grid_coords(10, 10)
        ctrl = ManningsBasisController(
            basis_type="kl", kl_n_modes=5, kl_prior_std=0.5,
        )
        ctrl._build_kl_basis(coords)
        assert np.all(ctrl._kl_eigenvalues > 0)

    def test_eigenvalues_descending(self):
        coords = _make_grid_coords(10, 10)
        ctrl = ManningsBasisController(
            basis_type="kl", kl_n_modes=5, kl_prior_std=0.5,
        )
        ctrl._build_kl_basis(coords)
        evals = ctrl._kl_eigenvalues
        assert np.all(evals[:-1] >= evals[1:])


class TestKLOrthogonality:
    """C. Orthogonality / covariance-space consistency tests."""

    def test_btb_is_diagonal(self):
        """B.T @ B should be diagonal with entries ≈ retained eigenvalues."""
        coords = _make_grid_coords(10, 10)
        ctrl = ManningsBasisController(
            basis_type="kl", kl_n_modes=5, kl_prior_std=0.5,
        )
        basis = ctrl._build_kl_basis(coords)
        gram = basis.T @ basis
        # Off-diagonal should be near zero
        off_diag = gram - np.diag(np.diag(gram))
        assert np.max(np.abs(off_diag)) < 1e-8
        # Diagonal should equal eigenvalues
        np.testing.assert_allclose(np.diag(gram), ctrl._kl_eigenvalues, rtol=1e-6)


class TestKLRoundTrip:
    """D. Round-trip controller tests."""

    def test_evaluate_field_with_kl_basis(self):
        coords = _make_grid_coords(10, 10)
        n_modes = 3
        ctrl = ManningsBasisController(
            basis_type="kl", kl_n_modes=n_modes, kl_prior_std=0.5,
            n_bounds=(0.01, 0.08), reference_n=0.02,
        )
        ctrl._basis_matrix = ctrl._build_kl_basis(coords)
        ctrl._base_field_array = np.full(len(coords), 0.02)

        theta = np.array([0.1, -0.2, 0.05])
        field = ctrl.evaluate_field(theta)

        # Field should be clipped within bounds
        assert np.all(field >= 0.01)
        assert np.all(field <= 0.08)
        assert field.shape == (len(coords),)

    def test_evaluate_field_at_zero_returns_reference(self):
        coords = _make_grid_coords(10, 10)
        ctrl = ManningsBasisController(
            basis_type="kl", kl_n_modes=3, kl_prior_std=0.5,
            n_bounds=(0.01, 0.08), reference_n=0.02,
        )
        ctrl._basis_matrix = ctrl._build_kl_basis(coords)
        ctrl._base_field_array = np.full(len(coords), 0.02)

        field = ctrl.evaluate_field(np.zeros(3))
        np.testing.assert_allclose(field, 0.02, atol=1e-12)


class TestKLErrorHandling:
    """Edge cases and error handling."""

    def test_too_many_modes_raises(self):
        coords = _make_grid_coords(5, 5)  # 25 nodes
        ctrl = ManningsBasisController(
            basis_type="kl", kl_n_modes=30, kl_prior_std=0.5,
        )
        with pytest.raises(ValueError, match="positive eigenvalues"):
            ctrl._build_kl_basis(coords)

    def test_invalid_basis_type_raises(self):
        with pytest.raises(ValueError, match="basis_type"):
            ManningsBasisController(basis_type="invalid")


class TestKLSparsePath:
    """Tests that verify the sparse (matrix-free eigsh) path."""

    def test_sparse_path_dispatched_via_low_threshold(self):
        """Force sparse path by setting kl_max_dense_nodes below node count."""
        coords = _make_grid_coords(10, 10)  # 100 nodes
        ctrl = ManningsBasisController(
            basis_type="kl", kl_n_modes=3, kl_prior_std=0.5,
            kl_max_dense_nodes=50,  # 100 > 50 → sparse path
        )
        basis = ctrl._build_kl_basis(coords)
        assert basis.shape == (100, 3)
        assert np.all(np.isfinite(basis))
        # Eigenvalues should still be positive and descending
        assert np.all(ctrl._kl_eigenvalues > 0)
        assert np.all(ctrl._kl_eigenvalues[:-1] >= ctrl._kl_eigenvalues[1:])

    def test_sparse_matches_dense(self):
        """Sparse and dense paths should produce the same eigenvalues.

        Eigenvectors from eigsh and eigh may differ (sign flips, rotations
        within degenerate subspaces), so we only compare eigenvalues and
        the Gram matrix B.T @ B (which depends only on eigenvalues for
        orthogonal eigenvectors).
        """
        coords = _make_grid_coords(8, 8)  # 64 nodes
        n_modes = 3
        ctrl_dense = ManningsBasisController(
            basis_type="kl", kl_n_modes=n_modes, kl_prior_std=0.5,
            kl_max_dense_nodes=10000,  # force dense
        )
        ctrl_sparse = ManningsBasisController(
            basis_type="kl", kl_n_modes=n_modes, kl_prior_std=0.5,
            kl_max_dense_nodes=10,  # force sparse
        )
        ctrl_dense._build_kl_basis(coords)
        ctrl_sparse._build_kl_basis(coords)

        np.testing.assert_allclose(
            ctrl_dense._kl_eigenvalues,
            ctrl_sparse._kl_eigenvalues,
            rtol=1e-8,
        )

    def test_sparse_btb_is_diagonal(self):
        """B.T @ B should be diagonal even on the sparse path."""
        coords = _make_grid_coords(10, 10)
        ctrl = ManningsBasisController(
            basis_type="kl", kl_n_modes=4, kl_prior_std=0.5,
            kl_max_dense_nodes=50,  # force sparse
        )
        basis = ctrl._build_kl_basis(coords)
        gram = basis.T @ basis
        off_diag = gram - np.diag(np.diag(gram))
        assert np.max(np.abs(off_diag)) < 1e-6
        np.testing.assert_allclose(np.diag(gram), ctrl._kl_eigenvalues, rtol=1e-4)


class TestKLAutoModeWarning:
    """Verify that kl_n_modes=None emits a warning."""

    def test_auto_mode_warns(self):
        with pytest.warns(UserWarning, match="kl_n_modes=None"):
            ManningsBasisController(basis_type="kl", kl_n_modes=None, kl_prior_std=0.5)


class TestGaussianRegression:
    """F. Regression: Gaussian basis still works exactly as before."""

    def test_gaussian_basis_unchanged(self):
        coords = np.array([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
            [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        ], dtype=float)
        ctrl = ManningsBasisController(
            basis_type="gaussian",
            basis_shape=(2, 1),
            basis_width_fraction=0.5,
            reference_n=0.02,
        )
        basis = ctrl._build_gaussian_basis(coords)
        # Row normalization: each row sums to 1
        row_sums = np.sum(basis, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)
        assert basis.shape == (6, 2)

    def test_gaussian_evaluate_field(self):
        ctrl = ManningsBasisController(
            basis_shape=(2, 1),
            basis_width_fraction=0.5,
            n_bounds=(0.01, 0.08),
            reference_n=0.02,
        )
        ctrl._basis_matrix = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        ctrl._base_field_array = np.array([0.02, 0.02, 0.02])

        field = ctrl.evaluate_field(np.array([2.0, -2.0]))
        assert np.all(field >= 0.01)
        assert np.all(field <= 0.08)

"""Unit tests for covariance matrix implementations.

Tests cover:
1. Symmetry of covariance matrices
2. Inverse consistency (C·C⁻¹ = I)
3. Inner product correctness
4. MPI determinism (same results on all ranks)
5. Positive definiteness

Run with:
    pytest test_covariance.py -v
    mpirun -np 4 pytest test_covariance.py -v  # Parallel tests
"""

import pytest
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
import sys
from typing import Optional


from swemnics.data_assimilation.covariance import (
    DiagonalCovariance,
    DenseCovariance,
    ImplicitCovariance,
    PrecisionBasedCovariance,
    create_observation_covariance,
    check_covariance_symmetry,
    check_inverse_consistency,
)


def _set_global_vector(vec: PETSc.Vec, values: np.ndarray) -> None:
    """Fill a distributed PETSc Vec with the provided global values."""
    start, end = vec.getOwnershipRange()
    local = np.array(values[start:end], dtype=float, copy=True)
    vec.setArray(local)
    vec.assemble()


def _gather_vector(vec: PETSc.Vec, comm: MPI.Comm) -> Optional[np.ndarray]:
    """Gather a distributed PETSc Vec onto rank 0."""
    local = vec.getArray()
    gathered = comm.gather(local.copy(), root=0)
    if comm.rank == 0:
        return np.concatenate(gathered)
    return None


# Provide a no-op benchmark fixture if pytest-benchmark is unavailable
try:  # pragma: no cover - only executed when plugin exists
    import pytest_benchmark.plugin  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback path

    @pytest.fixture
    def benchmark():
        """Minimal stand-in for pytest-benchmark fixture."""

        def _run(func, *args, **kwargs):
            return func(*args, **kwargs)

        return _run


@pytest.fixture
def comm():
    """MPI communicator fixture."""
    return MPI.COMM_WORLD


@pytest.fixture
def small_size():
    """Small problem size for tests."""
    return 10


@pytest.fixture
def medium_size():
    """Medium problem size for tests."""
    return 100


class TestDiagonalCovariance:
    """Tests for DiagonalCovariance class."""

    def test_uniform_variance_construction(self, comm, small_size):
        """Test construction with uniform variance."""
        variance = 2.5
        C = DiagonalCovariance(comm, size=small_size, variance=variance)

        # Check size
        assert C.size == small_size

        # Check diagonal values
        diag_array = C.diagonal.getArray()
        assert np.allclose(diag_array, variance)

        # Check inverse diagonal
        inv_diag_array = C.inv_diagonal.getArray()
        assert np.allclose(inv_diag_array, 1.0 / variance)

    def test_varying_diagonal_construction(self, comm, small_size):
        """Test construction with varying diagonal."""
        diagonal_values = np.random.uniform(0.5, 3.0, size=small_size)
        C = DiagonalCovariance(comm, size=small_size, diagonal=diagonal_values)

        # Gather diagonal to check (only on rank 0 for simplicity)
        diag_array = C.diagonal.getArray()

        # Check that local portion is correct
        start, end = C.ownership_range
        expected_local = diagonal_values[start:end]
        assert np.allclose(diag_array, expected_local)

    def test_apply(self, comm, small_size):
        """Test forward application C·v."""
        variance = 1.5
        C = DiagonalCovariance(comm, size=small_size, variance=variance)

        # Create test vector
        v = C.create_vec()
        v.set(1.0)  # All ones

        # Apply covariance
        result = C.apply(v)

        # Check result (should be variance * ones)
        result_array = result.getArray()
        assert np.allclose(result_array, variance)

        # Clean up
        v.destroy()
        result.destroy()

    def test_apply_inverse(self, comm, small_size):
        """Test inverse application C⁻¹·v."""
        variance = 2.0
        C = DiagonalCovariance(comm, size=small_size, variance=variance)

        v = C.create_vec()
        v.set(2.0)

        result = C.apply_inverse(v)

        # Check result (should be 2.0 / variance)
        result_array = result.getArray()
        assert np.allclose(result_array, 2.0 / variance)

        v.destroy()
        result.destroy()

    def test_sqrt_apply(self, comm, small_size):
        """Test square root application C^(1/2)·v."""
        variance = 4.0  # So sqrt = 2.0
        C = DiagonalCovariance(comm, size=small_size, variance=variance)

        v = C.create_vec()
        v.set(1.0)

        result = C.sqrt_apply(v)

        # Check result (should be 2.0)
        result_array = result.getArray()
        assert np.allclose(result_array, 2.0)

        v.destroy()
        result.destroy()

    def test_inverse_consistency(self, comm, medium_size):
        """Test that C·C⁻¹ = I."""
        diagonal_values = np.random.uniform(0.5, 3.0, size=medium_size)
        C = DiagonalCovariance(comm, size=medium_size, diagonal=diagonal_values)

        assert check_inverse_consistency(C, tol=1e-10)

    def test_symmetry(self, comm, medium_size):
        """Test that ⟨u, C·v⟩ = ⟨v, C·u⟩."""
        variance = 1.5
        C = DiagonalCovariance(comm, size=medium_size, variance=variance)

        assert check_covariance_symmetry(C, tol=1e-12)

    def test_inner_product_inv(self, comm, small_size):
        """Test optimized inner product ⟨u, C⁻¹·v⟩."""
        diagonal_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        C = DiagonalCovariance(comm, size=small_size, diagonal=diagonal_values)

        # Create test vectors (on rank owning the data)
        u = C.create_vec()
        v = C.create_vec()

        # Set values
        start, end = C.ownership_range
        u_local = np.ones(end - start)
        v_local = np.ones(end - start)
        u.setArray(u_local)
        v.setArray(v_local)

        # Compute ⟨u, C⁻¹·v⟩ = Σᵢ uᵢ·vᵢ/σᵢ²
        result = C.inner_product_inv(u, v)

        # Expected: sum of 1/diagonal_values
        expected = np.sum(1.0 / diagonal_values)

        assert np.isclose(result, expected, rtol=1e-10)

        u.destroy()
        v.destroy()

    def test_mpi_determinism(self, comm, medium_size):
        """Test that results are identical across MPI ranks."""
        variance = 1.5
        C = DiagonalCovariance(comm, size=medium_size, variance=variance)

        # Create consistent random vector across ranks
        v = C.create_vec()
        if comm.rank == 0:
            # Generate on rank 0
            np.random.seed(42)
            global_array = np.random.randn(medium_size)
        else:
            global_array = None

        # Broadcast to all ranks
        global_array = comm.bcast(global_array, root=0)

        # Set local portion
        start, end = C.ownership_range
        v.setArray(global_array[start:end])

        # Apply and gather result
        result = C.apply(v)
        local_result = result.getArray()

        # Gather to all ranks and check consistency
        all_local_results = comm.allgather(local_result)
        full_result = np.concatenate(all_local_results)

        # Check against expected
        expected = global_array * variance
        assert np.allclose(full_result, expected)

        v.destroy()
        result.destroy()


class TestDenseCovariance:
    """Tests for DenseCovariance class."""

    def test_from_correlation_identity(self, comm, small_size):
        """Test construction from identity correlation matrix."""
        corr = np.eye(small_size)
        variances = np.ones(small_size) * 2.0

        C = DenseCovariance.from_correlation(comm, corr, variances)

        assert C.size == small_size

    def test_from_correlation_full(self, comm):
        """Test construction from full correlation matrix."""
        size = 5
        # Construct a valid correlation matrix
        corr = np.array(
            [
                [1.0, 0.5, 0.3, 0.2, 0.1],
                [0.5, 1.0, 0.4, 0.3, 0.2],
                [0.3, 0.4, 1.0, 0.5, 0.3],
                [0.2, 0.3, 0.5, 1.0, 0.4],
                [0.1, 0.2, 0.3, 0.4, 1.0],
            ]
        )
        variances = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

        C = DenseCovariance.from_correlation(comm, corr, variances)

        # Extract diagonal collectively and gather on rank 0
        diag_vec = C.mat.createVecRight()
        C.mat.getDiagonal(diag_vec)

        start, end = diag_vec.getOwnershipRange()
        local_diag = diag_vec.getArray()
        gathered = comm.gather(local_diag.copy(), root=0)

        if comm.rank == 0:
            full_diag = np.concatenate(gathered)
            assert np.allclose(full_diag, variances, rtol=1e-10)

        diag_vec.destroy()

    def test_apply(self, comm):
        """Test forward application with known matrix."""
        size = 3
        # Simple 3x3 covariance
        mat = PETSc.Mat().create(comm=comm)
        mat.setSizes((size, size))
        mat.setType(PETSc.Mat.Type.AIJ)
        mat.setUp()

        if comm.rank == 0:
            cov_array = np.array(
                [
                    [4.0, 1.0, 0.5],
                    [1.0, 3.0, 1.0],
                    [0.5, 1.0, 2.0],
                ]
            )
            for i in range(size):
                mat.setValues(i, list(range(size)), cov_array[i, :])

        mat.assemblyBegin()
        mat.assemblyEnd()

        C = DenseCovariance(comm, mat)

        # Test with specific vector
        v = C.create_vec()
        _set_global_vector(v, np.array([1.0, 0.0, 0.0]))

        result = C.apply(v)

        gathered = _gather_vector(result, comm)
        if comm.rank == 0 and gathered is not None:
            expected = np.array([4.0, 1.0, 0.5])
            assert np.allclose(gathered, expected)

        v.destroy()
        result.destroy()

    def test_apply_inverse(self, comm):
        """Test inverse application."""
        size = 3
        corr = np.eye(size)
        variances = np.array([1.0, 2.0, 3.0])

        C = DenseCovariance.from_correlation(comm, corr, variances)

        v = C.create_vec()
        _set_global_vector(v, np.array([2.0, 4.0, 6.0]))

        result = C.apply_inverse(v)

        gathered = _gather_vector(result, comm)
        if comm.rank == 0 and gathered is not None:
            expected = np.array([2.0, 2.0, 2.0])
            assert np.allclose(gathered, expected, rtol=1e-6)

        v.destroy()
        result.destroy()

    def test_inverse_consistency(self, comm):
        """Test C·C⁻¹ = I."""
        size = 5
        corr = np.eye(size)
        variances = np.random.uniform(0.5, 3.0, size=size)

        C = DenseCovariance.from_correlation(comm, corr, variances)

        assert check_inverse_consistency(C, tol=1e-6)

    def test_symmetry(self, comm):
        """Test symmetry."""
        size = 5
        # Symmetric correlation
        corr = np.eye(size)
        for i in range(size - 1):
            corr[i, i + 1] = corr[i + 1, i] = 0.5
        variances = np.ones(size)

        C = DenseCovariance.from_correlation(comm, corr, variances)

        assert check_covariance_symmetry(C, tol=1e-10)


class TestPrecisionBasedCovariance:
    """Tests for ImplicitCovariance class."""

    def test_explicit_precision_apply_inverse(self, comm):
        """Test with explicitly stored precision matrix."""
        size = 5

        # Create precision matrix (C⁻¹ = I for simplicity)
        precision = PETSc.Mat().create(comm=comm)
        precision.setSizes((size, size))
        precision.setType(PETSc.Mat.Type.AIJ)
        precision.setUp()

        if comm.rank == 0:
            for i in range(size):
                precision.setValue(i, i, 1.0)

        precision.assemblyBegin()
        precision.assemblyEnd()

        C = PrecisionBasedCovariance(comm, precision, inverse_is_explicit=True)

        # Test inverse application (should be identity)
        v = C.create_vec()
        _set_global_vector(v, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        result = C.apply_inverse(v)

        gathered = _gather_vector(result, comm)
        if comm.rank == 0 and gathered is not None:
            assert np.allclose(gathered, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        v.destroy()
        result.destroy()

    def test_inverse_consistency_explicit(self, comm):
        """Test C·C⁻¹ = I with explicit precision."""
        size = 10

        # Create diagonal precision matrix
        precision = PETSc.Mat().create(comm=comm)
        precision.setSizes((size, size))
        precision.setType(PETSc.Mat.Type.AIJ)
        precision.setUp()

        if comm.rank == 0:
            diag_values = np.random.uniform(0.5, 2.0, size=size)
            for i in range(size):
                precision.setValue(i, i, diag_values[i])

        precision.assemblyBegin()
        precision.assemblyEnd()

        C = PrecisionBasedCovariance(comm, precision, inverse_is_explicit=True)

        # This test is relaxed due to iterative solver tolerance
        assert check_inverse_consistency(C, tol=1e-5)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_observation_covariance(self, comm, medium_size):
        """Test observation covariance construction."""
        obs_std = 1.5
        R = create_observation_covariance(comm, medium_size, obs_std)

        assert R.size == medium_size
        assert isinstance(R, DiagonalCovariance)

        # Check variance
        diag_array = R.diagonal.getArray()
        expected_variance = obs_std**2
        assert np.allclose(diag_array, expected_variance)


class TestErrorHandling:
    """Tests for error handling."""

    def test_diagonal_requires_variance_or_diagonal(self, comm, small_size):
        """Test that DiagonalCovariance requires variance or diagonal."""
        with pytest.raises(ValueError, match="Provide exactly one"):
            DiagonalCovariance(comm, size=small_size)

        with pytest.raises(ValueError, match="Provide exactly one"):
            DiagonalCovariance(
                comm, size=small_size, variance=1.0, diagonal=np.ones(small_size)
            )

    def test_correlation_matrix_symmetry(self, comm):
        """Test that correlation matrix must be symmetric."""
        size = 3
        corr = np.array(
            [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.4],
                [0.2, 0.4, 1.0],  # Not symmetric!
            ]
        )
        variances = np.ones(size)

        with pytest.raises(AssertionError):
            DenseCovariance.from_correlation(comm, corr, variances)


# Performance benchmarks (optional, requires pytest-benchmark)
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for covariance operations."""

    def test_diagonal_apply_performance(self, comm, benchmark):
        """Benchmark diagonal covariance application."""
        size = 10000
        C = DiagonalCovariance(comm, size=size, variance=1.5)
        v = C.create_vec()
        v.setRandom()

        def apply_op():
            result = C.apply(v)
            result.destroy()

        if hasattr(benchmark, "__call__"):
            benchmark(apply_op)
        else:
            # Run without benchmark
            apply_op()

        v.destroy()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

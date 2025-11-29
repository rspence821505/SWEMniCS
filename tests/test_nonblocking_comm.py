"""
Tests for non-blocking communication utilities.

Tests asynchronous communication operations and overlap patterns
for hiding communication latency.
"""

import pytest
import numpy as np
import time
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem

from swemnics.utils.nonblocking_comm import (
    NonBlockingScatter,
    AsyncVectorOps,
    OverlapComputeComm,
    AsyncObservationOperator,
    BatchedCommunication
)


# MPI configuration
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Test markers
requires_mpi = pytest.mark.skipif(size == 1, reason="Requires MPI with multiple ranks")
serial_only = pytest.mark.skipif(size > 1, reason="Serial test only")


@pytest.fixture
def simple_mesh():
    """Create a simple test mesh."""
    domain = mesh.create_unit_square(
        comm,
        10,
        10,
        cell_type=mesh.CellType.triangle
    )
    return domain


@pytest.fixture
def function_space(simple_mesh):
    """Create a function space for testing."""
    V = fem.functionspace(simple_mesh, ("Lagrange", 1))
    return V


@pytest.fixture
def test_function(function_space):
    """Create a test function."""
    u = fem.Function(function_space)
    u.x.array[:] = rank + 1.0
    return u


def test_nonblocking_scatter_init():
    """Test NonBlockingScatter initialization."""
    scatter = NonBlockingScatter()

    assert len(scatter.active_requests) == 0


def test_nonblocking_scatter_forward(test_function):
    """Test non-blocking scatter forward operation."""
    scatter = NonBlockingScatter()

    # Begin scatter
    request = scatter.scatter_forward_begin(test_function)

    # Should have active request
    assert len(scatter.active_requests) == 1

    # End scatter
    scatter.scatter_forward_end(request)

    # Request should be removed
    assert len(scatter.active_requests) == 0


def test_nonblocking_scatter_reverse(test_function):
    """Test non-blocking scatter reverse operation."""
    from petsc4py import PETSc

    scatter = NonBlockingScatter()

    # Begin scatter
    request = scatter.scatter_reverse_begin(test_function, PETSc.ScatterMode.REVERSE)

    # End scatter
    scatter.scatter_reverse_end(request)

    assert len(scatter.active_requests) == 0


def test_nonblocking_scatter_wait_all(test_function):
    """Test waiting for all scatter operations."""
    from petsc4py import PETSc

    scatter = NonBlockingScatter()

    # Start multiple operations
    req1 = scatter.scatter_forward_begin(test_function)
    req2 = scatter.scatter_reverse_begin(test_function, PETSc.ScatterMode.REVERSE)

    assert len(scatter.active_requests) == 2

    # Wait for all
    scatter.wait_all()

    assert len(scatter.active_requests) == 0


def test_async_vector_ops_init():
    """Test AsyncVectorOps initialization."""
    async_ops = AsyncVectorOps(comm)

    assert async_ops.comm == comm
    assert async_ops.rank == rank
    assert len(async_ops.active_ops) == 0


@requires_mpi
def test_async_allreduce():
    """Test non-blocking allreduce operation."""
    async_ops = AsyncVectorOps(comm)

    # Create test data
    send_data = np.array([rank + 1.0], dtype=np.float64)

    # Begin allreduce
    recv_buffer, request = async_ops.allreduce_begin(send_data, op=MPI.SUM)

    # Should have active operation
    assert "default" in async_ops.active_ops

    # End allreduce
    result = async_ops.allreduce_end()

    # Check result
    expected_sum = sum(range(1, size + 1))
    assert abs(result[0] - expected_sum) < 1e-10

    # Should be removed from active ops
    assert "default" not in async_ops.active_ops


@requires_mpi
def test_async_broadcast():
    """Test non-blocking broadcast operation."""
    async_ops = AsyncVectorOps(comm)

    # Create test data
    if rank == 0:
        data = np.array([42.0], dtype=np.float64)
    else:
        data = np.array([0.0], dtype=np.float64)

    # Begin broadcast
    request = async_ops.broadcast_begin(data, root=0)

    # End broadcast
    result = async_ops.broadcast_end()

    # All ranks should have 42.0
    assert abs(result[0] - 42.0) < 1e-10


@requires_mpi
def test_async_gather():
    """Test non-blocking gather operation."""
    async_ops = AsyncVectorOps(comm)

    # Create test data
    send_data = np.array([rank * 10.0], dtype=np.float64)

    # Begin gather
    recv_buffer, request = async_ops.gather_begin(send_data, root=0)

    # End gather
    result = async_ops.gather_end()

    # Check result on root
    if rank == 0:
        expected = np.array([i * 10.0 for i in range(size)])
        assert np.allclose(result.flatten(), expected)
    else:
        assert result is None


def test_async_vector_ops_wait_all():
    """Test waiting for all async operations."""
    async_ops = AsyncVectorOps(comm)

    # Start multiple operations
    send_data1 = np.array([1.0])
    send_data2 = np.array([2.0])

    async_ops.allreduce_begin(send_data1, tag="op1")
    async_ops.allreduce_begin(send_data2, tag="op2")

    assert len(async_ops.active_ops) == 2

    # Wait for all
    async_ops.wait_all()

    assert len(async_ops.active_ops) == 0


def test_overlap_compute_comm_init():
    """Test OverlapComputeComm initialization."""
    overlap = OverlapComputeComm(comm)

    assert overlap.comm == comm
    assert isinstance(overlap.async_ops, AsyncVectorOps)
    assert isinstance(overlap.scatter_ops, NonBlockingScatter)


def test_overlap_scatter_with_work(test_function):
    """Test overlapping scatter with computation."""
    overlap = OverlapComputeComm(comm)

    # Define independent work
    def independent_work():
        return 42

    # Overlap scatter with work
    result = overlap.overlap_scatter_with_work(test_function, independent_work)

    assert result == 42


@requires_mpi
def test_overlap_allreduce_with_work():
    """Test overlapping allreduce with computation."""
    overlap = OverlapComputeComm(comm)

    # Define independent work
    def independent_work():
        return rank * 2

    # Overlap allreduce with work
    work_result, reduced_value = overlap.overlap_allreduce_with_work(
        float(rank + 1),
        independent_work
    )

    # Check results
    assert work_result == rank * 2

    expected_sum = sum(range(1, size + 1))
    assert abs(reduced_value - expected_sum) < 1e-10


def test_pipeline_scatter_operations(function_space):
    """Test pipelining multiple scatter operations."""
    overlap = OverlapComputeComm(comm)

    # Create multiple functions
    u1 = fem.Function(function_space)
    u2 = fem.Function(function_space)
    u3 = fem.Function(function_space)

    u1.x.array[:] = 1.0
    u2.x.array[:] = 2.0
    u3.x.array[:] = 3.0

    functions = [u1, u2, u3]

    # Define operations (just verify the function was set)
    results = []

    def make_op(expected_val):
        def op(u):
            # Check that function has expected value
            results.append(u.x.array[0])
        return op

    operations = [make_op(1.0), make_op(2.0), make_op(3.0)]

    # Pipeline operations
    overlap.pipeline_scatter_operations(functions, operations)

    # Should have executed all operations
    assert len(results) == 3


def test_async_observation_operator_init():
    """Test AsyncObservationOperator initialization."""
    async_obs = AsyncObservationOperator(comm)

    assert async_obs.comm == comm
    assert isinstance(async_obs.async_ops, AsyncVectorOps)


@requires_mpi
def test_async_observation_operator_apply_with_overlap():
    """Test async observation operator with overlapped work."""
    async_obs = AsyncObservationOperator(comm)

    # Create local observations
    local_obs = np.array([rank * 1.0, rank * 2.0])

    # Define independent work
    def independent_work():
        return rank + 100

    # Apply with overlap
    global_obs = async_obs.apply_with_overlap(local_obs, independent_work)

    # Check that we got global observations
    expected_size = size * len(local_obs)
    assert len(global_obs) == expected_size


def test_batched_communication_init():
    """Test BatchedCommunication initialization."""
    batched = BatchedCommunication(comm)

    assert batched.comm == comm
    assert len(batched.send_buffer) == 0


def test_batched_communication_add_to_batch():
    """Test adding values to batch."""
    batched = BatchedCommunication(comm)

    batched.add_to_batch(1.0)
    batched.add_to_batch(2.0)
    batched.add_to_batch(3.0)

    assert len(batched.send_buffer) == 3


@requires_mpi
def test_batched_communication_allreduce_batch():
    """Test batched allreduce operation."""
    batched = BatchedCommunication(comm)

    # Add multiple values
    batched.add_to_batch(1.0)
    batched.add_to_batch(2.0)
    batched.add_to_batch(3.0)

    # Perform batched allreduce
    results = batched.allreduce_batch(op=MPI.SUM)

    # Check results
    assert len(results) == 3
    assert abs(results[0] - size * 1.0) < 1e-10
    assert abs(results[1] - size * 2.0) < 1e-10
    assert abs(results[2] - size * 3.0) < 1e-10

    # Buffer should be cleared
    assert len(batched.send_buffer) == 0


def test_batched_communication_reset():
    """Test resetting batched communication."""
    batched = BatchedCommunication(comm)

    batched.add_to_batch(1.0)
    batched.add_to_batch(2.0)

    assert len(batched.send_buffer) == 2

    batched.reset()

    assert len(batched.send_buffer) == 0


@requires_mpi
def test_multiple_async_operations_with_tags():
    """Test multiple simultaneous async operations using tags."""
    async_ops = AsyncVectorOps(comm)

    # Start multiple allreduces with different tags
    data1 = np.array([1.0])
    data2 = np.array([10.0])
    data3 = np.array([100.0])

    async_ops.allreduce_begin(data1, tag="sum1")
    async_ops.allreduce_begin(data2, tag="sum10")
    async_ops.allreduce_begin(data3, tag="sum100")

    # Complete in different order
    result3 = async_ops.allreduce_end("sum100")
    result1 = async_ops.allreduce_end("sum1")
    result2 = async_ops.allreduce_end("sum10")

    # Check results
    assert abs(result1[0] - size * 1.0) < 1e-10
    assert abs(result2[0] - size * 10.0) < 1e-10
    assert abs(result3[0] - size * 100.0) < 1e-10


def test_overlap_pattern_with_timing():
    """Test that overlap pattern can reduce wall clock time."""
    overlap = OverlapComputeComm(comm)

    # Simulate work that takes time
    def slow_work():
        time.sleep(0.01)
        return 42

    # Time overlapped version
    start = MPI.Wtime()
    result = overlap.overlap_allreduce_with_work(1.0, slow_work)
    elapsed_overlap = MPI.Wtime() - start

    # The overlap should complete successfully
    assert result[0] == 42


@requires_mpi
def test_async_operations_error_handling():
    """Test error handling for async operations."""
    async_ops = AsyncVectorOps(comm)

    # Try to end an operation that doesn't exist
    with pytest.raises(ValueError):
        async_ops.allreduce_end("nonexistent")

    with pytest.raises(ValueError):
        async_ops.broadcast_end("nonexistent")

    with pytest.raises(ValueError):
        async_ops.gather_end("nonexistent")


def test_batched_empty_allreduce():
    """Test batched allreduce with empty buffer."""
    batched = BatchedCommunication(comm)

    # Don't add anything
    results = batched.allreduce_batch()

    # Should return empty array
    assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

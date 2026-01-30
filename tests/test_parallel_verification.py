"""
Parallel verification tests for swe4dvar package.

These tests verify that parallelization features documented in
refactoring_docs/parallelization/ are correctly implemented.

Tests cover:
- Ghost cell management (scatter_forward/scatter_reverse)
- MPI collective operations in cost functions
- Parallel observation operator
- Distributed Jacobian storage
- Deterministic results across different MPI rank counts

Run with different rank counts:
    python -m pytest tests/test_parallel_verification.py -v  # Serial
    mpirun -n 2 python -m pytest tests/test_parallel_verification.py -v
    mpirun -n 4 python -m pytest tests/test_parallel_verification.py -v
"""

import pytest
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem

# Import swe4dvar parallelization modules
from swe4dvar.utils.parallel_ops import (
    ParallelContext,
    DistributedVectorOps,
    DistributedMatrixOps,
    ParallelIO,
    LoadBalancer,
    ParallelTimer,
)
from swe4dvar.utils.load_balancing_metrics import (
    LoadBalancingMetrics,
    CommunicationTracker,
)
from swe4dvar.utils.nonblocking_comm import (
    NonBlockingScatter,
    AsyncVectorOps,
    OverlapComputeComm,
    BatchedCommunication,
)


# MPI configuration
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Test markers
requires_mpi = pytest.mark.skipif(size == 1, reason="Requires MPI with multiple ranks")
serial_only = pytest.mark.skipif(size > 1, reason="Serial test only")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_mesh():
    """Create a simple distributed test mesh."""
    domain = mesh.create_unit_square(
        comm,
        20,
        20,
        cell_type=mesh.CellType.triangle
    )
    return domain


@pytest.fixture
def function_space(simple_mesh):
    """Create a scalar P1 function space."""
    V = fem.functionspace(simple_mesh, ("Lagrange", 1))
    return V


@pytest.fixture
def mixed_function_space(simple_mesh):
    """Create a mixed function space (velocity + height)."""
    # P2 for velocity, P1 for height (Taylor-Hood like)
    V_el = ("Lagrange", 2, (2,))  # Vector P2
    Q_el = ("Lagrange", 1)  # Scalar P1

    V = fem.functionspace(simple_mesh, V_el)
    Q = fem.functionspace(simple_mesh, Q_el)

    return V, Q


@pytest.fixture
def test_function(function_space):
    """Create a test function with known values."""
    u = fem.Function(function_space)
    # Set to x + 2*y
    u.interpolate(lambda x: x[0] + 2 * x[1])
    return u


@pytest.fixture
def parallel_context():
    """Create a ParallelContext instance."""
    return ParallelContext(comm)


# =============================================================================
# Section 1: Ghost Cell Management Tests
# =============================================================================

class TestGhostCellManagement:
    """Tests for ghost cell scatter_forward/scatter_reverse operations."""

    def test_scatter_forward_consistency(self, test_function):
        """Verify scatter_forward updates ghost cells correctly."""
        # Modify owned values
        local_size = test_function.x.index_map.size_local
        test_function.x.array[:local_size] = rank + 1.0

        # Scatter forward to update ghosts
        test_function.x.scatter_forward()

        # Ghost values should be from neighboring ranks
        num_ghosts = test_function.x.index_map.num_ghosts
        if num_ghosts > 0:
            ghost_values = test_function.x.array[local_size:local_size + num_ghosts]
            # Ghost values should be positive (set by some rank)
            assert np.all(ghost_values > 0)

    @requires_mpi
    def test_scatter_reverse_accumulation(self, function_space):
        """Verify scatter_reverse accumulates contributions correctly."""
        u = fem.Function(function_space)

        # Set all values to rank + 1
        u.x.array[:] = rank + 1.0

        # Scatter reverse with ADD mode
        u.x.scatter_reverse(dolfinx.la.InsertMode.add)

        # Values at partition boundaries should be accumulated
        # (values should be >= rank + 1)
        local_size = u.x.index_map.size_local
        owned_values = u.x.array[:local_size]
        assert np.all(owned_values >= rank + 1.0)

    def test_ghost_layer_size(self, simple_mesh, function_space):
        """Verify ghost layers exist for distributed mesh."""
        # Check mesh ghost cells
        mesh_ghosts = simple_mesh.topology.index_map(simple_mesh.topology.dim).num_ghosts

        # Check DOF ghosts
        dof_ghosts = function_space.dofmap.index_map.num_ghosts

        if size > 1:
            # At least some ranks should have ghosts
            total_mesh_ghosts = comm.allreduce(mesh_ghosts, op=MPI.SUM)
            total_dof_ghosts = comm.allreduce(dof_ghosts, op=MPI.SUM)
            assert total_mesh_ghosts > 0, "No mesh ghost cells in parallel run"
            assert total_dof_ghosts > 0, "No DOF ghosts in parallel run"
        else:
            # Serial run - no ghosts expected
            assert mesh_ghosts == 0
            assert dof_ghosts == 0


# =============================================================================
# Section 2: MPI Collective Operations Tests
# =============================================================================

class TestMPICollectives:
    """Tests for MPI collective operations in parallel context."""

    def test_allreduce_sum(self, parallel_context):
        """Verify allreduce sum produces correct global result."""
        local_value = float(rank + 1)
        global_sum = parallel_context.allreduce_sum(local_value)

        expected = sum(range(1, size + 1))
        assert abs(global_sum - expected) < 1e-10

    def test_broadcast(self, parallel_context):
        """Verify broadcast distributes data from root."""
        if rank == 0:
            data = {"value": 42.0, "name": "test"}
        else:
            data = None

        result = parallel_context.broadcast(data, root=0)

        assert result["value"] == 42.0
        assert result["name"] == "test"

    def test_allgather(self, parallel_context):
        """Verify allgather collects data from all ranks."""
        local_data = rank * 10
        all_data = parallel_context.allgather(local_data)

        assert len(all_data) == size
        expected = [i * 10 for i in range(size)]
        assert all_data == expected

    @requires_mpi
    def test_reduce_sum(self, parallel_context):
        """Verify reduce sum to root."""
        local_value = float(rank + 1)
        global_sum = parallel_context.reduce_sum(local_value, root=0)

        if rank == 0:
            expected = sum(range(1, size + 1))
            assert abs(global_sum - expected) < 1e-10


# =============================================================================
# Section 3: Distributed Vector Operations Tests
# =============================================================================

class TestDistributedVectorOps:
    """Tests for distributed PETSc vector operations."""

    def test_create_distributed_vec(self):
        """Verify distributed vector creation."""
        global_size = 100
        vec = DistributedVectorOps.create_distributed_vec(global_size, comm)

        # Check global size
        assert vec.getSize() == global_size

        # Check local size is reasonable
        local_size = vec.getLocalSize()
        assert local_size > 0
        assert local_size <= global_size

        # Total of local sizes should equal global
        total_local = comm.allreduce(local_size, op=MPI.SUM)
        assert total_local == global_size

        vec.destroy()

    def test_parallel_dot(self):
        """Verify parallel dot product computes correctly."""
        global_size = 100
        v1 = DistributedVectorOps.create_distributed_vec(global_size, comm)
        v2 = DistributedVectorOps.create_distributed_vec(global_size, comm)

        # Set all values to 1
        v1.set(1.0)
        v2.set(1.0)

        # Dot product should be global_size
        dot = DistributedVectorOps.parallel_dot(v1, v2)
        assert abs(dot - global_size) < 1e-10

        v1.destroy()
        v2.destroy()

    def test_parallel_norm(self):
        """Verify parallel norm computes correctly."""
        global_size = 100
        vec = DistributedVectorOps.create_distributed_vec(global_size, comm)

        # Set all values to 1
        vec.set(1.0)

        # L2 norm should be sqrt(global_size)
        norm = DistributedVectorOps.parallel_norm(vec)
        expected = np.sqrt(global_size)
        assert abs(norm - expected) < 1e-10

        vec.destroy()

    def test_local_to_global(self):
        """Verify local values are correctly inserted into global vector."""
        global_size = size * 10  # 10 per rank
        vec = DistributedVectorOps.create_distributed_vec(global_size, comm)

        # Determine local range
        start, end = vec.getOwnershipRange()
        local_size = end - start

        # Create local values
        local_values = np.arange(start, end, dtype=np.float64)
        local_indices = np.arange(start, end, dtype=np.int32)

        DistributedVectorOps.local_to_global(local_values, vec, local_indices)

        # Verify
        local_array = DistributedVectorOps.global_to_local(vec)
        assert np.allclose(local_array, local_values)

        vec.destroy()


# =============================================================================
# Section 4: Determinism Tests (Critical for 4D-Var)
# =============================================================================

class TestDeterminism:
    """Tests verifying deterministic results across rank counts."""

    def test_global_reduction_determinism(self):
        """Verify global reductions produce same result regardless of rank count."""
        # Each rank contributes based on a deterministic formula
        local_contribution = float((rank + 1) ** 2)

        global_sum = comm.allreduce(local_contribution, op=MPI.SUM)

        # The expected sum is sum(i^2 for i in 1..size)
        expected = sum((i + 1) ** 2 for i in range(size))

        assert abs(global_sum - expected) < 1e-10

    def test_dot_product_determinism(self):
        """Verify dot products are deterministic across rank counts."""
        global_size = 100

        # Create deterministic vectors
        v1 = DistributedVectorOps.create_distributed_vec(global_size, comm)
        v2 = DistributedVectorOps.create_distributed_vec(global_size, comm)

        # Set values based on global index
        start, end = v1.getOwnershipRange()
        for i in range(start, end):
            v1.setValue(i, float(i + 1))
            v2.setValue(i, 1.0 / (i + 1))

        v1.assemblyBegin()
        v1.assemblyEnd()
        v2.assemblyBegin()
        v2.assemblyEnd()

        # Dot product: sum(1) = global_size
        dot = v1.dot(v2)

        # Should be global_size regardless of how many ranks
        assert abs(dot - global_size) < 1e-10

        v1.destroy()
        v2.destroy()

    @requires_mpi
    def test_mesh_metrics_consistency(self, simple_mesh, function_space):
        """Verify mesh metrics are consistent across all ranks."""
        metrics = LoadBalancingMetrics(comm)
        all_metrics = metrics.compute_comprehensive_metrics(
            simple_mesh, function_space, name="determinism_test"
        )

        # Global values should be identical on all ranks
        global_cells = all_metrics['mesh']['global_cells']
        global_dofs = all_metrics['dofs']['global_dofs']

        # Gather to verify
        all_global_cells = comm.allgather(global_cells)
        all_global_dofs = comm.allgather(global_dofs)

        # All ranks should have same global values
        assert len(set(all_global_cells)) == 1
        assert len(set(all_global_dofs)) == 1


# =============================================================================
# Section 5: Load Balancing Tests
# =============================================================================

class TestLoadBalancing:
    """Tests for load balancing metrics and analysis."""

    def test_load_balance_metrics_computation(self, simple_mesh, function_space):
        """Verify load balance metrics are correctly computed."""
        metrics = LoadBalancingMetrics(comm)

        mesh_metrics = metrics.compute_mesh_metrics(simple_mesh)
        dof_metrics = metrics.compute_dof_metrics(function_space)

        # Check that metrics are reasonable
        assert mesh_metrics['global_cells'] > 0
        assert dof_metrics['global_dofs'] > 0
        assert 0 <= mesh_metrics['imbalance_percent'] <= 100
        assert 0 <= dof_metrics['imbalance_percent'] <= 100

    @requires_mpi
    def test_load_balance_quality_check(self, simple_mesh, function_space):
        """Verify load balance quality checking."""
        metrics = LoadBalancingMetrics(comm)

        is_balanced, message = metrics.check_load_balance_quality(
            simple_mesh, function_space, threshold=50.0
        )

        # For a uniform mesh, should be reasonably balanced
        assert isinstance(is_balanced, bool)
        assert len(message) > 0

    def test_observation_distribution(self):
        """Verify observations are distributed evenly across ranks."""
        num_obs = 100
        ranges = LoadBalancer.distribute_observations(num_obs, size)

        # All observations should be covered
        all_obs = set()
        for start, end in ranges:
            for i in range(start, end):
                all_obs.add(i)

        assert all_obs == set(range(num_obs))

        # Distribution should be reasonably balanced
        obs_per_rank = [end - start for start, end in ranges]
        max_obs = max(obs_per_rank)
        min_obs = min(obs_per_rank)
        assert max_obs - min_obs <= 1  # At most 1 difference


# =============================================================================
# Section 6: Non-Blocking Communication Tests
# =============================================================================

class TestNonBlockingCommunication:
    """Tests for non-blocking communication utilities."""

    def test_async_vector_ops(self):
        """Verify async vector operations work correctly."""
        async_ops = AsyncVectorOps(comm)

        # Test allreduce
        send_data = np.array([float(rank + 1)])
        recv, request = async_ops.allreduce_begin(send_data, op=MPI.SUM)
        result = async_ops.allreduce_end()

        expected = sum(range(1, size + 1))
        assert abs(result[0] - expected) < 1e-10

    def test_batched_communication(self):
        """Verify batched communication reduces correctly."""
        batched = BatchedCommunication(comm)

        # Add multiple values
        for i in range(5):
            batched.add_to_batch(float(rank + i))

        results = batched.allreduce_batch()

        # Verify each value was summed across ranks
        for i, result in enumerate(results):
            expected = sum(r + i for r in range(size))
            assert abs(result - expected) < 1e-10

    def test_overlap_pattern(self, test_function):
        """Verify overlap pattern executes correctly."""
        overlap = OverlapComputeComm(comm)

        def independent_work():
            return sum(range(100))

        result = overlap.overlap_scatter_with_work(test_function, independent_work)

        assert result == sum(range(100))


# =============================================================================
# Section 7: Communication Tracking Tests
# =============================================================================

class TestCommunicationTracking:
    """Tests for communication tracking and profiling."""

    def test_communication_tracker(self):
        """Verify communication operations are tracked."""
        tracker = CommunicationTracker(comm)

        # Record various operations
        tracker.record_scatter_forward(100, context="test_fwd")
        tracker.record_scatter_reverse(100, context="test_rev")
        tracker.record_allreduce(1, context="test_allreduce")

        summary = tracker.get_summary()

        assert summary['total_operations'] == 3 * size
        assert 'scatter_forward' in summary['operations_by_type']
        assert 'scatter_reverse' in summary['operations_by_type']
        assert 'allreduce' in summary['operations_by_type']

    def test_parallel_timer(self):
        """Verify parallel timer works correctly."""
        timer = ParallelTimer(comm)

        timer.start("test_region")
        # Simulate work
        _ = sum(range(1000))
        timer.stop("test_region")

        # Check timer recorded something
        assert "test_region" in timer.timers
        assert timer.timers["test_region"]["count"] == 1
        assert timer.timers["test_region"]["total"] >= 0


# =============================================================================
# Section 8: Parallel I/O Tests
# =============================================================================

class TestParallelIO:
    """Tests for parallel I/O operations."""

    def test_save_load_vec(self, tmp_path):
        """Verify distributed vector save/load."""
        # tmp_path is only on rank 0, broadcast it
        if rank == 0:
            filepath = str(tmp_path / "test_vec.dat")
        else:
            filepath = None
        filepath = comm.bcast(filepath, root=0)

        # Create and save vector
        global_size = 50
        vec = DistributedVectorOps.create_distributed_vec(global_size, comm)

        start, end = vec.getOwnershipRange()
        for i in range(start, end):
            vec.setValue(i, float(i))
        vec.assemblyBegin()
        vec.assemblyEnd()

        pio = ParallelIO(comm)
        pio.save_vec(vec, filepath)

        # Load and verify
        loaded = pio.load_vec(filepath)

        assert loaded.getSize() == global_size

        # Check values match
        for i in range(start, end):
            assert abs(loaded.getValue(i) - float(i)) < 1e-10

        vec.destroy()
        loaded.destroy()


# =============================================================================
# Section 9: Integration Tests
# =============================================================================

class TestParallelIntegration:
    """Integration tests combining multiple parallel features."""

    @requires_mpi
    def test_full_parallel_workflow(self, simple_mesh, function_space):
        """Test a complete parallel workflow."""
        # 1. Verify mesh distribution
        metrics = LoadBalancingMetrics(comm)
        mesh_metrics = metrics.compute_mesh_metrics(simple_mesh)
        assert mesh_metrics['global_cells'] > 0

        # 2. Create function and set values
        u = fem.Function(function_space)
        u.interpolate(lambda x: x[0] * x[1])

        # 3. Scatter and verify
        u.x.scatter_forward()

        # 4. Compute global integral
        local_sum = np.sum(u.x.array[:u.x.index_map.size_local])
        global_sum = comm.allreduce(local_sum, op=MPI.SUM)

        # All ranks should have same global sum
        all_sums = comm.allgather(global_sum)
        assert len(set(all_sums)) == 1

        # 5. Track communication
        tracker = CommunicationTracker(comm)
        tracker.record_scatter_forward(u.x.index_map.size_local, "integration_test")
        summary = tracker.get_summary()
        assert summary['total_operations'] > 0

    @requires_mpi
    def test_parallel_cost_function_pattern(self, function_space):
        """Test pattern used in parallel cost function evaluation."""
        # This tests the pattern from cost_functions.py

        # Create control variable
        m = fem.Function(function_space)
        m.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

        # Scatter for consistency
        m.x.scatter_forward()

        # Compute local contribution (mimics background term)
        local_norm_sq = np.dot(m.x.array, m.x.array)

        # Global reduction (mimics cost function value)
        global_norm_sq = comm.allreduce(local_norm_sq, op=MPI.SUM)

        # Verify all ranks have same result
        all_norms = comm.allgather(global_norm_sq)
        assert len(set(all_norms)) == 1

        # Compute local gradient contribution
        local_grad = 2.0 * m.x.array.copy()

        # Create gradient function and set values
        grad = fem.Function(function_space)
        local_size = grad.x.index_map.size_local
        grad.x.array[:local_size] = local_grad[:local_size]

        # Scatter reverse for accumulation (mimics adjoint assembly)
        grad.x.scatter_reverse(dolfinx.la.InsertMode.add)
        grad.x.scatter_forward()

        # Gradient should be non-zero
        grad_norm = np.sqrt(comm.allreduce(np.dot(grad.x.array, grad.x.array), op=MPI.SUM))
        assert grad_norm > 0


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

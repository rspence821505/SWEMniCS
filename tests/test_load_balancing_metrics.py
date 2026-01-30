"""
Tests for load balancing metrics module.

Tests comprehensive load balancing analysis including mesh distribution,
DOF distribution, and communication volume tracking.
"""

import pytest
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem

from swe4dvar.utils.load_balancing_metrics import (
    LoadBalancingMetrics,
    CommunicationTracker
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


def test_load_balancing_metrics_init():
    """Test LoadBalancingMetrics initialization."""
    metrics = LoadBalancingMetrics(comm)

    assert metrics.comm == comm
    assert metrics.rank == rank
    assert metrics.size == size
    assert len(metrics.metrics_history) == 0


def test_compute_mesh_metrics(simple_mesh):
    """Test mesh distribution metrics computation."""
    metrics = LoadBalancingMetrics(comm)
    mesh_metrics = metrics.compute_mesh_metrics(simple_mesh)

    # Check required fields
    assert 'global_cells' in mesh_metrics
    assert 'local_cells_per_rank' in mesh_metrics
    assert 'ghost_cells_per_rank' in mesh_metrics
    assert 'avg_local_cells' in mesh_metrics
    assert 'max_local_cells' in mesh_metrics
    assert 'min_local_cells' in mesh_metrics
    assert 'imbalance_percent' in mesh_metrics
    assert 'efficiency_percent' in mesh_metrics

    # Check values are reasonable
    assert mesh_metrics['global_cells'] > 0
    assert len(mesh_metrics['local_cells_per_rank']) == size
    assert mesh_metrics['imbalance_percent'] >= 0.0
    assert mesh_metrics['efficiency_percent'] <= 100.0


def test_compute_dof_metrics(function_space):
    """Test DOF distribution metrics computation."""
    metrics = LoadBalancingMetrics(comm)
    dof_metrics = metrics.compute_dof_metrics(function_space)

    # Check required fields
    assert 'global_dofs' in dof_metrics
    assert 'local_dofs_per_rank' in dof_metrics
    assert 'ghost_dofs_per_rank' in dof_metrics
    assert 'avg_local_dofs' in dof_metrics
    assert 'max_local_dofs' in dof_metrics
    assert 'min_local_dofs' in dof_metrics
    assert 'imbalance_percent' in dof_metrics

    # Check values
    assert dof_metrics['global_dofs'] > 0
    assert len(dof_metrics['local_dofs_per_rank']) == size
    assert sum(dof_metrics['local_dofs_per_rank']) == dof_metrics['global_dofs']


def test_compute_communication_metrics(simple_mesh, function_space):
    """Test communication volume metrics computation."""
    metrics = LoadBalancingMetrics(comm)
    comm_metrics = metrics.compute_communication_metrics(simple_mesh, function_space)

    # Check required fields
    assert 'ghost_cells_per_rank' in comm_metrics
    assert 'ghost_dofs_per_rank' in comm_metrics
    assert 'total_ghost_dofs' in comm_metrics
    assert 'avg_ghost_dofs' in comm_metrics
    assert 'comm_imbalance_percent' in comm_metrics

    # Check values
    assert comm_metrics['total_ghost_dofs'] >= 0
    assert len(comm_metrics['ghost_dofs_per_rank']) == size


def test_compute_comprehensive_metrics(simple_mesh, function_space):
    """Test comprehensive metrics computation."""
    metrics = LoadBalancingMetrics(comm)
    all_metrics = metrics.compute_comprehensive_metrics(
        simple_mesh,
        function_space,
        name="test_run"
    )

    # Check structure
    assert 'name' in all_metrics
    assert 'num_ranks' in all_metrics
    assert 'mesh' in all_metrics
    assert 'dofs' in all_metrics
    assert 'communication' in all_metrics

    assert all_metrics['name'] == "test_run"
    assert all_metrics['num_ranks'] == size

    # Check that it was stored in history
    assert len(metrics.metrics_history) == 1
    assert metrics.metrics_history[0] == all_metrics


def test_print_metrics(simple_mesh, function_space, capsys):
    """Test metrics printing (output on rank 0 only)."""
    metrics = LoadBalancingMetrics(comm)
    all_metrics = metrics.compute_comprehensive_metrics(
        simple_mesh,
        function_space,
        name="print_test"
    )

    metrics.print_metrics(all_metrics)

    # Only rank 0 should print
    if rank == 0:
        captured = capsys.readouterr()
        assert "Load Balancing Metrics Report" in captured.out
        assert "print_test" in captured.out


def test_print_detailed_per_rank_metrics(simple_mesh, function_space, capsys):
    """Test detailed per-rank metrics printing."""
    metrics = LoadBalancingMetrics(comm)
    all_metrics = metrics.compute_comprehensive_metrics(
        simple_mesh,
        function_space,
        name="detailed_test"
    )

    metrics.print_detailed_per_rank_metrics(all_metrics)

    if rank == 0:
        captured = capsys.readouterr()
        assert "Detailed Per-Rank Metrics" in captured.out
        assert "Local Cells" in captured.out
        assert "Ghost DOFs" in captured.out


def test_check_load_balance_quality(simple_mesh, function_space):
    """Test load balance quality checking."""
    metrics = LoadBalancingMetrics(comm)

    # Use a reasonable threshold
    is_balanced, message = metrics.check_load_balance_quality(
        simple_mesh,
        function_space,
        threshold=30.0
    )

    # Should be boolean
    assert isinstance(is_balanced, bool)
    assert isinstance(message, str)

    # For a simple uniform mesh, should be well balanced
    # (at least not terrible)
    assert len(message) > 0


@requires_mpi
def test_metrics_consistency_across_ranks(simple_mesh, function_space):
    """Test that metrics are consistent across all ranks."""
    metrics = LoadBalancingMetrics(comm)
    all_metrics = metrics.compute_comprehensive_metrics(
        simple_mesh,
        function_space,
        name="consistency_test"
    )

    # Global values should be same on all ranks (due to broadcast)
    global_cells = all_metrics['mesh']['global_cells']
    global_dofs = all_metrics['dofs']['global_dofs']

    # Gather to check consistency
    all_global_cells = comm.allgather(global_cells)
    all_global_dofs = comm.allgather(global_dofs)

    # All ranks should have the same global values
    assert len(set(all_global_cells)) == 1
    assert len(set(all_global_dofs)) == 1


def test_communication_tracker_init():
    """Test CommunicationTracker initialization."""
    tracker = CommunicationTracker(comm)

    assert tracker.comm == comm
    assert tracker.rank == rank
    assert tracker.size == size
    assert len(tracker.comm_log) == 0


def test_communication_tracker_record_operations():
    """Test recording communication operations."""
    tracker = CommunicationTracker(comm)

    # Record various operations
    tracker.record_scatter_forward(100, context="test_scatter_fwd")
    tracker.record_scatter_reverse(100, context="test_scatter_rev")
    tracker.record_allreduce(1, context="test_allreduce")
    tracker.record_broadcast(50, root=0, context="test_broadcast")

    # Check that operations were recorded
    assert len(tracker.comm_log) == 4

    # Check operation types
    types = [op['type'] for op in tracker.comm_log]
    assert 'scatter_forward' in types
    assert 'scatter_reverse' in types
    assert 'allreduce' in types
    assert 'broadcast' in types


def test_communication_tracker_summary():
    """Test communication summary generation."""
    tracker = CommunicationTracker(comm)

    # Record some operations
    tracker.record_scatter_forward(100, context="test1")
    tracker.record_scatter_forward(200, context="test2")
    tracker.record_allreduce(1, context="test3")

    summary = tracker.get_summary()

    # Check structure
    assert 'total_operations' in summary
    assert 'operations_by_type' in summary
    assert 'total_volume' in summary
    assert 'volume_by_type' in summary

    # Check values (should be same on all ranks after broadcast)
    assert summary['total_operations'] == 3 * size  # Each rank recorded 3
    assert 'scatter_forward' in summary['operations_by_type']
    assert 'allreduce' in summary['operations_by_type']


def test_communication_tracker_print_summary(capsys):
    """Test communication summary printing."""
    tracker = CommunicationTracker(comm)

    # Record operations
    tracker.record_scatter_forward(100, context="test")
    tracker.record_allreduce(1, context="test")

    tracker.print_summary()

    if rank == 0:
        captured = capsys.readouterr()
        assert "Communication Tracking Summary" in captured.out


def test_communication_tracker_clear():
    """Test clearing communication log."""
    tracker = CommunicationTracker(comm)

    # Record operations
    tracker.record_scatter_forward(100, context="test")
    tracker.record_allreduce(1, context="test")

    assert len(tracker.comm_log) > 0

    # Clear
    tracker.clear()

    assert len(tracker.comm_log) == 0


@requires_mpi
def test_load_balance_with_different_mesh_sizes():
    """Test load balancing metrics with different mesh resolutions."""
    metrics = LoadBalancingMetrics(comm)

    # Test with different resolutions
    resolutions = [5, 10, 20]

    for res in resolutions:
        test_mesh = mesh.create_unit_square(
            comm,
            res,
            res,
            cell_type=mesh.CellType.triangle
        )

        V = fem.functionspace(test_mesh, ("Lagrange", 1))

        mesh_metrics = metrics.compute_mesh_metrics(test_mesh)
        dof_metrics = metrics.compute_dof_metrics(V)

        # Higher resolution should have more cells/DOFs
        assert mesh_metrics['global_cells'] > 0
        assert dof_metrics['global_dofs'] > 0

        # Imbalance should be reasonable (< 50% for simple uniform mesh)
        assert mesh_metrics['imbalance_percent'] < 50.0
        assert dof_metrics['imbalance_percent'] < 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

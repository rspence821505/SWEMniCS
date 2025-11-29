"""
Tests for profiling utilities module.

Tests hierarchical timing, memory profiling, scalability analysis,
and comprehensive profiling capabilities.
"""

import pytest
import numpy as np
import time
from mpi4py import MPI

from swemnics.utils.profiling import (
    HierarchicalTimer,
    MemoryProfiler,
    ScalabilityAnalyzer,
    ComprehensiveProfiler,
    profile
)


# MPI configuration
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Test markers
requires_mpi = pytest.mark.skipif(size == 1, reason="Requires MPI with multiple ranks")
serial_only = pytest.mark.skipif(size > 1, reason="Serial test only")


def test_hierarchical_timer_init():
    """Test HierarchicalTimer initialization."""
    timer = HierarchicalTimer(comm)

    assert timer.comm == comm
    assert timer.rank == rank
    assert timer.size == size
    assert len(timer.timings) == 0
    assert len(timer.stack) == 0
    assert len(timer.hierarchy) == 0


def test_hierarchical_timer_simple_region():
    """Test timing a simple region."""
    timer = HierarchicalTimer(comm)

    with timer.region("test_region"):
        time.sleep(0.01)  # Sleep for 10ms

    # Check that region was recorded
    assert "test_region" in timer.timings
    assert timer.timings["test_region"]["count"] == 1
    assert timer.timings["test_region"]["total"] >= 0.01  # At least 10ms


def test_hierarchical_timer_nested_regions():
    """Test nested timing regions."""
    timer = HierarchicalTimer(comm)

    with timer.region("parent"):
        time.sleep(0.01)

        with timer.region("child1"):
            time.sleep(0.01)

        with timer.region("child2"):
            time.sleep(0.01)

    # Check hierarchy
    assert "parent" in timer.timings
    assert "child1" in timer.timings
    assert "child2" in timer.timings

    assert timer.timings["child1"]["parent"] == "parent"
    assert timer.timings["child2"]["parent"] == "parent"
    assert "child1" in timer.timings["parent"]["children"]
    assert "child2" in timer.timings["parent"]["children"]


def test_hierarchical_timer_multiple_calls():
    """Test timing region called multiple times."""
    timer = HierarchicalTimer(comm)

    for _ in range(3):
        with timer.region("repeated"):
            time.sleep(0.005)

    assert timer.timings["repeated"]["count"] == 3
    assert timer.timings["repeated"]["total"] >= 0.015  # At least 15ms total


def test_hierarchical_timer_summary():
    """Test getting timing summary."""
    timer = HierarchicalTimer(comm)

    with timer.region("test"):
        time.sleep(0.01)

    summary = timer.get_summary()

    # Summary should be available on all ranks (due to broadcast)
    assert "test" in summary
    assert 'min_time' in summary["test"]
    assert 'max_time' in summary["test"]
    assert 'avg_time' in summary["test"]
    assert 'imbalance_percent' in summary["test"]


def test_hierarchical_timer_print_report(capsys):
    """Test printing timing report."""
    timer = HierarchicalTimer(comm)

    with timer.region("parent"):
        with timer.region("child"):
            time.sleep(0.01)

    timer.print_report()

    if rank == 0:
        captured = capsys.readouterr()
        assert "Hierarchical Timing Report" in captured.out
        assert "parent" in captured.out


def test_hierarchical_timer_reset():
    """Test resetting timer."""
    timer = HierarchicalTimer(comm)

    with timer.region("test"):
        pass

    assert len(timer.timings) > 0

    timer.reset()

    assert len(timer.timings) == 0
    assert len(timer.stack) == 0
    assert len(timer.hierarchy) == 0


def test_memory_profiler_init():
    """Test MemoryProfiler initialization."""
    profiler = MemoryProfiler(comm)

    assert profiler.comm == comm
    assert profiler.rank == rank
    assert profiler.size == size
    assert len(profiler.snapshots) == 0


def test_memory_profiler_get_current_memory():
    """Test getting current memory usage."""
    profiler = MemoryProfiler(comm)

    mem_mb = profiler.get_current_memory_mb()

    # Should return a non-negative number
    assert mem_mb >= 0.0


def test_memory_profiler_snapshot():
    """Test taking memory snapshots."""
    profiler = MemoryProfiler(comm)

    profiler.snapshot("start")
    profiler.snapshot("middle")
    profiler.snapshot("end")

    assert len(profiler.snapshots) == 3
    assert profiler.snapshots[0]["label"] == "start"
    assert profiler.snapshots[1]["label"] == "middle"
    assert profiler.snapshots[2]["label"] == "end"


def test_memory_profiler_summary():
    """Test getting memory summary."""
    profiler = MemoryProfiler(comm)

    profiler.snapshot("test")

    summary = profiler.get_summary()

    # Should be available on all ranks
    assert "test" in summary
    assert 'min_memory_mb' in summary["test"]
    assert 'max_memory_mb' in summary["test"]
    assert 'avg_memory_mb' in summary["test"]
    assert 'total_memory_mb' in summary["test"]


def test_memory_profiler_print_report(capsys):
    """Test printing memory report."""
    profiler = MemoryProfiler(comm)

    profiler.snapshot("test")
    profiler.print_report()

    if rank == 0:
        captured = capsys.readouterr()
        assert "Memory Usage Report" in captured.out


def test_memory_profiler_reset():
    """Test resetting memory profiler."""
    profiler = MemoryProfiler(comm)

    profiler.snapshot("test")
    assert len(profiler.snapshots) > 0

    profiler.reset()
    assert len(profiler.snapshots) == 0


def test_scalability_analyzer_init():
    """Test ScalabilityAnalyzer initialization."""
    analyzer = ScalabilityAnalyzer(comm)

    assert analyzer.comm == comm
    assert analyzer.rank == rank
    assert analyzer.size == size
    assert len(analyzer.scaling_data) == 0


def test_scalability_analyzer_record_timing():
    """Test recording timing data."""
    analyzer = ScalabilityAnalyzer(comm)

    analyzer.record_timing(
        num_ranks=1,
        problem_size=1000,
        wall_time=10.0
    )

    analyzer.record_timing(
        num_ranks=2,
        problem_size=1000,
        wall_time=6.0
    )

    assert len(analyzer.scaling_data) == 2


def test_scalability_analyzer_strong_scaling():
    """Test strong scaling efficiency computation."""
    analyzer = ScalabilityAnalyzer(comm)

    # Simulate strong scaling data
    analyzer.record_timing(num_ranks=1, problem_size=1000, wall_time=10.0)
    analyzer.record_timing(num_ranks=2, problem_size=1000, wall_time=6.0)
    analyzer.record_timing(num_ranks=4, problem_size=1000, wall_time=3.5)

    results = analyzer.compute_strong_scaling_efficiency(baseline_ranks=1)

    assert 'baseline_ranks' in results
    assert 'baseline_time' in results
    assert 'efficiencies' in results

    assert results['baseline_ranks'] == 1
    assert results['baseline_time'] == 10.0
    assert len(results['efficiencies']) == 2  # For 2 and 4 ranks


def test_scalability_analyzer_weak_scaling():
    """Test weak scaling efficiency computation."""
    analyzer = ScalabilityAnalyzer(comm)

    # Simulate weak scaling data (problem size scales with ranks)
    analyzer.record_timing(num_ranks=1, problem_size=1000, wall_time=10.0)
    analyzer.record_timing(num_ranks=2, problem_size=2000, wall_time=11.0)
    analyzer.record_timing(num_ranks=4, problem_size=4000, wall_time=12.0)

    results = analyzer.compute_weak_scaling_efficiency()

    assert 'baseline_ranks' in results
    assert 'baseline_time' in results
    assert 'baseline_problem_size' in results
    assert 'efficiencies' in results

    assert len(results['efficiencies']) == 2  # For 2 and 4 ranks


def test_scalability_analyzer_print_strong_scaling_report(capsys):
    """Test printing strong scaling report."""
    analyzer = ScalabilityAnalyzer(comm)

    analyzer.record_timing(num_ranks=1, problem_size=1000, wall_time=10.0)
    analyzer.record_timing(num_ranks=2, problem_size=1000, wall_time=6.0)

    analyzer.print_strong_scaling_report(baseline_ranks=1)

    if rank == 0:
        captured = capsys.readouterr()
        assert "Strong Scaling Analysis" in captured.out


def test_scalability_analyzer_print_weak_scaling_report(capsys):
    """Test printing weak scaling report."""
    analyzer = ScalabilityAnalyzer(comm)

    analyzer.record_timing(num_ranks=1, problem_size=1000, wall_time=10.0)
    analyzer.record_timing(num_ranks=2, problem_size=2000, wall_time=11.0)

    analyzer.print_weak_scaling_report()

    if rank == 0:
        captured = capsys.readouterr()
        assert "Weak Scaling Analysis" in captured.out


def test_comprehensive_profiler_init():
    """Test ComprehensiveProfiler initialization."""
    profiler = ComprehensiveProfiler(comm)

    assert profiler.comm == comm
    assert profiler.rank == rank
    assert isinstance(profiler.timer, HierarchicalTimer)
    assert isinstance(profiler.memory, MemoryProfiler)
    assert isinstance(profiler.scalability, ScalabilityAnalyzer)


def test_comprehensive_profiler_profile_region():
    """Test profiling a region with comprehensive profiler."""
    profiler = ComprehensiveProfiler(comm)

    with profiler.profile_region("test_region"):
        time.sleep(0.01)

    # Check timer
    assert "test_region" in profiler.timer.timings

    # Check memory snapshots
    assert len(profiler.memory.snapshots) >= 2  # start and end


def test_comprehensive_profiler_print_full_report(capsys):
    """Test printing full profiling report."""
    profiler = ComprehensiveProfiler(comm)

    with profiler.profile_region("test"):
        time.sleep(0.01)

    profiler.print_full_report()

    if rank == 0:
        captured = capsys.readouterr()
        assert "Hierarchical Timing Report" in captured.out or "Memory Usage Report" in captured.out


def test_comprehensive_profiler_reset_all():
    """Test resetting all profiling data."""
    profiler = ComprehensiveProfiler(comm)

    with profiler.profile_region("test"):
        pass

    assert len(profiler.timer.timings) > 0
    assert len(profiler.memory.snapshots) > 0

    profiler.reset_all()

    assert len(profiler.timer.timings) == 0
    assert len(profiler.memory.snapshots) == 0


def test_profile_context_manager():
    """Test the profile context manager."""
    with profile("test_context", comm=comm, print_report=False) as profiler:
        time.sleep(0.01)

    # Should have created profiler
    assert isinstance(profiler, ComprehensiveProfiler)

    # Should have timed the region
    assert "test_context" in profiler.timer.timings


@requires_mpi
def test_hierarchical_timer_imbalance_detection():
    """Test detection of timing imbalance across ranks."""
    timer = HierarchicalTimer(comm)

    # Create artificial imbalance
    sleep_time = 0.01 * (rank + 1)

    with timer.region("imbalanced"):
        time.sleep(sleep_time)

    summary = timer.get_summary()

    # Should detect imbalance
    if size > 1:
        assert summary["imbalanced"]["imbalance_percent"] > 0


@requires_mpi
def test_memory_profiler_across_ranks():
    """Test memory profiling across multiple ranks."""
    profiler = MemoryProfiler(comm)

    # Each rank allocates different amount
    data = np.zeros(1000 * (rank + 1))

    profiler.snapshot("after_allocation")

    summary = profiler.get_summary()

    # Should show variation across ranks
    if size > 1:
        mem_data = summary["after_allocation"]
        # Max should be greater than min (different allocations)
        assert mem_data["max_memory_mb"] >= mem_data["min_memory_mb"]


def test_scalability_analyzer_efficiency_calculation():
    """Test efficiency calculation accuracy."""
    analyzer = ScalabilityAnalyzer(comm)

    # Perfect strong scaling
    analyzer.record_timing(num_ranks=1, problem_size=100, wall_time=10.0)
    analyzer.record_timing(num_ranks=2, problem_size=100, wall_time=5.0)
    analyzer.record_timing(num_ranks=4, problem_size=100, wall_time=2.5)

    results = analyzer.compute_strong_scaling_efficiency(baseline_ranks=1)

    # Check perfect efficiency
    for eff in results['efficiencies']:
        assert abs(eff['efficiency_percent'] - 100.0) < 0.1  # Should be ~100%


def test_nested_profiling_with_memory():
    """Test nested profiling regions with memory tracking."""
    profiler = ComprehensiveProfiler(comm)

    with profiler.profile_region("outer", snapshot_memory=True):
        data1 = np.zeros(1000)

        with profiler.profile_region("inner", snapshot_memory=True):
            data2 = np.zeros(1000)
            time.sleep(0.01)

    # Check hierarchy
    assert "outer" in profiler.timer.timings
    assert "inner" in profiler.timer.timings
    assert profiler.timer.timings["inner"]["parent"] == "outer"

    # Check memory snapshots
    assert len(profiler.memory.snapshots) >= 4  # start/end for each region


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

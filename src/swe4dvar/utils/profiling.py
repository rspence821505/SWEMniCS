"""
Advanced profiling utilities for parallel 4D-Var computations.

Provides comprehensive profiling tools including:
- Hierarchical timing
- Memory profiling
- Scalability analysis
- Performance visualization helpers
"""

from typing import Dict, List, Optional, Tuple, Any
import time
import sys
from contextlib import contextmanager
from collections import defaultdict
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


class HierarchicalTimer:
    """
    Hierarchical timer for nested profiling regions.

    Supports nested timing contexts with automatic hierarchy tracking.
    """

    def __init__(self, comm: MPI.Comm = None):
        """
        Initialize hierarchical timer.

        Args:
            comm: MPI communicator
        """
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Timing data: region_name -> {start, total, count, children}
        self.timings: Dict[str, Dict] = {}

        # Current timing stack for nesting
        self.stack: List[Tuple[str, float]] = []

        # Track parent-child relationships
        self.hierarchy: Dict[str, List[str]] = defaultdict(list)

    @contextmanager
    def region(self, name: str):
        """
        Time a named region with automatic nesting support.

        Args:
            name: Region name

        Example:
            with timer.region("forward_solve"):
                with timer.region("assembly"):
                    assemble_system()
                with timer.region("solve"):
                    solve_system()
        """
        # Track parent for hierarchy
        parent = self.stack[-1][0] if self.stack else None

        # Start timing
        self.comm.Barrier()  # Synchronize for consistent timing
        start_time = MPI.Wtime()

        # Initialize timing data if new region
        if name not in self.timings:
            self.timings[name] = {
                'total': 0.0,
                'count': 0,
                'parent': parent,
                'children': []
            }

        # Update hierarchy
        if parent and name not in self.hierarchy[parent]:
            self.hierarchy[parent].append(name)
            if name not in self.timings[parent]['children']:
                self.timings[parent]['children'].append(name)

        # Push to stack
        self.stack.append((name, start_time))

        try:
            yield
        finally:
            # Pop from stack
            region_name, region_start = self.stack.pop()

            # Stop timing
            self.comm.Barrier()
            elapsed = MPI.Wtime() - region_start

            # Update statistics
            self.timings[region_name]['total'] += elapsed
            self.timings[region_name]['count'] += 1

    def get_summary(self) -> Dict:
        """
        Get timing summary across all ranks.

        Returns:
            Dictionary with timing statistics (min, max, avg per region)
        """
        # Gather timings from all ranks
        all_timings = self.comm.gather(self.timings, root=0)

        summary = {}

        if self.rank == 0 and all_timings:
            # Get all region names
            all_regions = set()
            for timings in all_timings:
                all_regions.update(timings.keys())

            # Compute statistics for each region
            for region in all_regions:
                times = []
                counts = []

                for rank_timings in all_timings:
                    if region in rank_timings:
                        times.append(rank_timings[region]['total'])
                        counts.append(rank_timings[region]['count'])
                    else:
                        times.append(0.0)
                        counts.append(0)

                summary[region] = {
                    'min_time': min(times),
                    'max_time': max(times),
                    'avg_time': sum(times) / len(times),
                    'std_time': np.std(times),
                    'min_count': min(counts),
                    'max_count': max(counts),
                    'avg_count': sum(counts) / len(counts),
                    'imbalance_percent': ((max(times) - min(times)) / (sum(times) / len(times)) * 100)
                                        if sum(times) > 0 else 0.0,
                    'parent': self.timings[region]['parent'] if region in self.timings else None,
                    'children': self.timings[region]['children'] if region in self.timings else []
                }

        # Broadcast to all ranks
        summary = self.comm.bcast(summary, root=0)

        return summary

    def print_report(self, min_time_threshold: float = 0.001):
        """
        Print hierarchical timing report.

        Args:
            min_time_threshold: Minimum time (seconds) to include in report
        """
        summary = self.get_summary()

        if self.rank == 0:
            print("\n" + "="*90)
            print(f"Hierarchical Timing Report ({self.size} ranks)")
            print("="*90)

            # Find root regions (no parent)
            root_regions = [r for r, data in summary.items() if data['parent'] is None]

            # Recursively print hierarchy
            def print_region(region: str, indent: int = 0):
                data = summary[region]

                if data['avg_time'] < min_time_threshold:
                    return

                prefix = "  " * indent + "├─ " if indent > 0 else ""

                avg_per_call = data['avg_time'] / data['avg_count'] if data['avg_count'] > 0 else 0.0

                print(f"{prefix}{region}:")
                print(f"{'  ' * (indent + 1)}Time: "
                      f"min={data['min_time']:.4f}s, "
                      f"avg={data['avg_time']:.4f}s, "
                      f"max={data['max_time']:.4f}s")
                print(f"{'  ' * (indent + 1)}Calls: {data['avg_count']:.0f}, "
                      f"Avg/call: {avg_per_call:.4f}s, "
                      f"Imbalance: {data['imbalance_percent']:.1f}%")

                # Print children recursively
                for child in data['children']:
                    print_region(child, indent + 1)

            # Print all root regions
            for region in sorted(root_regions):
                print_region(region)

            print("="*90 + "\n")

    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.stack.clear()
        self.hierarchy.clear()


class MemoryProfiler:
    """
    Memory usage profiler for distributed computations.

    Tracks memory usage across MPI ranks.
    """

    def __init__(self, comm: MPI.Comm = None):
        """
        Initialize memory profiler.

        Args:
            comm: MPI communicator
        """
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Track memory snapshots
        self.snapshots: List[Dict] = []

    def get_current_memory_mb(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in megabytes (estimate via PETSc)
        """
        try:
            # Try to get PETSc memory info
            mem_usage = PETSc.Log.getMemoryUsage()
            return mem_usage / (1024.0 * 1024.0)  # Convert to MB
        except:
            # Fallback: try using resource module
            try:
                import resource
                usage = resource.getrusage(resource.RUSAGE_SELF)
                return usage.ru_maxrss / 1024.0  # Convert to MB (varies by OS)
            except:
                return 0.0

    def snapshot(self, label: str):
        """
        Take a memory snapshot.

        Args:
            label: Label for this snapshot
        """
        mem_mb = self.get_current_memory_mb()

        self.snapshots.append({
            'label': label,
            'memory_mb': mem_mb,
            'rank': self.rank
        })

    def get_summary(self) -> Dict:
        """
        Get memory usage summary across all ranks.

        Returns:
            Dictionary with memory statistics
        """
        # Gather snapshots from all ranks
        all_snapshots = self.comm.gather(self.snapshots, root=0)

        summary = {}

        if self.rank == 0 and all_snapshots:
            # Group by label
            labels = set()
            for snapshots in all_snapshots:
                for snapshot in snapshots:
                    labels.add(snapshot['label'])

            # Compute statistics for each label
            for label in labels:
                memories = []

                for rank_snapshots in all_snapshots:
                    for snapshot in rank_snapshots:
                        if snapshot['label'] == label:
                            memories.append(snapshot['memory_mb'])
                            break

                if memories:
                    summary[label] = {
                        'min_memory_mb': min(memories),
                        'max_memory_mb': max(memories),
                        'avg_memory_mb': sum(memories) / len(memories),
                        'total_memory_mb': sum(memories)
                    }

        # Broadcast to all ranks
        summary = self.comm.bcast(summary, root=0)

        return summary

    def print_report(self):
        """Print memory usage report."""
        summary = self.get_summary()

        if self.rank == 0:
            print("\n" + "="*70)
            print(f"Memory Usage Report ({self.size} ranks)")
            print("="*70)

            for label in sorted(summary.keys()):
                data = summary[label]
                print(f"\n{label}:")
                print(f"  Per-rank: min={data['min_memory_mb']:.1f} MB, "
                      f"avg={data['avg_memory_mb']:.1f} MB, "
                      f"max={data['max_memory_mb']:.1f} MB")
                print(f"  Total: {data['total_memory_mb']:.1f} MB")

            print("="*70 + "\n")

    def reset(self):
        """Reset all memory snapshots."""
        self.snapshots.clear()


class ScalabilityAnalyzer:
    """
    Analyzes scalability of parallel computations.

    Helps assess strong and weak scaling characteristics.
    """

    def __init__(self, comm: MPI.Comm = None):
        """
        Initialize scalability analyzer.

        Args:
            comm: MPI communicator
        """
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Store timing data for different rank counts
        self.scaling_data: List[Dict] = []

    def record_timing(
        self,
        num_ranks: int,
        problem_size: int,
        wall_time: float,
        metadata: Optional[Dict] = None
    ):
        """
        Record timing for scalability analysis.

        Args:
            num_ranks: Number of MPI ranks used
            problem_size: Problem size (e.g., number of DOFs)
            wall_time: Wall clock time in seconds
            metadata: Optional additional data
        """
        data = {
            'num_ranks': num_ranks,
            'problem_size': problem_size,
            'wall_time': wall_time,
            'metadata': metadata or {}
        }

        self.scaling_data.append(data)

    def compute_strong_scaling_efficiency(self, baseline_ranks: int = 1) -> Dict:
        """
        Compute strong scaling efficiency.

        Args:
            baseline_ranks: Baseline number of ranks for comparison

        Returns:
            Dictionary with strong scaling metrics
        """
        # Find baseline timing
        baseline = None
        for data in self.scaling_data:
            if data['num_ranks'] == baseline_ranks:
                baseline = data
                break

        if not baseline:
            return {}

        results = {
            'baseline_ranks': baseline_ranks,
            'baseline_time': baseline['wall_time'],
            'efficiencies': []
        }

        # Compute speedup and efficiency for each rank count
        for data in self.scaling_data:
            if data['num_ranks'] == baseline_ranks:
                continue

            speedup = baseline['wall_time'] / data['wall_time']
            ideal_speedup = data['num_ranks'] / baseline_ranks
            efficiency = speedup / ideal_speedup * 100.0

            results['efficiencies'].append({
                'num_ranks': data['num_ranks'],
                'time': data['wall_time'],
                'speedup': speedup,
                'ideal_speedup': ideal_speedup,
                'efficiency_percent': efficiency
            })

        return results

    def compute_weak_scaling_efficiency(self) -> Dict:
        """
        Compute weak scaling efficiency.

        Assumes problem size scales proportionally with rank count.

        Returns:
            Dictionary with weak scaling metrics
        """
        if not self.scaling_data:
            return {}

        # Use first entry as baseline
        baseline = min(self.scaling_data, key=lambda x: x['num_ranks'])

        results = {
            'baseline_ranks': baseline['num_ranks'],
            'baseline_time': baseline['wall_time'],
            'baseline_problem_size': baseline['problem_size'],
            'efficiencies': []
        }

        # Compute efficiency for each configuration
        for data in self.scaling_data:
            if data['num_ranks'] == baseline['num_ranks']:
                continue

            # Weak scaling: time should stay constant
            efficiency = (baseline['wall_time'] / data['wall_time']) * 100.0

            results['efficiencies'].append({
                'num_ranks': data['num_ranks'],
                'problem_size': data['problem_size'],
                'time': data['wall_time'],
                'efficiency_percent': efficiency
            })

        return results

    def print_strong_scaling_report(self, baseline_ranks: int = 1):
        """
        Print strong scaling report.

        Args:
            baseline_ranks: Baseline number of ranks
        """
        results = self.compute_strong_scaling_efficiency(baseline_ranks)

        if not results or self.rank != 0:
            return

        print("\n" + "="*70)
        print(f"Strong Scaling Analysis (baseline: {baseline_ranks} ranks)")
        print("="*70)
        print(f"Baseline time: {results['baseline_time']:.4f} s")
        print("\nRanks | Time (s) | Speedup | Ideal | Efficiency")
        print("-" * 70)

        for eff in results['efficiencies']:
            print(f"{eff['num_ranks']:5d} | {eff['time']:8.4f} | "
                  f"{eff['speedup']:7.2f} | {eff['ideal_speedup']:5.2f} | "
                  f"{eff['efficiency_percent']:6.1f}%")

        print("="*70 + "\n")

    def print_weak_scaling_report(self):
        """Print weak scaling report."""
        results = self.compute_weak_scaling_efficiency()

        if not results or self.rank != 0:
            return

        print("\n" + "="*70)
        print(f"Weak Scaling Analysis (baseline: {results['baseline_ranks']} ranks)")
        print("="*70)
        print(f"Baseline time: {results['baseline_time']:.4f} s")
        print(f"Baseline problem size: {results['baseline_problem_size']}")
        print("\nRanks | Problem Size | Time (s) | Efficiency")
        print("-" * 70)

        for eff in results['efficiencies']:
            print(f"{eff['num_ranks']:5d} | {eff['problem_size']:12d} | "
                  f"{eff['time']:8.4f} | {eff['efficiency_percent']:6.1f}%")

        print("="*70 + "\n")


class ComprehensiveProfiler:
    """
    Combines all profiling tools into a single interface.

    Provides one-stop profiling for 4D-Var computations.
    """

    def __init__(self, comm: MPI.Comm = None):
        """
        Initialize comprehensive profiler.

        Args:
            comm: MPI communicator
        """
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        # Initialize all profilers
        self.timer = HierarchicalTimer(comm)
        self.memory = MemoryProfiler(comm)
        self.scalability = ScalabilityAnalyzer(comm)

    @contextmanager
    def profile_region(self, name: str, snapshot_memory: bool = True):
        """
        Profile a code region with timing and optional memory tracking.

        Args:
            name: Region name
            snapshot_memory: Whether to take memory snapshots

        Example:
            with profiler.profile_region("forward_solve"):
                solve_forward_model()
        """
        if snapshot_memory:
            self.memory.snapshot(f"{name}_start")

        with self.timer.region(name):
            yield

        if snapshot_memory:
            self.memory.snapshot(f"{name}_end")

    def print_full_report(self):
        """Print complete profiling report."""
        self.timer.print_report()
        self.memory.print_report()

    def reset_all(self):
        """Reset all profiling data."""
        self.timer.reset()
        self.memory.reset()


@contextmanager
def profile(
    name: str = "main",
    comm: MPI.Comm = None,
    print_report: bool = True
):
    """
    Convenience context manager for quick profiling.

    Args:
        name: Name for the profiled region
        comm: MPI communicator
        print_report: Whether to print report on exit

    Example:
        with profile("4dvar_optimization"):
            run_optimization()
    """
    profiler = ComprehensiveProfiler(comm)

    with profiler.profile_region(name):
        yield profiler

    if print_report:
        profiler.print_full_report()

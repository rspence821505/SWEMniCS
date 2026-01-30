"""Timing utilities for performance monitoring.

This module provides context managers and utilities for tracking execution time
of code sections in scientific simulations.
"""

import time
from datetime import timedelta
from typing import Optional


def _get_mpi_rank() -> int:
    """Get the current MPI rank, or 0 if MPI is not available.

    Returns:
        Current MPI rank, or 0 if not running under MPI
    """
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except (ImportError, Exception):
        return 0


class Timer:
    """MPI-aware context manager for timing code sections.

    Automatically detects MPI rank and only prints from rank 0 to avoid
    duplicate output in parallel runs.

    Usage:
        # Basic usage (auto-detects MPI rank)
        with Timer("Operation name") as timer:
            # Code to time
            pass

        # Access elapsed time
        print(f"Elapsed: {timer.elapsed} seconds")

        # Disable output
        with Timer("Silent operation", verbose=False) as timer:
            pass

        # Auto-tracking for automatic summary (no manual dict needed)
        with Timer("Setup", verbose=False, track_key="setup"):
            pass
        Timer.print_summary()  # Uses auto-tracked timings

    Args:
        name: Name of the operation being timed
        verbose: If True, print timing information when exiting context
        rank: Optional MPI rank override. If None, automatically detects rank.
              Only rank 0 will print output (when verbose=True).
        track_key: Optional key for automatic tracking. If provided, elapsed time
                   will be stored in class-level registry for use with print_summary().

    Attributes:
        elapsed: Total elapsed time in seconds (available after __exit__)
    """

    # Class-level storage for auto-tracked timings
    _tracked_timings = {}

    def __init__(self, name="Operation", verbose=True, rank: Optional[int] = None, track_key: Optional[str] = None):
        self.name = name
        self.verbose = verbose
        self.rank = rank if rank is not None else _get_mpi_rank()
        self.elapsed = 0.0
        self.start = 0.0
        self.track_key = track_key

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self.start

        # Auto-track if key provided
        if self.track_key is not None:
            Timer._tracked_timings[self.track_key] = self.elapsed

        if self.verbose and self.rank == 0:
            print(
                f"{self.name}: {timedelta(seconds=self.elapsed)} ({self.elapsed:.2f}s)"
            )

    @staticmethod
    def print_summary(
        timings: Optional[dict] = None,
        total_key: str = "total",
        per_item_key: Optional[str] = None,
        num_items: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        """Print a formatted timing summary table.

        Args:
            timings: Optional dictionary of timing data. If None, uses auto-tracked timings
                     from Timers with track_key parameter.
            total_key: Key in timings dict that contains the total time (default: "total")
            per_item_key: Optional key for which to compute per-item average (e.g., "simulation" for per-timestep)
            num_items: Number of items for per-item average (e.g., number of timesteps)
            rank: Optional MPI rank. If None, auto-detects. Only rank 0 prints output.

        Example (manual dict):
            >>> timings = {
            ...     "setup": 0.5,
            ...     "simulation": 120.3,
            ...     "post_processing": 2.1,
            ...     "total": 122.9
            ... }
            >>> Timer.print_summary(timings, per_item_key="simulation", num_items=100)

        Example (auto-tracking):
            >>> with Timer("Setup", verbose=False, track_key="setup"):
            ...     pass
            >>> with Timer("Simulation", verbose=False, track_key="simulation"):
            ...     pass
            >>> Timer.print_summary(per_item_key="simulation", num_items=100)

            ======================================================================
            Timing Summary
            ======================================================================
            Setup:               0:00:00.500000 (  0.4%)
            Simulation:          0:02:00.300000 ( 97.9%)
            Post-processing:     0:00:02.100000 (  1.7%)
            ----------------------------------------------------------------------
            Total:               0:02:02.900000
            Per item:                             1.203 seconds
            ======================================================================
        """
        if rank is None:
            rank = _get_mpi_rank()

        # Only print from rank 0
        if rank != 0:
            return

        # Use auto-tracked timings if none provided
        if timings is None:
            timings = Timer._tracked_timings

        if total_key not in timings:
            raise ValueError(f"Total key '{total_key}' not found in timings dictionary")

        total_time = timings[total_key]

        # Print header
        print(f"\n{'='*70}")
        print("Timing Summary")
        print(f"{'='*70}")

        # Print each timing section (excluding total)
        for key, elapsed in timings.items():
            if key == total_key:
                continue

            # Format the key name (capitalize and replace underscores)
            display_name = key.replace("_", "-").capitalize()

            # Calculate percentage
            percentage = (elapsed / total_time * 100) if total_time > 0 else 0.0

            # Print row
            print(
                f"{display_name:<16} {str(timedelta(seconds=elapsed)):>20} ({percentage:5.1f}%)"
            )

        # Print separator
        print(f"{'-'*70}")

        # Print total
        print(f"{'Total:':<16} {str(timedelta(seconds=total_time)):>20}")

        # Print per-item average if requested
        if per_item_key is not None and num_items is not None and num_items > 0:
            if per_item_key not in timings:
                raise ValueError(f"Per-item key '{per_item_key}' not found in timings")

            per_item_time = timings[per_item_key] / num_items
            item_label = "Per item:"
            print(f"{item_label:<16} {per_item_time:>20.3f} seconds")

        # Print footer
        print(f"{'='*70}")

    @staticmethod
    def clear_tracked():
        """Clear auto-tracked timings.

        Useful when running multiple timing sessions in the same process.
        """
        Timer._tracked_timings.clear()

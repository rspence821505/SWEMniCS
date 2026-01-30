"""Newton solver diagnostics and logging.

This module provides the NewtonDiagnostics class for tracking Newton solver
convergence history with flexible output options.
"""

from typing import List, Optional, Dict, Any
import json
from pathlib import Path
from datetime import datetime


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


class NewtonDiagnostics:
    """Lightweight storage for Newton solver convergence diagnostics.

    This class tracks Newton iteration convergence data with flexible output:
    - Store in memory for post-processing
    - Print to console for monitoring
    - Log to file for persistent records

    Typical usage:
        diagnostics = NewtonDiagnostics(print_to_console=True, log_file="newton.log")

        # During Newton solve
        diagnostics.start_timestep(timestep=5, time=0.5)
        diagnostics.log_iteration(iteration=1, residual=1e-3, ...)
        diagnostics.end_timestep(converged=True, total_iters=3)

        # Analysis
        avg_iters = diagnostics.average_iterations()
        failed = diagnostics.get_failed_timesteps()
    """

    def __init__(
        self,
        print_to_console: bool = False,
        log_file: Optional[str] = None,
        store_history: bool = False,
        verbose: bool = False,
        rank: Optional[int] = None,
    ):
        """Initialize Newton diagnostics.

        Args:
            print_to_console: If True, print convergence info to stdout
            log_file: Optional path to log file. If provided, writes detailed logs.
            store_history: If True, store convergence history in memory
            verbose: If True, print detailed iteration info. If False, only summaries.
            rank: Optional MPI rank override. If None, automatically detects rank.
                  Only rank 0 will print output (when print_to_console=True).
        """
        self.print_to_console = print_to_console
        self.log_file = log_file
        self.store_history = store_history
        self.verbose = verbose
        self.rank = rank if rank is not None else _get_mpi_rank()

        # In-memory storage
        self.timestep_history: List[Dict[str, Any]] = []
        self._current_timestep: Optional[Dict[str, Any]] = None

        # Initialize log file if requested (only on rank 0)
        if self.log_file and self._should_print():
            self._init_log_file()

    def _should_print(self) -> bool:
        """Return True only on rank 0 for MPI-safe printing.

        Returns:
            True if this is rank 0, False otherwise
        """
        return self.rank == 0

    def print_config(self):
        """Print diagnostic configuration message (MPI-aware, only on rank 0).

        This method prints information about Newton diagnostics setup including
        whether logging is enabled and where files will be written.
        """
        if not self._should_print():
            return

        if self.log_file:
            print(f"Newton diagnostics will be logged to: {self.log_file}")

    def _init_log_file(self):
        """Initialize log file with header."""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.log_file, "w") as f:
            f.write(f"# Newton Solver Diagnostics Log\n")
            f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# {'='*70}\n\n")

    def start_timestep(self, timestep: int, time: float = None):
        """Start tracking a new timestep.

        Args:
            timestep: Timestep index
            time: Optional physical time value
        """
        self._current_timestep = {
            "timestep": timestep,
            "time": time,
            "iterations": [],
            "converged": None,
            "total_iterations": 0,
        }

        if self.print_to_console and self.verbose and self._should_print():
            time_str = f", t={time:.4e}" if time is not None else ""
            print(f"\n--- Timestep {timestep}{time_str} ---")

        if self.log_file:
            with open(self.log_file, "a") as f:
                time_str = f", time={time}" if time is not None else ""
                f.write(f"\nTimestep {timestep}{time_str}\n")

    def log_iteration(
        self,
        iteration: int,
        residual_norm: float,
        correction_norm: float = None,
        linear_iterations: int = None,
        linear_convergence: int = None,
        linear_residual: float = None,
        dx_0_norm: float = None,
    ):
        """Log information for a single Newton iteration.

        Args:
            iteration: Newton iteration number (0-indexed)
            residual_norm: Residual norm at this iteration
            correction_norm: Norm of Newton correction (optional)
            linear_iterations: Number of linear solver iterations (optional)
            linear_convergence: Linear solver convergence code (optional)
            linear_residual: Linear solver residual norm (optional)
            dx_0_norm: Reference norm of first correction for relative convergence (optional)
        """
        if self._current_timestep is None:
            raise RuntimeError("Must call start_timestep before log_iteration")

        # Store iteration data
        iter_data = {
            "iteration": iteration,
            "residual_norm": residual_norm,
            "correction_norm": correction_norm,
            "linear_iterations": linear_iterations,
            "linear_convergence": linear_convergence,
            "linear_residual": linear_residual,
            "dx_0_norm": dx_0_norm,
        }

        if self.store_history:
            self._current_timestep["iterations"].append(iter_data)

        # Print to console (mimicking current newton.py behavior)
        if self.print_to_console and self._should_print():
            if iteration == 0 and self.verbose:
                # Initial residual (before first iteration)
                print(f"Residual norm {residual_norm}")
            else:
                # Print linear solver info if available
                if (
                    linear_convergence is not None
                    and linear_iterations is not None
                    and self.verbose
                ):
                    print(
                        f"linear solver convergence {linear_convergence}, "
                        f"iterations {linear_iterations}, "
                        f"resid norm {linear_residual}"
                    )

                # Print Newton iteration info
                if self.verbose:
                    print(f"Newton Iteration {iteration}: ", end="")
                    if correction_norm is not None:
                        print(f"Correction norm {correction_norm}")
                    print(f"Residual norm {residual_norm}")

        # Log to file
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"  Iteration {iteration}:\n")
                f.write(f"    Residual norm: {residual_norm:.6e}\n")
                if correction_norm is not None:
                    f.write(f"    Correction norm: {correction_norm:.6e}\n")
                if dx_0_norm is not None:
                    f.write(f"    Reference norm (dx_0): {dx_0_norm:.6e}\n")
                if linear_iterations is not None:
                    f.write(f"    Linear iterations: {linear_iterations}\n")
                if linear_convergence is not None:
                    f.write(f"    Linear convergence: {linear_convergence}\n")
                if linear_residual is not None:
                    f.write(f"    Linear residual: {linear_residual:.6e}\n")

    def end_timestep(self, converged: bool, total_iterations: int, message: str = None):
        """End the current timestep and record final status.

        Args:
            converged: Whether Newton solver converged
            total_iterations: Total number of Newton iterations
            message: Optional convergence/failure message
        """
        if self._current_timestep is None:
            raise RuntimeError("Must call start_timestep before end_timestep")

        self._current_timestep["converged"] = converged
        self._current_timestep["total_iterations"] = total_iterations

        if self.store_history:
            self.timestep_history.append(self._current_timestep)

        # Print summary
        if self.print_to_console and self._should_print():
            if converged:
                print(f"Newton solver converged in {total_iterations} iterations")
            else:
                print(f"Newton solver FAILED after {total_iterations} iterations")
            if message:
                print(f"  {message}")

        # Log to file
        if self.log_file:
            with open(self.log_file, "a") as f:
                status = "CONVERGED" if converged else "FAILED"
                f.write(f"  Status: {status} in {total_iterations} iterations\n")
                if message:
                    f.write(f"  Message: {message}\n")

        self._current_timestep = None

    def get_timestep(self, timestep: int) -> Optional[Dict[str, Any]]:
        """Get diagnostics for a specific timestep.

        Args:
            timestep: Timestep index to retrieve

        Returns:
            Dictionary with timestep diagnostics or None if not found
        """
        for ts_data in self.timestep_history:
            if ts_data["timestep"] == timestep:
                return ts_data
        return None

    def get_failed_timesteps(self) -> List[int]:
        """Get list of timesteps where Newton solver failed to converge.

        Returns:
            List of timestep indices with convergence failures
        """
        return [
            ts["timestep"]
            for ts in self.timestep_history
            if not ts.get("converged", True)
        ]

    def average_iterations(self) -> float:
        """Compute average number of Newton iterations per timestep.

        Returns:
            Average iterations (only over converged timesteps)
        """
        converged = [
            ts["total_iterations"]
            for ts in self.timestep_history
            if ts.get("converged", False)
        ]
        return sum(converged) / len(converged) if converged else 0.0

    def max_iterations(self) -> int:
        """Get maximum number of iterations used in any timestep.

        Returns:
            Maximum iteration count
        """
        if not self.timestep_history:
            return 0
        return max(ts["total_iterations"] for ts in self.timestep_history)

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics for all timesteps.

        Returns:
            Dictionary with summary statistics
        """
        total = len(self.timestep_history)
        failed = len(self.get_failed_timesteps())
        converged = total - failed

        return {
            "total_timesteps": total,
            "converged": converged,
            "failed": failed,
            "convergence_rate": converged / total if total > 0 else 0.0,
            "average_iterations": self.average_iterations(),
            "max_iterations": self.max_iterations(),
        }

    def print_summary(self):
        """Print summary statistics to console (MPI-aware, only on rank 0)."""
        if not self._should_print():
            return

        summary = self.summary()
        print("\n" + "=" * 70)
        print("Newton Solver Summary")
        print("=" * 70)
        print(f"Total timesteps:     {summary['total_timesteps']}")
        print(f"Converged:           {summary['converged']}")
        print(f"Failed:              {summary['failed']}")
        print(f"Convergence rate:    {summary['convergence_rate']*100:.1f}%")
        print(f"Average iterations:  {summary['average_iterations']:.2f}")
        print(f"Max iterations:      {summary['max_iterations']}")
        print("=" * 70)

    def save_json(self, filename: str):
        """Save full diagnostics history to JSON file (MPI-aware, only on rank 0).

        Args:
            filename: Path to output JSON file
        """
        if not self._should_print():
            return

        output = {
            "summary": self.summary(),
            "timesteps": self.timestep_history,
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nNewton diagnostics saved to: {filename}\n")

    def clear(self):
        """Clear all stored diagnostics history."""
        self.timestep_history.clear()
        self._current_timestep = None

    def __repr__(self) -> str:
        summary = self.summary()
        return (
            f"NewtonDiagnostics("
            f"timesteps={summary['total_timesteps']}, "
            f"avg_iters={summary['average_iterations']:.2f}, "
            f"failed={summary['failed']})"
        )

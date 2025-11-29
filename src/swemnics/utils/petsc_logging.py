"""
PETSc logging and performance monitoring utilities.

Provides integration with PETSc's built-in logging infrastructure for
detailed performance analysis of parallel computations.
"""

from typing import Optional, Dict, List
import sys
from contextlib import contextmanager
from petsc4py import PETSc
from mpi4py import MPI


class PETScLogger:
    """
    Wrapper for PETSc logging infrastructure.

    Provides convenient access to PETSc's performance logging capabilities,
    including event tracking, stage creation, and log viewing.
    """

    def __init__(self, comm: MPI.Comm = None):
        """
        Initialize PETSc logger.

        Args:
            comm: MPI communicator (defaults to MPI.COMM_WORLD)
        """
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        # Track whether logging has been started
        self._logging_active = False

        # Store custom events
        self._custom_events: Dict[str, int] = {}

        # Store custom stages
        self._custom_stages: Dict[str, int] = {}

    def begin(self):
        """
        Begin PETSc logging.

        Must be called before any operations you want to log.
        """
        if not self._logging_active:
            PETSc.Log.begin()
            self._logging_active = True

    def view(self, viewer: Optional[PETSc.Viewer] = None):
        """
        View PETSc log summary.

        Args:
            viewer: PETSc viewer for output (defaults to stdout)
        """
        if self._logging_active:
            if viewer is None:
                # Default to ASCII viewer on stdout
                viewer = PETSc.Viewer().createASCII("-", comm=self.comm)
            PETSc.Log.view(viewer)

    def view_to_file(self, filename: str):
        """
        Write PETSc log to file.

        Args:
            filename: Output filename
        """
        if self._logging_active:
            viewer = PETSc.Viewer().createASCII(filename, "w", comm=self.comm)
            PETSc.Log.view(viewer)
            viewer.destroy()

    def register_event(self, name: str) -> int:
        """
        Register a custom event for logging.

        Args:
            name: Event name

        Returns:
            Event ID
        """
        if name not in self._custom_events:
            event_id = PETSc.Log.Event().register(name)
            self._custom_events[name] = event_id
        return self._custom_events[name]

    def push_event(self, name: str):
        """
        Start logging a custom event.

        Args:
            name: Event name (must be registered first)
        """
        if name not in self._custom_events:
            self.register_event(name)

        event_id = self._custom_events[name]
        PETSc.Log.Event().begin(event_id)

    def pop_event(self, name: str):
        """
        Stop logging a custom event.

        Args:
            name: Event name
        """
        if name in self._custom_events:
            event_id = self._custom_events[name]
            PETSc.Log.Event().end(event_id)

    @contextmanager
    def event(self, name: str):
        """
        Context manager for logging an event.

        Args:
            name: Event name

        Example:
            with logger.event("forward_solve"):
                # Code to profile
                solver.solve()
        """
        self.push_event(name)
        try:
            yield
        finally:
            self.pop_event(name)

    def register_stage(self, name: str) -> int:
        """
        Register a custom stage for logging.

        Stages are higher-level than events and can contain multiple events.

        Args:
            name: Stage name

        Returns:
            Stage ID
        """
        if name not in self._custom_stages:
            stage_id = PETSc.Log.Stage(name).id
            self._custom_stages[name] = stage_id
        return self._custom_stages[name]

    def push_stage(self, name: str):
        """
        Enter a logging stage.

        Args:
            name: Stage name (must be registered first)
        """
        if name not in self._custom_stages:
            self.register_stage(name)

        stage_id = self._custom_stages[name]
        PETSc.Log.Stage.push(stage_id)

    def pop_stage(self):
        """Exit current logging stage."""
        PETSc.Log.Stage.pop()

    @contextmanager
    def stage(self, name: str):
        """
        Context manager for a logging stage.

        Args:
            name: Stage name

        Example:
            with logger.stage("optimization"):
                # Optimization code
                optimizer.solve()
        """
        self.push_stage(name)
        try:
            yield
        finally:
            self.pop_stage()

    def get_flops(self) -> float:
        """
        Get floating point operations count.

        Returns:
            Total FLOPS across all ranks
        """
        return PETSc.Log.getFlops()


class PerformanceMonitor:
    """
    High-level performance monitoring using PETSc logging.

    Provides convenient methods for monitoring common 4D-Var operations.
    """

    def __init__(self, comm: MPI.Comm = None, auto_start: bool = True):
        """
        Initialize performance monitor.

        Args:
            comm: MPI communicator
            auto_start: Automatically start PETSc logging
        """
        self.logger = PETScLogger(comm)
        self.comm = self.logger.comm
        self.rank = self.logger.rank

        if auto_start:
            self.logger.begin()

        # Pre-register common events for 4D-Var
        self._register_common_events()

    def _register_common_events(self):
        """Register commonly used events for 4D-Var."""
        common_events = [
            "forward_model",
            "adjoint_model",
            "cost_function",
            "gradient",
            "hessian_vector_product",
            "observation_operator",
            "optimization_step",
            "linear_solve",
            "assembly",
            "checkpoint_save",
            "checkpoint_load"
        ]

        for event_name in common_events:
            self.logger.register_event(event_name)

    def monitor_forward_model(self):
        """Context manager for monitoring forward model."""
        return self.logger.event("forward_model")

    def monitor_adjoint_model(self):
        """Context manager for monitoring adjoint model."""
        return self.logger.event("adjoint_model")

    def monitor_cost_function(self):
        """Context manager for monitoring cost function evaluation."""
        return self.logger.event("cost_function")

    def monitor_gradient(self):
        """Context manager for monitoring gradient computation."""
        return self.logger.event("gradient")

    def monitor_optimization_step(self):
        """Context manager for monitoring optimization step."""
        return self.logger.event("optimization_step")

    def print_summary(self, filename: Optional[str] = None):
        """
        Print performance summary.

        Args:
            filename: Optional output file (prints to stdout if None)
        """
        if filename:
            self.logger.view_to_file(filename)
        else:
            self.logger.view()


class LoggingConfiguration:
    """
    Manages PETSc logging configuration options.

    Provides methods to configure various PETSc logging settings
    for detailed performance analysis.
    """

    @staticmethod
    def enable_detailed_logging():
        """
        Enable detailed PETSc logging with all options.

        Sets up comprehensive logging for in-depth performance analysis.
        """
        # Enable various PETSc logging options
        PETSc.Options().setValue("-log_view", "")
        PETSc.Options().setValue("-log_summary", "")

    @staticmethod
    def enable_memory_logging():
        """Enable memory usage logging."""
        PETSc.Options().setValue("-malloc_log", "")
        PETSc.Options().setValue("-memory_view", "")

    @staticmethod
    def enable_mpi_logging():
        """Enable MPI communication logging."""
        PETSc.Options().setValue("-log_view_memory", "")
        PETSc.Options().setValue("-log_mpe", "")

    @staticmethod
    def set_log_file(filename: str):
        """
        Set output file for PETSc log.

        Args:
            filename: Output filename
        """
        PETSc.Options().setValue("-log_view", f":{filename}")

    @staticmethod
    def disable_logging():
        """Disable PETSc logging."""
        PETSc.Options().setValue("-log_view", "false")

    @staticmethod
    def configure_from_dict(config: Dict[str, str]):
        """
        Configure logging from dictionary.

        Args:
            config: Dictionary of PETSc options
                   e.g., {"-log_view": "", "-log_summary": ""}
        """
        for key, value in config.items():
            PETSc.Options().setValue(key, value)


class TimingBreakdown:
    """
    Analyzes timing breakdown from PETSc logging data.

    Provides utilities to extract and analyze timing information
    from PETSc logs.
    """

    def __init__(self, comm: MPI.Comm = None):
        """
        Initialize timing breakdown analyzer.

        Args:
            comm: MPI communicator
        """
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def estimate_communication_fraction(
        self,
        total_time: float,
        computation_time: float
    ) -> Dict[str, float]:
        """
        Estimate communication vs computation time.

        Args:
            total_time: Total wall clock time
            computation_time: Time spent in local computation

        Returns:
            Dictionary with timing breakdown
        """
        # Estimate communication time
        comm_time = total_time - computation_time

        # Compute fractions
        comm_fraction = comm_time / total_time if total_time > 0 else 0.0
        comp_fraction = computation_time / total_time if total_time > 0 else 0.0

        breakdown = {
            'total_time': total_time,
            'computation_time': computation_time,
            'communication_time': comm_time,
            'computation_fraction': comp_fraction,
            'communication_fraction': comm_fraction
        }

        return breakdown

    def print_timing_breakdown(
        self,
        total_time: float,
        computation_time: float,
        target_comm_fraction: float = 0.2
    ):
        """
        Print formatted timing breakdown with assessment.

        Args:
            total_time: Total wall clock time
            computation_time: Time spent in local computation
            target_comm_fraction: Target communication fraction threshold
        """
        breakdown = self.estimate_communication_fraction(total_time, computation_time)

        if self.rank == 0:
            print("\n" + "="*70)
            print("Timing Breakdown")
            print("="*70)
            print(f"Total time:         {breakdown['total_time']:.4f} s")
            print(f"Computation time:   {breakdown['computation_time']:.4f} s "
                  f"({breakdown['computation_fraction']*100:.1f}%)")
            print(f"Communication time: {breakdown['communication_time']:.4f} s "
                  f"({breakdown['communication_fraction']*100:.1f}%)")

            # Assessment
            print("\n--- Assessment ---")
            if breakdown['communication_fraction'] <= target_comm_fraction:
                status = "GOOD - Communication overhead is acceptable"
            elif breakdown['communication_fraction'] <= 0.3:
                status = "FAIR - Consider reducing communication"
            else:
                status = "POOR - Communication is a significant bottleneck"

            print(f"Communication fraction: {breakdown['communication_fraction']*100:.1f}% "
                  f"(target: <{target_comm_fraction*100:.0f}%)")
            print(f"Status: {status}")
            print("="*70 + "\n")


def setup_default_logging(
    log_file: Optional[str] = None,
    enable_memory: bool = False,
    comm: MPI.Comm = None
) -> PETScLogger:
    """
    Set up default PETSc logging configuration.

    Convenience function to quickly set up logging with sensible defaults.

    Args:
        log_file: Optional output file for logs
        enable_memory: Whether to enable memory logging
        comm: MPI communicator

    Returns:
        Configured PETScLogger instance
    """
    comm = comm or MPI.COMM_WORLD

    # Configure logging options
    LoggingConfiguration.enable_detailed_logging()

    if enable_memory:
        LoggingConfiguration.enable_memory_logging()

    if log_file:
        LoggingConfiguration.set_log_file(log_file)

    # Create and start logger
    logger = PETScLogger(comm)
    logger.begin()

    return logger


@contextmanager
def petsc_log_context(
    log_file: Optional[str] = None,
    enable_memory: bool = False,
    comm: MPI.Comm = None
):
    """
    Context manager for PETSc logging.

    Automatically starts logging on entry and prints summary on exit.

    Args:
        log_file: Optional output file for logs
        enable_memory: Whether to enable memory logging
        comm: MPI communicator

    Example:
        with petsc_log_context("performance.log"):
            # Code to profile
            run_4dvar()
    """
    logger = setup_default_logging(log_file, enable_memory, comm)

    try:
        yield logger
    finally:
        # Print summary when exiting context
        if log_file:
            logger.view_to_file(log_file)
        else:
            logger.view()

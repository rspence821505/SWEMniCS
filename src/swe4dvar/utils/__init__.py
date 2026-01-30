"""Utility helpers for SWE4DVar solvers and data assimilation.

The :mod:`swe4dvar.utils` package gathers reusable building blocks that are
shared across solver implementations, including:

- finite-element compatibility helpers
- observation station management
- time-step data management for 4D-Var workflows
- solver storage containers
- MPI/parallel convenience wrappers
- visualization helpers
- load balancing metrics and analysis
- PETSc logging integration
- performance profiling tools
- non-blocking communication utilities
"""

from .fem_utilities import create_element, create_mixed_element
from .observation_stations import StationManager
from .timestep_manager import TimeStepDataManager
from .solver_storage import SolverStateStorage
from .parallel_ops import ParallelContext, DistributedVectorOps
from .visualization import SolverVisualizer
from .load_balancing_metrics import LoadBalancingMetrics, CommunicationTracker
from .petsc_logging import (
    PETScLogger,
    PerformanceMonitor,
    LoggingConfiguration,
    setup_default_logging,
    petsc_log_context
)
from .profiling import (
    HierarchicalTimer,
    MemoryProfiler,
    ScalabilityAnalyzer,
    ComprehensiveProfiler,
    profile
)
from .nonblocking_comm import (
    NonBlockingScatter,
    AsyncVectorOps,
    OverlapComputeComm,
    AsyncObservationOperator,
    BatchedCommunication
)
from .solver_parameters import (
    get_default_solver_params,
    get_solver_params_preset
)
from .output_paths import (
    OUTPUT_ROOT,
    LOGS_DIR,
    FIGURES_DIR,
    CHECKPOINTS_DIR,
    DATA_DIR,
    ensure_output_dirs,
    get_figure_path,
    get_data_path,
    get_log_path,
)

__all__ = [
    "create_element",
    "create_mixed_element",
    "StationManager",
    "TimeStepDataManager",
    "SolverStateStorage",
    "ParallelContext",
    "DistributedVectorOps",
    "SolverVisualizer",
    "LoadBalancingMetrics",
    "CommunicationTracker",
    "PETScLogger",
    "PerformanceMonitor",
    "LoggingConfiguration",
    "setup_default_logging",
    "petsc_log_context",
    "HierarchicalTimer",
    "MemoryProfiler",
    "ScalabilityAnalyzer",
    "ComprehensiveProfiler",
    "profile",
    "NonBlockingScatter",
    "AsyncVectorOps",
    "OverlapComputeComm",
    "AsyncObservationOperator",
    "BatchedCommunication",
    "get_default_solver_params",
    "get_solver_params_preset",
    "OUTPUT_ROOT",
    "LOGS_DIR",
    "FIGURES_DIR",
    "CHECKPOINTS_DIR",
    "DATA_DIR",
    "ensure_output_dirs",
    "get_figure_path",
    "get_data_path",
    "get_log_path",
]

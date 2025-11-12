"""Utility helpers for SWEMniCS solvers and data assimilation.

The :mod:`swemnics.utils` package gathers reusable building blocks that are
shared across solver implementations, including:

- finite-element compatibility helpers
- observation station management
- time-step data management for 4D-Var workflows
- solver storage containers
- MPI/parallel convenience wrappers
- visualization helpers
"""

from .fem_utilities import create_element, create_mixed_element
from .observation_stations import StationManager
from .timestep_manager import TimeStepDataManager
from .solver_storage import SolverStateStorage
from .parallel_ops import ParallelContext, DistributedVectorOps
from .visualization import SolverVisualizer

__all__ = [
    "create_element",
    "create_mixed_element",
    "StationManager",
    "TimeStepDataManager",
    "SolverStateStorage",
    "ParallelContext",
    "DistributedVectorOps",
    "SolverVisualizer",
]

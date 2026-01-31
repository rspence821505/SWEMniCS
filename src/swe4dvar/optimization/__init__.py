"""
Optimization module for SWE4DVar.

Provides optimization algorithms for 4D-Var data assimilation including
L-BFGS, Gauss-Newton, and PETSc TAO wrappers.
"""

from .optimizer_base import (
    Optimizer,
    LineSearch,
    TrustRegion,
    ConvergenceMonitor,
)
from .lbfgs import LBFGSOptimizer
from .gauss_newton import GaussNewtonOptimizer
from .petsc_tao_wrapper import PETScTAOWrapper

__all__ = [
    "Optimizer",
    "LineSearch",
    "TrustRegion",
    "ConvergenceMonitor",
    "LBFGSOptimizer",
    "GaussNewtonOptimizer",
    "PETScTAOWrapper",
]

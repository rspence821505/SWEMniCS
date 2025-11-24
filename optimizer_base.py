"""
Compatibility shim for legacy imports in tests.

Re-export the optimizer classes from the package module so tests that import
`optimizer_base` from the project root keep working.
"""

from swemnics.optimization.optimizer_base import (  # noqa: F401
    Optimizer,
    LineSearch,
    TrustRegion,
    ConvergenceMonitor,
)


"""
Optimization algorithms for 4D-Var data assimilation.

Recommended: Use PETScTAOWrapper or TAOOptimizerFactory for production use.
TAO provides battle-tested optimization algorithms with robust line search
and convergence monitoring.

Example
-------
>>> from swe4dvar.optimization import TAOOptimizerFactory
>>> optimizer = TAOOptimizerFactory.create_lbfgs(
...     cost_function,
...     memory_size=10,
...     options={'verbose': True}
... )
>>> m_optimal = optimizer.solve(m_initial)
"""

from .optimizer_base import (
    Optimizer,
    LineSearch,
    TrustRegion,
    ConvergenceMonitor,
)
from .petsc_tao_wrapper import PETScTAOWrapper, TAOOptimizerFactory
from .lbfgs import LBFGSOptimizer, PreconditionedLBFGS, BoundedLBFGS
from .gauss_newton import GaussNewtonOptimizer

# Factory aliases for easy TAO creation (recommended)
create_tao_optimizer = TAOOptimizerFactory.create_lbfgs
create_bounded_tao_optimizer = TAOOptimizerFactory.create_bounded_lbfgs
create_tao_trust_region = TAOOptimizerFactory.create_trust_region
create_tao_conjugate_gradient = TAOOptimizerFactory.create_conjugate_gradient

__all__ = [
    # Base classes
    "Optimizer",
    "LineSearch",
    "TrustRegion",
    "ConvergenceMonitor",
    # TAO (Recommended for production)
    "PETScTAOWrapper",
    "TAOOptimizerFactory",
    "create_tao_optimizer",
    "create_bounded_tao_optimizer",
    "create_tao_trust_region",
    "create_tao_conjugate_gradient",
    # Custom implementations (available but TAO preferred)
    "LBFGSOptimizer",
    "PreconditionedLBFGS",
    "BoundedLBFGS",
    "GaussNewtonOptimizer",
]

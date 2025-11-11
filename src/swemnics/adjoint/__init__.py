"""Adjoint modeling components for SWEMniCS.

This package contains the discrete adjoint solvers, tangent-linear
models, and checkpointing utilities needed for gradient-based data
assimilation and sensitivity analysis.  The key classes are re-exported
for convenience so callers can simply import from ``swemnics.adjoint``.
"""

from __future__ import annotations

from .adjoint_operators import (
    AdjointModel,
    ObservationAdjoint,
    CovarianceAdjoint,
    CompositeAdjoint,
    FiniteDifferenceAdjoint,
)
from .tangent_linear import (
    TangentLinearModel,
    ImplicitTLMSolver,
    FiniteDifferenceTLM,
)
from .implicit_adjoint import (
    ImplicitAdjointSolver,
    ImplicitAdjointStepAnalyzer,
    CheckpointedImplicitAdjoint,
)
from .checkpointing import (
    CheckpointingStrategy,
    CheckpointerBase,
    FullTrajectoryCheckpointer,
    StateOnlyCheckpointer,
    BinomialCheckpointer,
    CheckpointerFactory,
)

__all__ = [
    # Core adjoint models
    "AdjointModel",
    "ObservationAdjoint",
    "CovarianceAdjoint",
    "CompositeAdjoint",
    "FiniteDifferenceAdjoint",
    # Tangent-linear models
    "TangentLinearModel",
    "ImplicitTLMSolver",
    "FiniteDifferenceTLM",
    # Implicit adjoint solvers
    "ImplicitAdjointSolver",
    "ImplicitAdjointStepAnalyzer",
    "CheckpointedImplicitAdjoint",
    # Checkpointing utilities
    "CheckpointingStrategy",
    "CheckpointerBase",
    "FullTrajectoryCheckpointer",
    "StateOnlyCheckpointer",
    "BinomialCheckpointer",
    "CheckpointerFactory",
]

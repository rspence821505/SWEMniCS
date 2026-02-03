"""Adjoint module for SWE4DVar.

This module provides tools for computing adjoints and sensitivities
for 4D-Var data assimilation with implicit time-stepping schemes.

Submodules
----------
adjoint_operators : Base adjoint operations
    - AdjointModel
    - ObservationAdjoint
    - CovarianceAdjoint
    - CompositeAdjoint

implicit_adjoint : BDF2 implicit adjoint solver
    - ImplicitAdjointSolver
    - ImplicitAdjointStepAnalyzer

tangent_linear : Tangent Linear Model
    - TangentLinearModel
    - TLMValidator

checkpointing : Checkpointing strategies
    - FullTrajectoryCheckpointer
    - StateOnlyCheckpointer
    - BinomialCheckpointer
    - CheckpointingStrategy

Mathematical Background
-----------------------
For implicit BDF2 discretization, the adjoint equations are:

    J_n^T λ^n = (4/(2Δt))·M·λ^{n+1} - (1/(2Δt))·M·λ^{n+2} + forcing

where J_n is the Jacobian from the forward Newton solve.

Key insight: We reuse cached Jacobians from forward solve, providing
~50% cost savings compared to recomputation.

The gradient of the 4D-Var cost function is:
    ∇J(m) = B⁻¹(m - m_b) + λ₀

where λ₀ is obtained by backward integration of the adjoint equations.
"""

from __future__ import annotations

from .tangent_linear import TangentLinearModel, TLMValidator

# Import adjoint operators (when available)
try:
    from .adjoint_operators import (
        AdjointModel,
        ObservationAdjoint,
        CovarianceAdjoint,
        CompositeAdjoint,
    )
except ImportError:
    AdjointModel = None
    ObservationAdjoint = None
    CovarianceAdjoint = None
    CompositeAdjoint = None

# Import implicit adjoint solver (when available)
try:
    from .implicit_adjoint import (
        ImplicitAdjointSolver,
        ImplicitAdjointStepAnalyzer,
    )
except ImportError:
    ImplicitAdjointSolver = None
    ImplicitAdjointStepAnalyzer = None

# Import checkpointing (when available)
try:
    from .checkpointing import (
        FullTrajectoryCheckpointer,
        StateOnlyCheckpointer,
        BinomialCheckpointer,
        CheckpointingStrategy,
    )
except ImportError:
    FullTrajectoryCheckpointer = None
    StateOnlyCheckpointer = None
    BinomialCheckpointer = None
    CheckpointingStrategy = None

__all__ = [
    # Tangent Linear Model
    "TangentLinearModel",
    "TLMValidator",
    # Adjoint operators
    "AdjointModel",
    "ObservationAdjoint",
    "CovarianceAdjoint",
    "CompositeAdjoint",
    # Implicit adjoint
    "ImplicitAdjointSolver",
    "ImplicitAdjointStepAnalyzer",
    # Checkpointing
    "FullTrajectoryCheckpointer",
    "StateOnlyCheckpointer",
    "BinomialCheckpointer",
    "CheckpointingStrategy",
]

__version__ = "0.1.0"

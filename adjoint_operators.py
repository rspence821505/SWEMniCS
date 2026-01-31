"""
Compatibility wrapper for legacy imports used in tests.

Re-export the adjoint operator classes from the package module so
`from adjoint_operators import ...` continues to work while the code
resides under `swe4dvar.adjoint`.
"""

from swe4dvar.adjoint.adjoint_operators import (  # noqa: F401
    AdjointModel,
    ObservationAdjoint,
    CovarianceAdjoint,
    CompositeAdjoint,
    FiniteDifferenceAdjoint,
)


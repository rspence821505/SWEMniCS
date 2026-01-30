"""Forward modeling components for SWE4DVar.

This package bundles the shallow-water forward problems, solver
implementations, Newton utilities, and helper constructs used by the
examples and higher-level workflows.  The most commonly used classes are
re-exported here for convenience so callers can simply import from
``swe4dvar.forward`` without drilling into submodules.
"""

from __future__ import annotations

from .ADCIRC_2_FENICS import ADCIRCMesh
from .adcirc_problem import (
    ADCIRCBoundaries,
    ADCIRCProblem,
    ADCIRCTidalPotential,
)
from .newton import (
    CustomNewtonProblem,
    ElementBlockPreconditioner,
    NewtonSolver,
)
from .problems import (
    BaseProblem,
    ConvergenceProblem,
    DamProblem,
    FrictionLaw,
    IdealizedInlet,
    RainProblem,
    SlopedBeachProblem,
    TidalProblem,
    WellBalancedProblem,
)
from .solvers import (
    BaseSolver,
    CGImplicit,
    DGSolver,
    DGCGImplicit,
    DGImplicit,
    DGImplicitNonConservative,
    SUPGImplicit,
    get_solver,
)
from .variational_forms import (
    LinearizedVariationalForm,
    SWEVariationalForm,
    VariationalForm,
)

__all__ = [
    # ADCIRC utilities
    "ADCIRCMesh",
    "ADCIRCBoundaries",
    "ADCIRCProblem",
    "ADCIRCTidalPotential",
    # Problems
    "BaseProblem",
    "TidalProblem",
    "IdealizedInlet",
    "WellBalancedProblem",
    "RainProblem",
    "DamProblem",
    "ConvergenceProblem",
    "SlopedBeachProblem",
    "FrictionLaw",
    # Solvers
    "BaseSolver",
    "DGSolver",
    "DGImplicit",
    "DGImplicitNonConservative",
    "DGCGImplicit",
    "CGImplicit",
    "SUPGImplicit",
    "get_solver",
    # Newton utilities
    "CustomNewtonProblem",
    "ElementBlockPreconditioner",
    "NewtonSolver",
    # Variational forms
    "VariationalForm",
    "SWEVariationalForm",
    "LinearizedVariationalForm",
]

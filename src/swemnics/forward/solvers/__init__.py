"""Solver classes for steady-state and time-dependent shallow water equations.

This package contains various solver implementations for the shallow water
equations using different numerical methods including CG, DG, SUPG, and
mixed formulations.

Each Solver requires a Problem class to initialize, and Solvers inherit from
one another. New numerical methods can be implemented by inheriting from the
classes in this package.
"""

from .base_solver import BaseSolver
from .dg_solver import DGSolver
from .cg_implicit import CGImplicit
from .dg_implicit import DGImplicit
from .dg_implicit_nonconservative import DGImplicitNonConservative
from .supg_implicit import SUPGImplicit
from .dgcg_implicit import DGCGImplicit

# Solver factory
_get_solver = {
    "CG": CGImplicit,
    "SUPG": SUPGImplicit,
    "DGCG": DGCGImplicit,
    "DG": DGImplicit,
    "DGNC": DGImplicitNonConservative,
}


def get_solver(solver_type: str) -> type:
    """Get solver class by type string.

    Args:
        solver_type: One of 'CG', 'SUPG', 'DGCG', 'DG', 'DGNC'

    Returns:
        Solver class

    Raises:
        ValueError: If solver_type is unknown
    """
    try:
        return _get_solver[solver_type.upper()]
    except KeyError:
        raise ValueError(
            f"Unknown solver type {solver_type}, options available are: {list(_get_solver.keys())}"
        )


__all__ = [
    "BaseSolver",
    "DGSolver",
    "CGImplicit",
    "DGImplicit",
    "DGImplicitNonConservative",
    "SUPGImplicit",
    "DGCGImplicit",
    "get_solver",
]

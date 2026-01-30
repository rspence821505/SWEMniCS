"""Core physical utilities for SWE4DVar.

This subpackage provides reusable pieces tied to the shallow-water
physics: boundary-condition helpers, meteorological forcing utilities,
and the collection of physical constants used across solvers and
problems.  The most common symbols are re-exported for convenience.
"""

from __future__ import annotations

from .boundarycondition import BoundaryCondition, MarkBoundary
from .constants import (
    R,
    earth_elasticity,
    g,
    omega,
    p_air,
    p_water,
)
from .forcing import GriddedForcing

__all__ = [
    "BoundaryCondition",
    "MarkBoundary",
    "GriddedForcing",
    "g",
    "R",
    "omega",
    "p_water",
    "p_air",
    "earth_elasticity",
]

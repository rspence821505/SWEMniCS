"""
SWE4DVar: Shallow Water Equations 4D-Var Data Assimilation Framework

A comprehensive Python framework for solving the shallow water equations
using FEniCSx with advanced numerical methods, adjoint-based sensitivity
analysis, and 4D-Var data assimilation.

Features:
- Multiple discretizations: CG, DG, SUPG, mixed elements
- Time integration: Implicit Euler, Crank-Nicolson, BDF2
- 4D-Var data assimilation: Standard 4D-Var, DC-4DVar, DC-WME-4DVar
- Adjoint methods with Jacobian caching
- Full MPI parallelization

Example
-------
>>> from swe4dvar.forward.problems import TidalProblem
>>> from swe4dvar.forward.solvers import get_solver
>>>
>>> problem = TidalProblem(nx=40, ny=10, dt=3600, nt=168)
>>> solver = get_solver("SUPG")(problem, theta=1.0, p_degree=[1, 1])
>>> solver.time_loop({"rtol": 1e-5, "atol": 1e-6, "max_it": 10})
"""

from importlib.metadata import PackageNotFoundError, version

try:
    dist_name = "SWE4DVar"
    __version__ = version(dist_name)
except PackageNotFoundError:
    __version__ = "1.0.0"

# Core imports for convenient access
from .forward.problems import FrictionLaw

__all__ = [
    "__version__",
    "FrictionLaw",
]

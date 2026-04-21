"""dolfinx 0.9 / 0.10 compatibility shims.

Isolates the surface-level API differences between dolfinx 0.9 and 0.10 so
the rest of the project can be written against a single helper interface.

Only two shims are needed today:

1. ``interpolation_points(element)`` — 0.9 exposes this as a method,
   0.10 as an attribute.
2. ``create_vector_from_form(form)`` — 0.9 accepts ``Form``,
   0.10 requires ``list[FunctionSpace]``.

If/when the project drops 0.9 support, the call sites can be inlined with
the 0.10 API and this module deleted.
"""
from __future__ import annotations

import dolfinx
from dolfinx.fem import petsc as _petsc

# Version gate. dolfinx uses simple "0.9.0", "0.10.0.post5" style strings.
_DOLFINX_010_OR_NEWER: bool = not dolfinx.__version__.startswith("0.9")


def interpolation_points(element):
    """Return the interpolation points ndarray for a ``FiniteElement``.

    Cross-version: in dolfinx 0.9 ``element.interpolation_points`` is a
    method; in 0.10 it is an attribute (ndarray). Callers in this project
    should use this helper instead of calling ``.interpolation_points()``
    directly.
    """
    pts = element.interpolation_points
    return pts() if callable(pts) else pts


def create_vector_from_form(form):
    """Create a PETSc ``Vec`` sized for the given linear ``Form``.

    Cross-version replacement for ``dolfinx.fem.petsc.create_vector(form)``:
    dolfinx 0.10 requires a ``list[FunctionSpace]``. We recover the one
    relevant function space via ``form.function_spaces[0]``, which
    matches the 0.9 behavior exactly for linear forms.
    """
    if _DOLFINX_010_OR_NEWER:
        return _petsc.create_vector([form.function_spaces[0]])
    return _petsc.create_vector(form)


def create_petsc_vector_from_map(index_map, block_size):
    """Create a PETSc ``Vec`` from a DOLFINx IndexMap.

    Cross-version replacement for ``dolfinx.la.create_petsc_vector``:
    dolfinx 0.9 exposes it at ``dolfinx.la.create_petsc_vector``;
    0.10 relocated it to ``dolfinx.la.petsc.create_vector`` and
    dropped the top-level ``la.create_petsc_vector`` symbol.
    """
    if _DOLFINX_010_OR_NEWER:
        from dolfinx.la.petsc import create_vector as _create
        return _create(index_map, block_size)
    from dolfinx import la as _la
    return _la.create_petsc_vector(index_map, block_size)

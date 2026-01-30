"""Finite element utility functions for element and function space creation.

This module provides compatibility functions for creating finite elements
across different versions of UFL and Basix.
"""

from dolfinx import mesh
from ufl.finiteelement import AbstractFiniteElement


def create_element(mesh: mesh.Mesh, family: str, degree: int, shape: tuple[int] = ()):
    """Compatible element creation for UFL and Basix.

    Args:
        mesh: The mesh object
        family: Element family (e.g., 'CG', 'DG')
        degree: Polynomial degree
        shape: Shape tuple for vector elements. Defaults to () for scalar.

    Returns:
        Finite element compatible with the installed FEniCS version
    """
    try:
        from ufl import FiniteElement, VectorElement

        if shape == ():
            return FiniteElement(family, mesh.ufl_cell(), degree)
        else:
            assert len(shape) == 1
            return VectorElement(family, mesh.ufl_cell(), degree, dim=shape[0])
    except ImportError:
        from basix.ufl import element

        return element(family, mesh.basix_cell(), degree, shape=shape)


def create_mixed_element(elements: list[AbstractFiniteElement]):
    """Compatibility function for creating a mixed element.

    Args:
        elements: List of finite elements to combine

    Returns:
        Mixed element compatible with the installed FEniCS version
    """
    try:
        from ufl import MixedElement

        return MixedElement(elements)
    except ImportError:
        from basix.ufl import mixed_element

        return mixed_element(elements)

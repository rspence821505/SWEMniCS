"""Finite element utility functions for element and function space creation.

This module provides compatibility functions for creating finite elements
across different versions of UFL and Basix, as well as boundary DOF detection
utilities for adjoint-based methods.
"""

import numpy as np
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


def get_boundary_dofs(V, mesh) -> np.ndarray:
    """
    Get DOF indices that lie on the domain boundary.

    Uses topological detection to find all DOFs on boundary facets.
    This is essential for discrete adjoint methods (DTO) where boundary
    DOF gradients must be zeroed to ensure mathematical consistency.

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        Function space (can be scalar, vector, or mixed).
    mesh : dolfinx.mesh.Mesh
        Computational mesh.

    Returns
    -------
    boundary_dofs : np.ndarray
        Array of DOF indices on the boundary.

    Notes
    -----
    For the discrete adjoint method:
    - Dirichlet BCs fix boundary values in the forward problem
    - The adjoint must satisfy homogeneous BCs (λ = 0 at BC DOFs)
    - This function finds ALL boundary DOFs, regardless of BC type
    - Use this to pass bc_dof_indices to ImplicitAdjointSolver

    Example
    -------
    >>> from swe4dvar.utils import get_boundary_dofs
    >>> boundary_dofs = get_boundary_dofs(solver.V, problem.mesh)
    >>> adjoint_solver = ImplicitAdjointSolver(
    ...     forward_model, trajectory, jacobians, dt,
    ...     bc_dof_indices=set(boundary_dofs.tolist())
    ... )
    """
    import dolfinx
    from dolfinx.mesh import locate_entities_boundary

    tdim = mesh.topology.dim
    fdim = tdim - 1

    def on_boundary(x):
        """Mark all boundary points."""
        return np.full(x.shape[1], True)

    boundary_facets = locate_entities_boundary(mesh, fdim, on_boundary)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)

    return boundary_dofs

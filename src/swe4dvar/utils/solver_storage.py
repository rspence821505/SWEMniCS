"""Storage classes for solver state, Jacobians, and adjoints.

This module provides dedicated storage containers for different types of
solver data used in forward solves and data assimilation.
"""

from typing import List, Optional
import numpy as np
from petsc4py import PETSc


class SolverStateStorage:
    """Centralized storage for solver states, Jacobians, and adjoint data.

    This class manages arrays and matrices generated during time-stepping,
    particularly for 4D-Var data assimilation where historical data must
    be retained for adjoint computations.
    """

    def __init__(self):
        """Initialize empty storage containers."""
        # Forward solve data
        self.saved_states: List[np.ndarray] = []
        """List of state vectors from forward time integration"""

        self.saved_jacobians: List[PETSc.Mat] = []
        """List of Jacobian matrices from Newton solves (distributed)"""

        # Adjoint solve data
        self.saved_adjoints: List[PETSc.Mat] = []
        """List of adjoint Jacobian matrices (distributed)"""

        # Wetting/drying data
        self.dry_nodes: List[np.ndarray] = []
        """List of dry node indices at each timestep"""

        self.saved_bathy: List[np.ndarray] = []
        """List of bathymetry values at observation points"""

        self.saved_true_bathy: List[np.ndarray] = []
        """List of true bathymetry values (before wetting/drying adjustments)"""

    def clear(self):
        """Clear all stored data to free memory.

        For PETSc matrices (Jacobians and adjoints), this properly destroys
        them before clearing the list to avoid memory leaks.
        """
        # Clear numpy arrays - no special handling needed
        self.saved_states.clear()
        self.dry_nodes.clear()
        self.saved_bathy.clear()
        self.saved_true_bathy.clear()

        # Destroy PETSc matrices before clearing
        # Use defensive programming to avoid crashes
        for J in self.saved_jacobians:
            try:
                if hasattr(J, 'destroy') and callable(J.destroy):
                    J.destroy()
            except (PETSc.Error, RuntimeError, AttributeError):
                # Ignore errors - matrix may already be destroyed or invalid
                pass
        self.saved_jacobians.clear()

        for A in self.saved_adjoints:
            try:
                if hasattr(A, 'destroy') and callable(A.destroy):
                    A.destroy()
            except (PETSc.Error, RuntimeError, AttributeError):
                # Ignore errors - matrix may already be destroyed or invalid
                pass
        self.saved_adjoints.clear()

    def save_state(self, state: np.ndarray):
        """Save a state vector.

        Args:
            state: State vector to save (will be copied)
        """
        self.saved_states.append(state.copy())

    def save_jacobian(self, jacobian: PETSc.Mat):
        """Save a Jacobian matrix.

        Args:
            jacobian: Jacobian matrix to save (will be copied)
        """
        if hasattr(jacobian, 'copy'):
            self.saved_jacobians.append(jacobian.copy())
        else:
            # For testing purposes, allow non-PETSc objects
            self.saved_jacobians.append(jacobian)

    def save_adjoint(self, adjoint: PETSc.Mat):
        """Save an adjoint Jacobian matrix.

        Args:
            adjoint: Adjoint matrix to save (will be copied)
        """
        self.saved_adjoints.append(adjoint.copy())

    def save_dry_nodes(self, dry_indices: np.ndarray):
        """Save dry node indices.

        Args:
            dry_indices: Array of dry node indices
        """
        self.dry_nodes.append(dry_indices.copy())

    def save_bathymetry(self, bathy: np.ndarray, is_true_bathy: bool = False):
        """Save bathymetry values.

        Args:
            bathy: Bathymetry array
            is_true_bathy: If True, save to true_bathy storage
        """
        if is_true_bathy:
            self.saved_true_bathy.append(bathy.copy())
        else:
            self.saved_bathy.append(bathy.copy())

    def get_state(self, index: int) -> Optional[np.ndarray]:
        """Retrieve a saved state by index.

        Args:
            index: Index of the state to retrieve (supports negative indexing)

        Returns:
            State vector or None if index out of bounds
        """
        try:
            return self.saved_states[index]
        except IndexError:
            return None

    def get_jacobian(self, index: int) -> Optional[PETSc.Mat]:
        """Retrieve a saved Jacobian by index.

        Args:
            index: Index of the Jacobian to retrieve (supports negative indexing)

        Returns:
            Jacobian matrix or None if index out of bounds
        """
        try:
            return self.saved_jacobians[index]
        except IndexError:
            return None

    def num_states(self) -> int:
        """Get number of saved states."""
        return len(self.saved_states)

    def num_jacobians(self) -> int:
        """Get number of saved Jacobians."""
        return len(self.saved_jacobians)

    def num_adjoints(self) -> int:
        """Get number of saved adjoints."""
        return len(self.saved_adjoints)

    def estimate_memory_mb(self) -> dict:
        """Estimate memory usage in MB for each storage component.

        Returns:
            Dictionary with memory estimates for each component
        """
        bytes_to_mb = 1.0 / (1024**2)

        estimates = {}

        # State vectors
        if self.saved_states:
            state_bytes = sum(s.nbytes for s in self.saved_states)
            estimates["states"] = state_bytes * bytes_to_mb

        # Jacobians (rough estimate based on PETSc matrix info)
        if self.saved_jacobians:
            # This is approximate - actual memory depends on sparsity
            jac_info = self.saved_jacobians[0].getInfo()
            nnz = jac_info["nz_used"]
            estimates["jacobians"] = len(self.saved_jacobians) * nnz * 8 * bytes_to_mb

        # Adjoints (similar to Jacobians)
        if self.saved_adjoints:
            adj_info = self.saved_adjoints[0].getInfo()
            nnz = adj_info["nz_used"]
            estimates["adjoints"] = len(self.saved_adjoints) * nnz * 8 * bytes_to_mb

        # Wetting/drying data
        if self.dry_nodes:
            dry_bytes = sum(d.nbytes for d in self.dry_nodes)
            estimates["dry_nodes"] = dry_bytes * bytes_to_mb

        if self.saved_bathy:
            bathy_bytes = sum(b.nbytes for b in self.saved_bathy)
            estimates["bathymetry"] = bathy_bytes * bytes_to_mb

        if self.saved_true_bathy:
            true_bathy_bytes = sum(b.nbytes for b in self.saved_true_bathy)
            estimates["true_bathymetry"] = true_bathy_bytes * bytes_to_mb

        estimates["total"] = sum(estimates.values())

        return estimates

    def __repr__(self) -> str:
        return (
            f"SolverStateStorage("
            f"states={self.num_states()}, "
            f"jacobians={self.num_jacobians()}, "
            f"adjoints={self.num_adjoints()}, "
            f"dry_nodes={len(self.dry_nodes)})"
        )

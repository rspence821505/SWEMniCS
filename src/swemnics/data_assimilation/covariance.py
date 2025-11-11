"""Covariance matrix implementations for data assimilation.

This module provides MPI-safe covariance matrix classes for use in 4D-Var
data assimilation. All implementations use PETSc Vec/Mat for parallel operations.

Mathematical Background
-----------------------
In 4D-Var, we need to compute operations of the form:
    - C⁻¹·v  (inverse application)
    - C·v    (forward application)
    - ⟨u, C⁻¹·v⟩ (inner product with inverse)

where C represents either:
    - B: Background error covariance
    - R: Observation error covariance
    - L: Predictability covariance (for DC-4DVar)

References
----------
Spence et al. (2025): "Variational Data-Consistent Assimilation"
"""

from abc import ABC, abstractmethod
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from typing import Optional, Union, List
import scipy.sparse as sp


class CovarianceMatrix(ABC):
    """Abstract base class for covariance matrices.

    All covariance matrices must support:
    1. apply(v): Compute C·v
    2. apply_inverse(v): Compute C⁻¹·v
    3. sqrt_apply(v): Compute C^(1/2)·v (for sampling)
    4. inner_product_inv(u, v): Compute ⟨u, C⁻¹·v⟩

    All methods operate on PETSc Vec objects for MPI safety.

    Attributes
    ----------
    comm : MPI.Comm
        MPI communicator
    size : int
        Global size of the covariance matrix
    local_size : int
        Local size on this MPI rank
    """

    def __init__(self, comm: MPI.Comm, size: int):
        """Initialize covariance matrix.

        Parameters
        ----------
        comm : MPI.Comm
            MPI communicator
        size : int
            Global size of the covariance matrix
        """
        self.comm = comm
        self.size = size

        # Determine local size for this rank
        ownership_range = PETSc.Vec().create(comm=comm)
        ownership_range.setSizes((PETSc.DECIDE, size))
        ownership_range.setUp()
        self.local_size = ownership_range.getLocalSize()
        self.ownership_range = ownership_range.getOwnershipRange()
        ownership_range.destroy()

    @abstractmethod
    def apply(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply covariance matrix: out = C·v.

        Parameters
        ----------
        v : PETSc.Vec
            Input vector
        out : PETSc.Vec, optional
            Output vector (created if None)

        Returns
        -------
        PETSc.Vec
            Result of C·v
        """
        pass

    @abstractmethod
    def apply_inverse(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply inverse covariance matrix: out = C⁻¹·v.

        Parameters
        ----------
        v : PETSc.Vec
            Input vector
        out : PETSc.Vec, optional
            Output vector (created if None)

        Returns
        -------
        PETSc.Vec
            Result of C⁻¹·v
        """
        pass

    def sqrt_apply(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply square root of covariance: out = C^(1/2)·v.

        Used for generating samples from N(0, C).

        Parameters
        ----------
        v : PETSc.Vec
            Input vector (typically standard normal)
        out : PETSc.Vec, optional
            Output vector (created if None)

        Returns
        -------
        PETSc.Vec
            Result of C^(1/2)·v
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement sqrt_apply"
        )

    def inner_product_inv(self, u: PETSc.Vec, v: PETSc.Vec) -> float:
        """Compute ⟨u, C⁻¹·v⟩.

        This is the weighted inner product used in cost functionals.

        Parameters
        ----------
        u : PETSc.Vec
            First vector
        v : PETSc.Vec
            Second vector

        Returns
        -------
        float
            Value of ⟨u, C⁻¹·v⟩ (global reduction)
        """
        # Default implementation: compute C⁻¹·v then take inner product
        C_inv_v = self.apply_inverse(v)
        result = u.dot(C_inv_v)  # PETSc handles MPI reduction
        C_inv_v.destroy()
        return result

    def create_vec(self) -> PETSc.Vec:
        """Create a compatible PETSc vector.

        Returns
        -------
        PETSc.Vec
            Vector with correct size and parallel layout
        """
        vec = PETSc.Vec().create(comm=self.comm)
        vec.setSizes((self.local_size, self.size))
        vec.setUp()
        return vec


class DiagonalCovariance(CovarianceMatrix):
    """Diagonal covariance matrix: C = diag(σ²₁, σ²₂, ..., σ²ₙ).

    This is the most efficient representation when errors are uncorrelated.
    Common for observation error covariance R.

    Storage: O(n)
    Apply: O(n)
    Inverse: O(n)

    Examples
    --------
    >>> comm = MPI.COMM_WORLD
    >>> # Uniform variance σ² = 1.5
    >>> R = DiagonalCovariance(comm, size=100, variance=1.5)
    >>>
    >>> # Spatially varying variance
    >>> variances = np.random.uniform(0.5, 2.0, size=100)
    >>> R = DiagonalCovariance(comm, size=100, diagonal=variances)
    """

    def __init__(
        self,
        comm: MPI.Comm,
        size: int,
        variance: Optional[float] = None,
        diagonal: Optional[Union[np.ndarray, PETSc.Vec]] = None,
    ):
        """Initialize diagonal covariance matrix.

        Parameters
        ----------
        comm : MPI.Comm
            MPI communicator
        size : int
            Global size of matrix
        variance : float, optional
            Uniform variance (σ²) for all entries
        diagonal : np.ndarray or PETSc.Vec, optional
            Diagonal entries (variances). If ndarray, must have length = size.

        Notes
        -----
        Exactly one of variance or diagonal must be provided.
        """
        super().__init__(comm, size)

        if (variance is None and diagonal is None) or (
            variance is not None and diagonal is not None
        ):
            raise ValueError("Provide exactly one of 'variance' or 'diagonal'")

        # Create diagonal vector
        self.diagonal = self.create_vec()

        if variance is not None:
            self.diagonal.set(variance)
        else:
            if isinstance(diagonal, np.ndarray):
                # Distribute array across ranks
                start, end = self.ownership_range
                local_diagonal = diagonal[start:end]
                self.diagonal.setArray(local_diagonal)
            else:
                # Already a PETSc Vec
                self.diagonal.copy(diagonal)

        self.diagonal.assemble()

        # Precompute inverse for efficiency
        self.inv_diagonal = self.diagonal.duplicate()
        self.diagonal.copy(self.inv_diagonal)
        self.inv_diagonal.reciprocal()  # Element-wise 1/x

    def apply(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply diagonal covariance: out = C·v."""
        if out is None:
            out = self.create_vec()

        # Element-wise multiplication
        out.pointwiseMult(self.diagonal, v)
        return out

    def apply_inverse(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply inverse diagonal covariance: out = C⁻¹·v."""
        if out is None:
            out = self.create_vec()

        # Element-wise multiplication with precomputed inverse
        out.pointwiseMult(self.inv_diagonal, v)
        return out

    def sqrt_apply(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply square root: out = C^(1/2)·v = diag(σ₁, σ₂, ...)·v."""
        if out is None:
            out = self.create_vec()

        # Take sqrt of diagonal
        sqrt_diag = self.diagonal.duplicate()
        self.diagonal.copy(sqrt_diag)
        sqrt_diag.sqrtabs()  # Element-wise sqrt

        out.pointwiseMult(sqrt_diag, v)
        sqrt_diag.destroy()
        return out

    def inner_product_inv(self, u: PETSc.Vec, v: PETSc.Vec) -> float:
        """Optimized ⟨u, C⁻¹·v⟩ for diagonal case."""
        # ⟨u, diag(1/σ²)·v⟩ = Σᵢ uᵢ·vᵢ/σᵢ²
        temp = self.create_vec()
        temp.pointwiseMult(self.inv_diagonal, v)
        result = u.dot(temp)
        temp.destroy()
        return result

    def __del__(self):
        """Clean up PETSc resources."""
        if hasattr(self, "diagonal"):
            self.diagonal.destroy()
        if hasattr(self, "inv_diagonal"):
            self.inv_diagonal.destroy()


class DenseCovariance(CovarianceMatrix):
    """Dense covariance matrix stored as PETSc Mat.

    Use when the covariance has significant off-diagonal structure
    but the problem size is manageable (typically < 10,000 DoFs).

    Storage: O(n²)
    Apply: O(n²)
    Inverse: Precomputed via Cholesky or stored explicitly

    Examples
    --------
    >>> # Create from correlation matrix and variances
    >>> corr = np.eye(100)  # Identity correlation
    >>> variances = np.ones(100) * 2.0
    >>> B = DenseCovariance.from_correlation(
    ...     MPI.COMM_WORLD, corr, variances
    ... )
    """

    def __init__(
        self,
        comm: MPI.Comm,
        mat: PETSc.Mat,
        inverse_method: str = "cholesky",
    ):
        """Initialize dense covariance from PETSc matrix.

        Parameters
        ----------
        comm : MPI.Comm
            MPI communicator
        mat : PETSc.Mat
            Covariance matrix (must be symmetric positive definite)
        inverse_method : str, optional
            Method for computing inverse ('cholesky', 'lu', 'explicit')
        """
        size = mat.getSize()[0]
        super().__init__(comm, size)

        self.mat = mat
        self.inverse_method = inverse_method
        self._setup_inverse()

    def _setup_inverse(self):
        """Setup inverse application using Cholesky factorization."""
        if self.inverse_method == "cholesky":
            # Create KSP solver with Cholesky factorization
            self.ksp = PETSc.KSP().create(comm=self.comm)
            self.ksp.setOperators(self.mat)
            self.ksp.setType(PETSc.KSP.Type.PREONLY)
            pc = self.ksp.getPC()
            pc.setType(PETSc.PC.Type.CHOLESKY)
            self.ksp.setUp()
        else:
            raise NotImplementedError(
                f"Inverse method '{self.inverse_method}' not implemented"
            )

    def apply(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply dense covariance: out = C·v."""
        if out is None:
            out = self.create_vec()

        self.mat.mult(v, out)
        return out

    def apply_inverse(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply inverse via Cholesky solve: out = C⁻¹·v."""
        if out is None:
            out = self.create_vec()

        # Solve C·out = v
        self.ksp.solve(v, out)

        if self.ksp.getConvergedReason() < 0:
            raise RuntimeError(
                f"Covariance inverse solve failed: reason {self.ksp.getConvergedReason()}"
            )

        return out

    def sqrt_apply(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply Cholesky factor: out = L·v where C = L·Lᵀ."""
        # This requires extracting the Cholesky factor from the PC
        # For now, use eigendecomposition approach
        raise NotImplementedError(
            "sqrt_apply for DenseCovariance requires eigendecomposition"
        )

    @classmethod
    def from_correlation(
        cls,
        comm: MPI.Comm,
        correlation: np.ndarray,
        variances: np.ndarray,
        inverse_method: str = "cholesky",
    ) -> "DenseCovariance":
        """Construct covariance from correlation matrix and variances.

        C = D·Corr·D where D = diag(σ₁, σ₂, ..., σₙ)

        Parameters
        ----------
        comm : MPI.Comm
            MPI communicator
        correlation : np.ndarray, shape (n, n)
            Correlation matrix (entries in [-1, 1])
        variances : np.ndarray, shape (n,)
            Variance for each component (σ²)
        inverse_method : str, optional
            Inversion method

        Returns
        -------
        DenseCovariance
        """
        # Check inputs
        n = len(variances)
        assert correlation.shape == (n, n), "Correlation must be square"
        assert np.allclose(correlation, correlation.T), "Correlation must be symmetric"

        # Construct C = D·Corr·D
        std_devs = np.sqrt(variances)
        cov = correlation * np.outer(std_devs, std_devs)

        # Create PETSc Mat
        mat = PETSc.Mat().create(comm=comm)
        mat.setSizes((n, n))
        mat.setType(PETSc.Mat.Type.AIJ)
        mat.setUp()

        # Fill matrix (only on rank 0 for simplicity)
        if comm.rank == 0:
            for i in range(n):
                mat.setValues(i, list(range(n)), cov[i, :])

        mat.assemblyBegin()
        mat.assemblyEnd()

        return cls(comm, mat, inverse_method)

    def __del__(self):
        """Clean up PETSc resources."""
        if hasattr(self, "ksp"):
            self.ksp.destroy()
        if hasattr(self, "mat"):
            self.mat.destroy()


class ImplicitCovariance(CovarianceMatrix):
    """Implicit covariance via inverse application.

    Instead of storing C explicitly, store C⁻¹ or a way to apply C⁻¹.
    This is efficient when:
    1. C⁻¹ is sparse (e.g., precision matrix from SPDE)
    2. Problem size is large (> 10,000 DoFs)

    Common pattern: C = (αI - β∇²)⁻ᵏ (Matérn covariance)
    Then C⁻¹ = (αI - β∇²)ᵏ is sparse and easy to apply.

    Examples
    --------
    >>> # Precision matrix for background covariance
    >>> # C⁻¹ = αI + β·L where L is the graph Laplacian
    >>> alpha, beta = 1.0, 0.1
    >>> precision = alpha * identity + beta * laplacian
    >>> B = ImplicitCovariance(
    ...     comm, precision, inverse_is_explicit=True
    ... )
    """

    def __init__(
        self,
        comm: MPI.Comm,
        precision_mat: PETSc.Mat,
        inverse_is_explicit: bool = True,
    ):
        """Initialize implicit covariance.

        Parameters
        ----------
        comm : MPI.Comm
            MPI communicator
        precision_mat : PETSc.Mat
            The precision matrix C⁻¹ (if inverse_is_explicit=True)
            or the covariance matrix C (if inverse_is_explicit=False)
        inverse_is_explicit : bool, optional
            If True, precision_mat = C⁻¹ (fast inverse application)
            If False, precision_mat = C (requires solver for inverse)
        """
        size = precision_mat.getSize()[0]
        super().__init__(comm, size)

        self.precision_mat = precision_mat
        self.inverse_is_explicit = inverse_is_explicit

        self._preferred_pc = PETSc.PC.Type.HYPRE

        if not inverse_is_explicit:
            # Need to setup solver for C⁻¹·v
            self.ksp_inverse = PETSc.KSP().create(comm=comm)
            self.ksp_inverse.setOperators(precision_mat)
            self.ksp_inverse.setType(PETSc.KSP.Type.CG)
            pc = self.ksp_inverse.getPC()
            self._set_preconditioner(pc)
            self.ksp_inverse.setUp()

        # Setup solver for C·v (always need this)
        self.ksp_forward = PETSc.KSP().create(comm=comm)
        self.ksp_forward.setOperators(precision_mat)
        self.ksp_forward.setType(PETSc.KSP.Type.CG)
        pc_fwd = self.ksp_forward.getPC()
        self._set_preconditioner(pc_fwd)
        self.ksp_forward.setUp()

    def _set_preconditioner(self, pc: PETSc.PC) -> None:
        """Set PC type with graceful fallback if HYPRE is unavailable."""

        try:
            pc.setType(self._preferred_pc)
        except PETSc.Error:
            # HYPRE not available; fall back to simple Jacobi to keep tests portable
            pc.setType(PETSc.PC.Type.JACOBI)

    def apply(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply covariance: out = C·v."""
        if out is None:
            out = self.create_vec()

        if self.inverse_is_explicit:
            # C·v requires solving C⁻¹·out = v
            self.ksp_forward.solve(v, out)

            if self.ksp_forward.getConvergedReason() < 0:
                raise RuntimeError(
                    f"Covariance application failed: {self.ksp_forward.getConvergedReason()}"
                )
        else:
            # C is stored explicitly
            self.precision_mat.mult(v, out)

        return out

    def apply_inverse(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply precision: out = C⁻¹·v."""
        if out is None:
            out = self.create_vec()

        if self.inverse_is_explicit:
            # C⁻¹ is stored explicitly
            self.precision_mat.mult(v, out)
        else:
            # Need to solve C·out = v
            self.ksp_inverse.solve(v, out)

            if self.ksp_inverse.getConvergedReason() < 0:
                raise RuntimeError(
                    f"Precision application failed: {self.ksp_inverse.getConvergedReason()}"
                )

        return out

    def inner_product_inv(self, u: PETSc.Vec, v: PETSc.Vec) -> float:
        """Optimized ⟨u, C⁻¹·v⟩ when C⁻¹ is explicit."""
        if self.inverse_is_explicit:
            # Direct matrix-vector product
            temp = self.create_vec()
            self.precision_mat.mult(v, temp)
            result = u.dot(temp)
            temp.destroy()
            return result
        else:
            # Fall back to default implementation
            return super().inner_product_inv(u, v)

    def __del__(self):
        """Clean up PETSc resources."""
        if hasattr(self, "ksp_forward"):
            self.ksp_forward.destroy()
        if hasattr(self, "ksp_inverse"):
            self.ksp_inverse.destroy()
        if hasattr(self, "precision_mat"):
            self.precision_mat.destroy()


# Utility functions for constructing common covariance types


def create_observation_covariance(
    comm: MPI.Comm,
    n_obs: int,
    obs_std: float,
) -> DiagonalCovariance:
    """Create diagonal observation error covariance.

    R = σ²·I where σ is the observation standard deviation.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator
    n_obs : int
        Number of observations
    obs_std : float
        Observation standard deviation (σ)

    Returns
    -------
    DiagonalCovariance
        Observation error covariance R
    """
    variance = obs_std**2
    return DiagonalCovariance(comm, size=n_obs, variance=variance)


def create_background_covariance_from_ensemble(
    comm: MPI.Comm,
    ensemble: list[PETSc.Vec],
    inflation_factor: float = 1.0,
) -> DenseCovariance:
    """Create background covariance from ensemble of states.

    B = (1/N-1) Σᵢ (xᵢ - x̄)(xᵢ - x̄)ᵀ

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator
    ensemble : list of PETSc.Vec
        Ensemble members
    inflation_factor : float, optional
        Multiplicative inflation (α > 1 increases variance)

    Returns
    -------
    DenseCovariance
        Background error covariance B

    Notes
    -----
    This gathers the ensemble to rank 0 and computes the covariance.
    Not suitable for very large state spaces.
    """
    n_ensemble = len(ensemble)
    size = ensemble[0].getSize()

    # Gather ensemble to rank 0
    ensemble_array = None
    cov = None
    if comm.rank == 0:
        ensemble_array = np.zeros((n_ensemble, size))

    for i, member in enumerate(ensemble):
        full_vec = _gather_vec_to_root(member, comm)
        if comm.rank == 0 and full_vec is not None:
            ensemble_array[i, :] = full_vec

    if comm.rank == 0:
        mean = np.mean(ensemble_array, axis=0)
        centered = ensemble_array - mean[np.newaxis, :]
        cov = (centered.T @ centered) / (n_ensemble - 1)
        cov *= inflation_factor

    # Create PETSc Mat
    mat = PETSc.Mat().create(comm=comm)
    mat.setSizes((size, size))
    mat.setType(PETSc.Mat.Type.AIJ)
    mat.setUp()

    if comm.rank == 0 and cov is not None:
        for i in range(size):
            mat.setValues(i, list(range(size)), cov[i, :])

    mat.assemblyBegin()
    mat.assemblyEnd()

    return DenseCovariance(comm, mat)


def check_covariance_symmetry(C: CovarianceMatrix, tol: float = 1e-10) -> bool:
    """Return True if covariance matrix is symmetric.

    Tests: ⟨u, C·v⟩ = ⟨v, C·u⟩ for random vectors.

    Parameters
    ----------
    C : CovarianceMatrix
        Covariance matrix to test
    tol : float
        Tolerance for symmetry check

    Returns
    -------
    bool
        True if symmetric within tolerance
    """
    # Create random test vectors
    u = C.create_vec()
    v = C.create_vec()

    u.setRandom()
    v.setRandom()

    # Compute ⟨u, C·v⟩
    Cv = C.apply(v)
    lhs = u.dot(Cv)

    # Compute ⟨v, C·u⟩
    Cu = C.apply(u)
    rhs = v.dot(Cu)

    # Clean up
    Cv.destroy()
    Cu.destroy()
    u.destroy()
    v.destroy()

    return abs(lhs - rhs) < tol


def check_inverse_consistency(C: CovarianceMatrix, tol: float = 1e-8) -> bool:
    """Return True if C·C⁻¹ ≈ I.

    Parameters
    ----------
    C : CovarianceMatrix
        Covariance matrix to test
    tol : float
        Tolerance for identity check

    Returns
    -------
    bool
        True if C·C⁻¹·v ≈ v for random v
    """
    v = C.create_vec()
    v.setRandom()

    # Compute C·C⁻¹·v
    C_inv_v = C.apply_inverse(v)
    C_C_inv_v = C.apply(C_inv_v)

    # Check if result equals v
    C_C_inv_v.axpy(-1.0, v)  # C_C_inv_v -= v
    error = C_C_inv_v.norm()
    v_norm = v.norm()

    # Clean up
    C_inv_v.destroy()
    C_C_inv_v.destroy()
    v.destroy()

    return error / v_norm < tol
def _gather_vec_to_root(vec: PETSc.Vec, comm: MPI.Comm, root: int = 0) -> Optional[np.ndarray]:
    """Gather a distributed PETSc Vec onto the root rank as a flat numpy array."""

    start, end = vec.getOwnershipRange()
    local = vec.getArray().copy()
    local_sizes = comm.gather(end - start, root=root)
    local_arrays = comm.gather(local, root=root)

    if comm.rank != root:
        return None

    total = sum(local_sizes)
    dtype = local.dtype if local.size else float
    full = np.empty(total, dtype=dtype)
    offset = 0
    for chunk, size in zip(local_arrays, local_sizes):
        full[offset : offset + size] = chunk
        offset += size
    return full

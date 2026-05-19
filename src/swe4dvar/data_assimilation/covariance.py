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

    def __init__(self, comm: MPI.Comm, size: int, template_vec: Optional[PETSc.Vec] = None):
        """Initialize covariance matrix.

        Parameters
        ----------
        comm : MPI.Comm
            MPI communicator
        size : int
            Global size of the covariance matrix
        template_vec : PETSc.Vec, optional
            Template vector to match partitioning. If provided, the covariance
            will use the same MPI partitioning as this vector. If None, PETSc
            will automatically determine partitioning (may not match FEM vectors).
        """
        self.size = size

        # Determine comm + local size for this rank.
        # When a template vec is provided the covariance MUST live on the same
        # communicator (COMM_SELF obs vectors vs. COMM_WORLD covariance was the
        # np=2 bug at covariance.py:193). Template's getComm() wins over the
        # `comm` argument.
        #
        # petsc4py.PETSc.Vec.getComm() returns a `petsc4py.PETSc.Comm`, which
        # does NOT have `.allreduce` / `.Get_size` / etc. used by this class
        # (e.g. min_eigenvalue). Convert to mpi4py via .tompi4py() so self.comm
        # is always an mpi4py.MPI.Comm regardless of the provenance.
        if template_vec is not None:
            raw_comm = template_vec.getComm()
            self.comm = raw_comm.tompi4py() if hasattr(raw_comm, "tompi4py") else raw_comm
            self.local_size = template_vec.getLocalSize()
            self.ownership_range = template_vec.getOwnershipRange()
        else:
            # Mirror the conversion done in the template_vec branch: callers
            # sometimes pass PETSc.COMM_SELF/PETSc.COMM_WORLD (petsc4py.PETSc.Comm)
            # which lacks .allreduce/.Get_size used by this class. Convert via
            # .tompi4py() when available. Bug surfaced at Vista 703191 augmented
            # DC-WME (BlockDiagonalCovariance constructed with PETSc.COMM_SELF
            # by the augmented prototype) hitting AttributeError on min_eigenvalue.
            self.comm = comm.tompi4py() if hasattr(comm, "tompi4py") else comm
            # Let PETSc decide partitioning automatically
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

    def inner_product(
        self, v1: PETSc.Vec, v2: PETSc.Vec, inverse: bool = False
    ) -> float:
        """
        Compute C-weighted inner product: ⟨v1, C·v2⟩ or ⟨v1, C⁻¹·v2⟩.

        Args:
            v1: First vector
            v2: Second vector
            inverse: If True, use C⁻¹ instead of C

        Returns:
            Weighted inner product
        """
        if inverse:
            Cv2 = self.apply_inverse(v2)
        else:
            Cv2 = self.apply(v2)
        return v1.dot(Cv2)

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
        template_vec: Optional[PETSc.Vec] = None,
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
        template_vec : PETSc.Vec, optional
            Template vector for MPI partitioning. If provided, covariance will
            use same partitioning as this vector (important for FEM meshes).

        Notes
        -----
        Exactly one of variance or diagonal must be provided.
        """
        super().__init__(comm, size, template_vec=template_vec)

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

    def apply_sqrt_inverse(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply inverse square root: out = C^(-1/2)·v = diag(1/σ₁, 1/σ₂, ...)·v.

        For diagonal covariance C = diag(σ₁², σ₂², ..., σₙ²),
        C^(-1/2) = diag(1/σ₁, 1/σ₂, ..., 1/σₙ).

        Parameters
        ----------
        v : PETSc.Vec
            Input vector
        out : PETSc.Vec, optional
            Output vector (created if None)

        Returns
        -------
        PETSc.Vec
            Result of C^(-1/2)·v
        """
        if out is None:
            out = self.create_vec()

        # Compute 1/sqrt(diagonal)
        inv_sqrt_diag = self.diagonal.duplicate()
        self.diagonal.copy(inv_sqrt_diag)
        inv_sqrt_diag.sqrtabs()  # Element-wise sqrt
        inv_sqrt_diag.reciprocal()  # Element-wise 1/x

        out.pointwiseMult(inv_sqrt_diag, v)
        inv_sqrt_diag.destroy()
        return out

    def inner_product_inv(self, u: PETSc.Vec, v: PETSc.Vec) -> float:
        """Optimized ⟨u, C⁻¹·v⟩ for diagonal case."""
        # ⟨u, diag(1/σ²)·v⟩ = Σᵢ uᵢ·vᵢ/σᵢ²
        temp = self.create_vec()
        temp.pointwiseMult(self.inv_diagonal, v)
        result = u.dot(temp)
        temp.destroy()
        return result

    def min_eigenvalue(self) -> float:
        """Return minimum eigenvalue (minimum diagonal entry).

        For a diagonal matrix, eigenvalues are the diagonal entries.

        Returns
        -------
        float
            Minimum eigenvalue across all MPI ranks.
        """
        local_min = self.diagonal.min()[1]  # [0] is location, [1] is value
        global_min = self.comm.allreduce(local_min, op=MPI.MIN)
        return global_min

    def max_eigenvalue(self) -> float:
        """Return maximum eigenvalue (maximum diagonal entry).

        For a diagonal matrix, eigenvalues are the diagonal entries.

        Returns
        -------
        float
            Maximum eigenvalue across all MPI ranks.
        """
        local_max = self.diagonal.max()[1]  # [0] is location, [1] is value
        global_max = self.comm.allreduce(local_max, op=MPI.MAX)
        return global_max

    @classmethod
    def from_function_space(
        cls,
        function_space,
        variance: float,
        comm: MPI.Comm = None,
    ):
        """Create diagonal covariance matching a DOLFINx function space distribution.

        This ensures the covariance vector has the same MPI distribution as DOFs
        in the function space, which is required for proper parallel operation.

        Parameters
        ----------
        function_space : dolfinx.fem.FunctionSpace
            DOLFINx function space to match distribution of
        variance : float
            Uniform variance for all entries
        comm : MPI.Comm, optional
            MPI communicator (defaults to function space's communicator)

        Returns
        -------
        DiagonalCovariance
            Covariance with distribution matching the function space
        """
        from dolfinx import la

        if comm is None:
            comm = function_space.mesh.comm

        size = function_space.dofmap.index_map.size_global
        local_size = function_space.dofmap.index_map.size_local

        # Create instance with standard constructor
        # We need to override the diagonal vector creation
        instance = cls.__new__(cls)
        instance.comm = comm
        instance.size = size
        instance.local_size = local_size
        instance.ownership_range = (
            function_space.dofmap.index_map.local_range[0],
            function_space.dofmap.index_map.local_range[1],
        )

        # Create diagonal vector using function space's index_map
        from swe4dvar.utils.compat import create_petsc_vector_from_map as _cvm_
        instance.diagonal = _cvm_(
            function_space.dofmap.index_map,
            function_space.dofmap.index_map_bs,
        )
        instance.diagonal.set(variance)
        instance.diagonal.assemble()

        # Precompute inverse
        instance.inv_diagonal = instance.diagonal.duplicate()
        instance.diagonal.copy(instance.inv_diagonal)
        instance.inv_diagonal.reciprocal()

        return instance

    def __del__(self):
        """Clean up PETSc resources."""
        if hasattr(self, "diagonal"):
            self.diagonal.destroy()
        if hasattr(self, "inv_diagonal"):
            self.inv_diagonal.destroy()


class ScaledCovariance(CovarianceMatrix):
    """Scaled wrapper for an existing covariance matrix.

    Represents C_scaled = α · C where α is a scale factor and C is the
    underlying covariance matrix.

    This is useful for the WME formulation where L_wme = R/N, representing
    the reduced variance from averaging N observations.

    Operations:
    - apply(v) = α · C·v
    - apply_inverse(v) = (1/α) · C⁻¹·v

    Parameters
    ----------
    base_cov : CovarianceMatrix
        The underlying covariance matrix C.
    scale_factor : float
        The scale factor α.

    Examples
    --------
    >>> R = DiagonalCovariance(comm, size=100, variance=0.01)
    >>> L_wme = ScaledCovariance(R, scale_factor=1.0/24)  # 24 observations
    """

    def __init__(self, base_cov: CovarianceMatrix, scale_factor: float):
        """Initialize scaled covariance wrapper.

        Parameters
        ----------
        base_cov : CovarianceMatrix
            The underlying covariance matrix.
        scale_factor : float
            Scale factor α (L_scaled = α * L_base).
        """
        # Create a template vector from base covariance to ensure matching partitioning
        template = base_cov.create_vec()
        super().__init__(base_cov.comm, base_cov.size, template_vec=template)
        template.destroy()
        self.base_cov = base_cov
        self.scale_factor = scale_factor

    def apply(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply scaled covariance: out = α·C·v."""
        result = self.base_cov.apply(v, out)
        result.scale(self.scale_factor)
        return result

    def apply_inverse(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply scaled inverse: out = (1/α)·C⁻¹·v."""
        result = self.base_cov.apply_inverse(v, out)
        result.scale(1.0 / self.scale_factor)
        return result

    def sqrt_apply(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply scaled square root: out = √α·C^(1/2)·v."""
        result = self.base_cov.sqrt_apply(v, out)
        result.scale(np.sqrt(self.scale_factor))
        return result

    def min_eigenvalue(self) -> float:
        """Return minimum eigenvalue (scaled)."""
        return self.scale_factor * self.base_cov.min_eigenvalue()

    def max_eigenvalue(self) -> float:
        """Return maximum eigenvalue (scaled)."""
        return self.scale_factor * self.base_cov.max_eigenvalue()


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
        elif self.inverse_method == "iterative":
            # Iterative CG with Jacobi PC — avoids MUMPS factorization
            # entirely. Used for DC-WME's L_wme (3172771 segfaulted in
            # MUMPS MatSolveTranspose at W3 with the default cholesky
            # path; the factorization was reused across cycling windows
            # and the cached state went stale). L_wme is small (~581²)
            # and well-conditioned, so CG+Jacobi converges in <200 iter.
            self.ksp = PETSc.KSP().create(comm=self.comm)
            self.ksp.setOperators(self.mat)
            self.ksp.setType(PETSc.KSP.Type.CG)
            pc = self.ksp.getPC()
            pc.setType(PETSc.PC.Type.JACOBI)
            self.ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
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

    def min_eigenvalue(self) -> float:
        """Return minimum eigenvalue of the dense covariance matrix."""
        if not hasattr(self, '_min_eigenvalue'):
            # Extract dense matrix to numpy and compute eigenvalues
            n = self.size
            dense = np.zeros((n, n))
            rstart, rend = self.mat.getOwnershipRange()
            for i in range(rstart, rend):
                cols, vals = self.mat.getRow(i)
                for c, v in zip(cols, vals):
                    dense[i, c] = v
            eigvals = np.linalg.eigvalsh(dense)
            # Guard against tiny negative eigenvalues from numerical noise
            self._min_eigenvalue = float(max(eigvals[0], eigvals[-1] * 1e-14))
        return self._min_eigenvalue

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


class PrecisionBasedCovariance(CovarianceMatrix):
    """Implicit covariance via precision matrix.

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
    >>> B = PrecisionBasedCovariance(
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


class FullCovariance(CovarianceMatrix):
    """
    Full covariance matrix with explicit storage.

    Stores C as PETSc.Mat and uses KSP for inverse operations.
    """

    def __init__(self, matrix: PETSc.Mat, comm: MPI.Comm = None):
        """
        Initialize full covariance.

        Args:
            matrix: Covariance matrix (must be SPD)
            comm: MPI communicator
        """
        if comm is None:
            comm = matrix.getComm()
        super().__init__(comm, matrix.size[0])
        self.matrix = matrix

        # KSP solver for inverse operations
        self.ksp = PETSc.KSP().create(comm=self.comm)
        self.ksp.setOperators(self.matrix)
        self.ksp.setType(PETSc.KSP.Type.CG)
        self.ksp.getPC().setType(PETSc.PC.Type.JACOBI)

    def apply(self, v: PETSc.Vec) -> PETSc.Vec:
        """Matrix-vector product: w = C·v."""
        w = v.duplicate()
        self.matrix.mult(v, w)
        return w

    def apply_inverse(self, v: PETSc.Vec) -> PETSc.Vec:
        """Solve linear system: w = C⁻¹·v."""
        w = v.duplicate()
        self.ksp.solve(v, w)
        return w


class ImplicitCovariance(CovarianceMatrix):
    """
    Implicit covariance via operator.

    Defines C through an operator (e.g., differential operator)
    without explicit matrix storage.

    Example: C = (I - α∇²)⁻¹ (Matérn covariance)
    """

    def __init__(self, operator, inverse_operator, size: int, comm: MPI.Comm = None):
        """
        Initialize implicit covariance.

        Args:
            operator: Callable that applies C
            inverse_operator: Callable that applies C⁻¹
            size: Dimension
            comm: MPI communicator
        """
        if comm is None:
            comm = MPI.COMM_WORLD
        super().__init__(comm, size)
        self._operator = operator
        self._inverse_operator = inverse_operator

    def apply(self, v: PETSc.Vec) -> PETSc.Vec:
        """Apply operator."""
        return self._operator(v)

    def apply_inverse(self, v: PETSc.Vec) -> PETSc.Vec:
        """Apply inverse operator."""
        return self._inverse_operator(v)


class MaternCovariance(ImplicitCovariance):
    """
    Matérn covariance via SPDE approach.

    C = (κ² - ∇²)⁻ᵛ

    Implements efficient sampling and inverse via FEM.
    """

    def __init__(
        self,
        function_space,
        correlation_length: float,
        variance: float = 1.0,
        nu: float = 1.5,
        comm: MPI.Comm = None,
    ):
        """
        Initialize Matérn covariance.

        Args:
            function_space: FEniCSx function space
            correlation_length: Correlation length ℓ
            variance: Marginal variance σ²
            nu: Smoothness parameter
            comm: MPI communicator
        """
        self.V = function_space
        self.ell = correlation_length
        self.sigma2 = variance
        self.nu = nu

        # Build differential operator (κ² - ∇²)
        self._build_spde_operator()

        # Initialize base class with operators
        super().__init__(
            operator=self._apply_covariance,
            inverse_operator=self._apply_precision,
            size=function_space.dofmap.index_map.size_global,
            comm=comm,
        )

    def _build_spde_operator(self):
        """
        Build SPDE operator matrices.

        Assembles M (mass) and K (stiffness) matrices for
        (κ²M + K)u = f  ⟺  u = C·f
        """
        from dolfinx import fem
        from dolfinx.fem.petsc import assemble_matrix
        import ufl

        # κ = sqrt(8*nu) / ell for Matérn covariance
        kappa = np.sqrt(8.0 * self.nu) / self.ell

        # Define trial and test functions
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # Assemble mass matrix M: ∫ u·v dx
        mass_form = fem.form(ufl.inner(u, v) * ufl.dx)
        self.M = assemble_matrix(mass_form)
        self.M.assemble()

        # Assemble stiffness matrix K: ∫ ∇u·∇v dx
        stiffness_form = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
        self.K = assemble_matrix(stiffness_form)
        self.K.assemble()

        # Build operator matrix: L = κ²M + K
        self.L = self.M.copy()
        self.L.scale(kappa**2)
        self.L.axpy(1.0, self.K)  # L = κ²M + K
        self.L.assemble()

        # Create KSP solver for applying covariance (solving L·u = M·v)
        self.ksp_cov = PETSc.KSP().create(comm=self.comm)
        self.ksp_cov.setOperators(self.L)
        self.ksp_cov.setType(PETSc.KSP.Type.CG)
        pc = self.ksp_cov.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        self.ksp_cov.setUp()

    def _apply_covariance(self, v: PETSc.Vec) -> PETSc.Vec:
        """Solve (κ²M + K)u = M·v to apply C."""
        # Create RHS: b = M·v
        b = v.duplicate()
        self.M.mult(v, b)

        # Solve L·u = b where L = κ²M + K
        u = v.duplicate()
        self.ksp_cov.solve(b, u)

        if self.ksp_cov.getConvergedReason() < 0:
            raise RuntimeError(
                f"Matérn covariance solve failed: {self.ksp_cov.getConvergedReason()}"
            )

        b.destroy()
        return u

    def _apply_precision(self, v: PETSc.Vec) -> PETSc.Vec:
        """Apply precision Q = C⁻¹ via (κ²M + K)·M⁻¹."""
        # For Matérn: Q = C⁻¹ = (κ²M + K)·M⁻¹
        # So Q·v = (κ²M + K)·(M⁻¹·v)
        # We approximate M⁻¹ by solving M·w = v

        # Step 1: Solve M·w = v for w
        w = v.duplicate()
        ksp_mass = PETSc.KSP().create(comm=self.comm)
        ksp_mass.setOperators(self.M)
        ksp_mass.setType(PETSc.KSP.Type.CG)
        ksp_mass.getPC().setType(PETSc.PC.Type.JACOBI)
        ksp_mass.solve(v, w)

        if ksp_mass.getConvergedReason() < 0:
            raise RuntimeError(
                f"Mass matrix solve failed: {ksp_mass.getConvergedReason()}"
            )

        # Step 2: Apply L·w = (κ²M + K)·w
        result = v.duplicate()
        self.L.mult(w, result)

        # Clean up
        w.destroy()
        ksp_mass.destroy()

        return result

    def sqrt_apply(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """
        Sample from Matérn field: u = sqrt(C)·ξ where ξ ~ N(0,I).

        Solves (κ²M + K)u = M^{1/2}·ξ

        Parameters
        ----------
        v : PETSc.Vec
            Input vector (typically standard normal)
        out : PETSc.Vec, optional
            Output vector (created if None)

        Returns
        -------
        PETSc.Vec
            Result of C^{1/2}·v
        """
        # For Matérn covariance: C = (κ²M + K)⁻¹
        # We need C^{1/2}·v
        # Using the rational approximation or by solving:
        # (κ²M + K)·u = M^{1/2}·v

        # First, approximate M^{1/2}·v by lumped mass matrix approach
        # For simplicity, we use diagonal scaling based on mass matrix diagonal
        mass_diag = self.M.getDiagonal()
        mass_sqrt = mass_diag.duplicate()
        mass_diag.copy(mass_sqrt)
        mass_sqrt.sqrtabs()  # Element-wise sqrt

        # Compute b = M^{1/2}·v (approximate)
        b = v.duplicate()
        b.pointwiseMult(mass_sqrt, v)

        # Solve (κ²M + K)·u = b
        if out is None:
            out = v.duplicate()
        self.ksp_cov.solve(b, out)

        if self.ksp_cov.getConvergedReason() < 0:
            raise RuntimeError(
                f"Matérn sqrt sampling failed: {self.ksp_cov.getConvergedReason()}"
            )

        # Clean up
        mass_diag.destroy()
        mass_sqrt.destroy()
        b.destroy()

        return out


class BlockDiagonalCovariance(CovarianceMatrix):
    """
    Block diagonal covariance for mixed function spaces.

    C = diag(C₁, C₂, ..., Cₙ)

    Useful for (H, u, v) systems where each component
    has independent covariance structure.
    """

    def __init__(self, blocks: list[CovarianceMatrix], comm: MPI.Comm = None):
        """
        Initialize block diagonal covariance.

        Args:
            blocks: List of covariance matrices for each block
            comm: MPI communicator
        """
        if comm is None:
            comm = blocks[0].comm if blocks else MPI.COMM_WORLD
        total_size = sum(block.size for block in blocks)
        super().__init__(comm, total_size)
        self.blocks = blocks

        # Block offsets
        self.offsets = [0]
        for block in blocks:
            self.offsets.append(self.offsets[-1] + block.size)

    def apply(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply each block independently."""
        if out is None:
            out = self.create_vec()

        # Get arrays for input and output (read-only for v, writable for out)
        v_array = v.getArray(readonly=True)
        out_array = out.getArray()

        # Apply each block covariance to its corresponding sub-vector
        for i, block in enumerate(self.blocks):
            start = self.offsets[i]
            end = self.offsets[i + 1]

            # Extract sub-vector for this block
            v_sub = block.create_vec()
            v_sub_array = v_sub.getArray()
            v_sub_array[:] = v_array[start:end]
            v_sub.assemble()

            # Apply block covariance
            out_sub = block.apply(v_sub)

            # Copy result back to output
            out_sub_array = out_sub.getArray(readonly=True)
            out_array[start:end] = out_sub_array

            # Clean up
            v_sub.destroy()
            out_sub.destroy()

        # Assemble output (no need to restore arrays in petsc4py)
        out.assemble()

        return out

    def apply_inverse(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """Apply inverse of each block independently."""
        if out is None:
            out = self.create_vec()

        # Get arrays for input and output (read-only for v, writable for out)
        v_array = v.getArray(readonly=True)
        out_array = out.getArray()

        # Apply inverse of each block covariance to its corresponding sub-vector
        for i, block in enumerate(self.blocks):
            start = self.offsets[i]
            end = self.offsets[i + 1]

            # Extract sub-vector for this block
            v_sub = block.create_vec()
            v_sub_array = v_sub.getArray()
            v_sub_array[:] = v_array[start:end]
            v_sub.assemble()

            # Apply block inverse covariance
            out_sub = block.apply_inverse(v_sub)

            # Copy result back to output
            out_sub_array = out_sub.getArray(readonly=True)
            out_array[start:end] = out_sub_array

            # Clean up
            v_sub.destroy()
            out_sub.destroy()

        # Assemble output (no need to restore arrays in petsc4py)
        out.assemble()

        return out

    def min_eigenvalue(self) -> float:
        return min(block.min_eigenvalue() for block in self.blocks)

    def max_eigenvalue(self) -> float:
        return max(block.max_eigenvalue() for block in self.blocks)


class EnsembleCovariance(CovarianceMatrix):
    """
    Covariance matrix represented by ensemble of samples.

    C = (1/(N-1)) Σᵢ (xᵢ - x̄)(xᵢ - x̄)^T

    This is an efficient representation for covariance when you have
    ensemble samples but don't want to store the full dense matrix.

    Attributes
    ----------
    samples : List[PETSc.Vec]
        Ensemble members.
    mean : PETSc.Vec
        Ensemble mean.
    anomalies : List[PETSc.Vec]
        Centered ensemble (xᵢ - x̄).
    """

    def __init__(self, samples: List[PETSc.Vec], comm: MPI.Comm = None):
        """
        Initialize from ensemble samples.

        Parameters
        ----------
        samples : List[PETSc.Vec]
            Ensemble members.
        comm : MPI.Comm, optional
            MPI communicator (inferred from samples if not provided).
        """
        if not samples:
            raise ValueError("Cannot create EnsembleCovariance with empty samples")

        size = samples[0].getSize()
        if comm is None:
            comm = samples[0].getComm()

        super().__init__(comm, size)

        self.samples = samples
        self.n_samples = len(samples)

        # Compute ensemble mean
        self.mean = samples[0].duplicate()
        self.mean.zeroEntries()
        for s in samples:
            self.mean.axpy(1.0, s)
        self.mean.scale(1.0 / self.n_samples)

        # Compute anomalies
        self.anomalies = []
        for s in samples:
            a = s.duplicate()
            s.copy(a)
            a.axpy(-1.0, self.mean)
            self.anomalies.append(a)

    def apply(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """
        Apply covariance: C·v = (1/(N-1)) Σᵢ aᵢ(aᵢ^T·v).

        Parameters
        ----------
        v : PETSc.Vec
            Input vector.
        out : PETSc.Vec, optional
            Output vector (created if None).

        Returns
        -------
        PETSc.Vec
            C·v.
        """
        if out is None:
            out = self.create_vec()

        out.zeroEntries()

        for a in self.anomalies:
            coeff = a.dot(v)
            out.axpy(coeff, a)

        out.scale(1.0 / (self.n_samples - 1))
        return out

    def apply_inverse(self, v: PETSc.Vec, out: Optional[PETSc.Vec] = None) -> PETSc.Vec:
        """
        Apply inverse covariance using pseudo-inverse.

        Uses regularized inverse: (C + εI)^{-1}.

        Parameters
        ----------
        v : PETSc.Vec
            Input vector.
        out : PETSc.Vec, optional
            Output vector (created if None).

        Returns
        -------
        PETSc.Vec
            C^{-1}·v (regularized).
        """
        if out is None:
            out = self.create_vec()

        # For ensemble covariance, use iterative solver or pseudo-inverse
        # Here we use a simple regularized approach

        epsilon = 1e-6  # Regularization

        # Build the system matrix in ensemble space
        n = self.n_samples
        A = np.zeros((n, n))

        for i, ai in enumerate(self.anomalies):
            for j, aj in enumerate(self.anomalies):
                A[i, j] = ai.dot(aj) / (n - 1)

        # Regularize
        A += epsilon * np.eye(n)

        # Project v onto anomaly space
        b = np.array([a.dot(v) for a in self.anomalies])

        # Solve
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(A, b, rcond=None)[0]

        # Reconstruct
        out.zeroEntries()
        for i, a in enumerate(self.anomalies):
            out.axpy(c[i] / (n - 1), a)

        return out

    def min_eigenvalue(self) -> float:
        """Estimate minimum eigenvalue."""
        n = self.n_samples

        # Build covariance in ensemble space
        A = np.zeros((n, n))
        for i, ai in enumerate(self.anomalies):
            for j, aj in enumerate(self.anomalies):
                A[i, j] = ai.dot(aj) / (n - 1)

        eigvals = np.linalg.eigvalsh(A)
        return max(eigvals.min(), 1e-12)

    def max_eigenvalue(self) -> float:
        """Estimate maximum eigenvalue."""
        n = self.n_samples

        A = np.zeros((n, n))
        for i, ai in enumerate(self.anomalies):
            for j, aj in enumerate(self.anomalies):
                A[i, j] = ai.dot(aj) / (n - 1)

        eigvals = np.linalg.eigvalsh(A)
        return eigvals.max()


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


def _gather_vec_to_root(
    vec: PETSc.Vec, comm: MPI.Comm, root: int = 0
) -> Optional[np.ndarray]:
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

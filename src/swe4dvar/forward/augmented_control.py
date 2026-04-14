"""Augmented-control forward wrappers and parameter-sensitivity providers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
try:
    from mpi4py import MPI
except ModuleNotFoundError:  # pragma: no cover - import-light path for tests/docs
    class _SerialComm:
        def Get_size(self):
            return 1

        def Get_rank(self):
            return 0

        def Barrier(self):
            return None

    class _SerialMPI:
        COMM_WORLD = _SerialComm()

    MPI = _SerialMPI()
try:
    from petsc4py import PETSc
except ModuleNotFoundError:  # pragma: no cover - import-light path for tests/docs
    PETSc = None

from swe4dvar.control import ControlLayout, ControlVector
try:
    from swe4dvar.physics.forcing import ParametricWindForcing
except ModuleNotFoundError:  # pragma: no cover - import-light path for tests/docs
    ParametricWindForcing = None


@dataclass
class ParameterSensitivityBundle:
    """Timestep-wise state sensitivities with respect to theta."""

    parameter_names: list[str]
    sensitivities: list[list[PETSc.Vec]]
    derivative_mode: str = "unknown"

    @property
    def parameter_size(self) -> int:
        return len(self.parameter_names)


class ParameterController:
    """Abstract parameter-injection interface for augmented controls."""

    parameter_names: tuple[str, ...] = tuple()
    derivative_mode: str = "none"

    def bind(self, problem, solver) -> None:
        """Prepare any solver-level data structures needed by the controller."""

    def apply(self, problem, solver, theta: np.ndarray) -> None:
        raise NotImplementedError

    def parameter_size(self) -> int:
        return len(self.parameter_names)


class ParameterSensitivityProvider(ParameterController):
    """Explicit residual-derivative provider for augmented parameter controls."""

    derivative_mode: str = "residual_level"

    def compute_timestep_parameter_jacobian(self, problem, solver) -> list[PETSc.Vec]:
        raise NotImplementedError


class LowDimWindController(ParameterController):
    """Inject low-dimensional wind parameters through a parametric forcing model."""

    parameter_names = ("track_shift_km", "intensity_bias", "timing_offset_s")
    derivative_mode = "full_model_finite_difference"

    def __init__(self, forcing: ParametricWindForcing):
        if ParametricWindForcing is None:
            raise ModuleNotFoundError("Wind parameter controls require the forward physics stack")
        self.forcing = forcing

    def apply(self, problem, solver, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != self.parameter_size():
            raise ValueError(
                f"LowDimWindController expects {self.parameter_size()} parameters, got {theta.size}"
            )
        self.forcing.set_parameters(theta)
        if hasattr(problem, "forcing") and problem.forcing is not self.forcing:
            problem.forcing = self.forcing
        if hasattr(problem, "V") and problem.V is not None:
            self.forcing.set_V(
                problem.V.sub(0).collapse()[0] if hasattr(problem.V, "sub") else problem.V
            )
        if hasattr(problem, "update_boundary"):
            problem.update_boundary()


class ManningsBasisController(ParameterSensitivityProvider):
    """Manning's-n parameterization using a bounded exponential basis map.

    Supports two basis types:
      - "gaussian": Gaussian RBF basis with row-sum normalization (original).
      - "kl": Karhunen-Loève expansion from a squared-exponential covariance kernel.
        KL modes are orthogonal by construction, which avoids the collinearity
        that defeats the Gaussian basis for multi-parameter recovery.

    The Manning map is always: n(θ) = clip(n_ref ⊙ exp(B·θ), n_min, n_max)
    Only the construction of B differs between basis types.
    """

    derivative_mode = "semi_analytic_residual"

    # Default threshold for switching from dense eigendecomposition to
    # matrix-free sparse eigsh.  Configurable via kl_max_dense_nodes.
    _KL_MAX_DENSE_NODES_DEFAULT = 5000

    def __init__(
        self,
        *,
        basis_shape: tuple[int, int] = (3, 2),
        basis_width_fraction: float = 0.35,
        n_bounds: tuple[float, float] = (0.01, 0.08),
        reference_n: Optional[float] = None,
        basis_type: str = "gaussian",
        kl_correlation_length: Optional[float] = None,
        kl_prior_std: float = 0.5,
        kl_n_modes: Optional[int] = None,
        kl_max_dense_nodes: Optional[int] = None,
    ):
        self.basis_type = basis_type
        if basis_type not in ("gaussian", "kl"):
            raise ValueError(f"basis_type must be 'gaussian' or 'kl', got {basis_type!r}")

        # Gaussian-specific parameters
        self.basis_shape = tuple(int(v) for v in basis_shape)
        self.basis_width_fraction = float(basis_width_fraction)

        # KL-specific parameters
        self.kl_correlation_length = (
            None if kl_correlation_length is None else float(kl_correlation_length)
        )
        self.kl_prior_std = float(kl_prior_std)
        self.kl_n_modes = None if kl_n_modes is None else int(kl_n_modes)
        self.kl_max_dense_nodes = int(
            kl_max_dense_nodes if kl_max_dense_nodes is not None
            else self._KL_MAX_DENSE_NODES_DEFAULT
        )

        # Shared parameters
        self.n_bounds = (float(n_bounds[0]), float(n_bounds[1]))
        self.reference_n = None if reference_n is None else float(reference_n)

        # Parameter names are set after basis construction for KL (n_modes may be
        # auto-determined from the spectrum). For Gaussian, set them now.
        if basis_type == "gaussian":
            n_params = self.basis_shape[0] * self.basis_shape[1]
        else:
            if self.kl_n_modes is None:
                import warnings
                warnings.warn(
                    "KL basis with kl_n_modes=None (auto) will determine parameter count "
                    "only at bind() time.  parameter_size() returns 0 until then, which "
                    "breaks AugmentedForwardModelWrapper if it reads theta_size before "
                    "bind().  Set kl_n_modes explicitly to avoid this.",
                    stacklevel=2,
                )
            n_params = self.kl_n_modes if self.kl_n_modes is not None else 0
        self.parameter_names = tuple(f"mannings_basis_{j}" for j in range(n_params))
        self._bound_problem = None
        self._bound_solver = None
        self._basis_matrix: Optional[np.ndarray] = None
        self._base_field_array: Optional[np.ndarray] = None
        self._mannings_field_function = None
        self._dmannings_dtheta_functions = []
        self._basis_functions = []
        self._residual_derivative_forms = []

    def bind(self, problem, solver) -> None:
        if PETSc is None:
            raise ModuleNotFoundError("petsc4py is required for Manning sensitivity binding")
        if self._bound_solver is solver and self._bound_problem is problem:
            return

        from dolfinx import fem
        import ufl

        V_scalar = solver.V.sub(0).collapse()[0]
        coords = V_scalar.tabulate_dof_coordinates()[:, :2]
        if self.basis_type == "kl":
            basis_matrix = self._build_kl_basis(coords)
            # Update parameter names now that n_modes is resolved from the spectrum
            self.parameter_names = tuple(
                f"mannings_kl_{j}" for j in range(basis_matrix.shape[1])
            )
        else:
            basis_matrix = self._build_gaussian_basis(coords)
        base_field = self._resolve_reference_field(problem, V_scalar)

        basis_functions = []
        for j in range(basis_matrix.shape[1]):
            basis_fun = fem.Function(V_scalar)
            basis_fun.x.array[:] = basis_matrix[:, j]
            basis_fun.x.scatter_forward()
            basis_functions.append(basis_fun)

        mannings_field_fun = fem.Function(V_scalar)
        mannings_field_fun.x.array[:] = base_field
        mannings_field_fun.x.scatter_forward()

        dmannings_dtheta_functions = []
        for _ in range(basis_matrix.shape[1]):
            deriv_fun = fem.Function(V_scalar)
            deriv_fun.x.array[:] = 0.0
            deriv_fun.x.scatter_forward()
            dmannings_dtheta_functions.append(deriv_fun)

        problem.TAU = mannings_field_fun
        problem.TAU_const = mannings_field_fun
        problem.mannings_n_parametric = mannings_field_fun

        self._basis_matrix = basis_matrix
        self._base_field_array = base_field
        self._basis_functions = basis_functions
        self._mannings_field_function = mannings_field_fun
        self._dmannings_dtheta_functions = dmannings_dtheta_functions
        self._bound_problem = problem
        self._bound_solver = solver

        solver.parameter_sensitivity_provider = self
        solver.init_weak_form()

        self._residual_derivative_forms = [
            fem.form(ufl.derivative(solver.F, mannings_field_fun, deriv_fun))
            for deriv_fun in self._dmannings_dtheta_functions
        ]

    def apply(self, problem, solver, theta: np.ndarray) -> None:
        if PETSc is None:
            raise ModuleNotFoundError("petsc4py is required for Manning parameter controls")
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != self.parameter_size():
            raise ValueError(
                f"ManningsBasisController expects {self.parameter_size()} parameters, got {theta.size}"
            )
        if self._bound_solver is not solver or self._bound_problem is not problem:
            self.bind(problem, solver)
        raw_field = self._base_field_array * np.exp(self._basis_matrix @ theta)
        n_min, n_max = self.n_bounds
        clipped_field = np.clip(raw_field, n_min, n_max)
        self._mannings_field_function.x.array[:] = clipped_field
        self._mannings_field_function.x.scatter_forward()

        active_mask = np.logical_and(raw_field > n_min, raw_field < n_max).astype(float)
        for j, deriv_fun in enumerate(self._dmannings_dtheta_functions):
            deriv_fun.x.array[:] = raw_field * self._basis_matrix[:, j] * active_mask
            deriv_fun.x.scatter_forward()

    def compute_timestep_parameter_jacobian(self, problem, solver) -> list[PETSc.Vec]:
        if PETSc is None:
            raise ModuleNotFoundError("petsc4py is required for Manning sensitivity assembly")
        if self._bound_solver is not solver or self._bound_problem is not problem:
            raise RuntimeError("ManningsBasisController must be bound to the active solver")

        from dolfinx import fem

        derivative_vectors = []
        for form in self._residual_derivative_forms:
            vec = fem.petsc.create_vector(form)
            with vec.localForm() as loc_vec:
                loc_vec.set(0.0)
            fem.petsc.assemble_vector(vec, form)
            vec.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )
            vec.assemble()
            derivative_vectors.append(vec)
        return derivative_vectors

    def evaluate_field(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate Manning's-n field in coefficient space for diagnostics/tests."""
        if self._basis_matrix is None or self._base_field_array is None:
            raise RuntimeError("Controller must be bound before evaluating the Manning field")
        theta = np.asarray(theta, dtype=float).reshape(-1)
        raw = self._base_field_array * np.exp(self._basis_matrix @ theta)
        n_min, n_max = self.n_bounds
        return np.clip(raw, n_min, n_max)

    def _resolve_reference_field(self, problem, V_scalar) -> np.ndarray:
        if hasattr(problem, "mannings_n") and getattr(problem, "mannings_n") is not None:
            field = np.asarray(problem.mannings_n.x.array, dtype=float).copy()
        elif self.reference_n is not None:
            field = np.full(V_scalar.dofmap.index_map.size_local, self.reference_n, dtype=float)
        elif hasattr(problem, "TAU_const") and hasattr(problem.TAU_const, "value"):
            field = np.full(
                V_scalar.dofmap.index_map.size_local,
                float(problem.TAU_const.value),
                dtype=float,
            )
        elif hasattr(problem, "TAU") and isinstance(problem.TAU, (float, int)):
            field = np.full(
                V_scalar.dofmap.index_map.size_local,
                float(problem.TAU),
                dtype=float,
            )
        else:
            field = np.full(V_scalar.dofmap.index_map.size_local, 0.02, dtype=float)
        n_min, n_max = self.n_bounds
        return np.clip(field, n_min, n_max)

    def _build_gaussian_basis(self, coords: np.ndarray) -> np.ndarray:
        nx, ny = self.basis_shape
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        centers_x = np.linspace(x_min, x_max, nx)
        centers_y = np.linspace(y_min, y_max, ny)
        centers = np.array([(x, y) for y in centers_y for x in centers_x], dtype=float)
        width = self.basis_width_fraction * max(x_max - x_min, y_max - y_min)
        width = max(width, 1e-8)
        basis = np.zeros((coords.shape[0], centers.shape[0]), dtype=float)
        for j, center in enumerate(centers):
            dist_sq = np.sum((coords - center) ** 2, axis=1)
            basis[:, j] = np.exp(-0.5 * dist_sq / (width**2))
        row_sums = np.maximum(np.sum(basis, axis=1, keepdims=True), 1e-30)
        return basis / row_sums

    def _build_kl_basis(self, coords: np.ndarray) -> np.ndarray:
        """Build a Karhunen-Loève expansion basis from a squared-exponential kernel.

        Constructs the covariance kernel C[i,j] = σ² exp(-||x_i - x_j||² / (2L²))
        and computes the leading eigenpairs.

        For n_nodes > 5000, uses scipy.sparse.linalg.eigsh with a matrix-free
        linear operator to avoid building the full dense covariance matrix.
        For smaller meshes, uses dense numpy.linalg.eigh.

        Parameters are taken from self.kl_correlation_length, self.kl_prior_std,
        and self.kl_n_modes (set at __init__).
        """
        import logging
        logger = logging.getLogger(__name__)

        n_nodes = coords.shape[0]

        # Resolve correlation length: default to 1/4 of domain extent
        if self.kl_correlation_length is not None:
            L = self.kl_correlation_length
        else:
            extent = np.max(coords, axis=0) - np.min(coords, axis=0)
            L = 0.25 * np.max(extent)
        L = max(L, 1e-30)

        sigma = self.kl_prior_std

        # Resolve number of modes (needed upfront for sparse path)
        n_modes_requested = self.kl_n_modes

        # Memory estimate for the dense path:
        #   diff (n,n,d) + dist_sq (n,n) + C (n,n) + eigh output (n,n)
        #   ≈ 4 * n² * 8 bytes (for d=2)
        dense_peak_bytes = 4 * n_nodes * n_nodes * 8
        use_dense = n_nodes <= self.kl_max_dense_nodes

        if use_dense:
            logger.info(
                "KL basis: using dense eigendecomposition for %d nodes "
                "(est. peak %.0f MB, threshold %d nodes)",
                n_nodes, dense_peak_bytes / 1e6, self.kl_max_dense_nodes,
            )
        else:
            logger.info(
                "KL basis: using matrix-free sparse eigsh for %d nodes "
                "(dense would require %.0f MB, threshold %d nodes)",
                n_nodes, dense_peak_bytes / 1e6, self.kl_max_dense_nodes,
            )

        if use_dense:
            # Dense path: build full covariance and use eigh
            eigenvalues, eigenvectors = self._kl_dense_eigen(coords, sigma, L)

            # Discard numerically non-positive modes
            positive_mask = eigenvalues > 1e-12 * max(eigenvalues[0], 1e-30)
            n_positive = int(np.sum(positive_mask))

            if n_positive == 0:
                raise ValueError(
                    "KL basis: no positive eigenvalues found. Check that the covariance "
                    f"kernel parameters are reasonable (L={L:.4f}, σ={sigma:.4f}, "
                    f"n_nodes={n_nodes})"
                )

            if n_modes_requested is not None:
                n_modes = n_modes_requested
                if n_modes > n_positive:
                    raise ValueError(
                        f"KL basis: requested {n_modes} modes but only {n_positive} "
                        f"positive eigenvalues exist"
                    )
            else:
                cumulative = np.cumsum(eigenvalues[:n_positive])
                total = cumulative[-1]
                n_modes = int(np.searchsorted(cumulative, 0.99 * total)) + 1
                n_modes = max(1, min(n_modes, n_positive))

            retained_evals = eigenvalues[:n_modes]
            retained_evecs = eigenvectors[:, :n_modes]
        else:
            # Sparse path: matrix-free eigsh for large meshes
            if n_modes_requested is None:
                # For auto mode on large meshes, request a generous number and
                # truncate by variance afterward
                n_modes_requested = min(50, n_nodes - 2)
            eigenvalues, eigenvectors = self._kl_sparse_eigen(
                coords, sigma, L, n_modes_requested,
            )

            positive_mask = eigenvalues > 1e-12 * max(eigenvalues[0], 1e-30)
            n_positive = int(np.sum(positive_mask))
            if n_positive == 0:
                raise ValueError(
                    "KL basis: no positive eigenvalues from sparse solver"
                )

            if self.kl_n_modes is not None:
                n_modes = min(self.kl_n_modes, n_positive)
            else:
                cumulative = np.cumsum(eigenvalues[:n_positive])
                total = cumulative[-1]
                n_modes = int(np.searchsorted(cumulative, 0.99 * total)) + 1
                n_modes = max(1, min(n_modes, n_positive))

            retained_evals = eigenvalues[:n_modes]
            retained_evecs = eigenvectors[:, :n_modes]

        # Build basis matrix: B[:, i] = √λ_i · φ_i
        basis = retained_evecs * np.sqrt(retained_evals)[np.newaxis, :]

        if not np.all(np.isfinite(basis)):
            raise ValueError(
                "KL basis: non-finite values in basis matrix. "
                f"Eigenvalue range: [{retained_evals[-1]:.6e}, {retained_evals[0]:.6e}]"
            )

        # Store spectrum info for diagnostics.
        # Use the same positivity threshold (1e-12 * λ_max) for the denominator
        # so that variance_explained is consistent with the mode-retention logic.
        pos_threshold = 1e-12 * max(eigenvalues[0], 1e-30)
        total_positive_variance = float(np.sum(eigenvalues[eigenvalues > pos_threshold]))
        self._kl_eigenvalues = retained_evals.copy()
        self._kl_variance_explained = float(
            np.sum(retained_evals) / max(total_positive_variance, 1e-30)
        )

        return basis

    @staticmethod
    def _kl_dense_eigen(
        coords: np.ndarray, sigma: float, L: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dense eigendecomposition for small meshes (n_nodes <= 5000)."""
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        C = sigma ** 2 * np.exp(-0.5 * dist_sq / L ** 2)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        # Reverse to descending
        return eigenvalues[::-1].copy(), eigenvectors[:, ::-1].copy()

    @staticmethod
    def _kl_sparse_eigen(
        coords: np.ndarray, sigma: float, L: float, n_modes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Matrix-free eigendecomposition for large meshes using scipy.sparse.linalg.eigsh.

        Avoids building the full n×n covariance matrix. Instead, uses a LinearOperator
        that computes C @ v on-the-fly via batched kernel evaluation.
        """
        from scipy.sparse.linalg import LinearOperator, eigsh

        n = coords.shape[0]
        inv_2L2 = 0.5 / (L ** 2)
        s2 = sigma ** 2

        # Batch the matvec to limit memory: process rows in chunks
        chunk_size = min(2000, n)

        def matvec(v):
            result = np.zeros(n, dtype=float)
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk_coords = coords[start:end]                    # (chunk, d)
                diff = chunk_coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (chunk, n, d)
                dist_sq = np.sum(diff ** 2, axis=-1)                # (chunk, n)
                K_chunk = s2 * np.exp(-inv_2L2 * dist_sq)           # (chunk, n)
                result[start:end] = K_chunk @ v
            return result

        C_op = LinearOperator((n, n), matvec=matvec, dtype=float)
        eigenvalues, eigenvectors = eigsh(C_op, k=n_modes, which="LM")

        # eigsh returns in ascending order — reverse
        return eigenvalues[::-1].copy(), eigenvectors[:, ::-1].copy()


class AugmentedForwardModelWrapper:
    """Forward wrapper for packed controls c = [u0; theta]."""

    def __init__(
        self,
        solver,
        problem,
        solver_params: dict,
        *,
        control_layout: Optional[ControlLayout] = None,
        parameter_controller: Optional[ParameterController] = None,
        parameter_fd_steps: Optional[np.ndarray] = None,
        reference_state: Optional[np.ndarray] = None,
        t_start: float = 0.0,
    ):
        self.solver = solver
        self.problem = problem
        self.solver_params = solver_params
        self.dt = problem.dt
        self.nt = problem.nt
        self.t_start = t_start
        self.comm = MPI.COMM_WORLD
        self.var_form = getattr(solver, "var_form", None)
        self.parameter_controller = parameter_controller
        self.model_state_size = (
            solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs
        )

        if self.comm.Get_size() > 1 and parameter_controller is not None:
            raise NotImplementedError(
                "Augmented controls with parameters are currently supported only in serial."
            )

        theta_size = 0 if parameter_controller is None else parameter_controller.parameter_size()
        self.control_layout = control_layout or ControlLayout(
            state_size=self.model_state_size,
            theta_size=theta_size,
            comm=PETSc.COMM_SELF if PETSc is not None else None,
        )
        if self.control_layout.state_size not in (0, self.model_state_size):
            raise ValueError(
                f"control_layout.state_size={self.control_layout.state_size} must be either 0 "
                f"(parameter-only) or the model state size {self.model_state_size}"
            )

        self.reference_state = (
            np.asarray(reference_state, dtype=float).copy()
            if reference_state is not None
            else self.solver.u_n.x.array[: self.model_state_size].copy()
        )
        if self.reference_state.size != self.model_state_size:
            raise ValueError(
                f"reference_state has size {self.reference_state.size}, expected {self.model_state_size}"
            )

        self.parameter_fd_steps = (
            np.asarray(parameter_fd_steps, dtype=float).reshape(-1)
            if parameter_fd_steps is not None
            else np.full(theta_size, 1e-3, dtype=float)
        )
        if theta_size > 0 and self.parameter_fd_steps.size != theta_size:
            raise ValueError(
                f"parameter_fd_steps has size {self.parameter_fd_steps.size}, expected {theta_size}"
            )

        if self.parameter_controller is not None and hasattr(self.parameter_controller, "bind"):
            self.parameter_controller.bind(self.problem, self.solver)

        self._parameter_sensitivity_cache = {}

    def _hash_control(self, control) -> str:
        if isinstance(control, PETSc.Vec):
            arr = control.getArray(readonly=True)
        elif isinstance(control, ControlVector):
            arr = control.pack()
        else:
            arr = np.asarray(control, dtype=float).reshape(-1)
        return hashlib.md5(np.asarray(arr, dtype=float).tobytes()).hexdigest()

    def _coerce_control(self, control) -> ControlVector:
        if isinstance(control, ControlVector):
            return control
        if isinstance(control, PETSc.Vec):
            return self.control_layout.unpack_vec(control)
        return self.control_layout.unpack_array(np.asarray(control, dtype=float))

    def _apply_control(self, control: ControlVector) -> None:
        state = control.u0 if control.u0.size > 0 else self.reference_state
        state = np.asarray(state, dtype=float)
        if state.size != self.model_state_size:
            raise ValueError(
                f"Initial state has size {state.size}, expected {self.model_state_size}"
            )

        if self.parameter_controller is not None and self.control_layout.theta_size > 0:
            self.parameter_controller.apply(self.problem, self.solver, control.theta)

        self.solver.storage.clear()
        self.solver.u_n.x.array[: self.model_state_size] = state
        self.solver.u_n_old.x.array[: self.model_state_size] = state
        self.solver.u.x.array[: self.model_state_size] = state
        self.solver.u_n.x.scatter_forward()
        self.solver.u_n_old.x.scatter_forward()
        self.solver.u.x.scatter_forward()
        self.problem.t = self.t_start
        if hasattr(self.problem, "update_boundary"):
            self.problem.update_boundary()

    def solve(self, control, store_jacobians: bool = True):
        """Run the nonlinear forward model from a packed or logical control."""
        from dolfinx import la

        c = self._coerce_control(control)
        self._apply_control(c)

        store_parameter_derivatives = (
            store_jacobians
            and isinstance(self.parameter_controller, ParameterSensitivityProvider)
        )
        self.solver.time_loop(
            solver_parameters=self.solver_params,
            stations=np.array([[0.0, 0.0, 0.0]]),
            plot_every=9999,
            save_state=True,
            store_jacobians=store_jacobians,
            store_parameter_derivatives=store_parameter_derivatives,
            enable_video=False,
        )

        u_owned_size = (
            self.solver.V.dofmap.index_map.size_local * self.solver.V.dofmap.index_map_bs
        )
        trajectory = []
        for state_array in self.solver.storage.saved_states:
            vec = la.create_petsc_vector(
                self.solver.V.dofmap.index_map,
                self.solver.V.dofmap.index_map_bs,
            )
            vec.setArray(state_array[:u_owned_size])
            vec.assemble()
            trajectory.append(vec)

        jacobians = None
        if store_jacobians and len(self.solver.storage.saved_jacobians) > 0:
            jacobians = self.solver.storage.saved_jacobians.copy()

        return trajectory, jacobians

    def get_timestep_parameter_derivatives(
        self,
        control,
        *,
        base_trajectory=None,
        base_jacobians=None,
    ):
        if self.control_layout.theta_size == 0 or self.parameter_controller is None:
            return None

        if isinstance(self.parameter_controller, ParameterSensitivityProvider):
            if not self.solver.storage.saved_parameter_derivatives:
                self.solve(control, store_jacobians=True)
            return self.solver.storage.saved_parameter_derivatives
        return None

    def _build_parameter_sensitivity_bundle_from_residual_derivatives(
        self,
        control,
        *,
        base_trajectory,
        base_jacobians,
    ) -> Optional[ParameterSensitivityBundle]:
        if base_jacobians is None:
            _, base_jacobians = self.solve(control, store_jacobians=True)

        timestep_derivatives = self.get_timestep_parameter_derivatives(
            control,
            base_trajectory=base_trajectory,
            base_jacobians=base_jacobians,
        )
        if not timestep_derivatives:
            return None

        from swe4dvar.adjoint.tangent_linear import ForcedTangentLinearModel

        tlm = ForcedTangentLinearModel(self, base_trajectory, base_jacobians)
        parameter_names = list(self.parameter_controller.parameter_names)
        sensitivities: list[list[PETSc.Vec]] = []
        for p in range(self.control_layout.theta_size):
            timestep_forcings = [derivatives[p] for derivatives in timestep_derivatives]
            sensitivity_history = tlm.propagate_forcing(
                timestep_forcings,
                end_time=len(base_trajectory) - 1,
            )
            sensitivities.append(sensitivity_history)

        return ParameterSensitivityBundle(
            parameter_names=parameter_names,
            sensitivities=sensitivities,
            derivative_mode=getattr(self.parameter_controller, "derivative_mode", "unknown"),
        )

    def _build_parameter_sensitivity_bundle_via_finite_difference(
        self,
        control,
        *,
        base_trajectory=None,
    ) -> Optional[ParameterSensitivityBundle]:
        c = self._coerce_control(control)
        theta = c.theta.copy()
        state = c.u0.copy() if c.u0.size > 0 else self.reference_state.copy()
        sensitivities: list[list[PETSc.Vec]] = []
        for p, step in enumerate(self.parameter_fd_steps):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[p] += step
            theta_minus[p] -= step

            traj_plus, _ = self.solve(
                ControlVector(u0=state, theta=theta_plus),
                store_jacobians=False,
            )
            traj_minus, _ = self.solve(
                ControlVector(u0=state, theta=theta_minus),
                store_jacobians=False,
            )

            param_history = []
            for vec_plus, vec_minus in zip(traj_plus, traj_minus):
                sens = vec_plus.copy()
                sens.axpy(-1.0, vec_minus)
                sens.scale(1.0 / (2.0 * step))
                param_history.append(sens)

            for vec in traj_plus:
                vec.destroy()
            for vec in traj_minus:
                vec.destroy()

            sensitivities.append(param_history)

        return ParameterSensitivityBundle(
            parameter_names=list(self.parameter_controller.parameter_names),
            sensitivities=sensitivities,
            derivative_mode=getattr(self.parameter_controller, "derivative_mode", "full_model_finite_difference"),
        )

    def get_parameter_sensitivity_bundle(
        self,
        control,
        *,
        base_trajectory=None,
        base_jacobians=None,
    ) -> Optional[ParameterSensitivityBundle]:
        """Return timestep-wise state sensitivities du_k/dtheta_i."""
        if self.control_layout.theta_size == 0 or self.parameter_controller is None:
            return None

        c = self._coerce_control(control)
        cache_key = self._hash_control(c)
        if cache_key in self._parameter_sensitivity_cache:
            return self._parameter_sensitivity_cache[cache_key]

        if base_trajectory is None:
            base_trajectory, base_jacobians = self.solve(c, store_jacobians=True)

        if isinstance(self.parameter_controller, ParameterSensitivityProvider):
            bundle = self._build_parameter_sensitivity_bundle_from_residual_derivatives(
                c,
                base_trajectory=base_trajectory,
                base_jacobians=base_jacobians,
            )
        else:
            bundle = self._build_parameter_sensitivity_bundle_via_finite_difference(
                c,
                base_trajectory=base_trajectory,
            )

        self._parameter_sensitivity_cache[cache_key] = bundle
        return bundle

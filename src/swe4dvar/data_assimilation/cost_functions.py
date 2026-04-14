"""
Cost function implementations for 4D-Var data assimilation.

Implements standard 4D-Var, DC-4DVar, and DC-WME variants
following Spence et al. (2025).

Mathematical Formulations
-------------------------
Standard 4D-Var:
    J(m) = ½⟨m - m_b, B⁻¹(m - m_b)⟩
         + ½ Σ_k ⟨H_k(u_k) - y_k, R_k⁻¹(H_k(u_k) - y_k)⟩

DC-4DVar (Data-Consistent):
    J_DC(m) = J(m) - ½ Σ_k ⟨Q_k(m) - Q_k(m_b), L_k⁻¹(Q_k(m) - Q_k(m_b))⟩

DC-WME (Weighted Mean Error):
    Q_wme,k(m) = (1/√|I_k|) Σ_{j∈I_k} R_j^{-1/2}(H_j(M_{j:0}(m)) - y_j)
        where I_k := { j ∈ I : j ≤ k } and I is the observation index set.
    J_WME(m) = ½⟨m - m_b, B⁻¹(m - m_b)⟩
             + ½ ‖Q_wme(m)‖²
             - ½ ⟨Q_wme(m) - Q_wme(m_b), L_wme⁻¹(Q_wme(m) - Q_wme(m_b))⟩
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Union, Callable
from dataclasses import dataclass
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import hashlib
import time


@dataclass
class CostFunctionResult:
    """Container for cost function evaluation results."""

    value: float
    background_term: float
    observation_term: float
    predictability_term: float = 0.0


class CostFunction(ABC):
    """
    Abstract base class for 4D-Var cost functions.

    Defines the interface for computing cost function value,
    gradient, and Hessian-vector products.

    Attributes
    ----------
    forward_model : ForwardModel
        Forward model M_{k:0} that propagates initial conditions.
    obs_op : ObservationOperator
        Observation operator H_k mapping state to observations.
    B : CovarianceMatrix
        Background error covariance.
    R : CovarianceMatrix
        Observation error covariance.
    """

    def __init__(
        self,
        forward_model,
        observation_operator,
        background_cov,
        observation_cov,
        comm: Optional[MPI.Comm] = None,
        checkpointer=None,
    ):
        """
        Initialize cost function.

        Parameters
        ----------
        forward_model : ForwardModel
            Forward model M_{k:0}.
        observation_operator : ObservationOperator
            Observation operator H_k.
        background_cov : CovarianceMatrix
            Background error covariance B.
        observation_cov : CovarianceMatrix or Dict[int, CovarianceMatrix]
            Observation error covariance R_k. Can be a single covariance
            for all times or a dictionary mapping time indices to covariances.
        comm : MPI.Comm, optional
            MPI communicator. Defaults to MPI.COMM_WORLD.
        checkpointer : CheckpointerBase, optional
            Custom checkpointer for managing forward trajectory storage.
            If provided, overrides the forward model's default storage.
        """
        self.forward_model = forward_model
        self.obs_op = observation_operator
        self.B = background_cov
        self.R = observation_cov
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.checkpointer = checkpointer

        # Cache for forward trajectory and Jacobians
        self._trajectory: Optional[List[PETSc.Vec]] = None
        self._jacobians: Optional[List[PETSc.Mat]] = None
        self._current_control: Optional[PETSc.Vec] = None
        self._control_hash: Optional[str] = None  # Hash for efficient cache checking
        self.runtime_profiler = None

    def _profile_event(self, event: str, **payload) -> None:
        profiler = getattr(self, "runtime_profiler", None)
        if profiler is None:
            return
        try:
            profiler.log_event(event, **payload)
        except Exception:
            pass

    @abstractmethod
    def value(self, m: PETSc.Vec) -> float:
        """
        Compute cost function value J(m).

        Parameters
        ----------
        m : PETSc.Vec
            Control variable (initial condition).

        Returns
        -------
        float
            Cost function value.
        """
        pass

    @abstractmethod
    def gradient(self, m: PETSc.Vec) -> PETSc.Vec:
        """
        Compute gradient ∇J(m) via adjoint method.

        Parameters
        ----------
        m : PETSc.Vec
            Control variable.

        Returns
        -------
        PETSc.Vec
            Gradient vector.
        """
        pass

    def hessian_vector_product(self, m: PETSc.Vec, v: PETSc.Vec) -> PETSc.Vec:
        """
        Compute Hessian-vector product Hv using Gauss-Newton approximation.

        Parameters
        ----------
        m : PETSc.Vec
            Control variable.
        v : PETSc.Vec
            Direction vector.

        Returns
        -------
        PETSc.Vec
            H·v
        """
        raise NotImplementedError("Gauss-Newton Hessian not yet implemented")

    def _get_control_hash(self, m: PETSc.Vec) -> str:
        """
        Compute hash of control vector for caching.

        Uses MD5 hash of the vector's byte representation for efficient
        comparison without needing to allocate temporary vectors.

        Parameters
        ----------
        m : PETSc.Vec
            Control vector.

        Returns
        -------
        str
            MD5 hash string.
        """
        # Make a copy first to handle TAO vectors that may not allow direct array access
        # (TAO vectors can be in a special read-only state)
        try:
            m_bytes = m.getArray().tobytes()
        except Exception:
            # Fall back to copying the vector first
            m_copy = m.copy()
            m_bytes = m_copy.getArray().tobytes()
            m_copy.destroy()
        return hashlib.md5(m_bytes).hexdigest()

    def _run_forward_model(
        self, m: PETSc.Vec, store_jacobians: bool = True
    ) -> Tuple[List[PETSc.Vec], Optional[List[PETSc.Mat]]]:
        """
        Run forward model and cache trajectory.

        Uses hash-based caching to efficiently detect when the same control
        vector is used, avoiding redundant forward solves. This is critical
        for TAO efficiency, where value() and gradient() may be called
        separately for the same point.

        Parameters
        ----------
        m : PETSc.Vec
            Initial condition.
        store_jacobians : bool
            Whether to cache Jacobians for adjoint.

        Returns
        -------
        Tuple[List[PETSc.Vec], Optional[List[PETSc.Mat]]]
            (trajectory, jacobians) tuple.
        """
        # Compute hash for cache lookup
        m_hash = self._get_control_hash(m)
        self._profile_event(
            "cost.run_forward_model.enter",
            control_hash=m_hash,
            store_jacobians=bool(store_jacobians),
        )

        # Check if we can reuse cached trajectory
        if self._control_hash == m_hash and self._trajectory is not None:
            # If jacobians are requested but not cached, re-run
            if store_jacobians and self._jacobians is None:
                refresh_start = time.perf_counter()
                self._trajectory, self._jacobians = self.forward_model.solve(
                    m, store_jacobians=True
                )
                self._profile_event(
                    "cost.run_forward_model.cache_refresh",
                    duration_s=float(time.perf_counter() - refresh_start),
                    trajectory_len=(len(self._trajectory) if self._trajectory is not None else 0),
                    jacobian_len=(len(self._jacobians) if self._jacobians is not None else 0),
                )
            self._profile_event(
                "cost.run_forward_model.cache_hit",
                trajectory_len=(len(self._trajectory) if self._trajectory is not None else 0),
                jacobian_len=(len(self._jacobians) if self._jacobians is not None else 0),
            )
            return self._trajectory, self._jacobians

        # Clear solver storage and old trajectory/Jacobians to free memory.
        # Each eval stores 36 Jacobians at 52K DOFs (~504 MB); without clearing,
        # Armijo backtracking accumulates multiple sets and causes OOM.
        if hasattr(self.forward_model, 'solver') and hasattr(self.forward_model.solver, 'storage'):
            self.forward_model.solver.storage.clear()
        # Also release previous trajectory/Jacobian references so GC can reclaim.
        if self._trajectory is not None:
            for vec in self._trajectory:
                vec.destroy()
            self._trajectory = None
        if self._jacobians is not None:
            for jac in self._jacobians:
                if hasattr(jac, 'destroy'):
                    jac.destroy()
            self._jacobians = None
        # Force garbage collection to reclaim PETSc memory
        import gc
        gc.collect()

        # Enable Newton failure detection during optimization.
        # When Newton diverges, raise exception so cost function returns infinity
        # and TAO's Armijo line search backtracks to a smaller step.
        # The flag must be set on the actual CGImplicit solver (not the wrapper),
        # since time_loop() creates the Newton solver and propagates this flag.
        actual_solver = getattr(self.forward_model, 'solver', self.forward_model)
        actual_solver._raise_on_newton_failure = True

        # Run forward model
        solve_start = time.perf_counter()
        try:
            self._trajectory, self._jacobians = self.forward_model.solve(
                m, store_jacobians=store_jacobians
            )
        finally:
            actual_solver._raise_on_newton_failure = False
        self._profile_event(
            "cost.run_forward_model.solve_end",
            duration_s=float(time.perf_counter() - solve_start),
            trajectory_len=(len(self._trajectory) if self._trajectory is not None else 0),
            jacobian_len=(len(self._jacobians) if self._jacobians is not None else 0),
            store_jacobians=bool(store_jacobians),
        )
        self._current_control = m.copy()
        self._control_hash = m_hash

        # Check for NaN/inf or unreasonably large values in trajectory.
        # Newton divergence may produce finite but garbage states (e.g., huge residuals
        # without NaN if the linear solver diverges and the update is skipped).
        if self._trajectory:
            import numpy as np
            for i, state in enumerate(self._trajectory):
                arr = state.getArray()
                if not np.all(np.isfinite(arr)) or np.max(np.abs(arr)) > 1e4:
                    raise RuntimeError(
                        f"Forward model produced invalid states at timestep {i}: "
                        f"max|state|={np.max(np.abs(arr)):.2e}, "
                        f"has_nan={not np.all(np.isfinite(arr))}"
                    )

        # Warn if Jacobians were requested but not stored
        if store_jacobians and (self._jacobians is None or len(self._jacobians) == 0):
            import warnings
            warnings.warn(
                "Jacobian caching was requested but no Jacobians were stored. "
                "This will cause gradient computation to fail or produce incorrect results. "
                "Ensure your forward model's time_loop is called with store_jacobians=True "
                "and that observation_times are correctly specified.",
                UserWarning,
                stacklevel=3
            )

        return self._trajectory, self._jacobians

    def _get_observation_covariance(self, k: int):
        """Get observation covariance for time index k."""
        if isinstance(self.R, dict):
            return self.R.get(k, self.R.get(0, list(self.R.values())[0]))
        return self.R

    def clear_cache(self):
        """Clear cached trajectory and Jacobians."""
        self._trajectory = None
        self._jacobians = None
        self._current_control = None
        self._control_hash = None


class FourDVarCost(CostFunction):
    """
    Standard 4D-Var cost function.

    J(m) = ½⟨m - m_b, B⁻¹(m - m_b)⟩
         + ½ Σ_k ⟨H_k(u_k) - y_k, R_k⁻¹(H_k(u_k) - y_k)⟩

    The gradient is computed via the adjoint method:
        ∇J(m) = B⁻¹(m - m_b) + λ₀

    where λ₀ is the initial adjoint variable obtained by backward
    integration of the adjoint equations.

    Attributes
    ----------
    m_b : PETSc.Vec
        Background state.
    y_obs : List[PETSc.Vec]
        Observation vectors at each observation time.
    obs_times : List[int]
        Time indices where observations are available.
    """

    def __init__(
        self,
        forward_model,
        observation_operator,
        background_cov,
        observation_cov,
        m_background: PETSc.Vec,
        observations: List[PETSc.Vec],
        obs_times: List[int],
        comm: Optional[MPI.Comm] = None,
        checkpointer=None,
    ):
        """
        Initialize standard 4D-Var cost function.

        Parameters
        ----------
        forward_model : ForwardModel
            Forward model.
        observation_operator : ObservationOperator
            Observation operator.
        background_cov : CovarianceMatrix
            Background covariance B.
        observation_cov : CovarianceMatrix
            Observation covariance R.
        m_background : PETSc.Vec
            Background state m_b.
        observations : List[PETSc.Vec]
            List of observation vectors y_k.
        obs_times : List[int]
            List of observation time indices.
        comm : MPI.Comm, optional
            MPI communicator.
        checkpointer : CheckpointerBase, optional
            Custom checkpointer for managing forward trajectory storage.
            If provided, overrides the forward model's default storage.
        """
        super().__init__(
            forward_model, observation_operator, background_cov, observation_cov, comm, checkpointer
        )
        self.m_b = m_background
        self.y_obs = observations
        self.obs_times = obs_times
        self.gradient_smoother = None  # Optional callable: grad_array -> smoothed_grad_array

        # Validate inputs
        if len(observations) != len(obs_times):
            raise ValueError(
                f"Number of observations ({len(observations)}) must match "
                f"number of observation times ({len(obs_times)})"
            )

    def value(self, m: PETSc.Vec) -> float:
        """
        Compute standard 4D-Var cost.

        J(m) = ½⟨m - m_b, B⁻¹(m - m_b)⟩
             + ½ Σ_k ⟨H_k(u_k) - y_k, R_k⁻¹(H_k(u_k) - y_k)⟩

        Parameters
        ----------
        m : PETSc.Vec
            Control variable (initial condition).

        Returns
        -------
        float
            Cost function value. Returns float('inf') if forward model fails
            (e.g., due to unphysical states), allowing line search to reject the step.
        """
        # Run forward model with error handling for unphysical states
        try:
            trajectory, _ = self._run_forward_model(m, store_jacobians=False)
        except Exception as e:
            # Forward model failed (e.g., negative water depth, NaN)
            # Return infinity so line search rejects this step
            import warnings
            warnings.warn(
                f"Forward model failed during cost evaluation: {e}. "
                "Returning inf to reject line search step.",
                RuntimeWarning,
                stacklevel=2
            )
            return float('inf')

        # Background term: ½⟨m - m_b, B⁻¹(m - m_b)⟩
        background_term = self._compute_background_term(m)

        # Observation term: ½ Σ_k ⟨H_k(u_k) - y_k, R_k⁻¹(H_k(u_k) - y_k)⟩
        observation_term = self._compute_observation_term(trajectory)

        return background_term + observation_term

    def value_detailed(self, m: PETSc.Vec) -> CostFunctionResult:
        """
        Compute cost with detailed breakdown.

        Parameters
        ----------
        m : PETSc.Vec
            Control variable.

        Returns
        -------
        CostFunctionResult
            Result with individual terms.
        """
        trajectory, _ = self._run_forward_model(m, store_jacobians=False)
        background = self._compute_background_term(m)
        observation = self._compute_observation_term(trajectory)

        return CostFunctionResult(
            value=background + observation,
            background_term=background,
            observation_term=observation,
        )

    def _compute_background_term(self, m: PETSc.Vec) -> float:
        """Compute ½⟨m - m_b, B⁻¹(m - m_b)⟩."""
        # Compute deviation from background
        delta_m = m.duplicate()
        delta_m.waxpy(-1.0, self.m_b, m)  # delta_m = m - m_b

        # Apply B⁻¹
        B_inv_delta = self.B.apply_inverse(delta_m)

        # Compute inner product (uses MPI reduction internally)
        result = 0.5 * delta_m.dot(B_inv_delta)

        return result

    def _compute_observation_term(self, trajectory: List[PETSc.Vec]) -> float:
        """Compute ½ Σ_k ⟨H_k(u_k) - y_k, R_k⁻¹(H_k(u_k) - y_k)⟩."""
        total = 0.0

        for i, k in enumerate(self.obs_times):
            # Get state at observation time
            u_k = trajectory[k]

            # Apply observation operator: H_k(u_k)
            Hu_k = self.obs_op.forward(u_k)

            # Compute innovation: d_k = H_k(u_k) - y_k
            d_k = Hu_k.duplicate()
            d_k.waxpy(-1.0, self.y_obs[i], Hu_k)  # d_k = Hu_k - y_k

            # Apply R_k⁻¹
            R_k = self._get_observation_covariance(k)
            R_inv_d = R_k.apply_inverse(d_k)

            # Compute weighted inner product
            total += 0.5 * d_k.dot(R_inv_d)

        return total

    def gradient(self, m: PETSc.Vec) -> PETSc.Vec:
        """
        Compute gradient via adjoint method.

        ∇J(m) = B⁻¹(m - m_b) + λ₀

        Parameters
        ----------
        m : PETSc.Vec
            Control variable.

        Returns
        -------
        PETSc.Vec
            Gradient vector.
        """
        # Run forward model with Jacobian caching
        trajectory, jacobians = self._run_forward_model(m, store_jacobians=True)

        # Compute background gradient: B⁻¹(m - m_b)
        delta_m = m.duplicate()
        delta_m.waxpy(-1.0, self.m_b, m)
        grad_background = self.B.apply_inverse(delta_m)

        # Compute adjoint contribution via backward sweep
        lambda_0 = self._solve_adjoint(trajectory, jacobians)

        # Total gradient: ∇J = B⁻¹(m - m_b) + λ₀
        grad = grad_background.copy()
        grad.axpy(1.0, lambda_0)

        return grad

    def value_gradient(self, m: PETSc.Vec) -> Tuple[float, PETSc.Vec]:
        """
        Compute cost function value and gradient together (efficient for TAO).

        Uses single forward model run + adjoint solve, avoiding the double
        computation that occurs when calling value() then gradient() separately.
        This is the preferred method for TAO optimization callbacks.

        Parameters
        ----------
        m : PETSc.Vec
            Control variable (initial condition).

        Returns
        -------
        Tuple[float, PETSc.Vec]
            (cost_value, gradient_vector)

        Notes
        -----
        The efficiency gain comes from:
        1. Running forward model once with Jacobian caching enabled
        2. Computing both cost terms from the same trajectory
        3. Using the same Jacobians for the adjoint solve

        This avoids the typical pattern where value() and gradient() each
        independently run the forward model, doubling the computation cost.
        """
        # Run forward model once, storing Jacobians for adjoint
        try:
            trajectory, jacobians = self._run_forward_model(m, store_jacobians=True)
        except Exception as e:
            # Forward model failed - return infinity and zero gradient
            import warnings
            warnings.warn(
                f"Forward model failed during value_gradient: {e}. "
                "Returning inf cost and zero gradient.",
                RuntimeWarning,
                stacklevel=2
            )
            # Create zero gradient
            zero_grad = m.duplicate()
            zero_grad.zeroEntries()
            return float('inf'), zero_grad

        # Compute cost value (same as value())
        background_term = self._compute_background_term(m)
        observation_term = self._compute_observation_term(trajectory)
        cost = background_term + observation_term

        # Check for non-finite cost
        if not np.isfinite(cost):
            import warnings
            warnings.warn(
                f"Cost is not finite! background_term={background_term}, "
                f"observation_term={observation_term}, cost={cost}",
                RuntimeWarning,
                stacklevel=2
            )

        # Compute gradient (same as gradient() but reuses trajectory/jacobians)
        delta_m = m.duplicate()
        delta_m.waxpy(-1.0, self.m_b, m)
        grad_background = self.B.apply_inverse(delta_m)

        # Adjoint solve (uses cached jacobians)
        try:
            lambda_0 = self._solve_adjoint(trajectory, jacobians)
        except Exception as e:
            # Adjoint solve failed - return finite cost but background-only gradient
            import warnings
            import traceback
            warnings.warn(
                f"Adjoint solve failed during value_gradient: {e}\n"
                f"Traceback: {traceback.format_exc()}\n"
                "Returning finite cost but background-only gradient.",
                RuntimeWarning,
                stacklevel=2
            )
            # Return finite cost but only background gradient (observation gradient = 0)
            return cost, grad_background

        # Total gradient: ∇J = B⁻¹(m - m_b) + λ₀
        grad = grad_background.copy()
        grad.axpy(1.0, lambda_0)

        # Apply gradient smoothing if configured.
        # Smoothing filters high-frequency components from the adjoint gradient,
        # producing search directions that stay within Newton's convergence basin.
        # This is equivalent to B-preconditioning when B has spatial correlation.
        if self.gradient_smoother is not None:
            arr = grad.getArray().copy()
            arr = self.gradient_smoother(arr)
            grad.setArray(arr)

        return cost, grad

    def compute_background_gradient(self, m: PETSc.Vec) -> PETSc.Vec:
        """
        Compute only the background penalty gradient B⁻¹(m - m_b).

        This is useful when the forward model fails - we can still
        return a valid gradient pointing back toward the background,
        preventing TAO from thinking it has converged.

        Parameters
        ----------
        m : PETSc.Vec
            Current control variable.

        Returns
        -------
        PETSc.Vec
            Background gradient B⁻¹(m - m_b).
        """
        # Compute deviation from background
        delta_m = m.duplicate()
        delta_m.waxpy(-1.0, self.m_b, m)  # delta_m = m - m_b

        # Apply B⁻¹
        grad_background = self.B.apply_inverse(delta_m)

        return grad_background

    def _solve_adjoint(
        self, trajectory: List[PETSc.Vec], jacobians: List[PETSc.Mat]
    ) -> PETSc.Vec:
        """
        Solve adjoint equations backward in time.

        Parameters
        ----------
        trajectory : List[PETSc.Vec]
            Forward trajectory.
        jacobians : List[PETSc.Mat]
            Jacobians from forward solve.

        Returns
        -------
        PETSc.Vec
            Initial adjoint λ₀.
        """
        # Compute observation forcings at each time
        forcing_start = time.perf_counter()
        obs_forcings = self._compute_observation_forcings(trajectory)
        self._profile_event(
            "cost.solve_adjoint.forcings_ready",
            duration_s=float(time.perf_counter() - forcing_start),
            num_forcings=len(obs_forcings),
        )

        # Solve adjoint using implicit adjoint solver
        from ..adjoint.implicit_adjoint import ImplicitAdjointSolver
        from ..utils import get_boundary_dofs

        # Check if forward model has a variational form (for BDF2 time coefficients)
        # Look in multiple places since forward_model may be a wrapper
        variational_form = getattr(self.forward_model, 'var_form', None)
        if variational_form is None and hasattr(self.forward_model, 'solver'):
            # ForwardModelWrapper wraps the actual solver
            variational_form = getattr(self.forward_model.solver, 'var_form', None)

        # Get boundary DOFs for proper adjoint BC handling
        # Only zero boundary DOF gradients when there are STRONG Dirichlet BCs.
        # For DG with weakly-enforced BCs (dirichlet_bcs=[]), all DOFs are
        # part of the control space and should NOT be zeroed.
        bc_dof_indices = None
        problem = getattr(self.forward_model, 'problem', None)
        has_strong_bcs = (
            problem is not None
            and hasattr(problem, 'dirichlet_bcs')
            and len(problem.dirichlet_bcs) > 0
        )
        if has_strong_bcs:
            if hasattr(self.forward_model, 'solver') and hasattr(self.forward_model, 'problem'):
                V = self.forward_model.solver.V
                mesh = self.forward_model.problem.mesh
                boundary_dofs = get_boundary_dofs(V, mesh)
                bc_dof_indices = set(boundary_dofs.tolist())
            elif hasattr(self.forward_model, 'V') and hasattr(self.forward_model, 'mesh'):
                V = self.forward_model.V
                mesh = self.forward_model.mesh
                boundary_dofs = get_boundary_dofs(V, mesh)
                bc_dof_indices = set(boundary_dofs.tolist())

        adjoint_solver = ImplicitAdjointSolver(
            self.forward_model,
            trajectory,
            jacobians,
            self.forward_model.dt,
            variational_form=variational_form,
            bc_dof_indices=bc_dof_indices,  # Pass boundary DOFs for proper adjoint BCs
            flux_formulation=True,  # SWE uses conservative variables Q=[h, h*ux, h*uy]
        )

        # Terminal condition (usually zero)
        terminal = trajectory[-1].duplicate()
        terminal.zeroEntries()

        solve_start = time.perf_counter()
        lambda_0 = adjoint_solver.solve(terminal, obs_forcings)
        self._profile_event(
            "cost.solve_adjoint.solve_end",
            duration_s=float(time.perf_counter() - solve_start),
        )

        return lambda_0

    def _compute_observation_forcings(
        self, trajectory: List[PETSc.Vec]
    ) -> List[Optional[PETSc.Vec]]:
        """
        Compute adjoint forcing from observations at each time.

        Returns H_k^T R_k^{-1} (H_k(u_k) - y_k) at observation times.
        """
        N = len(trajectory)
        forcings = [None] * N

        for i, k in enumerate(self.obs_times):
            u_k = trajectory[k]

            # Forward observation: H_k(u_k)
            Hu_k = self.obs_op.forward(u_k)

            # Innovation: d_k = H_k(u_k) - y_k
            d_k = Hu_k.duplicate()
            d_k.waxpy(-1.0, self.y_obs[i], Hu_k)

            # Apply R_k⁻¹
            R_k = self._get_observation_covariance(k)
            R_inv_d = R_k.apply_inverse(d_k)

            # Apply adjoint observation operator: H_k^T R_k^{-1} d_k
            forcings[k] = self.obs_op.adjoint(R_inv_d)

        return forcings


class DCFourDVarCost(FourDVarCost):
    """
    Data-Consistent 4D-Var (DC-4DVar) cost function.

    J_DC(m) = J(m) - ½ Σ_k ⟨Q_k(m) - Q_k(m_b), L_k⁻¹(Q_k(m) - Q_k(m_b))⟩

    Includes predictability term to prevent assimilation of
    unpredictable small scales. The predictability term acts as
    "targeted unregularization" that reduces the impact of statistical
    bias in directions informed by observations.

    The gradient is:
        ∇J_DC(m) = ∇J(m) - Σ_k DQ_k^T L_k⁻¹(Q_k(m) - Q_k(m_b))

    where DQ_k is the Jacobian of the QoI map.

    Mathematical Background
    -----------------------
    The predicted error covariance L_k represents the push-forward of
    the background covariance through the QoI map:
        L_k = Q_k B Q_k^T

    The predictability assumption requires:
        λ_min(L_k) ≥ γ · λ_max(R_k)

    for some γ > 0, ensuring observations are trusted only when they
    improve upon the model's predicted uncertainty.

    Attributes
    ----------
    qoi_map : QoIMap
        Quantity of Interest map Q_k.
    L_k : Dict[int, CovarianceMatrix]
        Predicted error covariance at each observation time.
    m_b_qoi : Dict[int, PETSc.Vec]
        QoI evaluated at background: Q_k(m_b).
    """

    def __init__(
        self,
        forward_model,
        observation_operator,
        background_cov,
        observation_cov,
        m_background: PETSc.Vec,
        observations: List[PETSc.Vec],
        obs_times: List[int],
        qoi_map=None,
        predicted_cov: Optional[Dict] = None,
        gamma: float = 1.0,
        comm: Optional[MPI.Comm] = None,
        checkpointer=None,
    ):
        """
        Initialize DC-4DVar cost function.

        Parameters
        ----------
        forward_model : ForwardModel
            Forward model.
        observation_operator : ObservationOperator
            Observation operator.
        background_cov : CovarianceMatrix
            Background covariance B.
        observation_cov : CovarianceMatrix
            Observation covariance R.
        m_background : PETSc.Vec
            Background state.
        observations : List[PETSc.Vec]
            Observation vectors.
        obs_times : List[int]
            Observation time indices.
        qoi_map : QoIMap, optional
            Quantity of Interest map. If None, uses StandardQoI.
        predicted_cov : Dict[int, CovarianceMatrix], optional
            Pre-computed predicted covariances L_k. If None, computed
            from background covariance.
        gamma : float
            Scaling factor for predictability check (default 1.0).
        comm : MPI.Comm, optional
            MPI communicator.
        checkpointer : CheckpointerBase, optional
            Custom checkpointer for managing forward trajectory storage.
        """
        super().__init__(
            forward_model,
            observation_operator,
            background_cov,
            observation_cov,
            m_background,
            observations,
            obs_times,
            comm,
            checkpointer,
        )

        # Set QoI map (default to standard: Q_k = H_k ∘ M_{k:0})
        if qoi_map is None:
            from .qoi_maps import StandardQoI

            self.qoi_map = StandardQoI(forward_model, observation_operator)
        else:
            self.qoi_map = qoi_map

        # Predicted error covariance L_k
        self._L_k = predicted_cov
        self.gamma = gamma

        # Cache for background QoI values
        self._m_b_qoi: Optional[Dict[int, PETSc.Vec]] = None

        # Cache for linearized QoI maps
        self._linearized_qoi: Dict[int, object] = {}

    def value(self, m: PETSc.Vec) -> float:
        """
        Compute DC-4DVar cost with predictability term.

        J_DC(m) = J(m) - ½ Σ_k ⟨Q_k(m) - Q_k(m_b), L_k⁻¹(Q_k(m) - Q_k(m_b))⟩

        Parameters
        ----------
        m : PETSc.Vec
            Control variable.

        Returns
        -------
        float
            DC-4DVar cost function value.
        """
        # Compute standard 4D-Var cost
        J_standard = super().value(m)

        # Compute predictability term
        predictability_term = self._compute_predictability_term(m)

        return J_standard - predictability_term

    def value_detailed(self, m: PETSc.Vec) -> CostFunctionResult:
        """
        Compute cost with detailed breakdown.

        Parameters
        ----------
        m : PETSc.Vec
            Control variable.

        Returns
        -------
        CostFunctionResult
            Result with all terms.
        """
        standard_result = super().value_detailed(m)
        predictability = self._compute_predictability_term(m)

        return CostFunctionResult(
            value=standard_result.value - predictability,
            background_term=standard_result.background_term,
            observation_term=standard_result.observation_term,
            predictability_term=predictability,
        )

    def _compute_predictability_term(self, m: PETSc.Vec) -> float:
        """
        Compute ½ Σ_k ⟨Q_k(m) - Q_k(m_b), L_k⁻¹(Q_k(m) - Q_k(m_b))⟩.

        This term penalizes deviations from the background in the
        QoI space, weighted by the inverse predicted covariance.
        """
        # Ensure background QoI is computed
        self._ensure_background_qoi()

        # Ensure predicted covariances are available
        self._ensure_predicted_covariance()

        total = 0.0

        for i, k in enumerate(self.obs_times):
            # Evaluate QoI at current control: Q_k(m)
            Q_k_m = self.qoi_map.evaluate(m, k)

            # Get background QoI: Q_k(m_b)
            Q_k_mb = self._m_b_qoi[k]

            # Compute correction residual: q_k = Q_k(m) - Q_k(m_b)
            q_k = Q_k_m.duplicate()
            q_k.waxpy(-1.0, Q_k_mb, Q_k_m)

            # Apply L_k⁻¹
            L_k = self._get_predicted_covariance(k)
            L_inv_q = L_k.apply_inverse(q_k)

            # Compute weighted inner product
            total += 0.5 * q_k.dot(L_inv_q)

        return total

    def gradient(self, m: PETSc.Vec) -> PETSc.Vec:
        """
        Compute DC-4DVar gradient.

        ∇J_DC(m) = ∇J(m) - Σ_k DQ_k^T L_k⁻¹(Q_k(m) - Q_k(m_b))

        The predictability gradient correction removes the gradient
        contribution from unpredictable components.

        Parameters
        ----------
        m : PETSc.Vec
            Control variable.

        Returns
        -------
        PETSc.Vec
            Gradient vector.
        """
        # Compute standard 4D-Var gradient
        grad_standard = super().gradient(m)

        # Compute predictability gradient correction
        grad_predictability = self._compute_predictability_gradient(m)

        # DC gradient = standard gradient - predictability gradient
        grad_standard.axpy(-1.0, grad_predictability)

        return grad_standard

    def value_gradient(self, m: PETSc.Vec) -> Tuple[float, PETSc.Vec]:
        """
        Compute DC-4DVar cost and gradient together (efficient for TAO).

        Uses single forward model run + adjoint solve for the standard
        4D-Var component, then adds predictability corrections.

        Parameters
        ----------
        m : PETSc.Vec
            Control variable.

        Returns
        -------
        Tuple[float, PETSc.Vec]
            (cost_value, gradient_vector)
        """
        # Get standard 4D-Var value and gradient (uses single forward solve)
        J_standard, grad_standard = super().value_gradient(m)

        # Handle forward model failure
        if not np.isfinite(J_standard):
            return J_standard, grad_standard

        # Compute predictability term
        predictability_term = self._compute_predictability_term(m)

        # Compute predictability gradient correction
        grad_predictability = self._compute_predictability_gradient(m)

        # DC cost = standard cost - predictability term
        cost = J_standard - predictability_term

        # DC gradient = standard gradient - predictability gradient
        grad_standard.axpy(-1.0, grad_predictability)

        return cost, grad_standard

    def _compute_predictability_gradient(self, m: PETSc.Vec) -> PETSc.Vec:
        """
        Compute Σ_k DQ_k^T L_k⁻¹(Q_k(m) - Q_k(m_b)).

        This requires the adjoint of the linearized QoI map.
        """
        self._ensure_background_qoi()
        self._ensure_predicted_covariance()

        # Initialize gradient
        grad = m.duplicate()
        grad.zeroEntries()

        for i, k in enumerate(self.obs_times):
            # Evaluate QoI: Q_k(m)
            Q_k_m = self.qoi_map.evaluate(m, k)

            # Correction residual: q_k = Q_k(m) - Q_k(m_b)
            q_k = Q_k_m.duplicate()
            q_k.waxpy(-1.0, self._m_b_qoi[k], Q_k_m)

            # Apply L_k⁻¹: w_k = L_k⁻¹ q_k
            L_k = self._get_predicted_covariance(k)
            w_k = L_k.apply_inverse(q_k)

            # Linearize QoI at m
            linearized_qoi = self.qoi_map.linearize(m, k)

            # Apply adjoint: DQ_k^T w_k
            grad_k = linearized_qoi.apply_adjoint(w_k)

            # Accumulate
            grad.axpy(1.0, grad_k)

        return grad

    def _ensure_background_qoi(self):
        """Compute and cache QoI at background state."""
        if self._m_b_qoi is None:
            self._m_b_qoi = {}
            for k in self.obs_times:
                self._m_b_qoi[k] = self.qoi_map.evaluate(self.m_b, k)

    def _ensure_predicted_covariance(self):
        """Ensure predicted covariance L_k is available."""
        if self._L_k is None:
            self._L_k = {}
            for k in self.obs_times:
                self._L_k[k] = self._compute_predicted_covariance(k)

    def _get_predicted_covariance(self, k: int):
        """Get predicted covariance for time index k."""
        if isinstance(self._L_k, dict):
            return self._L_k.get(k)
        return self._L_k

    def _compute_predicted_covariance(self, k: int):
        """
        Compute L_k = Q_k B Q_k^T at observation time k.

        The predicted error covariance represents the push-forward
        of background uncertainty through the QoI map.

        For efficiency, we use either:
        1. Monte Carlo estimation (sampling-based)
        2. TLM-based computation (deterministic)

        Parameters
        ----------
        k : int
            Time index.

        Returns
        -------
        CovarianceMatrix
            Predicted error covariance L_k.
        """
        from .qoi_maps import QoICovarianceEstimator

        estimator = QoICovarianceEstimator(self.qoi_map, self.B, num_samples=100)
        return estimator.estimate(self.m_b, k)

    def check_predictability_assumption(self, k: int) -> Tuple[bool, float]:
        """
        Check if predictability assumption holds at time k.

        The predictability assumption requires:
            λ_min(L_k) ≥ γ · λ_max(R_k)

        Parameters
        ----------
        k : int
            Time index.

        Returns
        -------
        Tuple[bool, float]
            (assumption_holds, ratio) where ratio = λ_min(L_k) / λ_max(R_k).
        """
        self._ensure_predicted_covariance()

        L_k = self._get_predicted_covariance(k)
        R_k = self._get_observation_covariance(k)

        # Get eigenvalue bounds
        lambda_min_L = L_k.min_eigenvalue()
        lambda_max_R = R_k.max_eigenvalue()

        ratio = lambda_min_L / lambda_max_R
        holds = ratio >= self.gamma

        return holds, ratio

    def hessian_vector_product(self, m: PETSc.Vec, v: PETSc.Vec) -> PETSc.Vec:
        """
        Compute Gauss-Newton Hessian-vector product for DC-4DVar.

        H_GN · v ≈ B⁻¹·v + Σ_k J_k^T (R_k⁻¹ - L_k⁻¹) J_k · v

        where J_k = DQ_k is the Jacobian of the QoI map.

        Parameters
        ----------
        m : PETSc.Vec
            Control variable.
        v : PETSc.Vec
            Direction vector.

        Returns
        -------
        PETSc.Vec
            H_GN · v.
        """
        self._ensure_predicted_covariance()

        # Background term: B⁻¹·v
        Hv = self.B.apply_inverse(v)

        # Observation terms
        for i, k in enumerate(self.obs_times):
            # Linearize QoI at m
            linearized_qoi = self.qoi_map.linearize(m, k)

            # Forward: J_k · v
            Jv = linearized_qoi.apply(v)

            # Apply (R_k⁻¹ - L_k⁻¹)
            R_k = self._get_observation_covariance(k)
            L_k = self._get_predicted_covariance(k)

            # w = R_k⁻¹ · Jv
            w = R_k.apply_inverse(Jv)

            # w -= L_k⁻¹ · Jv
            L_inv_Jv = L_k.apply_inverse(Jv)
            w.axpy(-1.0, L_inv_Jv)

            # Adjoint: J_k^T · w
            JTw = linearized_qoi.apply_adjoint(w)

            # Accumulate
            Hv.axpy(1.0, JTw)

        return Hv


class DCWMEFourDVarCost(DCFourDVarCost):
    """
    DC-4DVar with Weighted Mean Error QoI.

    Q_wme,k(m) = (1/√|I_k|) Σ_{j∈I_k} R_j^{-1/2} [H_j(M_{j:0}(m)) - y_j]

    J_WME(m) = ½⟨m - m_b, B⁻¹(m - m_b)⟩
             + ½ ‖Q_wme(m)‖²
             - ½ ⟨Q_wme(m) - Q_wme(m_b), L_wme⁻¹(Q_wme(m) - Q_wme(m_b))⟩

    The WME formulation uses cumulative time-averaged innovation as QoI
    for improved stability with sparse observations. Key properties:

    1. Uncertainties decrease at rate proportional to number of observations
    2. Propagating MUD estimate produces unbiased sample mean of observed data
    3. Predictability assumption guaranteed to hold for sufficiently large N

    Attributes
    ----------
    L_wme : CovarianceMatrix
        Predicted covariance for WME QoI.
    """

    def __init__(
        self,
        forward_model,
        observation_operator,
        background_cov,
        observation_cov,
        m_background: PETSc.Vec,
        observations: List[PETSc.Vec],
        obs_times: List[int],
        predicted_cov_wme=None,
        n_l_wme_samples: int = 100,
        auto_inflate_B: bool = True,
        max_inflate_factor: float = None,  # Deprecated: Eq 38 inflation is uncapped
        predictability_gamma: float = 0.1,
        adaptive_gamma: bool = True,
        B_lwme=None,
        nonuniform_inflate: bool = False,
        eigenvalue_boost_target: float = 2.0,
        comm: Optional[MPI.Comm] = None,
        checkpointer=None,
    ):
        """
        Initialize DC-WME cost function.

        Parameters
        ----------
        forward_model : ForwardModel
            Forward model.
        observation_operator : ObservationOperator
            Observation operator.
        background_cov : CovarianceMatrix
            Background covariance B (used for B^{-1} in background term).
        observation_cov : CovarianceMatrix
            Observation covariance R.
        m_background : PETSc.Vec
            Background state.
        observations : List[PETSc.Vec]
            Observation vectors.
        obs_times : List[int]
            Observation time indices.
        predicted_cov_wme : CovarianceMatrix, optional
            Predicted covariance for WME. If None and n_l_wme_samples > 0,
            computes L_wme = J_wme B J_wme^T analytically via adjoint solves.
            If None and n_l_wme_samples == 0, falls back to L_wme = 2R.
        n_l_wme_samples : int, optional
            Controls L_wme computation. Any value > 0 triggers analytical
            computation of L_wme = J_wme B J_wme^T. Set to 0 to use the
            2R fallback. Default 100.
        auto_inflate_B : bool, optional
            If True, estimate the minimum background variance bound from the
            Gram matrix G = J_wme J_wme^T (paper eq. 38) and inflate B if
            needed to satisfy the predictability condition. Default True.
        max_inflate_factor : float, optional
            Deprecated for B inflation (Eq 38 inflation is now uncapped).
            Only used for nonuniform L_wme eigenvalue boosting when
            nonuniform_inflate=True. Default None.
        predictability_gamma : float, optional
            Relaxation parameter γ for the predictability condition (paper eq. 36).
            When adaptive_gamma=True (default), γ is spectrum-relative: the
            eigenvalue floor is γ * λ_max(L_wme), so γ represents the minimum
            fraction of the largest eigenvalue. Max amplification = 1/γ relative
            to the best-conditioned direction.
            When adaptive_gamma=False, γ is an absolute floor: λ_min(L_wme) ≥ γ.
            Default 0.1.
        adaptive_gamma : bool, optional
            If True, use spectrum-relative regularization where the eigenvalue
            floor adapts to the actual L_wme spectrum: γ_eff = γ * λ_max(L_wme).
            This preserves the natural eigenvalue structure for well-conditioned
            directions while stabilizing ill-conditioned ones.
            If False, use absolute floor γ (original behavior). Default True.
        B_lwme : CovarianceMatrix, optional
            Separate covariance for L_wme computation (L_wme = J B_lwme J^T).
            When provided, this is used instead of background_cov for L_wme,
            while background_cov is still used for B^{-1} in the background term.
            This allows using a spatially correlated B for L_wme amplification
            while keeping a well-conditioned diagonal B for the background term.
        comm : MPI.Comm, optional
            MPI communicator.
        checkpointer : CheckpointerBase, optional
            Custom checkpointer for managing forward trajectory storage.
        """
        # Create WME QoI map
        from .qoi_maps import WeightedMeanErrorQoI

        wme_qoi = WeightedMeanErrorQoI(
            forward_model,
            observation_operator,
            observations,
            observation_cov,
            obs_times=obs_times,
        )

        super().__init__(
            forward_model,
            observation_operator,
            background_cov,
            observation_cov,
            m_background,
            observations,
            obs_times,
            qoi_map=wme_qoi,
            predicted_cov=None,
            comm=comm,
            checkpointer=checkpointer,
        )

        # WME-specific predicted covariance
        self._L_wme = predicted_cov_wme
        self._auto_inflate_B = auto_inflate_B
        self._max_inflate_factor = max_inflate_factor  # Only used for nonuniform L_wme regularization
        self._predictability_gamma = predictability_gamma
        self._adaptive_gamma = adaptive_gamma

        # Separate B for L_wme computation (None = use self.B)
        self._B_lwme = B_lwme

        # Non-uniform eigenvalue inflation
        self._nonuniform_inflate = nonuniform_inflate
        self._eigenvalue_boost_target = eigenvalue_boost_target

        # Gram matrix diagnostics (populated during _compute_analytical_L_wme)
        self._gram_eigenvalues = None
        self._variance_bounds = None
        self._b_inflation_factor = 1.0
        self._B_original = None  # Store original B before inflation

        # Cache for WME accumulation
        self._wme_cache: Dict[str, PETSc.Vec] = {}

        # Pre-compute Q_wme(m_b) at initialization (computed once per assimilation window)
        # Note: We use store_jacobians=True because the optimizer typically starts at m_b
        # and will need Jacobians for the initial gradient computation. Storing them now
        # avoids a redundant forward solve.
        self._wme_cache["Q_wme_mb"] = self._compute_wme(self.m_b)

        # Share the trajectory from QoIMap back to cost function's cache
        # This prevents _run_forward_model from re-computing the m_b trajectory
        self._sync_trajectory_from_qoi(self.m_b)

        # Compute proper L_wme analytically if not provided
        if self._L_wme is None and n_l_wme_samples > 0:
            self._L_wme = self._compute_analytical_L_wme()

    def _sync_trajectory_from_qoi(self, m: PETSc.Vec) -> None:
        """
        Sync trajectory from QoIMap cache to cost function's cache.

        After QoIMap computes a trajectory (e.g., in _compute_wme), this method
        retrieves it and populates the cost function's cache so that _run_forward_model
        can reuse it instead of running another forward solve.
        """
        import hashlib

        try:
            m_bytes = m.getArray(readonly=True).tobytes()
        except Exception:
            m_copy = m.copy()
            m_bytes = m_copy.getArray().tobytes()
            m_copy.destroy()
        m_hash = hashlib.md5(m_bytes).hexdigest()

        # Check if QoIMap has the trajectory cached
        if m_hash in self.qoi_map._trajectory_cache:
            cached = self.qoi_map._trajectory_cache[m_hash]
            self._trajectory, self._jacobians = cached
            self._current_control = m.copy()
            self._control_hash = m_hash

    def _share_trajectory_with_qoi(
        self, m: PETSc.Vec, trajectory: List[PETSc.Vec], jacobians: Optional[List]
    ) -> None:
        """
        Share computed trajectory with QoIMap cache to avoid redundant forward solves.

        When _run_forward_model computes a trajectory, this method populates the
        QoIMap's trajectory cache so that subsequent _compute_wme calls can reuse
        the trajectory instead of running another forward solve.

        Parameters
        ----------
        m : PETSc.Vec
            Control variable used for the trajectory.
        trajectory : List[PETSc.Vec]
            Computed trajectory.
        jacobians : Optional[List]
            Optional Jacobians (may be None).
        """
        import hashlib

        # Compute hash matching QoIMap's caching scheme
        try:
            m_bytes = m.getArray(readonly=True).tobytes()
        except Exception:
            m_copy = m.copy()
            m_bytes = m_copy.getArray().tobytes()
            m_copy.destroy()
        m_hash = hashlib.md5(m_bytes).hexdigest()

        # Populate QoIMap's cache
        self.qoi_map._trajectory_cache[m_hash] = (trajectory, jacobians)

    def value(self, m: PETSc.Vec) -> float:
        """
        Compute DC-WME cost.

        J_WME(m) = ½⟨m - m_b, B⁻¹(m - m_b)⟩
                 + ½ ‖Q_wme(m)‖²
                 - ½ ⟨Q_wme(m) - Q_wme(m_b), L_wme⁻¹(Q_wme(m) - Q_wme(m_b))⟩

        Parameters
        ----------
        m : PETSc.Vec
            Control variable.

        Returns
        -------
        float
            DC-WME cost function value. Returns float('inf') if forward model fails.
        """
        try:
            # Background term
            background_term = self._compute_background_term(m)

            # WME data misfit: ½ ‖Q_wme(m)‖²
            Q_wme_m = self._compute_wme(m)
            data_misfit = 0.5 * Q_wme_m.dot(Q_wme_m)

            # Predictability term: ½ ⟨Q_wme(m) - Q_wme(m_b), L⁻¹(...)⟩
            predictability = self._compute_wme_predictability(m, Q_wme_m)

            return background_term + data_misfit - predictability
        except Exception as e:
            # Forward model failed (e.g., negative water depth, NaN)
            import warnings
            warnings.warn(
                f"Forward model failed during DC-WME cost evaluation: {e}. "
                "Returning inf to reject line search step.",
                RuntimeWarning,
                stacklevel=2
            )
            return float('inf')

    def _compute_wme(
        self,
        m: PETSc.Vec,
        store_jacobians: bool = True,
        obs_times_only: bool = False,
    ) -> PETSc.Vec:
        """
        Compute WME QoI: Q_wme(m) = (1/√N) Σ_k R_k^{-1/2}(H_k(u_k) - y_k).

        Parameters
        ----------
        m : PETSc.Vec
            Control variable.
        store_jacobians : bool
            Whether to store Jacobians. Set to False for evaluation-only
            (e.g., computing Q_wme(m_b) at initialization).
        obs_times_only : bool
            If True, only store trajectory at observation times.
            Useful when Jacobians are not needed (saves memory/time).

        Returns
        -------
        PETSc.Vec
            WME vector.
        """
        # Use final observation index K := max(I) for WME evaluation
        k_final = max(self.obs_times)
        return self.qoi_map.evaluate(
            m, k_final, store_jacobians=store_jacobians, obs_times_only=obs_times_only
        )

    def _compute_wme_predictability(self, m: PETSc.Vec, Q_wme_m: PETSc.Vec) -> float:
        """
        Compute WME predictability term.

        ½ ⟨Q_wme(m) - Q_wme(m_b), L_wme⁻¹(Q_wme(m) - Q_wme(m_b))⟩
        """
        # Ensure background WME is computed
        if "Q_wme_mb" not in self._wme_cache:
            self._wme_cache["Q_wme_mb"] = self._compute_wme(self.m_b)

        Q_wme_mb = self._wme_cache["Q_wme_mb"]

        # Correction residual
        delta_Q = Q_wme_m.duplicate()
        delta_Q.waxpy(-1.0, Q_wme_mb, Q_wme_m)

        # Ensure L_wme is available
        self._ensure_wme_predicted_covariance()

        # Apply L_wme⁻¹
        L_inv_delta = self._L_wme.apply_inverse(delta_Q)

        return 0.5 * delta_Q.dot(L_inv_delta)

    def _compute_gram_and_inflate_B(self, adjoint_vectors: list) -> None:
        """Compute Gram matrix G = J_wme J_wme^T and inflate B per Eq 38.

        Following Spence et al. (2025), Eq 38, the Gram matrix
        G[i,j] = a_i^T a_j (where a_i = J_wme^T e_i) determines the
        minimum background variance for the predictability condition:

            σ²_b ≥ γ / λ_min(G)

        Since J_wme absorbs R^{-1/2} and 1/√N, this is equivalent to
        σ²_b ≥ γ · σ_obs⁴ / (N · λ_min(Q_k Q_k^T)) from the paper.

        B is inflated by exactly the required factor — no cap applied.
        The inflation applies to B for L_wme computation. If no separate
        B_lwme exists, inflation applies to self.B (which also affects
        the background term).

        Parameters
        ----------
        adjoint_vectors : list of PETSc.Vec
            The adjoint vectors a_i = J_wme^T e_i already computed.
        """
        n_obs = len(adjoint_vectors)

        # Form Gram matrix G[i,j] = a_i^T a_j (no B involved)
        G = np.zeros((n_obs, n_obs))
        for i in range(n_obs):
            for j in range(i, n_obs):
                val = adjoint_vectors[i].dot(adjoint_vectors[j])
                G[i, j] = val
                G[j, i] = val

        gram_eigvals = np.linalg.eigvalsh(G)
        self._gram_eigenvalues = gram_eigvals

        lambda_min_G = max(gram_eigvals.min(), 1e-30)
        print(f"  [Eq 38] Gram matrix G ({n_obs}×{n_obs}):")
        print(f"    λ_min(G) = {gram_eigvals.min():.6e}")
        print(f"    λ_max(G) = {gram_eigvals.max():.6e}")
        print(f"    condition = {gram_eigvals.max() / lambda_min_G:.2f}")
        print(f"    rank (>1e-10) = {np.sum(gram_eigvals > 1e-10)}/{n_obs}")

        # Compute required σ²_b for the configured γ
        gamma_inflate = self._predictability_gamma
        required_var = gamma_inflate / lambda_min_G
        print(f"  [Eq 38] γ = {gamma_inflate}, required σ²_b = {required_var:.6e}")
        self._variance_bounds = {gamma_inflate: required_var}

        # Determine which B to use for L_wme computation
        B_for_lwme = self._B_lwme if self._B_lwme is not None else self.B
        if self._B_lwme is not None:
            print(f"  [Eq 38] Using separate B_lwme for L_wme computation")

        current_min_var = B_for_lwme.min_eigenvalue()
        print(f"  [Eq 38] Current min(B) = {current_min_var:.6e}")

        # Inflate B to satisfy Eq 38 — no cap, no heuristic override
        if self._auto_inflate_B:
            from .covariance import ScaledCovariance

            alpha = required_var / current_min_var if current_min_var > 0 else 1.0
            alpha = max(1.0, alpha)  # Never deflate

            if alpha > 1.0:
                if self._B_lwme is not None:
                    if self._B_original is None:
                        self._B_original = self._B_lwme
                    else:
                        self._B_lwme = self._B_original
                    self._B_lwme = ScaledCovariance(self._B_lwme, alpha)
                else:
                    if self._B_original is None:
                        self._B_original = self.B
                    else:
                        self.B = self._B_original
                    self.B = ScaledCovariance(self.B, alpha)

                self._b_inflation_factor = alpha
                B_for_lwme_inflated = self._B_lwme if self._B_lwme is not None else self.B
                new_min_var = B_for_lwme_inflated.min_eigenvalue()
                print(f"  [Eq 38] B INFLATED: α = {alpha:.4f} → min(B) = {new_min_var:.6e}")
            else:
                print(f"  [Eq 38] B already satisfies Eq 38 (no inflation needed)")

    def _compute_analytical_L_wme(self) -> 'CovarianceMatrix':
        """Compute L_wme = J_wme B J_wme^T analytically.

        Uses the observation-space column approach (exact, no sampling):
          1. For each basis vector e_i in obs space, compute a_i = J_wme^T e_i
          1b. Compute Gram matrix G[i,j] = a_i^T a_j and inflate B if needed
              (paper Section 6.2.1, eq. 38)
          2. Compute b_i = B a_i (using potentially inflated B)
          3. Form L_wme[i,j] = a_i^T b_j  (= e_i^T J_wme B J_wme^T e_j)

        This gives the exact predicted covariance. The cost is n_obs adjoint
        solves (one per observation DOF), which is much cheaper than forming
        the full n_state-column Jacobian. The Gram matrix computation adds
        zero extra adjoint solves (just n_obs² dot products).

        Regularization follows the relaxed predictability condition from
        Spence et al. (2025), equation (36): λ_min(L_wme) ≥ γ · λ_max(R_eff)
        where R_eff = I in the WME-normalized space, and γ << 1.

        Returns
        -------
        DenseCovariance
            Exact predicted covariance for WME QoI.
        """
        import time as _time

        # Linearize WME at m_b (reuses cached trajectory/Jacobians from __init__)
        linearized_wme = self.qoi_map.linearize(
            self.m_b,
            max(self.obs_times),
            trajectory=self._trajectory,
            jacobians=self._jacobians,
        )

        # Determine observation space dimension from cached Q_wme(m_b)
        Q_wme_mb = self._wme_cache.get("Q_wme_mb")
        if Q_wme_mb is None:
            Q_wme_mb = self._compute_wme(self.m_b)
        n_obs = Q_wme_mb.getSize()

        print(f"  Computing L_wme analytically ({n_obs} adjoint solves)...")
        t_start = _time.perf_counter()

        # Step 1: Compute a_i = J_wme^T e_i for each observation basis vector
        adjoint_vectors = []
        e_i = Q_wme_mb.duplicate()

        for i in range(n_obs):
            e_i.zeroEntries()
            e_i.setValue(i, 1.0)
            e_i.assemblyBegin()
            e_i.assemblyEnd()

            a_i = linearized_wme.apply_adjoint(e_i)
            adjoint_vectors.append(a_i)

            if (i + 1) % 25 == 0 or i == 0:
                elapsed = _time.perf_counter() - t_start
                print(f"    Adjoint solve {i + 1}/{n_obs} ({elapsed:.1f}s)")

        e_i.destroy()

        # Step 1b: Compute Gram matrix G and inflate B if needed
        self._compute_gram_and_inflate_B(adjoint_vectors)

        # Step 2: Compute b_j = B_lwme a_j for each j (uses potentially inflated B_lwme)
        B_for_lwme = self._B_lwme if self._B_lwme is not None else self.B
        B_adjoint_vectors = []
        for j in range(n_obs):
            b_j = B_for_lwme.apply(adjoint_vectors[j])
            B_adjoint_vectors.append(b_j)

        # Step 3: Form L_wme[i,j] = a_i^T b_j + I (exploit symmetry)
        # The +I comes from the observation noise contribution to the WME
        # predicted covariance: L_wme = J_wme B J_wme^T + I
        # where I arises from (1/N) sum_k R^{-1/2} Cov(eps_k) R^{-1/2} = I
        L_dense = np.eye(n_obs)  # Start with identity (+I term)
        for i in range(n_obs):
            for j in range(i, n_obs):
                val = adjoint_vectors[i].dot(B_adjoint_vectors[j])
                L_dense[i, j] += val
                if i != j:
                    L_dense[j, i] += val

        # Clean up PETSc vectors
        for v in adjoint_vectors:
            v.destroy()
        for v in B_adjoint_vectors:
            v.destroy()

        elapsed = _time.perf_counter() - t_start
        print(f"  L_wme matrix formed in {elapsed:.1f}s")

        # Eigenvalue diagnostics
        eigvals = np.linalg.eigvalsh(L_dense)
        print(f"  L_wme eigenvalues: [{eigvals.min():.6e}, {eigvals.max():.6e}]")
        print(f"  L_wme rank (>1e-10): {np.sum(eigvals > 1e-10)}/{n_obs}")
        n_above_1 = np.sum(eigvals > 1.0)
        print(f"  L_wme eigenvalues > 1.0: {n_above_1}/{n_obs} "
              f"(these get proper DC-WME correction)")
        if n_above_1 > 0:
            above_1 = eigvals[eigvals > 1.0]
            print(f"  Eigenvalues > 1: [{above_1.min():.6e}, {above_1.max():.6e}]")

        # Regularization: eigenvalue flooring with configurable floor
        #
        # In WME-normalized space, R_eff = I. The effective data weight per
        # eigendirection is:  w_i = 1 - 1/λ_i  (where λ_i = L_wme eigenvalue)
        #
        # Floor eigenvalues at γ (predictability_gamma) to cap L⁻¹ at 1/γ:
        #   γ = 1:   L⁻¹ ≤ 1.0, weight ≥ 0   (full DC removal allowed)
        #   γ = 2:   L⁻¹ ≤ 0.5, weight ≥ 0.5  (at most 50% data removed)
        #   γ = 5:   L⁻¹ ≤ 0.2, weight ≥ 0.8  (at most 20% data removed)
        #   γ = 10:  L⁻¹ ≤ 0.1, weight ≥ 0.9  (at most 10% data removed)
        #
        # Higher γ = more conservative DC correction, more stable optimization.
        # The cost function data contribution has weight w_i = 1 - 1/λ_i ≥ 1 - 1/γ > 0
        # for γ > 1, ensuring the cost is strictly bounded below.
        eigvals_full, eigvecs = np.linalg.eigh(L_dense)

        if self._adaptive_gamma:
            # Spectrum-relative: floor = γ * λ_max(L_wme)
            gamma_floor = self._predictability_gamma * eigvals_full.max()
            print(f"  Adaptive γ: {self._predictability_gamma} × λ_max={eigvals_full.max():.4e} → γ_eff={gamma_floor:.4e}")
        else:
            # Absolute floor
            gamma_floor = self._predictability_gamma

        n_floored = np.sum(eigvals_full < gamma_floor)
        n_natural = np.sum(eigvals_full >= gamma_floor)

        if self._nonuniform_inflate:
            # Non-uniform per-eigenvalue boosting: boost sub-threshold eigenvalues
            # towards target = boost_target × γ_floor. No multiplicative cap — Eq 38
            # requires uncapped inflation to satisfy predictability condition.
            target = self._eigenvalue_boost_target * gamma_floor
            eigvals_reg = eigvals_full.copy()
            for i in range(len(eigvals_full)):
                if eigvals_full[i] < gamma_floor:
                    eigvals_reg[i] = max(min(target, gamma_floor * self._eigenvalue_boost_target), gamma_floor)
            n_boosted = int(np.sum((eigvals_reg > gamma_floor) & (eigvals_full < gamma_floor)))
            n_still_floored = int(np.sum(eigvals_reg <= gamma_floor))
        else:
            eigvals_reg = np.maximum(eigvals_full, gamma_floor)
            n_boosted = 0
            n_still_floored = n_floored

        L_dense = eigvecs @ np.diag(eigvals_reg) @ eigvecs.T

        max_L_inv = 1.0 / gamma_floor
        min_weight = 1.0 - max_L_inv

        if n_natural > 0:
            natural = eigvals_full[eigvals_full >= gamma_floor]
            nat_min_weight = 1.0 - 1.0 / natural.min()
            nat_max_weight = 1.0 - 1.0 / natural.max()
            print(f"  Eigenvalue floor γ={gamma_floor:.1f}: "
                  f"{n_natural}/{n_obs} natural "
                  f"(λ=[{natural.min():.4e}, {natural.max():.4e}], "
                  f"weight=[{nat_min_weight:.4f}, {nat_max_weight:.4f}]), "
                  f"{n_floored}/{n_obs} floored "
                  f"(weight={min_weight:.4f}, max L⁻¹={max_L_inv:.4f})")
        else:
            print(f"  Eigenvalue floor γ={gamma_floor:.1f}: "
                  f"all {n_obs} floored "
                  f"(max raw λ={eigvals_full.max():.4e}, "
                  f"weight={min_weight:.4f}, max L⁻¹={max_L_inv:.4f})")

        if self._nonuniform_inflate:
            boosted_vals = eigvals_reg[(eigvals_reg > gamma_floor) & (eigvals_full < gamma_floor)]
            if n_boosted > 0:
                boosted_weights = 1.0 - 1.0 / boosted_vals
                print(f"  Non-uniform boost: {n_boosted} eigenvalues boosted above γ "
                      f"(target={target:.2f}), "
                      f"{n_still_floored} still floored")
                print(f"    Boosted λ range: [{boosted_vals.min():.4e}, {boosted_vals.max():.4e}], "
                      f"weight range: [{boosted_weights.min():.4f}, {boosted_weights.max():.4f}]")
            else:
                print(f"  Non-uniform boost: 0 eigenvalues boosted above γ "
                      f"(all {n_floored} below target={target:.4e})")
            n_effective = n_natural + n_boosted
            print(f"  Effective predictable directions: {n_effective}/{n_obs} "
                  f"({n_natural} natural + {n_boosted} boosted)")

        # Create DenseCovariance from dense matrix
        from .covariance import DenseCovariance
        comm = self.comm if self.comm is not None else MPI.COMM_WORLD

        mat = PETSc.Mat().create(comm=comm)
        mat.setSizes((n_obs, n_obs))
        mat.setType(PETSc.Mat.Type.DENSE)
        mat.setUp()

        if comm.rank == 0:
            for i in range(n_obs):
                mat.setValues(i, list(range(n_obs)), L_dense[i, :])

        mat.assemblyBegin()
        mat.assemblyEnd()

        print(f"  L_wme computed: {n_obs}x{n_obs} dense matrix")

        # Restore original B/B_lwme after L_wme computation.
        # The inflation was only needed to compute L_wme = J_wme B_infl J_wme^T.
        # The background term B⁻¹(m - m_b) should use the original (uninflated) B
        # to maintain strong regularization.
        if self._B_original is not None:
            if self._B_lwme is not None:
                print(f"  Restoring original B_lwme (background B was never modified)")
                self._B_lwme = self._B_original
            else:
                print(f"  Restoring original B for background term (inflation was for L_wme only)")
                self.B = self._B_original
            self._B_original = None

        return DenseCovariance(comm, mat)

    def _ensure_wme_predicted_covariance(self):
        """Ensure WME predicted covariance is available.

        Falls back to L_wme = 2R if no proper L_wme was computed.
        """
        if self._L_wme is None:
            # Fallback: L_wme = 2 * R
            R = self._get_observation_covariance(self.obs_times[0])
            from .covariance import ScaledCovariance
            self._L_wme = ScaledCovariance(R, scale_factor=2.0)

    def gradient(self, m: PETSc.Vec) -> PETSc.Vec:
        """
        Compute DC-WME gradient.

        ∇J_WME(m) = B⁻¹(m - m_b) + J^T(Q_wme - L_wme⁻¹(Q_wme - Q_wme,b))

        where J = (1/√N) Σ_k R_k^{-1/2} H_k M_k is constant for linear models.

        Parameters
        ----------
        m : PETSc.Vec
            Control variable.

        Returns
        -------
        PETSc.Vec
            Gradient vector.
        """
        # Run forward model for trajectory
        trajectory, jacobians = self._run_forward_model(m, store_jacobians=True)
        # Share trajectory with QoIMap to avoid redundant forward solve in _compute_wme
        self._share_trajectory_with_qoi(m, trajectory, jacobians)

        # Background gradient: B⁻¹(m - m_b)
        delta_m = m.duplicate()
        delta_m.waxpy(-1.0, self.m_b, m)
        grad = self.B.apply_inverse(delta_m)

        # Compute WME values
        Q_wme_m = self._compute_wme(m)

        if "Q_wme_mb" not in self._wme_cache:
            self._wme_cache["Q_wme_mb"] = self._compute_wme(self.m_b)
        Q_wme_mb = self._wme_cache["Q_wme_mb"]

        # Correction residual
        delta_Q = Q_wme_m.duplicate()
        delta_Q.waxpy(-1.0, Q_wme_mb, Q_wme_m)

        # Compute forcing: Q_wme - L_wme⁻¹(Q_wme - Q_wme,b)
        self._ensure_wme_predicted_covariance()
        L_inv_delta = self._L_wme.apply_inverse(delta_Q)

        # Use copy() not duplicate() - duplicate() creates empty vector!
        forcing = Q_wme_m.copy()
        forcing.axpy(-1.0, L_inv_delta)

        # Apply adjoint of WME Jacobian (pass trajectory to avoid redundant forward solve)
        linearized_wme = self.qoi_map.linearize(
            m, max(self.obs_times), trajectory=self._trajectory, jacobians=self._jacobians
        )
        grad_wme = linearized_wme.apply_adjoint(forcing)

        # Accumulate
        grad.axpy(1.0, grad_wme)

        # Zero boundary DOF gradients (same as standard 4D-Var adjoint)
        # BC DOFs are fixed and should not be modified by the optimizer
        grad = self._zero_boundary_gradient(grad)

        return grad

    def value_gradient(self, m: PETSc.Vec) -> Tuple[float, PETSc.Vec]:
        """
        Compute DC-WME cost and gradient together (efficient for TAO).

        Combines value and gradient computation to avoid redundant forward
        model evaluations.

        Parameters
        ----------
        m : PETSc.Vec
            Control variable.

        Returns
        -------
        Tuple[float, PETSc.Vec]
            (cost_value, gradient_vector)
        """
        # Run forward model once for trajectory and Jacobians
        try:
            stage_start = time.perf_counter()
            trajectory, jacobians = self._run_forward_model(m, store_jacobians=True)
            self._profile_event(
                "dcwme.value_gradient.forward_ready",
                duration_s=float(time.perf_counter() - stage_start),
                trajectory_len=len(trajectory),
                jacobian_len=(len(jacobians) if jacobians is not None else 0),
            )
            # Share trajectory with QoIMap to avoid redundant forward solve in _compute_wme
            self._share_trajectory_with_qoi(m, trajectory, jacobians)
        except Exception as e:
            import warnings
            warnings.warn(
                f"Forward model failed during value_gradient: {e}",
                RuntimeWarning,
                stacklevel=2
            )
            zero_grad = m.duplicate()
            zero_grad.zeroEntries()
            return float('inf'), zero_grad

        # === Compute cost value ===
        # Background term
        delta_m = m.duplicate()
        delta_m.waxpy(-1.0, self.m_b, m)
        B_inv_delta = self.B.apply_inverse(delta_m)
        background_term = 0.5 * delta_m.dot(B_inv_delta)

        # WME data misfit: ½||Q_wme(m)||²
        stage_start = time.perf_counter()
        Q_wme_m = self._compute_wme(m)
        self._profile_event(
            "dcwme.value_gradient.q_wme_ready",
            duration_s=float(time.perf_counter() - stage_start),
        )
        data_misfit = 0.5 * Q_wme_m.dot(Q_wme_m)

        # Predictability term
        stage_start = time.perf_counter()
        predictability = self._compute_wme_predictability(m, Q_wme_m)
        self._profile_event(
            "dcwme.value_gradient.predictability_ready",
            duration_s=float(time.perf_counter() - stage_start),
        )

        cost = background_term + data_misfit - predictability

        # === Compute gradient ===
        # Background gradient: B⁻¹(m - m_b) (reuse delta_m)
        grad = self.B.apply_inverse(delta_m)

        # Compute WME gradient contribution
        if "Q_wme_mb" not in self._wme_cache:
            self._wme_cache["Q_wme_mb"] = self._compute_wme(self.m_b)
        Q_wme_mb = self._wme_cache["Q_wme_mb"]

        # Correction residual
        delta_Q = Q_wme_m.duplicate()
        delta_Q.waxpy(-1.0, Q_wme_mb, Q_wme_m)

        # Compute forcing: Q_wme - L_wme⁻¹(Q_wme - Q_wme,b)
        self._ensure_wme_predicted_covariance()
        stage_start = time.perf_counter()
        L_inv_delta = self._L_wme.apply_inverse(delta_Q)
        self._profile_event(
            "dcwme.value_gradient.l_inv_ready",
            duration_s=float(time.perf_counter() - stage_start),
        )

        # Use copy() not duplicate() - duplicate() creates empty vector!
        forcing = Q_wme_m.copy()
        forcing.axpy(-1.0, L_inv_delta)

        # Apply adjoint of WME Jacobian (pass trajectory to avoid redundant forward solve)
        stage_start = time.perf_counter()
        linearized_wme = self.qoi_map.linearize(
            m, max(self.obs_times), trajectory=self._trajectory, jacobians=self._jacobians
        )
        self._profile_event(
            "dcwme.value_gradient.linearize_ready",
            duration_s=float(time.perf_counter() - stage_start),
        )
        stage_start = time.perf_counter()
        grad_wme = linearized_wme.apply_adjoint(forcing)
        self._profile_event(
            "dcwme.value_gradient.apply_adjoint_ready",
            duration_s=float(time.perf_counter() - stage_start),
        )

        # Accumulate
        grad.axpy(1.0, grad_wme)

        # Zero boundary DOF gradients (same as standard 4D-Var adjoint)
        grad = self._zero_boundary_gradient(grad)
        self._profile_event(
            "dcwme.value_gradient.complete",
            cost=float(cost),
        )

        return cost, grad

    def _zero_boundary_gradient(self, grad: PETSc.Vec) -> PETSc.Vec:
        """
        Zero gradient at boundary DOFs.

        BC DOFs are fixed in the forward problem, so their gradient should
        be zero to prevent the optimizer from modifying them.

        Parameters
        ----------
        grad : PETSc.Vec
            Gradient vector to modify in-place.

        Returns
        -------
        PETSc.Vec
            Modified gradient with zeros at boundary DOFs.
        """
        from ..utils import get_boundary_dofs

        # Get boundary DOFs using topological detection
        bc_dof_indices = None
        if hasattr(self.forward_model, 'solver') and hasattr(self.forward_model, 'problem'):
            V = self.forward_model.solver.V
            mesh = self.forward_model.problem.mesh
            boundary_dofs = get_boundary_dofs(V, mesh)
            bc_dof_indices = set(boundary_dofs.tolist())
        elif hasattr(self.forward_model, 'V') and hasattr(self.forward_model, 'mesh'):
            V = self.forward_model.V
            mesh = self.forward_model.mesh
            boundary_dofs = get_boundary_dofs(V, mesh)
            bc_dof_indices = set(boundary_dofs.tolist())

        if bc_dof_indices is not None and len(bc_dof_indices) > 0:
            grad_arr = grad.getArray()
            for dof in bc_dof_indices:
                if dof < len(grad_arr):
                    grad_arr[dof] = 0.0
            grad.setArray(grad_arr)

        return grad


# Factory function for creating cost functions
def create_cost_function(
    variant: str,
    forward_model,
    observation_operator,
    background_cov,
    observation_cov,
    m_background: PETSc.Vec,
    observations: List[PETSc.Vec],
    obs_times: List[int],
    **kwargs,
) -> CostFunction:
    """
    Factory function for creating cost functions.

    Parameters
    ----------
    variant : str
        Cost function variant: "4dvar", "dc", or "dc_wme".
    forward_model : ForwardModel
        Forward model.
    observation_operator : ObservationOperator
        Observation operator.
    background_cov : CovarianceMatrix
        Background covariance.
    observation_cov : CovarianceMatrix
        Observation covariance.
    m_background : PETSc.Vec
        Background state.
    observations : List[PETSc.Vec]
        Observation vectors.
    obs_times : List[int]
        Observation time indices.
    **kwargs
        Additional arguments for specific variants.

    Returns
    -------
    CostFunction
        Configured cost function instance.
    """
    variant = variant.lower()
    control_layout = getattr(forward_model, "control_layout", None)
    has_augmented_parameters = (
        control_layout is not None and getattr(control_layout, "theta_size", 0) > 0
    )

    if variant == "4dvar" or variant == "standard":
        if has_augmented_parameters:
            from .augmented_cost_functions import AugmentedFourDVarCost

            return AugmentedFourDVarCost(
                forward_model,
                observation_operator,
                background_cov,
                observation_cov,
                m_background,
                observations,
                obs_times,
            )
        return FourDVarCost(
            forward_model,
            observation_operator,
            background_cov,
            observation_cov,
            m_background,
            observations,
            obs_times,
        )
    elif variant == "dc" or variant == "dc_4dvar":
        if has_augmented_parameters:
            from .augmented_cost_functions import AugmentedDCFourDVarCost

            return AugmentedDCFourDVarCost(
                forward_model,
                observation_operator,
                background_cov,
                observation_cov,
                m_background,
                observations,
                obs_times,
                qoi_map=kwargs.get("qoi_map"),
                predicted_cov=kwargs.get("predicted_cov"),
                gamma=kwargs.get("gamma", 1.0),
            )
        return DCFourDVarCost(
            forward_model,
            observation_operator,
            background_cov,
            observation_cov,
            m_background,
            observations,
            obs_times,
            qoi_map=kwargs.get("qoi_map"),
            predicted_cov=kwargs.get("predicted_cov"),
            gamma=kwargs.get("gamma", 1.0),
        )
    elif variant == "dc_wme" or variant == "wme":
        if has_augmented_parameters:
            from .augmented_cost_functions import AugmentedDCWMEFourDVarCost

            return AugmentedDCWMEFourDVarCost(
                forward_model,
                observation_operator,
                background_cov,
                observation_cov,
                m_background,
                observations,
                obs_times,
                predicted_cov_wme=kwargs.get("predicted_cov_wme"),
                auto_inflate_B=kwargs.get("auto_inflate_B", True),
                predictability_gamma=kwargs.get("predictability_gamma", 0.1),
                adaptive_gamma=kwargs.get("adaptive_gamma", True),
                B_lwme=kwargs.get("B_lwme"),
                nonuniform_inflate=kwargs.get("nonuniform_inflate", False),
                eigenvalue_boost_target=kwargs.get("eigenvalue_boost_target", 2.0),
            )
        return DCWMEFourDVarCost(
            forward_model,
            observation_operator,
            background_cov,
            observation_cov,
            m_background,
            observations,
            obs_times,
            predicted_cov_wme=kwargs.get("predicted_cov_wme"),
            auto_inflate_B=kwargs.get("auto_inflate_B", True),
            predictability_gamma=kwargs.get("predictability_gamma", 0.1),
            adaptive_gamma=kwargs.get("adaptive_gamma", True),
            B_lwme=kwargs.get("B_lwme"),
            nonuniform_inflate=kwargs.get("nonuniform_inflate", False),
            eigenvalue_boost_target=kwargs.get("eigenvalue_boost_target", 2.0),
        )
    else:
        raise ValueError(f"Unknown cost function variant: {variant}")

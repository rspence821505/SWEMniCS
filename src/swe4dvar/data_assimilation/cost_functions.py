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

        # Check if we can reuse cached trajectory
        if self._control_hash == m_hash and self._trajectory is not None:
            # If jacobians are requested but not cached, re-run
            if store_jacobians and self._jacobians is None:
                self._trajectory, self._jacobians = self.forward_model.solve(
                    m, store_jacobians=True
                )
            return self._trajectory, self._jacobians

        # Run forward model
        self._trajectory, self._jacobians = self.forward_model.solve(
            m, store_jacobians=store_jacobians
        )
        self._current_control = m.copy()
        self._control_hash = m_hash

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

        # Total gradient
        grad = grad_background.duplicate()
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

        # Compute gradient (same as gradient() but reuses trajectory/jacobians)
        delta_m = m.duplicate()
        delta_m.waxpy(-1.0, self.m_b, m)
        grad_background = self.B.apply_inverse(delta_m)

        # Adjoint solve (uses cached jacobians)
        lambda_0 = self._solve_adjoint(trajectory, jacobians)

        # Total gradient
        grad = grad_background.duplicate()
        grad.axpy(1.0, lambda_0)

        return cost, grad

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
        obs_forcings = self._compute_observation_forcings(trajectory)

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
        # For discrete adjoint (DTO), we need to zero the adjoint at all boundary DOFs
        # This uses topological detection to find ALL boundary DOFs, including
        # those from wall BCs (velocity) that the problem may not explicitly track
        bc_dof_indices = None
        if hasattr(self.forward_model, 'solver') and hasattr(self.forward_model, 'problem'):
            # ForwardModelWrapper provides access to solver and problem
            V = self.forward_model.solver.V
            mesh = self.forward_model.problem.mesh
            boundary_dofs = get_boundary_dofs(V, mesh)
            bc_dof_indices = set(boundary_dofs.tolist())
        elif hasattr(self.forward_model, 'V') and hasattr(self.forward_model, 'mesh'):
            # Direct solver access
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
            bc_dof_indices=bc_dof_indices  # Pass boundary DOFs for proper adjoint BCs
        )

        # Terminal condition (usually zero)
        terminal = trajectory[-1].duplicate()
        terminal.zeroEntries()

        lambda_0 = adjoint_solver.solve(terminal, obs_forcings)

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
            Background covariance B.
        observation_cov : CovarianceMatrix
            Observation covariance R.
        m_background : PETSc.Vec
            Background state.
        observations : List[PETSc.Vec]
            Observation vectors.
        obs_times : List[int]
            Observation time indices.
        predicted_cov_wme : CovarianceMatrix, optional
            Predicted covariance for WME. If None, uses analytical
            approximation L_wme = R/N where N is the number of observations.
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

        # Cache for WME accumulation
        self._wme_cache: Dict[str, PETSc.Vec] = {}

        # Pre-compute Q_wme(m_b) at initialization (computed once per assimilation window)
        # This avoids the lazy computation on first value() call
        self._wme_cache["Q_wme_mb"] = self._compute_wme(self.m_b)

        # Share the trajectory from QoIMap back to cost function's cache
        # This prevents _run_forward_model from re-computing the m_b trajectory
        self._sync_trajectory_from_qoi(self.m_b)

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

    def _compute_wme(self, m: PETSc.Vec) -> PETSc.Vec:
        """
        Compute WME QoI: Q_wme(m) = (1/√N) Σ_k R_k^{-1/2}(H_k(u_k) - y_k).

        Parameters
        ----------
        m : PETSc.Vec
            Control variable.

        Returns
        -------
        PETSc.Vec
            WME vector.
        """
        # Use final observation index K := max(I) for WME evaluation
        k_final = max(self.obs_times)
        return self.qoi_map.evaluate(m, k_final)

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

    def _ensure_wme_predicted_covariance(self):
        """Ensure WME predicted covariance is available.

        For the WME formulation, the predicted covariance L_wme must satisfy
        the predictability condition:
            λ_max(R) < λ_min(L_wme)

        This ensures that the model's predicted uncertainty (L_wme) is greater
        than the observation uncertainty (R), which is required for the DC
        formulation to properly weight the observations.

        We use L_wme = 2 * R which satisfies:
            λ_min(L_wme) = 2 * λ_min(R) ≥ 2 * λ_max(R) > λ_max(R)
        for diagonal R with uniform entries.
        """
        if self._L_wme is None:
            # Get observation covariance at first observation time
            R = self._get_observation_covariance(self.obs_times[0])

            # L_wme = 2 * R ensures predictability condition: λ_max(R) < λ_min(L_wme)
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
            trajectory, jacobians = self._run_forward_model(m, store_jacobians=True)
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
        Q_wme_m = self._compute_wme(m)
        data_misfit = 0.5 * Q_wme_m.dot(Q_wme_m)

        # Predictability term
        predictability = self._compute_wme_predictability(m, Q_wme_m)

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
        grad = self._zero_boundary_gradient(grad)

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

    if variant == "4dvar" or variant == "standard":
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
        return DCWMEFourDVarCost(
            forward_model,
            observation_operator,
            background_cov,
            observation_cov,
            m_background,
            observations,
            obs_times,
            predicted_cov_wme=kwargs.get("predicted_cov_wme"),
        )
    else:
        raise ValueError(f"Unknown cost function variant: {variant}")

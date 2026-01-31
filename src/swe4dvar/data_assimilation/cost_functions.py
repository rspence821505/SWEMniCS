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
        """
        self.forward_model = forward_model
        self.obs_op = observation_operator
        self.B = background_cov
        self.R = observation_cov
        self.comm = comm if comm is not None else MPI.COMM_WORLD

        # Cache for forward trajectory and Jacobians
        self._trajectory: Optional[List[PETSc.Vec]] = None
        self._jacobians: Optional[List[PETSc.Mat]] = None
        self._current_control: Optional[PETSc.Vec] = None

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

    def _run_forward_model(
        self, m: PETSc.Vec, store_jacobians: bool = True
    ) -> Tuple[List[PETSc.Vec], Optional[List[PETSc.Mat]]]:
        """
        Run forward model and cache trajectory.

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
        # Check if we can reuse cached trajectory
        if self._current_control is not None:
            diff = m.duplicate()
            diff.waxpy(-1.0, self._current_control, m)
            if diff.norm() < 1e-14:
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
        """
        super().__init__(
            forward_model, observation_operator, background_cov, observation_cov, comm
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
            Cost function value.
        """
        # Run forward model
        trajectory, _ = self._run_forward_model(m, store_jacobians=False)

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

        # Check if forward model has a variational form (for BDF2 time coefficients)
        variational_form = getattr(self.forward_model, 'var_form', None)

        adjoint_solver = ImplicitAdjointSolver(
            self.forward_model,
            trajectory,
            jacobians,
            self.forward_model.dt,
            variational_form=variational_form  # NEW: Pass variational form if available
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
            Predicted covariance for WME. If None, estimated.
        comm : MPI.Comm, optional
            MPI communicator.
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
        )

        # WME-specific predicted covariance
        self._L_wme = predicted_cov_wme

        # Cache for WME accumulation
        self._wme_cache: Dict[str, PETSc.Vec] = {}

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
            DC-WME cost function value.
        """
        # Background term
        background_term = self._compute_background_term(m)

        # WME data misfit: ½ ‖Q_wme(m)‖²
        Q_wme_m = self._compute_wme(m)
        data_misfit = 0.5 * Q_wme_m.dot(Q_wme_m)

        # Predictability term: ½ ⟨Q_wme(m) - Q_wme(m_b), L⁻¹(...)⟩
        predictability = self._compute_wme_predictability(m, Q_wme_m)

        return background_term + data_misfit - predictability

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
        """Ensure WME predicted covariance is available."""
        if self._L_wme is None:
            from .qoi_maps import QoICovarianceEstimator

            estimator = QoICovarianceEstimator(self.qoi_map, self.B, num_samples=100)
            # Use final time for WME covariance
            self._L_wme = estimator.estimate(self.m_b, max(self.obs_times))

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

        forcing = Q_wme_m.duplicate()
        forcing.axpy(-1.0, L_inv_delta)

        # Apply adjoint of WME Jacobian
        linearized_wme = self.qoi_map.linearize(m, max(self.obs_times))
        grad_wme = linearized_wme.apply_adjoint(forcing)

        # Accumulate
        grad.axpy(1.0, grad_wme)

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

"""
Cost function implementations for 4D-Var data assimilation.

Implements standard 4D-Var, DC-4DVar, and DC-WME variants
following Spence et al. (2025).
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from petsc4py import PETSc


class CostFunction(ABC):
    """
    Abstract base class for 4D-Var cost functions.

    Defines the interface for computing cost function value,
    gradient, and Hessian-vector products.
    """

    def __init__(
        self, forward_model, observation_operator, background_cov, observation_cov
    ):
        """
        Initialize cost function.

        Args:
            forward_model: Forward model M_{k:0}
            observation_operator: Observation operator H_k
            background_cov: Background error covariance B
            observation_cov: Observation error covariance R_k
        """
        self.forward_model = forward_model
        self.obs_op = observation_operator
        self.B = background_cov
        self.R = observation_cov

        # Cache for forward trajectory
        self._trajectory = None
        self._jacobians = None

    @abstractmethod
    def value(self, m: PETSc.Vec) -> float:
        """
        Compute cost function value J(m).

        Args:
            m: Control variable (initial condition)

        Returns:
            Cost function value
        """
        pass

    @abstractmethod
    def gradient(self, m: PETSc.Vec) -> PETSc.Vec:
        """
        Compute gradient ∇J(m) via adjoint method.

        Args:
            m: Control variable

        Returns:
            Gradient vector
        """
        pass

    def hessian_vector_product(self, m: PETSc.Vec, v: PETSc.Vec) -> PETSc.Vec:
        """
        Compute Hessian-vector product Hv using Gauss-Newton approximation.

        Args:
            m: Control variable
            v: Direction vector

        Returns:
            H·v
        """
        raise NotImplementedError("Gauss-Newton Hessian not yet implemented")

    def _run_forward_model(
        self, m: PETSc.Vec, store_jacobians: bool = True
    ) -> Tuple[List, Optional[List]]:
        """
        Run forward model and cache trajectory.

        Args:
            m: Initial condition
            store_jacobians: Whether to cache Jacobians for adjoint

        Returns:
            (trajectory, jacobians) tuple
        """
        self._trajectory, self._jacobians = self.forward_model.solve(
            m, store_jacobians=store_jacobians
        )
        return self._trajectory, self._jacobians


class FourDVarCost(CostFunction):
    """
    Standard 4D-Var cost function.

    J(m) = ½⟨m - m_b, B⁻¹(m - m_b)⟩
         + ½ Σ_k ⟨H_k(u_k) - y_k, R_k⁻¹(H_k(u_k) - y_k)⟩
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
    ):
        """
        Initialize standard 4D-Var cost function.

        Args:
            forward_model: Forward model
            observation_operator: Observation operator
            background_cov: Background covariance B
            observation_cov: Observation covariance R
            m_background: Background state m_b
            observations: List of observation vectors y_k
            obs_times: List of observation time indices
        """
        super().__init__(
            forward_model, observation_operator, background_cov, observation_cov
        )
        self.m_b = m_background
        self.y_obs = observations
        self.obs_times = obs_times

    def value(self, m: PETSc.Vec) -> float:
        """Compute standard 4D-Var cost."""
        # TODO: Implement background term + observation term
        pass

    def gradient(self, m: PETSc.Vec) -> PETSc.Vec:
        """Compute gradient via adjoint method."""
        # TODO: Implement adjoint-based gradient computation
        pass


class DCFourDVarCost(CostFunction):
    """
    Data-Consistent 4D-Var (DC-4DVar) cost function.

    J_DC(m) = J(m) - ½ Σ_k ⟨Q_k(m) - Q_k(m_b), L_k⁻¹(Q_k(m) - Q_k(m_b))⟩

    Includes predictability term to prevent assimilation of
    unpredictable small scales.
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
    ):
        """
        Initialize DC-4DVar cost function.

        Args:
            forward_model: Forward model
            observation_operator: Observation operator
            background_cov: Background covariance B
            observation_cov: Observation covariance R
            m_background: Background state
            observations: Observation vectors
            obs_times: Observation time indices
            qoi_map: Quantity of Interest map Q_k
        """
        super().__init__(
            forward_model, observation_operator, background_cov, observation_cov
        )
        self.m_b = m_background
        self.y_obs = observations
        self.obs_times = obs_times
        self.qoi_map = qoi_map

        # Predicted error covariance L_k (computed from B)
        self._L_k = None

    def value(self, m: PETSc.Vec) -> float:
        """Compute DC-4DVar cost with predictability term."""
        # TODO: Implement standard term - predictability term
        pass

    def gradient(self, m: PETSc.Vec) -> PETSc.Vec:
        """Compute DC-4DVar gradient."""
        # TODO: Implement with predictability gradient correction
        pass

    def _compute_predicted_covariance(self, k: int) -> PETSc.Mat:
        """
        Compute L_k = Q_k B Q_k^T at observation time k.

        Args:
            k: Time index

        Returns:
            Predicted error covariance matrix
        """
        # TODO: Implement TLM-based covariance propagation
        pass


class DCWMEFourDVarCost(DCFourDVarCost):
    """
    DC-4DVar with Weighted Mean Error QoI.

    Q_wme,k(m) = (1/k) Σ_{j=0}^{k-1} (H_j(M_{j:0}(m)) - y_j)

    Uses cumulative time-averaged innovation as QoI
    for improved stability with sparse observations.
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
    ):
        """Initialize DC-WME cost function."""
        super().__init__(
            forward_model,
            observation_operator,
            background_cov,
            observation_cov,
            m_background,
            observations,
            obs_times,
        )

        # Cache for WME accumulation
        self._wme_accumulator = {}

    def value(self, m: PETSc.Vec) -> float:
        """Compute DC-WME cost."""
        # TODO: Implement with WME QoI
        pass

    def gradient(self, m: PETSc.Vec) -> PETSc.Vec:
        """Compute DC-WME gradient."""
        # TODO: Implement with WME adjoint contribution
        pass

    def _compute_wme_qoi(self, trajectory: List, k: int) -> PETSc.Vec:
        """
        Compute WME QoI at time k.

        Q_wme,k = (1/k) Σ_{j=0}^{k-1} (H_j(u_j) - y_j)

        Args:
            trajectory: Forward trajectory [u_0, ..., u_k]
            k: Current time index

        Returns:
            WME vector
        """
        # TODO: Implement weighted mean error accumulation
        pass

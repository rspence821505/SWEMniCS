"""
Quantity of Interest (QoI) maps for DC-4DVar.

Implements QoI operators Q_k: V → R^{m_k} and their linearizations
for computing predictability terms in DC-4DVar cost functions.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from petsc4py import PETSc


class QoIMap(ABC):
    """
    Abstract base class for Quantity of Interest maps.

    A QoI map extracts specific quantities from the model state
    that are compared between control and background runs.
    """

    def __init__(self, forward_model, observation_operator):
        """
        Initialize QoI map.

        Args:
            forward_model: Forward model M_{k:0}
            observation_operator: Observation operator H_k
        """
        self.forward_model = forward_model
        self.obs_op = observation_operator

    @abstractmethod
    def evaluate(self, m: PETSc.Vec, time_index: int) -> PETSc.Vec:
        """
        Evaluate QoI at time index k: Q_k(m).

        Args:
            m: Control variable (initial condition)
            time_index: Time index k

        Returns:
            QoI vector
        """
        pass

    @abstractmethod
    def linearize(self, m: PETSc.Vec, time_index: int) -> "LinearizedQoI":
        """
        Linearize QoI about state m at time k.

        Returns linear operator representing DQ_k(m).

        Args:
            m: Linearization point
            time_index: Time index k

        Returns:
            Linearized QoI operator
        """
        pass


class StandardQoI(QoIMap):
    """
    Standard QoI: direct observation of model state.

    Q_k(m) = H_k(M_{k:0}(m))

    Simply composes forward model with observation operator.
    """

    def evaluate(self, m: PETSc.Vec, time_index: int) -> PETSc.Vec:
        """
        Evaluate Q_k = H_k ∘ M_{k:0}.

        Args:
            m: Initial condition
            time_index: Target time index k

        Returns:
            Observed model state at time k
        """
        # Run forward model to time k
        trajectory, _ = self.forward_model.solve(m, store_jacobians=False)
        u_k = trajectory[time_index]

        # Apply observation operator
        return self.obs_op.forward(u_k)

    def linearize(self, m: PETSc.Vec, time_index: int) -> "LinearizedQoI":
        """
        Linearize standard QoI.

        DQ_k = H_k · TLM_{k:0}
        where TLM is the tangent linear model.
        """
        return LinearizedStandardQoI(self.forward_model, self.obs_op, m, time_index)


class WeightedMeanErrorQoI(QoIMap):
    """
    Weighted Mean Error QoI for DC-WME.

    Q_wme,k(m) = (1/k) Σ_{j=0}^{k-1} (H_j(M_{j:0}(m)) - y_j)

    Accumulates time-averaged innovation up to time k.
    """

    def __init__(
        self, forward_model, observation_operator, observations: List[PETSc.Vec]
    ):
        """
        Initialize WME QoI.

        Args:
            forward_model: Forward model
            observation_operator: Observation operator
            observations: True observation vectors y_j
        """
        super().__init__(forward_model, observation_operator)
        self.y_obs = observations

    def evaluate(self, m: PETSc.Vec, time_index: int) -> PETSc.Vec:
        """
        Evaluate WME QoI up to time k.

        Computes: Q_wme,k = (1/k) Σ_{j=0}^{k-1} (H_j(u_j) - y_j)

        Args:
            m: Initial condition
            time_index: Target time index k

        Returns:
            WME vector at time k
        """
        # TODO: Implement WME accumulation
        # 1. Run forward model to time k
        # 2. Apply observation operator at each time j ≤ k
        # 3. Accumulate innovations (H_j(u_j) - y_j)
        # 4. Divide by k
        pass

    def linearize(self, m: PETSc.Vec, time_index: int) -> "LinearizedQoI":
        """
        Linearize WME QoI.

        DQ_wme,k = (1/k) Σ_{j=0}^{k-1} H_j · TLM_{j:0}
        """
        return LinearizedWMEQoI(
            self.forward_model, self.obs_op, m, time_index, self.y_obs
        )


class LinearizedQoI(ABC):
    """
    Abstract base class for linearized QoI operators.

    Represents DQ_k(m̄): δm → δq
    """

    @abstractmethod
    def apply(self, delta_m: PETSc.Vec) -> PETSc.Vec:
        """
        Apply linearized QoI: δq = DQ_k(m̄)·δm.

        Args:
            delta_m: Perturbation in control space

        Returns:
            Perturbation in QoI space
        """
        pass

    @abstractmethod
    def apply_adjoint(self, delta_q: PETSc.Vec) -> PETSc.Vec:
        """
        Apply adjoint of linearized QoI: δm = DQ_k(m̄)^T·δq.

        Args:
            delta_q: Perturbation in QoI space

        Returns:
            Perturbation in control space
        """
        pass


class LinearizedStandardQoI(LinearizedQoI):
    """
    Linearized standard QoI: DQ_k = H_k · TLM_{k:0}.
    """

    def __init__(
        self, forward_model, observation_operator, m_bar: PETSc.Vec, time_index: int
    ):
        """
        Initialize linearized standard QoI.

        Args:
            forward_model: Forward model
            observation_operator: Observation operator
            m_bar: Linearization point
            time_index: Time index k
        """
        self.forward_model = forward_model
        self.obs_op = observation_operator
        self.m_bar = m_bar
        self.k = time_index

        # Cache trajectory at linearization point
        self._trajectory = None
        self._jacobians = None

    def apply(self, delta_m: PETSc.Vec) -> PETSc.Vec:
        """
        Apply DQ_k·δm via TLM.

        1. Run TLM to propagate δm to time k: δu_k = TLM_{k:0}·δm
        2. Apply observation operator: δq = H_k·δu_k
        """
        # TODO: Implement TLM propagation + observation
        pass

    def apply_adjoint(self, delta_q: PETSc.Vec) -> PETSc.Vec:
        """
        Apply DQ_k^T·δq via adjoint.

        1. Apply adjoint observation operator: δu_k = H_k^T·δq
        2. Run adjoint model backward: δm = ADJ_{k:0}·δu_k
        """
        # TODO: Implement adjoint observation + adjoint model
        pass


class LinearizedWMEQoI(LinearizedQoI):
    """
    Linearized WME QoI: DQ_wme,k = (1/k) Σ_{j=0}^{k-1} H_j · TLM_{j:0}.
    """

    def __init__(
        self,
        forward_model,
        observation_operator,
        m_bar: PETSc.Vec,
        time_index: int,
        observations: List[PETSc.Vec],
    ):
        """
        Initialize linearized WME QoI.

        Args:
            forward_model: Forward model
            observation_operator: Observation operator
            m_bar: Linearization point
            time_index: Time index k
            observations: True observations (not used in linearization)
        """
        self.forward_model = forward_model
        self.obs_op = observation_operator
        self.m_bar = m_bar
        self.k = time_index
        self.y_obs = observations

    def apply(self, delta_m: PETSc.Vec) -> PETSc.Vec:
        """
        Apply linearized WME via TLM.

        δq_wme = (1/k) Σ_{j=0}^{k-1} H_j · (TLM_{j:0}·δm)

        Requires running TLM to each observation time and accumulating.
        """
        # TODO: Implement accumulated TLM evaluation
        pass

    def apply_adjoint(self, delta_q: PETSc.Vec) -> PETSc.Vec:
        """
        Apply adjoint of linearized WME.

        δm = (1/k) Σ_{j=0}^{k-1} ADJ_{j:0} · H_j^T · δq_wme

        Requires running adjoint from each observation time and accumulating.
        """
        # TODO: Implement accumulated adjoint evaluation
        pass


class QoICovarianceEstimator:
    """
    Estimates predicted error covariance L_k = Q_k B Q_k^T.

    Uses Monte Carlo sampling or ensemble methods to approximate
    the push-forward of background covariance through QoI map.
    """

    def __init__(self, qoi_map: QoIMap, background_cov, num_samples: int = 100):
        """
        Initialize covariance estimator.

        Args:
            qoi_map: QoI map to push covariance through
            background_cov: Background covariance B
            num_samples: Number of Monte Carlo samples
        """
        self.qoi_map = qoi_map
        self.B = background_cov
        self.num_samples = num_samples

    def estimate(self, m_bar: PETSc.Vec, time_index: int) -> PETSc.Mat:
        """
        Estimate L_k ≈ (1/N) Σᵢ (Q_k·ξᵢ)(Q_k·ξᵢ)^T where ξᵢ ~ N(0, B).

        Args:
            m_bar: Linearization point (typically m_background)
            time_index: Time index k

        Returns:
            Estimated covariance matrix L_k
        """
        # TODO: Implement Monte Carlo or ensemble estimation
        # 1. Sample ξᵢ ~ N(0, B)
        # 2. Linearize QoI about m_bar
        # 3. Apply linearized QoI to samples
        # 4. Compute empirical covariance
        pass

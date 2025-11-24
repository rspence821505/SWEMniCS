"""
Quantity of Interest (QoI) maps for DC-4DVar.

Implements QoI operators Q_k: V → R^{m_k} and their linearizations
for computing predictability terms in DC-4DVar cost functions.

Mathematical Background
-----------------------
The QoI map Q_k represents the composition of the forward model and
observation operator:
    Q_k = H_k ∘ M_{k:0}

The linearized QoI (Jacobian) is:
    DQ_k = H_k · TLM_{k:0}

where TLM is the tangent linear model.

For DC-WME, the Weighted Mean Error QoI is:
    Q_wme(m) = (1/√N) Σ_{k=1}^{N} R_k^{-1/2} [H_k(M_{k:0}(m)) - y_k]
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np


class QoIMap(ABC):
    """
    Abstract base class for Quantity of Interest maps.

    A QoI map extracts specific quantities from the model state
    that are compared between control and background runs.

    Attributes
    ----------
    forward_model : ForwardModel
        Forward model M_{k:0}.
    obs_op : ObservationOperator
        Observation operator H_k.
    """

    def __init__(self, forward_model, observation_operator):
        """
        Initialize QoI map.

        Parameters
        ----------
        forward_model : ForwardModel
            Forward model M_{k:0}.
        observation_operator : ObservationOperator
            Observation operator H_k.
        """
        self.forward_model = forward_model
        self.obs_op = observation_operator

        # Cache for trajectories
        self._trajectory_cache: Dict[int, Tuple[List[PETSc.Vec], List]] = {}

    @abstractmethod
    def evaluate(self, m: PETSc.Vec, time_index: int) -> PETSc.Vec:
        """
        Evaluate QoI at time index k: Q_k(m).

        Parameters
        ----------
        m : PETSc.Vec
            Control variable (initial condition).
        time_index : int
            Time index k.

        Returns
        -------
        PETSc.Vec
            QoI vector.
        """
        pass

    @abstractmethod
    def linearize(self, m: PETSc.Vec, time_index: int) -> "LinearizedQoI":
        """
        Linearize QoI about state m at time k.

        Returns linear operator representing DQ_k(m).

        Parameters
        ----------
        m : PETSc.Vec
            Linearization point.
        time_index : int
            Time index k.

        Returns
        -------
        LinearizedQoI
            Linearized QoI operator.
        """
        pass

    def _get_trajectory(
        self, m: PETSc.Vec, store_jacobians: bool = True
    ) -> Tuple[List[PETSc.Vec], Optional[List]]:
        """Get or compute trajectory for given initial condition."""
        # Simple hash for caching (could be improved)
        m_hash = hash(m.norm())

        if m_hash not in self._trajectory_cache:
            trajectory, jacobians = self.forward_model.solve(m, store_jacobians)
            self._trajectory_cache[m_hash] = (trajectory, jacobians)

        return self._trajectory_cache[m_hash]

    def clear_cache(self):
        """Clear trajectory cache."""
        self._trajectory_cache.clear()


class StandardQoI(QoIMap):
    """
    Standard QoI: direct observation of model state.

    Q_k(m) = H_k(M_{k:0}(m))

    Simply composes forward model with observation operator.
    This is the QoI used in standard DC-4DVar.
    """

    def evaluate(self, m: PETSc.Vec, time_index: int) -> PETSc.Vec:
        """
        Evaluate Q_k = H_k ∘ M_{k:0}.

        Parameters
        ----------
        m : PETSc.Vec
            Initial condition.
        time_index : int
            Target time index k.

        Returns
        -------
        PETSc.Vec
            Observed model state at time k.
        """
        # Run forward model to time k
        trajectory, _ = self._get_trajectory(m, store_jacobians=False)

        if time_index >= len(trajectory):
            raise IndexError(
                f"Time index {time_index} exceeds trajectory length {len(trajectory)}"
            )

        u_k = trajectory[time_index]

        # Apply observation operator
        return self.obs_op.forward(u_k, time_index=time_index)

    def linearize(self, m: PETSc.Vec, time_index: int) -> "LinearizedQoI":
        """
        Linearize standard QoI.

        DQ_k = H_k · TLM_{k:0}

        where TLM is the tangent linear model.

        Parameters
        ----------
        m : PETSc.Vec
            Linearization point.
        time_index : int
            Time index k.

        Returns
        -------
        LinearizedStandardQoI
            Linearized QoI operator.
        """
        return LinearizedStandardQoI(self.forward_model, self.obs_op, m, time_index)


class WeightedMeanErrorQoI(QoIMap):
    """
    Weighted Mean Error QoI for DC-WME.

    Q_wme,k(m) = (1/√k) Σ_{j=0}^{k-1} R_j^{-1/2}(H_j(M_{j:0}(m)) - y_j)

    Accumulates time-averaged, precision-weighted innovation up to time k.

    This QoI has several desirable properties:
    1. Predictability assumption guaranteed for sufficiently large k
    2. Uncertainties decrease at rate proportional to observations
    3. Unbiased estimate of sample mean of observed data

    Attributes
    ----------
    y_obs : List[PETSc.Vec]
        True observation vectors.
    R_cov : CovarianceMatrix
        Observation error covariance (for R^{-1/2}).
    """

    def __init__(
        self,
        forward_model,
        observation_operator,
        observations: List[PETSc.Vec],
        observation_cov,
    ):
        """
        Initialize WME QoI.

        Parameters
        ----------
        forward_model : ForwardModel
            Forward model.
        observation_operator : ObservationOperator
            Observation operator.
        observations : List[PETSc.Vec]
            True observation vectors y_j.
        observation_cov : CovarianceMatrix
            Observation error covariance R.
        """
        super().__init__(forward_model, observation_operator)
        self.y_obs = observations
        self.R_cov = observation_cov

        # Cache for R^{-1/2} application
        self._R_sqrt_inv_cache: Dict = {}

    def evaluate(self, m: PETSc.Vec, time_index: int) -> PETSc.Vec:
        """
        Evaluate WME QoI up to time k.

        Computes: Q_wme,k = (1/√k) Σ_{j=0}^{k-1} R_j^{-1/2}(H_j(u_j) - y_j)

        Parameters
        ----------
        m : PETSc.Vec
            Initial condition.
        time_index : int
            Target time index k.

        Returns
        -------
        PETSc.Vec
            WME vector at time k.
        """
        if time_index <= 0:
            raise ValueError("time_index must be positive for WME")

        # Run forward model
        trajectory, _ = self._get_trajectory(m, store_jacobians=False)

        # Initialize accumulator
        wme = None
        num_obs = min(time_index, len(self.y_obs))

        for j in range(num_obs):
            # Get state at time j
            if j >= len(trajectory):
                break
            u_j = trajectory[j]

            # Apply observation operator: H_j(u_j)
            Hu_j = self.obs_op.forward(u_j, time_index=j)

            # Innovation: d_j = H_j(u_j) - y_j
            d_j = Hu_j.duplicate()
            d_j.waxpy(-1.0, self.y_obs[j], Hu_j)

            # Apply R_j^{-1/2}
            R_sqrt_inv_d = self._apply_R_sqrt_inv(d_j, j)

            # Accumulate
            if wme is None:
                wme = R_sqrt_inv_d.copy()
            else:
                wme.axpy(1.0, R_sqrt_inv_d)

        # Scale by 1/√k
        if wme is not None and num_obs > 0:
            wme.scale(1.0 / np.sqrt(num_obs))
        elif wme is None:
            # Return zero vector if no observations
            wme = self.y_obs[0].duplicate()
            wme.zeroEntries()

        return wme

    def _apply_R_sqrt_inv(self, v: PETSc.Vec, time_index: int) -> PETSc.Vec:
        """
        Apply R^{-1/2} to vector.

        For diagonal R = σ²I, this is simply v/σ.

        Parameters
        ----------
        v : PETSc.Vec
            Input vector.
        time_index : int
            Time index (for time-varying R).

        Returns
        -------
        PETSc.Vec
            R^{-1/2} · v.
        """
        # Get observation covariance for this time
        if hasattr(self.R_cov, "apply_sqrt_inverse"):
            return self.R_cov.apply_sqrt_inverse(v)
        else:
            # Fallback: use apply_inverse and estimate sqrt
            # For diagonal covariance, sqrt(R^{-1}) = R^{-1/2}
            R_inv_v = self.R_cov.apply_inverse(v)

            # Approximate sqrt via scaling (accurate for diagonal)
            result = v.duplicate()
            result.pointwiseMult(R_inv_v, v)
            result.sqrtabs()
            # Sign correction
            signs = v.duplicate()
            v.copy(signs)
            signs.sign()
            result.pointwiseMult(result, signs)

            return result

    def linearize(self, m: PETSc.Vec, time_index: int) -> "LinearizedQoI":
        """
        Linearize WME QoI.

        DQ_wme,k = (1/√k) Σ_{j=0}^{k-1} R_j^{-1/2} · H_j · TLM_{j:0}

        Parameters
        ----------
        m : PETSc.Vec
            Linearization point.
        time_index : int
            Time index k.

        Returns
        -------
        LinearizedWMEQoI
            Linearized WME operator.
        """
        return LinearizedWMEQoI(
            self.forward_model,
            self.obs_op,
            m,
            time_index,
            self.y_obs,
            self.R_cov,
        )


class LinearizedQoI(ABC):
    """
    Abstract base class for linearized QoI operators.

    Represents DQ_k(m̄): δm → δq

    The linearized QoI maps perturbations in the control space
    to perturbations in the QoI space.
    """

    @abstractmethod
    def apply(self, delta_m: PETSc.Vec) -> PETSc.Vec:
        """
        Apply linearized QoI: δq = DQ_k(m̄)·δm.

        Parameters
        ----------
        delta_m : PETSc.Vec
            Perturbation in control space.

        Returns
        -------
        PETSc.Vec
            Perturbation in QoI space.
        """
        pass

    @abstractmethod
    def apply_adjoint(self, delta_q: PETSc.Vec) -> PETSc.Vec:
        """
        Apply adjoint of linearized QoI: δm = DQ_k(m̄)^T·δq.

        Parameters
        ----------
        delta_q : PETSc.Vec
            Perturbation in QoI space.

        Returns
        -------
        PETSc.Vec
            Perturbation in control space.
        """
        pass


class LinearizedStandardQoI(LinearizedQoI):
    """
    Linearized standard QoI: DQ_k = H_k · TLM_{k:0}.

    The forward application propagates perturbations forward in time
    via the tangent linear model, then applies the observation operator.

    The adjoint application applies the adjoint observation operator,
    then propagates backward via the adjoint model.

    Attributes
    ----------
    forward_model : ForwardModel
        Forward model (for accessing TLM/adjoint).
    obs_op : ObservationOperator
        Observation operator.
    m_bar : PETSc.Vec
        Linearization point.
    k : int
        Target time index.
    """

    def __init__(
        self,
        forward_model,
        observation_operator,
        m_bar: PETSc.Vec,
        time_index: int,
    ):
        """
        Initialize linearized standard QoI.

        Parameters
        ----------
        forward_model : ForwardModel
            Forward model.
        observation_operator : ObservationOperator
            Observation operator.
        m_bar : PETSc.Vec
            Linearization point.
        time_index : int
            Time index k.
        """
        self.forward_model = forward_model
        self.obs_op = observation_operator
        self.m_bar = m_bar
        self.k = time_index

        # Cache trajectory and Jacobians at linearization point
        self._trajectory: Optional[List[PETSc.Vec]] = None
        self._jacobians: Optional[List[PETSc.Mat]] = None
        self._ensure_linearization()

    def _ensure_linearization(self):
        """Ensure trajectory/Jacobians are computed at linearization point."""
        if self._trajectory is None:
            self._trajectory, self._jacobians = self.forward_model.solve(
                self.m_bar, store_jacobians=True
            )

    def apply(self, delta_m: PETSc.Vec) -> PETSc.Vec:
        """
        Apply DQ_k·δm via TLM.

        1. Run TLM to propagate δm to time k: δu_k = TLM_{k:0}·δm
        2. Apply observation operator: δq = H_k·δu_k

        Parameters
        ----------
        delta_m : PETSc.Vec
            Perturbation in control space.

        Returns
        -------
        PETSc.Vec
            Perturbation in QoI space.
        """
        # Import TLM solver
        from ..adjoint.tangent_linear import TangentLinearModel

        tlm = TangentLinearModel(self.forward_model, self._trajectory, self._jacobians)

        # Propagate perturbation to time k
        delta_u_k = tlm.propagate(delta_m, target_time=self.k)

        # Apply observation operator (linearized at u_k)
        delta_q = self.obs_op.forward_linearized(
            delta_u_k, self._trajectory[self.k], time_index=self.k
        )

        return delta_q

    def apply_adjoint(self, delta_q: PETSc.Vec) -> PETSc.Vec:
        """
        Apply DQ_k^T·δq via adjoint.

        1. Apply adjoint observation operator: δu_k = H_k^T·δq
        2. Run adjoint model backward: δm = ADJ_{k:0}·δu_k

        Parameters
        ----------
        delta_q : PETSc.Vec
            Perturbation in QoI space.

        Returns
        -------
        PETSc.Vec
            Perturbation in control space.
        """
        # Apply adjoint observation operator
        delta_u_k = self.obs_op.adjoint(delta_q, time_index=self.k)

        # Run adjoint from time k to 0
        from ..adjoint.implicit_adjoint import ImplicitAdjointSolver

        adjoint_solver = ImplicitAdjointSolver(
            self.forward_model,
            self._trajectory[: self.k + 1],
            self._jacobians[: self.k] if self._jacobians else None,
            self.forward_model.dt,
        )

        # Create forcing vector (only at time k)
        forcings = [None] * (self.k + 1)
        forcings[self.k] = delta_u_k

        # Terminal condition
        terminal = delta_u_k.duplicate()
        terminal.zeroEntries()

        delta_m = adjoint_solver.solve(terminal, forcings)

        return delta_m


class LinearizedWMEQoI(LinearizedQoI):
    """
    Linearized WME QoI: DQ_wme,k = (1/√k) Σ_{j=0}^{k-1} R_j^{-1/2} · H_j · TLM_{j:0}.

    For the WME QoI, the linearization requires accumulating contributions
    from all observation times up to k.

    Attributes
    ----------
    forward_model : ForwardModel
        Forward model.
    obs_op : ObservationOperator
        Observation operator.
    m_bar : PETSc.Vec
        Linearization point.
    k : int
        Time index.
    y_obs : List[PETSc.Vec]
        Observations (not used in linearization, but kept for consistency).
    R_cov : CovarianceMatrix
        Observation covariance.
    """

    def __init__(
        self,
        forward_model,
        observation_operator,
        m_bar: PETSc.Vec,
        time_index: int,
        observations: List[PETSc.Vec],
        observation_cov,
    ):
        """
        Initialize linearized WME QoI.

        Parameters
        ----------
        forward_model : ForwardModel
            Forward model.
        observation_operator : ObservationOperator
            Observation operator.
        m_bar : PETSc.Vec
            Linearization point.
        time_index : int
            Time index k.
        observations : List[PETSc.Vec]
            True observations (not used in linearization).
        observation_cov : CovarianceMatrix
            Observation covariance R.
        """
        self.forward_model = forward_model
        self.obs_op = observation_operator
        self.m_bar = m_bar
        self.k = time_index
        self.y_obs = observations
        self.R_cov = observation_cov

        # Cache trajectory
        self._trajectory: Optional[List[PETSc.Vec]] = None
        self._jacobians: Optional[List[PETSc.Mat]] = None
        self._ensure_linearization()

    def _ensure_linearization(self):
        """Ensure trajectory is computed."""
        if self._trajectory is None:
            self._trajectory, self._jacobians = self.forward_model.solve(
                self.m_bar, store_jacobians=True
            )

    def apply(self, delta_m: PETSc.Vec) -> PETSc.Vec:
        """
        Apply linearized WME via TLM.

        δq_wme = (1/√k) Σ_{j=0}^{k-1} R_j^{-1/2} · H_j · (TLM_{j:0}·δm)

        Requires running TLM to each observation time and accumulating.

        Parameters
        ----------
        delta_m : PETSc.Vec
            Perturbation in control space.

        Returns
        -------
        PETSc.Vec
            Perturbation in WME space.
        """
        from ..adjoint.tangent_linear import TangentLinearModel

        tlm = TangentLinearModel(self.forward_model, self._trajectory, self._jacobians)

        # Accumulate contributions
        num_obs = min(self.k, len(self.y_obs))
        result = None

        for j in range(num_obs):
            # Propagate perturbation to time j
            delta_u_j = tlm.propagate(delta_m, target_time=j)

            # Apply linearized observation operator
            delta_Hu_j = self.obs_op.forward_linearized(
                delta_u_j, self._trajectory[j], time_index=j
            )

            # Apply R^{-1/2}
            scaled = self._apply_R_sqrt_inv(delta_Hu_j, j)

            # Accumulate
            if result is None:
                result = scaled.copy()
            else:
                result.axpy(1.0, scaled)

        # Scale by 1/√k
        if result is not None and num_obs > 0:
            result.scale(1.0 / np.sqrt(num_obs))
        else:
            # Return zero if no observations
            result = delta_m.duplicate()
            result.zeroEntries()

        return result

    def apply_adjoint(self, delta_q: PETSc.Vec) -> PETSc.Vec:
        """
        Apply adjoint of linearized WME.

        δm = (1/√k) Σ_{j=0}^{k-1} ADJ_{j:0} · H_j^T · R_j^{-1/2} · δq_wme

        Requires running adjoint from each observation time and accumulating.

        Parameters
        ----------
        delta_q : PETSc.Vec
            Perturbation in WME space.

        Returns
        -------
        PETSc.Vec
            Perturbation in control space.
        """
        from ..adjoint.implicit_adjoint import ImplicitAdjointSolver

        num_obs = min(self.k, len(self.y_obs))

        # Scale input by 1/√k
        delta_q_scaled = delta_q.duplicate()
        delta_q.copy(delta_q_scaled)
        if num_obs > 0:
            delta_q_scaled.scale(1.0 / np.sqrt(num_obs))

        # Accumulate adjoint contributions
        result = self.m_bar.duplicate()
        result.zeroEntries()

        for j in range(num_obs):
            # Apply R^{-1/2}^T = R^{-1/2} (symmetric)
            scaled = self._apply_R_sqrt_inv(delta_q_scaled, j)

            # Apply adjoint observation operator
            delta_u_j = self.obs_op.adjoint(scaled, time_index=j)

            # Run adjoint from time j to 0
            if j > 0:
                adjoint_solver = ImplicitAdjointSolver(
                    self.forward_model,
                    self._trajectory[: j + 1],
                    self._jacobians[:j] if self._jacobians else None,
                    self.forward_model.dt,
                )

                forcings = [None] * (j + 1)
                forcings[j] = delta_u_j

                terminal = delta_u_j.duplicate()
                terminal.zeroEntries()

                delta_m_j = adjoint_solver.solve(terminal, forcings)
            else:
                # At j=0, delta_m = delta_u_0
                delta_m_j = delta_u_j

            # Accumulate
            result.axpy(1.0, delta_m_j)

        return result

    def _apply_R_sqrt_inv(self, v: PETSc.Vec, time_index: int) -> PETSc.Vec:
        """Apply R^{-1/2} to vector."""
        if hasattr(self.R_cov, "apply_sqrt_inverse"):
            return self.R_cov.apply_sqrt_inverse(v)
        else:
            # Fallback for diagonal covariance
            R_inv_v = self.R_cov.apply_inverse(v)
            result = v.duplicate()
            result.pointwiseMult(R_inv_v, v)
            result.sqrtabs()
            signs = v.duplicate()
            v.copy(signs)
            signs.sign()
            result.pointwiseMult(result, signs)
            return result


class QoICovarianceEstimator:
    """
    Estimates predicted error covariance L_k = Q_k B Q_k^T.

    Uses Monte Carlo sampling or ensemble methods to approximate
    the push-forward of background covariance through QoI map.

    The estimator generates samples ξᵢ ~ N(0, B), applies the linearized
    QoI, and computes the empirical covariance.

    Attributes
    ----------
    qoi_map : QoIMap
        QoI map to push covariance through.
    B : CovarianceMatrix
        Background covariance.
    num_samples : int
        Number of Monte Carlo samples.
    """

    def __init__(self, qoi_map: QoIMap, background_cov, num_samples: int = 100):
        """
        Initialize covariance estimator.

        Parameters
        ----------
        qoi_map : QoIMap
            QoI map to push covariance through.
        background_cov : CovarianceMatrix
            Background covariance B.
        num_samples : int
            Number of Monte Carlo samples.
        """
        self.qoi_map = qoi_map
        self.B = background_cov
        self.num_samples = num_samples

    def estimate(self, m_bar: PETSc.Vec, time_index: int):
        """
        Estimate L_k ≈ (1/N) Σᵢ (DQ_k·ξᵢ)(DQ_k·ξᵢ)^T where ξᵢ ~ N(0, B).

        Parameters
        ----------
        m_bar : PETSc.Vec
            Linearization point (typically m_background).
        time_index : int
            Time index k.

        Returns
        -------
        CovarianceMatrix
            Estimated covariance matrix L_k.
        """
        from .covariance import EnsembleCovariance

        # Linearize QoI at m_bar
        linearized_qoi = self.qoi_map.linearize(m_bar, time_index)

        # Generate samples and propagate through linearized QoI
        samples = []

        for i in range(self.num_samples):
            # Sample ξᵢ ~ N(0, B) using B^{1/2}·η where η ~ N(0, I)
            xi = self._sample_from_background()

            # Apply linearized QoI: ζᵢ = DQ_k·ξᵢ
            zeta = linearized_qoi.apply(xi)
            samples.append(zeta)

        # Return ensemble covariance
        return EnsembleCovariance(samples)

    def _sample_from_background(self) -> PETSc.Vec:
        """
        Sample from N(0, B) using B^{1/2}.

        Returns
        -------
        PETSc.Vec
            Sample vector.
        """
        # Create standard normal sample
        eta = self.B.create_vec()
        rng = np.random.default_rng()
        local_size = eta.getLocalSize()
        eta.setArray(rng.standard_normal(local_size))

        # Apply B^{1/2}
        return self.B.apply_sqrt(eta)

    def estimate_tlm_based(
        self, m_bar: PETSc.Vec, time_index: int, num_directions: int = 50
    ):
        """
        Estimate L_k using TLM-based approach.

        Uses randomized SVD-like approach:
        L_k ≈ (DQ_k · B · DQ_k^T)

        For efficiency, we approximate using a low-rank representation.

        Parameters
        ----------
        m_bar : PETSc.Vec
            Linearization point.
        time_index : int
            Time index k.
        num_directions : int
            Number of random directions for low-rank approximation.

        Returns
        -------
        CovarianceMatrix
            Low-rank approximation of L_k.
        """
        from .covariance import LowRankCovariance

        linearized_qoi = self.qoi_map.linearize(m_bar, time_index)

        # Random directions in QoI space
        U = []
        S = []

        for i in range(num_directions):
            # Random direction
            omega = self._sample_from_background()

            # Forward: DQ_k · B^{1/2} · omega
            y = linearized_qoi.apply(omega)

            U.append(y)
            S.append(y.norm())

        return LowRankCovariance(U, S)


class EnsembleCovariance:
    """
    Covariance matrix represented by ensemble of samples.

    C = (1/(N-1)) Σᵢ (xᵢ - x̄)(xᵢ - x̄)^T

    Attributes
    ----------
    samples : List[PETSc.Vec]
        Ensemble members.
    mean : PETSc.Vec
        Ensemble mean.
    """

    def __init__(self, samples: List[PETSc.Vec]):
        """
        Initialize from ensemble samples.

        Parameters
        ----------
        samples : List[PETSc.Vec]
            Ensemble members.
        """
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

    def apply(self, v: PETSc.Vec) -> PETSc.Vec:
        """
        Apply covariance: C·v = (1/(N-1)) Σᵢ aᵢ(aᵢ^T·v).

        Parameters
        ----------
        v : PETSc.Vec
            Input vector.

        Returns
        -------
        PETSc.Vec
            C·v.
        """
        result = v.duplicate()
        result.zeroEntries()

        for a in self.anomalies:
            coeff = a.dot(v)
            result.axpy(coeff, a)

        result.scale(1.0 / (self.n_samples - 1))
        return result

    def apply_inverse(self, v: PETSc.Vec) -> PETSc.Vec:
        """
        Apply inverse covariance using pseudo-inverse.

        Uses regularized inverse: (C + εI)^{-1}.

        Parameters
        ----------
        v : PETSc.Vec
            Input vector.

        Returns
        -------
        PETSc.Vec
            C^{-1}·v (regularized).
        """
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
        result = v.duplicate()
        result.zeroEntries()
        for i, a in enumerate(self.anomalies):
            result.axpy(c[i] / (n - 1), a)

        return result

    def create_vec(self) -> PETSc.Vec:
        """Create a vector compatible with this covariance."""
        return self.samples[0].duplicate()

    def min_eigenvalue(self) -> float:
        """Estimate minimum eigenvalue."""
        # Use Gershgorin or power iteration for estimate
        # For now, return a conservative estimate
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

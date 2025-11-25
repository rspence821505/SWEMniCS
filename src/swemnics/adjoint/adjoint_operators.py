"""
Adjoint operators for backward sensitivity propagation.

Implements discrete adjoint of the forward model for efficient
gradient computation via the adjoint method.

This module provides the building blocks for computing adjoints
in 4D-Var data assimilation:
  - AdjointModel: Backward time integration of adjoint equations
  - ObservationAdjoint: H^T operator for observation space
  - CovarianceAdjoint: R^{-1} and B^{-1} precision operators
  - CompositeAdjoint: Chain of adjoint operators
  - FiniteDifferenceAdjoint: Verification via adjoint test

Author: Rylan Spence
Date: 2025
"""

from typing import List, Optional
from petsc4py import PETSc


class AdjointModel:
    """
    Discrete adjoint model for backward sensitivity propagation.

    Propagates adjoint variables backward in time using
    transposed Jacobians from forward solve.

    The adjoint equations for a discrete time-dependent system are:

        J_n^T · λ^n = forcing^n

    where:
        - J_n = ∂R/∂u^{n+1} is the Jacobian from forward solve
        - forcing^n includes observation terms and time coupling
        - λ^n is the adjoint (Lagrange multiplier) at time n

    Attributes
    ----------
    forward_model : object
        Forward model reference (for utility methods).
    trajectory : List[PETSc.Vec]
        Forward trajectory [u₀, u₁, ..., uₙ].
    jacobians : List[PETSc.Mat]
        Jacobian matrices from forward solve [J₀, J₁, ..., J_{n-1}].
    num_steps : int
        Number of time steps.

    Notes
    -----
    This is a base class. For implicit BDF2 schemes, use
    ImplicitAdjointSolver which provides the full time-coupling logic.
    """

    def __init__(
        self, forward_model, trajectory: List[PETSc.Vec], jacobians: List[PETSc.Mat]
    ):
        """
        Initialize adjoint model.

        Parameters
        ----------
        forward_model : object
            Nonlinear forward model (for utility methods).
        trajectory : List[PETSc.Vec]
            Forward trajectory [u₀, u₁, ..., uₙ].
        jacobians : List[PETSc.Mat]
            Jacobian matrices from forward solve.

        Raises
        ------
        ValueError
            If trajectory and jacobians have inconsistent lengths.
        """
        self.forward_model = forward_model
        self.trajectory = trajectory
        self.jacobians = jacobians

        self.num_steps = len(trajectory) - 1

        # Validate
        if len(jacobians) != self.num_steps:
            raise ValueError(
                f"Jacobians length ({len(jacobians)}) must equal "
                f"num_steps ({self.num_steps})"
            )

    def solve(self, terminal_condition: PETSc.Vec) -> PETSc.Vec:
        """
        Solve adjoint equations backward in time.

        Computes gradient of cost function w.r.t. initial condition
        given terminal adjoint value.

        Parameters
        ----------
        terminal_condition : PETSc.Vec
            Adjoint variable at final time λₙ.

        Returns
        -------
        PETSc.Vec
            Adjoint variable at initial time λ₀.

        Notes
        -----
        The backward sweep goes from n = N-1 down to n = 0,
        solving transpose systems at each step.
        """
        # Initialize adjoint at final time
        lambda_next_next = None  # λⁿ⁺² (for three-level schemes)
        lambda_next = terminal_condition.copy()  # λⁿ⁺¹

        # Backward sweep
        for n in range(self.num_steps - 1, -1, -1):
            # Solve adjoint step
            lambda_n = self._solve_adjoint_step(n, lambda_next, lambda_next_next)

            # Shift for next iteration
            lambda_next_next = lambda_next
            lambda_next = lambda_n

        return lambda_next

    def _solve_adjoint_step(
        self, n: int, lambda_next: PETSc.Vec, lambda_next_next: Optional[PETSc.Vec]
    ) -> PETSc.Vec:
        """
        Solve one adjoint time step.

        For BDF2 scheme, this is overridden by ImplicitAdjointSolver
        which includes the full three-level time-coupling logic.

        Parameters
        ----------
        n : int
            Time index.
        lambda_next : PETSc.Vec
            λⁿ⁺¹ from previous backward step.
        lambda_next_next : PETSc.Vec or None
            λⁿ⁺² from two steps back (for three-level schemes).

        Returns
        -------
        PETSc.Vec
            λⁿ computed via transpose solve.

        Raises
        ------
        NotImplementedError
            This base class method must be overridden.
        """
        raise NotImplementedError(
            "Use ImplicitAdjointSolver for BDF2 time-stepping. "
            "This base class provides the framework only."
        )


class ObservationAdjoint:
    """
    Adjoint of observation operator.

    Maps observation-space innovation back to state space via
    the transpose observation operator:

        state_contrib = H^T · innovation

    This is a key component of the 4D-Var gradient computation,
    where innovation = R^{-1}(H(u) - y).

    Attributes
    ----------
    obs_op : object
        Forward observation operator (must have adjoint() method).

    Examples
    --------
    >>> obs_adj = ObservationAdjoint(observation_operator)
    >>> innovation = R_inv @ (H(u) - y_obs)
    >>> state_contribution = obs_adj.apply(innovation)
    """

    def __init__(self, observation_operator):
        """
        Initialize observation adjoint.

        Parameters
        ----------
        observation_operator : object
            Forward observation operator with adjoint() method.
        """
        self.obs_op = observation_operator

    def apply(self, innovation: PETSc.Vec) -> PETSc.Vec:
        """
        Apply adjoint observation operator: H^T·d.

        This maps observation-space vectors back to state space.

        Parameters
        ----------
        innovation : PETSc.Vec
            Innovation vector d = H(u) - y or weighted innovation
            R^{-1}(H(u) - y).

        Returns
        -------
        PETSc.Vec
            Adjoint contribution in state space.

        Notes
        -----
        The observation operator adjoint satisfies:
            ⟨H·u, v⟩ = ⟨u, H^T·v⟩
        for all vectors u in state space and v in observation space.
        """
        return self.obs_op.adjoint(innovation)


class CovarianceAdjoint:
    """
    Adjoint operations involving covariance matrices.

    Handles inverse covariance operations (precision matrices)
    that appear in adjoint computations:

    - Observation error: R^{-1}·v
    - Background error: B^{-1}·v

    These precision matrices weight innovations and background
    departures in the 4D-Var cost function.

    Attributes
    ----------
    cov : object
        Covariance matrix (must have apply_inverse() method).

    Examples
    --------
    >>> cov_adj = CovarianceAdjoint(R_matrix)
    >>> weighted = cov_adj.weight_innovation(H(u) - y_obs)
    """

    def __init__(self, covariance):
        """
        Initialize covariance adjoint.

        Parameters
        ----------
        covariance : object
            Covariance matrix with apply_inverse() method.
        """
        self.cov = covariance

    def apply_precision(self, v: PETSc.Vec) -> PETSc.Vec:
        """
        Apply precision matrix: Q·v = C^{-1}·v.

        The precision matrix is the inverse of the covariance matrix.

        Parameters
        ----------
        v : PETSc.Vec
            Input vector.

        Returns
        -------
        PETSc.Vec
            Precision-weighted vector C^{-1}·v.

        Notes
        -----
        For diagonal covariance with variances σ²:
            Q = diag(1/σ₁², 1/σ₂², ..., 1/σₙ²)
        """
        return self.cov.apply_inverse(v)

    def weight_innovation(self, innovation: PETSc.Vec) -> PETSc.Vec:
        """
        Weight innovation by observation error covariance.

        Computes: w = R^{-1}·(H(u) - y)

        This is the observation contribution to the 4D-Var gradient.

        Parameters
        ----------
        innovation : PETSc.Vec
            Innovation H(u) - y.

        Returns
        -------
        PETSc.Vec
            Weighted innovation R^{-1}·(H(u) - y).

        Notes
        -----
        The weighted innovation appears in the gradient as:
            ∇J_obs = H^T · R^{-1} · (H(u) - y)
        """
        return self.apply_precision(innovation)

    def weight_background(self, departure: PETSc.Vec) -> PETSc.Vec:
        """
        Weight background departure by background error covariance.

        Computes: w = B^{-1}·(m - m_b)

        This is the background contribution to the 4D-Var gradient.

        Parameters
        ----------
        departure : PETSc.Vec
            Background departure m - m_b.

        Returns
        -------
        PETSc.Vec
            Weighted departure B^{-1}·(m - m_b).

        Notes
        -----
        The weighted departure is the background gradient term:
            ∇J_bg = B^{-1}·(m - m_b)

        The total gradient is:
            ∇J = ∇J_bg + λ₀
        where λ₀ comes from the adjoint sweep.
        """
        return self.apply_precision(departure)


class CompositeAdjoint:
    """
    Adjoint of composite operators.

    For composition f(g(x)), the adjoint is applied in reverse order:

        (f ∘ g)^T = g^T ∘ f^T

    This is essential for chaining multiple operators in the
    observation-to-state mapping.

    Attributes
    ----------
    operators : List[object]
        List of operators in forward order.

    Examples
    --------
    >>> # Forward: y = op2(op1(x))
    >>> composite = CompositeAdjoint([op1, op2])
    >>> # Adjoint: x_adj = op1.adjoint(op2.adjoint(y_adj))
    >>> result = composite.apply(y_adj)

    Notes
    -----
    The adjoint test for composite operators:
        ⟨(f∘g)(x), y⟩ = ⟨x, (g^T∘f^T)(y)⟩
    """

    def __init__(self, operators: List):
        """
        Initialize composite adjoint.

        Parameters
        ----------
        operators : List[object]
            List of operators in forward order.
            Each must have an adjoint() method.
        """
        self.operators = operators

    def apply(self, adjoint_input: PETSc.Vec) -> PETSc.Vec:
        """
        Apply composite adjoint by chaining in reverse order.

        The adjoint of f(g(x)) is computed as:
            1. Apply f^T to input
            2. Apply g^T to result

        Parameters
        ----------
        adjoint_input : PETSc.Vec
            Input to composite adjoint (from output space).

        Returns
        -------
        PETSc.Vec
            Output of composite adjoint (in input space).

        Notes
        -----
        For n operators [op₁, op₂, ..., opₙ], the adjoint applies:
            opₙ^T ∘ ... ∘ op₂^T ∘ op₁^T
        in that order.
        """
        result = adjoint_input.copy()

        # Apply adjoints in reverse order
        for op in reversed(self.operators):
            result = op.adjoint(result)

        return result


class FiniteDifferenceAdjoint:
    """
    Finite difference adjoint for testing/validation.

    Verifies adjoint correctness via the adjoint test:

        ⟨v, J·w⟩ = ⟨J^T·v, w⟩

    This is the fundamental property that all discrete adjoints
    must satisfy. The test computes both sides and checks their
    relative difference.

    Attributes
    ----------
    operator : object
        Operator to test (must have forward() method).
    epsilon : float
        Finite difference step size.

    Examples
    --------
    >>> fd_adj = FiniteDifferenceAdjoint(obs_operator)
    >>> error = fd_adj.verify_adjoint(obs_adjoint, state, adj_state)
    >>> assert error < 1e-10, "Adjoint verification failed"

    Notes
    -----
    A correctly implemented adjoint should have error < 1e-10
    in double precision arithmetic.
    """

    def __init__(self, operator, epsilon: float = 1e-6):
        """
        Initialize FD adjoint tester.

        Parameters
        ----------
        operator : object
            Operator to test (must have forward() method).
        epsilon : float, optional
            Finite difference step size (default: 1e-6).
        """
        self.operator = operator
        self.epsilon = epsilon

    def verify_adjoint(
        self, adjoint_operator, state: PETSc.Vec, adjoint_state: PETSc.Vec
    ) -> float:
        """
        Verify adjoint correctness via inner product test.

        Computes the relative error in the adjoint test:

            error = |⟨v, J·w⟩ - ⟨J^T·v, w⟩| / |⟨v, J·w⟩|

        Parameters
        ----------
        adjoint_operator : object
            Claimed adjoint operator (must have apply() method).
        state : PETSc.Vec
            Forward state w.
        adjoint_state : PETSc.Vec
            Adjoint state v.

        Returns
        -------
        float
            Relative error in adjoint test.

        Notes
        -----
        For a correctly implemented adjoint:
            - error < 1e-10 indicates success
            - error > 1e-6 indicates likely bug
            - error > 0.1 indicates definite bug

        The test works in parallel (MPI) by using global dot products.
        """
        # Forward: J·w
        Jw = self.operator.forward(state)

        # Adjoint: J^T·v
        JTv = adjoint_operator.apply(adjoint_state)

        # Inner products (these are global operations in parallel)
        lhs = adjoint_state.dot(Jw)  # ⟨v, J·w⟩
        rhs = JTv.dot(state)  # ⟨J^T·v, w⟩

        # Relative error
        if abs(lhs) < 1e-14:
            # Handle near-zero case
            return abs(lhs - rhs)
        else:
            return abs(lhs - rhs) / abs(lhs)

    def verify_adjoint_detailed(
        self, adjoint_operator, state: PETSc.Vec, adjoint_state: PETSc.Vec
    ) -> dict:
        """
        Verify adjoint with detailed diagnostics.

        Parameters
        ----------
        adjoint_operator : object
            Claimed adjoint operator.
        state : PETSc.Vec
            Forward state w.
        adjoint_state : PETSc.Vec
            Adjoint state v.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'error': Relative error
            - 'lhs': ⟨v, J·w⟩
            - 'rhs': ⟨J^T·v, w⟩
            - 'passed': Boolean indicating if test passed
        """
        # Forward: J·w
        Jw = self.operator.forward(state)

        # Adjoint: J^T·v
        JTv = adjoint_operator.apply(adjoint_state)

        # Inner products
        lhs = adjoint_state.dot(Jw)
        rhs = JTv.dot(state)

        # Compute error
        if abs(lhs) < 1e-14:
            error = abs(lhs - rhs)
        else:
            error = abs(lhs - rhs) / abs(lhs)

        return {"error": error, "lhs": lhs, "rhs": rhs, "passed": error < 1e-10}


class TimeIntegratedAdjoint:
    """
    Helper class for time-integrated adjoint operations.

    Handles accumulation of adjoint contributions over time,
    particularly useful for computing sensitivities to parameters
    that affect multiple time steps.

    Attributes
    ----------
    num_steps : int
        Number of time steps.
    accumulated_adjoint : PETSc.Vec
        Accumulated adjoint sensitivity.
    """

    def __init__(self, num_steps: int, template_vec: PETSc.Vec):
        """
        Initialize time-integrated adjoint.

        Parameters
        ----------
        num_steps : int
            Number of time steps to integrate over.
        template_vec : PETSc.Vec
            Template vector for creating accumulator.
        """
        self.num_steps = num_steps
        self.accumulated_adjoint = template_vec.copy()
        self.accumulated_adjoint.set(0.0)

    def accumulate(self, adjoint_contrib: PETSc.Vec, weight: float = 1.0):
        """
        Accumulate adjoint contribution.

        Parameters
        ----------
        adjoint_contrib : PETSc.Vec
            Adjoint contribution at current time.
        weight : float, optional
            Weight for this contribution (default: 1.0).
        """
        self.accumulated_adjoint.axpy(weight, adjoint_contrib)

    def get_result(self) -> PETSc.Vec:
        """
        Get accumulated adjoint result.

        Returns
        -------
        PETSc.Vec
            Total accumulated adjoint.
        """
        return self.accumulated_adjoint.copy()

    def reset(self):
        """Reset accumulator to zero."""
        self.accumulated_adjoint.set(0.0)

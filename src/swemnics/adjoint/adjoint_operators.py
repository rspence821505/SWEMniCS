"""
Adjoint operators for backward sensitivity propagation.

Implements discrete adjoint of the forward model for efficient
gradient computation via the adjoint method.
"""

from typing import List, Optional
from petsc4py import PETSc


class AdjointModel:
    """
    Discrete adjoint model for backward sensitivity propagation.

    Propagates adjoint variables backward in time using
    transposed Jacobians from forward solve.
    """

    def __init__(
        self, forward_model, trajectory: List[PETSc.Vec], jacobians: List[PETSc.Mat]
    ):
        """
        Initialize adjoint model.

        Args:
            forward_model: Nonlinear forward model
            trajectory: Forward trajectory [u₀, u₁, ..., uₙ]
            jacobians: Jacobian matrices from forward solve
        """
        self.forward_model = forward_model
        self.trajectory = trajectory
        self.jacobians = jacobians

        self.num_steps = len(trajectory) - 1

    def solve(self, terminal_condition: PETSc.Vec) -> PETSc.Vec:
        """
        Solve adjoint equations backward in time.

        Computes gradient of cost function w.r.t. initial condition
        given terminal adjoint value.

        Args:
            terminal_condition: Adjoint variable at final time λₙ

        Returns:
            Adjoint variable at initial time λ₀
        """
        # Initialize adjoint at final time
        lambda_next_next = None  # λⁿ⁺²
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

        For BDF2 scheme, this is overridden by ImplicitAdjointSolver.

        Args:
            n: Time index
            lambda_next: λⁿ⁺¹
            lambda_next_next: λⁿ⁺² (or None if not applicable)

        Returns:
            λⁿ
        """
        raise NotImplementedError("Use ImplicitAdjointSolver for BDF2")


class ObservationAdjoint:
    """
    Adjoint of observation operator.

    Maps observation-space innovation back to state space.
    """

    def __init__(self, observation_operator):
        """
        Initialize observation adjoint.

        Args:
            observation_operator: Forward observation operator
        """
        self.obs_op = observation_operator

    def apply(self, innovation: PETSc.Vec) -> PETSc.Vec:
        """
        Apply adjoint observation operator: Hᵀ·d.

        Args:
            innovation: Innovation vector d = H(u) - y

        Returns:
            Adjoint contribution in state space
        """
        return self.obs_op.adjoint(innovation)


class CovarianceAdjoint:
    """
    Adjoint operations involving covariance matrices.

    Handles inverse covariance operations in adjoint computation.
    """

    def __init__(self, covariance):
        """
        Initialize covariance adjoint.

        Args:
            covariance: Covariance matrix (with apply_inverse method)
        """
        self.cov = covariance

    def apply_precision(self, v: PETSc.Vec) -> PETSc.Vec:
        """
        Apply precision matrix: Q·v = C⁻¹·v.

        Args:
            v: Input vector

        Returns:
            Precision-weighted vector
        """
        return self.cov.apply_inverse(v)

    def weight_innovation(self, innovation: PETSc.Vec) -> PETSc.Vec:
        """
        Weight innovation by observation error covariance.

        w = R⁻¹·(H(u) - y)

        Args:
            innovation: Innovation H(u) - y

        Returns:
            Weighted innovation
        """
        return self.apply_precision(innovation)


class CompositeAdjoint:
    """
    Adjoint of composite operators.

    For composition f(g(x)), adjoint is: (∇g)ᵀ · (∇f)ᵀ
    Applied in reverse order.
    """

    def __init__(self, operators: List):
        """
        Initialize composite adjoint.

        Args:
            operators: List of operators in forward order
        """
        self.operators = operators

    def apply(self, adjoint_input: PETSc.Vec) -> PETSc.Vec:
        """
        Apply composite adjoint by chaining in reverse order.

        Args:
            adjoint_input: Input to composite adjoint

        Returns:
            Output of composite adjoint
        """
        result = adjoint_input.copy()

        # Apply adjoints in reverse order
        for op in reversed(self.operators):
            result = op.adjoint(result)

        return result


class FiniteDifferenceAdjoint:
    """
    Finite difference adjoint for testing/validation.

    Verifies adjoint correctness via adjoint test:
    ⟨v, J·w⟩ = ⟨Jᵀ·v, w⟩
    """

    def __init__(self, operator, epsilon: float = 1e-6):
        """
        Initialize FD adjoint tester.

        Args:
            operator: Operator to test (must have forward() method)
            epsilon: Finite difference step
        """
        self.operator = operator
        self.epsilon = epsilon

    def verify_adjoint(
        self, adjoint_operator, state: PETSc.Vec, adjoint_state: PETSc.Vec
    ) -> float:
        """
        Verify adjoint correctness via inner product test.

        Checks: |⟨v, J·w⟩ - ⟨Jᵀ·v, w⟩| / |⟨v, J·w⟩|

        Args:
            adjoint_operator: Claimed adjoint operator
            state: Forward state w
            adjoint_state: Adjoint state v

        Returns:
            Relative error in adjoint test
        """
        # Forward: J·w
        Jw = self.operator.forward(state)

        # Adjoint: Jᵀ·v
        JTv = adjoint_operator.apply(adjoint_state)

        # Inner products
        lhs = adjoint_state.dot(Jw)
        rhs = JTv.dot(state)

        # Relative error
        if abs(lhs) < 1e-14:
            return abs(lhs - rhs)
        else:
            return abs(lhs - rhs) / abs(lhs)

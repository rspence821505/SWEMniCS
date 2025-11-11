"""
Tangent Linear Model (TLM) for forward sensitivity propagation.

Implements linearized forward model: δuₙ₊₁ = J·δuₙ
where J is the Jacobian of the implicit time-stepping scheme.
"""

from typing import List, Optional, Tuple
from petsc4py import PETSc


class TangentLinearModel:
    """
    Tangent Linear Model for sensitivity propagation.

    Propagates perturbations forward in time using
    linearization of the nonlinear forward model.
    """

    def __init__(
        self,
        forward_model,
        trajectory: List[PETSc.Vec],
        jacobians: Optional[List[PETSc.Mat]] = None,
    ):
        """
        Initialize TLM from forward trajectory.

        Args:
            forward_model: Nonlinear forward model
            trajectory: Reference trajectory [u₀, u₁, ..., uₙ]
            jacobians: Optional precomputed Jacobians
        """
        self.forward_model = forward_model
        self.trajectory = trajectory
        self.jacobians = jacobians

    def propagate_perturbation(
        self, delta_u0: PETSc.Vec, start_time: int = 0, end_time: Optional[int] = None
    ) -> List[PETSc.Vec]:
        """
        Propagate initial perturbation forward in time.

        Solves: δuₙ₊₁ = J_n · δuₙ for n = start_time, ..., end_time-1

        Args:
            delta_u0: Initial perturbation
            start_time: Starting time index
            end_time: Ending time index (None = full trajectory)

        Returns:
            List of perturbations [δu₀, δu₁, ..., δuₖ]
        """
        # TODO: Implement TLM time stepping
        # For implicit BDF2:
        # J · δuₙ₊₁ = RHS(δuₙ, δuₙ₋₁)
        # where J is cached from forward solve
        pass

    def compute_sensitivity(
        self, delta_u0: PETSc.Vec, observation_operator, obs_time: int
    ) -> PETSc.Vec:
        """
        Compute sensitivity of observations to initial perturbation.

        δy = H · TLM(δu₀)

        Args:
            delta_u0: Initial perturbation
            observation_operator: Observation operator H
            obs_time: Observation time index

        Returns:
            Perturbation in observation space
        """
        # Propagate to observation time
        perturbations = self.propagate_perturbation(delta_u0, end_time=obs_time + 1)
        delta_u_obs = perturbations[obs_time]

        # Apply observation operator
        return observation_operator.forward(delta_u_obs)


class ImplicitTLMSolver:
    """
    Solver for implicit TLM time steps.

    For BDF2 implicit scheme, each TLM step requires solving:
    J · δuₙ₊₁ = RHS(δuₙ, δuₙ₋₁)
    """

    def __init__(self, dt: float):
        """
        Initialize implicit TLM solver.

        Args:
            dt: Time step size
        """
        self.dt = dt

        # KSP solver for linear systems
        self.ksp = None

    def solve_tlm_step(
        self, jacobian: PETSc.Mat, delta_u_n: PETSc.Vec, delta_u_nm1: PETSc.Vec
    ) -> PETSc.Vec:
        """
        Solve one TLM time step.

        For BDF2 linearization:
        J · δuₙ₊₁ = -(4/(2Δt)) M · δuₙ + (1/(2Δt)) M · δuₙ₋₁

        Args:
            jacobian: Jacobian matrix J = ∂R/∂u from forward solve
            delta_u_n: Perturbation at time n
            delta_u_nm1: Perturbation at time n-1

        Returns:
            Perturbation at time n+1
        """
        # TODO: Implement implicit TLM step
        # 1. Assemble RHS from BDF2 time coupling
        # 2. Solve J · δuₙ₊₁ = RHS
        pass

    def _assemble_tlm_rhs(
        self, delta_u_n: PETSc.Vec, delta_u_nm1: PETSc.Vec
    ) -> PETSc.Vec:
        """
        Assemble RHS for TLM step from BDF2 time coupling.

        RHS = -(4/(2Δt)) M · δuₙ + (1/(2Δt)) M · δuₙ₋₁

        Args:
            delta_u_n: Perturbation at time n
            delta_u_nm1: Perturbation at time n-1

        Returns:
            RHS vector
        """
        # TODO: Implement BDF2 time coupling
        pass

    def _setup_ksp(self, jacobian: PETSc.Mat):
        """
        Set up KSP solver for Jacobian system.

        Args:
            jacobian: System matrix
        """
        if self.ksp is None:
            self.ksp = PETSc.KSP().create()
            self.ksp.setType(PETSc.KSP.Type.GMRES)
            self.ksp.getPC().setType(PETSc.PC.Type.ILU)

        self.ksp.setOperators(jacobian)


class FiniteDifferenceTLM:
    """
    Finite difference approximation of TLM for testing.

    Computes TLM via forward differences:
    TLM(δu) ≈ (M(u + ε·δu) - M(u)) / ε
    """

    def __init__(self, forward_model, epsilon: float = 1e-6):
        """
        Initialize finite difference TLM.

        Args:
            forward_model: Nonlinear forward model
            epsilon: Finite difference step size
        """
        self.forward_model = forward_model
        self.epsilon = epsilon

    def apply(
        self, u_base: PETSc.Vec, delta_u: PETSc.Vec, num_steps: int = 1
    ) -> PETSc.Vec:
        """
        Apply TLM via finite differences.

        Args:
            u_base: Base state
            delta_u: Perturbation
            num_steps: Number of time steps to integrate

        Returns:
            Propagated perturbation
        """
        # Perturbed initial condition
        u_pert = u_base.copy()
        u_pert.axpy(self.epsilon, delta_u)

        # Solve both trajectories
        traj_base, _ = self.forward_model.solve(u_base, store_jacobians=False)
        traj_pert, _ = self.forward_model.solve(u_pert, store_jacobians=False)

        # Finite difference approximation
        delta_u_final = traj_pert[num_steps].copy()
        delta_u_final.axpy(-1.0, traj_base[num_steps])
        delta_u_final.scale(1.0 / self.epsilon)

        return delta_u_final

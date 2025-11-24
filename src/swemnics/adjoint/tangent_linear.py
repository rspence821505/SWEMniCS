"""
Tangent Linear Model (TLM) for forward sensitivity propagation.

Implements linearized forward model for propagating perturbations
through the nonlinear dynamics. For BDF2 implicit time-stepping:

    δu^{n+1} = TLM_step(δu^n, δu^{n-1})

where the TLM step requires solving:

    J · δu^{n+1} = RHS(δu^n, δu^{n-1})

and J is the Jacobian cached from the forward Newton solve.

Mathematical Background
-----------------------
For BDF2 discretization of the forward model:

    R(u^{n+1}; u^n, u^{n-1}) = (3u^{n+1} - 4u^n + u^{n-1})/(2Δt) + F(u^{n+1}) = 0

The linearization (TLM) gives:

    J · δu^{n+1} = (4/(2Δt))·M·δu^n - (1/(2Δt))·M·δu^{n-1}

where J = ∂R/∂u = (3/(2Δt))·M + ∂F/∂u is already computed during the forward solve.

Key Advantage
-------------
By reusing Jacobians from the forward Newton solve, the TLM has cost
approximately equal to one linear solve per time step (vs. full
nonlinear solve for finite difference approximation).
"""

from typing import List, Optional, Tuple, Dict
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np


class TangentLinearModel:
    """
    Tangent Linear Model for sensitivity propagation.

    Propagates perturbations forward in time using
    linearization of the nonlinear forward model.

    Attributes
    ----------
    forward_model : ForwardModel
        Nonlinear forward model (for accessing dt, mass matrix).
    trajectory : List[PETSc.Vec]
        Reference trajectory [u₀, u₁, ..., uₙ] from forward solve.
    jacobians : List[PETSc.Mat]
        Jacobian matrices from forward Newton solves.
    dt : float
        Time step size.
    num_steps : int
        Number of time steps in trajectory.

    Examples
    --------
    >>> # Create TLM from forward solve results
    >>> trajectory, jacobians = forward_model.solve(m, store_jacobians=True)
    >>> tlm = TangentLinearModel(forward_model, trajectory, jacobians)
    >>>
    >>> # Propagate perturbation to observation time
    >>> perturbations = tlm.propagate_perturbation(delta_m, end_time=10)
    >>> delta_u_k = perturbations[-1]
    """

    def __init__(
        self,
        forward_model,
        trajectory: List[PETSc.Vec],
        jacobians: Optional[List[PETSc.Mat]] = None,
    ):
        """
        Initialize TLM from forward trajectory.

        Parameters
        ----------
        forward_model : ForwardModel
            Nonlinear forward model.
        trajectory : List[PETSc.Vec]
            Reference trajectory [u₀, u₁, ..., uₙ].
        jacobians : List[PETSc.Mat], optional
            Precomputed Jacobians from forward solve. If None,
            Jacobians will be recomputed as needed (less efficient).
        """
        self.forward_model = forward_model
        self.trajectory = trajectory
        self.jacobians = jacobians

        # Extract time step size from forward model
        self.dt = getattr(forward_model, "dt", 1.0)
        self.num_steps = len(trajectory) - 1

        # Mass matrix for BDF2 time coupling (lazy initialization)
        self._mass_matrix: Optional[PETSc.Mat] = None

        # Implicit TLM solver
        self._tlm_solver: Optional[ImplicitTLMSolver] = None

        # Cache for propagated perturbations (keyed by delta_u0 id)
        self._perturbation_cache: Dict[int, List[PETSc.Vec]] = {}

    def propagate_perturbation(
        self,
        delta_u0: PETSc.Vec,
        start_time: int = 0,
        end_time: Optional[int] = None,
    ) -> List[PETSc.Vec]:
        """
        Propagate initial perturbation forward in time.

        Solves: δu^{n+1} = TLM_step(δu^n, δu^{n-1}) for n = start_time, ..., end_time-1

        For BDF2 implicit scheme, each step requires solving:

            J · δu^{n+1} = (4/(2Δt))·M·δu^n - (1/(2Δt))·M·δu^{n-1}

        where J is the Jacobian cached from the forward Newton solve.

        Parameters
        ----------
        delta_u0 : PETSc.Vec
            Initial perturbation.
        start_time : int
            Starting time index.
        end_time : int, optional
            Ending time index (None = full trajectory).

        Returns
        -------
        List[PETSc.Vec]
            List of perturbations [δu₀, δu₁, ..., δu_end].
        """
        if end_time is None:
            end_time = self.num_steps

        if end_time > self.num_steps:
            raise ValueError(
                f"end_time {end_time} exceeds trajectory length {self.num_steps}"
            )

        # Initialize solver if needed
        if self._tlm_solver is None:
            mass_matrix = self._get_mass_matrix()
            comm = delta_u0.getComm()
            self._tlm_solver = ImplicitTLMSolver(self.dt, mass_matrix, comm)

        # Initialize perturbation history
        perturbations = [delta_u0.copy()]

        # For BDF2, we need two previous perturbations
        # At n=0: δu^{-1} doesn't exist, so we use δu^{-1} = 0
        delta_u_nm1: Optional[PETSc.Vec] = None  # δu^{n-1}
        delta_u_n = delta_u0.copy()  # δu^n

        # Forward propagation through time steps
        for n in range(start_time, end_time):
            # Get Jacobian at this time step (cached or recomputed)
            J = self._get_jacobian(n)

            # Solve TLM step: J · δu^{n+1} = RHS(δu^n, δu^{n-1})
            delta_u_next = self._tlm_solver.solve_tlm_step(J, delta_u_n, delta_u_nm1)

            perturbations.append(delta_u_next.copy())

            # Shift history for next step
            delta_u_nm1 = delta_u_n
            delta_u_n = delta_u_next

        return perturbations

    def propagate(
        self,
        delta_u0: PETSc.Vec,
        target_time: int,
        cache_intermediate: bool = False,
    ) -> PETSc.Vec:
        """
        Propagate initial perturbation to a specific target time.

        Convenience method that returns only the final perturbation.

        Parameters
        ----------
        delta_u0 : PETSc.Vec
            Initial perturbation δu₀.
        target_time : int
            Target time index k.
        cache_intermediate : bool
            Whether to cache intermediate perturbations for reuse.

        Returns
        -------
        PETSc.Vec
            Propagated perturbation δu_k at target time.
        """
        if target_time <= 0:
            return delta_u0.copy()

        perturbations = self.propagate_perturbation(
            delta_u0, start_time=0, end_time=target_time
        )

        if cache_intermediate:
            cache_key = id(delta_u0)
            self._perturbation_cache[cache_key] = perturbations

        return perturbations[-1]

    def compute_sensitivity(
        self,
        delta_u0: PETSc.Vec,
        observation_operator,
        obs_time: int,
    ) -> PETSc.Vec:
        """
        Compute sensitivity of observations to initial perturbation.

        Evaluates: δy = H · TLM_{k:0}(δu₀)

        where H is the observation operator and TLM_{k:0} propagates
        perturbations from time 0 to time k.

        Parameters
        ----------
        delta_u0 : PETSc.Vec
            Initial perturbation.
        observation_operator : ObservationOperator
            Observation operator H.
        obs_time : int
            Observation time index.

        Returns
        -------
        PETSc.Vec
            Perturbation in observation space.
        """
        # Propagate perturbation to observation time
        perturbations = self.propagate_perturbation(delta_u0, end_time=obs_time + 1)
        delta_u_obs = perturbations[obs_time]

        # Apply observation operator (linearized if available)
        if hasattr(observation_operator, "forward_linearized"):
            return observation_operator.forward_linearized(
                delta_u_obs, self.trajectory[obs_time], time_index=obs_time
            )
        else:
            return observation_operator.forward(delta_u_obs, time_index=obs_time)

    def _get_jacobian(self, n: int) -> PETSc.Mat:
        """
        Get Jacobian at time step n.

        Uses cached Jacobian if available, otherwise recomputes.

        Parameters
        ----------
        n : int
            Time step index.

        Returns
        -------
        PETSc.Mat
            Jacobian matrix J = ∂R/∂u at time step n.
        """
        if self.jacobians is not None and n < len(self.jacobians):
            return self.jacobians[n]
        else:
            return self._compute_jacobian(n)

    def _compute_jacobian(self, n: int) -> PETSc.Mat:
        """
        Compute Jacobian at time step n (fallback when not cached).

        Parameters
        ----------
        n : int
            Time step index.

        Returns
        -------
        PETSc.Mat
            Jacobian matrix.
        """
        if hasattr(self.forward_model, "assemble_jacobian"):
            # Get states needed for Jacobian assembly
            u_np1 = self.trajectory[n + 1]
            u_n = self.trajectory[n]
            u_nm1 = self.trajectory[n - 1] if n > 0 else None
            return self.forward_model.assemble_jacobian(u_np1, u_n, u_nm1)
        else:
            # Fallback: identity scaled by BDF2 coefficient
            # Only valid for testing with trivial dynamics
            M = self._get_mass_matrix()
            J = M.duplicate()
            M.copy(J)
            J.scale(3.0 / (2.0 * self.dt))
            return J

    def _get_mass_matrix(self) -> PETSc.Mat:
        """
        Get or assemble mass matrix M.

        Returns
        -------
        PETSc.Mat
            Mass matrix.
        """
        if self._mass_matrix is not None:
            return self._mass_matrix

        # Try to get from forward model
        if hasattr(self.forward_model, "get_mass_matrix"):
            self._mass_matrix = self.forward_model.get_mass_matrix()
        elif hasattr(self.forward_model, "mass_matrix"):
            self._mass_matrix = self.forward_model.mass_matrix
        else:
            # Fallback: create identity matrix
            n = self.trajectory[0].getSize()
            comm = self.trajectory[0].getComm()

            self._mass_matrix = PETSc.Mat().createAIJ(size=[n, n], nnz=1, comm=comm)
            istart, iend = self._mass_matrix.getOwnershipRange()
            for i in range(istart, iend):
                self._mass_matrix.setValue(i, i, 1.0)
            self._mass_matrix.assemble()

        return self._mass_matrix

    def clear_cache(self):
        """Clear perturbation cache to free memory."""
        self._perturbation_cache.clear()


class ImplicitTLMSolver:
    """
    Solver for implicit TLM time steps.

    For BDF2 implicit scheme, each TLM step requires solving:

        J · δu^{n+1} = RHS(δu^n, δu^{n-1})

    where:
        - J = (3/(2Δt))·M + ∂F/∂u is the Jacobian from forward Newton solve
        - RHS = (4/(2Δt))·M·δu^n - (1/(2Δt))·M·δu^{n-1}

    Attributes
    ----------
    dt : float
        Time step size.
    mass_matrix : PETSc.Mat
        Mass matrix M for time coupling.
    ksp : PETSc.KSP
        Linear solver for Jacobian systems.
    """

    def __init__(
        self,
        dt: float,
        mass_matrix: Optional[PETSc.Mat] = None,
        comm: Optional[MPI.Comm] = None,
    ):
        """
        Initialize implicit TLM solver.

        Parameters
        ----------
        dt : float
            Time step size.
        mass_matrix : PETSc.Mat, optional
            Mass matrix for time coupling.
        comm : MPI.Comm, optional
            MPI communicator.
        """
        self.dt = dt
        self.mass_matrix = mass_matrix
        self.comm = comm if comm is not None else MPI.COMM_WORLD

        # KSP solver for linear systems
        self.ksp: Optional[PETSc.KSP] = None

    def solve_tlm_step(
        self,
        jacobian: PETSc.Mat,
        delta_u_n: PETSc.Vec,
        delta_u_nm1: Optional[PETSc.Vec] = None,
    ) -> PETSc.Vec:
        """
        Solve one TLM time step.

        For BDF2 linearization:

            J · δu^{n+1} = (4/(2Δt))·M·δu^n - (1/(2Δt))·M·δu^{n-1}

        Parameters
        ----------
        jacobian : PETSc.Mat
            Jacobian matrix J = ∂R/∂u from forward solve.
        delta_u_n : PETSc.Vec
            Perturbation at time n.
        delta_u_nm1 : PETSc.Vec, optional
            Perturbation at time n-1 (None for first step, treated as zero).

        Returns
        -------
        PETSc.Vec
            Perturbation at time n+1.
        """
        # Assemble RHS from BDF2 time coupling
        rhs = self._assemble_tlm_rhs(delta_u_n, delta_u_nm1)

        # Set up KSP solver with current Jacobian
        self._setup_ksp(jacobian)

        # Solve J · δu^{n+1} = rhs
        delta_u_next = rhs.duplicate()
        self.ksp.solve(rhs, delta_u_next)

        # Check convergence
        reason = self.ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"TLM linear solve failed with reason {reason}")

        return delta_u_next

    def _assemble_tlm_rhs(
        self,
        delta_u_n: PETSc.Vec,
        delta_u_nm1: Optional[PETSc.Vec],
    ) -> PETSc.Vec:
        """
        Assemble RHS for TLM step from BDF2 time coupling.

        RHS = (4/(2Δt))·M·δu^n - (1/(2Δt))·M·δu^{n-1}

        For the first step (when δu^{n-1} is None), we treat δu^{n-1} = 0:

            RHS = (4/(2Δt))·M·δu^n

        Parameters
        ----------
        delta_u_n : PETSc.Vec
            Perturbation at time n.
        delta_u_nm1 : PETSc.Vec, optional
            Perturbation at time n-1.

        Returns
        -------
        PETSc.Vec
            RHS vector.
        """
        M = self._get_mass_matrix(delta_u_n)

        # Compute (4/(2Δt))·M·δu^n
        rhs = delta_u_n.duplicate()
        M.mult(delta_u_n, rhs)
        rhs.scale(4.0 / (2.0 * self.dt))

        # Subtract (1/(2Δt))·M·δu^{n-1} if available
        if delta_u_nm1 is not None:
            temp = delta_u_n.duplicate()
            M.mult(delta_u_nm1, temp)
            rhs.axpy(-1.0 / (2.0 * self.dt), temp)
            temp.destroy()

        return rhs

    def _get_mass_matrix(self, template: PETSc.Vec) -> PETSc.Mat:
        """
        Get or create mass matrix.

        Parameters
        ----------
        template : PETSc.Vec
            Template vector for size/comm info.

        Returns
        -------
        PETSc.Mat
            Mass matrix.
        """
        if self.mass_matrix is not None:
            return self.mass_matrix

        # Create identity as default mass matrix
        n = template.getSize()
        self.mass_matrix = PETSc.Mat().createAIJ(size=[n, n], nnz=1, comm=self.comm)
        istart, iend = self.mass_matrix.getOwnershipRange()
        for i in range(istart, iend):
            self.mass_matrix.setValue(i, i, 1.0)
        self.mass_matrix.assemble()

        return self.mass_matrix

    def _setup_ksp(self, jacobian: PETSc.Mat):
        """
        Set up KSP solver for Jacobian system.

        Parameters
        ----------
        jacobian : PETSc.Mat
            System matrix (Jacobian).
        """
        if self.ksp is None:
            self.ksp = PETSc.KSP().create(self.comm)
            self.ksp.setType(PETSc.KSP.Type.GMRES)
            self.ksp.getPC().setType(PETSc.PC.Type.ILU)
            self.ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

        self.ksp.setOperators(jacobian)


class FiniteDifferenceTLM:
    """
    Finite difference approximation of TLM for testing.

    Computes TLM via forward differences:

        TLM(δu) ≈ (M(u + ε·δu) - M(u)) / ε

    where M is the forward model operator. This is useful for
    validating the analytical TLM implementation via Taylor
    remainder tests.

    Attributes
    ----------
    forward_model : ForwardModel
        Nonlinear forward model.
    epsilon : float
        Finite difference step size.
    """

    def __init__(self, forward_model, epsilon: float = 1e-6):
        """
        Initialize finite difference TLM.

        Parameters
        ----------
        forward_model : ForwardModel
            Nonlinear forward model.
        epsilon : float
            Finite difference step size (default 1e-6).
        """
        self.forward_model = forward_model
        self.epsilon = epsilon

    def apply(
        self,
        u_base: PETSc.Vec,
        delta_u: PETSc.Vec,
        num_steps: int = 1,
    ) -> PETSc.Vec:
        """
        Apply TLM via finite differences.

        Computes: (M_{k:0}(u + ε·δu) - M_{k:0}(u)) / ε

        Parameters
        ----------
        u_base : PETSc.Vec
            Base initial condition.
        delta_u : PETSc.Vec
            Perturbation direction.
        num_steps : int
            Number of time steps to integrate (target time).

        Returns
        -------
        PETSc.Vec
            Finite difference approximation of TLM(δu).
        """
        # Perturbed initial condition: u + ε·δu
        u_pert = u_base.duplicate()
        u_base.copy(u_pert)
        u_pert.axpy(self.epsilon, delta_u)

        # Solve both trajectories
        traj_base, _ = self.forward_model.solve(u_base, store_jacobians=False)
        traj_pert, _ = self.forward_model.solve(u_pert, store_jacobians=False)

        # Finite difference approximation: (u_pert(k) - u_base(k)) / ε
        delta_u_final = traj_pert[num_steps].duplicate()
        traj_pert[num_steps].copy(delta_u_final)
        delta_u_final.axpy(-1.0, traj_base[num_steps])
        delta_u_final.scale(1.0 / self.epsilon)

        return delta_u_final

    def apply_trajectory(
        self,
        u_base: PETSc.Vec,
        delta_u: PETSc.Vec,
    ) -> List[PETSc.Vec]:
        """
        Apply FD-TLM and return full perturbation trajectory.

        Parameters
        ----------
        u_base : PETSc.Vec
            Base initial condition.
        delta_u : PETSc.Vec
            Perturbation direction.

        Returns
        -------
        List[PETSc.Vec]
            FD approximation of perturbation trajectory [δu₀, δu₁, ...].
        """
        # Perturbed initial condition
        u_pert = u_base.duplicate()
        u_base.copy(u_pert)
        u_pert.axpy(self.epsilon, delta_u)

        # Solve both trajectories
        traj_base, _ = self.forward_model.solve(u_base, store_jacobians=False)
        traj_pert, _ = self.forward_model.solve(u_pert, store_jacobians=False)

        # Compute FD perturbation trajectory
        delta_traj = []
        for u_b, u_p in zip(traj_base, traj_pert):
            delta = u_p.duplicate()
            u_p.copy(delta)
            delta.axpy(-1.0, u_b)
            delta.scale(1.0 / self.epsilon)
            delta_traj.append(delta)

        return delta_traj


class TLMValidator:
    """
    Validator for TLM correctness using Taylor remainder tests.

    Verifies that the TLM correctly linearizes the forward model
    by checking Taylor remainder convergence:

        r(ε) = ||M(m + ε·δm) - M(m) - ε·TLM(δm)||

    should converge as O(ε²), i.e., r(ε/2)/r(ε) ≈ 0.25.

    Attributes
    ----------
    forward_model : ForwardModel
        Nonlinear forward model.
    tlm : TangentLinearModel
        Tangent linear model to validate.
    """

    def __init__(self, forward_model, tlm: TangentLinearModel):
        """
        Initialize validator.

        Parameters
        ----------
        forward_model : ForwardModel
            Nonlinear forward model.
        tlm : TangentLinearModel
            Tangent linear model to validate.
        """
        self.forward_model = forward_model
        self.tlm = tlm

    def taylor_test(
        self,
        m: PETSc.Vec,
        delta_m: PETSc.Vec,
        target_time: int,
        epsilons: Optional[List[float]] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Perform Taylor test for TLM correctness.

        Checks: M(m + ε·δm) - M(m) ≈ ε·TLM(m)·δm + O(ε²)

        Parameters
        ----------
        m : PETSc.Vec
            Linearization point (initial condition).
        delta_m : PETSc.Vec
            Perturbation direction.
        target_time : int
            Time index to evaluate at.
        epsilons : List[float], optional
            Perturbation magnitudes to test.

        Returns
        -------
        Tuple[List[float], List[float]]
            (remainders, convergence_ratios)
        """
        if epsilons is None:
            epsilons = [1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4]

        # Baseline trajectory: M(m)
        traj_base, _ = self.forward_model.solve(m, store_jacobians=False)
        u_base = traj_base[target_time]

        # TLM prediction: TLM_{k:0}·δm
        delta_u_tlm = self.tlm.propagate(delta_m, target_time=target_time)

        remainders = []
        for eps in epsilons:
            # Perturbed trajectory: M(m + ε·δm)
            m_pert = m.duplicate()
            m.copy(m_pert)
            m_pert.axpy(eps, delta_m)

            traj_pert, _ = self.forward_model.solve(m_pert, store_jacobians=False)
            u_pert = traj_pert[target_time]

            # Compute Taylor remainder:
            # r = ||M(m + ε·δm) - M(m) - ε·TLM·δm||
            diff = u_pert.duplicate()
            u_pert.copy(diff)
            diff.axpy(-1.0, u_base)  # M(m+ε·δm) - M(m)
            diff.axpy(-eps, delta_u_tlm)  # - ε·TLM·δm

            remainder = diff.norm()
            remainders.append(remainder)

        # Compute convergence ratios: r(ε_{i+1})/r(ε_i)
        ratios = []
        for i in range(len(remainders) - 1):
            if remainders[i] > 1e-14:
                ratios.append(remainders[i + 1] / remainders[i])
            else:
                ratios.append(0.0)

        return remainders, ratios

    def verify_convergence(
        self,
        m: PETSc.Vec,
        delta_m: PETSc.Vec,
        target_time: int,
        expected_rate: float = 0.25,
        tolerance: float = 0.15,
    ) -> Tuple[bool, List[float]]:
        """
        Verify TLM convergence rate.

        For correct TLM, the Taylor remainder should converge as O(ε²),
        meaning successive ratios should be approximately 0.25 when
        epsilon is halved.

        Parameters
        ----------
        m : PETSc.Vec
            Linearization point.
        delta_m : PETSc.Vec
            Perturbation direction.
        target_time : int
            Time index to evaluate at.
        expected_rate : float
            Expected convergence ratio (0.25 for O(ε²)).
        tolerance : float
            Acceptable deviation from expected rate.

        Returns
        -------
        Tuple[bool, List[float]]
            (converged_correctly, ratios)
        """
        _, ratios = self.taylor_test(m, delta_m, target_time)

        # Check if all ratios are within tolerance of expected rate
        converged = True
        for ratio in ratios:
            if ratio > 0 and abs(ratio - expected_rate) > tolerance:
                converged = False
                break

        return converged, ratios

    def compare_with_finite_difference(
        self,
        m: PETSc.Vec,
        delta_m: PETSc.Vec,
        target_time: int,
        fd_epsilon: float = 1e-6,
    ) -> float:
        """
        Compare TLM with finite difference approximation.

        Parameters
        ----------
        m : PETSc.Vec
            Initial condition.
        delta_m : PETSc.Vec
            Perturbation direction.
        target_time : int
            Time index to evaluate at.
        fd_epsilon : float
            Finite difference step size.

        Returns
        -------
        float
            Relative difference between TLM and FD.
        """
        # Analytical TLM result
        delta_u_tlm = self.tlm.propagate(delta_m, target_time=target_time)

        # Finite difference result
        fd_tlm = FiniteDifferenceTLM(self.forward_model, epsilon=fd_epsilon)
        delta_u_fd = fd_tlm.apply(m, delta_m, num_steps=target_time)

        # Compute relative difference
        diff = delta_u_tlm.duplicate()
        delta_u_tlm.copy(diff)
        diff.axpy(-1.0, delta_u_fd)

        fd_norm = delta_u_fd.norm()
        if fd_norm > 1e-14:
            rel_diff = diff.norm() / fd_norm
        else:
            rel_diff = diff.norm()

        return rel_diff

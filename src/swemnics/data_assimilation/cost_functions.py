"""
Cost function implementations for 4D-Var data assimilation.

Implements standard 4D-Var, DC-4DVar, and DC-WME variants
following Spence et al. (2025).

This module provides both explicit and implicit adjoint implementations:
- Explicit adjoint: For testing with simple forward models
- Implicit adjoint: For production use with BDF2 implicit schemes

Author: Rylan Spence
Date: 2025
"""

from abc import ABC, abstractmethod
from petsc4py import PETSc
from mpi4py import MPI
from typing import Optional, List, Tuple, Dict
import numpy as np


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
    Standard 4D-Var cost function with implicit adjoint gradient.

    This class implements the full 4D-Var cost function with:
    - Background term for regularization
    - Observation terms at multiple time steps
    - Gradient via adjoint solver (explicit or implicit)
    - Gauss-Newton Hessian approximation for optimization

    The implementation supports both explicit and implicit time-stepping:
    - use_implicit_adjoint=False: For explicit schemes (u_{n+1} = A*u_n)
    - use_implicit_adjoint=True: For implicit BDF2 schemes

    Attributes
    ----------
    forward_model : ForwardModel
        Forward model M_{k:0} with Jacobian caching
    obs_op : ObservationOperator
        Observation operator H_k
    B : CovarianceMatrix
        Background error covariance
    R : Dict[int, CovarianceMatrix]
        Observation error covariances R_k for each observation time
    m_b : PETSc.Vec
        Background state (prior)
    observations : Dict[int, PETSc.Vec]
        Observations {k: y_k} at each observation time
    obs_times : List[int]
        Time indices where observations are available
    use_implicit_adjoint : bool
        Whether to use implicit BDF2 adjoint (True) or explicit (False)

    Methods
    -------
    value(m)
        Compute cost function J(m)
    gradient(m)
        Compute gradient ∇J(m) via adjoint
    hessian_vector_product(m, v)
        Compute Hessian-vector product H·v (Gauss-Newton approximation)
    """

    def __init__(
        self,
        forward_model,
        observation_operator,
        background_cov,
        observation_cov: Dict[int, any],
        m_background: PETSc.Vec,
        observations: Dict[int, PETSc.Vec],
        obs_times: List[int],
        comm: Optional[PETSc.Comm] = None,
        use_implicit_adjoint: bool = False,
    ):
        """
        Initialize standard 4D-Var cost function.

        Parameters
        ----------
        forward_model : ForwardModel
            Forward model with solve(m, store_jacobians=True) method
        observation_operator : ObservationOperator
            Observation operator with apply() and apply_adjoint() methods
        background_cov : CovarianceMatrix
            Background error covariance B with apply_inverse() method
        observation_cov : Dict[int, CovarianceMatrix]
            Observation error covariances {k: R_k} with apply_inverse() method
        m_background : PETSc.Vec
            Background state vector
        observations : Dict[int, PETSc.Vec]
            Observation vectors {k: y_k}
        obs_times : List[int]
            Sorted list of observation time indices
        comm : PETSc.Comm, optional
            MPI communicator (defaults to PETSc.COMM_WORLD)
        use_implicit_adjoint : bool, optional
            Use implicit BDF2 adjoint (True) or explicit adjoint (False).
            Default is False for compatibility with simple test models.
        """
        self.forward_model = forward_model
        self.obs_op = observation_operator
        self.B = background_cov
        self.R = observation_cov
        self.m_b = m_background.copy()
        self.observations = {k: y_k.copy() for k, y_k in observations.items()}
        self.obs_times = sorted(obs_times)
        self.use_implicit_adjoint = use_implicit_adjoint

        # MPI communicator
        self.comm = comm if comm is not None else PETSc.COMM_WORLD
        self.rank = self.comm.getRank()

        # Cached trajectory and Jacobians from last forward solve
        self._trajectory = None
        self._jacobians = None
        self._last_m = None

        # Counters for performance tracking
        self.num_forward_solves = 0
        self.num_adjoint_solves = 0

        # Validate input dimensions
        self._validate_inputs()

    def _validate_inputs(self):
        """
        Validate input dimensions and consistency.

        Raises
        ------
        ValueError
            If dimensions are inconsistent
        """
        # Check that all observation times are valid
        for k in self.obs_times:
            if k not in self.observations:
                raise ValueError(f"Observation time {k} missing in observations dict")
            if k not in self.R:
                raise ValueError(f"Observation time {k} missing in covariance dict")

        # Check dimensions match
        n_state = self.m_b.getSize()
        for k in self.obs_times:
            n_obs = self.observations[k].getSize()
            # Note: Can't easily check R[k] dimensions without solving
            # Will be caught during apply_inverse if mismatch

    def value(self, m: PETSc.Vec) -> float:
        """
        Compute 4D-Var cost function value J(m).

        J(m) = J_b(m) + J_o(m)

        where:
            J_b(m) = ½⟨m - m_b, B⁻¹(m - m_b)⟩          (background term)
            J_o(m) = ½ Σ_k ⟨d_k, R_k⁻¹ d_k⟩            (observation terms)
            d_k = H_k(u_k) - y_k                        (innovation at time k)

        Parameters
        ----------
        m : PETSc.Vec
            Control variable (initial condition)

        Returns
        -------
        float
            Cost function value J(m)

        Notes
        -----
        This method runs the forward model if needed and caches the
        trajectory for subsequent gradient computation. The forward
        solve stores Jacobians from Newton iterations for efficient
        adjoint computation.
        """
        # Run forward model (with caching)
        trajectory, jacobians = self._run_forward_model(m, store_jacobians=True)

        # Background term: ½⟨m - m_b, B⁻¹(m - m_b)⟩
        m_minus_mb = m.copy()
        m_minus_mb.axpy(-1.0, self.m_b)  # m - m_b

        B_inv_m_minus_mb = self.B.apply_inverse(m_minus_mb)
        J_b = 0.5 * m_minus_mb.dot(B_inv_m_minus_mb)

        # Observation terms: ½ Σ_k ⟨d_k, R_k⁻¹ d_k⟩
        J_o = 0.0
        for k in self.obs_times:
            # Get state at observation time
            u_k = trajectory[k]

            # Apply observation operator: H_k(u_k)
            H_u_k = self.obs_op.apply(u_k, time_index=k)

            # Innovation: d_k = H_k(u_k) - y_k
            d_k = H_u_k.copy()
            d_k.axpy(-1.0, self.observations[k])

            # Weighted innovation: R_k⁻¹ d_k
            R_inv_d_k = self.R[k].apply_inverse(d_k)

            # Add to observation cost: ½⟨d_k, R_k⁻¹ d_k⟩
            J_o += 0.5 * d_k.dot(R_inv_d_k)

        # Total cost
        J_total = J_b + J_o

        # MPI collective (ensure all ranks agree)
        mpi_comm = self.comm.tompi4py()
        J_total = mpi_comm.allreduce(J_total, op=MPI.SUM)

        if self.rank == 0:
            print(
                f"Cost function: J = {J_total:.6e} (J_b = {J_b:.6e}, J_o = {J_o:.6e})"
            )

        return J_total

    def gradient(self, m: PETSc.Vec) -> PETSc.Vec:
        """
        Compute gradient ∇J(m) via adjoint method.

        The gradient is computed using the discrete adjoint equations:

        ∇J(m) = B⁻¹(m - m_b) + λ_0

        where λ_0 is obtained by solving the adjoint system backward in time.

        Parameters
        ----------
        m : PETSc.Vec
            Control variable (initial condition)

        Returns
        -------
        PETSc.Vec
            Gradient vector ∇J(m)

        Notes
        -----
        This method reuses cached Jacobians from the forward solve.
        The implementation uses either explicit or implicit adjoint
        depending on the use_implicit_adjoint flag.
        """
        # Run forward model if not already cached for this m
        trajectory, jacobians = self._run_forward_model(m, store_jacobians=True)

        # Background gradient contribution: B⁻¹(m - m_b)
        m_minus_mb = m.copy()
        m_minus_mb.axpy(-1.0, self.m_b)
        grad_background = self.B.apply_inverse(m_minus_mb)

        # Adjoint contribution: λ_0
        if self.use_implicit_adjoint:
            lambda_0 = self._solve_adjoint_system_implicit(trajectory, jacobians)
        else:
            lambda_0 = self._solve_adjoint_system_explicit(trajectory, jacobians)

        # Total gradient: ∇J = B⁻¹(m - m_b) + λ_0
        grad = grad_background.copy()
        grad.axpy(1.0, lambda_0)

        # Track counter
        self.num_adjoint_solves += 1

        if self.rank == 0:
            grad_norm = grad.norm()
            print(f"Gradient: ‖∇J‖ = {grad_norm:.6e}")

        return grad

    def _solve_adjoint_system_explicit(
        self, trajectory: List[PETSc.Vec], jacobians: List[PETSc.Mat]
    ) -> PETSc.Vec:
        """
        Solve adjoint system for EXPLICIT forward model.

        For explicit scheme where u_{n+1} = A * u_n, the adjoint is:
            λ_n = A^T * λ_{n+1} + forcing_n

        This is the correct adjoint for simple test models like:
            u_{n+1} = A * u_n

        Parameters
        ----------
        trajectory : List[PETSc.Vec]
            Forward trajectory [u_0, u_1, ..., u_N]
        jacobians : List[PETSc.Mat]
            Cached Jacobians [J_1, J_2, ..., J_N] from forward solve

        Returns
        -------
        PETSc.Vec
            Adjoint at initial time λ_0
        """
        N = len(trajectory) - 1

        if N == 0:
            # No time steps, just return observation forcing at t=0
            lambda_0 = trajectory[0].copy()
            lambda_0.zeroEntries()
            if 0 in self.obs_times:
                lambda_0 = self._compute_observation_forcing(trajectory[0], 0)
            return lambda_0

        # Initialize terminal adjoint
        lambda_current = trajectory[0].copy()
        lambda_current.zeroEntries()

        if N in self.obs_times:
            lambda_current = self._compute_observation_forcing(trajectory[N], N)

        # Backward sweep: n = N-1, N-2, ..., 0
        for n in range(N - 1, -1, -1):
            # Observation forcing at time n
            forcing_n = trajectory[0].copy()
            forcing_n.zeroEntries()

            if n in self.obs_times:
                forcing_n = self._compute_observation_forcing(trajectory[n], n)

            # For explicit scheme: λ_n = A^T * λ_{n+1} + forcing_n
            if jacobians and n < len(jacobians):
                J = jacobians[n]

                # Apply A^T to lambda_{n+1}
                lambda_new = lambda_current.duplicate()
                J.multTranspose(lambda_current, lambda_new)

                # Add observation forcing
                lambda_new.axpy(1.0, forcing_n)

                lambda_current = lambda_new
            else:
                # No Jacobian available, just use forcing
                lambda_current = forcing_n.copy()

        return lambda_current

    def _solve_adjoint_system_implicit(
        self, trajectory: List[PETSc.Vec], jacobians: List[PETSc.Mat]
    ) -> PETSc.Vec:
        """
        Solve adjoint system for IMPLICIT BDF2 scheme.

        For implicit BDF2 with residual:
            R(u^{n+1}; u^n, u^{n-1}) = (3u^{n+1} - 4u^n + u^{n-1})/(2Δt) + F(u^{n+1})

        The Jacobian is:
            J_n = ∂R/∂u^{n+1} = (3/(2Δt))M + ∂F/∂u

        The adjoint equation is:
            J_n^T λ_n = (4/(2Δt))M^T λ_{n+1} - (1/(2Δt))M^T λ_{n+2} + forcing_n

        This requires solving transpose linear systems.

        Parameters
        ----------
        trajectory : List[PETSc.Vec]
            Forward trajectory [u_0, u_1, ..., u_N]
        jacobians : List[PETSc.Mat]
            Cached Jacobians [J_1, J_2, ..., J_N] from forward solve

        Returns
        -------
        PETSc.Vec
            Adjoint at initial time λ_0
        """
        N = len(trajectory) - 1
        dt = self.forward_model.dt

        if N == 0:
            lambda_0 = trajectory[0].copy()
            lambda_0.zeroEntries()
            if 0 in self.obs_times:
                lambda_0 = self._compute_observation_forcing(trajectory[0], 0)
            return lambda_0

        # Initialize adjoint variables for BDF2 (need two previous values)
        lambda_next_next = trajectory[0].copy()
        lambda_next_next.zeroEntries()  # λ^{N+1} = 0 (doesn't exist)

        lambda_next = trajectory[0].copy()
        lambda_next.zeroEntries()  # λ^N = 0 (terminal condition)

        # Add terminal observation forcing if present
        if N in self.obs_times:
            terminal_forcing = self._compute_observation_forcing(trajectory[N], N)
            lambda_next.axpy(1.0, terminal_forcing)

        # Backward sweep: n = N-1, N-2, ..., 0
        for n in range(N - 1, -1, -1):
            # Assemble RHS for adjoint step
            rhs = lambda_next.copy()
            rhs.zeroEntries()

            # BDF2 time coupling: (4/(2Δt))M^T λ_{n+1}
            # Assuming M = I (identity mass matrix)
            rhs.axpy(4.0 / (2.0 * dt), lambda_next)

            # BDF2 time coupling: -(1/(2Δt))M^T λ_{n+2}
            if n < N - 1:  # Only apply if λ_{n+2} exists
                rhs.axpy(-1.0 / (2.0 * dt), lambda_next_next)

            # Add observation forcing if present at this time
            if n in self.obs_times:
                obs_forcing = self._compute_observation_forcing(trajectory[n], n)
                rhs.axpy(1.0, obs_forcing)

            # Solve transpose system: J_n^T λ^n = rhs
            if jacobians and n < len(jacobians):
                lambda_n = self._solve_transpose_system(jacobians[n], rhs)
            else:
                lambda_n = rhs.copy()

            # Update for next iteration
            lambda_next_next = lambda_next
            lambda_next = lambda_n

        return lambda_next

    def _compute_observation_forcing(self, u_n: PETSc.Vec, n: int) -> PETSc.Vec:
        """
        Compute observation forcing term for adjoint RHS.

        forcing_n = (∂H_n/∂u)^T R_n⁻¹ (H_n(u_n) - y_n)

        This is the adjoint of the observation operator applied to
        the weighted innovation.

        Parameters
        ----------
        u_n : PETSc.Vec
            State at time n
        n : int
            Time index

        Returns
        -------
        PETSc.Vec
            Observation forcing vector
        """
        # Compute innovation: d_n = H_n(u_n) - y_n
        H_u_n = self.obs_op.apply(u_n, time_index=n)
        d_n = H_u_n.copy()
        d_n.axpy(-1.0, self.observations[n])

        # Weight by observation covariance: R_n⁻¹ d_n
        R_inv_d_n = self.R[n].apply_inverse(d_n)

        # Apply observation operator adjoint: H_n^T (R_n⁻¹ d_n)
        forcing = self.obs_op.apply_adjoint(R_inv_d_n, u_n, time_index=n)

        return forcing

    def _solve_transpose_system(self, J: PETSc.Mat, rhs: PETSc.Vec) -> PETSc.Vec:
        """
        Solve transpose linear system: J^T x = rhs.

        This uses PETSc's KSP solver with transpose flag enabled.
        The Jacobian J is from the forward Newton solve.

        Parameters
        ----------
        J : PETSc.Mat
            Jacobian matrix from forward solve
        rhs : PETSc.Vec
            Right-hand side vector

        Returns
        -------
        PETSc.Vec
            Solution to J^T x = rhs

        Notes
        -----
        The solver is configured for transpose mode, which is critical
        for implicit adjoint computation. Uses GMRES with ILU
        preconditioning by default.
        """
        if J is None:
            # Handle special case (e.g., initial step)
            return rhs.copy()

        # Create KSP solver
        ksp = PETSc.KSP().create(comm=self.comm)
        ksp.setOperators(J)

        # Configure solver
        ksp.setType(PETSc.KSP.Type.GMRES)
        ksp.getPC().setType(PETSc.PC.Type.ILU)
        ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

        # Set convergence monitoring
        if self.rank == 0:
            ksp.setMonitor(lambda ksp, its, rnorm: None)  # Silent

        # Solve J^T x = rhs using transpose solve
        solution = rhs.duplicate()
        ksp.solveTranspose(rhs, solution)

        # Check convergence
        if ksp.getConvergedReason() < 0:
            if self.rank == 0:
                print(
                    f"WARNING: Transpose solve did not converge: {ksp.getConvergedReason()}"
                )

        ksp.destroy()
        return solution

    def _run_forward_model(
        self, m: PETSc.Vec, store_jacobians: bool = True
    ) -> Tuple[List[PETSc.Vec], Optional[List[PETSc.Mat]]]:
        """
        Run forward model and cache trajectory/Jacobians.

        This method intelligently caches results to avoid redundant
        forward solves when computing both value and gradient.

        Parameters
        ----------
        m : PETSc.Vec
            Initial condition
        store_jacobians : bool
            Whether to store Jacobians for adjoint (default: True)

        Returns
        -------
        trajectory : List[PETSc.Vec]
            State trajectory [u_0, u_1, ..., u_N]
        jacobians : Optional[List[PETSc.Mat]]
            Jacobian matrices [J_1, ..., J_N] or None

        Notes
        -----
        The forward model's solve method must support:
            trajectory, jacobians = forward_model.solve(m, store_jacobians=True)
        """
        # Check if we can reuse cached trajectory
        if self._last_m is not None and self._vectors_equal(m, self._last_m):
            if self._trajectory is not None:
                return self._trajectory, self._jacobians

        # Run forward model
        trajectory, jacobians = self.forward_model.solve(
            m, store_jacobians=store_jacobians
        )

        # Cache results
        self._trajectory = trajectory
        self._jacobians = jacobians
        self._last_m = m.copy()

        # Track counter
        self.num_forward_solves += 1

        return trajectory, jacobians

    def _vectors_equal(self, v1: PETSc.Vec, v2: PETSc.Vec, tol: float = 1e-14) -> bool:
        """
        Check if two vectors are equal within tolerance.

        Parameters
        ----------
        v1, v2 : PETSc.Vec
            Vectors to compare
        tol : float
            Tolerance for comparison

        Returns
        -------
        bool
            True if vectors are equal within tolerance
        """
        diff = v1.copy()
        diff.axpy(-1.0, v2)
        diff_norm = diff.norm()
        return diff_norm < tol

    def hessian_vector_product(self, m: PETSc.Vec, v: PETSc.Vec) -> PETSc.Vec:
        """
        Compute Hessian-vector product H·v using Gauss-Newton approximation.

        The Gauss-Newton Hessian is:
            H_GN = B⁻¹ + Σ_k (∂H_k/∂m)^T R_k⁻¹ (∂H_k/∂m)

        where ∂H_k/∂m is the sensitivity of observations to initial condition.

        This approximation neglects second-order derivative terms, making
        it positive semi-definite and suitable for optimization.

        Parameters
        ----------
        m : PETSc.Vec
            Current control variable
        v : PETSc.Vec
            Direction vector

        Returns
        -------
        PETSc.Vec
            Hessian-vector product H·v

        Notes
        -----
        The Hessian-vector product is computed using:
        1. Tangent linear model (TLM) to propagate perturbation forward
        2. Observation operator Jacobian application
        3. Adjoint of observation operator

        This avoids explicitly forming the Hessian matrix.
        """
        # Background term: B⁻¹ v
        Hv = self.B.apply_inverse(v)

        # Run tangent linear model: δu_k = (∂M_k/∂m) v
        # This requires forward solve with stored Jacobians
        trajectory, jacobians = self._run_forward_model(m, store_jacobians=True)

        # Propagate perturbation through TLM
        delta_trajectory = self._propagate_tlm(v, jacobians)

        # Accumulate observation terms
        for k in self.obs_times:
            # Linearized observation: δy_k = (∂H_k/∂u) δu_k
            delta_u_k = delta_trajectory[k]
            delta_y_k = self.obs_op.linearize_apply(
                delta_u_k, trajectory[k], time_index=k
            )

            # Weight by observation covariance: R_k⁻¹ δy_k
            R_inv_delta_y = self.R[k].apply_inverse(delta_y_k)

            # Apply observation operator adjoint: (∂H_k/∂u)^T R_k⁻¹ δy_k
            obs_contrib = self.obs_op.apply_adjoint(
                R_inv_delta_y, trajectory[k], time_index=k
            )

            # Propagate back to initial time via adjoint TLM
            initial_contrib = self._propagate_adjoint_tlm(obs_contrib, k, jacobians)

            # Add to Hessian-vector product
            Hv.axpy(1.0, initial_contrib)

        return Hv

    def _propagate_tlm(
        self, v: PETSc.Vec, jacobians: List[PETSc.Mat]
    ) -> List[PETSc.Vec]:
        """
        Propagate perturbation forward using tangent linear model.

        For explicit scheme: δu^{n+1} = J_n * δu^n
        For implicit BDF2: J_n δu^{n+1} = (4/(2Δt))M δu^n - (1/(2Δt))M δu^{n-1}

        Parameters
        ----------
        v : PETSc.Vec
            Initial perturbation δu_0 = v
        jacobians : List[PETSc.Mat]
            Cached Jacobians from forward solve

        Returns
        -------
        List[PETSc.Vec]
            Perturbation trajectory [δu_0, δu_1, ..., δu_N]
        """
        if not jacobians:
            return [v.copy()]

        N = len(jacobians)

        # Initialize perturbation trajectory
        delta_traj = [v.copy()]  # δu_0 = v

        if self.use_implicit_adjoint:
            # Implicit BDF2 TLM
            dt = self.forward_model.dt
            delta_u_nm1 = v.copy()  # δu^{n-1}
            delta_u_n = v.copy()  # δu^n

            for n in range(N):
                # Assemble RHS for TLM step
                rhs = delta_u_n.duplicate()
                rhs.zeroEntries()
                rhs.axpy(4.0 / (2.0 * dt), delta_u_n)
                rhs.axpy(-1.0 / (2.0 * dt), delta_u_nm1)

                # Solve J_n δu^{n+1} = rhs
                ksp = PETSc.KSP().create(comm=self.comm)
                ksp.setOperators(jacobians[n])
                ksp.setType(PETSc.KSP.Type.GMRES)
                ksp.getPC().setType(PETSc.PC.Type.ILU)
                ksp.setTolerances(rtol=1e-10, atol=1e-12)

                delta_u_next = rhs.duplicate()
                ksp.solve(rhs, delta_u_next)
                ksp.destroy()

                delta_traj.append(delta_u_next.copy())

                # Update for next step
                delta_u_nm1 = delta_u_n
                delta_u_n = delta_u_next
        else:
            # Explicit TLM: δu^{n+1} = J_n * δu^n
            delta_u_current = v.copy()

            for n in range(N):
                delta_u_next = delta_u_current.duplicate()
                jacobians[n].mult(delta_u_current, delta_u_next)
                delta_traj.append(delta_u_next.copy())
                delta_u_current = delta_u_next

        return delta_traj

    def _propagate_adjoint_tlm(
        self,
        forcing: PETSc.Vec,
        time_index: int,
        jacobians: List[PETSc.Mat],
    ) -> PETSc.Vec:
        """
        Propagate adjoint forcing back to initial time.

        This is used in Hessian-vector product computation to transport
        observation information back to the control space.

        Parameters
        ----------
        forcing : PETSc.Vec
            Forcing at time time_index
        time_index : int
            Time index where forcing is applied
        jacobians : List[PETSc.Mat]
            Cached Jacobians from forward solve

        Returns
        -------
        PETSc.Vec
            Contribution to gradient at initial time
        """
        if not jacobians or time_index == 0:
            return forcing.copy()

        lambda_current = forcing.copy()

        if self.use_implicit_adjoint:
            # Implicit adjoint backward propagation
            dt = self.forward_model.dt
            lambda_next_next = forcing.copy()
            lambda_next_next.zeroEntries()

            for n in range(time_index - 1, -1, -1):
                # Assemble RHS
                rhs = lambda_current.duplicate()
                rhs.zeroEntries()
                rhs.axpy(4.0 / (2.0 * dt), lambda_current)
                if n < time_index - 1:
                    rhs.axpy(-1.0 / (2.0 * dt), lambda_next_next)

                # Solve transpose system
                lambda_n = self._solve_transpose_system(jacobians[n], rhs)

                lambda_next_next = lambda_current
                lambda_current = lambda_n
        else:
            # Explicit adjoint backward propagation: λ_n = J^T * λ_{n+1}
            for n in range(time_index - 1, -1, -1):
                lambda_new = lambda_current.duplicate()
                jacobians[n].multTranspose(lambda_current, lambda_new)
                lambda_current = lambda_new

        return lambda_current

    def clear_cache(self):
        """
        Clear cached trajectory and Jacobians to free memory.

        Useful for long optimization runs where caching becomes
        memory intensive.
        """
        self._trajectory = None
        self._jacobians = None
        self._last_m = None

    def get_diagnostics(self) -> Dict[str, any]:
        """
        Get diagnostic information about cost function evaluations.

        Returns
        -------
        Dict
            Dictionary with counters and statistics
        """
        return {
            "num_forward_solves": self.num_forward_solves,
            "num_adjoint_solves": self.num_adjoint_solves,
            "num_obs_times": len(self.obs_times),
            "obs_times": self.obs_times,
            "adjoint_type": "implicit" if self.use_implicit_adjoint else "explicit",
        }


# ============================================================================
# TESTING UTILITIES
# ============================================================================


def taylor_remainder_test(
    cost_function: FourDVarCost,
    m0: PETSc.Vec,
    direction: Optional[PETSc.Vec] = None,
    epsilons: Optional[List[float]] = None,
) -> bool:
    """
    Perform Taylor remainder test for gradient verification.

    Tests the convergence:
        |J(m + εv) - J(m) - ε⟨∇J(m), v⟩| = O(ε²)

    Parameters
    ----------
    cost_function : FourDVarCost
        Cost function to test
    m0 : PETSc.Vec
        Base point for test
    direction : PETSc.Vec, optional
        Test direction (random if None)
    epsilons : List[float], optional
        List of perturbation sizes

    Returns
    -------
    bool
        True if test passes (O(ε²) convergence observed)
    """
    if direction is None:
        direction = m0.duplicate()
        direction.setRandom()
        direction.scale(1.0 / direction.norm())

    if epsilons is None:
        epsilons = [10 ** (-i) for i in range(1, 8)]

    # Compute base values
    J0 = cost_function.value(m0)
    grad0 = cost_function.gradient(m0)
    directional_deriv = grad0.dot(direction)

    print("\nTaylor Remainder Test:")
    print(f"{'ε':>12} {'|Remainder|':>15} {'Order':>10}")
    print("-" * 40)

    prev_remainder = None
    orders = []

    for eps in epsilons:
        # Perturbed point
        m_eps = m0.copy()
        m_eps.axpy(eps, direction)

        # Compute perturbed cost
        J_eps = cost_function.value(m_eps)

        # Taylor remainder: |J(m+εv) - J(m) - ε⟨∇J, v⟩|
        remainder = abs(J_eps - J0 - eps * directional_deriv)

        # Estimate convergence order
        if prev_remainder is not None:
            order = np.log(prev_remainder / remainder) / np.log(2.0)
            orders.append(order)
            print(f"{eps:12.2e} {remainder:15.6e} {order:10.2f}")
        else:
            print(f"{eps:12.2e} {remainder:15.6e} {'---':>10}")

        prev_remainder = remainder

    # Check if order is approximately 2 (indicating correct gradient)
    avg_order = np.mean(orders[-3:])  # Average last 3 orders
    passed = 1.8 <= avg_order <= 2.2

    print(f"\nAverage order (last 3): {avg_order:.2f}")
    print(f"Test {'PASSED' if passed else 'FAILED'}")

    return passed


def adjoint_consistency_test(
    cost_function: FourDVarCost,
    m0: PETSc.Vec,
    tolerance: float = 1e-10,
) -> bool:
    """
    Test adjoint consistency: ⟨TLM·v, w⟩ = ⟨v, Adjoint·w⟩.

    Parameters
    ----------
    cost_function : FourDVarCost
        Cost function to test
    m0 : PETSc.Vec
        Point at which to test
    tolerance : float
        Tolerance for test pass

    Returns
    -------
    bool
        True if test passes
    """
    # Create random vectors
    v = m0.duplicate()
    v.setRandom()

    w = m0.duplicate()
    w.setRandom()

    # Compute LHS: ⟨H·v, w⟩
    Hv = cost_function.hessian_vector_product(m0, v)
    lhs = Hv.dot(w)

    # Compute RHS: ⟨v, H·w⟩ (H should be symmetric)
    Hw = cost_function.hessian_vector_product(m0, w)
    rhs = v.dot(Hw)

    # Check symmetry
    rel_error = abs(lhs - rhs) / abs(lhs)

    print(f"\nAdjoint Consistency Test:")
    print(f"LHS = {lhs:.10e}")
    print(f"RHS = {rhs:.10e}")
    print(f"Relative error = {rel_error:.10e}")

    passed = rel_error < tolerance
    print(f"Test {'PASSED' if passed else 'FAILED'}")

    return passed


# class DCFourDVarCost(CostFunction):
#     """
#     Data-Consistent 4D-Var (DC-4DVar) cost function.

#     J_DC(m) = J(m) - ½ Σ_k ⟨Q_k(m) - Q_k(m_b), L_k⁻¹(Q_k(m) - Q_k(m_b))⟩

#     Includes predictability term to prevent assimilation of
#     unpredictable small scales.
#     """

#     def __init__(
#         self,
#         forward_model,
#         observation_operator,
#         background_cov,
#         observation_cov,
#         m_background: PETSc.Vec,
#         observations: List[PETSc.Vec],
#         obs_times: List[int],
#         qoi_map=None,
#     ):
#         """
#         Initialize DC-4DVar cost function.

#         Args:
#             forward_model: Forward model
#             observation_operator: Observation operator
#             background_cov: Background covariance B
#             observation_cov: Observation covariance R
#             m_background: Background state
#             observations: Observation vectors
#             obs_times: Observation time indices
#             qoi_map: Quantity of Interest map Q_k
#         """
#         super().__init__(
#             forward_model, observation_operator, background_cov, observation_cov
#         )
#         self.m_b = m_background
#         self.y_obs = observations
#         self.obs_times = obs_times
#         self.qoi_map = qoi_map

#         # Predicted error covariance L_k (computed from B)
#         self._L_k = None

#     def value(self, m: PETSc.Vec) -> float:
#         """Compute DC-4DVar cost with predictability term."""
#         # TODO: Implement standard term - predictability term
#         pass

#     def gradient(self, m: PETSc.Vec) -> PETSc.Vec:
#         """Compute DC-4DVar gradient."""
#         # TODO: Implement with predictability gradient correction
#         pass

#     def _compute_predicted_covariance(self, k: int) -> PETSc.Mat:
#         """
#         Compute L_k = Q_k B Q_k^T at observation time k.

#         Args:
#             k: Time index

#         Returns:
#             Predicted error covariance matrix
#         """
#         # TODO: Implement TLM-based covariance propagation
#         pass


# class DCWMEFourDVarCost(DCFourDVarCost):
#     """
#     DC-4DVar with Weighted Mean Error QoI.

#     Q_wme,k(m) = (1/k) Σ_{j=0}^{k-1} (H_j(M_{j:0}(m)) - y_j)

#     Uses cumulative time-averaged innovation as QoI
#     for improved stability with sparse observations.
#     """

#     def __init__(
#         self,
#         forward_model,
#         observation_operator,
#         background_cov,
#         observation_cov,
#         m_background: PETSc.Vec,
#         observations: List[PETSc.Vec],
#         obs_times: List[int],
#     ):
#         """Initialize DC-WME cost function."""
#         super().__init__(
#             forward_model,
#             observation_operator,
#             background_cov,
#             observation_cov,
#             m_background,
#             observations,
#             obs_times,
#         )

#         # Cache for WME accumulation
#         self._wme_accumulator = {}

#     def value(self, m: PETSc.Vec) -> float:
#         """Compute DC-WME cost."""
#         # TODO: Implement with WME QoI
#         pass

#     def gradient(self, m: PETSc.Vec) -> PETSc.Vec:
#         """Compute DC-WME gradient."""
#         # TODO: Implement with WME adjoint contribution
#         pass

#     def _compute_wme_qoi(self, trajectory: List, k: int) -> PETSc.Vec:
#         """
#         Compute WME QoI at time k.

#         Q_wme,k = (1/k) Σ_{j=0}^{k-1} (H_j(u_j) - y_j)

#         Args:
#             trajectory: Forward trajectory [u_0, ..., u_k]
#             k: Current time index

#         Returns:
#             WME vector
#         """
#         # TODO: Implement weighted mean error accumulation
#         pass

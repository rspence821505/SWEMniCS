"""
Implicit adjoint solver for BDF2 time-stepping scheme.

This module implements the discrete adjoint of the implicit BDF2
forward solver, which is critical for efficient gradient computation
in 4D-Var data assimilation.

Key Insight
-----------
For implicit BDF2 schemes, the Jacobian matrices J_n = ∂R/∂u^{n+1}
are already computed during forward Newton solves. We store and
reuse these Jacobians (transposed) for adjoint computation, providing
approximately 50% cost savings compared to recomputation.

Mathematical Background
-----------------------
The forward BDF2 discretization solves:
    R(u^{n+1}; u^n, u^{n-1}) = (3u^{n+1} - 4u^n + u^{n-1})/(2Δt) + F(u^{n+1}) = 0

The discrete adjoint equations become:
    J_n^T λ^n = (4/(2Δt))·M·λ^{n+1} - (1/(2Δt))·λ^{n+2} + forcing^n

where:
    - J_n = ∂R/∂u^{n+1} is the cached Jacobian from Newton solve
    - M is the mass matrix
    - forcing^n includes observation contributions

The adjoint sweep proceeds backward in time (n = N-1, ..., 0) solving
transpose linear systems at each step.

Author: Rylan Spence
Date: 2025
"""

from typing import List, Optional
from petsc4py import PETSc
import numpy as np


class ImplicitAdjointSolver:
    """
    Adjoint solver for implicit BDF2 time discretization.

    This solver implements the discrete adjoint of SWEMniCS's implicit
    BDF2 forward scheme. It reuses Jacobian matrices computed during
    forward Newton iterations, providing efficient gradient computation.

    The adjoint equations for BDF2 are:
        J_n^T λ^n = (4/(2Δt))·M·λ^{n+1} - (1/(2Δt))·M·λ^{n+2} + forcing^n

    Attributes
    ----------
    forward_model : ForwardModel
        Forward model (for mass matrix access).
    trajectory : List[PETSc.Vec]
        Forward trajectory [u_0, u_1, ..., u_N].
    jacobians : List[PETSc.Mat]
        Cached Jacobian matrices from forward solve.
    dt : float
        Time step size.
    comm : MPI.Comm
        MPI communicator.

    Notes
    -----
    This implementation is MPI-aware and maintains distributed
    computation throughout the adjoint solve. All operations use
    PETSc distributed data structures.
    """

    def __init__(
        self,
        forward_model,
        trajectory: List[PETSc.Vec],
        jacobians: List[PETSc.Mat],
        dt: float,
    ):
        """
        Initialize implicit adjoint solver.

        Parameters
        ----------
        forward_model : ForwardModel
            Forward model providing mass matrix and other utilities.
        trajectory : List[PETSc.Vec]
            Forward trajectory from forward solve.
        jacobians : List[PETSc.Mat]
            Cached Jacobian matrices from forward Newton solves.
            jacobians[n] corresponds to J_n = ∂R/∂u^{n+1}.
        dt : float
            Time step size used in forward solve.

        Raises
        ------
        ValueError
            If trajectory and jacobians have inconsistent lengths.
        """
        self.forward_model = forward_model
        self.trajectory = trajectory
        self.jacobians = jacobians
        self.dt = dt

        # Validate inputs
        self.num_steps = len(trajectory) - 1
        if jacobians is not None and len(jacobians) != self.num_steps:
            raise ValueError(
                f"Jacobians length ({len(jacobians)}) must match "
                f"num_steps ({self.num_steps})"
            )

        # Get communicator from trajectory
        self.comm = trajectory[0].getComm()

        # Cache for mass matrix (computed on first use)
        self._mass_matrix = None

    def solve(
        self,
        terminal_forcing: PETSc.Vec,
        observation_forcings: Optional[List[Optional[PETSc.Vec]]] = None,
    ) -> PETSc.Vec:
        """
        Solve adjoint equations backward in time.

        This method performs the adjoint sweep from time N to time 0,
        solving transpose linear systems at each step using cached
        Jacobians from the forward solve.

        Parameters
        ----------
        terminal_forcing : PETSc.Vec
            Forcing at final time λ_N (usually zero for standard 4D-Var).
        observation_forcings : Optional[List[Optional[PETSc.Vec]]]
            List of observation forcings at each time step.
            observation_forcings[n] = H_n^T R_n^{-1} (H_n(u_n) - y_n).
            None entries indicate no observations at that time.

        Returns
        -------
        PETSc.Vec
            Adjoint at initial time λ_0, which provides the gradient
            contribution: ∇J(m) = B^{-1}(m - m_b) + λ_0.

        Notes
        -----
        The adjoint sweep uses a three-level stencil due to BDF2:
            - λ^n (current, to be computed)
            - λ^{n+1} (one step into future)
            - λ^{n+2} (two steps into future, when available)

        For the last two steps (n = N-1, N-2), we handle the edge cases
        where λ^{n+2} may not exist.
        """
        if observation_forcings is None:
            observation_forcings = [None] * (self.num_steps + 1)

        # Validate observation forcings length
        if len(observation_forcings) != self.num_steps + 1:
            raise ValueError(
                f"observation_forcings length ({len(observation_forcings)}) "
                f"must be num_steps + 1 ({self.num_steps + 1})"
            )

        # Initialize adjoint at final time
        lambda_next_next = None  # λ^{n+2}
        lambda_next = terminal_forcing.copy()  # λ^{n+1}

        # Add observation forcing at final time if present
        if observation_forcings[self.num_steps] is not None:
            lambda_next.axpy(1.0, observation_forcings[self.num_steps])

        # Backward sweep: n = N-1, N-2, ..., 0
        for n in range(self.num_steps - 1, -1, -1):
            # Assemble forcing from observations and time coupling
            forcing = self._assemble_adjoint_forcing(
                n, lambda_next, lambda_next_next, observation_forcings[n]
            )

            # Solve transpose system: J_n^T · λ^n = forcing
            lambda_n = self._solve_transpose_system(n, forcing)

            # Shift adjoints for next iteration
            lambda_next_next = lambda_next
            lambda_next = lambda_n

        return lambda_next

    def _solve_transpose_system(self, n: int, forcing: PETSc.Vec) -> PETSc.Vec:
        """
        Solve J_n^T · λ = rhs using the DISTRIBUTED Jacobian.

        This is the core computational step of the adjoint solver.
        We reuse the Jacobian J_n = ∂R/∂u^{n+1} from the forward
        Newton solve, simply solving the transposed system.

        Parameters
        ----------
        n : int
            Time step index (0 ≤ n < num_steps).
        forcing : PETSc.Vec
            Right-hand side vector for the transpose system.

        Returns
        -------
        PETSc.Vec
            Solution λ^n to the transpose system.

        Notes
        -----
        We use GMRES with no preconditioning as a baseline. In production,
        you may want to add preconditioning (e.g., Block-Jacobi, ILU).

        The transpose solve is accomplished via PETSc's built-in
        transpose mode, which handles all the complexity of transposing
        the distributed matrix structure.
        """
        J = self.jacobians[n]

        # Create KSP solver for this step
        ksp = PETSc.KSP().create(J.getComm())
        ksp.setOperators(J)

        # Configure solver
        ksp.setType(PETSc.KSP.Type.GMRES)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.NONE)  # No preconditioning (baseline)

        # Set tight tolerances for adjoint accuracy
        ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

        # Allow command-line overrides
        ksp.setFromOptions()

        # Solve TRANSPOSE system: J^T · λ = forcing
        lambda_n = forcing.duplicate()
        ksp.solveTranspose(forcing, lambda_n)

        # Check convergence
        if not ksp.converged:
            reason = ksp.getConvergedReason()
            its = ksp.getIterationNumber()
            rnorm = ksp.getResidualNorm()
            raise RuntimeError(
                f"Adjoint solve failed at step {n}: "
                f"reason={reason}, iterations={its}, residual={rnorm}"
            )

        ksp.destroy()

        return lambda_n

    def _assemble_adjoint_forcing(
        self,
        n: int,
        lambda_next: PETSc.Vec,
        lambda_next_next: Optional[PETSc.Vec],
        obs_forcing: Optional[PETSc.Vec],
    ) -> PETSc.Vec:
        """
        Assemble RHS for adjoint step including time coupling and observations.

        For BDF2, the adjoint time coupling is:
            RHS = (4/(2Δt))·M·λ^{n+1} - (1/(2Δt))·M·λ^{n+2} + obs_forcing

        This three-level coupling arises from the BDF2 stencil in the
        forward discretization.

        Parameters
        ----------
        n : int
            Time step index.
        lambda_next : PETSc.Vec
            Adjoint at time n+1.
        lambda_next_next : Optional[PETSc.Vec]
            Adjoint at time n+2 (None for last step).
        obs_forcing : Optional[PETSc.Vec]
            Forcing from observation operator at time n.

        Returns
        -------
        PETSc.Vec
            Assembled forcing vector for transpose solve.

        Notes
        -----
        The mass matrix M represents the L2 projection operator.
        For standard finite element spaces, M is symmetric positive definite.
        """
        # Get mass matrix (cached after first call)
        M = self._get_mass_matrix()

        # Initialize forcing vector
        forcing = lambda_next.duplicate()

        # BDF2 time coupling: (4/(2Δt))·M·λ^{n+1}
        M.mult(lambda_next, forcing)
        forcing.scale(4.0 / (2.0 * self.dt))

        # Subtract (1/(2Δt))·M·λ^{n+2} if it exists
        if lambda_next_next is not None:
            temp = lambda_next.duplicate()
            M.mult(lambda_next_next, temp)
            forcing.axpy(-1.0 / (2.0 * self.dt), temp)
            temp.destroy()

        # Add observation forcing if present
        if obs_forcing is not None:
            forcing.axpy(1.0, obs_forcing)

        return forcing

    def _get_mass_matrix(self) -> PETSc.Mat:
        """
        Get or assemble mass matrix M.

        The mass matrix is required for BDF2 time coupling in the
        adjoint equations. For SWEMniCS, this is typically available
        from the forward model.

        Returns
        -------
        PETSc.Mat
            Mass matrix (identity if not provided by forward model).

        Notes
        -----
        We try several approaches in order of preference:
        1. Use forward_model.get_mass_matrix() if available
        2. Use forward_model.mass_matrix attribute if available
        3. Fall back to identity matrix (for testing/debugging)

        The identity fallback is only for simple test cases and should
        not be used in production.
        """
        if self._mass_matrix is None:
            # Try to get mass matrix from forward model
            if hasattr(self.forward_model, "get_mass_matrix"):
                self._mass_matrix = self.forward_model.get_mass_matrix()
            elif hasattr(self.forward_model, "mass_matrix"):
                self._mass_matrix = self.forward_model.mass_matrix
            else:
                # Fallback: create identity matrix (for testing only)
                # In production, forward model should provide mass matrix
                n_dofs_local, n_dofs_global = self.trajectory[0].getSizes()

                M = PETSc.Mat().createAIJ(
                    size=([n_dofs_local, n_dofs_global], [n_dofs_local, n_dofs_global]),
                    comm=self.comm,
                )
                M.setUp()

                # Set diagonal to 1 (identity) - only on owned rows
                start, end = M.getOwnershipRange()
                for i in range(start, end):
                    M.setValue(i, i, 1.0, addv=PETSc.InsertMode.INSERT)

                M.assemblyBegin()
                M.assemblyEnd()

                self._mass_matrix = M

                # Warn about identity fallback
                if self.comm.getRank() == 0:
                    import warnings

                    warnings.warn(
                        "Using identity mass matrix (fallback). "
                        "Forward model should provide proper mass matrix.",
                        UserWarning,
                    )

        return self._mass_matrix

    def get_diagnostics(self) -> dict:
        """
        Get diagnostic information about the adjoint solve.

        Returns
        -------
        dict
            Dictionary containing:
            - num_steps: Number of time steps
            - dt: Time step size
            - has_jacobians: Whether Jacobians are available
            - trajectory_size: Size of state vectors
        """
        diagnostics = {
            "num_steps": self.num_steps,
            "dt": self.dt,
            "has_jacobians": self.jacobians is not None,
        }

        if len(self.trajectory) > 0:
            diagnostics["trajectory_size"] = self.trajectory[0].getSize()

        return diagnostics


class ImplicitAdjointStepAnalyzer:
    """
    Analyzer for verifying implicit adjoint step correctness.

    This class provides utilities for validating that the adjoint
    time-stepping correctly implements the discrete adjoint of the
    BDF2 forward scheme.

    The key test is the adjoint consistency check:
        ⟨J·δu, λ⟩ = ⟨δu, J^T·λ⟩

    Methods
    -------
    verify_adjoint_step : Verify single adjoint step via adjoint test
    verify_time_coupling : Verify BDF2 time coupling correctness
    """

    def __init__(self, forward_model, dt: float):
        """
        Initialize analyzer.

        Parameters
        ----------
        forward_model : ForwardModel
            Forward model instance.
        dt : float
            Time step size.
        """
        self.forward_model = forward_model
        self.dt = dt

    def verify_adjoint_step(
        self,
        n: int,
        trajectory: List[PETSc.Vec],
        jacobians: List[PETSc.Mat],
        rtol: float = 1e-10,
    ) -> float:
        """
        Verify single adjoint step via adjoint consistency test.

        Tests that the transpose Jacobian is correctly applied:
            ⟨J·δu, λ⟩ = ⟨δu, J^T·λ⟩

        Parameters
        ----------
        n : int
            Time step index to verify.
        trajectory : List[PETSc.Vec]
            Forward trajectory.
        jacobians : List[PETSc.Mat]
            Jacobian matrices from forward solve.
        rtol : float, optional
            Relative tolerance for test (default: 1e-10).

        Returns
        -------
        float
            Relative error in adjoint test.

        Raises
        ------
        AssertionError
            If adjoint test fails (relative error > rtol).

        Notes
        -----
        This test is critical for validating the adjoint implementation.
        It should be run during development and as part of CI testing.
        """
        J = jacobians[n]
        u_ref = trajectory[n]

        # Create random perturbation δu
        delta_u = u_ref.duplicate()
        delta_u.setRandom()

        # Create random dual vector λ
        lambda_vec = u_ref.duplicate()
        lambda_vec.setRandom()

        # Compute LHS: ⟨J·δu, λ⟩
        J_delta_u = u_ref.duplicate()
        J.mult(delta_u, J_delta_u)
        lhs = J_delta_u.dot(lambda_vec)

        # Compute RHS: ⟨δu, J^T·λ⟩
        JT_lambda = u_ref.duplicate()
        J.multTranspose(lambda_vec, JT_lambda)
        rhs = delta_u.dot(JT_lambda)

        # Compute relative error
        rel_error = abs(lhs - rhs) / (abs(lhs) + 1e-16)

        # Clean up
        delta_u.destroy()
        lambda_vec.destroy()
        J_delta_u.destroy()
        JT_lambda.destroy()

        # Check error
        if rel_error > rtol:
            raise AssertionError(
                f"Adjoint consistency test failed at step {n}: "
                f"LHS={lhs}, RHS={rhs}, rel_error={rel_error}"
            )

        return rel_error

    def verify_time_coupling(
        self,
        lambda_n: PETSc.Vec,
        lambda_next: PETSc.Vec,
        lambda_next_next: Optional[PETSc.Vec],
        mass_matrix: PETSc.Mat,
    ) -> PETSc.Vec:
        """
        Verify BDF2 time coupling in adjoint.

        Computes and returns the time coupling term:
            (4/(2Δt))·M·λ^{n+1} - (1/(2Δt))·M·λ^{n+2}

        This can be used to verify that the time coupling is correctly
        assembled in the full adjoint solver.

        Parameters
        ----------
        lambda_n : PETSc.Vec
            Adjoint at time n (for reference/validation).
        lambda_next : PETSc.Vec
            Adjoint at time n+1.
        lambda_next_next : Optional[PETSc.Vec]
            Adjoint at time n+2.
        mass_matrix : PETSc.Mat
            Mass matrix M.

        Returns
        -------
        PETSc.Vec
            Time coupling contribution.

        Notes
        -----
        This utility is primarily for testing and debugging the
        adjoint implementation.
        """
        coupling = lambda_next.duplicate()

        # Contribution from λ^{n+1}
        mass_matrix.mult(lambda_next, coupling)
        coupling.scale(4.0 / (2.0 * self.dt))

        # Contribution from λ^{n+2} (if exists)
        if lambda_next_next is not None:
            temp = lambda_next.duplicate()
            mass_matrix.mult(lambda_next_next, temp)
            coupling.axpy(-1.0 / (2.0 * self.dt), temp)
            temp.destroy()

        return coupling


class CheckpointedImplicitAdjoint:
    """
    Implicit adjoint with checkpointing for memory efficiency.

    For long time horizons (N > 500), storing all Jacobians becomes
    memory-prohibitive. This class implements checkpointing strategies
    that trade computation for memory.

    Strategies
    ----------
    - Full storage: Store all states + Jacobians (fastest, N < 500)
    - State-only: Store states, recompute Jacobians (moderate, N < 2000)
    - Binomial: O(log N) checkpoints (slowest, any N)

    Notes
    -----
    This class is a stub for Sprint 2, Week 4 (Day 16).
    Full implementation will be completed in the checkpointing module.

    See Also
    --------
    swemnics.adjoint.checkpointing : Full checkpointing strategies
    """

    def __init__(self, forward_model, checkpointer, dt: float):
        """
        Initialize checkpointed adjoint.

        Parameters
        ----------
        forward_model : ForwardModel
            Forward model for recomputation.
        checkpointer : CheckpointerBase
            Checkpointing strategy instance.
        dt : float
            Time step size.
        """
        self.forward_model = forward_model
        self.checkpointer = checkpointer
        self.dt = dt

    def solve(
        self,
        terminal_forcing: PETSc.Vec,
        observation_forcings: Optional[List] = None,
    ) -> PETSc.Vec:
        """
        Solve adjoint with checkpointing.

        Uses checkpointing strategy to minimize memory while
        maintaining computational efficiency.

        Parameters
        ----------
        terminal_forcing : PETSc.Vec
            Terminal condition for adjoint.
        observation_forcings : Optional[List]
            Observation forcings at each time.

        Returns
        -------
        PETSc.Vec
            Adjoint at initial time.

        Notes
        -----
        To be implemented in Sprint 2, Week 4 (Day 16) as part of
        the checkpointing module refactoring.

        The implementation will:
        1. Identify checkpoint intervals
        2. For each interval:
           a. Restore state from checkpoint
           b. Recompute forward trajectory segment
           c. Solve adjoint backward over interval
        3. Accumulate adjoint contributions
        """
        raise NotImplementedError(
            "Checkpointed adjoint to be implemented in Sprint 2, Week 4 (Day 16). "
            "See REFACTORING_PLAN_DETAILED.md for full specification."
        )

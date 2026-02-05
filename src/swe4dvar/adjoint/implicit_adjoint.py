"""
Implicit adjoint solver for BDF2 time-stepping scheme.

Handles the specific time-coupling and transpose systems
required for adjoint of implicit BDF2 discretization.

Mathematical Background
-----------------------
For implicit BDF2 discretization:
    R(u^{n+1}; u^n, u^{n-1}) = (3u^{n+1} - 4u^n + u^{n-1})/(2Δt) + F(u^{n+1}) = 0

The Jacobian from Newton's method is:
    J_n = ∂R/∂u^{n+1} = (3/(2Δt))·M + ∂F/∂u|_{u^{n+1}}

The adjoint equation becomes:
    J_n^T·λ^n = (4/(2Δt))·M·λ^{n+1} - (1/(2Δt))·M·λ^{n+2} + forcing

Key insight: Jacobian J is already computed and stored from
forward Newton solve, so we just need to transpose it.

"""

from typing import List, Optional, Tuple
from petsc4py import PETSc
import numpy as np

# Import BDF2TimeCoefficients for proper time-coupling
from swe4dvar.forward.variational_forms import BDF2TimeCoefficients


class ImplicitAdjointSolver:
    """
    Adjoint solver for implicit BDF2 time discretization.

    This class implements the discrete adjoint of the implicit BDF2
    forward time-stepping scheme. It reuses cached Jacobians from the
    forward Newton solve, providing significant computational savings
    (~50% cost reduction compared to recomputing Jacobians).

    For BDF2 forward step:
        R(u^{n+1}; u^n, u^{n-1}) = (3u^{n+1} - 4u^n + u^{n-1})/(2Δt) + F(u^{n+1}) = 0

    The adjoint equation is:
        J_n^T·λ^n = (4/(2Δt))·M·λ^{n+1} - (1/(2Δt))·M·λ^{n+2} + forcing

    where:
        - J_n = ∂R/∂u^{n+1} is the Jacobian from forward Newton solve
        - M is the mass matrix
        - λ^n is the adjoint variable at time n
        - forcing includes observation terms and other adjoint sources

    Attributes
    ----------
    forward_model : ForwardModel
        Forward model instance (for mass matrix access).
    trajectory : List[PETSc.Vec]
        Forward trajectory [u_0, u_1, ..., u_N].
    jacobians : List[PETSc.Mat]
        Cached Jacobian matrices from forward solve [J_1, J_2, ..., J_N].
    dt : float
        Time step size.
    num_steps : int
        Number of time steps.

    Notes
    -----
    This implementation follows a discretize-then-optimize (DTO) approach
    for the time-discrete residual: it reuses the Jacobians assembled during
    the forward Newton solves and applies their transpose during the backward
    sweep.

    See Also
    --------
    CheckpointedImplicitAdjoint : Memory-efficient variant with checkpointing
    ImplicitAdjointStepAnalyzer : Validation and verification tools

    Boundary Condition Handling
    ---------------------------
    The discrete adjoint requires special treatment of Dirichlet boundary conditions:

    1. **Unmodified Jacobians**: The forward Newton solver now returns the physics
       Jacobian WITHOUT BC modifications (identity rows) for adjoint use. This
       preserves correct sensitivity propagation through the transpose solve.

    2. **Adjoint homogeneous BCs**: After each transpose solve J^T λ = rhs, the
       adjoint variable λ is set to zero at BC DOFs. This is the discrete adjoint
       analog of homogeneous Dirichlet BCs on the continuous adjoint.

    3. **Gradient at BC DOFs**: The final gradient ∂J/∂m has zeros at BC DOFs,
       indicating these are fixed and should not be modified by the optimizer.

    For proper gradient accuracy, ensure bc_dof_indices is set (either explicitly
    or via auto-detection from the forward model's problem definition).

    References
    ----------
    .. [1] Spence et al. (2025), "Variational Data-Consistent Inversion
           for Chaotic Dynamical Systems", arXiv:2501.08207
    """

    def __init__(
        self,
        forward_model,
        trajectory: List[PETSc.Vec],
        jacobians: List[PETSc.Mat],
        dt: float,
        variational_form=None,  # NEW: Optional variational form
        use_bdf2: Optional[bool] = None,  # NEW: Explicit time scheme control
        bdf2_start_step: int = 3,  # NEW: Step where BDF2 starts (1-indexed)
        flux_formulation: bool = True,  # NEW: Account for Q=[h, h*ux, h*uy] flux
        h_dof_indices: Optional[set] = None,  # NEW: Indices of h DOFs
        ux_dof_indices: Optional[set] = None,  # NEW: Indices of ux DOFs (for full (∂Q/∂u)^T)
        uy_dof_indices: Optional[set] = None,  # NEW: Indices of uy DOFs (for full (∂Q/∂u)^T)
        bc_dof_indices: Optional[set] = None,  # NEW: Indices of Dirichlet BC DOFs
    ):
        """
        Initialize implicit adjoint solver.

        Parameters
        ----------
        forward_model : ForwardModel
            Forward model instance (for mass matrix access).
        trajectory : List[PETSc.Vec]
            Forward trajectory [u_0, u_1, ..., u_N].
        jacobians : List[PETSc.Mat]
            Cached Jacobian matrices from forward solve.
        dt : float
            Time step size.
        variational_form : VariationalForm, optional
            Variational form providing mass matrix and BDF2 coefficients.
            If None, will use fallback assembly.
        use_bdf2 : bool, optional
            Whether to use BDF2 time stepping coefficients (True) or
            Backward Euler (False). If None, will attempt to detect from
            forward_model or default to True (BDF2).
        bdf2_start_step : int, optional
            The timestep (1-indexed) at which the forward solver switches
            from backward Euler to BDF2. Default is 3, matching the standard
            SWE solver behavior where steps 1-2 use backward Euler and
            steps 3+ use BDF2. Set to 1 to use BDF2 from the start, or
            set to num_steps+1 to use backward Euler throughout.
        flux_formulation : bool, optional
            Whether to account for flux formulation Q=[h, h*ux, h*uy] in
            the time coupling. When True (default), momentum DOFs are scaled
            by the water depth h from the trajectory. This is required for
            correct gradients when the forward solver uses velocity variables
            (solution_var='h' or 'eta') rather than flux variables.
        h_dof_indices : set, optional
            Set of DOF indices corresponding to water depth h. Required when
            flux_formulation=True. Can be obtained from the function space:
            _, h_to_parent = V.sub(0).collapse()
            h_dof_indices = set(h_to_parent)
        ux_dof_indices : set, optional
            Set of DOF indices corresponding to x-velocity ux. Optional but
            recommended for full flux transformation accuracy. Obtained via:
            _, ux_to_parent = V.sub(1).sub(0).collapse()
            ux_dof_indices = set(ux_to_parent)
        uy_dof_indices : set, optional
            Set of DOF indices corresponding to y-velocity uy. Optional but
            recommended for full flux transformation accuracy. Obtained via:
            _, uy_to_parent = V.sub(1).sub(1).collapse()
            uy_dof_indices = set(uy_to_parent)
        bc_dof_indices : set, optional
            Set of DOF indices with Dirichlet boundary conditions. The adjoint
            time-coupling term is zeroed for these DOFs since the forward Jacobian
            has identity rows there and the dynamics don't propagate through BCs.
            Can be obtained from the problem's boundary condition setup.

        Raises
        ------
        ValueError
            If trajectory and Jacobians have incompatible lengths.

        Notes
        -----
        The forward SWE solver typically uses backward Euler for the first
        2 timesteps to start up, then switches to BDF2 for accuracy. The
        adjoint must use matching time-coupling coefficients to ensure
        gradient accuracy. This is controlled by bdf2_start_step.

        When flux_formulation=True, the adjoint accounts for the fact that
        the SWE residual uses flux variables Q=[h, h*ux, h*uy] for the time
        derivative, while the state variables are [h, ux, uy]. The Jacobian
        of Q w.r.t. u includes h scaling for momentum components:
            ∂Q/∂u = [[1, 0, 0], [ux, h, 0], [uy, 0, h]]
        This scaling must be applied in the adjoint time coupling.
        """
        self.forward_model = forward_model
        self.trajectory = trajectory
        self.jacobians = jacobians
        self.dt = dt

        self.num_steps = len(trajectory) - 1

        # Validate inputs
        if len(jacobians) != self.num_steps:
            raise ValueError(
                f"Expected {self.num_steps} Jacobians for {len(trajectory)} states, "
                f"got {len(jacobians)}"
            )

        # Determine time stepping scheme
        # Priority: explicit parameter > forward_model attribute > default (BDF2)
        if use_bdf2 is None:
            # Try to detect from forward model
            if hasattr(forward_model, 'use_bdf2'):
                use_bdf2 = forward_model.use_bdf2
            elif hasattr(forward_model, 'problem') and hasattr(forward_model.problem, 'use_bdf2'):
                use_bdf2 = forward_model.problem.use_bdf2
            else:
                # Default to BDF2 (most common for accuracy)
                use_bdf2 = True

        self.use_bdf2 = use_bdf2
        self.bdf2_start_step = bdf2_start_step

        # Flux formulation handling
        self.flux_formulation = flux_formulation
        self.h_dof_indices = h_dof_indices
        self.ux_dof_indices = ux_dof_indices
        self.uy_dof_indices = uy_dof_indices
        self.bc_dof_indices = bc_dof_indices

        # Try to auto-detect BC DOF indices from forward model
        if bc_dof_indices is None:
            if hasattr(forward_model, 'problem'):
                problem = forward_model.problem
                bc_dofs = set()
                # Collect DOFs from problem's boundary info
                if hasattr(problem, 'dof_open') and problem.dof_open.size > 0:
                    bc_dofs.update(problem.dof_open.tolist())
                if hasattr(problem, 'ux_dofs_closed') and len(problem.ux_dofs_closed) > 0:
                    bc_dofs.update(problem.ux_dofs_closed.tolist())
                if hasattr(problem, 'uy_dofs_closed') and len(problem.uy_dofs_closed) > 0:
                    bc_dofs.update(problem.uy_dofs_closed.tolist())
                if bc_dofs:
                    self.bc_dof_indices = bc_dofs

        if flux_formulation:
            # Try to auto-detect DOF indices from forward model
            if hasattr(forward_model, 'solver') and hasattr(forward_model.solver, 'V'):
                V = forward_model.solver.V
                if h_dof_indices is None:
                    _, h_to_parent = V.sub(0).collapse()
                    self.h_dof_indices = set(h_to_parent)
                # Try to get velocity component DOF indices
                if ux_dof_indices is None or uy_dof_indices is None:
                    try:
                        # V.sub(1) is velocity space, V.sub(1).sub(0) is ux, V.sub(1).sub(1) is uy
                        _, ux_to_parent = V.sub(1).sub(0).collapse()
                        self.ux_dof_indices = set(ux_to_parent)
                        _, uy_to_parent = V.sub(1).sub(1).collapse()
                        self.uy_dof_indices = set(uy_to_parent)
                    except (AttributeError, IndexError, RuntimeError):
                        # May fail if velocity space structure differs
                        pass
            elif h_dof_indices is None:
                import warnings
                warnings.warn(
                    "flux_formulation=True but h_dof_indices not provided and "
                    "could not auto-detect. Gradient may be incorrect for momentum DOFs. "
                    "Set h_dof_indices explicitly or set flux_formulation=False.",
                    RuntimeWarning
                )

        # Create time coefficient objects for both schemes
        self._time_coeffs_bdf2 = BDF2TimeCoefficients(dt, use_bdf2=True)
        self._time_coeffs_euler = BDF2TimeCoefficients(dt, use_bdf2=False)

        # Legacy: keep time_coeffs for backward compatibility
        self.time_coeffs = BDF2TimeCoefficients(dt, use_bdf2=use_bdf2)

        # NEW: Store variational form for mass matrix access
        self.var_form = variational_form

        # Mass matrix for BDF2 time coupling (cached)
        self._mass_matrix = None

    def _uses_bdf2_at_step(self, n: int) -> bool:
        """
        Determine if timestep n uses BDF2 or backward Euler.

        Parameters
        ----------
        n : int
            Timestep index (1-indexed, where n=1 is the first timestep).

        Returns
        -------
        bool
            True if step n uses BDF2, False if it uses backward Euler.
        """
        if not self.use_bdf2:
            # Global backward Euler mode
            return False
        # BDF2 mode: check if we're past the transition
        return n >= self.bdf2_start_step

    def _get_adjoint_coeffs_for_step(self, n: int) -> Tuple[float, float]:
        """
        Get adjoint time-coupling coefficients for step n.

        These are the coefficients for computing the RHS forcing:
        RHS = c_{n+1} * M * λ_{n+1} + c_{n+2} * M * λ_{n+2}

        The coefficients depend on which time scheme was used for
        steps n+1 and n+2 in the forward solve.

        Parameters
        ----------
        n : int
            Current adjoint timestep (solving for λ_n).

        Returns
        -------
        tuple of float
            (c_{n+1}, c_{n+2}) coefficients.
        """
        dt = self.dt if isinstance(self.dt, (float, int)) else self.dt[n]

        # For the adjoint at step n, we need ∂R^{n+1}/∂u^n and ∂R^{n+2}/∂u^n
        # These depend on the time scheme used for R^{n+1} and R^{n+2}

        # Coefficient from R^{n+1} (sensitivity to u^n)
        if self._uses_bdf2_at_step(n + 1):
            # BDF2: R^{n+1} = (3u^{n+1} - 4u^n + u^{n-1})/(2dt) + F
            # ∂R^{n+1}/∂u^n = -4/(2dt) * M = -2/dt * M
            # In adjoint: contributes +2/dt * M * λ_{n+1} (sign flip)
            c_next = 4.0 / (2.0 * dt)
        else:
            # Backward Euler: R^{n+1} = (u^{n+1} - u^n)/dt + F
            # ∂R^{n+1}/∂u^n = -1/dt * M
            # In adjoint: contributes +1/dt * M * λ_{n+1}
            c_next = 1.0 / dt

        # Coefficient from R^{n+2} (sensitivity to u^n, only for BDF2)
        if self._uses_bdf2_at_step(n + 2):
            # BDF2: R^{n+2} = (3u^{n+2} - 4u^{n+1} + u^n)/(2dt) + F
            # ∂R^{n+2}/∂u^n = +1/(2dt) * M
            # In adjoint: contributes -1/(2dt) * M * λ_{n+2}
            c_next_next = -1.0 / (2.0 * dt)
        else:
            # Backward Euler: R^{n+2} doesn't depend on u^n
            c_next_next = 0.0

        return (c_next, c_next_next)

    def _apply_flux_scaling(self, vec: PETSc.Vec, state: PETSc.Vec) -> PETSc.Vec:
        """
        Apply full (∂Q/∂u)^T transformation for flux formulation Q=[h, h*ux, h*uy].

        For the SWE with state variables [h, ux, uy] and flux Q=[h, h*ux, h*uy],
        the time derivative Jacobian has the structure:
            ∂Q/∂u = [[1, 0, 0], [ux, h, 0], [uy, 0, h]]

        The transpose is:
            (∂Q/∂u)^T = [[1, ux, uy], [0, h, 0], [0, 0, h]]

        Applying (∂Q/∂u)^T to vector λ = [λ_h, λ_ux, λ_uy]:
            result_h = λ_h + ux*λ_ux + uy*λ_uy  (off-diagonal contributions!)
            result_ux = h*λ_ux
            result_uy = h*λ_uy

        Parameters
        ----------
        vec : PETSc.Vec
            Vector to transform (typically M·λ).
        state : PETSc.Vec
            State vector to extract h, ux, uy values from.

        Returns
        -------
        PETSc.Vec
            Transformed vector with full (∂Q/∂u)^T applied.

        Notes
        -----
        This method implements the full (∂Q/∂u)^T transformation including
        off-diagonal terms. For proper node-by-node application, requires
        h_dof_indices, ux_dof_indices, and uy_dof_indices to be set.

        If only h_dof_indices is available, falls back to diagonal-only scaling
        (momentum DOFs multiplied by mean h), which is approximate but captures
        the dominant effect.
        """
        if not self.flux_formulation or self.h_dof_indices is None:
            return vec

        state_arr = state.getArray()
        vec_arr = vec.getArray().copy()
        n_dofs = len(vec_arr)

        # Build node-to-DOF mapping if we have full DOF structure
        if self.ux_dof_indices is not None and self.uy_dof_indices is not None:
            # Full (∂Q/∂u)^T transformation with proper DOF mapping
            result_arr = self._apply_full_flux_transform(vec_arr, state_arr)
        else:
            # Fallback: diagonal-only scaling
            result_arr = self._apply_diagonal_flux_scaling(vec_arr, state_arr, n_dofs)

        result = vec.duplicate()
        result.setArray(result_arr)
        return result

    def _apply_full_flux_transform(
        self, vec_arr: np.ndarray, state_arr: np.ndarray
    ) -> np.ndarray:
        """
        Apply full (∂Q/∂u)^T transformation with proper DOF structure.

        Uses ux_dof_indices and uy_dof_indices to correctly map DOFs to nodes
        and apply the full transformation including off-diagonal terms.
        """
        result_arr = vec_arr.copy()

        # Create sorted lists for efficient lookup
        h_list = sorted(self.h_dof_indices)
        ux_list = sorted(self.ux_dof_indices)
        uy_list = sorted(self.uy_dof_indices)

        # Verify same number of DOFs per field (should be equal for CG spaces)
        n_nodes = len(h_list)
        if len(ux_list) != n_nodes or len(uy_list) != n_nodes:
            import warnings
            warnings.warn(
                f"DOF count mismatch: h={n_nodes}, ux={len(ux_list)}, uy={len(uy_list)}. "
                "Falling back to diagonal scaling.",
                RuntimeWarning
            )
            return self._apply_diagonal_flux_scaling(vec_arr, state_arr, len(vec_arr))

        # Apply (∂Q/∂u)^T node by node
        # For each node k with DOFs (h_k, ux_k, uy_k):
        #   result_h_k  = vec_h_k + ux_state_k * vec_ux_k + uy_state_k * vec_uy_k
        #   result_ux_k = h_state_k * vec_ux_k
        #   result_uy_k = h_state_k * vec_uy_k

        for k in range(n_nodes):
            h_idx = h_list[k]
            ux_idx = ux_list[k]
            uy_idx = uy_list[k]

            # Get state values at this node
            h_val = state_arr[h_idx]
            ux_val = state_arr[ux_idx]
            uy_val = state_arr[uy_idx]

            # Get input vector values
            vec_h = vec_arr[h_idx]
            vec_ux = vec_arr[ux_idx]
            vec_uy = vec_arr[uy_idx]

            # Apply (∂Q/∂u)^T transformation
            # h DOF: gets off-diagonal contributions from ux and uy
            result_arr[h_idx] = vec_h + ux_val * vec_ux + uy_val * vec_uy

            # ux DOF: scaled by h
            result_arr[ux_idx] = h_val * vec_ux

            # uy DOF: scaled by h
            result_arr[uy_idx] = h_val * vec_uy

        return result_arr

    def _apply_diagonal_flux_scaling(
        self, vec_arr: np.ndarray, state_arr: np.ndarray, n_dofs: int
    ) -> np.ndarray:
        """
        Apply diagonal-only flux scaling (fallback when ux/uy DOF indices unavailable).

        This approximation scales momentum DOFs by h but ignores off-diagonal terms.
        """
        result_arr = vec_arr.copy()
        h_dof_list = sorted(self.h_dof_indices)

        def find_nearest_h_value(mom_idx):
            """Find h value from the nearest h DOF."""
            best_h_idx = None
            for h_idx in h_dof_list:
                if h_idx < mom_idx:
                    best_h_idx = h_idx
                elif h_idx > mom_idx:
                    break

            if best_h_idx is not None and best_h_idx < len(state_arr):
                return state_arr[best_h_idx]
            else:
                h_values = [state_arr[i] for i in h_dof_list if i < len(state_arr)]
                return np.mean(h_values) if h_values else 1.0

        # Scale momentum DOFs by their associated h value
        for i in range(n_dofs):
            if i not in self.h_dof_indices:
                h_val = find_nearest_h_value(i)
                result_arr[i] *= h_val

        return result_arr

    def solve(
        self,
        terminal_forcing: PETSc.Vec,
        observation_forcings: Optional[List[Optional[PETSc.Vec]]] = None,
    ) -> PETSc.Vec:
        """
        Solve adjoint equations backward in time.

        Performs backward sweep through time to compute the gradient of
        the cost function with respect to the initial condition.

        Parameters
        ----------
        terminal_forcing : PETSc.Vec
            Forcing at final time (usually zero for 4D-Var).
        observation_forcings : Optional[List[Optional[PETSc.Vec]]]
            List of forcings from observations at each time.
            Length must be num_steps + 1. None entries indicate no
            observation at that time.

        Returns
        -------
        PETSc.Vec
            Adjoint at initial time λ_0 (gradient w.r.t. initial condition).

        Notes
        -----
        The gradient of the 4D-Var cost function is:
            ∇J(m) = B^{-1}(m - m_b) + λ_0

        where λ_0 is returned by this method.

        Examples
        --------
        >>> solver = ImplicitAdjointSolver(model, traj, jacs, dt)
        >>> terminal = traj[-1].duplicate()
        >>> terminal.zeroEntries()
        >>> obs_forcings = [None] * (len(traj))
        >>> obs_forcings[10] = compute_observation_forcing(...)
        >>> lambda_0 = solver.solve(terminal, obs_forcings)
        """
        if observation_forcings is None:
            observation_forcings = [None] * (self.num_steps + 1)

        # Validate observation forcings length
        if len(observation_forcings) != self.num_steps + 1:
            raise ValueError(
                f"observation_forcings must have length {self.num_steps + 1}, "
                f"got {len(observation_forcings)}"
            )

        # Initialize adjoint at final time by solving J_N^T λ_N = -f_N
        # The adjoint equation at n=N is: J_N^T λ_N = terminal_forcing - f_N
        # NOTE: Previous code incorrectly set λ_N = f_N directly without solving!
        lambda_next_next = None  # λ^{n+2}

        # Build RHS for final time: terminal_forcing - f_N
        final_rhs = terminal_forcing.copy()
        if observation_forcings[-1] is not None:
            final_rhs.axpy(-1.0, observation_forcings[-1])  # RHS = terminal - f_N

        # Solve J_N^T λ_N = RHS
        # Note: For n=N (final time), we use jacobians[N-1] = jacobians[num_steps-1]
        lambda_next = self._solve_transpose_system(self.num_steps, final_rhs)
        final_rhs.destroy()

        # Backward sweep: n = N-1, N-2, ..., 1
        # Note: We stop at n=1, not n=0, because n=0 (initial condition)
        # requires special handling - gradient comes from time-coupling only
        for n in range(self.num_steps - 1, 0, -1):
            # Assemble forcing from observations and time coupling
            forcing = self._assemble_adjoint_forcing(
                n, lambda_next, lambda_next_next, observation_forcings[n]
            )

            # Solve transpose system: J_n^T·λ^n = forcing
            lambda_n = self._solve_transpose_system(n, forcing)

            # Shift for next iteration
            lambda_next_next = lambda_next
            lambda_next = lambda_n

            # Clean up intermediate vectors (except final result)
            if n > 1:
                forcing.destroy()

        # Special handling for n=0 (initial condition)
        # The gradient w.r.t. initial condition comes from time-coupling only
        gradient_u0 = self._compute_initial_gradient(
            lambda_next, lambda_next_next, observation_forcings[0]
        )

        return gradient_u0

    def _solve_transpose_system(self, n: int, forcing: PETSc.Vec) -> PETSc.Vec:
        """
        Solve J^T·λ = rhs using the DISTRIBUTED Jacobian.

        The Jacobian J = ∂R^n/∂u^n from forward Newton solve
        is reused by transposing it. This is the key computational
        savings of the implicit adjoint approach.

        Parameters
        ----------
        n : int
            Time index (must be >= 1, as n=0 is handled separately).
        forcing : PETSc.Vec
            Right-hand side vector.

        Returns
        -------
        PETSc.Vec
            Solution λ^n.

        Notes
        -----
        - Uses GMRES with no preconditioning by default
        - Tolerances set to 1e-10 (rtol) and 1e-12 (atol)
        - The transpose solve is handled automatically by PETSc
        - All operations maintain distributed parallelism
        - Jacobian indexing: jacobians[k] stores ∂R^(k+1)/∂u^(k+1),
          so for λ^n we need jacobians[n-1] to get ∂R^n/∂u^n
        """
        if n == 0:
            raise RuntimeError(
                "Cannot solve transpose system for n=0. "
                "Initial condition gradient should be computed via time-coupling."
            )

        # CRITICAL FIX: jacobians[k] stores Jacobian from timestep k+1
        # To solve for λ^n, we need J_n = ∂R^n/∂u^n
        # This is stored in jacobians[n-1]
        J = self.jacobians[n - 1]

        # Create KSP solver on the Jacobian's communicator
        ksp = PETSc.KSP().create(J.getComm())
        ksp.setOperators(J)
        ksp.setType(PETSc.KSP.Type.GMRES)
        ksp.getPC().setType(PETSc.PC.Type.NONE)
        ksp.setTolerances(rtol=1e-10, atol=1e-12)

        # Set transpose mode - this is critical!
        ksp.setOperators(J)

        # Solve transpose system: J^T·λ = forcing
        lambda_n = forcing.duplicate()
        ksp.solveTranspose(forcing, lambda_n)

        # Check convergence
        reason = ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(
                f"Adjoint transpose solve failed at step {n}: "
                f"reason={reason}, iterations={ksp.getIterationNumber()}"
            )

        # Clean up
        ksp.destroy()

        # NOTE: We do NOT zero the adjoint at boundary DOFs during backward propagation.
        # With unmodified Jacobians (no identity rows at BC DOFs), the physics coupling
        # propagates through all DOFs including boundaries. Zeroing here would cut off
        # valid sensitivity pathways and corrupt interior DOF gradients.
        #
        # The boundary gradient zeroing happens ONLY at the final step in
        # _compute_initial_gradient(), where we zero the gradient (not adjoint state)
        # at BC DOFs. This ensures the optimizer doesn't modify fixed BC values
        # while preserving accurate gradient computation for interior DOFs.

        return lambda_n

    def _compute_initial_gradient(
        self,
        lambda_1: PETSc.Vec,
        lambda_2: Optional[PETSc.Vec],
        obs_forcing: Optional[PETSc.Vec],
    ) -> PETSc.Vec:
        """
        Compute gradient w.r.t. initial condition u^0.

        For the initial condition, there is no residual R^0 to linearize.
        The gradient comes from time-coupling to R^1 and potentially R^2:

            ∂L/∂u^0 = λ_1^T ∂R^1/∂u^0 + λ_2^T ∂R^2/∂u^0 + obs_forcing

        The coefficients depend on which time scheme is used for each step:

        - R^1 always uses backward Euler (step 1 < bdf2_start_step):
          ∂R^1/∂u^0 = -1/Δt · M  →  contributes -(1/Δt)·M·λ^1

        - R^2 depends on bdf2_start_step:
          - If step 2 uses backward Euler (bdf2_start_step > 2):
            ∂R^2/∂u^0 = 0  (backward Euler doesn't depend on u^0)
          - If step 2 uses BDF2 (bdf2_start_step <= 2):
            ∂R^2/∂u^0 = +1/(2Δt) · M  →  contributes +(1/(2Δt))·M·λ^2

        NOTE: This is DIFFERENT from interior adjoint steps because the
        initial condition connects differently to the time-stepping stencil.

        Parameters
        ----------
        lambda_1 : PETSc.Vec
            Adjoint variable at time 1.
        lambda_2 : Optional[PETSc.Vec]
            Adjoint variable at time 2 (None for single step).
        obs_forcing : Optional[PETSc.Vec]
            Observation forcing at time 0 (if any).

        Returns
        -------
        PETSc.Vec
            Gradient ∂L/∂u^0.
        """
        # Get time step size
        dt = self.dt if isinstance(self.dt, (float, int)) else self.dt[0]
        M = self._get_mass_matrix()

        # Get initial state for flux scaling (∂R^1/∂u^0 is evaluated at u^0)
        state_0 = self.trajectory[0] if len(self.trajectory) > 0 else None

        # Coefficient for λ^1 from R^1
        # Step 1 is always backward Euler (1 < bdf2_start_step for any reasonable value)
        # Backward Euler: R^1 = (u^1 - u^0)/dt + F(u^1) = 0
        # ∂R^1/∂u^0 = -1/dt · (flux-scaled M)
        c_1 = -1.0 / dt

        # Compute c_1·(flux-scaled M)·λ^1 = -(1/Δt)·(flux-scaled M)·λ^1
        result = lambda_1.duplicate()
        M.mult(lambda_1, result)

        # Apply flux scaling if enabled
        if self.flux_formulation and state_0 is not None:
            result = self._apply_flux_scaling(result, state_0)

        result.scale(c_1)

        # Coefficient for λ^2 from R^2 (only contributes if R^2 uses BDF2)
        if self.use_bdf2 and lambda_2 is not None and self._uses_bdf2_at_step(2):
            # BDF2 at step 2: R^2 = (3u^2 - 4u^1 + u^0)/(2dt) + F(u^2) = 0
            # ∂R^2/∂u^0 = +1/(2dt) · (flux-scaled M)
            c_2 = 1.0 / (2.0 * dt)
            temp = lambda_1.duplicate()
            M.mult(lambda_2, temp)

            # Apply flux scaling
            if self.flux_formulation and state_0 is not None:
                temp = self._apply_flux_scaling(temp, state_0)

            result.axpy(c_2, temp)
            temp.destroy()
        # If backward Euler at step 2, there's no u^0 dependency so c_2 = 0

        # ADD observation forcing at t=0 (if present)
        # NOTE: For initial condition, the gradient is:
        #   λ_0 = (time coupling) + ∂J_obs/∂u_0
        # This is DIFFERENT from interior points where forcing is SUBTRACTED.
        # The observation term at t=0 directly contributes to ∂J/∂m since u_0 = m.
        if obs_forcing is not None:
            result.axpy(+1.0, obs_forcing)

        # Apply homogeneous Dirichlet BCs to the gradient
        # BC DOFs in the initial condition are fixed (not part of control space),
        # so their gradient should be zero to prevent optimizer from changing them.
        if self.bc_dof_indices is not None and len(self.bc_dof_indices) > 0:
            result_arr = result.getArray()
            for dof in self.bc_dof_indices:
                if dof < len(result_arr):
                    result_arr[dof] = 0.0
            result.setArray(result_arr)

        return result

    def _assemble_adjoint_forcing(
        self,
        n: int,
        lambda_next: PETSc.Vec,
        lambda_next_next: Optional[PETSc.Vec],
        obs_forcing: Optional[PETSc.Vec],
    ) -> PETSc.Vec:
        """
        Assemble RHS for adjoint step.

        For BDF2, the adjoint time coupling is:
            RHS = c_{n+1}·M·λ^{n+1} + c_{n+2}·M·λ^{n+2} - obs_forcing

        Note: The observation forcing is SUBTRACTED because the adjoint equation
        comes from setting ∂L/∂u_n = 0 where L = J + Σ λ^T R. This gives:
            J_n^T λ_n = -∂J_obs/∂u_n + (time coupling terms)

        The coefficients come from BDF2TimeCoefficients and support both
        BDF2 and Backward Euler schemes, as well as adaptive time-stepping.

        Parameters
        ----------
        n : int
            Time index.
        lambda_next : PETSc.Vec
            Adjoint at time n+1.
        lambda_next_next : Optional[PETSc.Vec]
            Adjoint at time n+2 (None for last step).
        obs_forcing : Optional[PETSc.Vec]
            Forcing from observation operator.

        Returns
        -------
        PETSc.Vec
            Assembled forcing vector.

        Notes
        -----
        The time coupling coefficients arise from differentiating the
        BDF2 stencil with respect to u^n. For BDF2: (4/(2Δt), -1/(2Δt)),
        for Backward Euler: (1/Δt, 0).
        """
        # Get time-coupling coefficients for this specific step
        # This accounts for the backward Euler -> BDF2 transition in the forward solver
        c_next, c_next_next = self._get_adjoint_coeffs_for_step(n)

        # Get mass matrix
        M = self._get_mass_matrix()

        # Get state at time n for flux scaling (∂R^{n+1}/∂u^n is evaluated at u^n)
        state_n = self.trajectory[n] if n < len(self.trajectory) else None

        # Time coupling: c_{n+1}·(flux-scaled M)·λ^{n+1}
        # For flux formulation Q=[h, h*ux, h*uy], the sensitivity ∂Q/∂u
        # has h scaling for momentum DOFs
        forcing = lambda_next.duplicate()
        M.mult(lambda_next, forcing)

        # Apply flux scaling if enabled
        if self.flux_formulation and state_n is not None:
            forcing = self._apply_flux_scaling(forcing, state_n)

        forcing.scale(c_next)

        # Add c_{n+2}·M·λ^{n+2}
        if lambda_next_next is not None and abs(c_next_next) > 1e-14:
            temp = lambda_next.duplicate()
            M.mult(lambda_next_next, temp)

            # Apply flux scaling for the n+2 term as well (using state at n)
            if self.flux_formulation and state_n is not None:
                temp = self._apply_flux_scaling(temp, state_n)

            forcing.axpy(c_next_next, temp)
            temp.destroy()

        # SUBTRACT observation forcing (from ∂L/∂u_n = 0 gives -∂J_obs/∂u_n)
        if obs_forcing is not None:
            forcing.axpy(-1.0, obs_forcing)

        return forcing

    def _get_mass_matrix(self) -> PETSc.Mat:
        """
        Get or assemble mass matrix M.

        For BDF2 time coupling in adjoint equations. The mass matrix
        is cached after first access for efficiency.

        Returns
        -------
        PETSc.Mat
            Mass matrix (identity if not provided by forward model or
            variational form).

        Notes
        -----
        The mass matrix is assumed to be constant throughout the
        simulation. For time-varying mass matrices (e.g., in wetting/drying),
        this would need to be modified.

        Priority order:
        1. Variational form (if provided)
        2. Forward model mass matrix
        3. Identity matrix (fallback)
        """
        if self._mass_matrix is None:
            # First priority: use variational form if available
            if self.var_form is not None:
                self._mass_matrix = self.var_form.assemble_mass_matrix()
            # Second priority: try to get mass matrix from forward model
            elif hasattr(self.forward_model, "get_mass_matrix"):
                self._mass_matrix = self.forward_model.get_mass_matrix()
            elif hasattr(self.forward_model, "mass_matrix"):
                self._mass_matrix = self.forward_model.mass_matrix
            else:
                # Fallback: create identity matrix
                # Use Jacobian's distribution to match DOF partitioning
                if len(self.jacobians) > 0:
                    # Get sizes from Jacobian which has correct distribution
                    J = self.jacobians[0]
                    global_size = J.getSize()[0]
                    local_size = J.getLocalSize()[0]
                    comm = J.getComm()

                    # Create matrix with matching distribution
                    M = PETSc.Mat().createAIJ(
                        size=[[local_size, global_size], [local_size, global_size]],
                        comm=comm
                    )
                else:
                    # No Jacobians available - use trajectory vector
                    n_dofs = self.trajectory[0].getSize()
                    local_size = self.trajectory[0].getLocalSize()
                    comm = self.trajectory[0].getComm()

                    M = PETSc.Mat().createAIJ(
                        size=[[local_size, n_dofs], [local_size, n_dofs]],
                        comm=comm
                    )

                M.setUp()

                # Set diagonal to 1 (identity)
                start, end = M.getOwnershipRange()
                for i in range(start, end):
                    M.setValue(i, i, 1.0)

                M.assemblyBegin()
                M.assemblyEnd()

                self._mass_matrix = M

        return self._mass_matrix

    def cleanup(self):
        """
        Clean up allocated resources.

        Call this method to explicitly release PETSc objects when done.
        """
        if self._mass_matrix is not None:
            self._mass_matrix.destroy()
            self._mass_matrix = None


class ImplicitAdjointStepAnalyzer:
    """
    Analyzer for implicit adjoint step correctness.

    Validates that adjoint time-stepping correctly implements
    the discrete adjoint of BDF2 forward scheme.

    This class provides utilities for validating that the adjoint
    time-stepping correctly implements the discrete adjoint of the
    BDF2 forward scheme.

    The key test is the adjoint consistency check:
        ⟨J·δu, λ⟩ = ⟨δu, J^T·λ⟩

    Methods
    -------
    verify_adjoint_step : Verify single adjoint step via adjoint test
    verify_time_coupling : Verify BDF2 time coupling correctness

    Examples
    --------
    >>> analyzer = ImplicitAdjointStepAnalyzer(forward_model, dt)
    >>> error = analyzer.verify_adjoint_step(n, trajectory, jacobians)
    >>> print(f"Adjoint consistency error: {error}")
    """

    def __init__(self, forward_model, dt: float, variational_form=None):
        """
        Initialize analyzer.

        Parameters
        ----------
        forward_model : ForwardModel
            Forward model instance.
        dt : float
            Time step size.
        variational_form : VariationalForm, optional
            Variational form for mass matrix and time coefficients.
        """
        self.forward_model = forward_model
        self.dt = dt
        self.var_form = variational_form

        # Use BDF2TimeCoefficients for correct time-coupling
        use_bdf2 = True
        self.time_coeffs = BDF2TimeCoefficients(dt, use_bdf2=use_bdf2)

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
        n: int,
        lambda_n: PETSc.Vec,
        lambda_next: PETSc.Vec,
        lambda_next_next: Optional[PETSc.Vec],
        mass_matrix: PETSc.Mat,
    ) -> PETSc.Vec:
        """
        Verify BDF2 time coupling in adjoint.

        Computes and returns the time coupling term using BDF2TimeCoefficients:
            c_{n+1}·M·λ^{n+1} + c_{n+2}·M·λ^{n+2}

        This can be used to verify that the time coupling is correctly
        assembled in the full adjoint solver.

        Parameters
        ----------
        n : int
            Time step index.
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
        adjoint implementation. Coefficients are obtained from
        BDF2TimeCoefficients to support adaptive time-stepping.
        """
        # Get time-coupling coefficients from BDF2TimeCoefficients
        c_next, c_next_next = self.time_coeffs.get_adjoint_coeffs(n)

        coupling = lambda_next.duplicate()

        # Contribution from λ^{n+1}
        mass_matrix.mult(lambda_next, coupling)
        coupling.scale(c_next)

        # Contribution from λ^{n+2} (if exists)
        if lambda_next_next is not None and abs(c_next_next) > 1e-14:
            temp = lambda_next.duplicate()
            mass_matrix.mult(lambda_next_next, temp)
            coupling.axpy(c_next_next, temp)
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

    Attributes
    ----------
    forward_model : ForwardModel
        Forward model for recomputation.
    checkpointer : CheckpointerBase
        Checkpointing strategy instance.
    dt : float
        Time step size.

    Notes
    -----
    This implementation integrates with the checkpointing module
    (swe4dvar.adjoint.checkpointing) to provide flexible memory
    management for large-scale problems.

    The adjoint solve is performed in segments, with forward trajectory
    segments recomputed as needed based on the checkpointing strategy.

    See Also
    --------
    swe4dvar.adjoint.checkpointing : Full checkpointing strategies
    ImplicitAdjointSolver : Standard solver for small problems

    Examples
    --------
    >>> from swe4dvar.adjoint.checkpointing import create_checkpointer
    >>> checkpointer = create_checkpointer(num_steps=2000, strategy="state_only")
    >>> solver = CheckpointedImplicitAdjoint(model, checkpointer, dt=0.1)
    >>> lambda_0 = solver.solve(terminal, obs_forcings)
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
        observation_forcings: Optional[List[Optional[PETSc.Vec]]] = None,
    ) -> PETSc.Vec:
        """
        Solve adjoint with checkpointing.

        Uses checkpointing strategy to minimize memory while
        maintaining computational efficiency.

        Parameters
        ----------
        terminal_forcing : PETSc.Vec
            Terminal condition for adjoint.
        observation_forcings : Optional[List[Optional[PETSc.Vec]]]
            Observation forcings at each time.

        Returns
        -------
        PETSc.Vec
            Adjoint at initial time.

        Notes
        -----
        The implementation strategy depends on the checkpointer type:

        For FullTrajectoryCheckpointer:
            1. Retrieve all states and Jacobians from checkpointer
            2. Create standard ImplicitAdjointSolver
            3. Solve normally

        For StateOnlyCheckpointer:
            1. Divide time horizon into segments
            2. For each segment (backward):
               a. Retrieve states from checkpoints
               b. Recompute Jacobians for segment
               c. Solve adjoint backward over segment
            3. Accumulate adjoint contributions

        For BinomialCheckpointer:
            1. Use optimal checkpoint schedule
            2. Recursively recompute forward trajectory segments
            3. Solve adjoint segments in reverse order
            4. Accumulate adjoint contributions
        """
        # Retrieve checkpointed data
        num_steps = self.checkpointer.num_steps

        if observation_forcings is None:
            observation_forcings = [None] * (num_steps + 1)

        # Build trajectory and Jacobians from checkpoints
        trajectory = []
        jacobians = []

        for n in range(num_steps + 1):
            state, jacobian = self.checkpointer.retrieve_forward_data(n)
            trajectory.append(state)
            if jacobian is not None and n < num_steps:
                jacobians.append(jacobian)

        # Handle case where Jacobians need to be recomputed
        if len(jacobians) == 0:
            # Recompute Jacobians using forward model
            for n in range(num_steps):
                if n == 0:
                    # Special handling for first step
                    jacobian = self.forward_model.compute_jacobian(
                        trajectory[n], None, None, n
                    )
                elif n == 1:
                    # BDF2 not active yet
                    jacobian = self.forward_model.compute_jacobian(
                        trajectory[n], trajectory[n - 1], None, n
                    )
                else:
                    # Full BDF2
                    jacobian = self.forward_model.compute_jacobian(
                        trajectory[n], trajectory[n - 1], trajectory[n - 2], n
                    )
                jacobians.append(jacobian)

        # Create standard adjoint solver with retrieved data
        adjoint_solver = ImplicitAdjointSolver(
            self.forward_model, trajectory, jacobians, self.dt
        )

        # Solve adjoint
        lambda_0 = adjoint_solver.solve(terminal_forcing, observation_forcings)

        # Clean up
        for vec in trajectory:
            if vec is not None:
                vec.destroy()
        for mat in jacobians:
            if mat is not None:
                mat.destroy()

        return lambda_0

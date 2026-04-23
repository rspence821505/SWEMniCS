"""
Implicit adjoint solver for BDF2 time-stepping scheme.

Handles the specific time-coupling and transpose systems
required for adjoint of implicit BDF2 discretization.

Mathematical Background
-----------------------
For theta-blended BDF2/BE discretization:
    dQdt = θ·(1/Δt)·(1.5Q - 2Qn + 0.5Qn_old) + (1-θ)·(1/Δt)·(Q - Qn)

The time derivative coefficients are:
    α₀ = (0.5θ + 1)/Δt,  α₁ = -(θ + 1)/Δt,  α₂ = 0.5θ/Δt

The Jacobian from Newton's method includes α₀:
    J_n = ∂R/∂u^{n+1} = α₀·M + ∂F/∂u|_{u^{n+1}}

The adjoint equation becomes:
    J_n^T·λ^n = (θ+1)/Δt·M·λ^{n+1} - 0.5θ/Δt·M·λ^{n+2} + forcing

Key insight: Jacobian J is already computed and stored from
forward Newton solve, so we just need to transpose it.

"""

from typing import List, Optional, Tuple
from petsc4py import PETSc
import numpy as np

# Import BDF2TimeCoefficients for proper time-coupling
from swe4dvar.forward.variational_forms import BDF2TimeCoefficients


# ---------------------------------------------------------------------------
# Bisector diagnostic switch (set by _compute_eq38_from_tlm for 1 call only).
# When set to {"h": np.ndarray, "uv": np.ndarray}, _solve_transpose_system and
# _compute_initial_gradient emit component-resolved norms at every stage so
# we can see which line is killing u/v content. See
# docs/idealized_inlet_tlm_uv_bisector.md.
# ---------------------------------------------------------------------------
_UV_BISECTOR_CTX = {"comp_idx": None}

# One-shot flag: stored-Jacobian structural diagnostic fires on the FIRST
# backward solve it sees while armed, then disables itself for the rest of the
# sweep (avoids dumping 58×4 redundant reports). See
# docs/idealized_inlet_stored_jacobian_diagnostic.md.
_STORED_JAC_DIAG_FIRED = [False]

def _bisector_set_component_indices(comp_idx):
    _UV_BISECTOR_CTX["comp_idx"] = comp_idx
    _STORED_JAC_DIAG_FIRED[0] = False

def _bisector_clear():
    _UV_BISECTOR_CTX["comp_idx"] = None
    _STORED_JAC_DIAG_FIRED[0] = False

def _bisector_log(step_label: str, vec_arr: np.ndarray,
                  extras: Optional[dict] = None) -> None:
    ci = _UV_BISECTOR_CTX["comp_idx"]
    if ci is None:
        return
    try:
        from mpi4py import MPI as _MPI
        comm = _MPI.COMM_WORLD
        rank = comm.Get_rank()
        h  = vec_arr[ci["h"]]
        uv = vec_arr[ci["uv"]]
        tol = 1e-15
        # Collective: take global L2 norm across ranks
        h_sq_local  = float(np.sum(h * h))
        uv_sq_local = float(np.sum(uv * uv))
        h_nz_local  = int(np.sum(np.abs(h) > tol))
        uv_nz_local = int(np.sum(np.abs(uv) > tol))
        h_sq  = comm.allreduce(h_sq_local,  op=_MPI.SUM)
        uv_sq = comm.allreduce(uv_sq_local, op=_MPI.SUM)
        h_nz  = comm.allreduce(h_nz_local,  op=_MPI.SUM)
        uv_nz = comm.allreduce(uv_nz_local, op=_MPI.SUM)
        h_total  = comm.allreduce(len(h),  op=_MPI.SUM)
        uv_total = comm.allreduce(len(uv), op=_MPI.SUM)
        if rank == 0:
            msg = (f"[uv-bisector][GLOBAL] {step_label:<48s}  "
                   f"||h||={np.sqrt(h_sq):.3e}  "
                   f"||uv||={np.sqrt(uv_sq):.3e}  "
                   f"nz_h={h_nz}/{h_total}  nz_uv={uv_nz}/{uv_total}")
            if extras:
                for k, v in extras.items():
                    msg += f"  {k}={v}"
            print(msg, flush=True)
        # Also dump per-rank local view so we can see partition-asymmetry
        per_rank = (f"[uv-bisector][r{rank}] {step_label:<48s}  "
                    f"||h||_loc={np.sqrt(h_sq_local):.3e}  "
                    f"||uv||_loc={np.sqrt(uv_sq_local):.3e}  "
                    f"nz_h_loc={h_nz_local}/{len(h)}  "
                    f"nz_uv_loc={uv_nz_local}/{len(uv)}")
        if extras:
            for k, v in extras.items():
                per_rank += f"  {k}={v}"
        print(per_rank, flush=True)
    except Exception as _e:
        print(f"[uv-bisector] log failed at {step_label}: {_e}", flush=True)


def _stored_jacobian_structural_diag(J, n: int) -> None:
    """
    One-shot structural diagnostic on a stored Jacobian.

    Answers: is this matrix (a) real with zero diagonal, (b) empty, or
    (c) being mis-read? Emits metadata, magnitude histogram, diag/off-diag
    stats, and operator-action tests. Uses `_UV_BISECTOR_CTX["comp_idx"]`
    to do h/uv component analysis in operator-action tests.

    See docs/idealized_inlet_stored_jacobian_diagnostic.md.
    """
    from mpi4py import MPI as _MPI
    comm = _MPI.COMM_WORLD
    rank = comm.Get_rank()
    tag = "[jac-diag]"

    ci = _UV_BISECTOR_CTX["comp_idx"]

    # --- 1. Metadata ---
    gsize = J.getSize()
    lsize = J.getLocalSize()
    try:
        bsize = J.getBlockSize()
    except Exception:
        bsize = -1
    info = J.getInfo()
    nz_used = int(info.get("nz_used", -1))
    nz_allocated = int(info.get("nz_allocated", -1))
    try:
        fro = J.norm(PETSc.NormType.NORM_FROBENIUS)
    except Exception:
        fro = float("nan")
    try:
        inf_norm = J.norm(PETSc.NormType.NORM_INFINITY)
    except Exception:
        inf_norm = float("nan")

    # --- 2. Diagonal stats (allreduce across ranks) ---
    diag = J.getDiagonal()
    d = diag.getArray()
    n_diag_local = d.size
    abs_d = np.abs(d)
    # Local reductions
    if n_diag_local > 0:
        l_max = float(abs_d.max()); l_min = float(abs_d.min())
        l_sum = float(abs_d.sum()); l_nz = int(np.sum(abs_d > 0.0))
    else:
        l_max = 0.0; l_min = float("inf"); l_sum = 0.0; l_nz = 0
    g_max_diag = comm.allreduce(l_max, op=_MPI.MAX)
    g_min_diag = comm.allreduce(l_min, op=_MPI.MIN)
    g_sum_diag = comm.allreduce(l_sum, op=_MPI.SUM)
    g_count_diag = comm.allreduce(n_diag_local, op=_MPI.SUM)
    g_nz_diag = comm.allreduce(l_nz, op=_MPI.SUM)
    g_mean_diag = g_sum_diag / max(g_count_diag, 1)
    diag.destroy()

    # --- 3. Row-wise traversal: off-diag stats + magnitude histogram ---
    # Buckets: [0, 1e-20), [1e-20, 1e-10), [1e-10, 1e-5), [1e-5, ∞)
    # Tracked separately for diagonal vs off-diagonal entries.
    buckets_diag = np.zeros(4, dtype=np.int64)
    buckets_off = np.zeros(4, dtype=np.int64)

    row_start, row_end = J.getOwnershipRange()
    n_off_local = 0
    sum_off = 0.0
    max_off = 0.0
    min_off_nz = float("inf")  # min of NONZERO |off-diag|
    n_total_entries_local = 0

    def _bucket_counts(abs_vals):
        b = np.zeros(4, dtype=np.int64)
        b[0] = int(np.sum(abs_vals < 1e-20))
        b[1] = int(np.sum((abs_vals >= 1e-20) & (abs_vals < 1e-10)))
        b[2] = int(np.sum((abs_vals >= 1e-10) & (abs_vals < 1e-5)))
        b[3] = int(np.sum(abs_vals >= 1e-5))
        return b

    for i in range(row_start, row_end):
        cols, vals = J.getRow(i)
        abs_vals = np.abs(vals)
        n_total_entries_local += abs_vals.size
        diag_mask = (cols == i)
        off_mask = ~diag_mask
        if np.any(diag_mask):
            buckets_diag += _bucket_counts(abs_vals[diag_mask])
        if np.any(off_mask):
            off_vals = abs_vals[off_mask]
            buckets_off += _bucket_counts(off_vals)
            n_off_local += off_vals.size
            sum_off += float(off_vals.sum())
            if off_vals.size > 0:
                max_off = max(max_off, float(off_vals.max()))
                nz_off = off_vals[off_vals > 0.0]
                if nz_off.size > 0:
                    min_off_nz = min(min_off_nz, float(nz_off.min()))

    # Allreduce
    b_diag_g = np.zeros(4, dtype=np.int64)
    b_off_g  = np.zeros(4, dtype=np.int64)
    comm.Allreduce(buckets_diag, b_diag_g, op=_MPI.SUM)
    comm.Allreduce(buckets_off,  b_off_g,  op=_MPI.SUM)
    g_n_off   = comm.allreduce(n_off_local, op=_MPI.SUM)
    g_sum_off = comm.allreduce(sum_off, op=_MPI.SUM)
    g_max_off = comm.allreduce(max_off, op=_MPI.MAX)
    g_min_off = comm.allreduce(min_off_nz, op=_MPI.MIN)
    g_mean_off = (g_sum_off / max(g_n_off, 1))
    g_total_entries = comm.allreduce(n_total_entries_local, op=_MPI.SUM)

    # --- 4. Small dense snapshot (upper-left 3x3 via rank 0 only) ---
    snapshot = None
    if rank == 0:
        try:
            # rank 0 owns rows starting at 0, so rows 0,1,2 are local if lsize>=3
            if row_start == 0 and row_end >= 3:
                snapshot = J.getValues([0, 1, 2], [0, 1, 2])
        except Exception as _e:
            snapshot = f"(snapshot failed: {_e})"

    # --- 5. Operator-action tests ---
    # Build three test vectors with the same layout as J's domain
    def _op_test(label, setup_fn):
        x = J.createVecRight()
        setup_fn(x)
        x.assemble()
        y = J.createVecLeft()
        J.mult(x, y)
        arr = y.getArray()
        # Global L2
        l_norm2 = float(np.sum(arr * arr))
        g_norm2 = comm.allreduce(l_norm2, op=_MPI.SUM)
        g_norm = np.sqrt(g_norm2)
        # Component norms (use ci if available)
        if ci is not None:
            h_l = float(np.sum(arr[ci["h"]] ** 2))
            uv_l = float(np.sum(arr[ci["uv"]] ** 2))
            h_g = np.sqrt(comm.allreduce(h_l, op=_MPI.SUM))
            uv_g = np.sqrt(comm.allreduce(uv_l, op=_MPI.SUM))
        else:
            h_g = uv_g = float("nan")
        x.destroy(); y.destroy()
        return g_norm, h_g, uv_g

    def _set_ones(v): v.set(1.0)

    def _set_h_impulse(v):
        v.set(0.0)
        if ci is not None and len(ci["h"]) > 0:
            local_h = ci["h"]
            arr = v.getArray(readonly=False)
            arr[local_h[0]] = 1.0  # first h-DOF on each rank
            # getArray(readonly=False) returns the underlying buffer;
            # modifications persist. x.assemble() is called by the caller.

    def _set_uv_impulse(v):
        v.set(0.0)
        if ci is not None and len(ci["uv"]) > 0:
            local_uv = ci["uv"]
            arr = v.getArray(readonly=False)
            arr[local_uv[0]] = 1.0  # first uv-DOF on each rank
            # getArray(readonly=False) returns the underlying buffer;
            # modifications persist. x.assemble() is called by the caller.

    def _set_rand(v):
        arr = v.getArray(readonly=False)
        rng = np.random.RandomState(42 + rank)
        arr[:] = rng.randn(arr.size) * 1e-3
        # getArray returns the underlying buffer; assemble() by caller syncs.

    act_ones = _op_test("ones", _set_ones)
    act_himp = _op_test("h-impulse", _set_h_impulse)
    act_uvimp = _op_test("uv-impulse", _set_uv_impulse)
    act_rand = _op_test("rand(1e-3)", _set_rand)

    # --- 6. Rank-0 emit ---
    if rank == 0:
        print(f"\n{tag} ============================================================")
        print(f"{tag} Stored Jacobian structural diagnostic — n={n} (first fire only)")
        print(f"{tag} ============================================================")
        print(f"{tag} [metadata]")
        print(f"{tag}   global size        = {gsize}")
        print(f"{tag}   block size         = {bsize}")
        print(f"{tag}   nz_used            = {nz_used}")
        print(f"{tag}   nz_allocated       = {nz_allocated}")
        print(f"{tag}   total entries seen = {g_total_entries}")
        print(f"{tag}   Frobenius norm     = {fro:.6e}")
        print(f"{tag}   Infinity   norm    = {inf_norm:.6e}")
        print(f"{tag} [diagonal stats]")
        print(f"{tag}   # diag rows (global)      = {g_count_diag}")
        print(f"{tag}   # nonzero |diag| entries  = {g_nz_diag}")
        print(f"{tag}   max |diag|                = {g_max_diag:.6e}")
        print(f"{tag}   min |diag|                = {g_min_diag:.6e}")
        print(f"{tag}   mean |diag|               = {g_mean_diag:.6e}")
        print(f"{tag} [off-diagonal stats]")
        print(f"{tag}   # off-diag entries total  = {g_n_off}")
        print(f"{tag}   max |off-diag|            = {g_max_off:.6e}")
        print(f"{tag}   min |off-diag_nonzero|    = {g_min_off:.6e}")
        print(f"{tag}   mean |off-diag|           = {g_mean_off:.6e}")
        print(f"{tag} [magnitude histogram]   bucket           diag       off-diag")
        bucket_labels = ["[0, 1e-20)", "[1e-20, 1e-10)", "[1e-10, 1e-5)", "[1e-5, inf)"]
        for lab, bd, bo in zip(bucket_labels, b_diag_g, b_off_g):
            print(f"{tag}                   {lab:<15s}  {int(bd):>10d}  {int(bo):>10d}")
        print(f"{tag} [3x3 upper-left snapshot (rank 0, rows 0..2, cols 0..2)]")
        print(f"{tag}   {snapshot}")
        print(f"{tag} [operator-action tests: ||J·x||_total / ||J·x||_h / ||J·x||_uv]")
        print(f"{tag}   ones        = {act_ones[0]:.3e}  /  h={act_ones[1]:.3e}  /  uv={act_ones[2]:.3e}")
        print(f"{tag}   h-impulse   = {act_himp[0]:.3e}  /  h={act_himp[1]:.3e}  /  uv={act_himp[2]:.3e}")
        print(f"{tag}   uv-impulse  = {act_uvimp[0]:.3e}  /  h={act_uvimp[1]:.3e}  /  uv={act_uvimp[2]:.3e}")
        print(f"{tag}   rand(1e-3)  = {act_rand[0]:.3e}  /  h={act_rand[1]:.3e}  /  uv={act_rand[2]:.3e}")
        print(f"{tag} ============================================================\n", flush=True)


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
        theta: Optional[float] = None,  # NEW: Theta blending parameter for BDF2/BE
        adjoint_regularization: float = 0.0,  # NEW: shift εI on transpose solve for stability
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
        theta : float, optional
            Theta blending parameter for the BDF2/BE time stepping scheme.
            The forward solver uses: dQdt = theta*(BDF2) + (1-theta)*(BE).
            theta=1.0 gives pure BDF2, theta=0.5 gives blended scheme.
            If None, auto-detected from forward_model.solver.theta.

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

        # R2 handoff probe at adjoint init (one-shot, rank 0).
        try:
            from swe4dvar.utils.solver_storage import (
                _HANDOFF, _jac_handoff_log,
            )
            if not _HANDOFF.get("adjoint_fired", False) and jacobians:
                for idx in range(min(2, len(jacobians))):
                    _jac_handoff_log(f"adjoint_init_idx{idx}", jacobians[idx])
                _HANDOFF["adjoint_fired"] = True
        except Exception:
            pass

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

        # Theta blending parameter for BDF2/BE time stepping
        # theta=1.0 → pure BDF2, theta=0.5 → blended BDF2/BE (common default)
        # Auto-detect from forward model if not provided
        if theta is None:
            if hasattr(forward_model, 'solver') and hasattr(forward_model.solver, 'theta'):
                theta = forward_model.solver.theta
            elif hasattr(forward_model, 'theta'):
                theta = forward_model.theta
            else:
                theta = 1.0  # Default to pure BDF2 (backward compatible)
        self.theta = theta

        # Adjoint regularization: add εI to J before solving (J+εI)^T λ = f.
        # Damps near-null directions so direct solvers pick consistent solutions
        # across serial and distributed runs. Default 0.0 = no regularization.
        # Env var SWE4DVAR_ADJOINT_REG overrides the constructor default.
        import os
        env_reg = os.environ.get("SWE4DVAR_ADJOINT_REG")
        if env_reg is not None:
            try:
                adjoint_regularization = float(env_reg)
            except ValueError:
                pass
        self.adjoint_regularization = float(adjoint_regularization)

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

        # UFL-based flux mass matrix action (replaces pointwise approximation)
        self._flux_mass_form = None
        self._flux_state_func = None
        self._flux_lambda_func = None
        if flux_formulation:
            self._init_flux_mass_form()

    def _init_flux_mass_form(self):
        """
        Initialize UFL form for computing M_Q^T * lambda using proper FE integration.

        The flux mass matrix M_Q arises from the conservative time derivative:
            R_time = integral (dQ/dt) . v dx

        where Q = [h_wd, h_wd*ux, h_wd*uy] and the state variables are [h, ux, uy].
        Here h_wd = h + f(h) is the wetting-drying corrected depth (h_wd = h when
        WD is disabled). The cross-time derivative dR^{n+1}/du^n = -(c) * M_Q|_{u^n}.

        The transpose action M_Q^T * lambda is:
            (M_Q^T * lambda)_h  = integral dh_wd * (lambda_h + ux_s * lambda_ux + uy_s * lambda_uy) * v_h dx
            (M_Q^T * lambda)_ux = integral (h_wd * lambda_ux) * v_ux dx
            (M_Q^T * lambda)_uy = integral (h_wd * lambda_uy) * v_uy dx

        where dh_wd = dh_wd/dh is the derivative of the WD correction (1 when WD disabled).
        """
        import os
        os.environ.setdefault("CC", "/usr/bin/clang")

        V = None
        if hasattr(self.forward_model, 'solver') and hasattr(self.forward_model.solver, 'V'):
            V = self.forward_model.solver.V
        elif hasattr(self.forward_model, 'V'):
            V = self.forward_model.V

        if V is None:
            import warnings
            warnings.warn(
                "Cannot initialize UFL flux mass form: function space V not found. "
                "Falling back to pointwise flux scaling.",
                RuntimeWarning,
            )
            self.flux_formulation = False
            return

        # Detect wetting-drying from the forward model's problem
        problem = None
        if hasattr(self.forward_model, 'problem'):
            problem = self.forward_model.problem
        elif hasattr(self.forward_model, 'solver') and hasattr(self.forward_model.solver, 'problem'):
            problem = self.forward_model.solver.problem
        self._wd_active = problem is not None and hasattr(problem, 'wd') and problem.wd
        self._wd_alpha = getattr(problem, 'wd_alpha', 0.05) if self._wd_active else None

        try:
            from dolfinx import fem
            from ufl import TestFunction, split, inner, dx, sqrt as ufl_sqrt
            from petsc4py import PETSc

            self._flux_state_func = fem.Function(V)
            self._flux_lambda_func = fem.Function(V)

            v = TestFunction(V)
            h_v = split(v)[0]
            vel_v = split(v)[1]
            ux_v, uy_v = vel_v[0], vel_v[1]

            h_s = split(self._flux_state_func)[0]
            vel_s = split(self._flux_state_func)[1]
            ux_s, uy_s = vel_s[0], vel_s[1]

            h_l = split(self._flux_lambda_func)[0]
            vel_l = split(self._flux_lambda_func)[1]
            ux_l, uy_l = vel_l[0], vel_l[1]

            # Compute WD-corrected depth and its derivative
            # The forward solver uses Q = [h_wd, h_wd*ux, h_wd*uy] where:
            #   h_wd = h + 0.5*(sqrt(h^2 + alpha^2) - h)
            #   dh_wd/dh = 0.5 + 0.5*h/sqrt(h^2 + alpha^2)
            # Without WD: h_wd = h, dh_wd/dh = 1
            if self._wd_active:
                wd_alpha_sq = fem.Constant(V.mesh, PETSc.ScalarType(self._wd_alpha**2))
                h_wd = h_s + 0.5 * (ufl_sqrt(h_s**2 + wd_alpha_sq) - h_s)
                dh_wd = 0.5 + 0.5 * h_s / ufl_sqrt(h_s**2 + wd_alpha_sq)
            else:
                h_wd = h_s
                dh_wd = 1.0

            # M_Q^T * lambda as a linear form in v:
            # (dQ/du)^T = [[dh_wd,      dh_wd*ux, dh_wd*uy],
            #              [0,           h_wd,     0        ],
            #              [0,           0,        h_wd     ]]
            L = (inner(dh_wd * (h_l + ux_s * ux_l + uy_s * uy_l), h_v) * dx
                 + inner(h_wd * ux_l, ux_v) * dx
                 + inner(h_wd * uy_l, uy_v) * dx)

            self._flux_mass_form = fem.form(L)
        except Exception as e:
            import warnings
            warnings.warn(
                f"Failed to initialize UFL flux mass form: {e}. "
                "Falling back to pointwise flux scaling.",
                RuntimeWarning,
            )
            self._flux_mass_form = None

    def _compute_flux_mass_transpose_action(
        self, state: PETSc.Vec, lambda_vec: PETSc.Vec
    ) -> PETSc.Vec:
        """
        Compute M_Q^T * lambda using proper FE integration.

        Parameters
        ----------
        state : PETSc.Vec
            State vector at time n (provides h, ux, uy for flux scaling).
        lambda_vec : PETSc.Vec
            Adjoint vector to apply M_Q^T to.

        Returns
        -------
        PETSc.Vec
            Result of M_Q^T * lambda_vec.
        """
        if self._flux_mass_form is None:
            # Fallback to old pointwise method
            M = self._get_mass_matrix()
            result = lambda_vec.duplicate()
            M.mult(lambda_vec, result)
            result = self._apply_flux_scaling(result, state)
            return result

        from dolfinx import fem
        from dolfinx import la

        V = self._flux_state_func.function_space

        # Update state and lambda function values
        u_owned = V.dofmap.index_map.size_local
        state_arr = state.getArray()
        lambda_arr = lambda_vec.getArray()

        if len(state_arr) >= u_owned:
            self._flux_state_func.x.array[:u_owned] = state_arr[:u_owned]
        else:
            self._flux_state_func.x.array[:len(state_arr)] = state_arr
        self._flux_state_func.x.scatter_forward()

        if len(lambda_arr) >= u_owned:
            self._flux_lambda_func.x.array[:u_owned] = lambda_arr[:u_owned]
        else:
            self._flux_lambda_func.x.array[:len(lambda_arr)] = lambda_arr
        self._flux_lambda_func.x.scatter_forward()

        # Assemble linear form
        result_petsc = fem.petsc.assemble_vector(self._flux_mass_form)
        result_petsc.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        result_petsc.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # Convert to owned-only PETSc Vec matching the adjoint vectors
        from swe4dvar.utils.compat import create_petsc_vector_from_map as _cvm_
        result = _cvm_(V.dofmap.index_map, V.dofmap.index_map_bs)
        result.setArray(result_petsc.getArray()[:u_owned])
        result.assemble()

        result_petsc.destroy()
        return result

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
        # The forward solver uses theta-blended time stepping:
        #   dQdt = theta*(1/dt)*(1.5Q - 2Qn + 0.5Qn_old) + (1-theta)*(1/dt)*(Q - Qn)
        # Coefficient of Qn: (-theta - 1)/dt
        # So ∂R^{n+1}/∂u^n = (-theta-1)/dt * M_Q
        # In adjoint: contributes (theta+1)/dt * M_Q * λ_{n+1} (sign flip)
        if self._uses_bdf2_at_step(n + 1):
            c_next = (self.theta + 1.0) / dt
        else:
            # Backward Euler: R^{n+1} = (u^{n+1} - u^n)/dt + F
            # ∂R^{n+1}/∂u^n = -1/dt * M
            # In adjoint: contributes +1/dt * M * λ_{n+1}
            c_next = 1.0 / dt

        # Coefficient from R^{n+2} (sensitivity to u^n, only for BDF2)
        # Coefficient of Qn_old: 0.5*theta/dt
        # So ∂R^{n+2}/∂u^n = 0.5*theta/dt * M_Q
        # In adjoint: contributes -0.5*theta/dt * M_Q * λ_{n+2}
        if self._uses_bdf2_at_step(n + 2):
            c_next_next = -0.5 * self.theta / dt
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
        return_history: bool = False,
    ):
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
        PETSc.Vec or Tuple[PETSc.Vec, List[Optional[PETSc.Vec]]]
            By default returns the initial-time gradient contribution.
            When ``return_history=True``, also returns the timestep adjoint
            states ``[None, λ_1, ..., λ_N]``.

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

        lambda_history = [None] * (self.num_steps + 1) if return_history else None

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
        if lambda_history is not None:
            lambda_history[self.num_steps] = lambda_next.copy()
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
            if lambda_history is not None:
                lambda_history[n] = lambda_n.copy()

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
        if return_history:
            return gradient_u0, lambda_history
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
        - Tries direct LU factorization first (fastest, exact).
        - If LU fails (near-singular Jacobian), falls back to GMRES with
          ILU preconditioning (more robust to ill-conditioning).
        - The transpose solve is handled automatically by PETSc's
          solveTranspose().
        - Jacobian indexing: jacobians[k] stores ∂R^(k+1)/∂u^(k+1),
          so for λ^n we need jacobians[n-1] to get ∂R^n/∂u^n.
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

        # Stored-Jacobian structural diagnostic: one-shot, fires on the FIRST
        # J consumed while the bisector is armed. See
        # docs/idealized_inlet_stored_jacobian_diagnostic.md.
        if (_UV_BISECTOR_CTX["comp_idx"] is not None
                and not _STORED_JAC_DIAG_FIRED[0]):
            _STORED_JAC_DIAG_FIRED[0] = True
            try:
                _stored_jacobian_structural_diag(J, n)
            except Exception as _e:
                from mpi4py import MPI as _MPI
                if _MPI.COMM_WORLD.Get_rank() == 0:
                    print(f"[jac-diag] diagnostic raised: {_e}", flush=True)

        # Check for near-zero diagonal entries (dry nodes in wetting/drying).
        # Dry nodes produce zero rows in J, making J^T singular.
        # We regularize by using a shifted operator J + εI for the solve,
        # then zero out the adjoint at dry-node DOFs afterward.
        #
        # Threshold is RELATIVE to the largest diagonal magnitude of J, with
        # an absolute floor. The original absolute-only threshold (1e-20) was
        # mis-calibrated for DG-mixed-element problems whose natural diagonal
        # scale is small (idealized inlet showed every DOF < 1e-20), which
        # caused the safeguard to annihilate the entire adjoint vector.
        # See docs/idealized_inlet_tlm_uv_bisector.md.
        diag = J.getDiagonal()
        diag_arr = diag.getArray()

        # Collective MAX of |diag| across ranks so all ranks agree on scale.
        from mpi4py import MPI as _MPI
        local_max = float(np.max(np.abs(diag_arr))) if diag_arr.size > 0 else 0.0
        comm_j = J.getComm().tompi4py()
        diag_max = comm_j.allreduce(local_max, op=_MPI.MAX)

        abs_floor = 1e-20
        rel_floor = 1e-12 * diag_max if diag_max > 0.0 else 0.0
        tiny_thresh = max(abs_floor, rel_floor)
        tiny_mask = np.abs(diag_arr) < tiny_thresh
        n_regularized = int(np.sum(tiny_mask))
        diag.destroy()

        # UV bisector: per-component breakdown of tiny_mask + forcing norms
        _ci = _UV_BISECTOR_CTX["comp_idx"]
        if _ci is not None:
            tiny_h_count  = int(np.sum(tiny_mask[_ci["h"]]))
            tiny_uv_count = int(np.sum(tiny_mask[_ci["uv"]]))
            _bisector_log(
                f"n={n}  BEFORE solve (forcing)",
                forcing.getArray(),
                extras={"tiny_h": f"{tiny_h_count}/{len(_ci['h'])}",
                        "tiny_uv": f"{tiny_uv_count}/{len(_ci['uv'])}",
                        "n_tiny_total": n_regularized,
                        "diag_max": f"{diag_max:.3e}",
                        "tiny_thresh": f"{tiny_thresh:.3e}"},
            )

        # Two regularizations combined into one shifted copy of J:
        # 1. Dry-node regularization: set J[i,i] = 1 for near-zero diagonals
        # 2. Global adjoint regularization: add εI to J (damps near-null directions
        #    so serial and distributed LU/MUMPS converge to the same solution)
        # Do NOT modify J in-place — it's shared across multiple adjoint solves.
        eps = self.adjoint_regularization
        if n_regularized > 0 or eps > 0:
            J_reg = J.copy()
            shift_diag = J_reg.getDiagonal()
            shift_arr = shift_diag.getArray()
            if n_regularized > 0:
                shift_arr[tiny_mask] = 1.0  # Dry-node identity block
            if eps > 0:
                shift_arr = shift_arr + eps  # Global εI
            shift_diag.setArray(shift_arr)
            J_reg.setDiagonal(shift_diag)
            J_reg.assemble()
            shift_diag.destroy()
            J_solve = J_reg
        else:
            J_solve = J
            J_reg = None

        lambda_n = forcing.duplicate()

        # Iterative-solve override: env var SWE4DVAR_ADJOINT_ITERATIVE=1 forces
        # GMRES+ILU with tight tolerance instead of LU. Iterative Krylov methods
        # converge to the solution in the residual-norm sense, which tends to
        # be more reproducible across distribution layouts than direct LU
        # (which is sensitive to pivoting order and null-space projections).
        import os
        force_iter = os.environ.get("SWE4DVAR_ADJOINT_ITERATIVE", "0") == "1"

        if force_iter:
            ksp = PETSc.KSP().create(J_solve.getComm())
            ksp.setOperators(J_solve)
            ksp.setType(PETSc.KSP.Type.GMRES)
            # Use block Jacobi with ILU sub-blocks in MPI (ILU alone doesn't
            # work on mpiaij matrices in PETSc). In serial, this is equivalent
            # to plain ILU.
            comm_size = J_solve.getComm().getSize()
            pc = ksp.getPC()
            if comm_size > 1:
                pc.setType(PETSc.PC.Type.BJACOBI)
            else:
                pc.setType(PETSc.PC.Type.ILU)
            iter_rtol = float(os.environ.get("SWE4DVAR_ADJOINT_ITER_RTOL", "1e-10"))
            iter_atol = float(os.environ.get("SWE4DVAR_ADJOINT_ITER_ATOL", "1e-14"))
            iter_maxit = int(os.environ.get("SWE4DVAR_ADJOINT_ITER_MAXIT", "5000"))
            ksp.setTolerances(rtol=iter_rtol, atol=iter_atol, max_it=iter_maxit)
            ksp.setErrorIfNotConverged(False)
            ksp.solveTranspose(forcing, lambda_n)
            reason = ksp.getConvergedReason()
            ksp.destroy()
        else:
            # --- Strategy 1: Direct LU (fast, exact when it works) ---
            # Env var SWE4DVAR_ADJOINT_LU_SOLVER selects the factor-solver package:
            #   "mumps" (PETSc default at np>1 on LS6 — unstable on DG blocks
            #            with ~25K DOFs/rank at np=8 in this repo as of 2026-04-23)
            #   "superlu_dist" (alternate distributed LU; different pivoting)
            #   unset  = PETSc's own default
            import os as _os
            _lu_solver = _os.environ.get("SWE4DVAR_ADJOINT_LU_SOLVER", "").strip().lower()

            ksp = PETSc.KSP().create(J_solve.getComm())
            ksp.setOperators(J_solve)
            ksp.setType(PETSc.KSP.Type.PREONLY)
            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.LU)
            if _lu_solver:
                try:
                    pc.setFactorSolverType(_lu_solver)
                except Exception as _e:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"setFactorSolverType('{_lu_solver}') failed: {_e}. "
                        f"Falling back to PETSc default."
                    )
            ksp.setErrorIfNotConverged(False)

            ksp.solveTranspose(forcing, lambda_n)
            reason = ksp.getConvergedReason()

            # Aggressive cleanup: distributed LU factor-solver packages
            # (MUMPS, SuperLU_DIST) allocate large internal workspace that
            # the default ksp.destroy() path sometimes fails to release in
            # a per-iteration Gram-matrix loop (observed: ~9 GB/solve leak
            # with SuperLU_DIST at np=8 on 207K-DOF mesh, commit 9fef6b0).
            # Explicitly destroy the factor matrix, then force Python GC.
            try:
                _factor = pc.getFactorMatrix()
                if _factor is not None:
                    _factor.destroy()
            except Exception:
                pass  # not all PC configs expose a factor matrix
            ksp.destroy()
            import gc as _gc
            _gc.collect()

        # --- Strategy 2: GMRES + ILU fallback if LU failed ---
        if reason < 0:
            import logging
            logging.getLogger(__name__).warning(
                f"Adjoint LU transpose solve failed at step {n} (reason={reason}). "
                f"Falling back to GMRES+ILU."
            )

            ksp2 = PETSc.KSP().create(J_solve.getComm())
            ksp2.setOperators(J_solve)
            ksp2.setType(PETSc.KSP.Type.GMRES)
            ksp2.getPC().setType(PETSc.PC.Type.ILU)
            ksp2.setTolerances(rtol=1e-8, atol=1e-10, max_it=2000)
            ksp2.setErrorIfNotConverged(False)

            lambda_n.zeroEntries()
            ksp2.solveTranspose(forcing, lambda_n)
            reason2 = ksp2.getConvergedReason()
            iters2 = ksp2.getIterationNumber()
            # ILU PC also holds a factor matrix; same cleanup as LU path.
            try:
                _factor2 = ksp2.getPC().getFactorMatrix()
                if _factor2 is not None:
                    _factor2.destroy()
            except Exception:
                pass
            ksp2.destroy()
            import gc as _gc
            _gc.collect()

            if reason2 < 0:
                # --- Strategy 3: GMRES with no preconditioning (last resort) ---
                logging.getLogger(__name__).warning(
                    f"Adjoint GMRES+ILU fallback also failed at step {n} "
                    f"(reason={reason2}). Trying GMRES with no PC."
                )
                ksp3 = PETSc.KSP().create(J_solve.getComm())
                ksp3.setOperators(J_solve)
                ksp3.setType(PETSc.KSP.Type.GMRES)
                ksp3.getPC().setType(PETSc.PC.Type.NONE)
                ksp3.setTolerances(rtol=1e-6, atol=1e-8, max_it=5000)
                ksp3.setErrorIfNotConverged(False)

                lambda_n.zeroEntries()
                ksp3.solveTranspose(forcing, lambda_n)
                reason3 = ksp3.getConvergedReason()
                iters3 = ksp3.getIterationNumber()
                ksp3.destroy()

                if reason3 < 0:
                    if J_reg is not None:
                        J_reg.destroy()
                    raise RuntimeError(
                        f"Adjoint transpose solve failed at step {n} after all "
                        f"strategies: LU(reason={reason}), GMRES+ILU(reason={reason2}, "
                        f"iters={iters2}), GMRES(reason={reason3}, iters={iters3})"
                    )

        # Clean up regularized copy
        if J_reg is not None:
            J_reg.destroy()

        # UV bisector: state right after solve, before tiny-mask zeroing
        _ci = _UV_BISECTOR_CTX["comp_idx"]
        if _ci is not None:
            _bisector_log(f"n={n}  AFTER solveTranspose (pre tiny-mask)",
                          lambda_n.getArray())

        # Zero out adjoint at dry-node DOFs — these are numerically meaningless
        # after solving with the regularized Jacobian.
        if n_regularized > 0:
            arr = lambda_n.getArray()
            arr[tiny_mask] = 0.0
            lambda_n.setArray(arr)
            lambda_n.assemble()

        if _ci is not None:
            _bisector_log(f"n={n}  AFTER tiny-mask zeroing",
                          lambda_n.getArray())

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

        # Get initial state for flux scaling (∂R^1/∂u^0 is evaluated at u^0)
        state_0 = self.trajectory[0] if len(self.trajectory) > 0 else None

        # Coefficient for λ^1 from R^1
        # Step 1 is always backward Euler (1 < bdf2_start_step for any reasonable value)
        # Backward Euler: R^1 = (Q^1 - Q^0)/dt + F(u^1) = 0
        # ∂R^1/∂u^0 = -1/dt · M_Q|_{u^0}
        c_1 = -1.0 / dt

        # Compute c_1·M_Q^T·λ^1 using proper FE integration
        if self.flux_formulation and state_0 is not None and self._flux_mass_form is not None:
            result = self._compute_flux_mass_transpose_action(state_0, lambda_1)
        else:
            M = self._get_mass_matrix()
            result = lambda_1.duplicate()
            M.mult(lambda_1, result)

        result.scale(c_1)

        # Coefficient for λ^2 from R^2 (only contributes if R^2 uses BDF2)
        if self.use_bdf2 and lambda_2 is not None and self._uses_bdf2_at_step(2):
            # Theta-blended BDF2 at step 2:
            # Coefficient of Qn_old (= Q^0): 0.5*theta/dt
            # ∂R^2/∂u^0 = 0.5*theta/dt · M_Q|_{u^0}
            c_2 = 0.5 * self.theta / dt
            if self.flux_formulation and state_0 is not None and self._flux_mass_form is not None:
                temp = self._compute_flux_mass_transpose_action(state_0, lambda_2)
            else:
                M = self._get_mass_matrix()
                temp = lambda_1.duplicate()
                M.mult(lambda_2, temp)

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

        # UV bisector: gradient at n=0 BEFORE BC zeroing
        _ci = _UV_BISECTOR_CTX["comp_idx"]
        if _ci is not None:
            _bisector_log("n=0 (gradient_u0) BEFORE BC zeroing",
                          result.getArray())

        # Apply homogeneous Dirichlet BCs to the gradient
        # BC DOFs in the initial condition are fixed (not part of control space),
        # so their gradient should be zero to prevent optimizer from changing them.
        if self.bc_dof_indices is not None and len(self.bc_dof_indices) > 0:
            result_arr = result.getArray()
            # UV bisector: count BC DOFs landing on each component
            if _ci is not None:
                bc_arr = np.fromiter(self.bc_dof_indices, dtype=np.int64)
                bc_arr = bc_arr[bc_arr < len(result_arr)]
                h_set  = set(_ci["h"].tolist())
                uv_set = set(_ci["uv"].tolist())
                bc_in_h  = int(sum(1 for d in bc_arr if d in h_set))
                bc_in_uv = int(sum(1 for d in bc_arr if d in uv_set))
                print(f"[uv-bisector] BC mask: total={len(bc_arr)}  "
                      f"bc∩h={bc_in_h}/{len(_ci['h'])}  "
                      f"bc∩uv={bc_in_uv}/{len(_ci['uv'])}", flush=True)
            for dof in self.bc_dof_indices:
                if dof < len(result_arr):
                    result_arr[dof] = 0.0
            result.setArray(result_arr)

        if _ci is not None:
            _bisector_log("n=0 (gradient_u0) AFTER BC zeroing",
                          result.getArray())

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

        # Get state at time n for flux scaling (∂R^{n+1}/∂u^n is evaluated at u^n)
        state_n = self.trajectory[n] if n < len(self.trajectory) else None

        # Time coupling: c_{n+1}·M_Q^T·λ^{n+1}
        # For flux formulation Q=[h, h*ux, h*uy], M_Q includes the state-dependent
        # Jacobian ∂Q/∂u. Use proper FE integration for accuracy.
        if self.flux_formulation and state_n is not None and self._flux_mass_form is not None:
            # Proper UFL-based computation: M_Q^T * lambda
            forcing = self._compute_flux_mass_transpose_action(state_n, lambda_next)
        else:
            # Standard mass matrix (no flux formulation or fallback)
            M = self._get_mass_matrix()
            forcing = lambda_next.duplicate()
            M.mult(lambda_next, forcing)

        forcing.scale(c_next)

        # Add c_{n+2}·M_Q^T·λ^{n+2}
        if lambda_next_next is not None and abs(c_next_next) > 1e-14:
            if self.flux_formulation and state_n is not None and self._flux_mass_form is not None:
                temp = self._compute_flux_mass_transpose_action(state_n, lambda_next_next)
            else:
                M = self._get_mass_matrix()
                temp = lambda_next.duplicate()
                M.mult(lambda_next_next, temp)

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

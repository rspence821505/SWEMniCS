"""
Custom Newton solver for general nonlinear variational problems.

This was implemented because more control was desired over the Newton iteration
than provided by the built-in NonlinearProblem class.

MODIFICATIONS FOR 4D-VAR:
- CustomNewtonProblem.solve() now accepts return_jacobian flag
- Returns final Jacobian matrix for efficient adjoint computation
- Jacobian is reassembled at converged solution for correctness
- NewtonSolver.solve() also returns Jacobian when requested

DIAGNOSTICS:
- Integrated with NewtonDiagnostics for flexible logging/analysis
- Supports console printing, file logging, and in-memory storage
- Backward compatible with verbose printing behavior
"""

from dolfinx import fem as fe, nls, log, geometry, io, cpp
import dolfinx.fem.petsc as petsc
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from swe4dvar.utils.compat import create_vector_from_form as _cvf
from scipy.sparse import csr_matrix
import numpy.linalg as la
import sys
import time

from swe4dvar.utils.newton_diagnostics import NewtonDiagnostics


def petsc_to_csr(A):
    indptr, indices, data = A.getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=A.size)


class CustomNewtonProblem:
    """An all-in-one class that solves a nonlinear problem. . ."""

    def __init__(self, obj1, solver_parameters={}, diagnostics=None):
        """initialize the problem

        Args:
            obj1: Problem object with F, u, verbose, problem attributes
            solver_parameters: Dict of solver configuration parameters
            diagnostics: Optional NewtonDiagnostics instance for convergence tracking
        """
        self.u = obj1.u
        self.F = obj1.F
        self.residual = fe.form(self.F)
        self.verbose = obj1.verbose

        self.J = ufl.derivative(self.F, self.u)
        self.jacobian = fe.form(self.J)

        self.bcs = obj1.problem.dirichlet_bcs
        self.comm = obj1.problem.mesh.comm

        # relative tolerance for Newton solver
        self.rtol = 1e-5
        # absolute tolerance for Newton solver
        self.atol = 1e-6
        # max iteration number for Newton solver
        self.max_it = 5
        # relaxation parameter for Newton solver
        self.relaxation_parameter = 1.00
        # underlying linear solver
        # For serial: use direct solver (LU) which is robust to boundary conditions
        # For parallel: use GMRES with block Jacobi preconditioner
        if self.comm.Get_size() == 1:
            self.ksp_type = "preonly"  # Direct solver, no iterations
            self.pc_type = "lu"  # LU factorization
        else:
            self.ksp_type = "gmres"
            self.pc_type = "bjacobi"

        for k, v in solver_parameters.items():
            setattr(self, k, v)

        # Setup diagnostics for convergence tracking
        if diagnostics is None:
            # Default: match current verbose printing behavior
            self.diagnostics = NewtonDiagnostics(
                print_to_console=self.verbose,
                log_file=None,
                store_history=False,
                verbose=self.verbose,
            )
        else:
            self.diagnostics = diagnostics

        # Internal counter for tracking solve calls (used as timestep if not externally managed)
        self._solve_counter = 0

        self._solver_parameters = solver_parameters
        self._mesh = obj1.problem.mesh
        self.A = petsc.create_matrix(self.jacobian)
        self.L = _cvf(self.residual)
        self._setup_ksp()

    def _setup_ksp(self):
        """Create and configure the KSP linear solver.

        Can be called to rebuild the solver after a MUMPS/LU failure
        that corrupts the internal factorization state.
        """
        if hasattr(self, 'solver') and self.solver is not None:
            self.solver.destroy()

        self.solver = PETSc.KSP().create(self.comm)
        self.solver.setOptionsPrefix("newton_")
        self.solver.setTolerances(
            rtol=self._solver_parameters.get("ksp_rtol", 1e-8),
            atol=self._solver_parameters.get("ksp_atol", 1e-9),
            max_it=self._solver_parameters.get("ksp_max_it", 1000),
        )
        self.solver.setOperators(self.A)
        # CRITICAL: Must be False to prevent PETSc from aborting on MUMPS failures.
        # When True, MUMPS factorization errors (e.g., INFOG=-9) cause PETSc to
        # throw exceptions that corrupt internal solver state, making all subsequent
        # solves hang. Instead, we check convergence codes ourselves after each solve.
        self.solver.setErrorIfNotConverged(False)

        if self.pc_type == "element_block":
            self.pc = ElementBlockPreconditioner(self.A, self._mesh)
        else:
            self.pc = self.solver.getPC()
            self.pc.setType(self.pc_type)
            # MUMPS null-pivot detection for parallel LU path. Storm-peak
            # forward solves at v=20+ produce locally singular Jacobians
            # (h ≈ 2 m + huge wind drag → unfactorizable system) and MUMPS
            # bails with INFOG(1)=-9 → KSP_DIVERGED_PC_FAILED (code -11).
            # ICNTL(24)=1 enables null-pivot detection and adjustment.
            if self.pc_type == "lu":
                import os as _os
                if _os.environ.get("SWE4DVAR_FORWARD_MUMPS_NULLPIVOT") == "1":
                    opts = PETSc.Options()
                    opts["newton_pc_factor_mat_solver_type"] = "mumps"
                    opts["newton_mat_mumps_icntl_24"] = "1"
                    opts["newton_mat_mumps_cntl_3"] = _os.environ.get(
                        "SWE4DVAR_FORWARD_MUMPS_CNTL3", "1e-8")
                    opts["newton_mat_mumps_cntl_1"] = _os.environ.get(
                        "SWE4DVAR_FORWARD_MUMPS_CNTL1", "1e-6")
                    self.solver.setFromOptions()
                    if self.comm.rank == 0:
                        print(f"  [Newton MUMPS] null-pivot detection enabled "
                              f"(icntl_24=1, cntl_3={opts['newton_mat_mumps_cntl_3']}, "
                              f"cntl_1={opts['newton_mat_mumps_cntl_1']})",
                              flush=True)
            if self.pc_type == "bjacobi":
                opts = PETSc.Options()
                import os as _os
                sub_pc = _os.environ.get("SWE4DVAR_FORWARD_SUB_PC", "none")
                sub_max_it = _os.environ.get("SWE4DVAR_FORWARD_SUB_KSP_MAX_IT", "100")
                opts["newton_sub_pc_type"] = sub_pc
                opts["newton_sub_ksp_type"] = "gmres"
                opts["newton_sub_ksp_max_it"] = sub_max_it
                if sub_pc in ("ilu", "lu"):
                    opts["newton_sub_pc_factor_shift_type"] = _os.environ.get(
                        "SWE4DVAR_FORWARD_SUB_SHIFT", "NONZERO")
                self.solver.setFromOptions()
                # Force the sub-KSPs to be configured. PETSc creates sub-KSPs
                # lazily, after first solve. Setting via options DB alone can
                # miss them if the order isn't right. Explicitly iterate them
                # here to lock the sub PC/KSP types in.
                if sub_pc != "none":
                    try:
                        self.pc.setUp()
                        sub_ksps = self.pc.getASMSubKSP() if self.pc_type == "asm" else self.pc.getSubKSP()
                        for _sub in sub_ksps:
                            _sub.setType(PETSc.KSP.Type.GMRES)
                            _spc = _sub.getPC()
                            _spc.setType(sub_pc)
                            if sub_pc in ("ilu", "lu"):
                                try:
                                    _spc.setFactorShift(PETSc.PC.FactorShiftType.NONZERO)
                                except Exception:
                                    pass
                        if self.comm.rank == 0:
                            _check = self.pc.getSubKSP()[0].getPC().getType()
                            print(f"  [Newton KSP] outer={self.pc_type} "
                                  f"sub_pc={_check} sub_ksp_max_it={sub_max_it}",
                                  flush=True)
                    except Exception as _e:
                        if self.comm.rank == 0:
                            print(f"  [Newton KSP] sub-KSP config failed: {_e}",
                                  flush=True)

    def log(self, *msg):
        if self.comm.rank == 0:
            print(*msg)

    def destroy(self):
        """Release PETSc objects held by this Newton problem.

        Why: solve_init in cg_implicit.py creates a fresh CustomNewtonProblem
        every time_loop() call. Without explicit destroy of A, L, and the KSP
        (with its LU factorization), each cost-function eval leaks ~240 MB of
        C-level memory — observed as ~1.5 GB/eval RSS growth across cycling DA.
        """
        for attr in ("solver", "A", "L"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.destroy()
                except Exception:
                    pass
                setattr(self, attr, None)
        # Cached Newton update Function (Phase C-7) — backing Vec needs
        # explicit destroy.
        _dx = getattr(self, "_dx_cache", None)
        if _dx is not None:
            try:
                _dx.x.petsc_vec.destroy()
            except Exception:
                pass
            self._dx_cache = None
        # ElementBlockPreconditioner has internal PETSc objects too
        if getattr(self, "pc_type", None) == "element_block":
            pc_obj = getattr(self, "pc", None)
            if pc_obj is not None and hasattr(pc_obj, "destroy"):
                try:
                    pc_obj.destroy()
                except Exception:
                    pass
                self.pc = None

    def solve(self, u, max_it=5, return_jacobian=False, timestep=None, time=None):
        """Solve the nonlinear problem at u

        Args:
            u: Solution function to update in-place
            max_it: Maximum Newton iterations (default: 5)
            return_jacobian: If True, return final Jacobian for adjoint (default: False)
            timestep: Optional timestep number for diagnostics tracking
            time: Optional simulation time for diagnostics tracking

        Returns:
            If return_jacobian=False: None (solution stored in u)
            If return_jacobian=True: (None, J) where J is the final Jacobian matrix (PETSc.Mat)
                J = ∂R/∂u evaluated at the converged solution u

        Notes:
            For 4D-Var data assimilation, the Jacobian is reassembled at the converged
            solution to ensure correctness. During the Newton loop, the Jacobian is
            evaluated at u_i, then we update u_{i+1} = u_i + dx_i. When we break due
            to convergence, we need J evaluated at u_{i+1}, not u_i.
        """

        # Start diagnostics tracking for this solve
        if timestep is None:
            timestep = self._solve_counter
            self._solve_counter += 1
        self.diagnostics.start_timestep(timestep, time)

        # Cache the Newton update Function across solves (Phase C-7).
        # Without caching, every Newton solve allocates a fresh
        # ``fem.Function(u._V)`` whose backing PETSc Vec (~1.6 MB at 207K
        # DOFs) leaks at C level when Python drops the ref. With nt_da=6
        # timesteps × ~13 Newton iters × N cost evals, this churn pools
        # ~10 MB / eval. Caching by FunctionSpace identity keeps one Vec
        # for the lifetime of the Newton problem (now persistent across
        # cost evals via Phase C-1).
        _cached_dx = getattr(self, "_dx_cache", None)
        if _cached_dx is None or _cached_dx.function_space is not u._V:
            if _cached_dx is not None:
                try:
                    _cached_dx.x.petsc_vec.destroy()
                except Exception:
                    pass
            self._dx_cache = fe.Function(u._V)
        dx = self._dx_cache
        # Zero the cached Newton update at the start of each solve.
        dx.x.array[:] = 0.0
        i = 0
        converged = False
        rank = self.comm.rank
        A, L, solver = self.A, self.L, self.solver
        relaxation_parameter = self.relaxation_parameter

        while i < self.max_it:
            # Assemble Jacobian and residual at current u
            with L.localForm() as loc_L:
                loc_L.set(0)

            A.zeroEntries()

            # R2 handoff: one-shot — immediately after the FIRST A.zeroEntries()
            # that fires AFTER a Jacobian was saved to storage, check whether
            # the previously saved matrix still has its norm. If it collapses
            # here, the stored "copy" is aliasing A's value buffer.
            try:
                from swe4dvar.utils.solver_storage import (
                    _HANDOFF, _jac_handoff_log,
                )
                if (_HANDOFF.get("storage_fired", False)
                        and not _HANDOFF.get("postzero_fired", False)
                        and _HANDOFF.get("last_saved") is not None):
                    _jac_handoff_log("after_next_zeroEntries",
                                     _HANDOFF["last_saved"])
                    _HANDOFF["postzero_fired"] = True
            except Exception:
                pass
            petsc.assemble_matrix(A, self.jacobian, bcs=self.bcs)
            A.assemble()

            petsc.assemble_vector(L, self.residual)

            L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            L.scale(-1)
            # Compute b - J(u_D-u_(i-1))
            petsc.apply_lifting(
                L, [self.jacobian], [self.bcs], x0=[u.x.petsc_vec], alpha=1
            )
            # Set dx|_bc = u_{i-1}-u_D
            petsc.set_bc(L, self.bcs, u.x.petsc_vec, 1.0)
            L.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
            )

            # Get residual norm for logging
            residual_norm = L.norm(0)

            # Log iteration 0 (initial residual) on first loop
            if i == 0:
                self.diagnostics.log_iteration(
                    iteration=0,
                    residual_norm=residual_norm,
                )

            # Solve linear problem
            if self.pc_type == "element_block":
                new_A, new_rhs = self.pc.precondition(L)
                solver = PETSc.KSP().create(self.comm)
                solver.setType("gmres")
                solver.setTolerances(rtol=1e-8, atol=1e-9)
                solver.getPC().setType("mat")
                solver.setOperators(A, self.pc.mat)
                solver.solve(L, dx.x.petsc_vec)
            else:
                solver.solve(L, dx.x.petsc_vec)

            dx.x.scatter_forward()

            # Get linear solver diagnostics
            linear_iterations = solver.getIterationNumber()
            linear_convergence = solver.getConvergedReason()
            linear_residual = solver.getResidualNorm()

            if linear_convergence < 0:
                if self.comm.rank == 0:
                    print(f"  Linear solver diverged at timestep {timestep}, "
                          f"Newton iteration {i+1}: convergence code {linear_convergence}")

                # Fallback: try direct LU if iterative solver failed
                if self.pc_type != "lu":
                    if self.comm.rank == 0:
                        print(f"  Retrying Newton iteration {i+1} with direct LU solver...")
                    fallback_ksp = PETSc.KSP().create(self.comm)
                    fallback_ksp.setOperators(A)
                    fallback_ksp.setType(PETSc.KSP.Type.PREONLY)
                    fallback_ksp.getPC().setType(PETSc.PC.Type.LU)
                    fallback_ksp.setErrorIfNotConverged(False)
                    fallback_ksp.solve(L, dx.x.petsc_vec)
                    lu_reason = fallback_ksp.getConvergedReason()
                    fallback_ksp.destroy()

                    if lu_reason >= 0:
                        # LU succeeded — continue Newton iteration
                        if self.comm.rank == 0:
                            print(f"  LU fallback succeeded")
                        dx.x.scatter_forward()
                        u.x.array[:] += relaxation_parameter * dx.x.array[:]
                        i += 1
                        if i == 1:
                            self.dx_0_norm = dx.x.petsc_vec.norm(0)
                        if self.dx_0_norm > 1e-8:
                            dx.x.array[:] = np.array(dx.x.array[:] / self.dx_0_norm)
                        dx.x.petsc_vec.assemble()
                        correction_norm = dx.x.petsc_vec.norm(0)
                        self.diagnostics.log_iteration(
                            iteration=i, residual_norm=residual_norm,
                            correction_norm=correction_norm,
                            linear_iterations=0,
                        )
                        if correction_norm < self.atol:
                            converged = True
                            break
                        continue
                    else:
                        if self.comm.rank == 0:
                            print(f"  LU fallback also failed (reason={lu_reason})")

                # Rebuild KSP to recover from corrupted factorization state
                self._setup_ksp()
                break

            # Update u_{i+1} = u_i + alpha * dx_i
            u.x.array[:] += relaxation_parameter * dx.x.array[:]

            i += 1

            if i == 1:
                self.dx_0_norm = dx.x.petsc_vec.norm(0)

            # Normalize dx for convergence check (relative to first iteration)
            if self.dx_0_norm > 1e-8:
                dx.x.array[:] = np.array(dx.x.array[:] / self.dx_0_norm)
            dx.x.petsc_vec.assemble()

            # Compute norm of update
            correction_norm = dx.x.petsc_vec.norm(0)

            # Log this Newton iteration
            self.diagnostics.log_iteration(
                iteration=i,
                residual_norm=residual_norm,
                correction_norm=correction_norm,
                linear_iterations=linear_iterations,
                linear_convergence=linear_convergence,
                linear_residual=linear_residual,
                dx_0_norm=getattr(self, 'dx_0_norm', None),
            )

            # Check convergence
            if correction_norm < self.atol:
                converged = True
                break

            # Optionally reduce relaxation parameter if not converging
            if hasattr(self, "reduction_it"):
                if i and i % self.reduction_it == 0:
                    if self.verbose:
                        self.log("Still haven't converged. Reducing relax param")
                    relaxation_parameter /= 2

        # End diagnostics tracking for this timestep
        self.diagnostics.end_timestep(converged, i)

        # Expose per-timestep diagnostics for the forward-diag CSV reader
        # in cg_implicit.solve_timestep. Cheap public attributes.
        self.last_n_iters = int(i)
        self.last_converged = bool(converged)
        try:
            self.last_correction_norm = float(correction_norm) if i > 0 else float("nan")
        except Exception:
            self.last_correction_norm = float("nan")
        try:
            self.last_residual_norm = float(residual_norm)
        except Exception:
            self.last_residual_norm = float("nan")

        # Handle Newton solver failure
        if not converged:
            if self.comm.rank == 0:
                final_norm = correction_norm if i > 0 else float('nan')
                print(f"  WARNING: Newton solver did not converge at timestep {timestep} "
                      f"after {i} iterations (correction_norm={final_norm:.4e}, atol={self.atol:.1e})")
            # During optimization, raise exception so cost function returns infinity
            if getattr(self, 'raise_on_failure', False):
                raise RuntimeError(
                    f"Newton solver failed at timestep {timestep} after {i} iterations"
                )

        # Handle Jacobian return for 4D-Var
        if return_jacobian:
            # If we converged, the Jacobian A was assembled at u_i,
            # but u was updated to u_{i+1} = u_i + dx. For correct adjoint
            # computation, we need ∂R/∂u evaluated at the CONVERGED solution u_{i+1}.
            # Therefore, we reassemble the Jacobian one final time.
            #
            # CRITICAL FIX: For discrete adjoint (DTO), we return the UNMODIFIED
            # Jacobian (without BC rows set to identity). The BC-modified Jacobian
            # has identity rows at Dirichlet DOFs, which when transposed become
            # identity columns that block sensitivity propagation in the adjoint.
            #
            # The adjoint solver applies homogeneous Dirichlet BCs to the adjoint
            # solution (λ = 0 at BC DOFs) separately after solving J^T λ = rhs.
            # R1 reassembly trace (one-shot): localize at what exact stage the
            # stored Jacobian becomes the zero operator. See
            # docs/idealized_inlet_jacobian_reassembly_trace.md.
            _R1 = not getattr(CustomNewtonProblem, "_R1_FIRED", False)
            if _R1:
                CustomNewtonProblem._R1_FIRED = True

            if converged:
                if self.verbose:
                    self.log("Reassembling Jacobian at converged solution for adjoint (unmodified)")
                if _R1:
                    pre_zero_norm = A.norm(PETSc.NormType.NORM_FROBENIUS)
                    pre_nz = int(A.getInfo().get("nz_used", -1))
                A.zeroEntries()
                if _R1:
                    mid_norm = A.norm(PETSc.NormType.NORM_FROBENIUS)
                # Assemble WITHOUT bcs - this gives the true physics Jacobian
                # The BC-modified version was used for Newton iterations,
                # but the unmodified version is needed for correct adjoint gradients
                petsc.assemble_matrix(A, self.jacobian)  # No bcs parameter!
                A.assemble()
                if _R1:
                    post_assembly_norm = A.norm(PETSc.NormType.NORM_FROBENIUS)
                    post_nz = int(A.getInfo().get("nz_used", -1))

            # Return a copy to avoid issues with A being modified in subsequent timesteps
            J_final = A.copy()
            if _R1:
                copy_norm = J_final.norm(PETSc.NormType.NORM_FROBENIUS)
                copy_nz = int(J_final.getInfo().get("nz_used", -1))
                if self.comm.rank == 0:
                    print(f"[jac-reassembly] converged={converged} "
                          f"pre_zero={pre_zero_norm:.3e} "
                          f"mid={mid_norm:.3e} "
                          f"post={post_assembly_norm:.3e} "
                          f"copy={copy_norm:.3e} "
                          f"nz(pre/post/copy)={pre_nz}/{post_nz}/{copy_nz}",
                          flush=True)
            return None, J_final
        else:
            return None

    def assemble_A(self):
        """Assemble and return a copy of the Jacobian matrix at current state"""
        self.A.zeroEntries()
        petsc.assemble_matrix(self.A, self.jacobian, bcs=self.bcs)
        self.A.assemble()
        return self.A.copy()


class ElementBlockPreconditioner:

    def __init__(self, A, mesh):
        """Initialize the preconditioner from the"""

        dim = mesh.topology.dim
        num_cells = mesh.topology.index_map(dim).size_local
        print("local num cells", num_cells)
        block_size = A.size[0] // num_cells
        print("block size", block_size)
        self.block_size = block_size
        self.A = A
        mat = PETSc.Mat()
        mat.createAIJ(
            (A.size[0], A.size[0]),
            nnz=np.full(A.size[0], block_size, dtype=np.int32),
            comm=mesh.comm,
        )
        mat.setUp()
        mat.setBlockSize(block_size)
        self.mat = mat

    def precondition(self, rhs):
        """Apply the block preconditioner to A and the right hand side

        returns P^-1 * A, P^-1 * rhs
        """

        old_block_size = self.A.getBlockSize()
        self.A.setBlockSize(self.block_size)
        inv = self.A.invertBlockDiagonal()
        self.A.setBlockSize(old_block_size)
        start_ind, stop_ind = self.mat.owner_range
        block_inds = np.arange(
            start_ind // self.block_size, stop_ind // self.block_size + 1
        )
        block_inds = block_inds.astype(np.int32)
        self.mat.setValuesBlockedCSR(block_inds, block_inds[:-1], inv)
        self.mat.assemble()
        new_rhs = self.mat.createVecRight()
        self.mat.mult(rhs, new_rhs)
        return self.mat.matMult(self.A), new_rhs


class NewtonSolver:

    def __init__(
        self,
        obj1,
        # u_init = lambda x: np.ones(x.shape),
        solver_parameters={},
    ):
        """Solve the equation and save the result in u_sol"""

        prob = petsc.NonlinearProblem(obj1.F, obj1.u, bcs=obj1.problem.get_bcs())

        # the problem appears to be that the residual is humongous. . .
        res = fe.form(obj1.F)
        test_res = _cvf(res)
        petsc.assemble_vector(test_res, res)
        # print(test_res.getArray())
        print(f"Calling NewtonSolver with {obj1.problem.name} problem")
        self.solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, prob)
        print("Solver created")
        for k, v in solver_parameters.items():
            setattr(self.solver, k, v)
        self.solver.report = True
        self.solver.convergence_criterion = "incremental"
        self.solver.error_on_nonconvergence = False
        log.set_log_level(log.LogLevel.INFO)
        ksp = self.solver.krylov_solver

        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        viewer = PETSc.Viewer().createASCII("default_output.txt")
        ksp.view(viewer)

        opts[f"{option_prefix}ksp_type"] = "preonly"
        opts[f"{option_prefix}pc_type"] = "lu"  # "bjacobi"#"none"#"lu"#"gamg"

        opts[f"{option_prefix}pc_factor_solver_type"] = "mumps"

        ksp.setFromOptions()

        viewer = PETSc.Viewer().createASCII("linear_output.txt")
        ksp.view(viewer)
        viewer.destroy()

        # Use context manager to ensure file is properly closed
        with open("linear_output.txt", "r") as solver_output:
            for line in solver_output:
                print(line, end='')
        # print(self.u.vector.getArray())

    def solve(self, u, return_jacobian=False):
        """Solve the nonlinear problem

        Args:
            u: Solution function to update
            return_jacobian: If True, attempt to extract and return Jacobian (default: False)

        Returns:
            If return_jacobian=False: r (convergence reason)
            If return_jacobian=True: (r, J) where J is the Jacobian or None if unavailable

        Note: The built-in NewtonSolver doesn't easily expose the Jacobian.
              For 4D-Var applications, prefer using CustomNewtonProblem.
        """
        # print('before Newton', u.x.array[:])
        r = self.solver.solve(u)

        if return_jacobian:
            # The built-in NewtonSolver stores Jacobian in self.solver.A
            # We can try to extract it, but CustomNewtonProblem is preferred for 4D-Var
            try:
                J = self.solver.A.copy() if hasattr(self.solver, "A") else None
                return r, J
            except:
                print("Warning: Could not extract Jacobian from NewtonSolver.")
                print("For 4D-Var, use CustomNewtonProblem instead.")
                return r, None
        else:
            return r

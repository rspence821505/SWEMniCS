"""
Custom Newton solver for general nonlinear variational problems.

This was implemented because more control was desired over the Newton iteration
than provided by the built-in NonlinearProblem class.

MODIFICATIONS FOR 4D-VAR:
- CustomNewtonProblem.solve() now accepts return_jacobian flag
- Returns final Jacobian matrix for efficient adjoint computation
- Jacobian is reassembled at converged solution for correctness
- NewtonSolver.solve() also returns Jacobian when requested
"""

from dolfinx import fem as fe, nls, log, geometry, io, cpp
import dolfinx.fem.petsc as petsc
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from scipy.sparse import csr_matrix
import numpy.linalg as la
import sys
import time


def petsc_to_csr(A):
    indptr, indices, data = A.getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=A.size)


class CustomNewtonProblem:
    """An all-in-one class that solves a nonlinear problem. . ."""

    def __init__(self, obj1, solver_parameters={}):
        """initialize the problem

        F -- Ufl form
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

        self.A = petsc.create_matrix(self.jacobian)
        self.L = petsc.create_vector(self.residual)
        self.solver = PETSc.KSP().create(self.comm)

        self.solver.setTolerances(
            rtol=solver_parameters.get("ksp_rtol", 1e-8),
            atol=solver_parameters.get("ksp_atol", 1e-9),
            max_it=solver_parameters.get("ksp_max_it", 1000),
        )
        self.solver.setOperators(self.A)
        self.solver.setErrorIfNotConverged(
            solver_parameters.get("ksp_ErrorIfNotConverged", True)
        )

        if self.pc_type == "element_block":
            self.pc = ElementBlockPreconditioner(self.A, obj1.problem.mesh)
        else:
            self.pc = self.solver.getPC()
            self.pc.setType(self.pc_type)

    def log(self, *msg):
        if self.comm.rank == 0:
            print(*msg)

    def solve(self, u, max_it=5, return_jacobian=False):
        """Solve the nonlinear problem at u

        Args:
            u: Solution function to update in-place
            max_it: Maximum Newton iterations (default: 5)
            return_jacobian: If True, return final Jacobian for adjoint (default: False)

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

        dx = fe.Function(u._V)
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
            if self.verbose:
                self.log("Residual norm", L.norm(0))

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
            if self.verbose:
                self.log(
                    f"linear solver convergence {solver.getConvergedReason()}"
                    + f", iterations {solver.getIterationNumber()}, resid norm {solver.getResidualNorm()}"
                )
            if solver.getConvergedReason() == -9:
                sys.exit(1)

            # Update u_{i+1} = u_i + alpha * dx_i
            u.x.array[:] += relaxation_parameter * dx.x.array[:]

            i += 1

            if i == 1:
                self.dx_0_norm = dx.x.petsc_vec.norm(0)
                if self.verbose:
                    self.log("dx_0 norm,", self.dx_0_norm)

            # Normalize dx for convergence check (relative to first iteration)
            if self.dx_0_norm > 1e-8:
                dx.x.array[:] = np.array(dx.x.array[:] / self.dx_0_norm)
            dx.x.petsc_vec.assemble()

            # Compute norm of update
            correction_norm = dx.x.petsc_vec.norm(0)
            if self.verbose:
                self.log(f"Newton Iteration {i}: Correction norm {correction_norm}")

            # Check convergence
            if correction_norm < self.atol:
                converged = True
                if self.verbose:
                    self.log(f"Newton solver converged in {i} iterations")
                break

            # Optionally reduce relaxation parameter if not converging
            if hasattr(self, "reduction_it"):
                if i and i % self.reduction_it == 0:
                    if self.verbose:
                        self.log("Still haven't converged. Reducing relax param")
                    relaxation_parameter /= 2

        # Handle Jacobian return for 4D-Var
        if return_jacobian:
            # CRITICAL: If we converged, the Jacobian A was assembled at u_i,
            # but u was updated to u_{i+1} = u_i + dx. For correct adjoint
            # computation, we need ∂R/∂u evaluated at the CONVERGED solution u_{i+1}.
            # Therefore, we reassemble the Jacobian one final time.
            if converged:
                if self.verbose:
                    self.log("Reassembling Jacobian at converged solution for 4D-Var")
                A.zeroEntries()
                petsc.assemble_matrix(A, self.jacobian, bcs=self.bcs)
                A.assemble()

            # Return a copy to avoid issues with A being modified in subsequent timesteps
            J_final = A.copy()
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
        test_res = petsc.create_vector(res)
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

        solver_output = open("linear_output.txt", "r")
        for line in solver_output.readlines():
            print(line)
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

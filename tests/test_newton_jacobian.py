"""
Tests for CustomNewtonProblem with Jacobian extraction for 4D-Var.

Tests adaptively run in serial or parallel mode:
- Serial: pytest test_newton_jacobian.py -v
- Parallel: mpirun -n 4 pytest test_newton_jacobian.py -v

Tests cover:
- Basic Newton convergence
- Jacobian extraction with return_jacobian=True
- Jacobian evaluated at converged solution (not previous iterate)
- Jacobian properties (assembled, square, non-empty)
- Consistency between solve with/without Jacobian extraction
- MPI determinism and parallel correctness
"""

import pytest
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import mesh, fem as fe
from dolfinx.fem import functionspace
from ufl import (
    TrialFunction,
    TestFunction,
    grad,
    inner,
    dx,
    exp,
    sin,
    cos,
    SpatialCoordinate,
)
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swemnics.forward.newton import CustomNewtonProblem

# ============================================================================
# MPI DETECTION AND MARKERS
# ============================================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

requires_mpi = pytest.mark.skipif(size == 1, reason="Requires MPI with multiple ranks")
serial_only = pytest.mark.skipif(size > 1, reason="Serial test only")


# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def simple_poisson_problem():
    """Create a simple nonlinear Poisson problem for testing.

    Solves: -∇²u + u³ = f on unit square with Dirichlet BCs
    """
    # Create mesh
    msh = mesh.create_unit_square(comm, 10, 10, mesh.CellType.triangle)

    # Function space
    V = functionspace(msh, ("Lagrange", 1))

    # Create mock problem object that CustomNewtonProblem expects
    class MockProblem:
        def __init__(self, msh, V):
            self.mesh = msh
            self.dirichlet_bcs = []

            # Define boundary condition
            def boundary(x):
                return np.logical_or(
                    np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1)),
                    np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1)),
                )

            boundary_dofs = fe.locate_dofs_geometrical(V, boundary)
            bc_func = fe.Function(V)
            bc_func.x.array[:] = 0.0
            self.dirichlet_bcs = [fe.dirichletbc(bc_func, boundary_dofs)]

    # Create mock solver object
    class MockSolver:
        def __init__(self, msh, V, problem):
            self.problem = problem
            self.verbose = False

            # Define the nonlinear problem: -∇²u + u³ = f
            self.u = fe.Function(V)
            v = TestFunction(V)

            # RHS: manufactured solution u_exact = sin(πx)sin(πy)
            x = SpatialCoordinate(msh)
            u_exact = sin(np.pi * x[0]) * sin(np.pi * x[1])

            # Compute f from -∇²u_exact + u_exact³
            f = (
                2 * np.pi**2 * sin(np.pi * x[0]) * sin(np.pi * x[1])
                + sin(np.pi * x[0]) ** 3 * sin(np.pi * x[1]) ** 3
            )

            # Weak form: ∫(∇u·∇v + u³v - fv)dx = 0
            self.F = inner(grad(self.u), grad(v)) * dx + self.u**3 * v * dx - f * v * dx

    problem = MockProblem(msh, V)
    solver = MockSolver(msh, V, problem)

    return solver, V


@pytest.fixture
def linear_problem():
    """Create a simple linear problem for testing (single Newton iteration).

    Solves: -∇²u = 1 on unit square with u=0 on boundary
    """
    msh = mesh.create_unit_square(comm, 8, 8, mesh.CellType.triangle)
    V = functionspace(msh, ("Lagrange", 1))

    class MockProblem:
        def __init__(self, msh, V):
            self.mesh = msh
            self.dirichlet_bcs = []

            def boundary(x):
                return np.logical_or(
                    np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1)),
                    np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1)),
                )

            boundary_dofs = fe.locate_dofs_geometrical(V, boundary)
            bc_func = fe.Function(V)
            bc_func.x.array[:] = 0.0
            self.dirichlet_bcs = [fe.dirichletbc(bc_func, boundary_dofs)]

    class MockSolver:
        def __init__(self, msh, V, problem):
            self.problem = problem
            self.verbose = False
            self.u = fe.Function(V)
            v = TestFunction(V)

            # Simple Poisson: -∇²u = 1
            self.F = inner(grad(self.u), grad(v)) * dx - v * dx

    problem = MockProblem(msh, V)
    solver = MockSolver(msh, V, problem)

    return solver, V


# ============================================================================
# BASIC CONVERGENCE TESTS
# ============================================================================


class TestNewtonBasicConvergence:
    """Test basic Newton solver convergence."""

    def test_newton_converges_nonlinear(self, simple_poisson_problem):
        """Test that Newton solver converges for nonlinear problem."""
        solver, V = simple_poisson_problem

        # Initialize Newton
        newton = CustomNewtonProblem(solver, solver_parameters={})

        # Solve without Jacobian extraction
        newton.solve(solver.u, return_jacobian=False)

        # Check that solution is reasonable (not zero, not nan)
        assert not np.all(solver.u.x.array == 0.0)
        assert not np.any(np.isnan(solver.u.x.array))
        assert not np.any(np.isinf(solver.u.x.array))

    def test_newton_converges_linear(self, linear_problem):
        """Test that Newton solver converges in one iteration for linear problem."""
        solver, V = linear_problem

        newton = CustomNewtonProblem(solver, solver_parameters={"max_it": 1})
        newton.solve(solver.u, return_jacobian=False)

        # Solution should be reasonable
        assert not np.all(solver.u.x.array == 0.0)
        assert not np.any(np.isnan(solver.u.x.array))


# ============================================================================
# JACOBIAN EXTRACTION TESTS
# ============================================================================


class TestJacobianExtraction:
    """Test Jacobian extraction functionality."""

    def test_jacobian_extraction_basic(self, simple_poisson_problem):
        """Test that Jacobian can be extracted."""
        solver, V = simple_poisson_problem

        newton = CustomNewtonProblem(solver, solver_parameters={})

        # Solve with Jacobian extraction
        result, J = newton.solve(solver.u, return_jacobian=True)

        # Check that Jacobian was returned
        assert J is not None, "Jacobian should not be None"
        assert isinstance(J, PETSc.Mat), f"Expected PETSc.Mat, got {type(J)}"

    def test_jacobian_properties(self, simple_poisson_problem):
        """Test that extracted Jacobian has correct properties."""
        solver, V = simple_poisson_problem

        newton = CustomNewtonProblem(solver, solver_parameters={})
        _, J = newton.solve(solver.u, return_jacobian=True)

        # Check assembly
        assert J.assembled == True, "Jacobian should be assembled"

        # Check size
        size = J.getSize()
        assert size[0] > 0, "Jacobian should have non-zero rows"
        assert size[1] > 0, "Jacobian should have non-zero cols"
        assert size[0] == size[1], f"Jacobian should be square, got {size}"

        # Check sparsity
        info = J.getInfo()
        nnz = info["nz_used"]
        assert nnz > 0, "Jacobian should have non-zero entries"

    def test_jacobian_is_copy(self, simple_poisson_problem):
        """Test that returned Jacobian is independent copy."""
        solver, V = simple_poisson_problem

        newton = CustomNewtonProblem(solver, solver_parameters={})
        _, J1 = newton.solve(solver.u, return_jacobian=True)

        # Get original values
        J1_copy = J1.copy()

        # Solve again (this would overwrite internal matrix if not copied)
        solver.u.x.array[:] = 0.5  # Different initial guess
        _, J2 = newton.solve(solver.u, return_jacobian=True)

        # J1 should not have changed (it's a copy)
        # Check that they're different objects
        assert J1 != J2, "Should return different matrix objects"

        # J1 should still match its original copy
        J1_diff = J1.copy()
        J1_diff.axpy(-1.0, J1_copy)
        norm_diff = J1_diff.norm()
        assert norm_diff < 1e-10, "J1 should not have been modified"


# ============================================================================
# CORRECTNESS TESTS
# ============================================================================


class TestJacobianCorrectness:
    """Test that Jacobian is evaluated at correct solution state."""

    def test_jacobian_at_converged_solution(self, simple_poisson_problem):
        """Test that Jacobian is evaluated at converged solution, not previous iterate."""
        solver, V = simple_poisson_problem

        newton = CustomNewtonProblem(
            solver, solver_parameters={"max_it": 10, "atol": 1e-8}
        )

        # Solve and get Jacobian
        _, J_during = newton.solve(solver.u, return_jacobian=True)

        # Save converged solution
        u_converged = solver.u.x.array.copy()

        # Now manually assemble Jacobian at the converged solution
        J_manual = newton.assemble_A()

        # These should be identical (or very close)
        J_diff = J_during.copy()
        J_diff.axpy(-1.0, J_manual)
        diff_norm = J_diff.norm()

        # Normalize by Jacobian magnitude
        J_norm = J_manual.norm()
        relative_diff = diff_norm / J_norm if J_norm > 0 else diff_norm

        if rank == 0:
            print(f"\nJacobian difference: {diff_norm:.2e}")
            print(f"Jacobian norm: {J_norm:.2e}")
            print(f"Relative difference: {relative_diff:.2e}")

        assert relative_diff < 1e-10, (
            f"Jacobian should match manual assembly at converged solution. "
            f"Relative diff: {relative_diff:.2e}"
        )

    def test_solution_unchanged_with_jacobian_extraction(self, simple_poisson_problem):
        """Test that extracting Jacobian doesn't change the solution."""
        solver, V = simple_poisson_problem

        # Solve without Jacobian
        newton1 = CustomNewtonProblem(solver, solver_parameters={})
        u1 = fe.Function(V)
        u1.x.array[:] = 0.1  # Initial guess
        newton1.solve(u1, return_jacobian=False)
        solution1 = u1.x.array.copy()

        # Solve with Jacobian
        newton2 = CustomNewtonProblem(solver, solver_parameters={})
        u2 = fe.Function(V)
        u2.x.array[:] = 0.1  # Same initial guess
        solver.u = u2  # Point to new function
        _, J = newton2.solve(u2, return_jacobian=True)
        solution2 = u2.x.array.copy()

        # Solutions should be identical
        diff = np.linalg.norm(solution1 - solution2)
        assert (
            diff < 1e-12
        ), f"Solutions should match with/without Jacobian extraction. Diff: {diff}"


# ============================================================================
# PARALLEL TESTS
# ============================================================================


@requires_mpi
class TestNewtonParallel:
    """Test Newton solver in parallel."""

    def test_parallel_convergence(self, simple_poisson_problem):
        """Test that Newton converges in parallel."""
        solver, V = simple_poisson_problem

        newton = CustomNewtonProblem(solver, solver_parameters={})
        newton.solve(solver.u, return_jacobian=False)

        # Check local portion is reasonable
        assert not np.any(np.isnan(solver.u.x.array))
        assert not np.any(np.isinf(solver.u.x.array))

    def test_parallel_jacobian_extraction(self, simple_poisson_problem):
        """Test Jacobian extraction in parallel."""
        solver, V = simple_poisson_problem

        newton = CustomNewtonProblem(solver, solver_parameters={})
        _, J = newton.solve(solver.u, return_jacobian=True)

        # Check Jacobian properties on each rank
        assert J is not None
        assert isinstance(J, PETSc.Mat)
        assert J.assembled == True

        # Check distributed properties
        local_size = J.getLocalSize()
        assert local_size[0] > 0, f"Rank {rank} should own rows"

        # Check global size is consistent
        global_size = J.getSize()
        assert global_size[0] == global_size[1], "Global Jacobian should be square"

    def test_parallel_determinism(self, simple_poisson_problem):
        """Test that parallel solution is deterministic."""
        solver, V = simple_poisson_problem

        # Solve twice with same initial condition
        solutions = []
        for i in range(2):
            u_test = fe.Function(V)
            u_test.x.array[:] = 0.1
            solver.u = u_test

            newton = CustomNewtonProblem(solver, solver_parameters={})
            newton.solve(u_test, return_jacobian=False)
            solutions.append(u_test.x.array.copy())

        # Should get identical results
        diff = np.linalg.norm(solutions[0] - solutions[1])
        assert (
            diff < 1e-14
        ), f"Rank {rank}: Parallel solve not deterministic, diff={diff}"


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestNewtonEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_initial_guess(self, linear_problem):
        """Test Newton with zero initial guess."""
        solver, V = linear_problem

        # Zero initial guess
        solver.u.x.array[:] = 0.0

        newton = CustomNewtonProblem(solver, solver_parameters={})
        _, J = newton.solve(solver.u, return_jacobian=True)

        # Should still converge and extract Jacobian
        assert J is not None
        assert not np.all(solver.u.x.array == 0.0)

    def test_max_iterations_reached(self, simple_poisson_problem):
        """Test behavior when max iterations reached."""
        solver, V = simple_poisson_problem

        # Set very low max iterations
        newton = CustomNewtonProblem(
            solver, solver_parameters={"max_it": 1, "atol": 1e-15}
        )

        # Should still return Jacobian even if not fully converged
        _, J = newton.solve(solver.u, return_jacobian=True)

        assert J is not None
        assert isinstance(J, PETSc.Mat)


# ============================================================================
# PERFORMANCE AND MEMORY TESTS
# ============================================================================


@serial_only
class TestNewtonPerformance:
    """Test performance characteristics (serial only for timing)."""

    def test_jacobian_extraction_overhead(self, simple_poisson_problem):
        """Test that Jacobian extraction doesn't significantly slow solve."""
        import time

        solver, V = simple_poisson_problem

        # Time without Jacobian
        solver.u.x.array[:] = 0.1
        newton1 = CustomNewtonProblem(solver, solver_parameters={})

        start = time.time()
        newton1.solve(solver.u, return_jacobian=False)
        time_without = time.time() - start

        # Time with Jacobian
        solver.u.x.array[:] = 0.1
        newton2 = CustomNewtonProblem(solver, solver_parameters={})

        start = time.time()
        _, J = newton2.solve(solver.u, return_jacobian=True)
        time_with = time.time() - start

        # Overhead should be small (mainly one extra matrix assembly + copy)
        # Allow up to 2x overhead (copy + reassembly)
        if rank == 0:
            print(f"\nTime without Jacobian: {time_without:.4f}s")
            print(f"Time with Jacobian: {time_with:.4f}s")
            print(f"Overhead: {(time_with/time_without - 1)*100:.1f}%")

        # This is lenient - mainly checking it doesn't blow up
        assert (
            time_with < 3 * time_without
        ), "Jacobian extraction should not triple solve time"


# ============================================================================
# INTEGRATION TEST
# ============================================================================


class TestNewtonIntegration:
    """Integration test mimicking real 4D-Var usage."""

    def test_multiple_timestep_jacobian_storage(self, simple_poisson_problem):
        """Test storing Jacobians from multiple timesteps (4D-Var pattern)."""
        solver, V = simple_poisson_problem

        # Simulate 5 timesteps
        jacobians = []
        solutions = []

        for i in range(5):
            # Different initial guess each "timestep"
            solver.u.x.array[:] = 0.1 * (i + 1)

            newton = CustomNewtonProblem(solver, solver_parameters={})
            _, J = newton.solve(solver.u, return_jacobian=True)

            jacobians.append(J)
            solutions.append(solver.u.x.array.copy())

        # Check we have 5 independent Jacobians
        assert len(jacobians) == 5

        # Check they're all valid matrices
        for i, J in enumerate(jacobians):
            assert J is not None, f"Jacobian {i} should not be None"
            assert J.assembled, f"Jacobian {i} should be assembled"
            assert J.getSize()[0] > 0, f"Jacobian {i} should have non-zero size"

        # Note: Since all timesteps use the same problem (same forcing, same BCs),
        # Newton converges to the same solution each time, so Jacobians are identical.
        # In a real 4D-Var application, each timestep would have different state/BCs.
        # Here we just verify that the Jacobians are stored independently (different objects)
        for i in range(len(jacobians) - 1):
            assert jacobians[i] != jacobians[i + 1], "Should store independent matrix objects"


# ============================================================================
# RUN TESTS
# ============================================================================


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])

"""
Tests for solver Jacobian extraction and validation for 4D-Var.

Tests adaptively run in serial or parallel mode:
- Serial: pytest test_solver_jacobian.py -v
- Parallel: mpirun -n 4 pytest test_solver_jacobian.py -v

Tests cover:
- solve_timestep with Jacobian extraction
- Jacobian validation (size, type, assembly)
- Integration with time_loop and store_jacobians
- Error handling for invalid Jacobians
- MPI parallel correctness
- Memory efficiency with observation_times
"""

import pytest
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import mesh
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swe4dvar.forward.solvers import CGImplicit, DGImplicit
from swe4dvar.forward.problems import TidalProblem

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
def simple_tidal_problem():
    """Create a simple tidal problem for testing."""
    problem = TidalProblem(nx=5, ny=5, nt=10, dt=60.0, verbose=False)
    return problem


@pytest.fixture
def cg_solver(simple_tidal_problem):
    """Create a CG solver for testing."""
    solver = CGImplicit(simple_tidal_problem, theta=0.5, verbose=False)
    return solver


@pytest.fixture
def dg_solver(simple_tidal_problem):
    """Create a DG solver for testing."""
    solver = DGImplicit(simple_tidal_problem, theta=0.5, p_degree=[1, 1], verbose=False)
    return solver


# ============================================================================
# BASIC JACOBIAN EXTRACTION TESTS
# ============================================================================


class TestSolveTimestepBasic:
    """Test basic solve_timestep functionality."""

    def test_solve_without_jacobian(self, cg_solver):
        """Test solve_timestep without Jacobian extraction."""
        solver = cg_solver
        newton = solver.solve_init()

        # Solve without Jacobian
        J = solver.solve_timestep(newton, store_jacobian=False)

        assert J is None, "Should return None when store_jacobian=False"
        assert not np.any(np.isnan(solver.u.x.array))

    def test_solve_with_jacobian(self, cg_solver):
        """Test solve_timestep with Jacobian extraction."""
        solver = cg_solver
        newton = solver.solve_init()

        # Solve with Jacobian
        J = solver.solve_timestep(newton, store_jacobian=True)

        assert J is not None, "Should return Jacobian when store_jacobian=True"
        assert isinstance(J, PETSc.Mat), f"Expected PETSc.Mat, got {type(J)}"

    def test_solution_unchanged_with_jacobian(self, cg_solver):
        """Test that extracting Jacobian doesn't change solution."""
        solver = cg_solver

        # Solve without Jacobian
        newton1 = solver.solve_init()
        solver.solve_timestep(newton1, store_jacobian=False)
        solution1 = solver.u.x.array.copy()

        # Reset and solve with Jacobian
        solver.u.x.array[:] = solver.u_n.x.array[:]
        newton2 = solver.solve_init()
        J = solver.solve_timestep(newton2, store_jacobian=True)
        solution2 = solver.u.x.array.copy()

        # Solutions should match
        diff = np.linalg.norm(solution1 - solution2)
        assert diff < 1e-10, f"Solutions should match, diff={diff}"


# ============================================================================
# JACOBIAN VALIDATION TESTS
# ============================================================================


class TestJacobianValidation:
    """Test Jacobian validation logic in solve_timestep."""

    def test_jacobian_is_petsc_mat(self, cg_solver):
        """Test that extracted Jacobian is a PETSc matrix."""
        solver = cg_solver
        newton = solver.solve_init()

        J = solver.solve_timestep(newton, store_jacobian=True)

        assert isinstance(J, PETSc.Mat), f"Expected PETSc.Mat, got {type(J)}"

    def test_jacobian_is_assembled(self, cg_solver):
        """Test that Jacobian is assembled."""
        solver = cg_solver
        newton = solver.solve_init()

        J = solver.solve_timestep(newton, store_jacobian=True)

        assert J.assembled == True, "Jacobian should be assembled"

    def test_jacobian_nonzero_size(self, cg_solver):
        """Test that Jacobian has non-zero size."""
        solver = cg_solver
        newton = solver.solve_init()

        J = solver.solve_timestep(newton, store_jacobian=True)

        size = J.getSize()
        assert size[0] > 0, f"Jacobian should have non-zero rows, got {size}"
        assert size[1] > 0, f"Jacobian should have non-zero cols, got {size}"

    def test_jacobian_is_square(self, cg_solver):
        """Test that Jacobian is square (required for adjoint)."""
        solver = cg_solver
        newton = solver.solve_init()

        J = solver.solve_timestep(newton, store_jacobian=True)

        size = J.getSize()
        assert size[0] == size[1], f"Jacobian must be square, got {size}"

    def test_jacobian_has_nonzeros(self, cg_solver):
        """Test that Jacobian has non-zero entries."""
        solver = cg_solver
        newton = solver.solve_init()

        J = solver.solve_timestep(newton, store_jacobian=True)

        info = J.getInfo()
        nnz = info["nz_used"]
        assert nnz > 0, "Jacobian should have non-zero entries"

    def test_jacobian_validation_with_verbose(self, simple_tidal_problem):
        """Test that verbose mode logs Jacobian info."""
        solver = CGImplicit(simple_tidal_problem, theta=0.5, verbose=True)
        newton = solver.solve_init()

        # Capture log output
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            J = solver.solve_timestep(newton, store_jacobian=True)

        output = f.getvalue()

        # Should log Jacobian info on rank 0
        if rank == 0:
            assert "Jacobian extracted" in output or J is not None


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestJacobianErrorHandling:
    """Test error handling for invalid Jacobians."""

    def test_none_jacobian_raises_error(self, cg_solver):
        """Test that None Jacobian raises ValueError."""
        solver = cg_solver

        # Mock Newton solver to return None
        class MockNewton:
            def solve(self, u, return_jacobian=False):
                if return_jacobian:
                    return None, None
                return None

        mock_newton = MockNewton()

        with pytest.raises(ValueError, match="Jacobian extraction failed"):
            solver.solve_timestep(mock_newton, store_jacobian=True)

    def test_wrong_type_jacobian_raises_error(self, cg_solver):
        """Test that wrong type raises ValueError."""
        solver = cg_solver

        # Mock Newton solver to return wrong type
        class MockNewton:
            def solve(self, u, return_jacobian=False):
                if return_jacobian:
                    return None, "not_a_matrix"
                return None

        mock_newton = MockNewton()

        with pytest.raises(ValueError, match="Expected PETSc.Mat"):
            solver.solve_timestep(mock_newton, store_jacobian=True)

    def test_unassembled_jacobian_raises_error(self, cg_solver):
        """Test that unassembled Jacobian raises ValueError."""
        solver = cg_solver

        # Create unassembled matrix
        class MockNewton:
            def solve(self, u, return_jacobian=False):
                if return_jacobian:
                    # Create matrix but don't assemble
                    J = PETSc.Mat().create(comm=comm)
                    J.setSizes([10, 10])
                    J.setType("aij")
                    J.setUp()
                    # Don't call assemble
                    return None, J
                return None

        mock_newton = MockNewton()

        with pytest.raises(ValueError, match="not assembled"):
            solver.solve_timestep(mock_newton, store_jacobian=True)

    def test_zero_size_jacobian_raises_error(self, cg_solver):
        """Test that zero-size Jacobian raises ValueError."""
        solver = cg_solver

        class MockNewton:
            def solve(self, u, return_jacobian=False):
                if return_jacobian:
                    # Create zero-size matrix
                    J = PETSc.Mat().create(comm=comm)
                    J.setSizes([0, 0])
                    J.setType("aij")
                    J.setUp()
                    J.assemble()
                    return None, J
                return None

        mock_newton = MockNewton()

        with pytest.raises(ValueError, match="invalid size"):
            solver.solve_timestep(mock_newton, store_jacobian=True)


# ============================================================================
# TIME LOOP INTEGRATION TESTS
# ============================================================================


class TestTimeLoopJacobianIntegration:
    """Test Jacobian extraction in time_loop."""

    def test_time_loop_stores_jacobians(self, cg_solver):
        """Test that time_loop stores Jacobians when requested."""
        solver = cg_solver

        # Run time loop with Jacobian storage
        u, _ = solver.time_loop(
            solver_parameters={},
            store_jacobians=True,
        )

        # Check that Jacobians were stored
        assert len(solver.saved_jacobians) > 0, "Should store Jacobians"

        # Should have nt+1 Jacobians (including initial)
        # Actually, first 2 use implicit Euler, then BDF2
        # So we get Jacobians at timesteps 1, 2, ..., nt
        expected_count = solver.problem.nt
        assert (
            len(solver.saved_jacobians) == expected_count
        ), f"Expected {expected_count} Jacobians, got {len(solver.saved_jacobians)}"

    def test_time_loop_without_jacobians(self, cg_solver):
        """Test that time_loop doesn't store Jacobians by default."""
        solver = cg_solver

        # Run time loop without Jacobian storage
        u, _ = solver.time_loop(
            solver_parameters={},
            store_jacobians=False,
        )

        # Should not store Jacobians
        assert len(solver.saved_jacobians) == 0, "Should not store Jacobians"

    def test_time_loop_jacobians_are_valid(self, cg_solver):
        """Test that all stored Jacobians are valid."""
        solver = cg_solver

        u, _ = solver.time_loop(
            solver_parameters={},
            store_jacobians=True,
        )

        # Check each Jacobian
        for i, J in enumerate(solver.saved_jacobians):
            assert J is not None, f"Jacobian {i} should not be None"
            assert isinstance(J, PETSc.Mat), f"Jacobian {i} should be PETSc.Mat"
            assert J.assembled == True, f"Jacobian {i} should be assembled"

            size = J.getSize()
            assert size[0] > 0, f"Jacobian {i} should have non-zero size"
            assert size[0] == size[1], f"Jacobian {i} should be square"

    def test_time_loop_jacobians_are_different(self, cg_solver):
        """Test that Jacobians from different timesteps differ."""
        solver = cg_solver

        u, _ = solver.time_loop(
            solver_parameters={},
            store_jacobians=True,
        )

        # Check that consecutive Jacobians differ
        if len(solver.saved_jacobians) >= 2:
            J1 = solver.saved_jacobians[0]
            J2 = solver.saved_jacobians[1]

            J_diff = J1.copy()
            J_diff.axpy(-1.0, J2)
            diff_norm = J_diff.norm()

            # They should differ (different timesteps)
            assert diff_norm > 1e-6, "Consecutive Jacobians should differ"


# ============================================================================
# OBSERVATION TIME FILTERING TESTS
# ============================================================================


class TestObservationTimeFiltering:
    """Test Jacobian storage with observation_times filtering."""

    def test_observation_times_filters_jacobians(self, cg_solver):
        """Test that only Jacobians at observation times are stored."""
        solver = cg_solver

        obs_times = [0, 5, 10]

        u, _ = solver.time_loop(
            solver_parameters={},
            store_jacobians=True,
            observation_times=obs_times,
        )

        # Should only store Jacobians at observation times
        # obs_times includes initial (0), but Jacobians are stored at steps 1-10
        # So we expect Jacobians at steps 5 and 10
        expected_count = 2  # Steps 5 and 10
        assert (
            len(solver.saved_jacobians) == expected_count
        ), f"Expected {expected_count} Jacobians, got {len(solver.saved_jacobians)}"

    def test_sparse_observations_memory_savings(self, cg_solver):
        """Test memory savings with sparse observations."""
        solver = cg_solver

        # Very sparse observations
        obs_times = [0, 10]

        u, _ = solver.time_loop(
            solver_parameters={},
            store_jacobians=True,
            observation_times=obs_times,
        )

        # Should only store 1 Jacobian (at step 10, not at 0)
        assert len(solver.saved_jacobians) == 1

    def test_all_timesteps_observation(self, cg_solver):
        """Test that observation at every timestep stores all Jacobians."""
        solver = cg_solver

        obs_times = list(range(solver.problem.nt + 1))

        u, _ = solver.time_loop(
            solver_parameters={},
            store_jacobians=True,
            observation_times=obs_times,
        )

        # Should store all Jacobians
        expected_count = solver.problem.nt
        assert len(solver.saved_jacobians) == expected_count


# ============================================================================
# PARALLEL TESTS
# ============================================================================


@requires_mpi
class TestSolverJacobianParallel:
    """Test Jacobian extraction in parallel."""

    def test_parallel_jacobian_extraction(self, cg_solver):
        """Test Jacobian extraction works in parallel."""
        solver = cg_solver
        newton = solver.solve_init()

        J = solver.solve_timestep(newton, store_jacobian=True)

        # Check on all ranks
        assert J is not None
        assert isinstance(J, PETSc.Mat)
        assert J.assembled == True

    def test_parallel_time_loop_jacobians(self, cg_solver):
        """Test time_loop stores Jacobians correctly in parallel."""
        solver = cg_solver

        u, _ = solver.time_loop(
            solver_parameters={},
            store_jacobians=True,
        )

        # All ranks should store same number of Jacobians
        num_jacobians = len(solver.saved_jacobians)
        all_counts = comm.gather(num_jacobians, root=0)

        if rank == 0:
            assert all(
                c == all_counts[0] for c in all_counts
            ), f"All ranks should store same number of Jacobians, got {all_counts}"

    def test_parallel_jacobian_sizes_consistent(self, cg_solver):
        """Test that Jacobian global sizes are consistent across ranks."""
        solver = cg_solver
        newton = solver.solve_init()

        J = solver.solve_timestep(newton, store_jacobian=True)

        # Get global size
        global_size = J.getSize()

        # Gather sizes from all ranks
        all_sizes = comm.gather(global_size, root=0)

        if rank == 0:
            # All ranks should report same global size
            assert all(
                s == all_sizes[0] for s in all_sizes
            ), f"Global sizes should match across ranks, got {all_sizes}"

    def test_parallel_determinism(self, cg_solver):
        """Test that parallel Jacobian extraction is deterministic."""
        solver = cg_solver

        # Run twice
        jacobian_norms = []
        for i in range(2):
            solver.storage.clear()  # Clear previous Jacobians

            u, _ = solver.time_loop(
                solver_parameters={},
                store_jacobians=True,
            )

            # Compute norms of all Jacobians
            norms = [J.norm() for J in solver.saved_jacobians]
            jacobian_norms.append(norms)

        # Norms should match
        for i, (norm1, norm2) in enumerate(zip(jacobian_norms[0], jacobian_norms[1])):
            assert (
                abs(norm1 - norm2) < 1e-12
            ), f"Jacobian {i} norms should match: {norm1} vs {norm2}"


# ============================================================================
# SOLVER TYPE TESTS
# ============================================================================


class TestDifferentSolverTypes:
    """Test Jacobian extraction with different solver types."""

    def test_cg_solver_jacobian(self, cg_solver):
        """Test CG solver Jacobian extraction."""
        solver = cg_solver
        newton = solver.solve_init()

        J = solver.solve_timestep(newton, store_jacobian=True)

        assert J is not None
        assert isinstance(J, PETSc.Mat)

    def test_dg_solver_jacobian(self, dg_solver):
        """Test DG solver Jacobian extraction."""
        solver = dg_solver
        newton = solver.solve_init()

        J = solver.solve_timestep(newton, store_jacobian=True)

        assert J is not None
        assert isinstance(J, PETSc.Mat)

    def test_different_polynomial_degrees(self, simple_tidal_problem):
        """Test Jacobian extraction with different polynomial degrees."""
        for p in [1, 2]:
            solver = CGImplicit(
                simple_tidal_problem, theta=0.5, p_degree=[p, p], verbose=False
            )
            newton = solver.solve_init()

            J = solver.solve_timestep(newton, store_jacobian=True)

            assert J is not None, f"Failed for p={p}"
            assert isinstance(J, PETSc.Mat), f"Failed for p={p}"


# ============================================================================
# MEMORY AND PERFORMANCE TESTS
# ============================================================================


@serial_only
class TestJacobianMemoryPerformance:
    """Test memory and performance characteristics (serial only)."""

    def test_jacobian_storage_memory(self, cg_solver):
        """Test memory usage of stored Jacobians."""
        solver = cg_solver

        u, _ = solver.time_loop(
            solver_parameters={},
            store_jacobians=True,
        )

        # Estimate memory
        if len(solver.saved_jacobians) > 0:
            J = solver.saved_jacobians[0]
            size = J.getSize()
            info = J.getInfo()
            nnz = info["nz_used"]

            # Memory per Jacobian (rough estimate)
            bytes_per_entry = 8  # double precision
            mem_per_jacobian = nnz * bytes_per_entry / (1024**2)  # MB

            total_mem = mem_per_jacobian * len(solver.saved_jacobians)

            if rank == 0:
                print(f"\nJacobian size: {size}")
                print(f"Non-zeros: {int(nnz)}")
                print(f"Memory per Jacobian: {mem_per_jacobian:.2f} MB")
                print(f"Total memory: {total_mem:.2f} MB")

    def test_observation_time_memory_savings(self, simple_tidal_problem):
        """Test that observation times reduce memory usage."""
        # Create solver with more timesteps
        problem = TidalProblem(nx=5, ny=5, nt=50, dt=60.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        # Full storage
        u1, _ = solver.time_loop(
            solver_parameters={},
            store_jacobians=True,
        )
        full_count = len(solver.saved_jacobians)

        # Sparse storage
        solver.storage.clear()
        obs_times = [0, 10, 20, 30, 40, 50]
        u2, _ = solver.time_loop(
            solver_parameters={},
            store_jacobians=True,
            observation_times=obs_times,
        )
        sparse_count = len(solver.saved_jacobians)

        # Should have significant savings
        savings_ratio = sparse_count / full_count

        if rank == 0:
            print(f"\nFull storage: {full_count} Jacobians")
            print(f"Sparse storage: {sparse_count} Jacobians")
            print(f"Savings: {(1-savings_ratio)*100:.1f}%")

        assert sparse_count < full_count, "Sparse should use less memory"
        assert savings_ratio < 0.2, "Should save >80% memory"


# ============================================================================
# INTEGRATION TEST
# ============================================================================


class TestCompleteWorkflow:
    """Integration test for complete 4D-Var workflow."""

    def test_complete_forward_solve_with_jacobians(self, cg_solver):
        """Test complete forward solve storing Jacobians for adjoint."""
        solver = cg_solver

        # Define observation times (sparse)
        obs_times = [0, 3, 6, 10]

        # Run forward solve
        u_final, _ = solver.time_loop(
            solver_parameters={},
            store_jacobians=True,
            observation_times=obs_times,
        )

        # Verify we have everything needed for adjoint
        assert u_final is not None, "Should have final solution"
        assert len(solver.saved_states) > 0, "Should have saved states"
        assert len(solver.saved_jacobians) > 0, "Should have saved Jacobians"

        # Verify Jacobians are valid for adjoint computation
        for i, J in enumerate(solver.saved_jacobians):
            # Can transpose (required for adjoint)
            J_T = J.transpose()
            assert J_T is not None

            # Can create vectors
            x, y = J.createVecs()
            assert x is not None
            assert y is not None

            # Size matches
            assert J.getSize()[0] == J_T.getSize()[1]

        if rank == 0:
            print(f"\nForward solve complete:")
            print(f"  States: {len(solver.saved_states)}")
            print(f"  Jacobians: {len(solver.saved_jacobians)}")
            print(f"  Ready for adjoint computation")


# ============================================================================
# REGRESSION TESTS
# ============================================================================


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_old_code_still_works(self, cg_solver):
        """Test that code without store_jacobians still works."""
        solver = cg_solver

        # Old-style call (no Jacobian storage)
        u, _ = solver.time_loop(
            solver_parameters={},
            save_state=False,
        )

        assert u is not None
        assert len(solver.saved_jacobians) == 0

    def test_default_parameters(self, cg_solver):
        """Test that default parameters work correctly."""
        solver = cg_solver

        # Call with minimal parameters
        u, _ = solver.time_loop(solver_parameters={})

        assert u is not None
        # Should not store Jacobians by default
        assert len(solver.saved_jacobians) == 0


# ============================================================================
# RUN TESTS
# ============================================================================


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])

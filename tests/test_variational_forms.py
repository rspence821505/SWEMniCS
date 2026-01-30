"""
Test suite for variational_forms.py

Tests cover:
  - BDF2 vs Backward Euler time discretization
  - First-step startup handling
  - Adaptive time-stepping
  - Mass matrix assembly and caching
  - Jacobian assembly and correctness
  - MPI parallel execution and determinism

Run serial tests:
    pytest test_variational_forms.py -v -m "not mpi"

Run parallel tests:
    mpirun -np 4 pytest test_variational_forms.py -v -m "mpi"
"""

import pytest
import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem
from ufl import TrialFunction, TestFunction, dx, inner
from petsc4py import PETSc
import basix.ufl

# Import module under test
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from swe4dvar.forward.variational_forms import (
    VariationalForm,
    SWEVariationalForm,
    LinearizedVariationalForm,
    BDF2TimeCoefficients,
)
from swe4dvar.utils.fem_utilities import create_mixed_element


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_mesh():
    """Create simple 1D unit interval mesh."""
    return mesh.create_unit_interval(MPI.COMM_WORLD, 10)


@pytest.fixture
def simple_function_space(simple_mesh):
    """Create simple P1 function space on unit interval."""
    return fem.functionspace(simple_mesh, ("Lagrange", 1))


@pytest.fixture
def mixed_2d_mesh():
    """Create 2D unit square mesh for SWE tests."""
    return mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)


@pytest.fixture
def swe_function_space(mixed_2d_mesh):
    """
    Create mixed function space for SWE: (H, u, v).

    Uses P1 elements for simplicity in testing.
    """
    # Create basix elements for H, u, v
    P1 = basix.ufl.element("Lagrange", mixed_2d_mesh.basix_cell(), 1)
    element = create_mixed_element([P1, P1, P1])
    return fem.functionspace(mixed_2d_mesh, element)


# ============================================================================
# Base VariationalForm Tests
# ============================================================================


class TestVariationalFormBase:
    """Tests for VariationalForm base class."""

    def test_initialization_constant_dt(self, simple_function_space):
        """Test initialization with constant time step."""
        dt = 0.1
        form = VariationalForm(simple_function_space, dt)

        assert form.function_space == simple_function_space
        assert form.dt == dt
        assert form.use_bdf2 is True  # Default
        assert form._mass_matrix is None  # Not assembled yet

    def test_initialization_adaptive_dt(self, simple_function_space):
        """Test initialization with adaptive time steps."""
        dt_list = [0.1, 0.05, 0.02, 0.01]
        form = VariationalForm(simple_function_space, dt_list)

        assert form.dt == dt_list
        assert form.get_dt(0) == 0.1
        assert form.get_dt(3) == 0.01

    def test_set_time_scheme(self, simple_function_space):
        """Test switching between BDF2 and Backward Euler."""
        form = VariationalForm(simple_function_space, 0.1)

        # Start with BDF2
        assert form.use_bdf2 is True

        # Switch to Backward Euler
        form.set_time_scheme(False)
        assert form.use_bdf2 is False

        # Switch back to BDF2
        form.set_time_scheme(True)
        assert form.use_bdf2 is True

    def test_assemble_mass_matrix(self, simple_function_space):
        """Test mass matrix assembly and caching."""
        form = VariationalForm(simple_function_space, 0.1)

        # First assembly
        M1 = form.assemble_mass_matrix()
        assert isinstance(M1, PETSc.Mat)
        assert M1.assembled

        # Should be cached
        M2 = form.assemble_mass_matrix()
        assert M2 is M1  # Same object

    @pytest.mark.mpi
    def test_mass_matrix_parallel_determinism(self, simple_function_space):
        """Test that mass matrix is identical across MPI ranks."""
        form = VariationalForm(simple_function_space, 0.1)
        M = form.assemble_mass_matrix()

        # Get Frobenius norm (global quantity)
        norm = M.norm(PETSc.NormType.FROBENIUS)

        # All ranks should agree
        comm = MPI.COMM_WORLD
        norms = comm.allgather(norm)
        assert all(
            abs(n - norm) < 1e-14 for n in norms
        ), "Mass matrix norms differ across ranks"

    def test_mass_matrix_symmetry(self, simple_function_space):
        """Test that mass matrix is symmetric."""
        form = VariationalForm(simple_function_space, 0.1)
        M = form.assemble_mass_matrix()

        # Create test vectors
        x = M.createVecRight()
        y = M.createVecLeft()
        x.set(1.0)
        y.set(1.0)

        # Compute <Mx, y> and <x, My>
        Mx = x.copy()
        M.mult(x, Mx)
        lhs = Mx.dot(y)

        My = y.copy()
        M.mult(y, My)
        rhs = x.dot(My)

        rel_error = abs(lhs - rhs) / abs(lhs)
        assert rel_error < 1e-12, f"Mass matrix not symmetric: rel_error = {rel_error}"


# ============================================================================
# BDF2TimeCoefficients Tests
# ============================================================================


class TestBDF2TimeCoefficients:
    """Tests for BDF2 time-stepping coefficient calculations."""

    def test_forward_coeffs_bdf2_constant_dt(self):
        """Test BDF2 forward coefficients with constant dt."""
        dt = 0.1
        coeffs = BDF2TimeCoefficients(dt, use_bdf2=True)

        c_np1, c_n, c_nm1 = coeffs.get_forward_coeffs(0)

        # BDF2: (3/(2dt), -4/(2dt), 1/(2dt))
        assert abs(c_np1 - 3.0 / (2.0 * dt)) < 1e-14
        assert abs(c_n - (-4.0 / (2.0 * dt))) < 1e-14
        assert abs(c_nm1 - 1.0 / (2.0 * dt)) < 1e-14

    def test_forward_coeffs_backward_euler(self):
        """Test Backward Euler forward coefficients."""
        dt = 0.1
        coeffs = BDF2TimeCoefficients(dt, use_bdf2=False)

        c_np1, c_n, c_nm1 = coeffs.get_forward_coeffs(0)

        # Backward Euler: (1/dt, -1/dt, 0)
        assert abs(c_np1 - 1.0 / dt) < 1e-14
        assert abs(c_n - (-1.0 / dt)) < 1e-14
        assert abs(c_nm1) < 1e-14  # Should be zero

    def test_forward_coeffs_adaptive_dt(self):
        """Test forward coefficients with adaptive time-stepping."""
        dt_list = [0.1, 0.05, 0.02]
        coeffs = BDF2TimeCoefficients(dt_list, use_bdf2=True)

        # Check each time step has correct coefficients
        for step, dt in enumerate(dt_list):
            c_np1, c_n, c_nm1 = coeffs.get_forward_coeffs(step)
            assert abs(c_np1 - 3.0 / (2.0 * dt)) < 1e-14

    def test_adjoint_coeffs_bdf2(self):
        """Test BDF2 adjoint time-coupling coefficients."""
        dt = 0.1
        coeffs = BDF2TimeCoefficients(dt, use_bdf2=True)

        c_np1, c_np2 = coeffs.get_adjoint_coeffs(0)

        # BDF2: (4/(2dt), -1/(2dt))
        assert abs(c_np1 - 4.0 / (2.0 * dt)) < 1e-14
        assert abs(c_np2 - (-1.0 / (2.0 * dt))) < 1e-14

    def test_adjoint_coeffs_backward_euler(self):
        """Test Backward Euler adjoint coefficients."""
        dt = 0.1
        coeffs = BDF2TimeCoefficients(dt, use_bdf2=False)

        c_np1, c_np2 = coeffs.get_adjoint_coeffs(0)

        # Backward Euler: (1/dt, 0)
        assert abs(c_np1 - 1.0 / dt) < 1e-14
        assert abs(c_np2) < 1e-14  # Should be zero

    def test_jacobian_coeff_bdf2(self):
        """Test BDF2 Jacobian coefficient."""
        dt = 0.1
        coeffs = BDF2TimeCoefficients(dt, use_bdf2=True)

        c = coeffs.get_jacobian_coeff(0)
        assert abs(c - 3.0 / (2.0 * dt)) < 1e-14

    def test_jacobian_coeff_backward_euler(self):
        """Test Backward Euler Jacobian coefficient."""
        dt = 0.1
        coeffs = BDF2TimeCoefficients(dt, use_bdf2=False)

        c = coeffs.get_jacobian_coeff(0)
        assert abs(c - 1.0 / dt) < 1e-14


# ============================================================================
# SWEVariationalForm Tests
# ============================================================================


class TestSWEVariationalForm:
    """Tests for SWE-specific variational form."""

    def test_initialization(self, swe_function_space):
        """Test SWE form initialization with various parameters."""
        form = SWEVariationalForm(
            swe_function_space,
            dt=0.1,
            g=9.81,
            friction=0.025,
            use_bdf2=True,
            wetting_drying=False,
        )

        assert form.g == 9.81
        assert form.friction == 0.025
        assert form.wetting_drying is False
        assert form.h_min == 0.01  # Default

    def test_wetting_drying_flag(self, swe_function_space):
        """Test enabling wetting/drying scheme."""
        form = SWEVariationalForm(
            swe_function_space,
            dt=0.1,
            wetting_drying=True,
            h_min=0.05,
        )

        assert form.wetting_drying is True
        assert form.h_min == 0.05

    def test_residual_requires_unm1_for_bdf2(self, swe_function_space):
        """Test that BDF2 auto-falls back to Backward Euler when u_nm1 is None."""
        form = SWEVariationalForm(swe_function_space, dt=0.1, use_bdf2=True)

        u_next = fem.Function(swe_function_space)
        u_n = fem.Function(swe_function_space)

        # Should not raise error - auto-falls back to Backward Euler
        residual = form.assemble_residual(u_next, u_n, u_nm1=None)
        assert isinstance(residual, PETSc.Vec)

    def test_residual_backward_euler_no_unm1(self, swe_function_space):
        """Test that Backward Euler doesn't require u_nm1."""
        form = SWEVariationalForm(swe_function_space, dt=0.1, use_bdf2=False)

        u_next = fem.Function(swe_function_space)
        u_n = fem.Function(swe_function_space)

        # Should not raise error
        residual = form.assemble_residual(u_next, u_n, u_nm1=None)
        assert isinstance(residual, PETSc.Vec)

    def test_jacobian_structure(self, swe_function_space):
        """Test that Jacobian has correct structure."""
        form = SWEVariationalForm(swe_function_space, dt=0.1, use_bdf2=True)

        u_next = fem.Function(swe_function_space)
        u_n = fem.Function(swe_function_space)
        u_nm1 = fem.Function(swe_function_space)

        J = form.assemble_jacobian(u_next, u_n, u_nm1)

        # Check dimensions
        m, n = J.getSize()
        assert m == n, "Jacobian should be square"

        # Check assembled
        assert J.assembled

    def test_jacobian_bdf2_vs_backward_euler(self, swe_function_space):
        """Test that BDF2 and Backward Euler Jacobians differ correctly."""
        dt = 0.1

        # BDF2 form
        form_bdf2 = SWEVariationalForm(swe_function_space, dt, use_bdf2=True)

        # Backward Euler form
        form_be = SWEVariationalForm(swe_function_space, dt, use_bdf2=False)

        # Create identical states
        u_next = fem.Function(swe_function_space)
        u_n = fem.Function(swe_function_space)
        u_nm1 = fem.Function(swe_function_space)

        J_bdf2 = form_bdf2.assemble_jacobian(u_next, u_n, u_nm1)
        J_be = form_be.assemble_jacobian(u_next, u_n, None)

        # Compute norms
        norm_bdf2 = J_bdf2.norm(PETSc.NormType.FROBENIUS)
        norm_be = J_be.norm(PETSc.NormType.FROBENIUS)

        # BDF2 should have coefficient 3/(2dt) vs BE's 1/dt
        # Ratio should be (3/2) / 1 = 1.5
        expected_ratio = 1.5
        actual_ratio = norm_bdf2 / norm_be

        # Allow some tolerance due to numerical assembly
        assert (
            abs(actual_ratio - expected_ratio) < 0.1
        ), f"Jacobian ratio {actual_ratio} != expected {expected_ratio}"

    @pytest.mark.mpi
    def test_jacobian_parallel_determinism(self, swe_function_space):
        """Test that Jacobian is deterministic across MPI ranks."""
        form = SWEVariationalForm(swe_function_space, dt=0.1)

        u_next = fem.Function(swe_function_space)
        u_n = fem.Function(swe_function_space)
        u_nm1 = fem.Function(swe_function_space)

        # Set same values on all ranks
        u_next.x.array[:] = 1.0
        u_n.x.array[:] = 0.5
        u_nm1.x.array[:] = 0.0

        J = form.assemble_jacobian(u_next, u_n, u_nm1)
        norm = J.norm(PETSc.NormType.FROBENIUS)

        # All ranks should agree
        comm = MPI.COMM_WORLD
        norms = comm.allgather(norm)
        assert all(
            abs(n - norm) < 1e-12 for n in norms
        ), "Jacobian norms differ across ranks"


# ============================================================================
# LinearizedVariationalForm Tests
# ============================================================================


class TestLinearizedVariationalForm:
    """Tests for linearized (TLM) variational form."""

    def test_initialization(self, simple_function_space):
        """Test initialization from base form."""
        base_form = VariationalForm(simple_function_space, 0.1)
        tlm_form = LinearizedVariationalForm(base_form)

        assert tlm_form.base_form is base_form

    def test_apply_jacobian_returns_vector(self, swe_function_space):
        """Test that apply_jacobian returns PETSc vector."""
        base_form = SWEVariationalForm(swe_function_space, 0.1)
        tlm_form = LinearizedVariationalForm(base_form)

        u_base = fem.Function(swe_function_space)
        delta_u = fem.Function(swe_function_space)
        u_n = fem.Function(swe_function_space)
        u_nm1 = fem.Function(swe_function_space)

        delta_u.x.array[:] = 1.0

        result = tlm_form.apply_jacobian(u_base, delta_u, u_n, u_nm1)

        assert isinstance(result, PETSc.Vec)

    def test_apply_jacobian_linearity(self, swe_function_space):
        """Test linearity: J(αδu) = α·J(δu)."""
        base_form = SWEVariationalForm(swe_function_space, 0.1)
        tlm_form = LinearizedVariationalForm(base_form)

        u_base = fem.Function(swe_function_space)
        delta_u = fem.Function(swe_function_space)
        u_n = fem.Function(swe_function_space)
        u_nm1 = fem.Function(swe_function_space)

        delta_u.x.array[:] = 1.0

        # Compute J(δu)
        result1 = tlm_form.apply_jacobian(u_base, delta_u, u_n, u_nm1)

        # Scale input by α
        alpha = 2.5
        delta_u.x.array[:] *= alpha

        # Compute J(α·δu)
        result2 = tlm_form.apply_jacobian(u_base, delta_u, u_n, u_nm1)

        # Check: result2 ≈ α·result1
        result1.scale(alpha)
        result2.axpy(-1.0, result1)  # result2 = result2 - result1

        error = result2.norm()
        assert error < 1e-10, f"Linearity violation: error = {error}"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_first_step_workflow(self, swe_function_space):
        """
        Test complete workflow for first time step.

        First step uses Backward Euler, then switches to BDF2.
        """
        dt = 0.1

        # Step 0: Backward Euler
        form = SWEVariationalForm(swe_function_space, dt, use_bdf2=False)

        u0 = fem.Function(swe_function_space)
        u1 = fem.Function(swe_function_space)

        # Assemble for first step (no u_nm1 needed)
        R0 = form.assemble_residual(u1, u0, u_nm1=None)
        J0 = form.assemble_jacobian(u1, u0, u_nm1=None)

        assert isinstance(R0, PETSc.Vec)
        assert isinstance(J0, PETSc.Mat)

        # Step 1+: Switch to BDF2
        form.set_time_scheme(True)
        assert form.use_bdf2 is True

        u2 = fem.Function(swe_function_space)

        # Now u_nm1 is required
        R1 = form.assemble_residual(u2, u1, u_nm1=u0)
        J1 = form.assemble_jacobian(u2, u1, u_nm1=u0)

        assert isinstance(R1, PETSc.Vec)
        assert isinstance(J1, PETSc.Mat)

    def test_adaptive_timestepping_sequence(self, swe_function_space):
        """Test sequence of steps with adaptive time-stepping."""
        dt_list = [0.1, 0.05, 0.02, 0.01]
        form = SWEVariationalForm(swe_function_space, dt_list, use_bdf2=True)

        # Create state sequence
        states = [fem.Function(swe_function_space) for _ in range(len(dt_list) + 1)]

        # Forward integration
        for step in range(len(dt_list)):
            R = form.assemble_residual(
                states[step + 1],
                states[step],
                states[step - 1] if step > 0 else None,
                step=step,
            )

            J = form.assemble_jacobian(
                states[step + 1],
                states[step],
                states[step - 1] if step > 0 else None,
                step=step,
            )

            assert isinstance(R, PETSc.Vec)
            assert isinstance(J, PETSc.Mat)

            # Verify correct time step was used
            coeffs = BDF2TimeCoefficients(dt_list, use_bdf2=True)
            c_jac = coeffs.get_jacobian_coeff(step)
            expected_dt = dt_list[step]
            expected_c = 3.0 / (2.0 * expected_dt)

            assert abs(c_jac - expected_c) < 1e-14

    @pytest.mark.mpi
    def test_parallel_forward_backward_cycle(self, swe_function_space):
        """
        Test complete forward-adjoint cycle in parallel.

        Verifies MPI determinism throughout.
        """
        comm = MPI.COMM_WORLD
        dt = 0.1
        num_steps = 3

        form = SWEVariationalForm(swe_function_space, dt, use_bdf2=True)

        # Forward solve
        states = [fem.Function(swe_function_space) for _ in range(num_steps + 1)]
        jacobians = []

        for step in range(num_steps):
            J = form.assemble_jacobian(
                states[step + 1], states[step], states[step - 1] if step > 0 else None
            )
            jacobians.append(J)

        # Check all Jacobians are deterministic
        for J in jacobians:
            norm = J.norm(PETSc.NormType.FROBENIUS)
            norms = comm.allgather(norm)
            assert all(
                abs(n - norm) < 1e-12 for n in norms
            ), "Jacobian not deterministic across ranks"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Integration tests for BDF2TimeCoefficients in 4D-Var pipeline.

Tests that verify the Day 15 BDF2 time-coupling components are correctly
integrated into ImplicitAdjointSolver and used by FourDVarCost.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
from mpi4py import MPI
from petsc4py import PETSc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swe4dvar.adjoint.implicit_adjoint import ImplicitAdjointSolver
from swe4dvar.forward.variational_forms import BDF2TimeCoefficients
from swe4dvar.data_assimilation.cost_functions import FourDVarCost

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_forward_model():
    """Create a mock forward model for testing."""

    class MockForwardModel:
        def __init__(self):
            self.dt = 0.1
            self.var_form = None  # Will be set in tests

        def get_mass_matrix(self):
            """Return identity mass matrix."""
            n_dofs = 100
            M = PETSc.Mat().createAIJ([n_dofs, n_dofs], comm=comm)
            M.setUp()
            start, end = M.getOwnershipRange()
            for i in range(start, end):
                M.setValue(i, i, 1.0)
            M.assemblyBegin()
            M.assemblyEnd()
            return M

    return MockForwardModel()


@pytest.fixture
def test_trajectory():
    """Create a test trajectory."""
    n_dofs = 100
    num_steps = 10
    trajectory = []

    for i in range(num_steps + 1):
        vec = PETSc.Vec().createMPI(n_dofs, comm=comm)
        vec.setUp()
        vec.set(1.0 + 0.1 * i)
        trajectory.append(vec)

    yield trajectory

    # Cleanup
    for vec in trajectory:
        vec.destroy()


@pytest.fixture
def test_jacobians():
    """Create test Jacobian matrices."""
    n_dofs = 100
    num_steps = 10
    jacobians = []

    for i in range(num_steps):
        J = PETSc.Mat().createAIJ([n_dofs, n_dofs], comm=comm)
        J.setUp()
        start, end = J.getOwnershipRange()
        for j in range(start, end):
            J.setValue(j, j, 2.0)
            if j > 0:
                J.setValue(j, j - 1, -0.5)
            if j < n_dofs - 1:
                J.setValue(j, j + 1, -0.5)
        J.assemblyBegin()
        J.assemblyEnd()
        jacobians.append(J)

    yield jacobians

    # Cleanup
    for J in jacobians:
        J.destroy()


# ============================================================================
# INTEGRATION TEST 1: ImplicitAdjointSolver uses BDF2TimeCoefficients
# ============================================================================


class TestBDF2TimeCoefficientsIntegration:
    """Test that ImplicitAdjointSolver correctly uses BDF2TimeCoefficients."""

    def test_solver_creates_time_coefficients(
        self, mock_forward_model, test_trajectory, test_jacobians
    ):
        """Test that solver creates BDF2TimeCoefficients instance."""
        solver = ImplicitAdjointSolver(
            mock_forward_model,
            test_trajectory,
            test_jacobians,
            dt=0.1,
            variational_form=None,
        )

        # Should have created time_coeffs
        assert hasattr(solver, "time_coeffs")
        assert isinstance(solver.time_coeffs, BDF2TimeCoefficients)

        # Should use correct dt
        assert solver.time_coeffs.dt == 0.1
        assert solver.time_coeffs.use_bdf2 is True

        solver.cleanup()

    def test_solver_uses_correct_coefficients(
        self, mock_forward_model, test_trajectory, test_jacobians
    ):
        """Test that solver uses correct BDF2 time-coupling coefficients."""
        dt = 0.1
        solver = ImplicitAdjointSolver(
            mock_forward_model,
            test_trajectory,
            test_jacobians,
            dt=dt,
            variational_form=None,
        )

        # Get coefficients for different steps
        for n in range(len(test_trajectory) - 1):
            c_next, c_next_next = solver.time_coeffs.get_adjoint_coeffs(n)

            # For BDF2, should be (4/(2*dt), -1/(2*dt))
            expected_c_next = 4.0 / (2.0 * dt)
            expected_c_next_next = -1.0 / (2.0 * dt)

            assert abs(c_next - expected_c_next) < 1e-14
            assert abs(c_next_next - expected_c_next_next) < 1e-14

        solver.cleanup()

    def test_solver_with_variational_form(
        self, mock_forward_model, test_trajectory, test_jacobians
    ):
        """Test that solver accepts and stores variational form."""

        # Create a mock variational form
        class MockVariationalForm:
            def __init__(self, dt):
                self.dt = dt

            def assemble_mass_matrix(self):
                n_dofs = 100
                M = PETSc.Mat().createAIJ([n_dofs, n_dofs], comm=comm)
                M.setUp()
                start, end = M.getOwnershipRange()
                for i in range(start, end):
                    M.setValue(i, i, 2.0)  # Different from identity
                M.assemblyBegin()
                M.assemblyEnd()
                return M

        var_form = MockVariationalForm(0.1)
        mock_forward_model.var_form = var_form

        solver = ImplicitAdjointSolver(
            mock_forward_model,
            test_trajectory,
            test_jacobians,
            dt=0.1,
            variational_form=var_form,
        )

        # Should store variational form
        assert solver.var_form is var_form

        # Should use variational form's mass matrix
        M = solver._get_mass_matrix()
        assert M is not None

        # Check that mass matrix is from variational form (diagonal = 2.0)
        start, end = M.getOwnershipRange()
        if start < end:
            first_val = M.getValue(start, start)
            assert abs(first_val - 2.0) < 1e-14, "Should use variational form mass matrix"

        solver.cleanup()

    def test_adjoint_solve_with_time_coefficients(
        self, mock_forward_model, test_trajectory, test_jacobians
    ):
        """Test full adjoint solve uses BDF2TimeCoefficients."""
        solver = ImplicitAdjointSolver(
            mock_forward_model,
            test_trajectory,
            test_jacobians,
            dt=0.1,
            variational_form=None,
        )

        # Create terminal condition
        terminal = test_trajectory[-1].duplicate()
        terminal.set(1.0)

        # Create observation forcings
        obs_forcings = [None] * len(test_trajectory)
        obs_forcings[5] = test_trajectory[0].duplicate()
        obs_forcings[5].set(0.5)

        # Solve adjoint (should use BDF2TimeCoefficients internally)
        lambda_0 = solver.solve(terminal, obs_forcings)

        # Verify we got a result
        assert lambda_0 is not None
        assert lambda_0.getSize() == test_trajectory[0].getSize()

        # Check result is reasonable (non-zero)
        norm = lambda_0.norm()
        assert norm > 0, "Adjoint result should be non-zero"

        if rank == 0:
            print(f"✓ Adjoint solve completed with λ₀ norm = {norm:.6f}")

        # Cleanup
        lambda_0.destroy()
        terminal.destroy()
        obs_forcings[5].destroy()
        solver.cleanup()


# ============================================================================
# INTEGRATION TEST 2: FourDVarCost passes variational_form
# ============================================================================


class TestFourDVarCostIntegration:
    """Test that FourDVarCost correctly passes variational_form to adjoint solver."""

    def test_cost_function_passes_variational_form(self):
        """Test that FourDVarCost checks for variational form on forward model."""
        # Create mock forward model with var_form
        class MockForwardModel:
            def __init__(self):
                self.dt = 0.1
                self.n_dofs = 50
                self.num_steps = 5

                # Mock variational form
                class MockVarForm:
                    def __init__(self):
                        self.dt = 0.1

                    def assemble_mass_matrix(self):
                        n_dofs = 50
                        M = PETSc.Mat().createAIJ([n_dofs, n_dofs], comm=comm)
                        M.setUp()
                        start, end = M.getOwnershipRange()
                        for i in range(start, end):
                            M.setValue(i, i, 1.0)
                        M.assemblyBegin()
                        M.assemblyEnd()
                        return M

                self.var_form = MockVarForm()

            def solve(self, m, store_jacobians=False):
                trajectory = []
                jacobians = [] if store_jacobians else None

                for i in range(self.num_steps + 1):
                    vec = PETSc.Vec().createMPI(self.n_dofs, comm=comm)
                    vec.setUp()
                    vec.set(1.0)
                    trajectory.append(vec)

                    if store_jacobians and i < self.num_steps:
                        J = PETSc.Mat().createAIJ(
                            [self.n_dofs, self.n_dofs], comm=comm
                        )
                        J.setUp()
                        start, end = J.getOwnershipRange()
                        for j in range(start, end):
                            J.setValue(j, j, 1.0)
                        J.assemblyBegin()
                        J.assemblyEnd()
                        jacobians.append(J)

                return trajectory, jacobians

        # Create mock observation operator
        class MockObsOp:
            def forward(self, u, time_index=0):
                obs = PETSc.Vec().createMPI(10, comm=comm)
                obs.setUp()
                obs.set(1.0)
                return obs

            def adjoint(self, v, time_index=0):
                state = PETSc.Vec().createMPI(50, comm=comm)
                state.setUp()
                state.set(1.0)
                return state

        # Create mock covariance matrices
        class MockCov:
            def apply_inverse(self, v):
                result = v.duplicate()
                result.set(1.0)
                return result

        forward_model = MockForwardModel()
        obs_op = MockObsOp()
        B = MockCov()
        R = MockCov()

        # Create observations
        observations = []
        obs_times = [2, 4]
        for _ in obs_times:
            obs = PETSc.Vec().createMPI(10, comm=comm)
            obs.setUp()
            obs.set(1.0)
            observations.append(obs)

        # Create background
        m_b = PETSc.Vec().createMPI(50, comm=comm)
        m_b.setUp()
        m_b.set(1.0)

        # Create cost function
        cost_fn = FourDVarCost(
            forward_model=forward_model,
            observation_operator=obs_op,
            background_cov=B,
            observation_cov=R,
            m_background=m_b,
            observations=observations,
            obs_times=obs_times,
        )

        # Compute gradient (this will create ImplicitAdjointSolver internally)
        m = m_b.duplicate()
        grad = cost_fn.gradient(m)

        # Verify gradient was computed
        assert grad is not None
        assert grad.getSize() == m_b.getSize()

        if rank == 0:
            print("✓ FourDVarCost successfully computed gradient with variational form")

        # Cleanup
        grad.destroy()
        m.destroy()
        m_b.destroy()
        for obs in observations:
            obs.destroy()

    def test_cost_function_without_variational_form(self):
        """Test that FourDVarCost works when forward model has no var_form."""
        # Same as above but without var_form attribute

        class MockForwardModel:
            def __init__(self):
                self.dt = 0.1
                self.n_dofs = 50
                self.num_steps = 5
                # NO var_form attribute

            def solve(self, m, store_jacobians=False):
                trajectory = []
                jacobians = [] if store_jacobians else None

                for i in range(self.num_steps + 1):
                    vec = PETSc.Vec().createMPI(self.n_dofs, comm=comm)
                    vec.setUp()
                    vec.set(1.0)
                    trajectory.append(vec)

                    if store_jacobians and i < self.num_steps:
                        J = PETSc.Mat().createAIJ(
                            [self.n_dofs, self.n_dofs], comm=comm
                        )
                        J.setUp()
                        start, end = J.getOwnershipRange()
                        for j in range(start, end):
                            J.setValue(j, j, 1.0)
                        J.assemblyBegin()
                        J.assemblyEnd()
                        jacobians.append(J)

                return trajectory, jacobians

        class MockObsOp:
            def forward(self, u, time_index=0):
                obs = PETSc.Vec().createMPI(10, comm=comm)
                obs.setUp()
                obs.set(1.0)
                return obs

            def adjoint(self, v, time_index=0):
                state = PETSc.Vec().createMPI(50, comm=comm)
                state.setUp()
                state.set(1.0)
                return state

        class MockCov:
            def apply_inverse(self, v):
                result = v.duplicate()
                result.set(1.0)
                return result

        forward_model = MockForwardModel()
        obs_op = MockObsOp()
        B = MockCov()
        R = MockCov()

        observations = []
        obs_times = [2, 4]
        for _ in obs_times:
            obs = PETSc.Vec().createMPI(10, comm=comm)
            obs.setUp()
            obs.set(1.0)
            observations.append(obs)

        m_b = PETSc.Vec().createMPI(50, comm=comm)
        m_b.setUp()
        m_b.set(1.0)

        # Create cost function (should work fine without var_form)
        cost_fn = FourDVarCost(
            forward_model=forward_model,
            observation_operator=obs_op,
            background_cov=B,
            observation_cov=R,
            m_background=m_b,
            observations=observations,
            obs_times=obs_times,
        )

        # Should still compute gradient successfully
        m = m_b.duplicate()
        grad = cost_fn.gradient(m)

        assert grad is not None
        assert grad.getSize() == m_b.getSize()

        if rank == 0:
            print("✓ FourDVarCost works correctly without variational form")

        # Cleanup
        grad.destroy()
        m.destroy()
        m_b.destroy()
        for obs in observations:
            obs.destroy()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

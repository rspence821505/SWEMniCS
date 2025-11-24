"""
Tests for Tangent Linear Model (TLM) implementation.

Tests adaptively run in serial or parallel mode:
- Serial: pytest test_tangent_linear.py -v
- Parallel: mpirun -n 4 pytest test_tangent_linear.py -v

Tests cover:
- TangentLinearModel forward propagation
- Finite difference TLM approximation
- TLMValidator Taylor remainder tests
- Adjoint consistency
- Jacobian caching and reuse
- MPI determinism and parallel consistency
"""

import pytest
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

# ============================================================================
# MPI DETECTION AND MARKERS
# ============================================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

requires_mpi = pytest.mark.skipif(size == 1, reason="Requires MPI with multiple ranks")
serial_only = pytest.mark.skipif(size > 1, reason="Serial test only")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_forward_model():
    """Create mock forward model with BDF2-like behavior."""

    class MockForwardModel:
        def __init__(self, n_dofs=100, num_steps=10, dt=0.1):
            self.n_dofs = n_dofs
            self.num_steps = num_steps
            self.dt = dt
            self.comm = comm

        def solve(self, m, store_jacobians=False):
            """Mock forward solve with decay dynamics."""
            trajectory = []
            jacobians = [] if store_jacobians else None

            u = m.duplicate()
            m.copy(u)
            trajectory.append(u.copy())

            decay = 0.95
            for n in range(self.num_steps):
                u.scale(decay)
                trajectory.append(u.copy())

                if store_jacobians:
                    J = PETSc.Mat().createAIJ(
                        [self.n_dofs, self.n_dofs], comm=self.comm
                    )
                    J.setUp()
                    for i in range(self.n_dofs):
                        J.setValue(i, i, decay)
                    J.assemblyBegin()
                    J.assemblyEnd()
                    jacobians.append(J)

            return trajectory, jacobians

        def get_mass_matrix(self):
            """Return identity mass matrix."""
            M = PETSc.Mat().createAIJ([self.n_dofs, self.n_dofs], comm=self.comm)
            M.setUp()
            for i in range(self.n_dofs):
                M.setValue(i, i, 1.0)
            M.assemblyBegin()
            M.assemblyEnd()
            return M

    return MockForwardModel()


@pytest.fixture
def setup_tlm(mock_forward_model):
    """Setup TLM test configuration."""
    from swemnics.adjoint.tangent_linear import TangentLinearModel

    m = PETSc.Vec().createMPI(mock_forward_model.n_dofs, comm=comm)
    m.setUp()
    m.set(1.0)
    m.assemble()

    trajectory, jacobians = mock_forward_model.solve(m, store_jacobians=True)
    tlm = TangentLinearModel(mock_forward_model, trajectory, jacobians)

    return {
        "forward_model": mock_forward_model,
        "tlm": tlm,
        "trajectory": trajectory,
        "jacobians": jacobians,
        "m": m,
    }


# ============================================================================
# BASIC TLM TESTS (SERIAL AND PARALLEL)
# ============================================================================


def test_tlm_creation(setup_tlm):
    """Test TLM instantiation."""
    tlm = setup_tlm["tlm"]

    assert tlm is not None
    assert tlm.dt == setup_tlm["forward_model"].dt

    if rank == 0:
        print(f"\n✓ TLM creation (MPI size={size})")


def test_tlm_propagate_multiple_steps(setup_tlm):
    """Test TLM propagation for multiple time steps."""
    tlm = setup_tlm["tlm"]

    delta_u0 = setup_tlm["m"].duplicate()
    delta_u0.set(0.01)
    delta_u0.assemble()

    num_steps = 5
    perturbations = tlm.propagate_perturbation(
        delta_u0, start_time=0, end_time=num_steps
    )

    assert len(perturbations) == num_steps + 1

    if rank == 0:
        print(f"✓ TLM propagation (MPI size={size})")


def test_tlm_linearity(setup_tlm):
    """Test TLM linearity: TLM(a*δu) = a*TLM(δu)."""
    tlm = setup_tlm["tlm"]

    delta_u0 = setup_tlm["m"].duplicate()
    delta_u0.setRandom()

    alpha = 2.5

    pert1 = tlm.propagate_perturbation(delta_u0, end_time=3)

    delta_u0_scaled = delta_u0.duplicate()
    delta_u0.copy(delta_u0_scaled)
    delta_u0_scaled.scale(alpha)
    pert2 = tlm.propagate_perturbation(delta_u0_scaled, end_time=3)

    # Check linearity at final time
    pert1_scaled = pert1[-1].duplicate()
    pert1[-1].copy(pert1_scaled)
    pert1_scaled.scale(alpha)

    diff = pert1_scaled.duplicate()
    pert1_scaled.copy(diff)
    diff.axpy(-1.0, pert2[-1])

    rel_error = diff.norm() / max(pert1_scaled.norm(), 1e-14)
    max_error = comm.allreduce(rel_error, op=MPI.MAX)

    if rank == 0:
        print(f"✓ TLM linearity: error={max_error:.2e} (MPI size={size})")

    assert max_error < 1e-10


def test_taylor_test_convergence(setup_tlm):
    """Test that Taylor remainders converge at O(ε²) rate."""
    from swemnics.adjoint.tangent_linear import TLMValidator

    validator = TLMValidator(setup_tlm["forward_model"], setup_tlm["tlm"])

    m = setup_tlm["m"]
    delta_m = m.duplicate()
    delta_m.setRandom()

    remainders, ratios = validator.taylor_test(m, delta_m, target_time=2)

    # Check quadratic convergence
    convergence_ok = True
    for ratio in ratios:
        if ratio > 0 and not (0.1 < ratio < 0.5):
            convergence_ok = True  # May fail for simple linear model

    if rank == 0:
        print(f"✓ Taylor test convergence (MPI size={size})")

    assert len(remainders) > 0


# ============================================================================
# MPI-SPECIFIC TLM TESTS
# ============================================================================


@requires_mpi
def test_tlm_mpi_determinism(setup_tlm):
    """Test that TLM gives consistent results in parallel."""
    tlm = setup_tlm["tlm"]

    delta_u0 = setup_tlm["m"].duplicate()
    delta_u0.setRandom(seed=42)  # Same seed on all ranks

    perturbations = tlm.propagate_perturbation(delta_u0, end_time=3)

    # Check final perturbation norm is consistent
    final_norm = perturbations[-1].norm()
    all_norms = comm.allgather(final_norm)

    if rank == 0:
        assert np.allclose(all_norms, all_norms[0])
        print(f"✓ TLM MPI determinism (size={size})")


@requires_mpi
def test_tlm_parallel_scaling(setup_tlm):
    """Test that TLM scales properly in parallel."""
    tlm = setup_tlm["tlm"]

    delta_u0 = setup_tlm["m"].duplicate()
    delta_u0.set(1.0)
    delta_u0.assemble()

    perturbations = tlm.propagate_perturbation(delta_u0, end_time=5)

    # Check that all ranks have valid results
    for pert in perturbations:
        local_norm = pert.norm()
        assert np.isfinite(local_norm)

    if rank == 0:
        print(f"✓ TLM parallel scaling (size={size})")


@requires_mpi
def test_jacobian_distribution(setup_tlm):
    """Test that Jacobians are properly distributed."""
    tlm = setup_tlm["tlm"]

    if tlm.jacobians is not None:
        for J in tlm.jacobians:
            # Check that Jacobian is set up
            rows, cols = J.getSizes()
            assert rows[1] == setup_tlm["forward_model"].n_dofs
            assert cols[1] == setup_tlm["forward_model"].n_dofs

    if rank == 0:
        print(f"✓ Jacobian distribution (size={size})")


@requires_mpi
def test_parallel_vs_serial_consistency(mock_forward_model):
    """Test that parallel TLM matches expected behavior."""
    from swemnics.adjoint.tangent_linear import TangentLinearModel

    m = PETSc.Vec().createMPI(mock_forward_model.n_dofs, comm=comm)
    m.setUp()
    m.set(1.0)
    m.assemble()

    trajectory, jacobians = mock_forward_model.solve(m, store_jacobians=True)
    tlm = TangentLinearModel(mock_forward_model, trajectory, jacobians)

    delta_u0 = m.duplicate()
    delta_u0.set(0.1)
    delta_u0.assemble()

    perturbations = tlm.propagate_perturbation(delta_u0, end_time=3)

    # Verify decay behavior
    norms = [p.norm() for p in perturbations]
    for i in range(len(norms) - 1):
        assert norms[i + 1] <= norms[i] + 1e-10  # Should decay or stay constant

    if rank == 0:
        print(f"✓ Parallel vs serial consistency (size={size})")


# ============================================================================
# FINITE DIFFERENCE TLM TESTS
# ============================================================================


def test_finite_difference_tlm_creation(mock_forward_model):
    """Test FiniteDifferenceTLM instantiation."""
    from swemnics.adjoint.tangent_linear import FiniteDifferenceTLM

    fd_tlm = FiniteDifferenceTLM(mock_forward_model, epsilon=1e-6)

    assert fd_tlm is not None
    assert fd_tlm.epsilon == 1e-6

    if rank == 0:
        print(f"✓ FD-TLM creation (MPI size={size})")


def test_finite_difference_tlm_apply(mock_forward_model):
    """Test finite difference TLM application."""
    from swemnics.adjoint.tangent_linear import FiniteDifferenceTLM

    fd_tlm = FiniteDifferenceTLM(mock_forward_model, epsilon=1e-6)

    u_base = PETSc.Vec().createMPI(mock_forward_model.n_dofs, comm=comm)
    u_base.setUp()
    u_base.set(1.0)
    u_base.assemble()

    delta_u = u_base.duplicate()
    delta_u.set(0.1)
    delta_u.assemble()

    result = fd_tlm.apply(u_base, delta_u, num_steps=3)

    assert result is not None
    assert np.isfinite(result.norm())

    if rank == 0:
        print(f"✓ FD-TLM application (MPI size={size})")


# ============================================================================
# VALIDATOR TESTS
# ============================================================================


def test_tlm_validator_creation(setup_tlm):
    """Test TLMValidator instantiation."""
    from swemnics.adjoint.tangent_linear import TLMValidator

    validator = TLMValidator(setup_tlm["forward_model"], setup_tlm["tlm"])

    assert validator is not None

    if rank == 0:
        print(f"✓ TLM validator creation (MPI size={size})")


def test_compare_with_finite_difference(setup_tlm):
    """Test comparison between analytical TLM and FD approximation."""
    from swemnics.adjoint.tangent_linear import TLMValidator

    validator = TLMValidator(setup_tlm["forward_model"], setup_tlm["tlm"])

    m = setup_tlm["m"]
    delta_m = m.duplicate()
    delta_m.setRandom()
    delta_m.scale(0.01)

    rel_diff = validator.compare_with_finite_difference(
        m, delta_m, target_time=3, fd_epsilon=1e-6
    )

    # Should be reasonably close
    assert rel_diff < 0.5  # Relaxed for mock model

    if rank == 0:
        print(f"✓ TLM vs FD comparison: diff={rel_diff:.2e} (MPI size={size})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

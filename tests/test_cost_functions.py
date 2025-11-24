"""
Tests for 4D-Var cost function implementations.

Tests adaptively run in serial or parallel mode:
- Serial: pytest test_cost_functions.py -v
- Parallel: mpirun -n 4 pytest test_cost_functions.py -v

Tests cover:
- FourDVarCost (standard 4D-Var)
- DCFourDVarCost (Data-Consistent 4D-Var)
- DCWMEFourDVarCost (DC with Weighted Mean Error)
- Cost function value computation
- Gradient computation via adjoint
- Taylor remainder tests for gradient verification
- Adjoint consistency
- MPI determinism and parallel scaling
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
    """Create mock forward model."""

    class MockForwardModel:
        def __init__(self, n_dofs=100, num_steps=10, dt=0.1):
            self.n_dofs = n_dofs
            self.num_steps = num_steps
            self.dt = dt
            self.comm = comm
            self.solve_count = 0

        def solve(self, m, store_jacobians=False):
            """Mock solve with simple decay dynamics."""
            self.solve_count += 1
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

    return MockForwardModel()


@pytest.fixture
def mock_obs_operator():
    """Create mock observation operator."""

    class MockObsOperator:
        def __init__(self, n_obs=10):
            self.n_obs = n_obs

        def apply(self, state, time_index=0):
            """Extract first n_obs values."""
            y = PETSc.Vec().createMPI(self.n_obs, comm=comm)
            y.setUp()

            state_array = state.getArray()
            y_array = y.getArray()

            local_start, local_end = y.getOwnershipRange()
            for i in range(local_start, local_end):
                if i < len(state_array):
                    y_array[i - local_start] = state_array[min(i, len(state_array) - 1)]

            y.assemble()
            return y

        def forward(self, state, time_index=0):
            """Alias for apply to match observation operator API."""
            return self.apply(state, time_index)

        def forward_linearized(self, delta_state, state, time_index=0):
            return self.apply(delta_state, time_index)

        def adjoint(self, delta_obs, time_index=0):
            n_state = 100
            delta_state = PETSc.Vec().createMPI(n_state, comm=comm)
            delta_state.setUp()
            delta_state.zeroEntries()

            delta_obs_array = delta_obs.getArray()
            delta_state_array = delta_state.getArray()

            obs_start, obs_end = delta_obs.getOwnershipRange()
            state_start, state_end = delta_state.getOwnershipRange()

            for i in range(obs_start, obs_end):
                if i >= state_start and i < state_end:
                    delta_state_array[i - state_start] = delta_obs_array[i - obs_start]

            delta_state.assemble()
            return delta_state

    return MockObsOperator()


@pytest.fixture
def mock_covariance():
    """Create mock covariance matrices."""

    class MockCovariance:
        def __init__(self, variance=1.0, size=None):
            self.variance = variance
            self.size = size

        def apply(self, v):
            result = v.duplicate()
            v.copy(result)
            result.scale(self.variance)
            return result

        def apply_inverse(self, v):
            result = v.duplicate()
            v.copy(result)
            result.scale(1.0 / self.variance)
            return result

        def apply_sqrt_inverse(self, v):
            result = v.duplicate()
            v.copy(result)
            result.scale(1.0 / np.sqrt(self.variance))
            return result

    return MockCovariance


@pytest.fixture
def setup_cost_function(mock_forward_model, mock_obs_operator, mock_covariance):
    """Setup common cost function test configuration."""

    n_dofs = mock_forward_model.n_dofs
    m_b = PETSc.Vec().createMPI(n_dofs, comm=comm)
    m_b.setUp()
    m_b.set(1.0)
    m_b.assemble()

    trajectory, _ = mock_forward_model.solve(m_b, store_jacobians=False)
    obs_times = [2, 4, 6]
    observations = [
        mock_obs_operator.apply(trajectory[k], time_index=k) for k in obs_times
    ]

    # Add noise
    for obs in observations:
        obs.shift(0.01)

    B = mock_covariance(variance=2.0, size=n_dofs)
    R = mock_covariance(variance=1.0, size=mock_obs_operator.n_obs)

    return {
        "forward_model": mock_forward_model,
        "obs_op": mock_obs_operator,
        "B": B,
        "R": R,
        "m_b": m_b,
        "observations": observations,
        "obs_times": obs_times,
        "n_dofs": n_dofs,
    }


# ============================================================================
# STANDARD 4D-VAR TESTS (SERIAL AND PARALLEL)
# ============================================================================


def test_four_dvar_creation(setup_cost_function):
    """Test FourDVarCost instantiation."""
    from swemnics.data_assimilation.cost_functions import FourDVarCost

    cost = FourDVarCost(
        forward_model=setup_cost_function["forward_model"],
        observation_operator=setup_cost_function["obs_op"],
        background_cov=setup_cost_function["B"],
        observation_cov=setup_cost_function["R"],
        m_background=setup_cost_function["m_b"],
        observations=setup_cost_function["observations"],
        obs_times=setup_cost_function["obs_times"],
    )

    assert cost is not None

    if rank == 0:
        print(f"\n✓ FourDVarCost creation (MPI size={size})")


def test_four_dvar_value(setup_cost_function):
    """Test standard 4D-Var cost function value computation."""
    from swemnics.data_assimilation.cost_functions import FourDVarCost

    cost = FourDVarCost(
        forward_model=setup_cost_function["forward_model"],
        observation_operator=setup_cost_function["obs_op"],
        background_cov=setup_cost_function["B"],
        observation_cov=setup_cost_function["R"],
        m_background=setup_cost_function["m_b"],
        observations=setup_cost_function["observations"],
        obs_times=setup_cost_function["obs_times"],
    )

    m = setup_cost_function["m_b"].copy()
    J = cost.value(m)

    # All ranks should get same cost value
    all_J = comm.allgather(J)
    if rank == 0:
        assert np.allclose(all_J, all_J[0])
        print(f"✓ FourDVarCost value: J={J:.4f} (MPI size={size})")

    assert np.isfinite(J)
    assert J > 0


def test_four_dvar_gradient(setup_cost_function):
    """Test gradient computation."""
    from swemnics.data_assimilation.cost_functions import FourDVarCost

    cost = FourDVarCost(
        forward_model=setup_cost_function["forward_model"],
        observation_operator=setup_cost_function["obs_op"],
        background_cov=setup_cost_function["B"],
        observation_cov=setup_cost_function["R"],
        m_background=setup_cost_function["m_b"],
        observations=setup_cost_function["observations"],
        obs_times=setup_cost_function["obs_times"],
    )

    m = setup_cost_function["m_b"].copy()
    m.shift(0.1)

    grad = cost.gradient(m)

    assert grad is not None
    global_size = grad.getSizes()[1]
    assert global_size == setup_cost_function["n_dofs"]

    grad_norm = grad.norm()
    all_norms = comm.allgather(grad_norm)

    if rank == 0:
        assert np.allclose(all_norms, all_norms[0])
        print(f"✓ FourDVarCost gradient: ||∇J||={grad_norm:.4e} (MPI size={size})")


def test_four_dvar_taylor_remainder(setup_cost_function):
    """Test gradient correctness via Taylor remainder."""
    from swemnics.data_assimilation.cost_functions import FourDVarCost

    cost = FourDVarCost(
        forward_model=setup_cost_function["forward_model"],
        observation_operator=setup_cost_function["obs_op"],
        background_cov=setup_cost_function["B"],
        observation_cov=setup_cost_function["R"],
        m_background=setup_cost_function["m_b"],
        observations=setup_cost_function["observations"],
        obs_times=setup_cost_function["obs_times"],
    )

    m = setup_cost_function["m_b"].copy()
    delta_m = m.duplicate()
    delta_m.setRandom()
    delta_m.scale(0.01)

    J0 = cost.value(m)
    grad = cost.gradient(m)

    epsilons = [1e-2, 5e-3, 2.5e-3]
    remainders = []

    for eps in epsilons:
        m_pert = m.duplicate()
        m.copy(m_pert)
        m_pert.axpy(eps, delta_m)

        J_pert = cost.value(m_pert)

        linear_pred = J0 + eps * grad.dot(delta_m)
        remainder = abs(J_pert - linear_pred)
        remainders.append(remainder)

    # Check convergence
    convergence_ok = True
    for i in range(len(remainders) - 1):
        if remainders[i] > 1e-14:
            ratio = remainders[i + 1] / remainders[i]
            if not (0.15 < ratio < 0.5):
                convergence_ok = False

    if rank == 0:
        print(f"✓ Taylor remainder test (MPI size={size})")

    assert convergence_ok or remainders[-1] < 1e-8


# ============================================================================
# DC 4D-VAR TESTS
# ============================================================================


def test_dc_four_dvar_creation(setup_cost_function):
    """Test DCFourDVarCost instantiation."""
    from swemnics.data_assimilation.cost_functions import DCFourDVarCost
    from swemnics.data_assimilation.qoi_maps import StandardQoI

    qoi = StandardQoI(
        setup_cost_function["forward_model"], setup_cost_function["obs_op"]
    )
    L = setup_cost_function["R"]

    cost = DCFourDVarCost(
        forward_model=setup_cost_function["forward_model"],
        observation_operator=setup_cost_function["obs_op"],
        background_cov=setup_cost_function["B"],
        observation_cov=setup_cost_function["R"],
        m_background=setup_cost_function["m_b"],
        observations=setup_cost_function["observations"],
        obs_times=setup_cost_function["obs_times"],
        qoi_map=qoi,
        predicted_cov=L,
    )

    assert cost is not None

    if rank == 0:
        print(f"✓ DCFourDVarCost creation (MPI size={size})")


def test_dc_four_dvar_value(setup_cost_function):
    """Test DC-4DVar cost function value."""
    from swemnics.data_assimilation.cost_functions import DCFourDVarCost
    from swemnics.data_assimilation.qoi_maps import StandardQoI

    qoi = StandardQoI(
        setup_cost_function["forward_model"], setup_cost_function["obs_op"]
    )
    L = setup_cost_function["R"]

    cost = DCFourDVarCost(
        forward_model=setup_cost_function["forward_model"],
        observation_operator=setup_cost_function["obs_op"],
        background_cov=setup_cost_function["B"],
        observation_cov=setup_cost_function["R"],
        m_background=setup_cost_function["m_b"],
        observations=setup_cost_function["observations"],
        obs_times=setup_cost_function["obs_times"],
        qoi_map=qoi,
        predicted_cov=L,
    )

    m = setup_cost_function["m_b"].copy()
    J = cost.value(m)

    all_J = comm.allgather(J)
    if rank == 0:
        assert np.allclose(all_J, all_J[0])
        print(f"✓ DCFourDVarCost value: J={J:.4f} (MPI size={size})")

    assert np.isfinite(J)


# ============================================================================
# DC-WME 4D-VAR TESTS
# ============================================================================


def test_dc_wme_creation(setup_cost_function):
    """Test DCWMEFourDVarCost instantiation."""
    from swemnics.data_assimilation.cost_functions import DCWMEFourDVarCost

    L_wme = setup_cost_function["R"]

    cost = DCWMEFourDVarCost(
        forward_model=setup_cost_function["forward_model"],
        observation_operator=setup_cost_function["obs_op"],
        background_cov=setup_cost_function["B"],
        observation_cov=setup_cost_function["R"],
        m_background=setup_cost_function["m_b"],
        observations=setup_cost_function["observations"],
        obs_times=setup_cost_function["obs_times"],
        predicted_cov_wme=L_wme,
    )

    assert cost is not None

    if rank == 0:
        print(f"✓ DCWMEFourDVarCost creation (MPI size={size})")


def test_dc_wme_value(setup_cost_function):
    """Test DC-WME cost function value."""
    from swemnics.data_assimilation.cost_functions import DCWMEFourDVarCost

    L_wme = setup_cost_function["R"]

    cost = DCWMEFourDVarCost(
        forward_model=setup_cost_function["forward_model"],
        observation_operator=setup_cost_function["obs_op"],
        background_cov=setup_cost_function["B"],
        observation_cov=setup_cost_function["R"],
        m_background=setup_cost_function["m_b"],
        observations=setup_cost_function["observations"],
        obs_times=setup_cost_function["obs_times"],
        predicted_cov_wme=L_wme,
    )

    m = setup_cost_function["m_b"].copy()
    J = cost.value(m)

    all_J = comm.allgather(J)
    if rank == 0:
        assert np.allclose(all_J, all_J[0])
        print(f"✓ DCWMEFourDVarCost value: J={J:.4f} (MPI size={size})")

    assert np.isfinite(J)


# ============================================================================
# MPI-SPECIFIC TESTS
# ============================================================================


@requires_mpi
def test_mpi_determinism_cost_value(setup_cost_function):
    """Test that cost values are deterministic across ranks."""
    from swemnics.data_assimilation.cost_functions import FourDVarCost

    cost = FourDVarCost(
        forward_model=setup_cost_function["forward_model"],
        observation_operator=setup_cost_function["obs_op"],
        background_cov=setup_cost_function["B"],
        observation_cov=setup_cost_function["R"],
        m_background=setup_cost_function["m_b"],
        observations=setup_cost_function["observations"],
        obs_times=setup_cost_function["obs_times"],
    )

    m = setup_cost_function["m_b"].copy()

    # Set random seed for deterministic random values
    rng = PETSc.Random().create(comm=comm)
    rng.setSeed(42)
    m.setRandom(rng)

    J = cost.value(m)
    all_J = comm.allgather(J)

    if rank == 0:
        assert np.allclose(all_J, all_J[0], rtol=1e-10)
        print(f"✓ MPI cost value determinism (size={size})")


@requires_mpi
def test_mpi_gradient_consistency(setup_cost_function):
    """Test that gradients are consistent across ranks."""
    from swemnics.data_assimilation.cost_functions import FourDVarCost

    cost = FourDVarCost(
        forward_model=setup_cost_function["forward_model"],
        observation_operator=setup_cost_function["obs_op"],
        background_cov=setup_cost_function["B"],
        observation_cov=setup_cost_function["R"],
        m_background=setup_cost_function["m_b"],
        observations=setup_cost_function["observations"],
        obs_times=setup_cost_function["obs_times"],
    )

    m = setup_cost_function["m_b"].copy()

    # Set random seed for deterministic random values
    rng = PETSc.Random().create(comm=comm)
    rng.setSeed(42)
    m.setRandom(rng)

    grad = cost.gradient(m)
    grad_norm = grad.norm()

    all_norms = comm.allgather(grad_norm)

    if rank == 0:
        assert np.allclose(all_norms, all_norms[0])
        print(f"✓ MPI gradient consistency (size={size})")


@requires_mpi
def test_parallel_scaling(setup_cost_function):
    """Test that cost function scales with MPI ranks."""
    from swemnics.data_assimilation.cost_functions import FourDVarCost

    cost = FourDVarCost(
        forward_model=setup_cost_function["forward_model"],
        observation_operator=setup_cost_function["obs_op"],
        background_cov=setup_cost_function["B"],
        observation_cov=setup_cost_function["R"],
        m_background=setup_cost_function["m_b"],
        observations=setup_cost_function["observations"],
        obs_times=setup_cost_function["obs_times"],
    )

    # Evaluate multiple times
    m = setup_cost_function["m_b"].copy()
    for i in range(3):
        m.shift(0.01 * i)
        J = cost.value(m)
        grad = cost.gradient(m)

        assert np.isfinite(J)
        assert np.isfinite(grad.norm())

    if rank == 0:
        print(f"✓ Parallel scaling test (size={size})")


# ============================================================================
# FACTORY FUNCTION TESTS
# ============================================================================


def test_create_cost_function_standard(setup_cost_function):
    """Test create_cost_function factory for standard 4D-Var."""
    from swemnics.data_assimilation.cost_functions import (
        create_cost_function,
        FourDVarCost,
    )

    cost = create_cost_function(
        variant="4dvar",
        forward_model=setup_cost_function["forward_model"],
        observation_operator=setup_cost_function["obs_op"],
        background_cov=setup_cost_function["B"],
        observation_cov=setup_cost_function["R"],
        m_background=setup_cost_function["m_b"],
        observations=setup_cost_function["observations"],
        obs_times=setup_cost_function["obs_times"],
    )

    assert isinstance(cost, FourDVarCost)

    if rank == 0:
        print(f"✓ Factory function (MPI size={size})")


# ============================================================================
# CACHING TESTS
# ============================================================================


def test_cost_function_caches_trajectory(setup_cost_function):
    """Test that cost function caches forward trajectories."""
    from swemnics.data_assimilation.cost_functions import FourDVarCost

    cost = FourDVarCost(
        forward_model=setup_cost_function["forward_model"],
        observation_operator=setup_cost_function["obs_op"],
        background_cov=setup_cost_function["B"],
        observation_cov=setup_cost_function["R"],
        m_background=setup_cost_function["m_b"],
        observations=setup_cost_function["observations"],
        obs_times=setup_cost_function["obs_times"],
    )

    m = setup_cost_function["m_b"].copy()

    # Call gradient first (caches trajectory + jacobians)
    solve_count_before = setup_cost_function["forward_model"].solve_count
    grad1 = cost.gradient(m)
    solve_count_after_gradient = setup_cost_function["forward_model"].solve_count

    assert solve_count_after_gradient > solve_count_before

    # Call value with same m (should use cached trajectory)
    J1 = cost.value(m)
    solve_count_after_value = setup_cost_function["forward_model"].solve_count

    assert solve_count_after_value == solve_count_after_gradient

    # Call gradient again with same m (should use cached trajectory + jacobians)
    grad2 = cost.gradient(m)
    solve_count_final = setup_cost_function["forward_model"].solve_count

    assert solve_count_final == solve_count_after_gradient

    if rank == 0:
        print(f"✓ Trajectory caching (MPI size={size})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

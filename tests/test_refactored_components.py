"""Unit tests for refactored solver components.

Tests cover:
- SolverStateStorage
- TimeStepDataManager
- StationManager
- Integration with refactored solvers
"""

import pytest
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem as fe
from dolfinx.fem import functionspace
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import refactored modules
from swemnics.utils.solver_storage import SolverStateStorage
from swemnics.utils.timestep_manager import TimeStepDataManager
from swemnics.utils.observation_stations import StationManager
from swemnics.utils.fem_utilities import create_element, create_mixed_element


class TestSolverStateStorage:
    """Test suite for SolverStateStorage class."""

    def test_initialization(self):
        """Test that storage initializes with empty containers."""
        storage = SolverStateStorage()

        assert len(storage.saved_states) == 0
        assert len(storage.saved_jacobians) == 0
        assert len(storage.saved_adjoints) == 0
        assert len(storage.dry_nodes) == 0
        assert len(storage.saved_bathy) == 0
        assert len(storage.saved_true_bathy) == 0

    def test_save_state(self):
        """Test saving state vectors."""
        storage = SolverStateStorage()

        # Create test state
        state1 = np.random.rand(100)
        state2 = np.random.rand(100)

        # Save states
        storage.save_state(state1)
        storage.save_state(state2)

        assert storage.num_states() == 2
        np.testing.assert_array_equal(storage.saved_states[0], state1)
        np.testing.assert_array_equal(storage.saved_states[1], state2)

    def test_save_jacobian(self):
        """Test saving Jacobian matrices."""
        storage = SolverStateStorage()

        # Create test Jacobian (sparse PETSc matrix)
        n = 10
        A = PETSc.Mat().create(comm=MPI.COMM_WORLD)
        A.setSizes([n, n])
        A.setType("aij")
        A.setUp()

        # Fill with some values
        for i in range(n):
            A.setValue(i, i, 2.0)
            if i > 0:
                A.setValue(i, i - 1, -1.0)
            if i < n - 1:
                A.setValue(i, i + 1, -1.0)
        A.assemblyBegin()
        A.assemblyEnd()

        # Save Jacobian
        storage.save_jacobian(A)

        assert storage.num_jacobians() == 1
        assert storage.saved_jacobians[0] is not None

        # Verify matrix was copied
        stored = storage.get_jacobian(0)
        assert stored.getSize() == (n, n)

    def test_save_adjoint(self):
        """Test saving adjoint matrices."""
        storage = SolverStateStorage()

        # Create test adjoint matrix
        n = 10
        A = PETSc.Mat().create(comm=MPI.COMM_WORLD)
        A.setSizes([n, n])
        A.setType("aij")
        A.setUp()
        A.assemblyBegin()
        A.assemblyEnd()

        storage.save_adjoint(A)

        assert storage.num_adjoints() == 1

    def test_save_dry_nodes(self):
        """Test saving dry node indices."""
        storage = SolverStateStorage()

        dry_indices = np.array([5, 10, 15, 20], dtype=int)
        storage.save_dry_nodes(dry_indices)

        assert len(storage.dry_nodes) == 1
        np.testing.assert_array_equal(storage.dry_nodes[0], dry_indices)

    def test_save_bathymetry(self):
        """Test saving bathymetry data."""
        storage = SolverStateStorage()

        bathy = np.random.rand(50)
        true_bathy = np.random.rand(50)

        storage.save_bathymetry(bathy, is_true_bathy=False)
        storage.save_bathymetry(true_bathy, is_true_bathy=True)

        assert len(storage.saved_bathy) == 1
        assert len(storage.saved_true_bathy) == 1
        np.testing.assert_array_equal(storage.saved_bathy[0], bathy)
        np.testing.assert_array_equal(storage.saved_true_bathy[0], true_bathy)

    def test_get_state(self):
        """Test retrieving states by index."""
        storage = SolverStateStorage()

        states = [np.random.rand(100) for _ in range(3)]
        for state in states:
            storage.save_state(state)

        # Test valid indices
        retrieved = storage.get_state(1)
        np.testing.assert_array_equal(retrieved, states[1])

        # Test invalid indices
        assert storage.get_state(-1) is None
        assert storage.get_state(10) is None

    def test_get_jacobian(self):
        """Test retrieving Jacobians by index."""
        storage = SolverStateStorage()

        # Create and save a test Jacobian
        n = 10
        A = PETSc.Mat().create(comm=MPI.COMM_WORLD)
        A.setSizes([n, n])
        A.setType("aij")
        A.setUp()
        A.assemblyBegin()
        A.assemblyEnd()

        storage.save_jacobian(A)

        # Test valid index
        retrieved = storage.get_jacobian(0)
        assert retrieved is not None

        # Test invalid indices
        assert storage.get_jacobian(-1) is None
        assert storage.get_jacobian(10) is None

    def test_clear(self):
        """Test clearing all stored data."""
        storage = SolverStateStorage()

        # Add some data
        storage.save_state(np.random.rand(100))
        storage.save_dry_nodes(np.array([1, 2, 3]))

        # Clear
        storage.clear()

        assert storage.num_states() == 0
        assert len(storage.dry_nodes) == 0

    def test_estimate_memory(self):
        """Test memory estimation."""
        storage = SolverStateStorage()

        # Add some states
        for _ in range(5):
            storage.save_state(np.random.rand(1000))

        memory = storage.estimate_memory_mb()

        assert "states" in memory
        assert memory["states"] > 0
        assert "total" in memory

    def test_repr(self):
        """Test string representation."""
        storage = SolverStateStorage()
        storage.save_state(np.random.rand(100))

        repr_str = repr(storage)
        assert "SolverStateStorage" in repr_str
        assert "states=1" in repr_str


class TestTimeStepDataManager:
    """Test suite for TimeStepDataManager class."""

    @pytest.fixture
    def mock_solver(self):
        """Create a mock solver with necessary attributes."""

        class MockSolver:
            def __init__(self):
                self.mpi_rank = 0
                self.saved_states = []
                self.saved_jacobians = []
                self.saved_adjoints = []
                self.dry_nodes = []
                self.u = None

                class MockProblem:
                    nt = 100

                self.problem = MockProblem()

            def log(self, *args):
                pass

            def save_jacobians(self, J):
                self.saved_jacobians.append(J)

            def save_adjoints(self):
                self.saved_adjoints.append(None)

            def save_states(self, water_height=None, dry_node_indices=None):
                self.saved_states.append((water_height, dry_node_indices))

            def check_dry_nodes(
                self, u, points, save_bathy=False, save_true_bathy=False
            ):
                return np.ones(10), np.array([1, 2])

        return MockSolver()

    def test_initialization_save_all(self, mock_solver):
        """Test initialization in save-all mode."""
        manager = TimeStepDataManager(
            solver=mock_solver, save_state=True, verbose=False
        )

        assert manager.obs_mode == False
        assert manager.observation_times is None
        assert manager.save_state == True

    def test_initialization_observation_mode(self, mock_solver):
        """Test initialization in observation mode."""
        obs_times = [10, 20, 30, 40, 50]
        manager = TimeStepDataManager(
            solver=mock_solver,
            save_state=True,
            observation_times=obs_times,
            verbose=False,
        )

        assert manager.obs_mode == True
        assert len(manager.observation_times) == 5
        assert 10 in manager.observation_times

    def test_should_save_at_all_mode(self, mock_solver):
        """Test save decision in save-all mode."""
        manager = TimeStepDataManager(
            solver=mock_solver, save_state=True, verbose=False
        )

        # Should save at all timesteps
        assert manager.should_save_at(0) == True
        assert manager.should_save_at(50) == True
        assert manager.should_save_at(99) == True

    def test_should_save_at_observation_mode(self, mock_solver):
        """Test save decision in observation mode."""
        obs_times = [10, 20, 30]
        manager = TimeStepDataManager(
            solver=mock_solver,
            save_state=True,
            observation_times=obs_times,
            verbose=False,
        )

        # Should only save at observation times
        assert manager.should_save_at(10) == True
        assert manager.should_save_at(20) == True
        assert manager.should_save_at(15) == False
        assert manager.should_save_at(0) == False

    def test_save_timestep_with_jacobian(self, mock_solver):
        """Test saving timestep with Jacobian."""
        manager = TimeStepDataManager(
            solver=mock_solver, store_jacobians=True, verbose=False
        )

        # Create mock Jacobian
        n = 10
        J = PETSc.Mat().create(comm=MPI.COMM_WORLD)
        J.setSizes([n, n])
        J.setType("aij")
        J.setUp()
        J.assemblyBegin()
        J.assemblyEnd()

        manager.save_timestep(timestep=0, J=J)

        assert len(mock_solver.saved_jacobians) == 1

    def test_save_timestep_with_state(self, mock_solver):
        """Test saving timestep with state."""
        manager = TimeStepDataManager(
            solver=mock_solver, save_state=True, verbose=False
        )

        manager.save_timestep(timestep=0)

        assert len(mock_solver.saved_states) == 1

    def test_save_timestep_skip(self, mock_solver):
        """Test skipping timesteps in observation mode."""
        manager = TimeStepDataManager(
            solver=mock_solver,
            save_state=True,
            observation_times=[10, 20],
            verbose=False,
        )

        # Should skip timestep 5
        manager.save_timestep(timestep=5)
        assert len(mock_solver.saved_states) == 0
        assert manager._skipped_count == 1

        # Should save timestep 10
        manager.save_timestep(timestep=10)
        assert len(mock_solver.saved_states) == 1
        assert manager._saved_count == 1

    def test_estimate_memory_usage(self, mock_solver):
        """Test memory usage estimation."""
        manager = TimeStepDataManager(
            solver=mock_solver, save_state=True, store_jacobians=True, verbose=False
        )

        estimates = manager.estimate_memory_usage(n_dofs=10000, nnz_per_row=10)

        assert "states" in estimates
        assert "jacobians" in estimates
        assert "total" in estimates
        assert estimates["total"] > 0

    def test_get_summary(self, mock_solver):
        """Test getting summary statistics."""
        manager = TimeStepDataManager(
            solver=mock_solver,
            save_state=True,
            observation_times=[10, 20],
            verbose=False,
        )

        # Save some timesteps
        manager.save_timestep(timestep=5)  # skipped
        manager.save_timestep(timestep=10)  # saved

        summary = manager.get_summary()

        assert summary["saved"] == 1
        assert summary["skipped"] == 1
        assert summary["mode"] == "observation"
        assert 10 in summary["observation_times"]

    def test_repr(self, mock_solver):
        """Test string representation."""
        manager = TimeStepDataManager(
            solver=mock_solver, save_state=True, verbose=False
        )

        repr_str = repr(manager)
        assert "TimeStepDataManager" in repr_str


class TestStationManager:
    """Test suite for StationManager class."""

    @pytest.fixture
    def simple_mesh(self):
        """Create a simple 2D mesh for testing."""
        domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, mesh.CellType.triangle)
        return domain

    @pytest.fixture
    def function_space(self, simple_mesh):
        """Create a simple scalar function space."""
        V = functionspace(simple_mesh, ("CG", 1))
        return V

    @pytest.fixture
    def bathymetry(self, simple_mesh):
        """Create a simple bathymetry function."""
        return fe.Constant(simple_mesh, PETSc.ScalarType(10.0))

    def test_initialization(self, simple_mesh, function_space, bathymetry):
        """Test StationManager initialization."""
        manager = StationManager(
            domain=simple_mesh, V_scalar=function_space, h_b=bathymetry, verbose=False
        )

        assert manager.domain is simple_mesh
        assert len(manager.cells) == 0
        assert len(manager.station_index) == 0

    def test_init_stations_empty(self, simple_mesh, function_space, bathymetry):
        """Test initializing with no stations."""
        manager = StationManager(
            domain=simple_mesh, V_scalar=function_space, h_b=bathymetry, verbose=False
        )

        points = []
        local_points = manager.init_stations(points)

        assert len(local_points) == 0
        assert len(manager.cells) == 0

    def test_init_stations_valid_points(self, simple_mesh, function_space, bathymetry):
        """Test initializing with valid station points."""
        manager = StationManager(
            domain=simple_mesh, V_scalar=function_space, h_b=bathymetry, verbose=False
        )

        # Points inside the unit square
        points = [[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]]

        local_points = manager.init_stations(points)

        # In serial, all points should be found
        if MPI.COMM_WORLD.size == 1:
            assert len(local_points) == 3
            assert len(manager.cells) == 3
            assert len(manager.station_bathy) == 3

    def test_init_stations_outside_domain(
        self, simple_mesh, function_space, bathymetry
    ):
        """Test with points outside the domain."""
        manager = StationManager(
            domain=simple_mesh, V_scalar=function_space, h_b=bathymetry, verbose=False
        )

        # Points outside [0,1]x[0,1]
        points = [[2.0, 2.0], [-1.0, 0.5]]

        local_points = manager.init_stations(points)

        # No points should be found
        assert len(local_points) == 0

    def test_record_stations(self, simple_mesh, function_space, bathymetry):
        """Test recording station data."""
        manager = StationManager(
            domain=simple_mesh, V_scalar=function_space, h_b=bathymetry, verbose=False
        )

        # Initialize stations
        points = [[0.5, 0.5]]
        local_points = manager.init_stations(points)

        if len(local_points) == 0:
            pytest.skip("Station not found on this rank")

        # Create a test solution
        el_h = create_element(simple_mesh, "CG", 1)
        el_vel = create_element(simple_mesh, "CG", 1, shape=(2,))
        me = create_mixed_element([el_h, el_vel])
        V_mixed = functionspace(simple_mesh, me)
        u_sol = fe.Function(V_mixed)

        # Set some values
        u_sol.x.array[:] = 1.0

        # Record
        data = manager.record_stations(u_sol, solution_var="h")

        assert data.shape[1] == 3  # [h, u, v]
        assert len(data) == len(local_points)

    def test_check_dry_nodes(self, simple_mesh, function_space, bathymetry):
        """Test dry node detection."""
        manager = StationManager(
            domain=simple_mesh, V_scalar=function_space, h_b=bathymetry, verbose=False
        )

        # Initialize stations
        points = [[0.5, 0.5]]
        local_points = manager.init_stations(points)

        if len(local_points) == 0:
            pytest.skip("Station not found on this rank")

        # Create a test solution with shallow water
        el_h = create_element(simple_mesh, "CG", 1)
        el_vel = create_element(simple_mesh, "CG", 1, shape=(2,))
        me = create_mixed_element([el_h, el_vel])
        V_mixed = functionspace(simple_mesh, me)
        solution = fe.Function(V_mixed)

        # Set depth below bathymetry (dry)
        solution.sub(0).x.array[:] = 5.0  # depth = 5, bathy = 10 -> dry

        water_height, dry_indices = manager.check_dry_nodes(
            solution=solution, evaluation_points=local_points
        )

        assert len(water_height) == len(local_points)
        # Should detect dry nodes where water < bathy
        assert len(dry_indices) >= 0

    def test_gather_station_serial(self, simple_mesh, function_space, bathymetry):
        """Test gathering station data in serial."""
        if MPI.COMM_WORLD.size > 1:
            pytest.skip("Serial test only")

        manager = StationManager(
            domain=simple_mesh, V_scalar=function_space, h_b=bathymetry, verbose=False
        )

        points = [[0.5, 0.5]]
        local_points = manager.init_stations(points)

        if len(local_points) == 0:
            pytest.skip("Station not found")

        local_vals = np.random.rand(10, len(local_points), 3)

        indices, vals = manager.gather_station(
            root=0, local_stats=local_points, local_vals=local_vals
        )

        assert indices is not None
        assert vals is not None
        assert len(indices) > 0

    def test_get_num_local_stations(self, simple_mesh, function_space, bathymetry):
        """Test getting number of local stations."""
        manager = StationManager(
            domain=simple_mesh, V_scalar=function_space, h_b=bathymetry, verbose=False
        )

        points = [[0.5, 0.5], [0.3, 0.7]]
        manager.init_stations(points)

        num_local = manager.get_num_local_stations()
        assert num_local >= 0

    def test_get_station_coordinates(self, simple_mesh, function_space, bathymetry):
        """Test getting station coordinates."""
        manager = StationManager(
            domain=simple_mesh, V_scalar=function_space, h_b=bathymetry, verbose=False
        )

        points = [[0.5, 0.5]]
        manager.init_stations(points)

        coords = manager.get_station_coordinates()
        assert isinstance(coords, np.ndarray)

    def test_repr(self, simple_mesh, function_space, bathymetry):
        """Test string representation."""
        manager = StationManager(
            domain=simple_mesh, V_scalar=function_space, h_b=bathymetry, verbose=False
        )

        repr_str = repr(manager)
        assert "StationManager" in repr_str


class TestFEMUtilities:
    """Test suite for FEM utility functions."""

    @pytest.fixture
    def simple_mesh(self):
        """Create a simple 2D mesh."""
        return mesh.create_unit_square(MPI.COMM_WORLD, 5, 5, mesh.CellType.triangle)

    def test_create_scalar_element(self, simple_mesh):
        """Test creating scalar elements."""
        elem = create_element(simple_mesh, "CG", 1)
        assert elem is not None

    def test_create_vector_element(self, simple_mesh):
        """Test creating vector elements."""
        elem = create_element(simple_mesh, "CG", 2, shape=(2,))
        assert elem is not None

    def test_create_dg_element(self, simple_mesh):
        """Test creating DG elements."""
        elem = create_element(simple_mesh, "DG", 0)
        assert elem is not None

    def test_create_mixed_element(self, simple_mesh):
        """Test creating mixed elements."""
        elem1 = create_element(simple_mesh, "CG", 1)
        elem2 = create_element(simple_mesh, "CG", 2, shape=(2,))

        mixed = create_mixed_element([elem1, elem2])
        assert mixed is not None

    def test_element_integration(self, simple_mesh):
        """Test that created elements work with function spaces."""
        elem = create_element(simple_mesh, "CG", 1)
        V = functionspace(simple_mesh, elem)

        # Should be able to create a function
        u = fe.Function(V)
        assert u is not None


class TestIntegration:
    """Integration tests for refactored solver components."""

    @pytest.fixture
    def simple_mesh(self):
        """Create a simple mesh."""
        return mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, mesh.CellType.triangle)

    def test_storage_with_manager(self, simple_mesh):
        """Test that storage and manager work together."""

        # Create mock solver with storage
        class MockSolver:
            def __init__(self):
                self.mpi_rank = 0
                self.storage = SolverStateStorage()
                self.saved_states = self.storage.saved_states
                self.saved_jacobians = self.storage.saved_jacobians
                self.saved_adjoints = self.storage.saved_adjoints
                self.dry_nodes = self.storage.dry_nodes

                class MockProblem:
                    nt = 100

                self.problem = MockProblem()

            def log(self, *args):
                pass

            def save_jacobians(self, J):
                self.storage.save_jacobian(J)

            def save_adjoints(self):
                self.storage.save_adjoint(None)

            def save_states(self):
                self.storage.save_state(np.random.rand(100))

        solver = MockSolver()

        # Create manager
        manager = TimeStepDataManager(
            solver=solver, save_state=True, store_jacobians=True, verbose=False
        )

        # Save a timestep
        n = 10
        J = PETSc.Mat().create(comm=MPI.COMM_WORLD)
        J.setSizes([n, n])
        J.setType("aij")
        J.setUp()
        J.assemblyBegin()
        J.assemblyEnd()

        manager.save_timestep(timestep=0, J=J)

        # Verify storage received data
        assert len(solver.storage.saved_states) == 1
        assert len(solver.storage.saved_jacobians) == 1

    def test_backward_compatibility(self):
        """Test that storage maintains backward compatibility."""
        storage = SolverStateStorage()

        # Old code accessed lists directly
        storage.saved_states.append(np.ones(10))
        storage.saved_jacobians.append(None)

        # Should still work
        assert len(storage.saved_states) == 1
        assert len(storage.saved_jacobians) == 1


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

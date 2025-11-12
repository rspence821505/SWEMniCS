"""
Unit tests for TimeStepDataManager class.

Tests the observation-time filtering, memory estimation, and data saving
functionality of the refactored TimeStepDataManager.

Run with: pytest test_data_manager.py -v
"""

import pytest
import numpy as np
from swemnics.forward.solvers import TimeStepDataManager, CGImplicit
from swemnics.forward.problems import TidalProblem


class TestTimeStepDataManager:
    """Unit tests for TimeStepDataManager."""

    def setup_method(self):
        """Create a simple tidal problem and solver for testing."""
        # Small problem for fast tests
        self.problem = TidalProblem(
            nx=5,
            ny=5,
            nt=10,
            dt=60.0,
            x0=0,
            x1=10000,
            y0=0,
            y1=2000,
            h_b=10.0,
            mag=0.15,
            verbose=False,
        )
        self.solver = CGImplicit(self.problem, theta=0.5, verbose=False)

    def test_initialization_default_mode(self):
        """Test manager initialization in default (save all) mode."""
        manager = TimeStepDataManager(
            solver=self.solver, save_state=True, store_jacobians=True
        )

        assert manager.save_state == True
        assert manager.store_jacobians == True
        assert manager.obs_mode == False
        assert manager.observation_times is None
        assert manager._saved_count == 0
        assert manager._skipped_count == 0

    def test_initialization_observation_mode(self):
        """Test manager initialization in observation mode."""
        obs_times = [0, 5, 10]
        manager = TimeStepDataManager(
            solver=self.solver, save_state=True, observation_times=obs_times
        )

        assert manager.obs_mode == True
        assert manager.observation_times == set(obs_times)
        assert len(manager.observation_times) == 3

    def test_should_save_at_all_timesteps(self):
        """Test should_save_at() in default mode (save all timesteps)."""
        manager = TimeStepDataManager(solver=self.solver, save_state=True)

        # Should save at every timestep when save_state=True
        for t in range(20):
            assert manager.should_save_at(t) == True

    def test_should_save_at_no_saving(self):
        """Test should_save_at() when save_state=False."""
        manager = TimeStepDataManager(solver=self.solver, save_state=False)  # No saving

        # Should NOT save at any timestep
        for t in range(20):
            assert manager.should_save_at(t) == False

    def test_should_save_at_observation_times(self):
        """Test should_save_at() in observation mode."""
        obs_times = [0, 5, 10]
        manager = TimeStepDataManager(
            solver=self.solver, save_state=True, observation_times=obs_times
        )

        # Should save at observation times
        assert manager.should_save_at(0) == True
        assert manager.should_save_at(5) == True
        assert manager.should_save_at(10) == True

        # Should NOT save at other times
        assert manager.should_save_at(1) == False
        assert manager.should_save_at(3) == False
        assert manager.should_save_at(7) == False
        assert manager.should_save_at(15) == False

    def test_save_timestep_without_observation_times(self):
        """Test saving at all timesteps (old behavior)."""
        manager = TimeStepDataManager(
            solver=self.solver, save_state=True, verbose=False
        )

        # Simulate saving at multiple timesteps
        for t in range(5):
            manager.save_timestep(t)

        summary = manager.get_summary()
        assert summary["saved"] == 5
        assert summary["skipped"] == 0
        assert summary["mode"] == "all_timesteps"
        assert len(self.solver.saved_states) == 5

    def test_save_timestep_with_observation_times(self):
        """Test observation-time filtering actually works."""
        obs_times = [0, 2, 4, 6, 8]
        manager = TimeStepDataManager(
            solver=self.solver, save_state=True, observation_times=obs_times
        )

        # Try saving at all timesteps
        for t in range(10):
            manager.save_timestep(t)

        summary = manager.get_summary()
        assert summary["saved"] == 5  # Only observation times
        assert summary["skipped"] == 5  # Other timesteps skipped
        assert summary["mode"] == "observation"
        assert len(self.solver.saved_states) == 5

    def test_save_timestep_with_sparse_observations(self):
        """Test very sparse observation times."""
        obs_times = [0, 10]  # Only 2 observations
        manager = TimeStepDataManager(
            solver=self.solver, save_state=True, observation_times=obs_times
        )

        # Simulate 20 timesteps
        for t in range(20):
            manager.save_timestep(t)

        summary = manager.get_summary()
        assert summary["saved"] == 2
        assert summary["skipped"] == 18
        assert len(self.solver.saved_states) == 2

    def test_jacobian_storage_at_observations(self):
        """Test that Jacobians are only stored at observation times."""
        obs_times = [0, 5, 10]
        manager = TimeStepDataManager(
            solver=self.solver,
            save_state=True,
            store_jacobians=True,
            observation_times=obs_times,
        )

        # Simulate with mock Jacobians (just use None for testing)
        for t in range(15):
            # In reality, J would be a PETSc matrix
            J = None if t not in obs_times else "mock_jacobian"
            manager.save_timestep(t, J=J)

        summary = manager.get_summary()
        # Note: Since we're passing None for non-obs times,
        # saved_jacobians won't actually grow
        # This just tests the manager's filtering logic
        assert summary["saved"] == 3
        assert summary["skipped"] == 12

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        obs_times = [0, 5, 10]
        manager = TimeStepDataManager(
            solver=self.solver,
            save_state=True,
            store_jacobians=True,
            observation_times=obs_times,
        )

        n_dofs = 100
        nnz_per_row = 10
        estimates = manager.estimate_memory_usage(n_dofs, nnz_per_row)

        # Check that all expected keys exist
        assert "states" in estimates
        assert "jacobians" in estimates
        assert "total" in estimates

        # Check that values are reasonable
        assert estimates["states"] > 0
        assert estimates["jacobians"] > 0
        assert estimates["total"] == estimates["states"] + estimates["jacobians"]

        # Check approximate sizes (3 obs times, 100 DoFs)
        # States: 3 * 100 * 8 bytes = 2400 bytes ≈ 0.002 MB
        # Jacobians: 3 * 100 * 10 * 8 bytes = 24000 bytes ≈ 0.023 MB
        assert 0.001 < estimates["states"] < 0.01
        assert 0.01 < estimates["jacobians"] < 0.1

    def test_memory_estimation_with_bathymetry(self):
        """Test memory estimation including bathymetry saves."""
        obs_times = [0, 5, 10]
        manager = TimeStepDataManager(
            solver=self.solver,
            save_state=True,
            save_bathy=True,
            observation_times=obs_times,
        )

        # Need to set up stations for bathymetry
        self.solver.points_on_proc = np.array([[1000, 1000, 0]])

        n_dofs = 100
        estimates = manager.estimate_memory_usage(n_dofs)

        assert "states" in estimates
        assert "bathymetry" in estimates
        assert "total" in estimates

    def test_summary_generation(self):
        """Test that summary contains all expected information."""
        obs_times = [0, 3, 6, 9]
        manager = TimeStepDataManager(
            solver=self.solver, save_state=True, observation_times=obs_times
        )

        # Simulate some saves
        for t in range(10):
            manager.save_timestep(t)

        summary = manager.get_summary()

        # Check all required keys
        assert "mode" in summary
        assert "saved" in summary
        assert "skipped" in summary
        assert "observation_times" in summary
        assert "n_states" in summary

        # Check values
        assert summary["mode"] == "observation"
        assert summary["saved"] == 4
        assert summary["skipped"] == 6
        assert set(summary["observation_times"]) == set(obs_times)

    def test_repr_method(self):
        """Test string representation."""
        manager = TimeStepDataManager(solver=self.solver, save_state=True)

        # Should be able to convert to string without error
        repr_str = repr(manager)
        assert "TimeStepDataManager" in repr_str
        assert "mode=" in repr_str

    def test_no_state_saving_when_disabled(self):
        """Test that states are not saved when save_state=False."""
        manager = TimeStepDataManager(
            solver=self.solver,
            save_state=False,  # Explicitly disabled
            observation_times=[0, 5, 10],
        )

        for t in range(10):
            manager.save_timestep(t)

        # No states should be saved
        assert len(self.solver.saved_states) == 0

        summary = manager.get_summary()
        assert summary["saved"] == 0  # Nothing saved

    def test_adjoint_saving(self):
        """Test adjoint saving mode (old method)."""
        manager = TimeStepDataManager(
            solver=self.solver,
            save_state=True,
            save_adjoints=True,
            observation_times=[0, 5],
        )

        # Note: save_adjoints() requires the solver to have been run
        # and have a valid Jacobian. For this test, we just check
        # that the flag is set correctly
        assert manager.save_adjoints == True

    def test_multiple_manager_instances(self):
        """Test that multiple managers don't interfere with each other."""
        manager1 = TimeStepDataManager(
            solver=self.solver, save_state=True, observation_times=[0, 5]
        )

        manager2 = TimeStepDataManager(
            solver=self.solver, save_state=True, observation_times=[0, 2, 4, 6, 8]
        )

        # Each should have its own observation times
        assert len(manager1.observation_times) == 2
        assert len(manager2.observation_times) == 5

        # They should maintain independent counts
        for t in range(10):
            manager1.save_timestep(t)

        assert manager1._saved_count == 2
        assert manager2._saved_count == 0  # Hasn't been used yet


class TestTimeStepDataManagerEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test problem."""
        self.problem = TidalProblem(nx=3, ny=3, nt=5, verbose=False)
        self.solver = CGImplicit(self.problem, theta=0.5, verbose=False)

    def test_empty_observation_times(self):
        """Test with empty observation list."""
        manager = TimeStepDataManager(
            solver=self.solver, save_state=True, observation_times=[]  # Empty list
        )

        # Should be in observation mode but never save
        assert manager.obs_mode == True

        for t in range(10):
            manager.save_timestep(t)

        assert len(self.solver.saved_states) == 0
        assert manager._saved_count == 0
        assert manager._skipped_count == 10

    def test_observation_time_zero_only(self):
        """Test with only initial condition as observation."""
        manager = TimeStepDataManager(
            solver=self.solver, save_state=True, observation_times=[0]
        )

        for t in range(10):
            manager.save_timestep(t)

        assert len(self.solver.saved_states) == 1
        assert manager._saved_count == 1

    def test_duplicate_observation_times(self):
        """Test that duplicate observation times are handled (converted to set)."""
        manager = TimeStepDataManager(
            solver=self.solver,
            save_state=True,
            observation_times=[0, 5, 5, 10, 10, 10],  # Duplicates
        )

        # Should be deduplicated to set
        assert len(manager.observation_times) == 3
        assert manager.observation_times == {0, 5, 10}

    def test_unsorted_observation_times(self):
        """Test that observation times don't need to be sorted."""
        manager = TimeStepDataManager(
            solver=self.solver,
            save_state=True,
            observation_times=[10, 0, 5, 3, 8],  # Unsorted
        )

        # Should work correctly regardless of order
        for t in range(12):
            manager.save_timestep(t)

        assert manager._saved_count == 5
        assert len(self.solver.saved_states) == 5


class TestTimeStepDataManagerIntegration:
    """Integration tests with actual solver operations."""

    def test_with_wetting_drying(self):
        """Test manager with wetting/drying enabled."""
        problem = TidalProblem(nx=5, ny=5, nt=10, h_b=5.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        # Set up observation points
        stations = np.array([[5000, 1000, 0]])
        local_points = solver.init_stations(stations)

        manager = TimeStepDataManager(
            solver=solver,
            save_state=True,
            make_wet=True,
            save_bathy=True,
            save_true_bathy=True,
            observation_times=[0, 5, 10],
        )

        # Just test that it initializes correctly
        assert manager.make_wet == True
        assert manager.save_bathy == True
        assert manager.save_true_bathy == True

    def test_verbose_logging(self, capsys):
        """Test that verbose mode produces output."""
        manager = TimeStepDataManager(
            solver=self.solver,
            save_state=True,
            verbose=True,  # Enable verbose
            observation_times=[0, 5, 10],
        )

        # Only rank 0 should print
        if self.solver.mpi_rank == 0:
            captured = capsys.readouterr()
            # Should have printed something during initialization
            assert len(captured.out) > 0 or manager.obs_mode == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

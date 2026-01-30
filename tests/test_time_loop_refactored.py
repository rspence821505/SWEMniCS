"""
Integration tests for refactored time_loop() method.

Tests the complete time_loop functionality with TimeStepDataManager,
including backward compatibility, observation-time filtering, and
memory efficiency improvements.

Run with: pytest test_time_loop_refactored.py -v
"""

import pytest
import numpy as np
from swe4dvar.forward.solvers import CGImplicit, DGImplicit
from swe4dvar.forward.problems import TidalProblem, WellBalancedProblem


class TestTimeLoopBackwardCompatibility:
    """Test that refactored time_loop maintains backward compatibility."""

    def test_basic_time_loop_no_saving(self):
        """Test basic time loop without any saving."""
        problem = TidalProblem(nx=5, ny=5, nt=20, dt=60.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        # Basic time loop - no saving
        u, _ = solver.time_loop(solver_parameters={})

        # Should complete without saving anything
        assert len(solver.saved_states) == 0
        assert len(solver.saved_jacobians) == 0
        assert u is not None

    def test_backward_compatible_save_all_states(self):
        """Test that old usage (save_state=True) still works."""
        problem = TidalProblem(nx=5, ny=5, nt=20, dt=60.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        # Old usage pattern - save all states
        u, _ = solver.time_loop(
            solver_parameters={}, save_state=True, make_wet=False  # Old parameter
        )

        # Should have saved all timesteps (nt + 1 including initial)
        assert len(solver.saved_states) == 21  # 20 timesteps + initial
        assert u is not None

    def test_backward_compatible_with_wetting_drying(self):
        """Test old W/D usage pattern still works."""
        problem = TidalProblem(nx=5, ny=5, nt=10, dt=60.0, h_b=5.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        stations = np.array([[5000, 1000, 0]])

        # Old W/D pattern
        u, _ = solver.time_loop(
            solver_parameters={},
            stations=stations,
            save_state=True,
            make_wet=True,
            save_bathy=True,
            save_true_bathy=True,
        )

        # Should save all states with W/D modifications
        assert len(solver.saved_states) == 11
        assert len(solver.saved_bathy) > 0
        assert len(solver.saved_true_bathy) > 0
        assert len(solver.dry_nodes) > 0


class TestTimeLoopObservationMode:
    """Test new observation-time mode functionality."""

    def test_observation_mode_basic(self):
        """Test basic observation-time filtering."""
        problem = TidalProblem(nx=5, ny=5, nt=20, dt=60.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        # New usage with observation times
        obs_times = [0, 5, 10, 15, 20]
        u, _ = solver.time_loop(
            solver_parameters={},
            save_state=True,  # Ignored when obs_times provided
            observation_times=obs_times,  # NEW parameter
        )

        # Should only save at observation times
        assert len(solver.saved_states) == 5
        assert u is not None

    def test_observation_mode_sparse(self):
        """Test very sparse observations (high compression)."""
        problem = TidalProblem(nx=10, ny=10, nt=100, dt=60.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        # Very sparse observations - every 25 timesteps
        obs_times = [0, 25, 50, 75, 100]
        u, _ = solver.time_loop(solver_parameters={}, observation_times=obs_times)

        # Should only save 5 states instead of 101
        assert len(solver.saved_states) == 5

        # Memory savings: 101/5 = ~20x reduction
        full_save_count = 101
        actual_save_count = len(solver.saved_states)
        compression_ratio = full_save_count / actual_save_count
        assert compression_ratio > 15  # At least 15x savings

    def test_observation_mode_with_jacobians(self):
        """Test Jacobian storage at observation times only."""
        problem = TidalProblem(nx=5, ny=5, nt=20, dt=60.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        obs_times = [0, 10, 20]
        u, _ = solver.time_loop(
            solver_parameters={}, store_jacobians=True, observation_times=obs_times
        )

        # Should only store 3 Jacobians (at obs times)
        # Note: Actual count may vary depending on whether initial condition
        # gets a Jacobian, but should be close to len(obs_times)
        assert len(solver.saved_jacobians) <= len(obs_times)
        assert len(solver.saved_jacobians) > 0

    def test_observation_mode_first_and_last(self):
        """Test saving only first and last timesteps."""
        problem = TidalProblem(nx=5, ny=5, nt=50, dt=60.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        obs_times = [0, 50]  # Only initial and final
        u, _ = solver.time_loop(solver_parameters={}, observation_times=obs_times)

        assert len(solver.saved_states) == 2

    def test_observation_mode_with_wd(self):
        """Test observation mode combined with wetting/drying."""
        problem = TidalProblem(nx=5, ny=5, nt=20, dt=60.0, h_b=5.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        stations = np.array([[5000, 1000, 0]])
        obs_times = [0, 5, 10, 15, 20]

        u, _ = solver.time_loop(
            solver_parameters={},
            stations=stations,
            make_wet=True,
            save_bathy=True,
            save_true_bathy=True,
            observation_times=obs_times,
        )

        # Should save states only at observation times
        assert len(solver.saved_states) == 5

        # W/D data should also only be at observation times
        assert len(solver.saved_bathy) == 5
        assert len(solver.saved_true_bathy) == 5
        assert len(solver.dry_nodes) == 5


class TestTimeLoopMemorySavings:
    """Test memory efficiency improvements."""

    def test_memory_savings_states_only(self):
        """Test memory savings with state-only storage."""
        problem = TidalProblem(nx=10, ny=10, nt=100, dt=60.0, verbose=False)

        # Full save
        solver1 = CGImplicit(problem, theta=0.5, verbose=False)
        solver1.time_loop(solver_parameters={}, save_state=True)
        full_memory = len(solver1.saved_states)

        # Sparse save (10%)
        solver2 = CGImplicit(problem, theta=0.5, verbose=False)
        obs_times = list(range(0, 101, 10))  # Every 10th timestep
        solver2.time_loop(solver_parameters={}, observation_times=obs_times)
        sparse_memory = len(solver2.saved_states)

        # Should have ~10x less memory
        ratio = full_memory / sparse_memory
        assert ratio > 8  # At least 8x savings
        assert ratio < 12  # But not more than 12x

    def test_memory_savings_with_jacobians(self):
        """Test memory savings are even larger with Jacobians."""
        problem = TidalProblem(nx=5, ny=5, nt=50, dt=60.0, verbose=False)

        # Sparse observations (10%)
        solver = CGImplicit(problem, theta=0.5, verbose=False)
        obs_times = list(range(0, 51, 10))  # Every 10th

        solver.time_loop(
            solver_parameters={}, store_jacobians=True, observation_times=obs_times
        )

        # Jacobians stored should be much less than total timesteps
        jacobian_count = len(solver.saved_jacobians)
        total_timesteps = 51

        # Should be close to observation count
        assert jacobian_count <= len(obs_times)

        # Significant memory savings
        savings_ratio = total_timesteps / jacobian_count
        assert savings_ratio > 5  # At least 5x savings

    def test_no_memory_impact_when_disabled(self):
        """Test that observation_times=None has no memory overhead."""
        problem = TidalProblem(nx=5, ny=5, nt=20, dt=60.0, verbose=False)

        # Without observation_times (old behavior)
        solver1 = CGImplicit(problem, theta=0.5, verbose=False)
        solver1.time_loop(solver_parameters={})

        # With observation_times=None (should be identical)
        solver2 = CGImplicit(problem, theta=0.5, verbose=False)
        solver2.time_loop(solver_parameters={}, observation_times=None)

        # Should have same behavior
        assert len(solver1.saved_states) == len(solver2.saved_states)
        assert len(solver1.saved_jacobians) == len(solver2.saved_jacobians)


class TestTimeLoopWithDifferentSolvers:
    """Test time_loop works with different solver types."""

    def test_with_dg_solver(self):
        """Test observation mode with DG solver."""
        problem = WellBalancedProblem(nx=5, ny=5, nt=20, dt=1.0, verbose=False)
        solver = DGImplicit(problem, theta=0.5, p_degree=[1, 1], verbose=False)

        obs_times = [0, 10, 20]
        u, _ = solver.time_loop(solver_parameters={}, observation_times=obs_times)

        assert len(solver.saved_states) == 3
        assert u is not None

    def test_with_cg_solver_high_order(self):
        """Test with higher-order CG elements."""
        problem = TidalProblem(nx=5, ny=5, nt=10, dt=60.0, verbose=False)
        # Higher order elements
        solver = CGImplicit(problem, theta=0.5, p_degree=[2, 2], verbose=False)

        obs_times = [0, 5, 10]
        u, _ = solver.time_loop(solver_parameters={}, observation_times=obs_times)

        assert len(solver.saved_states) == 3


class TestTimeLoopEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_observation_list(self):
        """Test with empty observation list."""
        problem = TidalProblem(nx=5, ny=5, nt=10, dt=60.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        # Empty observation list - should save nothing
        u, _ = solver.time_loop(solver_parameters={}, observation_times=[])

        assert len(solver.saved_states) == 0
        assert u is not None

    def test_observation_at_every_timestep(self):
        """Test observation at every timestep (same as save_state=True)."""
        problem = TidalProblem(nx=5, ny=5, nt=10, dt=60.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        # Observation at every timestep
        obs_times = list(range(11))  # [0, 1, 2, ..., 10]
        u, _ = solver.time_loop(solver_parameters={}, observation_times=obs_times)

        # Should save all states (same as old behavior)
        assert len(solver.saved_states) == 11

    def test_observation_beyond_nt(self):
        """Test observation times beyond nt are ignored."""
        problem = TidalProblem(nx=5, ny=5, nt=10, dt=60.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        # Some observation times beyond nt
        obs_times = [0, 5, 10, 15, 20, 100]
        u, _ = solver.time_loop(solver_parameters={}, observation_times=obs_times)

        # Should only save at valid timesteps (0, 5, 10)
        assert len(solver.saved_states) == 3

    def test_negative_observation_times_ignored(self):
        """Test that negative observation times are handled."""
        problem = TidalProblem(nx=5, ny=5, nt=10, dt=60.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        # Include some negative times
        obs_times = [-5, 0, 5, 10]
        u, _ = solver.time_loop(solver_parameters={}, observation_times=obs_times)

        # Should only save at valid times (0, 5, 10)
        # Note: -5 won't be reached during time loop
        assert len(solver.saved_states) == 3


class TestTimeLoopForFourDVar:
    """Test time_loop for typical 4D-Var usage patterns."""

    def test_4dvar_typical_usage(self):
        """Test typical 4D-Var pattern: sparse obs + Jacobians."""
        problem = TidalProblem(nx=10, ny=10, nt=100, dt=60.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        # Typical 4D-Var: observations every hour (60 timesteps if dt=60s)
        obs_times = list(range(0, 101, 20))  # Every 20 timesteps

        stations = np.array([[5000, 1000, 0], [7500, 1000, 0]])

        u, _ = solver.time_loop(
            solver_parameters={"atol": 1e-8},
            stations=stations,
            store_jacobians=True,
            observation_times=obs_times,
        )

        # Should have sparse storage
        assert len(solver.saved_states) == len(obs_times)
        assert len(solver.saved_jacobians) <= len(obs_times)

        # Huge memory savings compared to full storage
        full_storage = 101
        actual_storage = len(solver.saved_states)
        assert full_storage / actual_storage > 4  # At least 4x savings

    def test_4dvar_with_wetting_drying(self):
        """Test 4D-Var with wetting/drying modifications."""
        problem = TidalProblem(nx=10, ny=10, nt=50, dt=60.0, h_b=5.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        stations = np.array([[5000, 1000, 0]])
        obs_times = list(range(0, 51, 10))

        u, _ = solver.time_loop(
            solver_parameters={},
            stations=stations,
            make_wet=True,
            save_bathy=True,
            save_true_bathy=True,
            store_jacobians=True,
            observation_times=obs_times,
        )

        # All data should be at observation times only
        assert len(solver.saved_states) == len(obs_times)
        assert len(solver.saved_bathy) == len(obs_times)
        assert len(solver.saved_true_bathy) == len(obs_times)
        assert len(solver.dry_nodes) == len(obs_times)

    def test_4dvar_adjoint_workflow_preparation(self):
        """Test that forward solve prepares data for adjoint."""
        problem = TidalProblem(nx=5, ny=5, nt=20, dt=60.0, verbose=False)
        solver = CGImplicit(problem, theta=0.5, verbose=False)

        obs_times = [0, 5, 10, 15, 20]

        # Forward solve with everything needed for adjoint
        u, _ = solver.time_loop(
            solver_parameters={},
            store_jacobians=True,
            save_state=True,
            observation_times=obs_times,
        )

        # Check that we have everything for adjoint computation
        assert len(solver.saved_states) == len(obs_times)
        assert len(solver.saved_jacobians) > 0

        # States and Jacobians should be synchronized
        # (may differ by 1 if initial condition doesn't have Jacobian)
        assert abs(len(solver.saved_states) - len(solver.saved_jacobians)) <= 1


class TestTimeLoopPerformance:
    """Performance-related tests."""

    def test_minimal_overhead_without_observations(self):
        """Test that observation_times=None adds no overhead."""
        import time

        problem = TidalProblem(nx=10, ny=10, nt=50, dt=60.0, verbose=False)

        # Time old behavior
        solver1 = CGImplicit(problem, theta=0.5, verbose=False)
        start1 = time.time()
        solver1.time_loop(solver_parameters={})
        time1 = time.time() - start1

        # Time new behavior with None
        solver2 = CGImplicit(problem, theta=0.5, verbose=False)
        start2 = time.time()
        solver2.time_loop(solver_parameters={}, observation_times=None)
        time2 = time.time() - start2

        # Should be approximately the same time (within 50% difference)
        # (allow wide margin for timing variability in tests)
        time_ratio = max(time1, time2) / min(time1, time2)
        assert time_ratio < 1.5, f"Performance regression: {time_ratio:.2f}x slower"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

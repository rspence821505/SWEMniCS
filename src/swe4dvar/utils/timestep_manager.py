"""Time-stepping data management for forward solves and data assimilation.

This module provides the TimeStepDataManager class which coordinates saving
of states, Jacobians, and other data during time integration loops.
"""

from typing import Optional, Sequence
from petsc4py import PETSc
import time


class TimeStepDataManager:
    """Centralized manager for saving timestep data during time loops.

    This class determines which timesteps require data storage and coordinates
    the saving of states, Jacobians, adjoints, and wetting/drying information.
    It supports both "save everything" and "observation-only" modes.
    """

    def __init__(
        self,
        solver,
        save_state: bool = False,
        make_wet: bool = False,
        save_bathy: bool = False,
        save_true_bathy: bool = False,
        store_jacobians: bool = False,
        store_parameter_derivatives: bool = False,
        save_adjoints: bool = False,
        observation_times: Optional[Sequence[int]] = None,
        verbose: bool = False,
    ):
        """Initialize the data manager.

        Args:
            solver: Reference to the solver object
            save_state: Whether to save state vectors
            make_wet: Apply wetting/drying adjustments before saving
            save_bathy: Save bathymetry at observation points
            save_true_bathy: Save true bathymetry and dry node indices
            store_jacobians: Store Jacobian matrices for 4D-Var
            save_adjoints: Save adjoint matrices
            observation_times: Optional sequence of timestep indices to save.
                If None, saves at all timesteps (when other flags are True).
            verbose: Enable verbose logging
        """
        self.solver = solver
        self.save_state = save_state
        self.make_wet = make_wet
        self.save_bathy = save_bathy
        self.save_true_bathy = save_true_bathy
        self.store_jacobians = store_jacobians
        self.store_parameter_derivatives = store_parameter_derivatives
        self.save_adjoints = save_adjoints
        self.verbose = verbose

        if observation_times is not None:
            self.observation_times = set(observation_times)
            self.obs_mode = True
        else:
            self.observation_times = None
            self.obs_mode = False

        self._saved_count = 0
        self._skipped_count = 0
        self._start_time = None

    def should_save_at(self, timestep: int) -> bool:
        """Return True if data should be saved at the given timestep.

        Args:
            timestep: Current timestep index (0-based)

        Returns:
            True if this timestep should trigger data saving
        """
        if self.observation_times is None:
            return (
                self.save_state
                or self.store_jacobians
                or self.store_parameter_derivatives
                or self.save_adjoints
            )
        return timestep in self.observation_times

    def save_timestep(
        self, timestep: int, J: Optional[PETSc.Mat] = None, local_points=None
    ):
        """Save configured data for the current timestep.

        Args:
            timestep: Current timestep index (0-based).
            J: Optional Jacobian (distributed PETSc.Mat, never gathered).
            local_points: Observation coordinates for wetting/drying logic.
        """
        if self._start_time is None:
            self._start_time = time.time()

        if not self.should_save_at(timestep):
            self._skipped_count += 1
            return

        # Track if anything was actually saved
        saved_anything = False

        if J is not None and self.store_jacobians:
            self.solver.save_jacobians(J)
            saved_anything = True

        # Residual derivatives dR_k/dtheta only exist for solved timesteps k >= 1.
        # Saving a synthetic row at timestep 0 misaligns the augmented adjoint
        # accumulation with lambda-history indexing.
        if (
            self.store_parameter_derivatives
            and timestep > 0
            and hasattr(self.solver, "save_parameter_derivatives")
        ):
            self.solver.save_parameter_derivatives()
            saved_anything = True

        if self.save_adjoints and timestep > 0:
            self.solver.save_adjoints()
            saved_anything = True

        if self.save_state:
            self._save_state_with_wd(local_points)
            saved_anything = True

        # Only increment saved count if something was actually saved
        if saved_anything:
            self._saved_count += 1

        # Disabled: This progress message is redundant with tqdm progress bar
        # if self.verbose and self.solver.mpi_rank == 0 and self._saved_count % 10 == 0:
        #     elapsed = time.time() - self._start_time
        #     rate = self._saved_count / elapsed if elapsed > 0 else 0.0
        #     self.solver.log(
        #         f"DataManager: Saved {self._saved_count} timesteps ({rate:.1f} saves/sec)"
        #     )

    def _save_state_with_wd(self, local_points):
        """Internal helper to save state with optional wetting/drying handling.

        Args:
            local_points: Local observation point coordinates for wetting/drying
        """
        if self.make_wet and local_points is not None and len(local_points) > 0:
            water_height, dry_node_indices = self.solver.check_dry_nodes(
                self.solver.u,
                local_points,
                save_bathy=self.save_bathy,
                save_true_bathy=self.save_true_bathy,
            )
            self.solver.save_states(
                water_height=water_height,
                dry_node_indices=dry_node_indices,
            )
        else:
            self.solver.save_states()

    def estimate_memory_usage(self, n_dofs: int, nnz_per_row: int = 10) -> dict:
        """Estimate memory usage for stored data (in MB).

        Args:
            n_dofs: Number of degrees of freedom per state
            nnz_per_row: Estimated nonzeros per row in Jacobian (default 10)

        Returns:
            Dictionary with memory estimates per component
        """
        bytes_per_float = 8
        n_saves = (
            len(self.observation_times) if self.obs_mode else self.solver.problem.nt
        )
        estimates: dict[str, float] = {}

        if self.save_state:
            state_mb = (n_saves * n_dofs * bytes_per_float) / (1024**2)
            estimates["states"] = state_mb

        if self.store_jacobians:
            jac_mb = (n_saves * n_dofs * nnz_per_row * bytes_per_float) / (1024**2)
            estimates["jacobians"] = jac_mb

        if self.save_adjoints:
            adj_mb = (n_saves * n_dofs * nnz_per_row * bytes_per_float) / (1024**2)
            estimates["adjoints"] = adj_mb

        if self.save_bathy:
            # Estimate bathymetry storage (assuming similar size to states)
            # This is a rough estimate based on number of observation points
            n_points = len(getattr(self.solver, 'points_on_proc', []))
            bathy_mb = (n_saves * n_points * bytes_per_float) / (1024**2)
            estimates["bathymetry"] = bathy_mb

        estimates["total"] = sum(estimates.values())
        return estimates

    def get_summary(self) -> dict:
        """Return summary statistics for saved data.

        Returns:
            Dictionary containing save statistics and storage counts
        """
        return {
            "saved": self._saved_count,
            "skipped": self._skipped_count,
            "mode": "observation" if self.obs_mode else "all_timesteps",
            "observation_times": (
                list(self.observation_times) if self.obs_mode else None
            ),
            "n_states": len(self.solver.saved_states),
            "n_jacobians": len(self.solver.saved_jacobians),
            "n_adjoints": len(self.solver.saved_adjoints),
            "n_parameter_derivative_steps": len(
                getattr(self.solver, "saved_parameter_derivatives", [])
            ),
            "n_dry_nodes": len(self.solver.dry_nodes),
        }

    def print_summary(self):
        """Print summary statistics to console (only from rank 0 in MPI)."""
        # Only print from rank 0 in parallel runs
        if self.solver.mpi_rank != 0:
            return

        summary = self.get_summary()
        print("\n" + "=" * 70)
        print("TimeStep Data Manager Summary")
        print("=" * 70)
        print(f"Mode:                {summary['mode']}")
        print(f"Timesteps saved:     {summary['saved']}")
        print(f"Timesteps skipped:   {summary['skipped']}")
        if summary['observation_times'] is not None:
            print(f"Observation times:   {len(summary['observation_times'])} timesteps")
        print(f"\nStored data:")
        print(f"  States:            {summary['n_states']}")
        print(f"  Jacobians:         {summary['n_jacobians']}")
        print(f"  Adjoints:          {summary['n_adjoints']}")
        print(f"  Parameter derivs:  {summary['n_parameter_derivative_steps']}")
        print(f"  Dry nodes:         {summary['n_dry_nodes']}")
        print("=" * 70)

    def __repr__(self) -> str:
        summary = self.get_summary()
        return (
            f"TimeStepDataManager(mode={summary['mode']}, "
            f"saved={summary['saved']}, skipped={summary['skipped']})"
        )

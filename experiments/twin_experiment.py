#!/usr/bin/env python3
"""
Generalized Twin Experiment Framework for Data Assimilation.

This module provides a reusable framework for running twin experiments
with any shallow water problem (TidalProblem, ADCIRCProblem, etc.).

Twin Experiment Overview:
1. Generate "truth" trajectory using forward model with known initial condition
2. Create synthetic observations by sampling truth + noise
3. Perturb initial condition to create "background" state
4. Run data assimilation to recover initial condition from observations
5. Evaluate results by comparing analysis to truth

Supported DA Methods:
- 4D-Var: Standard variational data assimilation
- DC-WME: Data-Consistent Weighted Mean Error 4D-Var

Usage:
    # As a library
    from twin_experiment import TwinExperiment

    experiment = TwinExperiment(
        problem=my_problem,
        solver=my_solver,
        method="4dvar",
        obs_fraction=0.5,
    )
    results = experiment.run()

    # As a command-line tool (see run_twin_experiment.py)
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from mpi4py import MPI
from petsc4py import PETSc

from swe4dvar.forward.solvers import get_solver
from swe4dvar.data_assimilation import (
    FourDVarCost,
    DCWMEFourDVarCost,
    DiagonalCovariance,
    PointObservationOperator,
)
from swe4dvar.utils import get_default_solver_params
from swe4dvar.utils.output_paths import DATA_DIR, ensure_output_dirs


@dataclass
class TwinExperimentConfig:
    """Configuration for a twin experiment."""

    # DA method
    method: str = "4dvar"  # "4dvar" or "dcwme"

    # Observation configuration
    obs_fraction: float = 0.5
    obs_frequency: int = 1
    obs_noise_level: float = 0.01
    obs_points_file: Optional[str] = None  # JSON file with pre-selected points
    interior_only: bool = True  # Only observe interior nodes

    # Background error configuration
    background_error_std: float = 0.1

    # Optimization configuration
    max_iterations: int = 50
    gradient_tolerance: float = 1e-6
    cost_tolerance: float = 1e-8
    use_bounds: bool = True
    h_min: float = 0.01

    # Covariance configuration
    component_aware_cov: bool = False

    # Output configuration
    output_dir: str = "outputs/data"
    save_trajectories: bool = False
    verbose: bool = True

    # Random seeds for reproducibility
    obs_seed: int = 42
    background_seed: int = 123

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class TwinExperimentResults:
    """Results from a twin experiment."""

    # Method and problem info
    method: str = ""
    problem_name: str = ""

    # Cost function history
    cost_history: List[float] = field(default_factory=list)
    gradient_norm_history: List[float] = field(default_factory=list)

    # Error metrics
    background_error: float = 0.0
    analysis_error: float = 0.0
    error_reduction: float = 0.0

    # Innovation statistics
    innovation_mean: float = 0.0
    innovation_std: float = 0.0

    # Runtime information
    num_iterations: int = 0
    converged: bool = False
    wall_time: float = 0.0

    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)

    # Problem-specific info
    problem_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def save(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "TwinExperimentResults":
        """Load results from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)


class ForwardModelWrapper:
    """
    Wrapper to adapt SWE4DVar solvers to the forward model interface
    expected by the cost functions.
    """

    def __init__(self, solver, problem, solver_params: dict):
        self.solver = solver
        self.problem = problem
        self.solver_params = solver_params
        self.dt = problem.dt
        self.nt = problem.nt
        self.comm = MPI.COMM_WORLD
        self.var_form = getattr(solver, "var_form", None)

    def solve(
        self, m: PETSc.Vec, store_jacobians: bool = True
    ) -> Tuple[List[PETSc.Vec], Optional[List]]:
        """Run forward model from initial condition m."""
        self.solver.storage.clear()

        # Set initial condition
        m_local = m.copy()
        m_array = m_local.getArray()

        u_owned_size = self.solver.V.dofmap.index_map.size_local
        if len(m_array) == len(self.solver.u_n.x.array):
            self.solver.u_n.x.array[:] = m_array
            self.solver.u_n_old.x.array[:] = m_array
            self.solver.u.x.array[:] = m_array
        elif len(m_array) == u_owned_size:
            self.solver.u_n.x.array[:u_owned_size] = m_array
            self.solver.u_n_old.x.array[:u_owned_size] = m_array
            self.solver.u.x.array[:u_owned_size] = m_array
            self.solver.u_n.x.scatter_forward()
            self.solver.u_n_old.x.scatter_forward()
            self.solver.u.x.scatter_forward()
        else:
            raise ValueError(
                f"Initial condition size {len(m_array)} does not match "
                f"solver DOFs (owned={u_owned_size}, total={len(self.solver.u_n.x.array)})"
            )
        m_local.destroy()

        self.problem.t = 0.0

        self.solver.time_loop(
            solver_parameters=self.solver_params,
            stations=np.array([[0.0, 0.0, 0.0]]),
            plot_every=9999,
            save_state=True,
            store_jacobians=store_jacobians,
            enable_video=False,
        )

        from dolfinx import la

        u_owned_size = self.solver.V.dofmap.index_map.size_local
        trajectory = []
        for state_array in self.solver.storage.saved_states:
            vec = la.create_petsc_vector(
                self.solver.V.dofmap.index_map,
                self.solver.V.dofmap.index_map_bs,
            )
            vec.setArray(state_array[:u_owned_size])
            vec.assemble()
            trajectory.append(vec)

        jacobians = None
        if store_jacobians and len(self.solver.storage.saved_jacobians) > 0:
            jacobians = self.solver.storage.saved_jacobians.copy()

        return trajectory, jacobians


class TwinExperiment:
    """
    Generalized twin experiment framework for data assimilation.

    This class encapsulates the entire twin experiment workflow and can
    work with any problem that inherits from BaseProblem/TidalProblem.

    Parameters
    ----------
    problem : BaseProblem
        The problem instance (TidalProblem, ADCIRCProblem, etc.).
    solver : Solver
        The solver instance (DGImplicit, SUPGImplicit, etc.).
    config : TwinExperimentConfig
        Experiment configuration.
    solver_params : dict, optional
        Solver parameters for Newton iterations.
    comm : MPI.Comm, optional
        MPI communicator.
    """

    def __init__(
        self,
        problem,
        solver,
        config: Optional[TwinExperimentConfig] = None,
        solver_params: Optional[dict] = None,
        comm: Optional[MPI.Comm] = None,
    ):
        self.problem = problem
        self.solver = solver
        self.config = config or TwinExperimentConfig()
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        # Default solver params if not provided
        self.solver_params = solver_params or get_default_solver_params(
            rtol=1e-5,
            atol=1e-6,
            max_it=10,
            relaxation_parameter=1.0,
            comm=self.comm,
            error_if_not_converged=True,
        )

        # Extract problem name
        self.problem_name = type(problem).__name__

        # Results storage
        self.truth_trajectory = None
        self.analysis_trajectory = None
        self.observations = None
        self.m_true = None
        self.m_background = None
        self.m_analysis = None

    def log(self, *args, **kwargs):
        """Print on rank 0 only."""
        if self.rank == 0 and self.config.verbose:
            print(*args, **kwargs)

    def run(self) -> TwinExperimentResults:
        """
        Run the complete twin experiment.

        Returns
        -------
        results : TwinExperimentResults
            Experiment results including error metrics and convergence history.
        """
        ensure_output_dirs()
        start_time = time.time()

        self._print_header()

        # Step 1: Generate truth trajectory
        self.log("\nStep 1: Generating truth trajectory...")
        self._generate_truth()

        # Step 2: Setup observations
        self.log("\nStep 2: Setting up observations...")
        obs_points, obs_operator, obs_times = self._setup_observations()

        # Step 3: Generate synthetic observations
        self.log("\nStep 3: Generating synthetic observations...")
        self.observations, obs_noise_stds = self._generate_observations(
            obs_operator, obs_times
        )

        # Step 4: Setup background state
        self.log("\nStep 4: Setting up background state...")
        background_error = self._setup_background()

        # Step 5: Setup covariance matrices
        self.log("\nStep 5: Setting up covariance matrices...")
        B, R = self._setup_covariances(obs_operator, obs_noise_stds)

        # Step 6: Create forward model wrapper
        self.log("\nStep 6: Creating forward model wrapper...")
        forward_model = self._create_forward_model()

        # Step 7: Setup cost function
        self.log(f"\nStep 7: Setting up {self.config.method.upper()} cost function...")
        cost_function = self._setup_cost_function(
            forward_model, obs_operator, B, R, obs_times
        )

        # Step 8: Run optimization
        self.log("\nStep 8: Running optimization...")
        optimizer, opt_time = self._run_optimization(cost_function)

        # Step 9: Evaluate results
        self.log("\nStep 9: Evaluating results...")
        analysis_error, error_reduction, innov_mean, innov_std = self._evaluate_results(
            obs_operator, obs_times, background_error
        )

        # Build results
        total_time = time.time() - start_time

        cost_history = [h["cost"] for h in optimizer.convergence_history]
        gradient_history = [h["grad_norm"] for h in optimizer.convergence_history]

        results = TwinExperimentResults(
            method=self.config.method,
            problem_name=self.problem_name,
            cost_history=cost_history,
            gradient_norm_history=gradient_history,
            background_error=background_error,
            analysis_error=analysis_error,
            error_reduction=error_reduction,
            innovation_mean=innov_mean,
            innovation_std=innov_std,
            num_iterations=optimizer.iteration,
            converged=optimizer.converged,
            wall_time=total_time,
            config=self.config.to_dict(),
            problem_config=self._get_problem_config(),
        )

        # Save results
        self._save_results(results)
        self._print_summary(results)

        # Cleanup
        self._cleanup()

        return results

    def _print_header(self):
        """Print experiment header."""
        if self.rank == 0:
            print("=" * 70)
            print(f"{self.problem_name} {self.config.method.upper()} Twin Experiment")
            print("=" * 70)
            print(f"MPI ranks: {self.comm.Get_size()}")
            print(f"Time step: {self.problem.dt} s")
            print(f"Number of time steps: {self.problem.nt}")
            print(f"Observation fraction: {self.config.obs_fraction}")
            print(f"Observation frequency: every {self.config.obs_frequency} timesteps")
            print(f"Noise level: {self.config.obs_noise_level}")
            print(f"Background error: {self.config.background_error_std}")
            print("=" * 70)

    def _generate_truth(self):
        """Generate truth trajectory by running forward model."""
        # Run forward model
        self.solver.time_loop(
            solver_parameters=self.solver_params,
            stations=np.array([[0.0, 0.0, 0.0]]),
            plot_every=9999,
            save_state=True,
            store_jacobians=True,
            enable_video=False,
            monitor_progress=(self.rank == 0 and self.config.verbose),
        )

        # Store truth trajectory
        self.truth_trajectory = []
        for state_array in self.solver.storage.saved_states:
            vec = PETSc.Vec().createWithArray(state_array.copy(), comm=self.comm)
            self.truth_trajectory.append(vec)

        # True initial condition
        self.m_true = self.truth_trajectory[0].copy()

        self.log(f"  Truth trajectory: {len(self.truth_trajectory)} states")

    def _setup_observations(self):
        """Setup observation points and operator."""
        if self.config.obs_points_file is not None:
            # Load from file
            with open(self.config.obs_points_file, "r") as f:
                obs_data = json.load(f)
            obs_points = np.array(obs_data["coordinates"])
            self.log(
                f"  Loaded {len(obs_points)} observation points from {self.config.obs_points_file}"
            )
        elif self.config.interior_only:
            # Generate interior-only points
            obs_points = self._generate_interior_observation_points()
            self.log(
                f"  Generated {len(obs_points)} interior observation points"
            )
        else:
            # Generate from all mesh nodes
            obs_points = self._generate_observation_points()
            self.log(f"  Generated {len(obs_points)} observation points")

        # Create observation operator
        obs_operator = PointObservationOperator(
            self.solver.V, obs_points, comm=self.comm
        )

        # Observation times
        obs_times = list(
            range(
                self.config.obs_frequency,
                self.problem.nt + 1,
                self.config.obs_frequency,
            )
        )
        self.log(
            f"  Observation times: {len(obs_times)} "
            f"(every {self.config.obs_frequency} timesteps)"
        )

        return obs_points, obs_operator, obs_times

    def _generate_observation_points(self) -> np.ndarray:
        """Generate random observation points from mesh nodes."""
        rng = np.random.default_rng(self.config.obs_seed)
        coords = self.problem.mesh.geometry.x
        n_points = coords.shape[0]
        n_obs = int(n_points * self.config.obs_fraction)
        indices = rng.choice(n_points, size=n_obs, replace=False)

        obs_points = np.zeros((n_obs, 3))
        obs_points[:, : coords.shape[1]] = coords[indices, :]
        return obs_points

    def _generate_interior_observation_points(
        self, boundary_tol: float = 1e-10
    ) -> np.ndarray:
        """Generate observation points from interior mesh nodes only."""
        rng = np.random.default_rng(self.config.obs_seed)
        coords = self.problem.mesh.geometry.x

        # Domain bounds
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        # Interior nodes mask
        interior_mask = (
            (coords[:, 0] > x_min + boundary_tol)
            & (coords[:, 0] < x_max - boundary_tol)
            & (coords[:, 1] > y_min + boundary_tol)
            & (coords[:, 1] < y_max - boundary_tol)
        )
        interior_indices = np.where(interior_mask)[0]

        if len(interior_indices) == 0:
            raise ValueError("No interior nodes found. Mesh may be too coarse.")

        n_obs = max(1, int(len(interior_indices) * self.config.obs_fraction))
        selected = rng.choice(
            len(interior_indices), size=min(n_obs, len(interior_indices)), replace=False
        )
        selected_indices = interior_indices[selected]

        obs_points = np.zeros((len(selected_indices), 3))
        obs_points[:, : coords.shape[1]] = coords[selected_indices, :]
        return obs_points

    def _generate_observations(self, obs_operator, obs_times):
        """Generate synthetic observations from truth trajectory."""
        rng = np.random.default_rng(self.config.obs_seed)
        observations = []
        noise_stds = []

        for k in obs_times:
            if k >= len(self.truth_trajectory):
                raise IndexError(
                    f"Observation time {k} exceeds trajectory length "
                    f"{len(self.truth_trajectory)}"
                )

            H_u = obs_operator.forward(self.truth_trajectory[k])
            H_u_array = H_u.getArray()

            signal_magnitude = np.abs(H_u_array).mean() + 1e-10
            noise_std = self.config.obs_noise_level * signal_magnitude
            noise_stds.append(noise_std)

            noise = rng.normal(0, noise_std, size=H_u_array.shape)
            noisy_obs = H_u_array + noise

            obs_vec = PETSc.Vec().createSeq(len(noisy_obs), comm=PETSc.COMM_SELF)
            obs_vec.setArray(noisy_obs)
            obs_vec.assemble()
            observations.append(obs_vec)

        self.log(f"  Observations generated with mean noise std: {np.mean(noise_stds):.6f}")
        return observations, np.array(noise_stds)

    def _setup_background(self) -> float:
        """Setup background state with perturbation from truth."""
        rng = np.random.default_rng(self.config.background_seed)
        truth_array = self.m_true.getArray()

        truth_magnitude = np.abs(truth_array).mean() + 1e-10
        error_magnitude = self.config.background_error_std * truth_magnitude
        perturbation = rng.normal(0, error_magnitude, size=truth_array.shape)

        self.m_background = self.m_true.duplicate()
        self.m_background.setArray(truth_array + perturbation)
        self.m_background.assemble()

        # Compute background error
        diff = self.m_background.copy()
        diff.axpy(-1.0, self.m_true)
        background_error = np.sqrt(diff.dot(diff) / diff.getSize())
        diff.destroy()

        self.log(f"  Background RMS error: {background_error:.6f}")
        return background_error

    def _setup_covariances(self, obs_operator, obs_noise_stds):
        """Setup background and observation covariance matrices."""
        state_size = self.m_true.getSize()
        truth_magnitude = np.abs(self.m_true.getArray()).mean()

        if self.config.component_aware_cov:
            # Component-aware variance
            h_var, uv_var = self._estimate_component_variances()
            B = self._create_component_aware_covariance(state_size, h_var, uv_var)
            self.log(f"  Background covariance: component-aware")
            self.log(f"    h variance: {h_var:.6e}, u/v variance: {uv_var:.6e}")
        else:
            # Uniform variance
            background_variance = (
                self.config.background_error_std * truth_magnitude
            ) ** 2
            B = DiagonalCovariance(self.comm, state_size, variance=background_variance)
            self.log(f"  Background covariance: diagonal, variance = {background_variance:.6e}")

        # Observation covariance
        n_obs = obs_operator.get_num_observations()
        obs_variance = obs_noise_stds.mean() ** 2
        R = DiagonalCovariance(self.comm, n_obs, variance=obs_variance)
        self.log(f"  Observation covariance: diagonal, variance = {obs_variance:.6e}")

        return B, R

    def _estimate_component_variances(self, n_vars: int = 3):
        """Estimate component-specific variances from truth state."""
        arr = self.m_true.getArray()
        h_values = arr[0::n_vars]
        uv_values = np.concatenate([arr[j::n_vars] for j in range(1, n_vars)])

        h_mag = np.abs(h_values).mean() + 1e-10
        uv_mag = np.abs(uv_values).mean() + 1e-10

        h_variance = (self.config.background_error_std * h_mag) ** 2
        uv_variance = (self.config.background_error_std * uv_mag) ** 2

        return h_variance, uv_variance

    def _create_component_aware_covariance(
        self, state_size: int, h_variance: float, velocity_variance: float, n_vars: int = 3
    ):
        """Create component-aware diagonal covariance."""
        variances = np.zeros(state_size)
        n_nodes = state_size // n_vars
        for i in range(n_nodes):
            variances[i * n_vars] = h_variance
            for j in range(1, n_vars):
                variances[i * n_vars + j] = velocity_variance

        return DiagonalCovariance(self.comm, state_size, diagonal=variances)

    def _create_forward_model(self):
        """Create forward model wrapper."""
        self.solver.storage.clear()
        self.problem.t = 0.0
        return ForwardModelWrapper(self.solver, self.problem, self.solver_params)

    def _setup_cost_function(self, forward_model, obs_operator, B, R, obs_times):
        """Setup the DA cost function."""
        if self.config.method == "4dvar":
            cost_function = FourDVarCost(
                forward_model=forward_model,
                observation_operator=obs_operator,
                background_cov=B,
                observation_cov=R,
                m_background=self.m_background,
                observations=self.observations,
                obs_times=obs_times,
                comm=self.comm,
            )
        elif self.config.method == "dcwme":
            cost_function = DCWMEFourDVarCost(
                forward_model=forward_model,
                observation_operator=obs_operator,
                background_cov=B,
                observation_cov=R,
                m_background=self.m_background,
                observations=self.observations,
                obs_times=obs_times,
                predicted_cov_wme=None,
                comm=self.comm,
            )
        else:
            raise ValueError(f"Unknown DA method: {self.config.method}")

        # Wrap with boundary gradient zeroing if needed
        if self.config.interior_only:
            boundary_dofs = self._get_boundary_dofs()
            cost_function = ZeroBoundaryGradientCost(cost_function, boundary_dofs)
            self.log(f"  Zeroing {len(boundary_dofs)} boundary DOF gradients")

        return cost_function

    def _get_boundary_dofs(self) -> np.ndarray:
        """Get DOF indices on domain boundary."""
        import dolfinx
        from dolfinx.mesh import locate_entities_boundary

        mesh = self.problem.mesh
        V = self.solver.V
        tdim = mesh.topology.dim
        fdim = tdim - 1

        def on_boundary(x):
            return np.full(x.shape[1], True)

        boundary_facets = locate_entities_boundary(mesh, fdim, on_boundary)
        return dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)

    def _run_optimization(self, cost_function):
        """Run the optimization."""
        from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper

        opt_options = {
            "max_iterations": self.config.max_iterations,
            "gradient_tolerance": self.config.gradient_tolerance,
            "cost_tolerance": self.config.cost_tolerance,
            "verbose": (self.rank == 0),
        }

        if self.config.use_bounds:
            lower, upper = self._create_physical_bounds()
            tao_type = "blmvm"
            self.log(f"  Using bounded L-BFGS (h_min={self.config.h_min})")
        else:
            lower, upper = None, None
            tao_type = "lmvm"
            self.log("  Using unbounded L-BFGS")

        optimizer = PETScTAOWrapper(
            cost_function,
            tao_type=tao_type,
            lower_bounds=lower,
            upper_bounds=upper,
            options=opt_options,
        )

        opt_start = time.time()
        self.m_analysis = optimizer.solve(self.m_background.copy())
        opt_time = time.time() - opt_start

        self.log(f"\n  Optimization completed in {opt_time:.2f} seconds")
        self.log(f"  Iterations: {optimizer.iteration}")
        self.log(f"  Converged: {optimizer.converged}")

        return optimizer, opt_time

    def _create_physical_bounds(self, n_vars: int = 3):
        """Create physical bounds for the control variable."""
        lower = self.m_background.duplicate()
        upper = self.m_background.duplicate()

        lower_array = lower.getArray()
        upper_array = upper.getArray()

        n_dofs = len(lower_array)
        n_nodes = n_dofs // n_vars

        for i in range(n_nodes):
            lower_array[i * n_vars] = self.config.h_min
            upper_array[i * n_vars] = 1e10
            for j in range(1, n_vars):
                lower_array[i * n_vars + j] = -1e10
                upper_array[i * n_vars + j] = 1e10

        lower.setArray(lower_array)
        upper.setArray(upper_array)
        return lower, upper

    def _evaluate_results(self, obs_operator, obs_times, background_error):
        """Evaluate analysis results."""
        # Analysis error
        diff = self.m_analysis.copy()
        diff.axpy(-1.0, self.m_true)
        analysis_error = np.sqrt(diff.dot(diff) / diff.getSize())
        diff.destroy()

        error_reduction = (background_error - analysis_error) / background_error * 100

        self.log(f"  Analysis RMS error: {analysis_error:.6f}")
        self.log(f"  Error reduction: {error_reduction:.1f}%")

        # Run analysis forward for innovation statistics
        self.solver.storage.clear()
        self.problem.t = 0.0

        m_analysis_array = self.m_analysis.getArray()
        u_owned_size = self.solver.V.dofmap.index_map.size_local

        if len(m_analysis_array) == len(self.solver.u_n.x.array):
            self.solver.u_n.x.array[:] = m_analysis_array
            self.solver.u_n_old.x.array[:] = m_analysis_array
            self.solver.u.x.array[:] = m_analysis_array
        else:
            self.solver.u_n.x.array[:u_owned_size] = m_analysis_array
            self.solver.u_n_old.x.array[:u_owned_size] = m_analysis_array
            self.solver.u.x.array[:u_owned_size] = m_analysis_array
            self.solver.u_n.x.scatter_forward()
            self.solver.u_n_old.x.scatter_forward()
            self.solver.u.x.scatter_forward()

        self.solver.time_loop(
            solver_parameters=self.solver_params,
            stations=np.array([[0.0, 0.0, 0.0]]),
            plot_every=9999,
            save_state=True,
            enable_video=False,
        )

        self.analysis_trajectory = []
        for state_array in self.solver.storage.saved_states:
            vec = PETSc.Vec().createWithArray(state_array.copy(), comm=self.comm)
            self.analysis_trajectory.append(vec)

        # Innovation statistics
        all_innovations = []
        for i, k in enumerate(obs_times):
            H_u = obs_operator.forward(self.analysis_trajectory[k])
            H_u_array = H_u.getArray()
            y_array = self.observations[i].getArray()
            innovation = y_array - H_u_array
            all_innovations.extend(innovation.tolist())

        all_innovations = np.array(all_innovations)
        innov_mean = float(np.mean(all_innovations))
        innov_std = float(np.std(all_innovations))

        self.log(f"  Innovation mean: {innov_mean:.6f}")
        self.log(f"  Innovation std: {innov_std:.6f}")

        return analysis_error, error_reduction, innov_mean, innov_std

    def _get_problem_config(self) -> Dict[str, Any]:
        """Extract problem configuration for saving."""
        config = {
            "problem_type": self.problem_name,
            "dt": self.problem.dt,
            "nt": self.problem.nt,
        }

        # Add problem-specific attributes
        if hasattr(self.problem, "nx"):
            config["nx"] = self.problem.nx
        if hasattr(self.problem, "ny"):
            config["ny"] = self.problem.ny
        if hasattr(self.problem, "adios_file"):
            config["adios_file"] = self.problem.adios_file

        return config

    def _save_results(self, results: TwinExperimentResults):
        """Save experiment results."""
        if self.rank == 0:
            output_path = Path(self.config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            filename = f"{self.problem_name.lower()}_{self.config.method}_results.json"
            filepath = output_path / filename
            results.save(str(filepath))
            self.log(f"\nResults saved to: {filepath}")

    def _print_summary(self, results: TwinExperimentResults):
        """Print experiment summary."""
        if self.rank == 0:
            print("\n" + "=" * 70)
            print(f"SUMMARY: {self.problem_name} {self.config.method.upper()} Experiment")
            print("=" * 70)
            print(f"Background error:  {results.background_error:.6f}")
            print(f"Analysis error:    {results.analysis_error:.6f}")
            print(f"Error reduction:   {results.error_reduction:.1f}%")
            print(f"Iterations:        {results.num_iterations}")
            print(f"Converged:         {results.converged}")
            print(f"Total time:        {results.wall_time:.2f} s")
            print("=" * 70)

    def _cleanup(self):
        """Cleanup PETSc vectors."""
        if self.truth_trajectory:
            for vec in self.truth_trajectory:
                vec.destroy()
        if self.analysis_trajectory:
            for vec in self.analysis_trajectory:
                vec.destroy()
        if self.observations:
            for vec in self.observations:
                vec.destroy()
        if self.m_true:
            self.m_true.destroy()
        if self.m_background:
            self.m_background.destroy()
        if self.m_analysis:
            self.m_analysis.destroy()


class ZeroBoundaryGradientCost:
    """Wrapper that zeros gradient at boundary DOFs."""

    def __init__(self, base_cost, boundary_dofs: np.ndarray):
        self.base_cost = base_cost
        self.boundary_dofs = boundary_dofs

    def value(self, m: PETSc.Vec) -> float:
        return self.base_cost.value(m)

    def gradient(self, m: PETSc.Vec) -> PETSc.Vec:
        grad = self.base_cost.gradient(m)
        grad_arr = grad.getArray()
        grad_arr[self.boundary_dofs] = 0.0
        grad.setArray(grad_arr)
        return grad

    def value_gradient(self, m: PETSc.Vec):
        cost, grad = self.base_cost.value_gradient(m)
        grad_arr = grad.getArray()
        grad_arr[self.boundary_dofs] = 0.0
        grad.setArray(grad_arr)
        return cost, grad

    def clear_cache(self):
        if hasattr(self.base_cost, "clear_cache"):
            self.base_cost.clear_cache()


# Problem factory for CLI
PROBLEM_REGISTRY = {}


def register_problem(name: str, factory: Callable):
    """Register a problem factory for CLI usage."""
    PROBLEM_REGISTRY[name] = factory


def create_problem(name: str, **kwargs):
    """Create a problem instance by name."""
    if name not in PROBLEM_REGISTRY:
        raise ValueError(
            f"Unknown problem: {name}. Available: {list(PROBLEM_REGISTRY.keys())}"
        )
    return PROBLEM_REGISTRY[name](**kwargs)


# Register built-in problems
def _register_builtin_problems():
    """Register built-in problem types."""
    from swe4dvar.forward.problems import TidalProblem, DamProblem

    def create_tidal(**kwargs):
        return TidalProblem(**kwargs)

    def create_dam(**kwargs):
        return DamProblem(**kwargs)

    register_problem("tidal", create_tidal)
    register_problem("dam", create_dam)
    register_problem("dam_break", create_dam)

    # Try to register ADCIRC problem
    try:
        from swe4dvar.forward.adcirc_problem import ADCIRCProblem

        def create_adcirc(**kwargs):
            return ADCIRCProblem(**kwargs)

        register_problem("adcirc", create_adcirc)
        register_problem("shinnecock", create_adcirc)
    except ImportError:
        pass


_register_builtin_problems()

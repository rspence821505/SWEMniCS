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

    # Physics perturbation for inverse crime avoidance
    # Bathymetry perturbation
    perturb_bathymetry: bool = False
    bathymetry_noise_std: float = 0.5  # Absolute (m) for additive, relative for multiplicative
    bathymetry_noise_type: str = "additive"  # "additive" or "multiplicative"
    bathymetry_correlation_length: float = 500.0  # meters

    # Friction perturbation
    perturb_friction: bool = False
    friction_scale_factor: float = 1.0  # α: friction_perturbed = friction_true * α

    # Perturbation seed for reproducibility
    perturbation_seed: int = 456

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
        # Force direct solver (LU) for stability - GMRES+ILU can diverge
        # on larger meshes or with physics perturbations
        self.solver_params = solver_params or get_default_solver_params(
            rtol=1e-5,
            atol=1e-6,
            max_it=10,
            relaxation_parameter=1.0,
            comm=self.comm,
            error_if_not_converged=True,
            # Override to use direct solver for stability
            ksp_type="preonly",
            pc_type="lu",
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

        # Step 1: Generate truth trajectory (with TRUE parameters)
        self.log("\nStep 1: Generating truth trajectory...")
        self._generate_truth()

        # Step 1b: Apply physics perturbations (for inverse crime avoidance)
        if self.config.perturb_bathymetry or self.config.perturb_friction:
            self.log("\nStep 1b: Applying physics perturbations...")
            self._apply_physics_perturbations()

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

        # Build results
        total_time = time.time() - start_time

        cost_history = [h["cost"] for h in optimizer.convergence_history]
        gradient_history = [h["grad_norm"] for h in optimizer.convergence_history]

        # Step 9: Evaluate results
        self.log("\nStep 9: Evaluating results...")
        # If optimization failed (cost returned 1e20), skip forward evaluation
        if cost_history and cost_history[-1] >= 1e19:
            self.log("  Skipping forward evaluation due to optimization failure")
            analysis_error = background_error
            error_reduction = 0.0
            innov_mean = 0.0
            innov_std = 0.0
            self.analysis_trajectory = None
        else:
            analysis_error, error_reduction, innov_mean, innov_std = self._evaluate_results(
                obs_operator, obs_times, background_error
            )

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

            # Physics perturbation info
            if self.config.perturb_bathymetry or self.config.perturb_friction:
                print("-" * 70)
                print("Physics Perturbation (Inverse Crime Avoidance):")
                if self.config.perturb_bathymetry:
                    print(f"  Bathymetry: {self.config.bathymetry_noise_type} noise, "
                          f"std={self.config.bathymetry_noise_std}, "
                          f"corr_len={self.config.bathymetry_correlation_length}m")
                if self.config.perturb_friction:
                    print(f"  Friction: scale factor = {self.config.friction_scale_factor}")
            else:
                print("-" * 70)
                print("Physics Perturbation: None (inverse crime)")

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

    def _apply_physics_perturbations(self):
        """
        Apply physics perturbations to avoid inverse crime.

        This modifies the problem's bathymetry and/or friction parameters
        AFTER truth generation, so the DA uses perturbed parameters while
        the truth was generated with the original parameters.

        For bathymetry perturbation with solution_var="h", we must also update
        the solver's initial condition to maintain physical consistency:
          h_new = h_old + (h_b_perturbed - h_b_old)
        This ensures eta (surface elevation) remains unchanged.
        """
        rng = np.random.default_rng(self.config.perturbation_seed)

        if self.config.perturb_bathymetry:
            # Store old bathymetry values before perturbation
            old_h_b_values = self._get_bathymetry_values()
            self._perturb_bathymetry(rng)
            new_h_b_values = self._get_bathymetry_values()

            # For solution_var="h", update initial condition to maintain eta consistency
            if hasattr(self.problem, 'solution_var') and self.problem.solution_var == "h":
                self._adjust_initial_condition_for_bathymetry(old_h_b_values, new_h_b_values)

        if self.config.perturb_friction:
            self._perturb_friction()

    def _get_bathymetry_values(self) -> np.ndarray:
        """Get current bathymetry values as an array."""
        h_b = self.problem.h_b
        if hasattr(h_b, 'x'):
            return h_b.x.array.copy()
        elif hasattr(h_b, 'value'):
            # Constant - return scalar as array
            return np.array([float(h_b.value)])
        else:
            # Numeric value
            return np.array([float(h_b)])

    def _adjust_initial_condition_for_bathymetry(self, old_h_b: np.ndarray, new_h_b: np.ndarray):
        """
        Adjust solver initial condition when bathymetry is perturbed for solution_var="h".

        For solution_var="h", the state variable is total water depth h = h_b + eta.
        When bathymetry changes, we need to adjust h to keep eta consistent:
          h_new = h_old + (h_b_new - h_b_old)

        This also clears cached boundary conditions that depend on bathymetry.
        """
        self.log("  Adjusting initial condition for bathymetry change (solution_var=h)...")

        # Get the delta in bathymetry
        if len(old_h_b) == 1 and len(new_h_b) == 1:
            # Constant bathymetry
            delta_h_b = new_h_b[0] - old_h_b[0]
            self.log(f"  Uniform bathymetry change: {delta_h_b:+.3f}m")
        else:
            # Spatially varying bathymetry
            delta_h_b = new_h_b - old_h_b
            self.log(f"  Bathymetry change: min={delta_h_b.min():+.3f}m, max={delta_h_b.max():+.3f}m, "
                     f"mean={delta_h_b.mean():+.3f}m")

        # Update solver state variables (u_n, u_n_old, u)
        # For mixed elements, h is the first component
        n_vars = 3  # h, ux, uy

        for state_var in [self.solver.u_n, self.solver.u_n_old, self.solver.u]:
            arr = state_var.x.array
            n_dofs = len(arr)
            n_nodes = n_dofs // n_vars

            if isinstance(delta_h_b, np.ndarray) and len(delta_h_b) > 1:
                # Spatially varying - delta_h_b aligns with mesh nodes
                # But DOFs might be interlaced differently
                # For CG elements, DOFs are typically grouped by component
                h_dofs = np.arange(0, n_dofs, n_vars)  # Every n_vars DOF is h
                if len(delta_h_b) == n_nodes:
                    arr[h_dofs] += delta_h_b
                else:
                    # Size mismatch - use mean
                    self.log(f"  Warning: Bathymetry array size {len(delta_h_b)} != n_nodes {n_nodes}, using mean")
                    arr[h_dofs] += delta_h_b.mean()
            else:
                # Uniform change
                h_dofs = np.arange(0, n_dofs, n_vars)
                arr[h_dofs] += float(np.mean(delta_h_b))

            state_var.x.scatter_forward()

        # Update h_init if it exists
        if hasattr(self.problem, 'h_init') and self.problem.h_init is not None:
            h_init = self.problem.h_init
            if hasattr(h_init, 'x'):
                if isinstance(delta_h_b, np.ndarray) and len(delta_h_b) == len(h_init.x.array):
                    h_init.x.array[:] += delta_h_b
                else:
                    h_init.x.array[:] += float(np.mean(delta_h_b))
                h_init.x.scatter_forward()

        # Clear cached boundary values that depend on bathymetry
        if hasattr(self.problem, '_hb_boundary'):
            delattr(self.problem, '_hb_boundary')
            self.log("  Cleared cached boundary values")

        # Update boundary conditions
        if hasattr(self.problem, 'update_boundary'):
            self.problem.update_boundary()
            self.log("  Updated boundary conditions")

    def _perturb_bathymetry(self, rng: np.random.Generator):
        """
        Perturb the bathymetry field.

        For additive noise: b_perturbed = b_true + noise
        For multiplicative noise: b_perturbed = b_true * (1 + noise)

        If h_b is a Constant, only uniform perturbation is applied (mean of noise field).
        For spatially-varying perturbation, h_b must be a Function.
        """
        from dolfinx import fem as fe

        # Get the bathymetry field
        if not hasattr(self.problem, 'h_b') or self.problem.h_b is None:
            self.log("  Warning: No bathymetry field found, skipping perturbation")
            return

        h_b = self.problem.h_b

        # Check if h_b is a Function or a Constant/Expression
        if hasattr(h_b, 'x'):
            # It's a Function - apply spatially-varying perturbation
            n_points = len(h_b.x.array)

            # Generate smooth noise field
            noise = self._generate_smooth_noise(
                rng,
                n_points,
                self.config.bathymetry_noise_std,
                self.config.bathymetry_correlation_length,
            )

            if self.config.bathymetry_noise_type == "additive":
                h_b.x.array[:] += noise
                self.log(f"  Applied additive bathymetry noise (std={self.config.bathymetry_noise_std}m)")
            elif self.config.bathymetry_noise_type == "multiplicative":
                h_b.x.array[:] *= (1.0 + noise)
                self.log(f"  Applied multiplicative bathymetry noise (std={self.config.bathymetry_noise_std*100:.1f}%)")
            else:
                raise ValueError(f"Unknown bathymetry noise type: {self.config.bathymetry_noise_type}")

            # Ensure bathymetry stays positive (for stability)
            min_depth = self.config.h_min
            h_b.x.array[h_b.x.array < min_depth] = min_depth

            # Log statistics
            self.log(f"  Perturbed bathymetry: min={h_b.x.array.min():.2f}m, "
                     f"max={h_b.x.array.max():.2f}m, mean={h_b.x.array.mean():.2f}m")

            # Scatter forward to ensure consistency across processes
            h_b.x.scatter_forward()

        elif hasattr(h_b, 'value'):
            # It's a Constant - apply uniform perturbation
            # Generate a single random perturbation value
            original_value = float(h_b.value)

            if self.config.bathymetry_noise_type == "additive":
                # Sample from normal distribution
                perturbation = rng.normal(0, self.config.bathymetry_noise_std)
                new_value = original_value + perturbation
                self.log(f"  Bathymetry is Constant: applying uniform additive perturbation")
                self.log(f"  Original: {original_value:.2f}m, Perturbation: {perturbation:+.3f}m, "
                         f"New: {new_value:.2f}m")
            elif self.config.bathymetry_noise_type == "multiplicative":
                perturbation = rng.normal(0, self.config.bathymetry_noise_std)
                new_value = original_value * (1.0 + perturbation)
                self.log(f"  Bathymetry is Constant: applying uniform multiplicative perturbation")
                self.log(f"  Original: {original_value:.2f}m, Factor: {1.0 + perturbation:.3f}, "
                         f"New: {new_value:.2f}m")
            else:
                raise ValueError(f"Unknown bathymetry noise type: {self.config.bathymetry_noise_type}")

            # Ensure bathymetry stays positive
            new_value = max(new_value, self.config.h_min)
            h_b.value = new_value

        else:
            # It's a UFL expression or something we can't handle
            self.log("  Warning: Bathymetry is a UFL expression, cannot perturb")
            self.log("  Consider using a Function or Constant for bathymetry to enable perturbation")

    def _perturb_friction(self):
        """
        Perturb the friction coefficient by uniform scaling.

        friction_perturbed = friction_true * scale_factor

        Handles various friction representations:
        - TAU_const as Constant (dolfinx Constant)
        - TAU_const as float (set during make_Friction)
        - mannings_n as Function (ADCIRCProblem)
        - TAU as Constant or Function
        """
        from dolfinx import fem as fe
        from petsc4py.PETSc import ScalarType

        alpha = self.config.friction_scale_factor

        if alpha == 1.0:
            self.log("  Friction scale factor is 1.0, no perturbation applied")
            return

        # Check for different friction representations
        friction_perturbed = False

        # Case 1: TAU_const - should be a Constant after our fix to get_friction
        if hasattr(self.problem, 'TAU_const') and self.problem.TAU_const is not None:
            tau_const = self.problem.TAU_const

            if hasattr(tau_const, 'value'):
                # It's a dolfinx Constant
                original = float(tau_const.value)
                tau_const.value = original * alpha
                self.log(f"  Scaled TAU_const: {original:.6f} -> {original * alpha:.6f} (factor={alpha})")
                friction_perturbed = True
            elif isinstance(tau_const, (int, float)):
                # It's a plain number - create a new Constant
                # Note: This won't affect the weak form that was already compiled
                original = float(tau_const)
                new_value = original * alpha
                self.problem.TAU_const = fe.Constant(self.problem.mesh, ScalarType(new_value))
                self.log(f"  Warning: TAU_const was a float, converted to Constant: {original:.6f} -> {new_value:.6f}")
                self.log(f"  Note: This may not affect an already-compiled weak form")
                friction_perturbed = True

        # Case 2: mannings_n (Function) - used by ADCIRCProblem
        if hasattr(self.problem, 'mannings_n') and self.problem.mannings_n is not None:
            if hasattr(self.problem.mannings_n, 'x'):
                original_mean = np.mean(self.problem.mannings_n.x.array)
                self.problem.mannings_n.x.array[:] *= alpha
                new_mean = np.mean(self.problem.mannings_n.x.array)
                self.log(f"  Scaled Manning's n: mean {original_mean:.6f} -> {new_mean:.6f} (factor={alpha})")
                friction_perturbed = True

        # Case 3: TAU (could be a constant, function, or float)
        if hasattr(self.problem, 'TAU') and self.problem.TAU is not None and not friction_perturbed:
            tau = self.problem.TAU

            if hasattr(tau, 'x'):
                # It's a Function
                original_mean = np.mean(tau.x.array)
                tau.x.array[:] *= alpha
                new_mean = np.mean(tau.x.array)
                self.log(f"  Scaled TAU (Function): mean {original_mean:.6f} -> {new_mean:.6f} (factor={alpha})")
                friction_perturbed = True
            elif hasattr(tau, 'value'):
                # It's a Constant
                original = float(tau.value)
                tau.value = original * alpha
                self.log(f"  Scaled TAU (Constant): {original:.6f} -> {original * alpha:.6f} (factor={alpha})")
                friction_perturbed = True
            elif isinstance(tau, (int, float)):
                # It's a plain number - create a new Constant
                original = float(tau)
                new_value = original * alpha
                self.problem.TAU = fe.Constant(self.problem.mesh, ScalarType(new_value))
                self.log(f"  Warning: TAU was a float, converted to Constant: {original:.6f} -> {new_value:.6f}")
                friction_perturbed = True

        if not friction_perturbed:
            self.log("  Warning: No friction field found to perturb")

    def _generate_smooth_noise(
        self,
        rng: np.random.Generator,
        n_points: int,
        std: float,
        correlation_length: float,
    ) -> np.ndarray:
        """
        Generate spatially correlated noise field.

        Uses a simple moving average approach for smoothing.
        For more sophisticated applications, consider using
        Gaussian processes or FFT-based methods.
        """
        coords = self.problem.mesh.geometry.x[:, :2]  # x, y coordinates

        if len(coords) != n_points:
            # Fall back to white noise if coordinate mismatch
            self.log(f"  Warning: Coordinate size mismatch, using white noise")
            return rng.normal(0, std, n_points)

        # Generate raw noise
        raw_noise = rng.normal(0, std, n_points)

        # Simple distance-weighted smoothing using a Gaussian kernel
        # This is O(n^2) but acceptable for moderate mesh sizes
        # For large meshes, consider using scipy.spatial.KDTree for efficiency

        if n_points > 10000:
            # For large meshes, use a KDTree-based approach
            from scipy.spatial import KDTree

            tree = KDTree(coords)
            smoothed_noise = np.zeros(n_points)

            # Query neighbors within correlation_length
            for i in range(n_points):
                # Find all points within 2*correlation_length
                neighbors = tree.query_ball_point(coords[i], 2 * correlation_length)
                if len(neighbors) > 0:
                    distances = np.linalg.norm(coords[neighbors] - coords[i], axis=1)
                    weights = np.exp(-distances**2 / (2 * correlation_length**2))
                    weights /= weights.sum()
                    smoothed_noise[i] = np.dot(weights, raw_noise[neighbors])
                else:
                    smoothed_noise[i] = raw_noise[i]
        else:
            # For smaller meshes, compute full distance matrix
            from scipy.spatial.distance import cdist

            dist_matrix = cdist(coords, coords)
            weights = np.exp(-dist_matrix**2 / (2 * correlation_length**2))
            weights /= weights.sum(axis=1, keepdims=True)
            smoothed_noise = weights @ raw_noise

        # Rescale to match desired standard deviation
        current_std = np.std(smoothed_noise)
        if current_std > 0:
            smoothed_noise = smoothed_noise * (std / current_std)

        return smoothed_noise

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

    def _get_component_dof_indices(self):
        """Get DOF indices for each component (h, u, v) using function space structure.

        Returns tuple of (h_indices, u_indices, v_indices) as numpy arrays.
        """
        V = self.solver.V
        n_subspaces = V.ufl_element().num_sub_elements

        if n_subspaces == 2:
            # Mixed element (h, [u,v]) - common for DG/SUPG
            h_dofs_sub = V.sub(0).dofmap.list.flatten()
            uv_dofs_sub = V.sub(1).dofmap.list.flatten()

            # uv has block size 2 (u, v interleaved)
            # Get unique sorted indices
            h_indices = np.unique(h_dofs_sub)
            uv_indices = np.unique(uv_dofs_sub)

            # Split uv into u and v (they're interleaved with block size 2)
            u_indices = uv_indices[0::2]
            v_indices = uv_indices[1::2]

            return h_indices, u_indices, v_indices
        else:
            # Fallback: assume interleaved (h, u, v)
            n_dofs = len(self.solver.u.x.array)
            h_indices = np.arange(0, n_dofs, 3)
            u_indices = np.arange(1, n_dofs, 3)
            v_indices = np.arange(2, n_dofs, 3)
            return h_indices, u_indices, v_indices

    def _setup_background(self, n_vars: int = 3) -> float:
        """Setup background state with component-aware perturbation from truth.

        Uses different perturbation magnitudes for h vs u,v to avoid creating
        unrealistic velocity perturbations that cause numerical instability.
        """
        rng = np.random.default_rng(self.config.background_seed)
        truth_array = self.m_true.getArray()

        # Get actual DOF indices for each component using function space
        h_indices, u_indices, v_indices = self._get_component_dof_indices()

        # Extract component values using proper indices
        h_values = truth_array[h_indices]
        u_values = truth_array[u_indices]
        v_values = truth_array[v_indices]

        h_magnitude = np.abs(h_values).mean() + 1e-10
        # Use max velocity magnitude with a floor of 0.1 m/s to prevent huge relative errors
        uv_magnitude = max(np.abs(u_values).max(), np.abs(v_values).max(), 0.1)

        # Create component-specific perturbations
        perturbation = np.zeros_like(truth_array)
        h_error = self.config.background_error_std * h_magnitude
        uv_error = self.config.background_error_std * uv_magnitude

        self.log(f"  Component-aware perturbation: h_std={h_error:.4f}, uv_std={uv_error:.4f}")

        # Perturb each component with appropriate magnitude
        perturbation[h_indices] = rng.normal(0, h_error, len(h_indices))
        perturbation[u_indices] = rng.normal(0, uv_error, len(u_indices))
        perturbation[v_indices] = rng.normal(0, uv_error, len(v_indices))

        background_array = truth_array + perturbation

        # Enforce physical constraints on h (water depth must be positive)
        # Use the h_indices from component analysis (not interleaved assumption)
        h_min = self.config.h_min
        n_clipped = np.sum(background_array[h_indices] < h_min)
        if n_clipped > 0:
            self.log(f"  Clipping {n_clipped} h values to h_min={h_min}")
            background_array[h_indices] = np.maximum(background_array[h_indices], h_min)

        self.m_background = self.m_true.duplicate()
        self.m_background.setArray(background_array)
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
        """Create physical bounds for the control variable.

        Uses proper component DOF indices for mixed elements where DOFs
        may not be interleaved in the simple [h,u,v,h,u,v,...] pattern.
        """
        lower = self.m_background.duplicate()
        upper = self.m_background.duplicate()

        lower_array = lower.getArray()
        upper_array = upper.getArray()

        # Use proper component DOF indices
        h_indices, u_indices, v_indices = self._get_component_dof_indices()

        # h: constrained to be positive (h >= h_min)
        lower_array[h_indices] = self.config.h_min
        upper_array[h_indices] = 1e10

        # u, v: unconstrained
        lower_array[u_indices] = -1e10
        upper_array[u_indices] = 1e10
        lower_array[v_indices] = -1e10
        upper_array[v_indices] = 1e10

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
        # Create a fresh solver to avoid state corruption from optimization
        from swe4dvar.forward.solvers import get_solver
        solver_type = type(self.solver).__name__
        if "DG" in solver_type:
            eval_solver = get_solver("DG")(self.problem, theta=0.5, p_degree=[1, 1], verbose=False)
        else:
            eval_solver = get_solver("SUPG")(self.problem, theta=0.5, p_degree=[1, 1], verbose=False)

        self.problem.t = 0.0

        # Update boundary conditions for t=0
        if hasattr(self.problem, 'update_boundary'):
            self.problem.update_boundary()

        m_analysis_array = self.m_analysis.getArray()
        u_owned_size = eval_solver.V.dofmap.index_map.size_local

        if len(m_analysis_array) == len(eval_solver.u_n.x.array):
            eval_solver.u_n.x.array[:] = m_analysis_array
            eval_solver.u_n_old.x.array[:] = m_analysis_array
            eval_solver.u.x.array[:] = m_analysis_array
        else:
            eval_solver.u_n.x.array[:u_owned_size] = m_analysis_array
            eval_solver.u_n_old.x.array[:u_owned_size] = m_analysis_array
            eval_solver.u.x.array[:u_owned_size] = m_analysis_array
            eval_solver.u_n.x.scatter_forward()
            eval_solver.u_n_old.x.scatter_forward()
            eval_solver.u.x.scatter_forward()

        eval_solver.time_loop(
            solver_parameters=self.solver_params,
            stations=np.array([[0.0, 0.0, 0.0]]),
            plot_every=9999,
            save_state=True,
            enable_video=False,
        )

        self.analysis_trajectory = []
        for state_array in eval_solver.storage.saved_states:
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

        # Add physics perturbation info
        config["inverse_crime_avoidance"] = {
            "perturb_bathymetry": self.config.perturb_bathymetry,
            "perturb_friction": self.config.perturb_friction,
        }
        if self.config.perturb_bathymetry:
            config["inverse_crime_avoidance"]["bathymetry_noise_type"] = self.config.bathymetry_noise_type
            config["inverse_crime_avoidance"]["bathymetry_noise_std"] = self.config.bathymetry_noise_std
            config["inverse_crime_avoidance"]["bathymetry_correlation_length"] = self.config.bathymetry_correlation_length
        if self.config.perturb_friction:
            config["inverse_crime_avoidance"]["friction_scale_factor"] = self.config.friction_scale_factor

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
        # Ensure integer dtype (empty arrays default to float64, causing index errors)
        self.boundary_dofs = np.asarray(boundary_dofs, dtype=int)

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

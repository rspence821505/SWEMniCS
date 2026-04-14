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
from swe4dvar.forward.augmented_control import AugmentedForwardModelWrapper
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
    background_correlation_length: float = 0.0  # Spatial correlation length (m). 0 = white noise.

    # Optimization configuration
    max_iterations: int = 50
    max_funcs: Optional[int] = None  # Max total function evaluations (None = unlimited)
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

    # Cycling 4D-Var
    n_windows: int = 1  # Number of cycling windows (1 = single window)

    # DC-WME L_wme estimation
    l_wme_samples: int = 100  # >0 = analytical L_wme via adjoint, 0 = 2R fallback
    auto_inflate_B: bool = True  # Auto-inflate B based on Gram matrix bound (paper eq. 38)
    predictability_gamma: float = 0.1  # Relaxation γ for predictability condition (paper eq. 36, 38)
    adaptive_gamma: bool = True  # Spectrum-relative γ: floor = γ * λ_max(L_wme)
    nonuniform_inflate: bool = False  # Per-eigenvalue boosting for sub-threshold L_wme eigenvalues
    eigenvalue_boost_target: float = 2.0  # Target eigenvalue as multiple of γ

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

    # DAMetrics-computed metrics
    mean_rmse: float = 0.0  # Mean RMSE across observation times
    data_misfit: float = 0.0  # Total data misfit term

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


class ForwardModelWrapper(AugmentedForwardModelWrapper):
    """
    Wrapper to adapt SWE4DVar solvers to the forward model interface
    expected by the cost functions.
    """

    def __init__(
        self,
        solver,
        problem,
        solver_params: dict,
        t_start: float = 0.0,
        control_layout=None,
        parameter_controller=None,
        parameter_fd_steps=None,
        reference_state=None,
    ):
        super().__init__(
            solver=solver,
            problem=problem,
            solver_params=solver_params,
            t_start=t_start,
            control_layout=control_layout,
            parameter_controller=parameter_controller,
            parameter_fd_steps=parameter_fd_steps,
            reference_state=reference_state,
        )

    def get_mass_matrix(self) -> PETSc.Mat:
        """Assemble the mass matrix for the function space using UFL."""
        if hasattr(self, '_mass_matrix_cache'):
            return self._mass_matrix_cache

        import os
        os.environ.setdefault("CC", "/usr/bin/clang")

        from ufl import TrialFunction, TestFunction, inner, dx
        from dolfinx import fem

        V = self.solver.V
        u = TrialFunction(V)
        v = TestFunction(V)
        mass_form = inner(u, v) * dx

        M = fem.petsc.assemble_matrix(fem.form(mass_form))
        M.assemble()

        self._mass_matrix_cache = M
        return M

    def solve(
        self, m: PETSc.Vec, store_jacobians: bool = True
    ) -> Tuple[List[PETSc.Vec], Optional[List]]:
        """Run forward model from a control vector.

        By default this remains the original state-only behavior. When a
        control layout and parameter controller are attached, ``m`` may be a
        packed augmented control ``[u_0; theta]`` or a pure-parameter control
        with an empty state block.
        """
        return super().solve(m, store_jacobians=store_jacobians)


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
            max_it=25,
            relaxation_parameter=1.0,
            comm=self.comm,
            error_if_not_converged=False,
            reduction_it=10,
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

        # Dispatch to cycling if n_windows > 1
        if self.config.n_windows > 1:
            return self._run_cycling(start_time)

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
        B, R, B_lwme = self._setup_covariances(obs_operator, obs_noise_stds)

        # Step 6: Create forward model wrapper
        self.log("\nStep 6: Creating forward model wrapper...")
        forward_model = self._create_forward_model()

        # Step 7: Setup cost function
        self.log(f"\nStep 7: Setting up {self.config.method.upper()} cost function...")
        cost_function = self._setup_cost_function(
            forward_model, obs_operator, B, R, obs_times, B_lwme=B_lwme
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
            mean_rmse = 0.0
            data_misfit = 0.0
            self.analysis_trajectory = None
        else:
            analysis_error, error_reduction, innov_mean, innov_std, mean_rmse, data_misfit = self._evaluate_results(
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
            mean_rmse=mean_rmse,
            data_misfit=data_misfit,
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
        # Store DA window start time (after warmup) for correct evaluation
        self.t_da_start = self.problem.t

        # Run forward model
        self.solver.time_loop(
            solver_parameters=self.solver_params,
            stations=np.array([[0.0, 0.0, 0.0]]),
            plot_every=9999,
            save_state=True,
            store_jacobians=False,  # Truth Jacobians not needed by DA (saves ~2.8GB at 52K DOFs)
            enable_video=False,
            monitor_progress=(self.rank == 0 and self.config.verbose),
        )

        # Store truth trajectory using FEM-consistent vector creation
        # CRITICAL: Must use la.create_petsc_vector with the FEM index map to ensure
        # vectors have global size = sum of owned DOFs (excluding ghosts).
        # Using createWithArray(full_array, comm=COMM_WORLD) would include ghost DOFs
        # in the global size, causing size mismatches with vectors created during
        # the forward model solve (which use la.create_petsc_vector).
        from dolfinx import la
        u_owned_size = self.solver.V.dofmap.index_map.size_local * self.solver.V.dofmap.index_map_bs
        self.truth_trajectory = []
        for state_array in self.solver.storage.saved_states:
            vec = la.create_petsc_vector(
                self.solver.V.dofmap.index_map,
                self.solver.V.dofmap.index_map_bs,
            )
            vec.setArray(state_array[:u_owned_size])
            vec.assemble()
            self.truth_trajectory.append(vec)

        # True initial condition
        self.m_true = self.truth_trajectory[0].copy()

        # Free solver storage now that we've extracted the trajectory
        self.solver.storage.clear()

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

        # Observation times (include t=0 if obs_frequency divides evenly)
        obs_times = list(
            range(
                0,
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
        """Generate observation points from interior mesh nodes only.

        IMPORTANT: In parallel, mesh.geometry.x is LOCAL to each rank.
        We must gather ALL coordinates to rank 0, generate points there,
        then broadcast to all ranks for consistency.
        """
        # Gather all coordinates to rank 0
        local_coords = self.problem.mesh.geometry.x
        all_coords = self.comm.gather(local_coords, root=0)

        if self.comm.rank == 0:
            # Concatenate all coordinates from all ranks, deduplicate, and sort
            # to ensure deterministic obs selection regardless of MPI partition count.
            # (DG meshes duplicate geometry points at partition boundaries)
            coords_all = np.vstack(all_coords)
            _, unique_idx = np.unique(
                np.round(coords_all[:, :2], decimals=10), axis=0, return_index=True
            )
            coords = coords_all[unique_idx]  # sorted by np.unique

            rng = np.random.default_rng(self.config.obs_seed)

            # Domain bounds (now using GLOBAL unique coordinates)
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
        else:
            obs_points = None

        # Broadcast observation points from rank 0 to all ranks
        obs_points = self.comm.bcast(obs_points, root=0)
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

    def _get_component_dof_indices(self, owned_only: bool = False):
        """Get DOF indices for each component (h, u, v) using function space structure.

        Args:
            owned_only: If True, filter to only owned DOFs (excluding ghost DOFs).
                       Use this when indexing into PETSc vectors created with
                       la.create_petsc_vector which only contain owned DOFs.

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

            if owned_only:
                # Filter to only owned DOFs (exclude ghost DOFs)
                u_owned_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
                h_indices = h_indices[h_indices < u_owned_size]
                u_indices = u_indices[u_indices < u_owned_size]
                v_indices = v_indices[v_indices < u_owned_size]

            return h_indices, u_indices, v_indices
        else:
            # Fallback: assume interleaved (h, u, v)
            n_dofs = len(self.solver.u.x.array)
            if owned_only:
                n_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
            h_indices = np.arange(0, n_dofs, 3)
            u_indices = np.arange(1, n_dofs, 3)
            v_indices = np.arange(2, n_dofs, 3)
            return h_indices, u_indices, v_indices

    def _build_smoothing_matrix(self, component_indices: np.ndarray, correlation_length: float):
        """Build Gaussian spatial smoothing matrix for DOFs of one component.

        Uses a sparse KD-tree approach: only computes weights for neighbors
        within 3*correlation_length (beyond which exp(-d²/2L²) < 1e-4).
        This avoids building a dense n×n matrix at large DOF counts.

        Args:
            component_indices: DOF indices in the parent space for one component
            correlation_length: Gaussian correlation length scale (meters)

        Returns:
            Sparse CSR smoothing matrix of shape (n_dofs, n_dofs)
        """
        from scipy.spatial import cKDTree
        from scipy import sparse

        # Get DOF coordinates via collapsed sub-space
        V = self.solver.V
        h_sub = V.sub(0)
        h_space, h_map = h_sub.collapse()
        all_coords = h_space.tabulate_dof_coordinates()[:, :2]  # (n_pts, 2)

        # Map parent DOF indices to collapsed indices
        parent_to_collapsed = np.full(max(h_map) + 1, -1, dtype=int)
        for collapsed_i, parent_i in enumerate(h_map):
            parent_to_collapsed[parent_i] = collapsed_i

        # Get coordinates in the order of component_indices
        collapsed_idx = parent_to_collapsed[component_indices]
        coords = all_coords[collapsed_idx]  # (n_component_dofs, 2)

        n = len(coords)
        radius = 3.0 * correlation_length  # exp(-9/2) ≈ 0.011, negligible beyond this
        two_L_sq = 2.0 * correlation_length ** 2

        # Build sparse Gaussian kernel using KD-tree neighbor queries
        tree = cKDTree(coords)
        rows, cols, data = [], [], []
        for i in range(n):
            neighbors = tree.query_ball_point(coords[i], radius)
            if len(neighbors) == 0:
                neighbors = [i]  # self-connection as fallback
            dists_sq = np.sum((coords[neighbors] - coords[i]) ** 2, axis=1)
            weights = np.exp(-dists_sq / two_L_sq)
            rows.extend([i] * len(neighbors))
            cols.extend(neighbors)
            data.extend(weights)

        G = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

        # Normalize rows so G @ white_noise has unit variance
        # Var(G @ z) = sum(G_ij^2) for z ~ N(0,I), so normalize by sqrt(row sum of G^2)
        G_sq = G.multiply(G)
        row_norms = np.sqrt(np.array(G_sq.sum(axis=1)).flatten())
        row_norms[row_norms == 0] = 1.0  # avoid division by zero
        # Scale rows by 1/row_norm
        inv_norms = sparse.diags(1.0 / row_norms)
        G = inv_norms @ G

        self.log(f"  Smoothing matrix: {n}x{n}, nnz={G.nnz} ({100*G.nnz/n**2:.1f}% dense)")
        return G

    def _setup_background(self, n_vars: int = 3) -> float:
        """Setup background state with component-aware perturbation from truth.

        Uses different perturbation magnitudes for h vs u,v to avoid creating
        unrealistic velocity perturbations that cause numerical instability.
        When background_correlation_length > 0, applies Gaussian spatial smoothing
        to produce spatially correlated perturbations that persist as coherent
        structures under forward model integration.
        """
        rng = np.random.default_rng(self.config.background_seed)
        truth_array = self.m_true.getArray()

        # Get actual DOF indices for each component using function space
        # Use owned_only=True since m_true is a PETSc vector with only owned DOFs
        h_indices, u_indices, v_indices = self._get_component_dof_indices(owned_only=True)

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

        # Generate white noise per component
        h_noise = rng.normal(0, 1.0, len(h_indices))
        u_noise = rng.normal(0, 1.0, len(u_indices))
        v_noise = rng.normal(0, 1.0, len(v_indices))

        # Apply spatial correlation if requested
        L = self.config.background_correlation_length
        if L > 0:
            smoothing_matrix = self._build_smoothing_matrix(h_indices, L)
            h_noise = smoothing_matrix @ h_noise
            u_noise = smoothing_matrix @ u_noise
            v_noise = smoothing_matrix @ v_noise
            self.log(f"  Applied spatial correlation: L={L:.0f} m")

        # Scale to desired standard deviation
        perturbation[h_indices] = h_error * h_noise
        perturbation[u_indices] = uv_error * u_noise
        perturbation[v_indices] = uv_error * v_noise

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
            # CRITICAL FIX: Use m_true as template to ensure covariance uses same MPI partitioning
            # as FEM state vectors. Without this, PETSc.DECIDE partitioning may differ from
            # FEM mesh partitioning, causing MPI errors in pointwiseMult operations.
            B = DiagonalCovariance(self.comm, state_size, variance=background_variance,
                                  template_vec=self.m_true)
            self.log(f"  Background covariance: diagonal, variance = {background_variance:.6e}")

        # Add spatial correlation if requested (for L_wme computation only)
        # Skip dense B_lwme construction for standard 4dvar — it's only needed for DC-WME
        # and builds a dense n_state × n_state matrix that's infeasible at large DOF counts.
        B_lwme = None
        corr_len = self.config.background_correlation_length
        if corr_len > 0 and self.config.method in ("dcwme", "dc"):
            B_lwme = self._add_spatial_correlation(B, corr_len)

        # Observation covariance
        n_obs = obs_operator.get_num_observations()
        obs_variance = obs_noise_stds.mean() ** 2
        # CRITICAL FIX: Create template observation vector to ensure R uses same partitioning
        # as observation vectors from obs_operator.forward(). Observation vectors are sequential
        # (COMM_SELF) on each rank, so R must also be sequential/local.
        template_obs = obs_operator.forward(self.m_true)
        R = DiagonalCovariance(self.comm, n_obs, variance=obs_variance,
                              template_vec=template_obs)
        template_obs.destroy()  # Clean up template
        self.log(f"  Observation covariance: diagonal, variance = {obs_variance:.6e}")

        return B, R, B_lwme

    def _add_spatial_correlation(self, B_diag, correlation_length: float):
        """Convert diagonal B to spatially correlated B using Gaussian kernel.

        Builds B_corr = D^{1/2} C D^{1/2} where D is the diagonal variance
        and C is a Gaussian correlation matrix based on DOF distances.

        Parameters
        ----------
        B_diag : DiagonalCovariance
            Original diagonal background covariance.
        correlation_length : float
            Gaussian correlation length scale (meters).

        Returns
        -------
        DenseCovariance
            Spatially correlated background covariance.
        """
        from scipy.spatial.distance import cdist
        from swe4dvar.data_assimilation.covariance import DenseCovariance

        state_size = B_diag.size

        # Get DOF coordinates via cell centroids.
        # For DG elements, DOFs within a cell share the centroid position.
        # This creates rank deficiency in the correlation matrix, but the
        # regularization parameter ε controls the minimum eigenvalue.
        mesh = self.problem.mesh
        tdim = mesh.topology.dim
        num_cells = mesh.topology.index_map(tdim).size_local
        geom = mesh.geometry.x

        cell_centroids = np.zeros((num_cells, 2))
        for cell_idx in range(num_cells):
            cell_verts = mesh.geometry.dofmap[cell_idx]
            cell_centroids[cell_idx] = geom[cell_verts, :2].mean(axis=0)

        dofmap = self.solver.V.dofmap
        dof_coords = np.zeros((state_size, 2))
        for cell_idx in range(num_cells):
            for dof in dofmap.cell_dofs(cell_idx):
                dof_coords[dof] = cell_centroids[cell_idx]

        # Build Gaussian correlation matrix with regularization.
        # ε controls the minimum eigenvalue of the correlation matrix.
        # Larger ε = better conditioned but weaker correlation effect.
        dist_matrix = cdist(dof_coords, dof_coords, metric='euclidean')
        correlation = np.exp(-dist_matrix**2 / (2 * correlation_length**2))
        eps = 1e-6
        correlation = (1 - eps) * correlation + eps * np.eye(state_size)

        # Get variances from diagonal B
        if hasattr(B_diag, 'diagonal'):
            variances = B_diag.diagonal.getArray().copy()
        else:
            # Uniform variance fallback
            variances = np.full(state_size, B_diag.min_eigenvalue())

        B_corr = DenseCovariance.from_correlation(
            self.comm, correlation, variances, inverse_method="cholesky"
        )

        self.log(f"  Spatial correlation: Gaussian L={correlation_length:.0f}m")
        self.log(f"    Dense B: {state_size}x{state_size}, "
                 f"min eigenvalue ≈ {B_corr.min_eigenvalue():.4e}")

        return B_corr

    def _estimate_component_variances(self):
        """Estimate component-specific variances from truth state.

        Uses the same component magnitudes as _setup_background() to ensure
        B covariance matches the actual perturbation statistics.

        Environment variable overrides (for controlled experiments):
          SWE4DVAR_H_VARIANCE  — override h component variance
          SWE4DVAR_UV_VARIANCE — override u/v component variance
        """
        import os

        arr = self.m_true.getArray()
        h_indices, u_indices, v_indices = self._get_component_dof_indices(owned_only=True)

        h_mag = np.abs(arr[h_indices]).mean() + 1e-10
        uv_mag = max(np.abs(arr[u_indices]).max(), np.abs(arr[v_indices]).max(), 0.1)

        h_variance = (self.config.background_error_std * h_mag) ** 2
        uv_variance = (self.config.background_error_std * uv_mag) ** 2

        # Allow environment variable overrides for controlled experiments
        h_override = os.environ.get("SWE4DVAR_H_VARIANCE")
        uv_override = os.environ.get("SWE4DVAR_UV_VARIANCE")
        if h_override is not None:
            h_variance = float(h_override)
            self.log(f"  [override] h_variance = {h_variance:.6e} (from SWE4DVAR_H_VARIANCE)")
        if uv_override is not None:
            uv_variance = float(uv_override)
            self.log(f"  [override] uv_variance = {uv_variance:.6e} (from SWE4DVAR_UV_VARIANCE)")

        return h_variance, uv_variance

    def _create_component_aware_covariance(
        self, state_size: int, h_variance: float, velocity_variance: float
    ):
        """Create component-aware diagonal covariance using correct DOF indices."""
        h_indices, u_indices, v_indices = self._get_component_dof_indices(owned_only=True)

        variances = np.zeros(state_size)
        variances[h_indices] = h_variance
        variances[u_indices] = velocity_variance
        variances[v_indices] = velocity_variance

        return DiagonalCovariance(self.comm, state_size, diagonal=variances,
                                  template_vec=self.m_true)

    def _create_forward_model(self, t_start: float = 0.0):
        """Create forward model wrapper."""
        self.solver.storage.clear()
        self.problem.t = t_start
        return ForwardModelWrapper(self.solver, self.problem, self.solver_params, t_start=t_start)

    def _setup_cost_function(self, forward_model, obs_operator, B, R, obs_times, B_lwme=None):
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
                n_l_wme_samples=self.config.l_wme_samples,
                auto_inflate_B=self.config.auto_inflate_B,
                predictability_gamma=self.config.predictability_gamma,
                adaptive_gamma=self.config.adaptive_gamma,
                B_lwme=B_lwme,
                nonuniform_inflate=self.config.nonuniform_inflate,
                eigenvalue_boost_target=self.config.eigenvalue_boost_target,
                comm=self.comm,
            )
        else:
            raise ValueError(f"Unknown DA method: {self.config.method}")

        # DISABLE M^{-1} preconditioning - causes ill-conditioning with DG
        # Mass matrix range [187, 460513] creates 2450x scaling variation
        # This makes gradients tiny and optimizer stuck
        # TODO: Investigate proper preconditioning (maybe scaled M^{-1} or different approach)
        if False and hasattr(forward_model, 'get_mass_matrix'):
            M = forward_model.get_mass_matrix()
            cost_function = MassMatrixPreconditionedCost(cost_function, M)
            diag = M.getDiagonal()
            diag_arr = diag.getArray()
            self.log(f"  Applied M^{{-1}} gradient preconditioning (M diag: min={diag_arr.min():.2f}, max={diag_arr.max():.2f}, mean={diag_arr.mean():.2f})")
            diag.destroy()
        else:
            self.log(f"  M^{{-1}} preconditioning DISABLED (causes ill-conditioning)")

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
            "gradient_relative_tolerance": 0.0,  # Disable grtol; use gatol only
            "cost_tolerance": self.config.cost_tolerance,
            "verbose": (self.rank == 0),
            "line_search_initial_step": 1.0,  # Standard for L-BFGS; NaN detection + Armijo backtracking handles overshoots
            "line_search_max_funcs": 50,  # Extra backtracking attempts for Armijo when forward model fails
        }
        if self.config.max_funcs is not None:
            opt_options["max_funcs"] = self.config.max_funcs
        if hasattr(self, "optimization_iteration_callback"):
            opt_options["iteration_callback"] = self.optimization_iteration_callback

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
        # Start optimization from perturbed background state (standard 4D-Var).
        # With spatially smooth perturbations (background_correlation_length > 0),
        # m_background is dynamically consistent and the forward model can integrate it.
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

        # Use proper component DOF indices (owned only since these are PETSc vectors)
        h_indices, u_indices, v_indices = self._get_component_dof_indices(owned_only=True)

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
        """Evaluate analysis results using DAMetrics."""
        from swe4dvar.data_assimilation.metrics import DAMetrics

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

        # CRITICAL FIX: Use the same start time as truth generation (not t=0!)
        # The truth was generated after warmup at t=t_da_start
        # Analysis must start from the same time to match observations
        self.problem.t = getattr(self, 't_da_start', 0.0)

        # Update boundary conditions for the CORRECT time (t_da_start, not t=0!)
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

        # Create observation dictionaries for DAMetrics
        obs_dict = {obs_times[i]: self.observations[i] for i in range(len(obs_times))}
        obs_op_dict = {k: obs_operator for k in obs_times}

        # Use DAMetrics for RMSE and data misfit
        metrics = DAMetrics(obs_dict, obs_op_dict)
        rmse_dict = metrics.compute_rmse(self.analysis_trajectory)
        data_misfit = metrics.compute_data_misfit(self.analysis_trajectory)
        mean_rmse = float(np.mean(list(rmse_dict.values()))) if rmse_dict else 0.0

        self.log(f"  Mean RMSE: {mean_rmse:.6f}")
        self.log(f"  Data misfit: {data_misfit:.6f}")

        # Innovation statistics (keep existing calculation for backward compatibility)
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

        return analysis_error, error_reduction, innov_mean, innov_std, mean_rmse, data_misfit

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

    def _run_cycling(self, start_time: float) -> TwinExperimentResults:
        """Run cycling 4D-Var with multiple assimilation windows.

        Splits the total simulation into n_windows shorter windows.
        The analysis from each window is propagated forward to provide
        the background for the next window.
        """
        n_windows = self.config.n_windows
        total_nt = self.problem.nt
        window_nt = total_nt // n_windows

        if total_nt % n_windows != 0:
            raise ValueError(
                f"total_nt ({total_nt}) must be divisible by n_windows ({n_windows})"
            )

        self.log(f"\n  Cycling 4D-Var: {n_windows} windows x {window_nt} timesteps "
                 f"({window_nt * self.problem.dt / 3600:.1f}h each)")

        # Step 1: Generate truth trajectory for FULL simulation
        self.log("\nStep 1: Generating truth trajectory (full)...")
        self._generate_truth()

        # Step 1b: Apply physics perturbations
        if self.config.perturb_bathymetry or self.config.perturb_friction:
            self.log("\nStep 1b: Applying physics perturbations...")
            self._apply_physics_perturbations()

        # Step 2: Setup observations for FULL trajectory
        self.log("\nStep 2: Setting up observations (full)...")
        obs_points, obs_operator, global_obs_times = self._setup_observations()

        # Step 3: Generate synthetic observations for FULL trajectory
        self.log("\nStep 3: Generating synthetic observations (full)...")
        self.observations, obs_noise_stds = self._generate_observations(
            obs_operator, global_obs_times
        )

        # Step 4: Setup initial background state (perturbed IC at t=0)
        self.log("\nStep 4: Setting up background state...")
        background_error = self._setup_background()

        # Step 5: Setup covariance matrices
        self.log("\nStep 5: Setting up covariance matrices...")
        B, R, B_lwme = self._setup_covariances(obs_operator, obs_noise_stds)

        # Cycling loop
        all_cost_history = []
        all_gradient_history = []
        total_iterations = 0
        window_0_analysis_error = None
        last_converged = False

        for w in range(n_windows):
            t_start_window = w * window_nt * self.problem.dt
            self.log(f"\n{'='*60}")
            self.log(f"  Window {w+1}/{n_windows}: t = {t_start_window/3600:.1f}h - "
                     f"{(w+1) * window_nt * self.problem.dt / 3600:.1f}h")
            self.log(f"{'='*60}")

            # a) Temporarily set problem.nt to window length
            self.problem.nt = window_nt

            # b) Subset observations for this window
            # Global obs times in range (w*window_nt, (w+1)*window_nt]
            window_obs_indices = []
            window_local_times = []
            for i, gt in enumerate(global_obs_times):
                if w * window_nt < gt <= (w + 1) * window_nt:
                    window_obs_indices.append(i)
                    window_local_times.append(gt - w * window_nt)

            window_observations = [self.observations[i] for i in window_obs_indices]
            self.log(f"  Window obs: {len(window_local_times)} observations at local times {window_local_times}")

            if len(window_observations) == 0:
                self.log(f"  WARNING: No observations in this window, skipping optimization")
                # Propagate background forward to next window
                if w < n_windows - 1:
                    self.m_background = self._propagate_forward(
                        self.m_background, window_nt, t_start_window
                    )
                continue

            # c) Create forward model for this window
            forward_model = self._create_forward_model(t_start=t_start_window)

            # d) Setup cost function with window-specific observations
            # Temporarily swap observations for cost function setup
            orig_observations = self.observations
            self.observations = window_observations
            cost_function = self._setup_cost_function(
                forward_model, obs_operator, B, R, window_local_times, B_lwme=B_lwme
            )
            self.observations = orig_observations

            # e) Run optimization
            self.log(f"  Running optimization for window {w+1}...")
            optimizer, opt_time = self._run_optimization(cost_function)

            # Accumulate history
            window_cost = [h["cost"] for h in optimizer.convergence_history]
            window_grad = [h["grad_norm"] for h in optimizer.convergence_history]
            all_cost_history.extend(window_cost)
            all_gradient_history.extend(window_grad)
            total_iterations += optimizer.iteration
            last_converged = optimizer.converged

            self.log(f"  Window {w+1} result: {optimizer.iteration} iterations, "
                     f"converged={optimizer.converged}")

            # Compute window 0 analysis error (for comparison with single-window)
            if w == 0:
                diff = self.m_analysis.copy()
                diff.axpy(-1.0, self.m_true)
                window_0_analysis_error = np.sqrt(diff.dot(diff) / diff.getSize())
                diff.destroy()
                self.log(f"  Window 0 analysis error: {window_0_analysis_error:.6f}")

            # f) Propagate analysis forward to get background for next window
            if w < n_windows - 1:
                self.log(f"  Propagating analysis forward to next window...")
                next_background = self._propagate_forward(
                    self.m_analysis, window_nt, t_start_window
                )
                self.m_background = next_background

        # Restore problem.nt
        self.problem.nt = total_nt

        # Results: use window 0 analysis error as the primary metric
        total_time = time.time() - start_time
        analysis_error = window_0_analysis_error if window_0_analysis_error is not None else background_error
        error_reduction = (background_error - analysis_error) / background_error * 100

        self.log(f"\n  Cycling complete: {total_iterations} total iterations across {n_windows} windows")
        self.log(f"  Background error: {background_error:.6f}")
        self.log(f"  Analysis error (window 0): {analysis_error:.6f}")
        self.log(f"  Error reduction: {error_reduction:.1f}%")

        results = TwinExperimentResults(
            method=self.config.method,
            problem_name=self.problem_name,
            cost_history=all_cost_history,
            gradient_norm_history=all_gradient_history,
            background_error=background_error,
            analysis_error=analysis_error,
            error_reduction=error_reduction,
            mean_rmse=0.0,
            data_misfit=0.0,
            innovation_mean=0.0,
            innovation_std=0.0,
            num_iterations=total_iterations,
            converged=last_converged,
            wall_time=total_time,
            config=self.config.to_dict(),
            problem_config=self._get_problem_config(),
        )

        self._save_results(results)
        self._print_summary(results)
        self._cleanup()

        return results

    def _propagate_forward(
        self, m: PETSc.Vec, nt: int, t_start: float
    ) -> PETSc.Vec:
        """Run the forward model from state m for nt timesteps starting at t_start.

        Returns the final state as a PETSc Vec.
        """
        self.solver.storage.clear()

        # Set initial condition
        m_array = m.getArray()
        u_owned_size = self.solver.V.dofmap.index_map.size_local
        if len(m_array) == len(self.solver.u_n.x.array):
            self.solver.u_n.x.array[:] = m_array
            self.solver.u_n_old.x.array[:] = m_array
            self.solver.u.x.array[:] = m_array
        else:
            self.solver.u_n.x.array[:u_owned_size] = m_array
            self.solver.u_n_old.x.array[:u_owned_size] = m_array
            self.solver.u.x.array[:u_owned_size] = m_array
            self.solver.u_n.x.scatter_forward()
            self.solver.u_n_old.x.scatter_forward()
            self.solver.u.x.scatter_forward()

        # Set time and nt for propagation
        orig_nt = self.problem.nt
        self.problem.nt = nt
        self.problem.t = t_start

        self.solver.time_loop(
            solver_parameters=self.solver_params,
            stations=np.array([[0.0, 0.0, 0.0]]),
            plot_every=9999,
            save_state=True,
            enable_video=False,
        )

        # Get final state
        final_state = self.solver.storage.saved_states[-1]
        final_vec = PETSc.Vec().createWithArray(final_state.copy(), comm=self.comm)

        self.problem.nt = orig_nt
        return final_vec


class MassMatrixPreconditionedCost:
    """Wrapper that preconditions the gradient by M^{-1}.

    For DG elements, the adjoint gradient includes the mass matrix M
    (from ∂R/∂u₀ = -M/dt), which makes ||∇J|| ~ O(M). This prevents
    the optimizer from finding a step size.

    Preconditioning by M^{-1} converts the functional derivative to the
    Riesz gradient in the L²-inner product, giving ||g|| ~ O(1).
    The converged solution is identical (M is SPD so g=0 iff ∇J=0).

    Uses diagonal M^{-1} scaling (lumped mass) instead of full M^{-1}
    to preserve the gradient direction more faithfully. The diagonal
    mass matrix handles per-element scaling without the severe direction
    distortion that can occur with full M^{-1} on meshes with highly
    variable element sizes.
    """

    def __init__(self, base_cost, mass_matrix: PETSc.Mat):
        self.base_cost = base_cost
        self.M = mass_matrix
        # Use diagonal (lumped) mass matrix for scaling
        self._M_diag = mass_matrix.getDiagonal()
        self._M_diag_inv = self._M_diag.duplicate()
        diag_arr = self._M_diag.getArray()
        inv_arr = np.where(diag_arr > 0, 1.0 / diag_arr, 0.0)
        self._M_diag_inv.setArray(inv_arr)

    def value(self, m: PETSc.Vec) -> float:
        return self.base_cost.value(m)

    def gradient(self, m: PETSc.Vec) -> PETSc.Vec:
        grad = self.base_cost.gradient(m)
        precond_grad = grad.duplicate()
        precond_grad.pointwiseMult(grad, self._M_diag_inv)
        return precond_grad

    def value_gradient(self, m: PETSc.Vec):
        cost, grad = self.base_cost.value_gradient(m)
        precond_grad = grad.duplicate()
        precond_grad.pointwiseMult(grad, self._M_diag_inv)
        if not hasattr(self, '_logged'):
            self._logged = True
            raw_norm = grad.norm()
            precond_norm = precond_grad.norm()
            print(f"  [diag M⁻¹ precond] cost={cost:.4f}, ||grad_raw||={raw_norm:.4e}, ||grad_precond||={precond_norm:.4e}")
        return cost, precond_grad

    def clear_cache(self):
        if hasattr(self.base_cost, "clear_cache"):
            self.base_cost.clear_cache()


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

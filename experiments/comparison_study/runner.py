"""
Experiment runner for the comparison study.

This module provides the ComparisonRunner class that orchestrates parameter
sweeps, handles errors gracefully, and captures diagnostics.

Key design principles:
1. Never stop the full sweep - catch all exceptions and continue
2. Record failure details with diagnostics for debugging
3. Save incrementally - write results after each experiment
4. Handle missing data in plotting
"""

import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from .config import ComparisonStudyConfig, SweepConfig, EXPERIMENTS_WITH_MODEL_ERROR
from .diagnostics import (
    ExperimentDiagnostics,
    classify_failure,
    generate_diagnostic_report,
)

# Import twin experiment framework
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig


@dataclass
class ExperimentResult:
    """Result of a single experiment."""

    experiment_id: str
    status: str  # "success", "failed", "skipped"
    method: str
    sweep_parameter: str
    sweep_value: Any
    result: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    failure_type: Optional[str] = None
    wall_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "method": self.method,
            "sweep_parameter": self.sweep_parameter,
            "sweep_value": self.sweep_value,
            "result": self.result,
            "diagnostics": self.diagnostics,
            "error": self.error,
            "failure_type": self.failure_type,
            "wall_time": self.wall_time,
        }


class ComparisonRunner:
    """Runner for 4D-Var vs DC-WME comparison experiments.

    Orchestrates parameter sweeps with robust error handling and diagnostics.
    """

    def __init__(
        self,
        base_config: ComparisonStudyConfig,
        sweep_config: SweepConfig,
        timeout: float = 1800.0,  # 30 minutes per experiment
    ):
        """Initialize the runner.

        Args:
            base_config: Base problem and DA configuration
            sweep_config: Sweep parameters for experiments
            timeout: Maximum time (seconds) per experiment
        """
        self.base_config = base_config
        self.sweep_config = sweep_config
        self.timeout = timeout
        self.results: Dict[str, ExperimentResult] = {}

        # Create output directories
        self.output_dir = Path(base_config.output_dir)
        self.data_dir = self.output_dir / "data"
        self.figures_dir = self.output_dir / "figures"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def _create_problem_solver(self) -> tuple:
        """Create fresh problem and solver instances."""
        problem = TidalProblem(
            nx=self.base_config.nx,
            ny=self.base_config.ny,
            dt=self.base_config.dt,
            nt=self.base_config.nt,
        )
        solver = get_solver(self.base_config.solver_type)(
            problem, theta=0.5, p_degree=self.base_config.p_degree
        )
        return problem, solver

    def _create_twin_config(
        self,
        method: str,
        sweep_param: str,
        sweep_value: Any,
    ) -> TwinExperimentConfig:
        """Create TwinExperimentConfig for a specific experiment.

        Args:
            method: "4dvar" or "dcwme"
            sweep_param: Parameter being swept
            sweep_value: Value for this experiment

        Returns:
            Configured TwinExperimentConfig
        """
        # Start with base values
        config_kwargs = {
            "method": method,
            "obs_frequency": self.base_config.obs_frequency,
            "obs_fraction": self.base_config.obs_fraction,
            "obs_noise_level": self.base_config.obs_noise_level,
            "background_error_std": self.base_config.background_error_std,
            "max_iterations": self.base_config.max_iterations,
            "gradient_tolerance": self.base_config.gradient_tolerance,
            "obs_seed": self.base_config.obs_seed,
            "background_seed": self.base_config.background_seed,
            "perturbation_seed": self.base_config.perturbation_seed,
            "verbose": False,  # Reduce output during sweeps
            "interior_only": True,  # Critical for DG
            "use_bounds": True,
            "h_min": 0.01,
        }

        # Apply physics perturbation for non-friction sweeps
        # (friction sweep varies friction directly, others use default model error)
        if sweep_param != "friction":
            config_kwargs["perturb_friction"] = True
            config_kwargs["friction_scale_factor"] = self.base_config.friction_scale_factor

        # Apply sweep parameter
        if sweep_param == "friction":
            config_kwargs["perturb_friction"] = sweep_value > 1.0
            config_kwargs["friction_scale_factor"] = sweep_value
        elif sweep_param == "obs_freq":
            config_kwargs["obs_frequency"] = sweep_value
        elif sweep_param == "obs_fraction":
            config_kwargs["obs_fraction"] = sweep_value
        elif sweep_param == "noise":
            config_kwargs["obs_noise_level"] = sweep_value
        elif sweep_param == "background":
            config_kwargs["background_error_std"] = sweep_value
        elif sweep_param == "bathymetry":
            config_kwargs["perturb_bathymetry"] = sweep_value > 0
            config_kwargs["bathymetry_noise_std"] = sweep_value

        return TwinExperimentConfig(**config_kwargs)

    def _run_single_experiment(
        self,
        method: str,
        sweep_param: str,
        sweep_value: Any,
    ) -> ExperimentResult:
        """Run a single experiment with diagnostics.

        Args:
            method: "4dvar" or "dcwme"
            sweep_param: Parameter being swept
            sweep_value: Value for this experiment

        Returns:
            ExperimentResult with status, result, and diagnostics
        """
        exp_id = f"{method}_{sweep_param}_{sweep_value}"
        diagnostics = ExperimentDiagnostics()
        start_time = time.time()

        try:
            # Create fresh problem/solver for each experiment
            problem, solver = self._create_problem_solver()
            config = self._create_twin_config(method, sweep_param, sweep_value)

            # Run experiment
            experiment = TwinExperiment(problem, solver, config)

            # Hook diagnostics into cost function if available
            if hasattr(experiment, "cost_function") and experiment.cost_function:
                diagnostics.attach_to_cost_function(experiment.cost_function)

            result = experiment.run()
            wall_time = time.time() - start_time

            # Capture additional diagnostics from result
            diagnostics.cost_history = result.cost_history
            diagnostics.gradient_norm_history = result.gradient_norm_history

            # Try to capture cost breakdown
            if hasattr(experiment, "cost_function") and experiment.cost_function:
                diagnostics.capture_final_state(experiment.cost_function)

            return ExperimentResult(
                experiment_id=exp_id,
                status="success",
                method=method,
                sweep_parameter=sweep_param,
                sweep_value=sweep_value,
                result=result.to_dict(),
                diagnostics=diagnostics.to_dict(),
                wall_time=wall_time,
            )

        except Exception as e:
            wall_time = time.time() - start_time
            diagnostics.failure_iteration = len(diagnostics.cost_history)
            diagnostics.failure_traceback = traceback.format_exc()
            failure_type = classify_failure(diagnostics.to_dict())

            return ExperimentResult(
                experiment_id=exp_id,
                status="failed",
                method=method,
                sweep_parameter=sweep_param,
                sweep_value=sweep_value,
                error=str(e),
                failure_type=failure_type,
                diagnostics=diagnostics.to_dict(),
                wall_time=wall_time,
            )

    def _save_results(self, sweep_name: str):
        """Save current results to JSON file."""
        output_file = self.data_dir / f"{sweep_name}_sweep.json"
        results_dict = {
            exp_id: result.to_dict() for exp_id, result in self.results.items()
        }
        results_dict["_metadata"] = {
            "base_config": self.base_config.to_dict(),
            "sweep_name": sweep_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

    def _load_existing_results(self, sweep_name: str) -> Dict[str, ExperimentResult]:
        """Load existing results to enable resume."""
        output_file = self.data_dir / f"{sweep_name}_sweep.json"
        if not output_file.exists():
            return {}

        with open(output_file, "r") as f:
            data = json.load(f)

        # Remove metadata
        data.pop("_metadata", None)

        # Convert to ExperimentResult objects
        results = {}
        for exp_id, result_dict in data.items():
            results[exp_id] = ExperimentResult(
                experiment_id=result_dict["experiment_id"],
                status=result_dict["status"],
                method=result_dict["method"],
                sweep_parameter=result_dict["sweep_parameter"],
                sweep_value=result_dict["sweep_value"],
                result=result_dict.get("result"),
                diagnostics=result_dict.get("diagnostics"),
                error=result_dict.get("error"),
                failure_type=result_dict.get("failure_type"),
                wall_time=result_dict.get("wall_time", 0.0),
            )
        return results

    def run_sweep(
        self,
        sweep_name: str,
        sweep_values: Optional[List[Any]] = None,
        methods: Optional[List[str]] = None,
        resume: bool = True,
    ) -> Dict[str, ExperimentResult]:
        """Run a parameter sweep.

        Args:
            sweep_name: Name of the sweep (e.g., "friction", "obs_freq")
            sweep_values: Values to sweep (uses defaults from SweepConfig if None)
            methods: Methods to compare (uses defaults from SweepConfig if None)
            resume: If True, skip already-completed experiments

        Returns:
            Dictionary of ExperimentResult objects
        """
        if sweep_values is None:
            sweep_values = self.sweep_config.get_sweep_values(sweep_name)
        if methods is None:
            methods = self.sweep_config.methods

        # Load existing results for resume
        if resume:
            self.results = self._load_existing_results(sweep_name)
            if self.results:
                print(f"Resuming from {len(self.results)} existing experiments")

        total_experiments = len(sweep_values) * len(methods)
        completed = 0

        print(f"\n{'=' * 60}")
        print(f"Running {sweep_name} sweep: {len(sweep_values)} values x {len(methods)} methods")
        print(f"{'=' * 60}\n")

        for value in sweep_values:
            for method in methods:
                exp_id = f"{method}_{sweep_name}_{value}"
                completed += 1

                # Skip if already completed
                if exp_id in self.results:
                    status = self.results[exp_id].status
                    print(f"[{completed}/{total_experiments}] {exp_id} - SKIPPED (already {status})")
                    continue

                # Run experiment
                print(f"[{completed}/{total_experiments}] {exp_id} - RUNNING...", end=" ", flush=True)
                result = self._run_single_experiment(method, sweep_name, value)
                self.results[exp_id] = result

                # Print result
                if result.status == "success":
                    error_red = result.result.get("error_reduction", 0)
                    iters = result.result.get("num_iterations", 0)
                    print(
                        f"COMPLETED (error_reduction={error_red:.1f}%, "
                        f"{iters} iters, {result.wall_time:.1f}s)"
                    )
                else:
                    print(
                        f"FAILED ({result.failure_type}: {result.error[:50]}...)"
                    )

                # Save after each experiment
                self._save_results(sweep_name)

        # Generate diagnostic report
        report_path = self.output_dir / f"{sweep_name}_diagnostic_report.md"
        results_dict = {exp_id: r.to_dict() for exp_id, r in self.results.items()}
        generate_diagnostic_report(results_dict, report_path, sweep_name)
        print(f"\nDiagnostic report saved to: {report_path}")

        return self.results

    def run_friction_sweep(self, resume: bool = True) -> Dict[str, ExperimentResult]:
        """Run friction scale sweep (PRIMARY experiment)."""
        return self.run_sweep("friction", resume=resume)

    def run_obs_frequency_sweep(self, resume: bool = True) -> Dict[str, ExperimentResult]:
        """Run observation frequency sweep."""
        return self.run_sweep("obs_freq", resume=resume)

    def run_obs_fraction_sweep(self, resume: bool = True) -> Dict[str, ExperimentResult]:
        """Run observation fraction sweep."""
        return self.run_sweep("obs_fraction", resume=resume)

    def run_noise_sweep(self, resume: bool = True) -> Dict[str, ExperimentResult]:
        """Run observation noise sweep."""
        return self.run_sweep("noise", resume=resume)

    def run_background_sweep(self, resume: bool = True) -> Dict[str, ExperimentResult]:
        """Run background error sweep."""
        return self.run_sweep("background", resume=resume)

    def run_bathymetry_sweep(self, resume: bool = True) -> Dict[str, ExperimentResult]:
        """Run bathymetry noise sweep (OPTIONAL)."""
        return self.run_sweep("bathymetry", resume=resume)

    def run_all_sweeps(self, resume: bool = True) -> Dict[str, Dict[str, ExperimentResult]]:
        """Run all parameter sweeps.

        Returns:
            Dictionary mapping sweep name to results
        """
        all_results = {}

        # Run friction sweep first (primary)
        print("\n" + "=" * 60)
        print("FRICTION SWEEP (PRIMARY)")
        print("=" * 60)
        all_results["friction"] = self.run_friction_sweep(resume=resume)

        # Run other sweeps
        for sweep_name in ["obs_freq", "obs_fraction", "noise", "background"]:
            print(f"\n{'=' * 60}")
            print(f"{sweep_name.upper()} SWEEP")
            print("=" * 60)
            self.results = {}  # Reset for new sweep
            all_results[sweep_name] = self.run_sweep(sweep_name, resume=resume)

        return all_results


def run_single_experiment_verbose(
    method: str,
    sweep_param: str,
    sweep_value: Any,
    base_config: Optional[ComparisonStudyConfig] = None,
    save_intermediate: bool = False,
    output_dir: Optional[Path] = None,
) -> ExperimentResult:
    """Run a single experiment with verbose diagnostics for debugging.

    This is useful for investigating specific failures.

    Args:
        method: "4dvar" or "dcwme"
        sweep_param: Parameter being swept
        sweep_value: Value for this experiment
        base_config: Base configuration (uses defaults if None)
        save_intermediate: If True, save intermediate states
        output_dir: Directory for debug output

    Returns:
        ExperimentResult with full diagnostics
    """
    if base_config is None:
        base_config = ComparisonStudyConfig(diagnostic_level="verbose")

    sweep_config = SweepConfig()
    runner = ComparisonRunner(base_config, sweep_config)

    # Override diagnostic level to verbose
    runner.base_config.diagnostic_level = "verbose"

    result = runner._run_single_experiment(method, sweep_param, sweep_value)

    # Save to debug directory if requested
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        debug_file = output_dir / f"{result.experiment_id}_debug.json"
        with open(debug_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Debug output saved to: {debug_file}")

    return result

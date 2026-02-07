"""
Diagnostics module for the comparison study.

Provides comprehensive diagnostics capture and analysis to understand
WHY solvers fail, not just THAT they fail.

Diagnostic Levels:
- minimal: Final cost, error reduction, wall time only (~0% overhead)
- standard: Cost/gradient history, cost breakdown, failure info (~5% overhead)
- verbose: Everything including per-timestep forward model health (~20% overhead)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path


@dataclass
class ExperimentDiagnostics:
    """Comprehensive diagnostics for debugging solver behavior.

    Captures optimization trajectory, cost function breakdown, and forward
    model health to enable diagnosis of solver failures.
    """

    # Optimization trajectory
    cost_history: List[float] = field(default_factory=list)
    gradient_norm_history: List[float] = field(default_factory=list)
    step_size_history: List[float] = field(default_factory=list)

    # Cost function breakdown (at final iteration)
    background_term: Optional[float] = None
    observation_term: Optional[float] = None

    # Forward model health (verbose mode only)
    min_depth_per_timestep: List[float] = field(default_factory=list)
    max_velocity_per_timestep: List[float] = field(default_factory=list)
    forward_solver_iterations: List[int] = field(default_factory=list)

    # Condition indicators (when available)
    estimated_condition_number: Optional[float] = None
    max_cfl_number: Optional[float] = None

    # Failure information
    failure_iteration: Optional[int] = None
    failure_traceback: Optional[str] = None
    last_valid_state: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "cost_history": self.cost_history,
            "gradient_norm_history": self.gradient_norm_history,
            "step_size_history": self.step_size_history,
            "background_term": self.background_term,
            "observation_term": self.observation_term,
            "min_depth_per_timestep": self.min_depth_per_timestep,
            "max_velocity_per_timestep": self.max_velocity_per_timestep,
            "forward_solver_iterations": self.forward_solver_iterations,
            "estimated_condition_number": self.estimated_condition_number,
            "max_cfl_number": self.max_cfl_number,
            "failure_iteration": self.failure_iteration,
            "failure_traceback": self.failure_traceback,
            # Don't serialize last_valid_state (too large)
            "has_last_valid_state": self.last_valid_state is not None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentDiagnostics":
        """Create from dictionary."""
        diag = cls()
        diag.cost_history = data.get("cost_history", [])
        diag.gradient_norm_history = data.get("gradient_norm_history", [])
        diag.step_size_history = data.get("step_size_history", [])
        diag.background_term = data.get("background_term")
        diag.observation_term = data.get("observation_term")
        diag.min_depth_per_timestep = data.get("min_depth_per_timestep", [])
        diag.max_velocity_per_timestep = data.get("max_velocity_per_timestep", [])
        diag.forward_solver_iterations = data.get("forward_solver_iterations", [])
        diag.estimated_condition_number = data.get("estimated_condition_number")
        diag.max_cfl_number = data.get("max_cfl_number")
        diag.failure_iteration = data.get("failure_iteration")
        diag.failure_traceback = data.get("failure_traceback")
        return diag

    def attach_to_cost_function(self, cost_function):
        """Hook into cost function to capture per-iteration data.

        This monkey-patches the cost function's value_gradient method
        to record cost and gradient norm at each iteration.
        """
        original_vg = cost_function.value_gradient

        def instrumented_vg(m, g=None):
            cost, grad = original_vg(m, g)
            self.cost_history.append(float(cost))
            if grad is not None:
                try:
                    grad_arr = grad.getArray(readonly=True)
                    self.gradient_norm_history.append(float(np.linalg.norm(grad_arr)))
                except Exception:
                    # If we can't read the gradient, just skip
                    pass
            return cost, grad

        cost_function.value_gradient = instrumented_vg

    def capture_final_state(self, cost_function):
        """Capture cost breakdown from final iteration.

        Attempts to extract background and observation terms separately.
        This is cost-function specific and may not work for all implementations.
        """
        # Try to get cost breakdown (implementation-specific)
        if hasattr(cost_function, "J_b"):
            self.background_term = float(cost_function.J_b)
        if hasattr(cost_function, "J_o"):
            self.observation_term = float(cost_function.J_o)

        # Alternative: try to get from last cost evaluation
        if hasattr(cost_function, "_last_background_term"):
            self.background_term = float(cost_function._last_background_term)
        if hasattr(cost_function, "_last_observation_term"):
            self.observation_term = float(cost_function._last_observation_term)


def classify_failure(diagnostics: Dict[str, Any]) -> str:
    """Classify the failure mode based on diagnostic patterns.

    Returns a string describing the failure type to help with debugging.

    Failure types:
    - immediate_failure: Failed before any optimization
    - gradient_explosion: Gradient became NaN or inf
    - gradient_divergence: Gradient increased rapidly
    - negative_depth: Forward model produced negative depths
    - forward_model_failure: Cost function returned inf
    - slow_convergence: Ran out of iterations without converging
    - unknown: Couldn't determine failure mode
    """
    grad_hist = diagnostics.get("gradient_norm_history", [])
    cost_hist = diagnostics.get("cost_history", [])
    min_depths = diagnostics.get("min_depth_per_timestep", [])

    if not grad_hist and not cost_hist:
        return "immediate_failure"

    # Check for gradient explosion (NaN or inf)
    if grad_hist:
        for g in grad_hist:
            if np.isnan(g) or np.isinf(g):
                return "gradient_explosion"

        # Check for gradient divergence (10x increase between iterations)
        if len(grad_hist) >= 2 and grad_hist[-1] > grad_hist[-2] * 10:
            return "gradient_divergence"

    # Check for negative depths
    if min_depths and min(min_depths) < 0:
        return "negative_depth"

    # Check for forward model failure (inf cost)
    if cost_hist:
        for c in cost_hist:
            if np.isinf(c):
                return "forward_model_failure"

    # Check for slow convergence (if gradient is still large)
    if grad_hist and len(grad_hist) > 0:
        if grad_hist[-1] > 1e-3:  # Arbitrary threshold
            return "slow_convergence"

    return "unknown"


def compute_failure_thresholds(
    results: Dict[str, Any], parameter_name: str = "friction_scale"
) -> Dict[str, Optional[float]]:
    """Identify the parameter value at which each method first fails.

    Args:
        results: Dictionary of experiment results
        parameter_name: Name of the parameter to analyze

    Returns:
        Dictionary mapping method name to failure threshold (None if no failure)
    """
    thresholds = {"4dvar": None, "dcwme": None}

    for method in ["4dvar", "dcwme"]:
        # Find all experiments for this method, sorted by parameter value
        method_experiments = [
            (exp_id, r)
            for exp_id, r in results.items()
            if r.get("method") == method and parameter_name in r
        ]
        method_experiments.sort(key=lambda x: x[1].get(parameter_name, 0))

        # Find first failure
        for exp_id, r in method_experiments:
            if r.get("status") == "failed":
                thresholds[method] = r.get(parameter_name)
                break

    return thresholds


def generate_diagnostic_report(
    results: Dict[str, Any], output_path: Path, parameter_name: str = "friction_scale"
) -> str:
    """Generate human-readable diagnostic report.

    Args:
        results: Dictionary of experiment results
        output_path: Path to write the report
        parameter_name: Primary parameter being swept

    Returns:
        Report content as string
    """
    lines = ["# Experiment Diagnostic Report\n"]

    # Summary statistics
    n_success = sum(1 for r in results.values() if r.get("status") == "success")
    n_failed = sum(1 for r in results.values() if r.get("status") == "failed")
    lines.append(f"## Summary\n")
    lines.append(f"- Total experiments: {n_success + n_failed}")
    lines.append(f"- Successful: {n_success}")
    lines.append(f"- Failed: {n_failed}\n")

    # Failure analysis
    if n_failed > 0:
        lines.append("## Failed Experiments\n")
        for exp_id, r in results.items():
            if r.get("status") == "failed":
                diag = r.get("diagnostics", {})
                lines.append(f"### {exp_id}\n")
                lines.append(f"- **Error**: `{r.get('error', 'Unknown')}`")
                lines.append(f"- **Failure type**: {r.get('failure_type', 'unknown')}")
                lines.append(
                    f"- **Failed at iteration**: {diag.get('failure_iteration', 'N/A')}"
                )

                grad_hist = diag.get("gradient_norm_history", [])
                if grad_hist:
                    lines.append(f"- **Last gradient norm**: {grad_hist[-1]:.2e}")

                lines.append(
                    f"- **Background term**: {diag.get('background_term', 'N/A')}"
                )
                lines.append(
                    f"- **Observation term**: {diag.get('observation_term', 'N/A')}"
                )

                if diag.get("failure_traceback"):
                    lines.append(
                        f"\n<details><summary>Full traceback</summary>\n\n```\n{diag['failure_traceback']}\n```\n</details>\n"
                    )
                lines.append("")

    # Method comparison
    lines.append("## Method Robustness Comparison\n")
    thresholds = compute_failure_thresholds(results, parameter_name)
    for method, threshold in thresholds.items():
        if threshold is not None:
            lines.append(f"- **{method.upper()}**: Failed at {parameter_name}={threshold}")
        else:
            lines.append(f"- **{method.upper()}**: Completed all experiments")

    # Robustness interpretation
    if thresholds["4dvar"] and thresholds["dcwme"]:
        ratio = thresholds["dcwme"] / thresholds["4dvar"]
        lines.append(
            f"\n**Interpretation**: DC-WME is {(ratio - 1) * 100:.0f}% more robust to {parameter_name} perturbation."
        )
    elif thresholds["4dvar"] and not thresholds["dcwme"]:
        lines.append(
            f"\n**Interpretation**: DC-WME completed all experiments while 4D-Var failed at {parameter_name}={thresholds['4dvar']}."
        )
    elif not thresholds["4dvar"] and not thresholds["dcwme"]:
        lines.append("\n**Interpretation**: Both methods completed all experiments.")

    # Gradient convergence summary
    lines.append("\n## Gradient Convergence Summary\n")
    lines.append("| Experiment | Method | Initial Grad | Final Grad | Ratio |")
    lines.append("|------------|--------|-------------|------------|-------|")
    for exp_id, r in sorted(results.items()):
        if r.get("status") == "success":
            diag = r.get("diagnostics", {})
            grad_hist = diag.get("gradient_norm_history", [])
            if len(grad_hist) >= 2:
                initial = grad_hist[0]
                final = grad_hist[-1]
                ratio = final / initial if initial > 0 else float("inf")
                lines.append(
                    f"| {exp_id} | {r.get('method', 'N/A')} | {initial:.2e} | {final:.2e} | {ratio:.2e} |"
                )

    report_content = "\n".join(lines)

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_content)

    return report_content


# Failure type descriptions for reference
FAILURE_TYPE_DESCRIPTIONS = {
    "immediate_failure": "Failed before any optimization began. Usually indicates problem setup issue.",
    "gradient_explosion": "Gradient became NaN or inf. Indicates unstable adjoint or ill-conditioned Jacobian.",
    "gradient_divergence": "Gradient increased rapidly (10x+ between iterations). Indicates optimizer instability.",
    "negative_depth": "Forward model produced negative water depths. Physics perturbation too extreme.",
    "forward_model_failure": "Cost function returned inf. Forward model produced invalid states.",
    "slow_convergence": "Ran out of iterations without converging. May need more iterations or different optimizer settings.",
    "unknown": "Could not determine failure mode from diagnostics.",
}


def get_failure_fix_suggestion(failure_type: str) -> str:
    """Get a suggested fix for a failure type."""
    suggestions = {
        "immediate_failure": "Check problem setup, background error std, and initial condition.",
        "gradient_explosion": "Reduce physics perturbation magnitude. Try friction_scale closer to 1.0.",
        "gradient_divergence": "Reduce optimizer step size or use more conservative line search.",
        "negative_depth": "Reduce bathymetry_noise_std. Ensure h_min bound is enforced.",
        "forward_model_failure": "Reduce physics perturbation. Check for extreme parameter values.",
        "slow_convergence": "Increase max_iterations or relax gradient_tolerance.",
        "unknown": "Enable verbose diagnostics and re-run to capture more information.",
    }
    return suggestions.get(failure_type, "No suggestion available.")

#!/usr/bin/env python
"""
Test Inverse Crime Avoidance in Twin Experiments

This script tests the twin experiment framework with physics perturbations
to avoid the "inverse crime" - using the exact same model for both truth
generation and data assimilation.

Experimental Design Matrix:
| Experiment | Bathymetry | Friction | Purpose                      |
|------------|------------|----------|------------------------------|
| Baseline   | Same       | Same     | Sanity check (inverse crime) |
| A          | Perturbed  | Same     | Bathymetry error only        |
| B          | Same       | Perturbed| Friction error only          |
| C          | Perturbed  | Perturbed| Combined (realistic)         |

Each experiment is run with both 4D-Var and DC-WME methods.

LIMITATIONS:
- TidalProblem uses solution_var="h" (total depth), which makes bathymetry
  perturbation problematic since the initial condition becomes inconsistent.
- TidalProblem's friction is set as a float in make_Friction, so friction
  perturbation also doesn't work correctly.
- For proper inverse crime testing, use ADCIRCProblem/Shinnecock which uses
  Functions for bathymetry and friction.

Usage:
    python test_inverse_crime.py
    python test_inverse_crime.py --save-plots
    python test_inverse_crime.py --nx 10 --ny 5 --nt 24
"""

import argparse
import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.twin_experiment import TwinExperiment, TwinExperimentConfig
from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    name: str
    method: str
    background_error: float
    analysis_error: float
    error_reduction: float
    converged: bool
    num_iterations: int
    cost_history: list


def create_problem_and_solver(nx: int, ny: int, dt: float, nt: int, solver_type: str = "SUPG"):
    """Create a fresh problem and solver instance."""
    problem = TidalProblem(nx=nx, ny=ny, dt=dt, nt=nt)
    if solver_type.upper() == "DG":
        solver = get_solver("DG")(problem, theta=0.5, p_degree=[1, 1], verbose=False)
    else:
        solver = get_solver("SUPG")(problem, theta=0.5, p_degree=[1, 1], verbose=False)
    return problem, solver


def run_experiment(
    name: str,
    config: TwinExperimentConfig,
    nx: int,
    ny: int,
    dt: float,
    nt: int,
    verbose: bool = True,
    solver_type: str = "SUPG",
) -> ExperimentResult:
    """Run a single twin experiment and return results."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

    problem, solver = create_problem_and_solver(nx, ny, dt, nt, solver_type)
    experiment = TwinExperiment(problem, solver, config)
    results = experiment.run()

    if verbose:
        print(f"\nResults for {name}:")
        print(f"  Background error: {results.background_error:.6f}")
        print(f"  Analysis error:   {results.analysis_error:.6f}")
        print(f"  Error reduction:  {results.error_reduction:.1f}%")
        print(f"  Converged:        {results.converged}")
        print(f"  Iterations:       {results.num_iterations}")

    return ExperimentResult(
        name=name,
        method=config.method,
        background_error=results.background_error,
        analysis_error=results.analysis_error,
        error_reduction=results.error_reduction,
        converged=results.converged,
        num_iterations=results.num_iterations,
        cost_history=results.cost_history,
    )


def run_all_experiments(
    nx: int = 5,
    ny: int = 3,
    dt: float = 3600.0,
    nt: int = 12,
    bathymetry_noise_std: float = 0.5,
    bathymetry_correlation_length: float = 500.0,
    friction_scale_factor: float = 1.15,
    max_iterations: int = 30,
    verbose: bool = True,
    skip_perturbation: bool = True,
    friction_only: bool = False,
    solver_type: str = "SUPG",
) -> Dict[str, Dict[str, ExperimentResult]]:
    """
    Run all experiments from the inverse crime avoidance matrix.

    Args:
        skip_perturbation: If True, only run baseline experiments (no physics
            perturbation). Set to False for problems that support perturbation
            like ADCIRCProblem/Shinnecock.
        friction_only: If True, only test friction perturbation (skip bathymetry).
            Useful for TidalProblem which has issues with bathymetry perturbation.

    Returns:
        Dictionary with structure: {method: {experiment_name: result}}
    """
    # Common experiment settings
    common_config = {
        "obs_fraction": 0.5,
        "obs_frequency": 1,
        "obs_noise_level": 0.01,
        "background_error_std": 0.1,
        "max_iterations": max_iterations,
        "verbose": False,
        "obs_seed": 42,
        "background_seed": 123,
        "perturbation_seed": 456,
    }

    # Define experiment configurations
    if skip_perturbation:
        # Only run baseline for TidalProblem (perturbation not supported)
        experiments = {
            "Baseline": {
                "perturb_bathymetry": False,
                "perturb_friction": False,
            },
        }
    elif friction_only:
        # Friction perturbation only (for TidalProblem which has bathy issues)
        experiments = {
            "Baseline": {
                "perturb_bathymetry": False,
                "perturb_friction": False,
            },
            "B: Friction": {
                "perturb_bathymetry": False,
                "perturb_friction": True,
                "friction_scale_factor": friction_scale_factor,
            },
        }
    else:
        experiments = {
            "Baseline": {
                "perturb_bathymetry": False,
                "perturb_friction": False,
            },
            "A: Bathy": {
                "perturb_bathymetry": True,
                "bathymetry_noise_std": bathymetry_noise_std,
                "bathymetry_noise_type": "additive",
                "bathymetry_correlation_length": bathymetry_correlation_length,
                "perturb_friction": False,
            },
            "B: Friction": {
                "perturb_bathymetry": False,
                "perturb_friction": True,
                "friction_scale_factor": friction_scale_factor,
            },
            "C: Combined": {
                "perturb_bathymetry": True,
                "bathymetry_noise_std": bathymetry_noise_std,
                "bathymetry_noise_type": "additive",
                "bathymetry_correlation_length": bathymetry_correlation_length,
                "perturb_friction": True,
                "friction_scale_factor": friction_scale_factor,
            },
        }

    methods = ["4dvar", "dcwme"]
    all_results = {method: {} for method in methods}

    if verbose:
        print("\n" + "=" * 70)
        print("INVERSE CRIME AVOIDANCE TEST")
        print("=" * 70)
        print(f"Problem: {nx}x{ny} elements, dt={dt}s, nt={nt}")
        print(f"Solver: {solver_type.upper()}")
        if skip_perturbation:
            print("NOTE: Physics perturbation disabled for TidalProblem")
            print("      Use --enable-perturbation or --friction-only for perturbation tests")
        elif friction_only:
            print("NOTE: Friction-only perturbation mode (bathymetry skipped)")
            print(f"Friction perturbation: {friction_scale_factor}x scale factor")
        else:
            print(f"Bathymetry perturbation: {bathymetry_noise_std}m additive noise")
            print(f"Friction perturbation: {friction_scale_factor}x scale factor")
        print("=" * 70)

    # Run all experiments
    for method in methods:
        for exp_name, exp_config in experiments.items():
            config = TwinExperimentConfig(
                **common_config,
                method=method,
                **exp_config,
            )
            full_name = f"{exp_name} ({method.upper()})"
            result = run_experiment(full_name, config, nx, ny, dt, nt, verbose, solver_type)
            all_results[method][exp_name] = result

    return all_results


def print_summary(results: Dict[str, Dict[str, ExperimentResult]]):
    """Print summary table of all results."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(
        f"{'Experiment':<15} {'Method':<8} {'Bg Error':<12} {'An Error':<12} "
        f"{'Reduction':<12} {'Converged'}"
    )
    print("-" * 80)

    for method in ["4dvar", "dcwme"]:
        for exp_name, result in results[method].items():
            print(
                f"{exp_name:<15} {method.upper():<8} {result.background_error:<12.6f} "
                f"{result.analysis_error:<12.6f} {result.error_reduction:<12.1f}% "
                f"{result.converged}"
            )

    print("=" * 80)


def print_analysis(results: Dict[str, Dict[str, ExperimentResult]]):
    """Print analysis of model error impact."""
    experiments = list(results["4dvar"].keys())

    # Only show model error analysis if we have perturbation experiments
    if len(experiments) > 1:
        print("\n" + "=" * 60)
        print("ANALYSIS: Impact of Model Error on DA Performance")
        print("=" * 60)

        for method in ["4dvar", "dcwme"]:
            print(f"\n{method.upper()}:")
            baseline_reduction = results[method]["Baseline"].error_reduction

            for exp_name, result in results[method].items():
                if exp_name != "Baseline":
                    degradation = baseline_reduction - result.error_reduction
                    print(f"  {exp_name}: {degradation:.1f}% degradation from baseline")

    # Compare methods
    print("\n" + "=" * 60)
    print("ROBUSTNESS COMPARISON: DC-WME vs 4D-Var")
    print("=" * 60)

    for exp_name in experiments:
        reduction_4dvar = results["4dvar"][exp_name].error_reduction
        reduction_dcwme = results["dcwme"][exp_name].error_reduction
        diff = reduction_dcwme - reduction_4dvar

        if diff > 0:
            better = "DC-WME"
        elif diff < 0:
            better = "4D-Var"
        else:
            better = "Tie"

        print(f"{exp_name}: {better} better by {abs(diff):.1f}% error reduction")


def save_plots(results: Dict[str, Dict[str, ExperimentResult]], output_dir: str = "outputs/figures"):
    """Save comparison plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    experiments = list(results["4dvar"].keys())
    x = np.arange(len(experiments))
    width = 0.35

    # Plot 1: Error comparison for each method
    for method in ["4dvar", "dcwme"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        bg_errors = [results[method][exp].background_error for exp in experiments]
        an_errors = [results[method][exp].analysis_error for exp in experiments]
        reductions = [results[method][exp].error_reduction for exp in experiments]

        # Errors
        axes[0].bar(x - width / 2, bg_errors, width, label="Background Error", alpha=0.8)
        axes[0].bar(x + width / 2, an_errors, width, label="Analysis Error", alpha=0.8)
        axes[0].set_xlabel("Experiment")
        axes[0].set_ylabel("RMS Error")
        axes[0].set_title(f"{method.upper()}: Errors")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(experiments, rotation=45, ha="right")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)

        # Error reduction
        colors = ["green" if r > 0 else "red" for r in reductions]
        axes[1].bar(x, reductions, color=colors, alpha=0.8)
        axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[1].set_xlabel("Experiment")
        axes[1].set_ylabel("Error Reduction (%)")
        axes[1].set_title(f"{method.upper()}: Error Reduction")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(experiments, rotation=45, ha="right")
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"inverse_crime_{method}.png"), dpi=150)
        plt.close()

    # Plot 2: Method comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    reductions_4dvar = [results["4dvar"][exp].error_reduction for exp in experiments]
    reductions_dcwme = [results["dcwme"][exp].error_reduction for exp in experiments]

    bars1 = ax.bar(x - width / 2, reductions_4dvar, width, label="4D-Var", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width / 2, reductions_dcwme, width, label="DC-WME", color="darkorange", alpha=0.8)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Error Reduction (%)")
    ax.set_title("Error Reduction: 4D-Var vs DC-WME")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars1, reductions_4dvar):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, val in zip(bars2, reductions_dcwme):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inverse_crime_comparison.png"), dpi=150)
    plt.close()

    # Plot 3: Convergence history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for method, ax in zip(["4dvar", "dcwme"], axes):
        for exp_name, result in results[method].items():
            if result.cost_history:
                normalized_cost = np.array(result.cost_history) / result.cost_history[0]
                ax.semilogy(normalized_cost, label=exp_name, linewidth=2)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Normalized Cost (J/J₀)")
        ax.set_title(f"{method.upper()} Convergence History")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inverse_crime_convergence.png"), dpi=150)
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Test inverse crime avoidance in twin experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Problem parameters
    parser.add_argument("--nx", type=int, default=5, help="Elements in x direction (default: 5)")
    parser.add_argument("--ny", type=int, default=3, help="Elements in y direction (default: 3)")
    parser.add_argument("--dt", type=float, default=3600.0, help="Time step in seconds (default: 3600)")
    parser.add_argument("--nt", type=int, default=12, help="Number of timesteps (default: 12)")

    # Perturbation parameters
    parser.add_argument(
        "--bathy-noise", type=float, default=0.5, help="Bathymetry noise std in meters (default: 0.5)"
    )
    parser.add_argument(
        "--bathy-corr-len", type=float, default=500.0, help="Bathymetry correlation length (default: 500)"
    )
    parser.add_argument(
        "--friction-scale", type=float, default=1.15, help="Friction scale factor (default: 1.15)"
    )

    # Optimization parameters
    parser.add_argument("--max-iter", type=int, default=30, help="Maximum iterations (default: 30)")

    # Output options
    parser.add_argument("--save-plots", action="store_true", help="Save comparison plots")
    parser.add_argument("--output-dir", type=str, default="outputs/figures", help="Output directory for plots")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    # Perturbation options
    parser.add_argument(
        "--enable-perturbation",
        action="store_true",
        help="Enable physics perturbation experiments (for problems that support it)"
    )
    parser.add_argument(
        "--friction-only",
        action="store_true",
        help="Test only friction perturbation (skip bathymetry experiments)"
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="SUPG",
        choices=["SUPG", "DG"],
        help="Solver type (default: SUPG)"
    )

    args = parser.parse_args()

    # Run experiments
    results = run_all_experiments(
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        nt=args.nt,
        bathymetry_noise_std=args.bathy_noise,
        bathymetry_correlation_length=args.bathy_corr_len,
        friction_scale_factor=args.friction_scale,
        max_iterations=args.max_iter,
        verbose=not args.quiet,
        skip_perturbation=not args.enable_perturbation and not args.friction_only,
        friction_only=args.friction_only,
        solver_type=args.solver,
    )

    # Print results
    print_summary(results)
    print_analysis(results)

    # Save plots if requested
    if args.save_plots:
        save_plots(results, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()

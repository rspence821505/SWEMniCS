#!/usr/bin/env python3
"""
Analyze and compare results from serial DA experiments.

This script loads results from all experiments and generates:
1. Cost convergence comparison plots
2. Analysis vs truth error comparison
3. Innovation statistics
4. Summary tables

Outputs are saved to outputs/figures/ and outputs/data/.

Usage:
    python analyze_results.py [--output-dir outputs]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Try to import matplotlib (may not be available in all environments)
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plots will not be generated.")

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.utils.output_paths import FIGURES_DIR, DATA_DIR, ensure_output_dirs

from da_experiment_utils import (
    DAExperimentResults,
    load_all_results,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze serial DA experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory"
    )
    return parser.parse_args()


def load_results(data_dir: Path) -> Dict[str, DAExperimentResults]:
    """
    Load all experiment results from JSON files.

    Parameters
    ----------
    data_dir : Path
        Directory containing result JSON files.

    Returns
    -------
    results : Dict[str, DAExperimentResults]
        Dictionary mapping experiment name to results.
    """
    results = {}

    # Expected result files
    expected_files = [
        ("tidal_4dvar", "tidal_4dvar_results.json"),
        ("tidal_dcwme", "tidal_dcwme_results.json"),
        ("dam_break_4dvar", "dam_break_4dvar_results.json"),
        ("dam_break_dcwme", "dam_break_dcwme_results.json"),
    ]

    for name, filename in expected_files:
        filepath = data_dir / filename
        if filepath.exists():
            results[name] = DAExperimentResults.load(str(filepath))
            print(f"  Loaded: {filename}")
        else:
            print(f"  Missing: {filename}")

    return results


def plot_cost_convergence(
    results: Dict[str, DAExperimentResults],
    figures_dir: Path
) -> Optional[str]:
    """
    Plot cost function convergence for all experiments.

    Parameters
    ----------
    results : Dict[str, DAExperimentResults]
        Experiment results.
    figures_dir : Path
        Output directory for figures.

    Returns
    -------
    filepath : str or None
        Path to saved figure, or None if matplotlib unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Colors and styles
    colors = {
        "4dvar": "blue",
        "dcwme": "red"
    }
    linestyles = {
        "4dvar": "-",
        "dcwme": "--"
    }

    # Tidal case
    ax = axes[0]
    ax.set_title("Tidal Case: Cost Convergence", fontsize=12)
    for key, res in results.items():
        if "tidal" in key and res.cost_history:
            method = "4dvar" if "4dvar" in key else "dcwme"
            label = "4D-Var" if method == "4dvar" else "DC-WME-4DVar"
            ax.semilogy(
                res.cost_history,
                color=colors[method],
                linestyle=linestyles[method],
                label=label,
                linewidth=2
            )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost Function")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Dam break case
    ax = axes[1]
    ax.set_title("Dam Break Case: Cost Convergence", fontsize=12)
    for key, res in results.items():
        if "dam_break" in key and res.cost_history:
            method = "4dvar" if "4dvar" in key else "dcwme"
            label = "4D-Var" if method == "4dvar" else "DC-WME-4DVar"
            ax.semilogy(
                res.cost_history,
                color=colors[method],
                linestyle=linestyles[method],
                label=label,
                linewidth=2
            )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost Function")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = figures_dir / "cost_convergence_comparison.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return str(filepath)


def plot_error_comparison(
    results: Dict[str, DAExperimentResults],
    figures_dir: Path
) -> Optional[str]:
    """
    Plot background vs analysis error comparison.

    Parameters
    ----------
    results : Dict[str, DAExperimentResults]
        Experiment results.
    figures_dir : Path
        Output directory for figures.

    Returns
    -------
    filepath : str or None
        Path to saved figure, or None if matplotlib unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Tidal case
    ax = axes[0]
    ax.set_title("Tidal Case: Error Reduction", fontsize=12)

    tidal_results = {k: v for k, v in results.items() if "tidal" in k}
    if tidal_results:
        methods = []
        background_errors = []
        analysis_errors = []

        for key in ["tidal_4dvar", "tidal_dcwme"]:
            if key in tidal_results:
                res = tidal_results[key]
                label = "4D-Var" if "4dvar" in key else "DC-WME"
                methods.append(label)
                background_errors.append(res.background_error)
                analysis_errors.append(res.analysis_error)

        x = np.arange(len(methods))
        width = 0.35

        bars1 = ax.bar(x - width/2, background_errors, width, label='Background', color='gray')
        bars2 = ax.bar(x + width/2, analysis_errors, width, label='Analysis', color='steelblue')

        ax.set_xlabel("Method")
        ax.set_ylabel("RMS Error")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # Dam break case
    ax = axes[1]
    ax.set_title("Dam Break Case: Error Reduction", fontsize=12)

    dam_results = {k: v for k, v in results.items() if "dam_break" in k}
    if dam_results:
        methods = []
        background_errors = []
        analysis_errors = []

        for key in ["dam_break_4dvar", "dam_break_dcwme"]:
            if key in dam_results:
                res = dam_results[key]
                label = "4D-Var" if "4dvar" in key else "DC-WME"
                methods.append(label)
                background_errors.append(res.background_error)
                analysis_errors.append(res.analysis_error)

        x = np.arange(len(methods))
        width = 0.35

        bars1 = ax.bar(x - width/2, background_errors, width, label='Background', color='gray')
        bars2 = ax.bar(x + width/2, analysis_errors, width, label='Analysis', color='steelblue')

        ax.set_xlabel("Method")
        ax.set_ylabel("RMS Error")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    filepath = figures_dir / "error_comparison.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return str(filepath)


def plot_gradient_convergence(
    results: Dict[str, DAExperimentResults],
    figures_dir: Path
) -> Optional[str]:
    """
    Plot gradient norm convergence for all experiments.

    Parameters
    ----------
    results : Dict[str, DAExperimentResults]
        Experiment results.
    figures_dir : Path
        Output directory for figures.

    Returns
    -------
    filepath : str or None
        Path to saved figure, or None if matplotlib unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"4dvar": "blue", "dcwme": "red"}
    linestyles = {"4dvar": "-", "dcwme": "--"}

    # Tidal case
    ax = axes[0]
    ax.set_title("Tidal Case: Gradient Convergence", fontsize=12)
    for key, res in results.items():
        if "tidal" in key and res.gradient_norm_history:
            method = "4dvar" if "4dvar" in key else "dcwme"
            label = "4D-Var" if method == "4dvar" else "DC-WME-4DVar"
            ax.semilogy(
                res.gradient_norm_history,
                color=colors[method],
                linestyle=linestyles[method],
                label=label,
                linewidth=2
            )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Dam break case
    ax = axes[1]
    ax.set_title("Dam Break Case: Gradient Convergence", fontsize=12)
    for key, res in results.items():
        if "dam_break" in key and res.gradient_norm_history:
            method = "4dvar" if "4dvar" in key else "dcwme"
            label = "4D-Var" if method == "4dvar" else "DC-WME-4DVar"
            ax.semilogy(
                res.gradient_norm_history,
                color=colors[method],
                linestyle=linestyles[method],
                label=label,
                linewidth=2
            )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = figures_dir / "gradient_convergence_comparison.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return str(filepath)


def plot_innovation_statistics(
    results: Dict[str, DAExperimentResults],
    figures_dir: Path
) -> Optional[str]:
    """
    Plot innovation statistics comparison.

    Parameters
    ----------
    results : Dict[str, DAExperimentResults]
        Experiment results.
    figures_dir : Path
        Output directory for figures.

    Returns
    -------
    filepath : str or None
        Path to saved figure, or None if matplotlib unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Innovation mean
    ax = axes[0]
    ax.set_title("Innovation Mean (Bias)", fontsize=12)

    all_methods = []
    all_means = []
    all_colors = []

    for key in ["tidal_4dvar", "tidal_dcwme", "dam_break_4dvar", "dam_break_dcwme"]:
        if key in results:
            res = results[key]
            case = "Tidal" if "tidal" in key else "Dam"
            method = "4D" if "4dvar" in key else "WME"
            all_methods.append(f"{case}\n{method}")
            all_means.append(res.innovation_mean)
            all_colors.append("steelblue" if "4dvar" in key else "coral")

    x = np.arange(len(all_methods))
    ax.bar(x, all_means, color=all_colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Innovation Mean")
    ax.set_xticks(x)
    ax.set_xticklabels(all_methods)
    ax.grid(True, alpha=0.3, axis='y')

    # Innovation std
    ax = axes[1]
    ax.set_title("Innovation Std (Spread)", fontsize=12)

    all_stds = []
    for key in ["tidal_4dvar", "tidal_dcwme", "dam_break_4dvar", "dam_break_dcwme"]:
        if key in results:
            all_stds.append(results[key].innovation_std)

    ax.bar(x, all_stds, color=all_colors)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Innovation Std")
    ax.set_xticks(x)
    ax.set_xticklabels(all_methods)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    filepath = figures_dir / "innovation_statistics.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return str(filepath)


def generate_summary_table(results: Dict[str, DAExperimentResults]) -> str:
    """
    Generate a text summary table of all results.

    Parameters
    ----------
    results : Dict[str, DAExperimentResults]
        Experiment results.

    Returns
    -------
    table : str
        Formatted summary table.
    """
    lines = []
    lines.append("=" * 90)
    lines.append("SERIAL DA EXPERIMENT RESULTS SUMMARY")
    lines.append("=" * 90)
    lines.append("")

    # Header
    header = f"{'Experiment':<25} {'Bg Error':>12} {'An Error':>12} {'Reduction':>10} {'Iter':>6} {'Conv':>6} {'Time(s)':>10}"
    lines.append(header)
    lines.append("-" * 90)

    # Results by experiment
    for key in ["tidal_4dvar", "tidal_dcwme", "dam_break_4dvar", "dam_break_dcwme"]:
        if key in results:
            res = results[key]
            name = key.replace("_", " ").title()
            conv = "Yes" if res.converged else "No"
            line = f"{name:<25} {res.background_error:>12.6f} {res.analysis_error:>12.6f} {res.error_reduction:>9.1f}% {res.num_iterations:>6} {conv:>6} {res.wall_time:>10.2f}"
            lines.append(line)

    lines.append("-" * 90)
    lines.append("")

    # Method comparison
    lines.append("METHOD COMPARISON")
    lines.append("-" * 40)

    # Tidal comparison
    if "tidal_4dvar" in results and "tidal_dcwme" in results:
        r1, r2 = results["tidal_4dvar"], results["tidal_dcwme"]
        lines.append(f"Tidal Case:")
        lines.append(f"  4D-Var error reduction:     {r1.error_reduction:.1f}%")
        lines.append(f"  DC-WME error reduction:     {r2.error_reduction:.1f}%")
        diff = r2.error_reduction - r1.error_reduction
        lines.append(f"  DC-WME advantage:           {diff:+.1f}%")
        lines.append("")

    # Dam break comparison
    if "dam_break_4dvar" in results and "dam_break_dcwme" in results:
        r1, r2 = results["dam_break_4dvar"], results["dam_break_dcwme"]
        lines.append(f"Dam Break Case:")
        lines.append(f"  4D-Var error reduction:     {r1.error_reduction:.1f}%")
        lines.append(f"  DC-WME error reduction:     {r2.error_reduction:.1f}%")
        diff = r2.error_reduction - r1.error_reduction
        lines.append(f"  DC-WME advantage:           {diff:+.1f}%")
        lines.append("")

    lines.append("=" * 90)

    return "\n".join(lines)


def save_combined_results(
    results: Dict[str, DAExperimentResults],
    data_dir: Path
) -> str:
    """
    Save combined results to a single JSON file.

    Parameters
    ----------
    results : Dict[str, DAExperimentResults]
        All experiment results.
    data_dir : Path
        Output directory.

    Returns
    -------
    filepath : str
        Path to saved file.
    """
    combined = {
        "experiments": {k: v.to_dict() for k, v in results.items()},
        "summary": {
            "total_experiments": len(results),
            "methods": ["4dvar", "dcwme"],
            "test_cases": ["tidal", "dam_break"],
        }
    }

    filepath = data_dir / "serial_da_results.json"
    with open(filepath, 'w') as f:
        json.dump(combined, f, indent=2)

    return str(filepath)


def main():
    """Run analysis of serial DA experiments."""
    args = parse_args()

    # Ensure output directories exist
    ensure_output_dirs()

    data_dir = Path(args.output_dir) / "data"
    figures_dir = Path(args.output_dir) / "figures"

    print("=" * 70)
    print("Serial DA Experiment Analysis")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Figures directory: {figures_dir}")
    print("")

    # Load results
    print("Loading experiment results...")
    results = load_results(data_dir)

    if not results:
        print("\nNo results found. Please run the experiments first.")
        print("Use: ./run_serial_experiments.sh")
        return

    print(f"\nLoaded {len(results)} experiment results.")

    # Generate figures
    print("\nGenerating figures...")

    if MATPLOTLIB_AVAILABLE:
        filepath = plot_cost_convergence(results, figures_dir)
        if filepath:
            print(f"  - Cost convergence: {filepath}")

        filepath = plot_error_comparison(results, figures_dir)
        if filepath:
            print(f"  - Error comparison: {filepath}")

        filepath = plot_gradient_convergence(results, figures_dir)
        if filepath:
            print(f"  - Gradient convergence: {filepath}")

        filepath = plot_innovation_statistics(results, figures_dir)
        if filepath:
            print(f"  - Innovation statistics: {filepath}")
    else:
        print("  (matplotlib not available - skipping plots)")

    # Generate summary table
    print("\nGenerating summary...")
    summary = generate_summary_table(results)
    print(summary)

    # Save combined results
    filepath = save_combined_results(results, data_dir)
    print(f"\nCombined results saved to: {filepath}")

    # Save summary to file
    summary_filepath = data_dir / "serial_da_summary.txt"
    with open(summary_filepath, 'w') as f:
        f.write(summary)
    print(f"Summary saved to: {summary_filepath}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

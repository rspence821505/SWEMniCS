"""
Plotting utilities for the comparison study.

Generates visualizations for comparing 4D-Var and DC-WME performance
across different parameter sweeps.

Key plots:
- Convergence comparison
- Error reduction bar charts
- Gradient analysis (for debugging)
- Summary heatmaps
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Try to import matplotlib, warn if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting functions will not work.")


def check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )


def load_sweep_results(filepath: Path) -> Dict[str, Any]:
    """Load sweep results from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    # Remove metadata
    data.pop("_metadata", None)
    return data


def get_successful_experiments(results: Dict[str, Any]) -> Dict[str, Any]:
    """Filter to only successful experiments."""
    return {k: v for k, v in results.items() if v.get("status") == "success"}


def get_failed_experiments(results: Dict[str, Any]) -> Dict[str, Any]:
    """Filter to only failed experiments."""
    return {k: v for k, v in results.items() if v.get("status") == "failed"}


def plot_convergence_comparison(
    results: Dict[str, Any],
    sweep_param: str,
    output_path: Path,
    title_prefix: str = "",
):
    """Plot cost convergence for all experiments in a sweep.

    Creates a grid of subplots showing cost history for each parameter value,
    with 4D-Var and DC-WME overlaid.

    Args:
        results: Dictionary of experiment results
        sweep_param: Name of the swept parameter (for labeling)
        output_path: Path to save the figure
        title_prefix: Prefix for the plot title
    """
    check_matplotlib()

    # Group by sweep value
    by_value: Dict[Any, Dict[str, Any]] = {}
    for exp_id, result in results.items():
        if result.get("status") != "success":
            continue
        value = result.get("sweep_value")
        if value not in by_value:
            by_value[value] = {}
        by_value[value][result.get("method")] = result

    if not by_value:
        print("No successful experiments to plot")
        return

    # Create subplot grid
    n_values = len(by_value)
    n_cols = min(2, n_values)
    n_rows = (n_values + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_values == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = {"4dvar": "blue", "dcwme": "red"}
    line_styles = {"4dvar": "-", "dcwme": "--"}

    for idx, (value, methods) in enumerate(sorted(by_value.items())):
        ax = axes[idx]

        for method, result in methods.items():
            cost_hist = result.get("result", {}).get("cost_history", [])
            if cost_hist:
                ax.semilogy(
                    cost_hist,
                    color=colors.get(method, "gray"),
                    linestyle=line_styles.get(method, "-"),
                    label=method.upper(),
                    linewidth=2,
                )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost (log scale)")
        ax.set_title(f"{sweep_param}={value}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(by_value), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"{title_prefix}Convergence Comparison: {sweep_param} Sweep", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_error_reduction_comparison(
    results: Dict[str, Any],
    sweep_param: str,
    output_path: Path,
    title_prefix: str = "",
):
    """Plot error reduction comparison as grouped bar chart.

    Args:
        results: Dictionary of experiment results
        sweep_param: Name of the swept parameter (for x-axis)
        output_path: Path to save the figure
        title_prefix: Prefix for the plot title
    """
    check_matplotlib()

    # Extract data by method and sweep value
    data: Dict[str, Dict[Any, float]] = {"4dvar": {}, "dcwme": {}}
    failed: Dict[str, Dict[Any, bool]] = {"4dvar": {}, "dcwme": {}}

    for exp_id, result in results.items():
        method = result.get("method")
        value = result.get("sweep_value")
        if method not in data:
            continue

        if result.get("status") == "success":
            error_red = result.get("result", {}).get("error_reduction", 0)
            data[method][value] = error_red
            failed[method][value] = False
        else:
            data[method][value] = 0
            failed[method][value] = True

    if not any(data.values()):
        print("No data to plot")
        return

    # Get all unique values and sort
    all_values = sorted(set(list(data["4dvar"].keys()) + list(data["dcwme"].keys())))
    x = np.arange(len(all_values))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    colors = {"4dvar": "steelblue", "dcwme": "coral"}
    hatch_failed = {"4dvar": "//", "dcwme": "\\\\"}

    for i, method in enumerate(["4dvar", "dcwme"]):
        heights = [data[method].get(v, 0) for v in all_values]
        is_failed = [failed[method].get(v, True) for v in all_values]

        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            heights,
            width,
            label=method.upper(),
            color=colors[method],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add hatch pattern for failed experiments
        for bar, fail in zip(bars, is_failed):
            if fail:
                bar.set_hatch(hatch_failed[method])
                bar.set_alpha(0.5)

    ax.set_xlabel(sweep_param)
    ax.set_ylabel("Error Reduction (%)")
    ax.set_title(f"{title_prefix}Error Reduction: {sweep_param} Sweep")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in all_values])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Add legend for failed experiments
    failed_patch = mpatches.Patch(
        facecolor="white", edgecolor="black", hatch="//", label="Failed"
    )
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [failed_patch])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_gradient_analysis(
    results: Dict[str, Any],
    sweep_param: str,
    output_path: Path,
    title_prefix: str = "",
):
    """Plot gradient norm histories for debugging.

    Shows gradient convergence patterns to identify instability.

    Args:
        results: Dictionary of experiment results
        sweep_param: Name of the swept parameter
        output_path: Path to save the figure
        title_prefix: Prefix for the plot title
    """
    check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, 10))

    for method, ax in zip(["4dvar", "dcwme"], axes):
        method_results = [
            (r.get("sweep_value"), r)
            for r in results.values()
            if r.get("method") == method
        ]
        method_results.sort(key=lambda x: x[0])

        for i, (value, result) in enumerate(method_results):
            diag = result.get("diagnostics", {})
            grad_hist = diag.get("gradient_norm_history", [])

            if not grad_hist:
                continue

            color = colors[i % len(colors)]
            is_failed = result.get("status") == "failed"
            linestyle = "--" if is_failed else "-"
            alpha = 0.5 if is_failed else 1.0

            label = f"{sweep_param}={value}"
            if is_failed:
                label += " (FAILED)"

            ax.semilogy(
                grad_hist,
                color=color,
                linestyle=linestyle,
                alpha=alpha,
                label=label,
                linewidth=2,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Gradient Norm (log scale)")
        ax.set_title(f"{method.upper()} Gradient Convergence")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{title_prefix}Gradient Analysis: {sweep_param} Sweep", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_method_robustness(
    results: Dict[str, Any],
    sweep_param: str,
    output_path: Path,
    title_prefix: str = "",
):
    """Plot DC-WME advantage over 4D-Var across the sweep.

    Shows the difference in error reduction between methods.

    Args:
        results: Dictionary of experiment results
        sweep_param: Name of the swept parameter
        output_path: Path to save the figure
        title_prefix: Prefix for the plot title
    """
    check_matplotlib()

    # Extract error reduction by value and method
    data: Dict[Any, Dict[str, Optional[float]]] = {}

    for result in results.values():
        value = result.get("sweep_value")
        method = result.get("method")

        if value not in data:
            data[value] = {"4dvar": None, "dcwme": None}

        if result.get("status") == "success":
            data[value][method] = result.get("result", {}).get("error_reduction", 0)
        else:
            data[value][method] = None

    if not data:
        print("No data to plot")
        return

    # Calculate advantage
    values = sorted(data.keys())
    advantage = []
    for v in values:
        dv4 = data[v].get("4dvar")
        dcwme = data[v].get("dcwme")
        if dv4 is not None and dcwme is not None:
            advantage.append(dcwme - dv4)
        else:
            advantage.append(None)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot bars
    x = np.arange(len(values))
    colors = ["green" if a is not None and a > 0 else "red" for a in advantage]
    heights = [a if a is not None else 0 for a in advantage]

    bars = ax.bar(x, heights, color=colors, edgecolor="black", linewidth=0.5)

    # Mark missing data
    for i, a in enumerate(advantage):
        if a is None:
            ax.annotate(
                "N/A",
                xy=(i, 0),
                ha="center",
                va="bottom",
                fontsize=10,
                color="gray",
            )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel(sweep_param)
    ax.set_ylabel("DC-WME Advantage (% points)")
    ax.set_title(f"{title_prefix}DC-WME vs 4D-Var Robustness: {sweep_param} Sweep")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in values])
    ax.grid(True, axis="y", alpha=0.3)

    # Add annotation
    ax.annotate(
        "DC-WME better",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
        color="green",
    )
    ax.annotate(
        "4D-Var better",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=10,
        color="red",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_wall_time_comparison(
    results: Dict[str, Any],
    sweep_param: str,
    output_path: Path,
    title_prefix: str = "",
):
    """Plot wall time comparison between methods.

    Args:
        results: Dictionary of experiment results
        sweep_param: Name of the swept parameter
        output_path: Path to save the figure
        title_prefix: Prefix for the plot title
    """
    check_matplotlib()

    # Extract wall time by value and method
    data: Dict[str, Dict[Any, float]] = {"4dvar": {}, "dcwme": {}}

    for result in results.values():
        value = result.get("sweep_value")
        method = result.get("method")
        wall_time = result.get("wall_time", 0)

        if method in data:
            data[method][value] = wall_time

    if not any(data.values()):
        print("No data to plot")
        return

    all_values = sorted(set(list(data["4dvar"].keys()) + list(data["dcwme"].keys())))
    x = np.arange(len(all_values))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"4dvar": "steelblue", "dcwme": "coral"}

    for i, method in enumerate(["4dvar", "dcwme"]):
        heights = [data[method].get(v, 0) for v in all_values]
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            heights,
            width,
            label=method.upper(),
            color=colors[method],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel(sweep_param)
    ax.set_ylabel("Wall Time (seconds)")
    ax.set_title(f"{title_prefix}Wall Time: {sweep_param} Sweep")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in all_values])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_plots(
    results_file: Path,
    output_dir: Path,
    sweep_param: str,
    title_prefix: str = "",
):
    """Generate all plots for a sweep.

    Args:
        results_file: Path to the sweep results JSON file
        output_dir: Directory to save figures
        sweep_param: Name of the swept parameter
        title_prefix: Prefix for plot titles
    """
    check_matplotlib()

    results = load_sweep_results(results_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots for {sweep_param} sweep...")

    # Convergence comparison
    plot_convergence_comparison(
        results,
        sweep_param,
        output_dir / f"convergence_{sweep_param}.png",
        title_prefix,
    )

    # Error reduction comparison
    plot_error_reduction_comparison(
        results,
        sweep_param,
        output_dir / f"error_reduction_{sweep_param}.png",
        title_prefix,
    )

    # Gradient analysis
    plot_gradient_analysis(
        results,
        sweep_param,
        output_dir / f"gradient_analysis_{sweep_param}.png",
        title_prefix,
    )

    # Method robustness
    plot_method_robustness(
        results,
        sweep_param,
        output_dir / f"robustness_{sweep_param}.png",
        title_prefix,
    )

    # Wall time comparison
    plot_wall_time_comparison(
        results,
        sweep_param,
        output_dir / f"wall_time_{sweep_param}.png",
        title_prefix,
    )

    print(f"All plots saved to: {output_dir}")

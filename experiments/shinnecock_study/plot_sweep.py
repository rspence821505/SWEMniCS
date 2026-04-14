"""Plot Phase 6 parameter sweep results: 4D-Var vs Static DC-WME.

Reads individual result JSONs from outputs/shinnecock_study/data/phase6_*.json
and produces a multi-panel comparison figure.
"""

import argparse
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Display labels for each dimension
DIM_LABELS = {
    "noise": "Observation Noise Level",
    "obs_density": "Observation Fraction",
    "obs_frequency": "Observation Frequency (timesteps)",
    "bg_error": "Background Error Std",
    "predictability_gamma": "Predictability γ (Eq 38)",
    "window_length": "DA Window Length (timesteps)",
    "model_error": "Friction Scale Factor",
}

BASELINE_VALUES = {
    "noise": 0.01,
    "obs_density": 0.1,
    "obs_frequency": 6,
    "bg_error": 0.02,
    "predictability_gamma": 0.1,
    "window_length": 12,
    "model_error": 1.15,
}

DIM_ORDER = ["noise", "obs_density", "obs_frequency", "bg_error",
             "predictability_gamma", "window_length", "model_error"]


def load_individual_results(data_dir):
    """Load all phase6 result JSONs and group by sweep dimension."""
    sweeps = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "phase6_*_results.json"))):
        name = os.path.basename(f).replace("phase6_", "").replace("_results.json", "")
        parts = name.rsplit("_", 1)
        method_key = parts[-1]  # a (4dvar) or b (dcwme)
        dim_val = parts[0]

        dim = None
        val = None
        for prefix in DIM_ORDER:
            if dim_val.startswith(prefix + "_"):
                dim = prefix
                val_str = dim_val[len(prefix) + 1:]
                val = float(val_str.replace("p", "."))
                break

        if dim is None:
            continue

        if dim not in sweeps:
            sweeps[dim] = {}
        if val not in sweeps[dim]:
            sweeps[dim][val] = {}

        with open(f) as fh:
            d = json.load(fh)

        r = d.get("results", {})
        sweeps[dim][val][method_key] = {
            "status": d.get("status", "unknown"),
            "error_reduction": r.get("error_reduction"),
            "background_error": r.get("background_error"),
            "analysis_error": r.get("analysis_error"),
            "num_iterations": r.get("num_iterations"),
            "converged": r.get("converged"),
        }

    return sweeps


def plot_sweep(data_dir, output_path=None):
    """Generate comparison plots for all sweep dimensions."""
    sweeps = load_individual_results(data_dir)

    dims = [d for d in DIM_ORDER if d in sweeps]
    n_dims = len(dims)

    if n_dims == 0:
        print("No results found.")
        return

    fig, axes = plt.subplots(n_dims, 1, figsize=(8, 3.5 * n_dims), squeeze=False)
    fig.suptitle("Phase 6: Parameter Sensitivity — 4D-Var vs DC-WME (static)",
                 fontsize=14, fontweight="bold", y=1.01)

    for i, dim_name in enumerate(dims):
        ax = axes[i, 0]
        label = DIM_LABELS[dim_name]
        baseline_val = BASELINE_VALUES.get(dim_name)
        dim_data = sweeps[dim_name]

        values = sorted(dim_data.keys())

        fdvar_x, fdvar_y, fdvar_fail_x = [], [], []
        dcwme_x, dcwme_y, dcwme_fail_x = [], [], []

        for val in values:
            entries = dim_data[val]
            # 4D-Var (method a)
            a = entries.get("a", {})
            if a.get("status") == "success" and a.get("error_reduction") is not None:
                fdvar_x.append(val)
                fdvar_y.append(a["error_reduction"])
            else:
                fdvar_fail_x.append(val)

            # DC-WME (method b)
            b = entries.get("b", {})
            if b.get("status") == "success" and b.get("error_reduction") is not None:
                dcwme_x.append(val)
                dcwme_y.append(b["error_reduction"])
            else:
                dcwme_fail_x.append(val)

        # Plot lines
        if fdvar_x:
            ax.plot(fdvar_x, fdvar_y, "o-", color="#2166ac", label="4D-Var",
                    linewidth=2, markersize=7, zorder=3)
        if fdvar_fail_x:
            ax.plot(fdvar_fail_x, [0] * len(fdvar_fail_x), "x",
                    color="#2166ac", markersize=10, markeredgewidth=2.5,
                    label="4D-Var (failed)" if not fdvar_x else None)

        if dcwme_x:
            ax.plot(dcwme_x, dcwme_y, "s--", color="#b2182b", label="DC-WME (static)",
                    linewidth=2, markersize=7, zorder=3)
        if dcwme_fail_x:
            ax.plot(dcwme_fail_x, [0] * len(dcwme_fail_x), "x",
                    color="#b2182b", markersize=10, markeredgewidth=2.5,
                    label="DC-WME (failed)" if not dcwme_x else None)

        # Reference lines
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        if baseline_val is not None:
            ax.axvline(x=baseline_val, color="gray", linestyle="--",
                       linewidth=1, alpha=0.5, label=f"Baseline ({baseline_val})")

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Error Reduction (%)", fontsize=11)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Log scale for noise and bg_error
        if dim_name in ("noise", "bg_error"):
            ax.set_xscale("log")

        # Y-axis: ensure 0 is visible, add some padding
        all_y = fdvar_y + dcwme_y
        if all_y:
            ymin = min(min(all_y), -1)
            ymax = max(all_y) * 1.15
            ax.set_ylim(ymin, ymax)

    plt.tight_layout()

    if output_path is None:
        output_path = Path(data_dir).parent / "phase6_sweep_comparison.png"

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

    # Also generate individual per-dimension plots for the paper
    for dim_name in dims:
        fig_single, ax = plt.subplots(figsize=(6, 4))
        label = DIM_LABELS[dim_name]
        baseline_val = BASELINE_VALUES.get(dim_name)
        dim_data = sweeps[dim_name]
        values = sorted(dim_data.keys())

        fdvar_x, fdvar_y, fdvar_fail_x = [], [], []
        dcwme_x, dcwme_y, dcwme_fail_x = [], [], []

        for val in values:
            entries = dim_data[val]
            a = entries.get("a", {})
            if a.get("status") == "success" and a.get("error_reduction") is not None:
                fdvar_x.append(val)
                fdvar_y.append(a["error_reduction"])
            else:
                fdvar_fail_x.append(val)

            b = entries.get("b", {})
            if b.get("status") == "success" and b.get("error_reduction") is not None:
                dcwme_x.append(val)
                dcwme_y.append(b["error_reduction"])
            else:
                dcwme_fail_x.append(val)

        if fdvar_x:
            ax.plot(fdvar_x, fdvar_y, "o-", color="#2166ac", label="4D-Var",
                    linewidth=2.5, markersize=8, zorder=3)
        if fdvar_fail_x:
            ax.plot(fdvar_fail_x, [0] * len(fdvar_fail_x), "x",
                    color="#2166ac", markersize=12, markeredgewidth=3)

        if dcwme_x:
            ax.plot(dcwme_x, dcwme_y, "s--", color="#b2182b", label="DC-WME (static)",
                    linewidth=2.5, markersize=8, zorder=3)
        if dcwme_fail_x:
            ax.plot(dcwme_fail_x, [0] * len(dcwme_fail_x), "x",
                    color="#b2182b", markersize=12, markeredgewidth=3)

        ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        if baseline_val is not None:
            ax.axvline(x=baseline_val, color="gray", linestyle="--",
                       linewidth=1, alpha=0.5, label=f"Baseline ({baseline_val})")

        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("Error Reduction (%)", fontsize=12)
        ax.set_title(f"Parameter Sensitivity: {label}", fontsize=13)
        ax.legend(loc="best", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        if dim_name in ("noise", "bg_error"):
            ax.set_xscale("log")

        all_y = fdvar_y + dcwme_y
        if all_y:
            ymin = min(min(all_y), -1)
            ymax = max(all_y) * 1.15
            ax.set_ylim(ymin, ymax)

        plt.tight_layout()
        single_path = Path(data_dir).parent / f"phase6_sweep_{dim_name}.png"
        fig_single.savefig(single_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {single_path}")
        plt.close(fig_single)


def main():
    parser = argparse.ArgumentParser(description="Plot Phase 6 sweep results")
    parser.add_argument("--data-dir",
                        default="outputs/shinnecock_study/data",
                        help="Directory containing phase6_*_results.json files")
    parser.add_argument("--output", default=None, help="Output plot path (combined)")
    args = parser.parse_args()

    plot_sweep(args.data_dir, args.output)


if __name__ == "__main__":
    main()

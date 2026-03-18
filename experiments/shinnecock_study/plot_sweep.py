"""Plot Phase 6 parameter sweep results: 4D-Var vs Static DC-WME."""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Display labels for each dimension
DIM_LABELS = {
    "noise": ("Observation Noise Level", "obs_noise_level"),
    "obs_density": ("Observation Fraction", "obs_fraction"),
    "obs_frequency": ("Observation Frequency (timesteps)", "obs_frequency"),
    "bg_error": ("Background Error Std", "background_error_std"),
    "cov_inflation": ("Covariance Inflation Factor", "cov_inflation_factor"),
    "window_length": ("DA Window Length (timesteps)", "nt_da"),
    "model_error": ("Friction Scale Factor", "friction_scale_factor"),
}

BASELINE_VALUES = {
    "noise": 0.01,
    "obs_density": 0.1,
    "obs_frequency": 6,
    "bg_error": 0.02,
    "cov_inflation": 10.0,
    "window_length": 12,
    "model_error": 1.15,
}


def extract_results(dim_results):
    """Extract values and error reductions from dimension results."""
    values = []
    fdvar_errs = []
    dcwme_errs = []
    dcwme_spreads = []

    for point in dim_results:
        val = point["value"]
        values.append(val)

        # 4D-Var
        fdvar = point.get("4dvar", {})
        if fdvar.get("status") == "failed":
            fdvar_errs.append(None)
        else:
            fdvar_errs.append(fdvar.get("results", {}).get("error_reduction"))

        # DC-WME
        dcwme = point.get("dcwme", {})
        if dcwme.get("status") == "failed":
            dcwme_errs.append(None)
            dcwme_spreads.append(None)
        else:
            dcwme_errs.append(dcwme.get("results", {}).get("error_reduction"))
            # Eigenvalue spread
            diag = dcwme.get("dcwme_diagnostics", {})
            eigs = diag.get("l_wme_eigenvalues", [])
            if eigs:
                spread = max(eigs) / max(min(eigs), 1e-30)
                dcwme_spreads.append(spread)
            else:
                dcwme_spreads.append(None)

    return values, fdvar_errs, dcwme_errs, dcwme_spreads


def plot_sweep(summary_path, output_path=None):
    """Generate comparison plots for all sweep dimensions."""
    with open(summary_path) as f:
        all_results = json.load(f)

    dims = [d for d in DIM_LABELS if d in all_results]
    n_dims = len(dims)

    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 4 * n_dims), squeeze=False)

    for i, dim_name in enumerate(dims):
        ax = axes[i, 0]
        label, param = DIM_LABELS[dim_name]
        baseline_val = BASELINE_VALUES.get(dim_name)

        values, fdvar_errs, dcwme_errs, dcwme_spreads = extract_results(
            all_results[dim_name]
        )

        x = np.array(values, dtype=float)

        # Plot 4D-Var
        fdvar_y = []
        fdvar_x = []
        fdvar_fail_x = []
        for xi, yi in zip(x, fdvar_errs):
            if yi is not None:
                fdvar_x.append(xi)
                fdvar_y.append(yi)
            else:
                fdvar_fail_x.append(xi)

        if fdvar_x:
            ax.plot(fdvar_x, fdvar_y, "o-", color="tab:blue", label="4D-Var",
                    linewidth=2, markersize=6)
        if fdvar_fail_x:
            ax.plot(fdvar_fail_x, [0] * len(fdvar_fail_x), "x",
                    color="tab:blue", markersize=10, markeredgewidth=2)

        # Plot DC-WME
        dcwme_y = []
        dcwme_x = []
        dcwme_fail_x = []
        for xi, yi in zip(x, dcwme_errs):
            if yi is not None:
                dcwme_x.append(xi)
                dcwme_y.append(yi)
            else:
                dcwme_fail_x.append(xi)

        if dcwme_x:
            ax.plot(dcwme_x, dcwme_y, "s--", color="tab:red", label="DC-WME (static)",
                    linewidth=2, markersize=6)
        if dcwme_fail_x:
            ax.plot(dcwme_fail_x, [0] * len(dcwme_fail_x), "x",
                    color="tab:red", markersize=10, markeredgewidth=2)

        # Reference lines
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        if baseline_val is not None:
            ax.axvline(x=baseline_val, color="gray", linestyle="--",
                       linewidth=1, alpha=0.5, label=f"Baseline ({baseline_val})")

        # Secondary y-axis for eigenvalue spread
        valid_spreads = [(xi, si) for xi, si in zip(x, dcwme_spreads)
                         if si is not None]
        if valid_spreads:
            ax2 = ax.twinx()
            sx, sy = zip(*valid_spreads)
            ax2.plot(sx, sy, "^:", color="tab:orange", alpha=0.6,
                     label="L_wme spread", markersize=5)
            ax2.set_ylabel("L_wme eigenvalue spread (max/min)", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax.set_xlabel(label)
        ax.set_ylabel("Error Reduction (%)")
        ax.set_title(f"Sweep: {label}")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Use log scale for noise and bg_error
        if dim_name in ("noise", "bg_error"):
            ax.set_xscale("log")

    plt.tight_layout()

    if output_path is None:
        output_path = Path(summary_path).parent.parent / "phase6_sweep_comparison.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Phase 6 sweep results")
    parser.add_argument("--input", required=True, help="Path to phase6_summary.json")
    parser.add_argument("--output", default=None, help="Output plot path")
    args = parser.parse_args()

    plot_sweep(args.input, args.output)


if __name__ == "__main__":
    main()

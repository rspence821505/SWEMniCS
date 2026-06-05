#!/usr/bin/env python3
r"""Presentation-quality log-scale per-window analysis-RMSE comparison.

Companion plot for Section 4 of ``docs/idealized_filter.tex``. The
figure visualizes the three-way head-to-head between QPCA-EnDCF,
stochastic 4D-EnKF, and the traditional sequential stochastic EnKF
from Table~1 of the chapter, on a logarithmic y-axis spanning the
~36 orders of magnitude that separate the bounded QPCA-EnDCF
trajectory from the floating-point overflow that the two stochastic
baselines reach.

Three shaded horizontal bands distinguish three operating regimes:

  - ``Physical regime`` (a few cm to ~10 m) — the dynamic range of a
    storm-surge field where an analysis can plausibly be interpreted.
  - ``Diverged but bounded`` (10 to 10^10 m) — finite-precision numbers
    that have left the physical regime but have not yet exhausted the
    floating-point representation.
  - ``Numerical overflow`` (above 10^10 m) — the regime in which the
    stored numbers no longer correspond to any physical quantity and
    the analysis has crossed into floating-point exhaustion.

The three filter trajectories are colored to convey their narrative
role: QPCA-EnDCF in a calm navy (the bounded, working filter); 4D-EnKF
in amber (severe failure); sequential EnKF in crimson (catastrophic
failure). Marker labels at the endpoints report the exact RMSE values
from the chapter's Table~1.

Outputs (resolved relative to the repository root):

    docs/filt_headtohead_logrmse.png
    docs/filt_headtohead_logrmse.pdf

The data are hard-coded against the chapter's
``tab:filt_headtohead`` and the underlying experiment outputs in
``results/three_way_ls6/`` (LS6 job 3203539 + 3205065). To regenerate
from those JSONs instead of the hard-coded table, pass
``--from-json``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = REPO_ROOT / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Configuration & data
# ---------------------------------------------------------------------------


DEFAULT_OUTPUT = Path("docs/filt_headtohead_logrmse.png")
DEFAULT_DPI = 400

# Table 1 values from docs/idealized_filter.tex
# (head-to-head analysis RMSE per cycling window at the chapter's
# operating-point configuration: cycling-4, κ=1, λ_infl=1.05,
# 5 km Gaspari-Cohn localization, N=10, seed=2026, vmax=15).
WINDOWS = [1, 2, 3, 4]
QPCA = [0.206, 0.396, 0.424, 1.454]
ENKF4D = [0.234, 2.22e5, 1.77e13, 6.73e32]
SEQ_ENKF = [0.293, 1.47e36, 1.47e36, 1.47e36]


# Color tokens — QPCA in a calm navy (the "stable, working" filter),
# 4D-EnKF in amber (severe failure), sequential EnKF in crimson
# (catastrophic failure). The three colors are far enough apart on
# the color wheel that they remain distinct at slide distance and
# under projector/printer color reproduction.
COLOR_QPCA = "#1F4E79"
COLOR_ENKF4D = "#F58518"
COLOR_SEQ_ENKF = "#D62728"

# Regime band colors — pastel tints that read as "background context"
# rather than data. Chosen so they sit clearly below the data lines.
COLOR_BAND_PHYSICAL = "#A8D5BA"   # pale sage green
COLOR_BAND_DIVERGED = "#F4E0A8"   # pale amber
COLOR_BAND_OVERFLOW = "#E8B4B8"   # pale rose


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def _apply_presentation_style() -> None:
    """Match the typography used in the companion observations and
    storm-tracks figures so the three plots present as a coherent set.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.weight": "bold",
            "mathtext.fontset": "dejavuserif",
            "mathtext.default": "bf",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": 17.5,
            "axes.titlepad": 14.0,
            "axes.labelsize": 14.0,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "black",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "legend.fontsize": 11.0,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
            "savefig.dpi": DEFAULT_DPI,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _draw_regime_bands(ax, x_lo: float, x_hi: float,
                       y_lo: float, y_hi: float) -> None:
    """Shade the three operating regimes as horizontal background
    bands. The boundaries are chosen so a reader can pin each filter's
    trajectory to a physical interpretation at a glance.
    """
    # Physical regime: cm to ~10 m water-surface elevation error.
    ax.axhspan(y_lo, 10.0, facecolor=COLOR_BAND_PHYSICAL,
               alpha=0.42, zorder=0, linewidth=0)
    # Diverged but bounded: between ten meters and ten gigameters
    # (numerical limit of the 64-bit mantissa is around 10^308, but
    # well past 10^10 m the values are operationally meaningless).
    ax.axhspan(10.0, 1e10, facecolor=COLOR_BAND_DIVERGED,
               alpha=0.42, zorder=0, linewidth=0)
    # Numerical overflow: above 10^10 m. Effectively floating-point
    # exhaustion for any analysis quantity tied to the SWE state.
    ax.axhspan(1e10, y_hi, facecolor=COLOR_BAND_OVERFLOW,
               alpha=0.42, zorder=0, linewidth=0)

    # Band labels sit at the right edge so they do not collide with
    # data lines or the legend on the upper left.
    label_x = x_hi - 0.02
    band_label_kwargs = dict(
        ha="right", va="center",
        fontsize=10.5, fontweight="bold",
        zorder=1,
    )
    ax.text(label_x, np.sqrt(y_lo * 10.0), "Physical regime",
            color="#0F4D2C", **band_label_kwargs)
    ax.text(label_x, np.sqrt(10.0 * 1e10), "Diverged but bounded",
            color="#7A5300", **band_label_kwargs)
    ax.text(label_x, np.sqrt(1e10 * y_hi), "Numerical overflow",
            color="#7B1F22", **band_label_kwargs)


def _draw_filter_line(ax, y_values, color: str, label: str,
                      marker_style: str = "o") -> None:
    """Plot a single filter trajectory with a thick line and prominent
    end-of-line markers. ``label`` is used only for the legend.
    """
    # Clip any zero / negative entries to the y-axis floor so the log
    # scale does not blow up. The chapter's RMSEs are all strictly
    # positive but this is defensive.
    y = np.array(y_values, dtype=float)
    y = np.maximum(y, 1e-2)
    ax.plot(
        WINDOWS, y,
        color=color, linewidth=3.0,
        marker=marker_style, markersize=12.0,
        markerfacecolor=color, markeredgecolor="white",
        markeredgewidth=1.6,
        label=label, zorder=4,
    )


def _annotate_endpoint(ax, x: float, y: float, text: str, color: str,
                       x_offset: float = 0.18, y_factor: float = 1.6,
                       ha: str = "left", va: str = "center") -> None:
    """Place a labeled callout near an endpoint of one of the lines.

    ``y_factor`` multiplies the y position (since we are on a log
    axis, the label is offset by a constant factor up or down).
    """
    ax.annotate(
        text,
        xy=(x, y), xytext=(x + x_offset, y * y_factor),
        ha=ha, va=va,
        fontsize=11.5, fontweight="bold",
        color=color, zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.30,rounding_size=0.18",
            facecolor="white", edgecolor=color,
            linewidth=0.8,
        ),
        arrowprops=dict(
            arrowstyle="-", color=color,
            connectionstyle="arc3,rad=0.0",
            linewidth=0.7, shrinkA=2, shrinkB=4,
        ),
    )


def _draw_legend(ax) -> None:
    """Compact legend identifying the three filters."""
    handles = [
        Line2D([0], [0], color=COLOR_QPCA, linewidth=3.0,
               marker="o", markerfacecolor=COLOR_QPCA,
               markeredgecolor="white", markersize=10.0,
               label="QPCA-EnDCF (spectrally truncated 4D)"),
        Line2D([0], [0], color=COLOR_ENKF4D, linewidth=3.0,
               marker="o", markerfacecolor=COLOR_ENKF4D,
               markeredgecolor="white", markersize=10.0,
               label="Stochastic 4D-EnKF"),
        Line2D([0], [0], color=COLOR_SEQ_ENKF, linewidth=3.0,
               marker="o", markerfacecolor=COLOR_SEQ_ENKF,
               markeredgecolor="white", markersize=10.0,
               label="Sequential stochastic EnKF"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=10.5,
        handlelength=2.6,
        handletextpad=0.7,
        borderaxespad=0.6,
    )
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
        txt.set_color("black")
    leg.set_zorder(7)


def make_plot(output_file: Path, dpi: int = DEFAULT_DPI,
              save_pdf: bool = True) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _apply_presentation_style()

    fig, ax = plt.subplots(figsize=(11.2, 6.6), dpi=dpi)

    # Y-axis range: from a tenth of a meter (below the smallest
    # physical-regime value of ~0.2 m) up to ten orders of magnitude
    # past the worst data point so the overflow band has visual
    # weight without compressing the data lines.
    y_lo = 1e-1
    y_hi = 1e40
    x_lo, x_hi = 0.55, 4.45

    _draw_regime_bands(ax, x_lo, x_hi, y_lo, y_hi)

    _draw_filter_line(ax, QPCA, COLOR_QPCA,
                      "QPCA-EnDCF (spectrally truncated 4D)")
    _draw_filter_line(ax, ENKF4D, COLOR_ENKF4D, "Stochastic 4D-EnKF")
    _draw_filter_line(ax, SEQ_ENKF, COLOR_SEQ_ENKF,
                      "Sequential stochastic EnKF")

    # Endpoint annotations — picked so each line is labeled with its
    # final-window value (or the value at which it saturates, in the
    # sequential case which pins at the same overflow scale from W2
    # onward).
    _annotate_endpoint(ax, 4, QPCA[-1],
                       f"{QPCA[-1]:.2f} m\n(bounded)",
                       COLOR_QPCA, x_offset=-1.10, y_factor=8.0,
                       ha="right", va="center")
    _annotate_endpoint(ax, 4, ENKF4D[-1],
                       r"$6.7 \times 10^{32}$ m",
                       COLOR_ENKF4D, x_offset=-1.20, y_factor=18.0,
                       ha="right", va="center")
    _annotate_endpoint(ax, 2, SEQ_ENKF[1],
                       r"$1.5 \times 10^{36}$ m" "\n(saturated)",
                       COLOR_SEQ_ENKF, x_offset=0.20, y_factor=4.0,
                       ha="left", va="center")

    # Axes chrome.
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(WINDOWS)
    ax.set_xticklabels([f"$W_{w}$" for w in WINDOWS])
    ax.set_xlabel("Cycling window")
    ax.set_ylabel("Analysis RMSE (m)")
    ax.set_title("QPCA-EnDCF vs Stochastic Ensemble Kalman Baselines",
                 loc="left", color="black")

    _draw_legend(ax)

    # Force-bold all tick labels and axis labels after the final draw
    # in case any auto-tick regeneration during the log-scale setup
    # silently reset weights.
    fig.canvas.draw()
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")

    fig.savefig(output_file, dpi=dpi)
    if save_pdf:
        fig.savefig(output_file.with_suffix(".pdf"), dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument("--no-pdf", action="store_true",
                   help="Skip the vector PDF export.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output = (args.output if args.output.is_absolute()
              else REPO_ROOT / args.output)
    make_plot(output, dpi=int(args.dpi), save_pdf=not args.no_pdf)
    print(f"Wrote: {output}")
    if not args.no_pdf:
        print(f"Wrote: {output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

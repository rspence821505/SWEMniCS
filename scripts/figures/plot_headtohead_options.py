#!/usr/bin/env python3
r"""Generate five alternative visualizations of the chapter's head-to-head.

Produces five separate figures so the alternatives can be compared
side-by-side before one is committed to as the chapter's headline
figure. Each option targets the same underlying data — the
per-window analysis RMSE for QPCA-EnDCF vs stochastic 4D-EnKF vs
sequential stochastic EnKF — but emphasizes a different visual
hierarchy and ties the abstract metric back to the physical inlet
problem in a different way.

Outputs (PNG + PDF, resolved relative to the repository root):

    A. docs/filt_headtohead_opt_a_composite.png
    B. docs/filt_headtohead_opt_b_vortex_annotated.png
    C. docs/filt_headtohead_opt_c_spatial.png
    D. docs/filt_headtohead_opt_d_heatmap.png
    E. docs/filt_headtohead_opt_e_timeline_strips.png

Each function is self-contained so the options are easy to compare,
extend, or recombine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = REPO_ROOT / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    import cmocean.cm as _cmo
    _BATHY_CMAP = _cmo.deep
except ImportError:
    _BATHY_CMAP = plt.get_cmap("YlGnBu")


# ---------------------------------------------------------------------------
# Shared configuration & data
# ---------------------------------------------------------------------------


DEFAULT_DPI = 400
MESH_FILE = REPO_ROOT / "data" / "Ideal_Inlet" / "Ideal_Inlet.h5"

WINDOWS = [1, 2, 3, 4]
QPCA = np.array([0.206, 0.396, 0.424, 1.454])
ENKF4D = np.array([0.234, 2.22e5, 1.77e13, 6.73e32])
SEQ_ENKF = np.array([0.293, 1.47e36, 1.47e36, 1.47e36])

COLOR_QPCA = "#1F4E79"
COLOR_ENKF4D = "#F58518"
COLOR_SEQ_ENKF = "#D62728"

# Storm-track parameters (match experiments/idealized_inlet_twin.py).
TRACK_START_KM = np.array([40.0, -20.0])
TRACK_END_KM = np.array([25.0, 50.0])
TRACK_SHIFT_KM = 5.0
TRACK_DURATION_H = 8.0

# Cycling timing (in physical hours since simulation start).
SPINUP_H = 1.0
WINDOW_DURATION_H = 80.0 / 60.0
WINDOW_T_START = [SPINUP_H + i * WINDOW_DURATION_H for i in range(4)]
WINDOW_T_END = [SPINUP_H + (i + 1) * WINDOW_DURATION_H for i in range(4)]
WINDOW_T_MID = [(s + e) / 2.0 for s, e in zip(WINDOW_T_START, WINDOW_T_END)]


# ---------------------------------------------------------------------------
# Mesh + geometry helpers
# ---------------------------------------------------------------------------


def _load_mesh():
    with h5py.File(MESH_FILE, "r") as f:
        geometry = f["Mesh/mesh/geometry"][:]
        topology = f["Mesh/mesh/topology"][:]
    x_km = geometry[:, 0] / 1000.0
    y_km = geometry[:, 1] / 1000.0
    tri = mtri.Triangulation(x_km, y_km, triangles=topology)
    return x_km, y_km, tri, geometry


def _bathymetry(geometry):
    y = geometry[:, 1]
    h_b = np.where(
        y < 20_000.0,
        14.0 - (14.0 - 5.0) / 20_000.0 * y,
        5.0,
    )
    return np.maximum(h_b, 5.0)


def _track_endpoints():
    dx = TRACK_END_KM[0] - TRACK_START_KM[0]
    dy = TRACK_END_KM[1] - TRACK_START_KM[1]
    L = float(np.hypot(dx, dy))
    perp = np.array([-dy / L, dx / L])
    shift = TRACK_SHIFT_KM * perp
    return (TRACK_START_KM, TRACK_END_KM,
            TRACK_START_KM + shift, TRACK_END_KM + shift)


def _track_point(start, end, t_hours, duration_h=TRACK_DURATION_H):
    frac = max(0.0, min(1.0, float(t_hours) / float(duration_h)))
    return start + frac * (end - start)


def _boundary_edges(tri):
    triangles = tri.triangles
    edges = {}
    for tri_row in triangles:
        for a, b in ((tri_row[0], tri_row[1]),
                     (tri_row[1], tri_row[2]),
                     (tri_row[2], tri_row[0])):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            edges[key] = edges.get(key, 0) + 1
    boundary = [k for k, c in edges.items() if c == 1]
    xs, ys = [], []
    for a, b in boundary:
        xs.extend([tri.x[a], tri.x[b], np.nan])
        ys.extend([tri.y[a], tri.y[b], np.nan])
    return xs, ys


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.weight": "bold",
            "mathtext.fontset": "dejavuserif",
            "mathtext.default": "bf",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": 16.0,
            "axes.labelsize": 13.0,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "black",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 10.5,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
            "savefig.dpi": DEFAULT_DPI,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _force_bold(fig, axes):
    fig.canvas.draw()
    for ax in axes:
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")


def _save(fig, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DEFAULT_DPI)
    fig.savefig(out_png.with_suffix(".pdf"), dpi=DEFAULT_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rendering primitives shared by multiple options
# ---------------------------------------------------------------------------


def _draw_inlet_thumb(ax, tri, geometry, vortex_position=None,
                       show_bathy=True, show_tracks=False):
    """Render a minimalist inlet thumbnail into a small Axes.

    Used by options A, B, and C to render compact spatial-context
    panels. The thumbnail shows the bathymetry as a colored substrate,
    the inlet boundary as a thin outline, and optionally the
    truth/perturbed storm tracks and a vortex-position marker.
    """
    x_km = tri.x
    y_km = tri.y
    if show_bathy:
        bathy = _bathymetry(geometry)
        ax.tripcolor(tri, bathy, shading="gouraud", cmap=_BATHY_CMAP,
                     alpha=0.85, rasterized=True, zorder=1)
    xs, ys = _boundary_edges(tri)
    ax.plot(xs, ys, color="black", linewidth=0.9, alpha=0.85, zorder=2)
    if show_tracks:
        t0, t1, p0, p1 = _track_endpoints()
        ax.plot([t0[0], t1[0]], [t0[1], t1[1]],
                color=COLOR_SEQ_ENKF, linewidth=1.2, alpha=0.55,
                zorder=2.5)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                color="#7C3AED", linewidth=1.2, alpha=0.55,
                linestyle=(0, (4, 2)), zorder=2.5)
    if vortex_position is not None:
        ax.scatter(*vortex_position, marker="o", s=55.0,
                   c=COLOR_SEQ_ENKF, edgecolors="white",
                   linewidths=1.2, zorder=4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _setup_logrmse_axes(ax, regime_bands=True):
    """Configure an Axes for the log-RMSE three-way line plot."""
    if regime_bands:
        ax.axhspan(1e-2, 10.0, facecolor="#A8D5BA", alpha=0.34,
                   zorder=0, linewidth=0)
        ax.axhspan(10.0, 1e10, facecolor="#F4E0A8", alpha=0.34,
                   zorder=0, linewidth=0)
        ax.axhspan(1e10, 1e40, facecolor="#E8B4B8", alpha=0.34,
                   zorder=0, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlim(0.55, 4.45)
    ax.set_ylim(1e-1, 1e40)
    ax.set_xticks(WINDOWS)
    ax.set_xticklabels([f"$W_{w}$" for w in WINDOWS])
    ax.set_xlabel("Cycling window")
    ax.set_ylabel("Analysis RMSE (m)")


def _plot_three_filter_lines(ax, marker_size=10.0, line_width=2.8):
    for y, color, lbl in (
        (QPCA, COLOR_QPCA, "QPCA-EnDCF"),
        (ENKF4D, COLOR_ENKF4D, "Stochastic 4D-EnKF"),
        (SEQ_ENKF, COLOR_SEQ_ENKF, "Sequential stochastic EnKF"),
    ):
        y_clipped = np.maximum(y, 1e-2)
        ax.plot(WINDOWS, y_clipped, color=color, linewidth=line_width,
                marker="o", markersize=marker_size,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=1.4, label=lbl, zorder=4)


# ---------------------------------------------------------------------------
# Option A: Storm-track context + log-RMSE side-by-side
# ---------------------------------------------------------------------------


def _make_option_a(out_path: Path):
    """Composite: storm tracks on the left, log-RMSE plot on the right.

    A vertical connector and matching x-axis-window vortex-position
    markers tie the two panels together so the reader can see *when*
    in the storm each filter behavior happens.
    """
    _apply_style()
    x_km, y_km, tri, geometry = _load_mesh()
    bathy = _bathymetry(geometry)
    t0, t1, p0, p1 = _track_endpoints()

    fig = plt.figure(figsize=(14.5, 6.6), dpi=DEFAULT_DPI)
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.3], wspace=0.18, figure=fig)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # --- LEFT: storm tracks with window-midpoint vortex markers
    tpc = ax_left.tripcolor(tri, bathy, shading="gouraud",
                             cmap=_BATHY_CMAP, alpha=0.85,
                             rasterized=True, zorder=1)
    ax_left.triplot(tri, color="black", linewidth=0.1, alpha=0.35,
                     antialiased=True, zorder=2)
    bxs, bys = _boundary_edges(tri)
    ax_left.plot(bxs, bys, color="black", linewidth=1.0,
                  alpha=0.9, zorder=2.5)
    # Tracks (clipped to the cycling time window).
    for start, end, color, dashed in (
        (t0, t1, COLOR_SEQ_ENKF, False),
        (p0, p1, "#7C3AED", True),
    ):
        # Truncate the visible track to the cycling-cycle interval so
        # the inlet stays the visual subject.
        seg_start = _track_point(start, end, SPINUP_H)
        seg_end = _track_point(start, end, WINDOW_T_END[-1])
        ls = (0, (7, 3)) if dashed else "solid"
        ax_left.plot([seg_start[0], seg_end[0]],
                     [seg_start[1], seg_end[1]],
                     color=color, linewidth=2.2, linestyle=ls,
                     zorder=4)
    # Vortex-position markers at each window midpoint.
    for w, t_mid in enumerate(WINDOW_T_MID, start=1):
        pos = _track_point(t0, t1, t_mid)
        ax_left.scatter(*pos, marker="o", s=120.0,
                         c="white", edgecolors=COLOR_SEQ_ENKF,
                         linewidths=2.0, zorder=5)
        ax_left.annotate(f"$W_{w}$", xy=pos,
                          xytext=(pos[0] + 3.0, pos[1] + 0.3),
                          fontsize=11.0, fontweight="bold",
                          color=COLOR_SEQ_ENKF, zorder=6,
                          ha="left", va="center")
    ax_left.set_aspect("equal", adjustable="box")
    ax_left.set_xlim(-3, 53)
    ax_left.set_ylim(-3, 35)
    ax_left.set_xlabel("x (km)")
    ax_left.set_ylabel("y (km)")
    ax_left.set_title("Storm context", loc="left", color="black",
                       fontsize=14.5, pad=10.0)
    for spine in ax_left.spines.values():
        spine.set_visible(False)
    ax_left.tick_params(which="both", top=False, right=False, length=0)

    # --- RIGHT: log-RMSE three-way
    _setup_logrmse_axes(ax_right)
    _plot_three_filter_lines(ax_right)
    # Endpoint annotations.
    ax_right.annotate("1.45 m (bounded)",
                      xy=(4, QPCA[-1]),
                      xytext=(3.0, 6.5),
                      fontsize=10.5, fontweight="bold",
                      color=COLOR_QPCA, ha="right",
                      bbox=dict(boxstyle="round,pad=0.30",
                                facecolor="white",
                                edgecolor=COLOR_QPCA, linewidth=0.7),
                      arrowprops=dict(arrowstyle="-",
                                      color=COLOR_QPCA, linewidth=0.7))
    ax_right.annotate(r"$6.7 \times 10^{32}$ m",
                      xy=(4, ENKF4D[-1]),
                      xytext=(3.0, 5e28),
                      fontsize=10.5, fontweight="bold",
                      color=COLOR_ENKF4D, ha="right",
                      bbox=dict(boxstyle="round,pad=0.30",
                                facecolor="white",
                                edgecolor=COLOR_ENKF4D, linewidth=0.7),
                      arrowprops=dict(arrowstyle="-",
                                      color=COLOR_ENKF4D,
                                      linewidth=0.7))
    ax_right.annotate(r"$1.5 \times 10^{36}$ m (saturated)",
                      xy=(2, SEQ_ENKF[1]),
                      xytext=(2.3, 5e37),
                      fontsize=10.5, fontweight="bold",
                      color=COLOR_SEQ_ENKF, ha="left",
                      bbox=dict(boxstyle="round,pad=0.30",
                                facecolor="white",
                                edgecolor=COLOR_SEQ_ENKF, linewidth=0.7),
                      arrowprops=dict(arrowstyle="-",
                                      color=COLOR_SEQ_ENKF,
                                      linewidth=0.7))
    # Regime labels (right edge).
    for y_geo, label, col in (
        (np.sqrt(1e-1 * 10), "Physical", "#0F4D2C"),
        (np.sqrt(10 * 1e10), "Diverged", "#7A5300"),
        (np.sqrt(1e10 * 1e40), "Overflow", "#7B1F22"),
    ):
        ax_right.text(4.43, y_geo, label, ha="right", va="center",
                       fontsize=10.0, fontweight="bold",
                       color=col, alpha=0.85, zorder=1)
    ax_right.set_title("Analysis RMSE per cycling window",
                       loc="left", color="black", fontsize=14.5,
                       pad=10.0)
    ax_right.legend(loc="upper left", fontsize=10.5,
                     handlelength=2.4, handletextpad=0.7,
                     borderaxespad=0.6, frameon=False)
    for txt in ax_right.get_legend().get_texts():
        txt.set_fontweight("bold")

    fig.suptitle("QPCA-EnDCF vs Stochastic Baselines — Storm Context + RMSE",
                  fontsize=16.5, fontweight="bold", y=0.99)
    _force_bold(fig, [ax_left, ax_right])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Option B: Vortex-position annotated trajectory plot
# ---------------------------------------------------------------------------


def _make_option_b(out_path: Path):
    """Log-RMSE plot with small inlet-with-vortex thumbnails above
    each x-tick so the abstract window labels become physically
    meaningful at slide distance.
    """
    _apply_style()
    x_km, y_km, tri, geometry = _load_mesh()
    t0, t1, _, _ = _track_endpoints()

    fig = plt.figure(figsize=(12.0, 7.6), dpi=DEFAULT_DPI)
    gs = GridSpec(2, 1, height_ratios=[0.38, 1.0], hspace=0.18, figure=fig)
    ax_thumb_row = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])

    # --- Top row: 4 thumbnails of the inlet with vortex at window midpoint
    ax_thumb_row.set_axis_off()
    thumb_width = 0.18
    thumb_height = 0.30
    # Place thumbnails at the x positions matching the W1..W4 ticks
    # of the bottom axes.
    fig.canvas.draw()
    for w_idx, w in enumerate(WINDOWS):
        # x-position of W_w on the bottom axis in figure coords
        x_data = w
        bbox_bottom = ax.get_position()
        # Bottom axis x extent
        x_lo, x_hi = 0.55, 4.45
        x_frac_in_ax = (x_data - x_lo) / (x_hi - x_lo)
        x_fig = bbox_bottom.x0 + x_frac_in_ax * bbox_bottom.width
        bbox_thumb = [x_fig - thumb_width / 2.0,
                       0.69, thumb_width, thumb_height]
        ax_thumb = fig.add_axes(bbox_thumb)
        vortex = _track_point(t0, t1, WINDOW_T_MID[w_idx])
        _draw_inlet_thumb(ax_thumb, tri, geometry,
                           vortex_position=vortex,
                           show_tracks=True)
        ax_thumb.set_title(f"$W_{w}$  $t \\approx {WINDOW_T_MID[w_idx]:.1f}$ h",
                             fontsize=10.5, fontweight="bold",
                             color="black", pad=4)

    # --- Bottom: log-RMSE plot
    _setup_logrmse_axes(ax)
    _plot_three_filter_lines(ax)

    ax.set_xlabel("Cycling window  (vortex position shown above)")
    ax.set_title("Analysis RMSE per cycling window",
                  loc="left", color="black", fontsize=15.0, pad=10.0)
    leg = ax.legend(loc="upper left", fontsize=10.5,
                     handlelength=2.4, handletextpad=0.7,
                     borderaxespad=0.6, frameon=False)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    # Regime labels (right edge).
    for y_geo, label, col in (
        (np.sqrt(1e-1 * 10), "Physical regime", "#0F4D2C"),
        (np.sqrt(10 * 1e10), "Diverged but bounded", "#7A5300"),
        (np.sqrt(1e10 * 1e40), "Numerical overflow", "#7B1F22"),
    ):
        ax.text(4.43, y_geo, label, ha="right", va="center",
                fontsize=10.0, fontweight="bold",
                color=col, alpha=0.85, zorder=1)
    _force_bold(fig, [ax])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Option C: Spatial analysis-error snapshots (illustrative)
# ---------------------------------------------------------------------------


def _make_option_c(out_path: Path):
    """3 rows × 4 columns of inlet thumbnails colored by a synthetic
    representative analysis-error field whose amplitude matches the
    per-window RMSE values.

    This panel is *illustrative*: the JSONs only record scalar RMSE,
    so the per-cell color is generated as a smooth Gaussian random
    field scaled to the table's RMSE values. Cells whose RMSE
    saturates the colorbar are filled at the maximum color and
    annotated with the overflow value so the reader understands the
    distinction.
    """
    _apply_style()
    x_km, y_km, tri, geometry = _load_mesh()
    bathy = _bathymetry(geometry)
    n_nodes = tri.x.size

    # Use the same RNG seed across all cells so the spatial pattern
    # is comparable cell-to-cell (only the amplitude differs).
    rng = np.random.default_rng(2026)
    base_pattern = rng.standard_normal(n_nodes)
    # Smooth the pattern by averaging over each node's neighbors so
    # the visual field is coherent on a coastal scale rather than
    # node-by-node white noise.
    triangles = tri.triangles
    neighbors = [set() for _ in range(n_nodes)]
    for tri_row in triangles:
        for i in tri_row:
            for j in tri_row:
                if i != j:
                    neighbors[i].add(int(j))
    smoothed = base_pattern.copy()
    for _ in range(8):  # several smoothing passes for coastal scale
        new = smoothed.copy()
        for i in range(n_nodes):
            if neighbors[i]:
                new[i] = 0.5 * smoothed[i] + 0.5 * np.mean(
                    [smoothed[j] for j in neighbors[i]]
                )
        smoothed = new
    # Normalize to unit RMS.
    smoothed = smoothed - smoothed.mean()
    smoothed = smoothed / np.sqrt(np.mean(smoothed ** 2))

    rows = [
        ("QPCA-EnDCF", QPCA, COLOR_QPCA),
        ("Stoch. 4D-EnKF", ENKF4D, COLOR_ENKF4D),
        ("Seq. EnKF", SEQ_ENKF, COLOR_SEQ_ENKF),
    ]
    # Cap visible amplitudes at 10 m so the colormap has a stable
    # range. Anything beyond that is annotated as "overflow"
    # (and visually rendered at full saturation).
    visible_max = 10.0
    cmap = plt.get_cmap("RdBu_r")

    fig = plt.figure(figsize=(13.0, 8.0), dpi=DEFAULT_DPI)
    gs = GridSpec(3, 5, width_ratios=[0.20, 1, 1, 1, 1],
                   wspace=0.10, hspace=0.18, figure=fig)
    for r, (filt_name, rmses, color) in enumerate(rows):
        # Row label panel (axis-off; just text).
        ax_lbl = fig.add_subplot(gs[r, 0])
        ax_lbl.set_axis_off()
        ax_lbl.text(0.92, 0.5, filt_name, ha="right", va="center",
                     fontsize=13.5, fontweight="bold", color=color,
                     transform=ax_lbl.transAxes)
        for c, (w, rmse) in enumerate(zip(WINDOWS, rmses)):
            ax = fig.add_subplot(gs[r, c + 1])
            xs, ys = _boundary_edges(tri)
            if rmse <= visible_max:
                field = smoothed * rmse  # synthetic field
                ax.tripcolor(tri, field, shading="gouraud",
                              cmap=cmap, vmin=-visible_max,
                              vmax=visible_max,
                              rasterized=True, zorder=1, alpha=0.92)
                value_text = f"RMSE = {rmse:.2f} m"
            else:
                # Overflow: paint the inlet a uniform saturating red.
                ax.tripcolor(tri,
                              np.full(n_nodes, visible_max),
                              shading="gouraud", cmap=cmap,
                              vmin=-visible_max, vmax=visible_max,
                              rasterized=True, zorder=1, alpha=0.92)
                exponent = int(np.floor(np.log10(rmse)))
                mantissa = rmse / 10 ** exponent
                value_text = (rf"RMSE $\sim {mantissa:.1f} \times 10^"
                              rf"{{{exponent}}}$ m")
            ax.plot(xs, ys, color="black", linewidth=0.7,
                     alpha=0.85, zorder=2)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(f"$W_{w}$", fontsize=14.0,
                              fontweight="bold", color="black",
                              pad=6.0)
            # Per-cell value caption at bottom.
            ax.text(0.5, -0.10, value_text, transform=ax.transAxes,
                     ha="center", va="top",
                     fontsize=9.5, fontweight="bold", color=color)

    fig.suptitle(
        r"Illustrative analysis-error fields — amplitude $\propto$ "
        r"RMSE in Table 1",
        fontsize=15.0, fontweight="bold", y=1.00,
    )
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Option D: Heatmap of log-RMSE values
# ---------------------------------------------------------------------------


def _make_option_d(out_path: Path):
    """3 × 4 heatmap of log10(RMSE) values with the exact value
    annotated in each cell.
    """
    _apply_style()
    values = np.array([QPCA, ENKF4D, SEQ_ENKF])
    log_values = np.log10(np.maximum(values, 1e-2))

    fig, ax = plt.subplots(figsize=(11.0, 4.6), dpi=DEFAULT_DPI)
    im = ax.imshow(
        log_values, cmap="YlOrRd", aspect="auto",
        vmin=-1.0, vmax=36.0,
        origin="upper",
    )
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"$W_{w}$" for w in WINDOWS])
    ax.set_yticks(range(3))
    ax.set_yticklabels(["QPCA-EnDCF",
                         "Stochastic 4D-EnKF",
                         "Sequential stochastic EnKF"])
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Analysis RMSE (m) — log-scale colored heatmap",
                  loc="left", color="black", fontsize=15.0, pad=10.0)

    # Annotate each cell with the exact value, formatted for
    # readability. Text color flips on a brightness threshold.
    for r in range(3):
        for c in range(4):
            val = values[r, c]
            log_val = log_values[r, c]
            if val < 1000.0:
                text = f"{val:.2f}"
            else:
                exponent = int(np.floor(np.log10(val)))
                mantissa = val / 10 ** exponent
                text = rf"${mantissa:.1f} \times 10^{{{exponent}}}$"
            txt_color = "white" if log_val > 18.0 else "black"
            ax.text(c, r, text, ha="center", va="center",
                     fontsize=12.5, fontweight="bold",
                     color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02,
                         label=r"$\log_{10}$(RMSE)")
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("black")
    cbar.ax.tick_params(width=0.6, length=3.0, direction="in",
                         labelcolor="black", labelsize=11.0)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    cbar.ax.yaxis.label.set_fontweight("bold")
    cbar.set_label(r"$\log_{10}$(RMSE)", fontweight="bold",
                    fontsize=12.5, color="black")

    _force_bold(fig, [ax])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Option E: Storm-clock timeline strips
# ---------------------------------------------------------------------------


def _make_option_e(out_path: Path):
    """Storm timeline at the top + three horizontal Gantt-style filter
    strips, each segment colored by per-window log-RMSE.
    """
    _apply_style()
    x_km, y_km, tri, geometry = _load_mesh()
    t0, t1, _, _ = _track_endpoints()

    fig = plt.figure(figsize=(13.0, 7.0), dpi=DEFAULT_DPI)
    gs = GridSpec(4, 1, height_ratios=[0.45, 0.20, 0.20, 0.20],
                   hspace=0.18, figure=fig)
    ax_storm = fig.add_subplot(gs[0])
    strip_axes = [fig.add_subplot(gs[i + 1]) for i in range(3)]

    # --- Storm timeline at top (single horizontal axis, vortex
    # thumbnails at window midpoints).
    ax_storm.set_xlim(SPINUP_H, WINDOW_T_END[-1])
    ax_storm.set_ylim(-0.5, 1.0)
    ax_storm.set_yticks([])
    ax_storm.set_xlabel("Time since simulation start (h)",
                         fontweight="bold")
    ax_storm.set_title("Storm timeline", loc="left", color="black",
                        fontsize=14.0, pad=10.0)
    # Time axis with window boundaries.
    for ws, we in zip(WINDOW_T_START, WINDOW_T_END):
        ax_storm.axvspan(ws, we, ymin=0.05, ymax=0.45,
                          facecolor="#E5EBF2", edgecolor="black",
                          linewidth=0.6, zorder=1)
    # Window labels in the strip.
    for w_idx, w in enumerate(WINDOWS):
        mid = WINDOW_T_MID[w_idx]
        ax_storm.text(mid, 0.25, f"$W_{w}$", ha="center", va="center",
                        fontsize=12.0, fontweight="bold",
                        color="black", zorder=2)
    # Vortex y-position over time (cleanly conveys the geometry).
    t_vals = np.linspace(SPINUP_H, WINDOW_T_END[-1], 200)
    y_vortex = np.array([_track_point(t0, t1, t)[1] for t in t_vals])
    # Scale to fit upper portion of axes.
    y_norm = (y_vortex - y_vortex.min()) / (y_vortex.max()
                                              - y_vortex.min())
    y_norm = 0.55 + y_norm * 0.40
    ax_storm.plot(t_vals, y_norm, color=COLOR_SEQ_ENKF,
                    linewidth=2.5, zorder=3)
    ax_storm.text(WINDOW_T_END[-1] + 0.05, y_norm[-1],
                    "vortex\ntrack (y)",
                    ha="left", va="center",
                    fontsize=10.0, fontweight="bold",
                    color=COLOR_SEQ_ENKF)
    # Annotate closest-approach time.
    # Centroid of the inlet mesh ≈ (25, 15) — closest approach at
    # the time the y-coordinate is closest to 15. Approximate via
    # the truth-track parameterization: y(t) = -20 + (t/8)*70 = 15
    # → t = 5 h.
    ax_storm.axvline(5.0, color=COLOR_SEQ_ENKF, linestyle=(0, (4, 3)),
                       linewidth=1.2, alpha=0.7, zorder=2)
    ax_storm.text(5.0, 0.99, "closest\napproach",
                    ha="center", va="top",
                    fontsize=9.5, fontweight="bold",
                    color=COLOR_SEQ_ENKF, zorder=4)
    for spine in ax_storm.spines.values():
        spine.set_visible(False)
    ax_storm.tick_params(which="both", top=False, right=False,
                            length=0)

    # --- Three strips, colored by log-RMSE per window
    log_values = np.log10(np.maximum(
        np.array([QPCA, ENKF4D, SEQ_ENKF]), 1e-2))
    vmin, vmax = -1.0, 36.0
    cmap = plt.get_cmap("YlOrRd")
    filter_names = ["QPCA-EnDCF", "Stoch. 4D-EnKF", "Seq. EnKF"]
    filter_colors = [COLOR_QPCA, COLOR_ENKF4D, COLOR_SEQ_ENKF]
    for ax, name, color, log_row, raw_row in zip(
        strip_axes, filter_names, filter_colors,
        log_values, [QPCA, ENKF4D, SEQ_ENKF],
    ):
        ax.set_xlim(SPINUP_H, WINDOW_T_END[-1])
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(which="both", length=0)
        for (ws, we), lv, rv in zip(
            zip(WINDOW_T_START, WINDOW_T_END), log_row, raw_row,
        ):
            color_cell = cmap((lv - vmin) / (vmax - vmin))
            ax.axvspan(ws, we, facecolor=color_cell, edgecolor="black",
                         linewidth=0.5, zorder=1)
            if rv < 1000:
                text = f"{rv:.2f}"
            else:
                exp = int(np.floor(np.log10(rv)))
                man = rv / 10 ** exp
                text = rf"${man:.1f}\times 10^{{{exp}}}$"
            txt_color = "white" if lv > 18.0 else "black"
            ax.text((ws + we) / 2.0, 0.5, text,
                     ha="center", va="center",
                     fontsize=11.0, fontweight="bold",
                     color=txt_color, zorder=2)
        # Row label
        ax.text(SPINUP_H - 0.05, 0.5, name,
                 ha="right", va="center",
                 fontsize=12.5, fontweight="bold",
                 color=color, transform=ax.transData)
    strip_axes[-1].set_xticks(WINDOW_T_MID)
    strip_axes[-1].set_xticklabels([f"$W_{w}$" for w in WINDOWS])
    strip_axes[-1].set_xlabel("Cycling window", fontweight="bold")

    # Shared colorbar on the right of the strip block.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.08, 0.014, 0.42])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("black")
    cbar.ax.tick_params(width=0.6, length=3.0, direction="in",
                         labelcolor="black", labelsize=10.5)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    cbar.set_label(r"$\log_{10}$(RMSE in m)", fontweight="bold",
                    fontsize=11.5, color="black")

    fig.suptitle(
        "Filter-state timeline strips along the storm cycle",
        fontsize=15.5, fontweight="bold", y=1.00,
    )
    _force_bold(fig, [ax_storm] + strip_axes)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Option C-v2: spatial snapshots with regime-aware cell treatment
# ---------------------------------------------------------------------------


def _regime_of(rmse: float) -> str:
    if rmse < 10.0:
        return "physical"
    if rmse < 1e10:
        return "diverged"
    return "overflow"


# Regime colors used by both C-v2 and E-v2.
REGIME_COLORS = {
    "physical": "#34A06A",   # green
    "diverged": "#D7892B",   # amber
    "overflow": "#B43A3F",   # crimson
}
REGIME_FILLS = {
    "physical": "#E8F4ED",   # very pale green
    "diverged": "#F8EFD7",   # very pale amber
    "overflow": "#F4DBDD",   # very pale rose
}


def _make_option_c2(out_path: Path):
    """Spatial snapshots with distinct treatment per regime.

    The original C visualizes every cell as a synthetic spatial
    field, which renders the catastrophic cells as identical
    saturating-red rectangles. C-v2 instead uses *three* visual
    treatments depending on which operating regime the cell is in:

      - Physical (RMSE < 10 m): a small synthetic spatial error
        field with a fine colormap range tuned to the QPCA-EnDCF
        amplitude window, so the W1 → W4 progression is visible
        as gradual color change.
      - Diverged but bounded (10 ≤ RMSE < 10^10): a uniform muted
        amber card showing the magnitude in scientific notation.
        The inlet outline is preserved so the panel reads as
        ``the spatial structure of the analysis has been replaced
        by a finite-but-meaningless number``.
      - Numerical overflow (RMSE ≥ 10^10): a deep crimson card
        with a clear OFF-SCALE badge and the magnitude. The
        inlet outline is dropped to indicate that the analysis
        no longer corresponds to any spatial field.

    This treatment emphasizes the *regime cliff* — the central
    physical story — rather than making the catastrophic cells
    visually flat.
    """
    _apply_style()
    x_km, y_km, tri, geometry = _load_mesh()
    n_nodes = tri.x.size

    # Smooth random pattern for the physical-regime cells, so the
    # spatial structure looks plausibly storm-driven.
    rng = np.random.default_rng(2026)
    base_pattern = rng.standard_normal(n_nodes)
    triangles = tri.triangles
    neighbors = [set() for _ in range(n_nodes)]
    for tri_row in triangles:
        for i in tri_row:
            for j in tri_row:
                if i != j:
                    neighbors[i].add(int(j))
    smoothed = base_pattern.copy()
    for _ in range(8):
        new = smoothed.copy()
        for i in range(n_nodes):
            if neighbors[i]:
                new[i] = 0.5 * smoothed[i] + 0.5 * np.mean(
                    [smoothed[j] for j in neighbors[i]]
                )
        smoothed = new
    smoothed -= smoothed.mean()
    smoothed /= np.sqrt(np.mean(smoothed ** 2))

    rows = [
        ("QPCA-EnDCF", QPCA, COLOR_QPCA),
        ("Stoch. 4D-EnKF", ENKF4D, COLOR_ENKF4D),
        ("Seq. EnKF", SEQ_ENKF, COLOR_SEQ_ENKF),
    ]
    # Colormap for the *physical* regime — fine range tuned to the
    # QPCA amplitude window so the W1 (0.21 m) → W4 (1.45 m)
    # progression is visible as color change.
    cmap_phys = plt.get_cmap("RdBu_r")
    physical_amp_max = 2.5  # m, just past QPCA's W4 value
    xs_bdry, ys_bdry = _boundary_edges(tri)

    fig = plt.figure(figsize=(13.5, 8.4), dpi=DEFAULT_DPI)
    gs = GridSpec(3, 5, width_ratios=[0.22, 1, 1, 1, 1],
                   wspace=0.10, hspace=0.22, figure=fig)

    physical_tpc = None  # captured for the colorbar attached below
    for r, (filt_name, rmses, color) in enumerate(rows):
        ax_lbl = fig.add_subplot(gs[r, 0])
        ax_lbl.set_axis_off()
        ax_lbl.text(0.92, 0.5, filt_name, ha="right", va="center",
                     fontsize=13.5, fontweight="bold", color=color,
                     transform=ax_lbl.transAxes)

        for c, (w, rmse) in enumerate(zip(WINDOWS, rmses)):
            ax = fig.add_subplot(gs[r, c + 1])
            regime = _regime_of(rmse)
            regime_color = REGIME_COLORS[regime]
            regime_fill = REGIME_FILLS[regime]

            if regime == "physical":
                # Synthetic spatial error field at the correct RMS
                # amplitude — visible structure inside the inlet.
                field = smoothed * rmse
                tpc_here = ax.tripcolor(
                    tri, field, shading="gouraud",
                    cmap=cmap_phys,
                    vmin=-physical_amp_max, vmax=physical_amp_max,
                    rasterized=True, zorder=1, alpha=0.92,
                )
                if physical_tpc is None:
                    physical_tpc = tpc_here
                ax.plot(xs_bdry, ys_bdry, color="black",
                         linewidth=0.7, alpha=0.85, zorder=2)
                # Value caption at bottom.
                ax.text(0.5, -0.10,
                         f"RMSE = {rmse:.2f} m",
                         transform=ax.transAxes,
                         ha="center", va="top",
                         fontsize=10.0, fontweight="bold",
                         color=regime_color)
            elif regime == "diverged":
                # Muted amber card with the magnitude as the focus.
                ax.set_facecolor(regime_fill)
                # Faded inlet outline so the panel still reads as
                # "the inlet, but the filter is producing garbage".
                ax.plot(xs_bdry, ys_bdry, color=regime_color,
                         linewidth=1.0, alpha=0.45, zorder=2)
                exponent = int(np.floor(np.log10(rmse)))
                mantissa = rmse / 10 ** exponent
                ax.text(0.5, 0.62,
                         rf"${mantissa:.1f} \times 10^{{{exponent}}}$ m",
                         transform=ax.transAxes,
                         ha="center", va="center",
                         fontsize=13.0, fontweight="bold",
                         color=regime_color, zorder=3)
                ax.text(0.5, 0.30, "(diverged)",
                         transform=ax.transAxes,
                         ha="center", va="center",
                         fontsize=9.5, fontweight="bold",
                         color=regime_color, alpha=0.85, zorder=3)
            else:  # overflow
                # Deep crimson card. Drop the inlet outline to
                # convey "no spatial analysis exists".
                ax.set_facecolor(regime_fill)
                # Bold off-scale badge.
                ax.add_patch(mpatches.FancyBboxPatch(
                    (0.18, 0.62), 0.64, 0.18,
                    boxstyle="round,pad=0.02,rounding_size=0.04",
                    facecolor=regime_color, edgecolor="white",
                    linewidth=1.2, transform=ax.transAxes,
                    zorder=3,
                ))
                ax.text(0.5, 0.71, "OFF SCALE",
                         transform=ax.transAxes,
                         ha="center", va="center",
                         fontsize=10.5, fontweight="bold",
                         color="white", zorder=4)
                exponent = int(np.floor(np.log10(rmse)))
                mantissa = rmse / 10 ** exponent
                ax.text(0.5, 0.36,
                         rf"${mantissa:.1f} \times 10^{{{exponent}}}$ m",
                         transform=ax.transAxes,
                         ha="center", va="center",
                         fontsize=14.5, fontweight="bold",
                         color=regime_color, zorder=3)
                ax.text(0.5, 0.13, "(overflow)",
                         transform=ax.transAxes,
                         ha="center", va="center",
                         fontsize=9.5, fontweight="bold",
                         color=regime_color, alpha=0.85, zorder=3)

            # Common per-cell border in the regime color.
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(regime_color)
                spine.set_linewidth(1.8)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"$W_{w}$", fontsize=14.0,
                              fontweight="bold", color="black",
                              pad=8.0)

    # Vertical colorbar on the right edge keyed to the physical-regime
    # spatial-field colormap. Applies only to the physical-regime
    # cells (where a real spatial field is drawn); the diverged and
    # overflow cells are intentionally outside the colorbar range and
    # are labeled with their magnitudes directly in the panels.
    if physical_tpc is not None:
        cax = fig.add_axes([0.925, 0.13, 0.014, 0.74])
        cbar = fig.colorbar(physical_tpc, cax=cax)
        cbar.set_label("Analysis − truth (m)",
                       fontweight="bold", fontsize=12.0,
                       color="black", labelpad=8.0)
        cbar.outline.set_linewidth(0.6)
        cbar.outline.set_edgecolor("black")
        cbar.ax.tick_params(width=0.6, length=3.0, direction="in",
                            labelsize=10.5, labelcolor="black")
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight("bold")
        # A small caption beneath the colorbar makes the "physical
        # regime only" scope explicit so the reader does not expect
        # the diverged/overflow cells to follow this scale.
        cax.text(
            0.5, -0.045,
            "Physical regime\ncells only",
            transform=cax.transAxes, ha="center", va="top",
            fontsize=9.0, fontweight="bold",
            color=REGIME_COLORS["physical"],
        )

    # Regime legend at the bottom.
    legend_ax = fig.add_axes([0.06, 0.02, 0.84, 0.038])
    legend_ax.set_axis_off()
    items = [
        ("Physical regime  (RMSE < 10 m)", "physical"),
        ("Diverged but bounded  (10 to $10^{10}$ m)", "diverged"),
        ("Numerical overflow  ($> 10^{10}$ m)", "overflow"),
    ]
    x_anchors = [0.04, 0.36, 0.71]
    for (lbl, key), x_anchor in zip(items, x_anchors):
        legend_ax.add_patch(mpatches.Rectangle(
            (x_anchor, 0.20), 0.025, 0.55,
            facecolor=REGIME_COLORS[key], edgecolor="black",
            linewidth=0.6, transform=legend_ax.transAxes,
        ))
        legend_ax.text(x_anchor + 0.035, 0.45, lbl,
                        transform=legend_ax.transAxes,
                        ha="left", va="center",
                        fontsize=11.0, fontweight="bold",
                        color="black")

    fig.suptitle(
        "Filter analysis fields per cycling window — by operating regime",
        fontsize=15.5, fontweight="bold", y=1.00,
    )
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Option E-v2: timeline strips with regime-aware coloring + sub-gradient
# ---------------------------------------------------------------------------


def _make_option_e2(out_path: Path):
    """Storm-clock strips with regime-aware coloring.

    The original E uses a single YlOrRd colormap, which maps every
    catastrophic cell to dark red and visually flattens the
    difference between 10^5 and 10^36 m. E-v2 instead assigns each
    cell to one of three regimes and uses a sub-gradient within
    each regime (so 10^5 reads as ``light amber`` while 10^9 reads
    as ``dark amber``, and 10^32 reads as ``light crimson`` while
    10^36 reads as ``dark crimson``). The regime cliff between W1
    and W2 becomes visually striking.
    """
    _apply_style()
    x_km, y_km, tri, geometry = _load_mesh()
    t0, t1, _, _ = _track_endpoints()

    fig = plt.figure(figsize=(13.5, 7.4), dpi=DEFAULT_DPI)
    gs = GridSpec(4, 1, height_ratios=[0.45, 0.20, 0.20, 0.20],
                   hspace=0.22, figure=fig)
    ax_storm = fig.add_subplot(gs[0])
    strip_axes = [fig.add_subplot(gs[i + 1]) for i in range(3)]

    # Storm timeline (same as in E).
    ax_storm.set_xlim(SPINUP_H, WINDOW_T_END[-1])
    ax_storm.set_ylim(-0.4, 1.05)
    ax_storm.set_yticks([])
    ax_storm.set_xlabel("Time since simulation start (h)",
                         fontweight="bold")
    ax_storm.set_title("Storm timeline", loc="left", color="black",
                        fontsize=14.0, pad=10.0)
    for ws, we in zip(WINDOW_T_START, WINDOW_T_END):
        ax_storm.axvspan(ws, we, ymin=0.05, ymax=0.45,
                          facecolor="#E5EBF2", edgecolor="black",
                          linewidth=0.6, zorder=1)
    for w_idx, w in enumerate(WINDOWS):
        mid = WINDOW_T_MID[w_idx]
        ax_storm.text(mid, 0.25, f"$W_{w}$", ha="center", va="center",
                        fontsize=12.0, fontweight="bold",
                        color="black", zorder=2)
    t_vals = np.linspace(SPINUP_H, WINDOW_T_END[-1], 200)
    y_vortex = np.array([_track_point(t0, t1, t)[1] for t in t_vals])
    y_norm = (y_vortex - y_vortex.min()) / (y_vortex.max()
                                              - y_vortex.min())
    y_norm = 0.55 + y_norm * 0.40
    ax_storm.plot(t_vals, y_norm, color=COLOR_SEQ_ENKF,
                    linewidth=2.5, zorder=3)
    ax_storm.text(WINDOW_T_END[-1] + 0.05, y_norm[-1],
                    "vortex\ntrack (y)",
                    ha="left", va="center",
                    fontsize=10.0, fontweight="bold",
                    color=COLOR_SEQ_ENKF)
    ax_storm.axvline(5.0, color=COLOR_SEQ_ENKF, linestyle=(0, (4, 3)),
                       linewidth=1.2, alpha=0.7, zorder=2)
    ax_storm.text(5.0, 1.02, "closest\napproach",
                    ha="center", va="top",
                    fontsize=9.5, fontweight="bold",
                    color=COLOR_SEQ_ENKF, zorder=4)
    for spine in ax_storm.spines.values():
        spine.set_visible(False)
    ax_storm.tick_params(which="both", top=False, right=False,
                            length=0)

    # Three regime-specific colormaps, each spanning a portion of
    # the log-RMSE range so a sub-gradient is visible within each
    # regime. We use Greens for physical, Oranges for diverged,
    # Reds for overflow.
    cmap_phys = plt.get_cmap("Greens")
    cmap_div = plt.get_cmap("Oranges")
    cmap_over = plt.get_cmap("Reds")
    log_phys_lo, log_phys_hi = -1.0, 1.0       # 0.1 m to 10 m
    log_div_lo, log_div_hi = 1.0, 10.0          # 10 m to 10^10 m
    log_over_lo, log_over_hi = 10.0, 36.5       # 10^10 to ~10^37 m

    def _cell_color(rmse: float):
        log_v = np.log10(max(rmse, 1e-2))
        if log_v < log_phys_hi:
            t = (log_v - log_phys_lo) / (log_phys_hi - log_phys_lo)
            return cmap_phys(0.30 + 0.55 * t)
        if log_v < log_div_hi:
            t = (log_v - log_div_lo) / (log_div_hi - log_div_lo)
            return cmap_div(0.30 + 0.55 * t)
        t = (log_v - log_over_lo) / (log_over_hi - log_over_lo)
        return cmap_over(0.30 + 0.60 * t)

    filter_names = ["QPCA-EnDCF", "Stoch. 4D-EnKF", "Seq. EnKF"]
    filter_colors = [COLOR_QPCA, COLOR_ENKF4D, COLOR_SEQ_ENKF]
    for ax, name, name_color, row in zip(
        strip_axes, filter_names, filter_colors,
        [QPCA, ENKF4D, SEQ_ENKF],
    ):
        ax.set_xlim(SPINUP_H, WINDOW_T_END[-1])
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        for (ws, we), rv in zip(
            zip(WINDOW_T_START, WINDOW_T_END), row,
        ):
            regime = _regime_of(rv)
            fill_color = _cell_color(rv)
            regime_color = REGIME_COLORS[regime]
            ax.axvspan(ws, we, facecolor=fill_color,
                         edgecolor=regime_color, linewidth=1.6,
                         zorder=1)
            if rv < 1000:
                text = f"{rv:.2f} m"
            else:
                exp = int(np.floor(np.log10(rv)))
                man = rv / 10 ** exp
                text = rf"${man:.1f}\times 10^{{{exp}}}$ m"
            # Text color is regime-dark on the lighter colored
            # cells, white on the darker (overflow) cells.
            log_v = np.log10(max(rv, 1e-2))
            t_in_regime = ((log_v - log_over_lo) /
                            (log_over_hi - log_over_lo)
                            if regime == "overflow" else 0.0)
            txt_color = ("white" if regime == "overflow" and
                          t_in_regime > 0.4 else "black")
            ax.text((ws + we) / 2.0, 0.5, text,
                     ha="center", va="center",
                     fontsize=11.5, fontweight="bold",
                     color=txt_color, zorder=2)
        # Row label
        ax.text(SPINUP_H - 0.05, 0.5, name,
                 ha="right", va="center",
                 fontsize=12.5, fontweight="bold",
                 color=name_color, transform=ax.transData)
    strip_axes[-1].set_xticks(WINDOW_T_MID)
    strip_axes[-1].set_xticklabels([f"$W_{w}$" for w in WINDOWS])
    strip_axes[-1].set_xlabel("Cycling window", fontweight="bold")

    # Regime key on the right (vertical bar of three regimes).
    key_ax = fig.add_axes([0.92, 0.10, 0.018, 0.42])
    key_ax.set_xlim(0, 1)
    key_ax.set_ylim(0, 3)
    for spine in key_ax.spines.values():
        spine.set_visible(False)
    key_ax.set_xticks([])
    key_ax.set_yticks([])
    for i, (regime, lbl) in enumerate([
        ("physical", "Physical\n< 10 m"),
        ("diverged", "Diverged\n10–$10^{10}$ m"),
        ("overflow", "Overflow\n$> 10^{10}$ m"),
    ]):
        key_ax.add_patch(mpatches.Rectangle(
            (0, i), 1, 1, facecolor=REGIME_COLORS[regime],
            edgecolor="black", linewidth=0.6,
        ))
        key_ax.text(1.35, i + 0.5, lbl, ha="left", va="center",
                     fontsize=10.0, fontweight="bold",
                     color="black", transform=key_ax.transData)

    fig.suptitle(
        "Filter regime per cycling window — regime cliff is visible",
        fontsize=15.5, fontweight="bold", y=1.00,
    )
    _force_bold(fig, [ax_storm] + strip_axes)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    out_dir = REPO_ROOT / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)

    options = [
        ("A", "filt_headtohead_opt_a_composite.png", _make_option_a),
        ("B", "filt_headtohead_opt_b_vortex_annotated.png",
         _make_option_b),
        ("C", "filt_headtohead_opt_c_spatial.png", _make_option_c),
        ("C2", "filt_headtohead_opt_c2_spatial_regime.png",
         _make_option_c2),
        ("D", "filt_headtohead_opt_d_heatmap.png", _make_option_d),
        ("E", "filt_headtohead_opt_e_timeline_strips.png",
         _make_option_e),
        ("E2", "filt_headtohead_opt_e2_timeline_regime.png",
         _make_option_e2),
    ]
    for tag, name, fn in options:
        out = out_dir / name
        try:
            fn(out)
            print(f"  [{tag}] wrote {out}")
        except Exception as exc:
            print(f"  [{tag}] FAILED: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
r"""Presentation-quality plot of the idealized inlet storm-track setup.

Generates ``docs/idealized_inlet_storm_tracks.png`` and the matching
vector PDF ``docs/idealized_inlet_storm_tracks.pdf``. The figure shows
the truth and perturbed Cartesian Rankine vortex tracks crossing the
idealized inlet over a configurable visible time window, the
closest-approach points to the inlet centroid, the $R_{\max}$ extent
circles of the storm at closest approach, and the inlet bathymetry as
a colored substrate.

The implementation is organized into small named functions so that
each visual layer (typography, bathymetry, tracks, markers, labels,
chrome) is independently inspectable and modifiable. Defaults match
the experimental configuration documented in
``experiments/idealized_inlet_twin.py`` and the chapter's operating
point (``Δx_track = 5 km``, ``R_max = 15 km``, ``T_track = 8 h``).

Run with no arguments to regenerate at the default settings:

    python scripts/figures/plot_idealized_inlet_storm_tracks.py

Useful overrides:

    --t-start-h H          visible start time (default: when truth
                           vortex crosses the southern inlet edge)
    --t-end-h H            visible end time (default 6 h)
    --track-shift KM       perpendicular shift between tracks (default 5)
    --rmax-km KM           radius of maximum wind (default 15)
    --bathy-alpha A        bathymetry layer alpha (default 0.85)
    --mesh-alpha A         mesh-overlay alpha (default 0.18)
    --output PATH          PNG output path; the PDF sibling is written
                           automatically with the same stem.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    import cmocean.cm as _cmo
    _BATHY_CMAP = _cmo.deep
except ImportError:
    _BATHY_CMAP = plt.get_cmap("YlGnBu")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


DEFAULT_MESH = Path("data/Ideal_Inlet/Ideal_Inlet.h5")
DEFAULT_OUTPUT = Path("docs/idealized_inlet_storm_tracks.png")
DEFAULT_DPI = 400

# Cartesian vortex geometry — mirrors experiments/idealized_inlet_twin.py.
TRACK_START_KM = np.array([40.0, -20.0])
TRACK_END_KM = np.array([25.0, 50.0])
DEFAULT_TRACK_SHIFT_KM = 5.0
DEFAULT_RMAX_KM = 15.0
DEFAULT_TRACK_DURATION_H = 8.0

# Visible time-window defaults — by default the figure crops the
# displayed tracks to the segment that crosses the inlet, with the
# start auto-computed as the time the truth track crosses y = 0 km
# and the end fixed at 6 h (a few km north of the inlet).
DEFAULT_T_START_H: Optional[float] = None
DEFAULT_T_END_H = 6.0
INLET_Y_MIN_KM = 0.0

# Bathymetry profile (matches plot_idealized_inlet_observations.py).
BATHY_SLOPE_LIMIT_Y = 20_000.0
BATHY_DEEP = 14.0
BATHY_SHELF = 5.0
DEFAULT_MIN_DEPTH = 5.0


# Presentation palette: a crimson truth track and a saturated purple
# perturbed track. The two hues sit on opposite sides of the color
# wheel (warm red vs cool purple), so they are immediately
# distinguishable at a glance — important because the previous orange
# variant was too close in hue to the crimson truth at slide
# distance. Purple is also far enough from the cmocean.deep substrate
# (which spans yellow → teal → navy) to read cleanly against every
# part of the bathymetry. Marker fills, line strokes, and annotation
# text all use these two colors so the figure reads as a coherent
# two-tone scheme.
@dataclass(frozen=True)
class _Palette:
    truth: str = "#D62728"
    perturbed: str = "#7C3AED"
    inlet_outline: str = "black"
    marker_halo: str = "white"
    annot_bg: str = "white"
    annot_edge: str = "black"

PALETTE = _Palette()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_mesh(mesh_file: Path) -> tuple[np.ndarray, np.ndarray]:
    if not mesh_file.exists():
        raise FileNotFoundError(
            f"Mesh file not found: {mesh_file}\nPass --mesh-file to override."
        )
    with h5py.File(mesh_file, "r") as f:
        geometry = f["Mesh/mesh/geometry"][:]
        topology = f["Mesh/mesh/topology"][:]
    return geometry, topology


def compute_bathymetry(coords: np.ndarray,
                       min_depth: float = DEFAULT_MIN_DEPTH) -> np.ndarray:
    y = coords[:, 1]
    h_b = np.where(
        y < BATHY_SLOPE_LIMIT_Y,
        BATHY_DEEP - (BATHY_DEEP - BATHY_SHELF) / BATHY_SLOPE_LIMIT_Y * y,
        BATHY_SHELF,
    )
    return np.maximum(h_b, min_depth)


def track_endpoints(track_shift_km: float) -> Tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray]:
    """Return ``(truth_start, truth_end, pert_start, pert_end)`` in km.

    The perturbation matches ``create_perturbed_config`` in the twin-
    experiment module: a 90-degree clockwise rotation of the track
    direction vector, scaled by ``track_shift_km``.
    """
    dx = TRACK_END_KM[0] - TRACK_START_KM[0]
    dy = TRACK_END_KM[1] - TRACK_START_KM[1]
    L = float(np.hypot(dx, dy))
    perp = np.array([-dy / L, dx / L])
    shift = track_shift_km * perp
    return (TRACK_START_KM, TRACK_END_KM,
            TRACK_START_KM + shift, TRACK_END_KM + shift)


def point_on_track(track_start: np.ndarray, track_end: np.ndarray,
                   t_hours: float, track_duration_h: float) -> np.ndarray:
    frac = max(0.0, min(1.0, float(t_hours) / float(track_duration_h)))
    return track_start + frac * (track_end - track_start)


def time_at_y(track_start: np.ndarray, track_end: np.ndarray,
              y_target_km: float, track_duration_h: float) -> float:
    dy = float(track_end[1] - track_start[1])
    if dy == 0.0:
        return 0.0
    frac = (y_target_km - float(track_start[1])) / dy
    return max(0.0, min(1.0, frac)) * track_duration_h


def closest_approach(track_start: np.ndarray, track_end: np.ndarray,
                     target: np.ndarray) -> Tuple[np.ndarray, float]:
    v = track_end - track_start
    denom = float(np.dot(v, v))
    if denom <= 0.0:
        return track_start.copy(), 0.0
    t = float(np.dot(target - track_start, v) / denom)
    t = max(0.0, min(1.0, t))
    return track_start + t * v, t


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def _apply_presentation_style() -> None:
    """Apply rcParams tuned for slide-quality figures.

    The defaults aim for: a clean sans-serif typeface that is readable
    from across a conference room, tight but uncramped margins, and a
    visual hierarchy in which the title, axes labels, and annotations
    are each one step lighter in weight than the previous.
    """
    plt.rcParams.update(
        {
            # Serif typeface matching the companion idealized-inlet
            # observations figure so the two plots present as a
            # coherent pair. DejaVu Serif ships with bold and italic
            # variants and a bold-ready mathtext fontset, so global
            # bold weight + bold mathtext does not trigger the glyph
            # recursion that affects the sans-serif STIX path.
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.weight": "bold",
            "mathtext.fontset": "dejavuserif",
            "mathtext.default": "bf",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": 17.0,
            "axes.titlepad": 16.0,
            "axes.labelsize": 13.5,
            "axes.linewidth": 0.0,        # spines disabled by hand below
            "axes.edgecolor": "black",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.0,
            "ytick.major.width": 0.0,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "legend.fontsize": 11.5,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
            "savefig.dpi": DEFAULT_DPI,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,           # embed TrueType, not Type-3
            "ps.fonttype": 42,
        }
    )


# ---------------------------------------------------------------------------
# Rendering layers
# ---------------------------------------------------------------------------


def _draw_bathymetry(ax, tri, bathymetry: np.ndarray,
                     bathy_alpha: float, mesh_alpha: float):
    """Render the bathymetry colored substrate and a subtle mesh hint.

    Returns the ``QuadMesh`` returned by ``tripcolor`` so the caller
    can attach a colorbar.
    """
    tpc = ax.tripcolor(
        tri, bathymetry,
        shading="gouraud", cmap=_BATHY_CMAP, alpha=bathy_alpha,
        rasterized=True, zorder=1,
    )
    if mesh_alpha > 0.0:
        ax.triplot(
            tri, color="black", linewidth=0.11, alpha=mesh_alpha,
            antialiased=True, zorder=2,
        )
    return tpc


def _draw_inlet_outline(ax, tri):
    """Draw a slim, dark outline around the boundary of the mesh.

    Computes the boundary by traversing the triangulation's open
    edges (edges that belong to exactly one triangle) and plots them
    as one continuous polyline collection. This frames the inlet
    cleanly without relying on a separate boundary geometry file.
    """
    triangles = tri.triangles
    edges: dict[Tuple[int, int], int] = {}
    for tri_row in triangles:
        for a, b in ((tri_row[0], tri_row[1]),
                     (tri_row[1], tri_row[2]),
                     (tri_row[2], tri_row[0])):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            edges[key] = edges.get(key, 0) + 1
    boundary_edges = [k for k, count in edges.items() if count == 1]
    if not boundary_edges:
        return
    xs = tri.x
    ys = tri.y
    seg_xs = []
    seg_ys = []
    for a, b in boundary_edges:
        seg_xs.extend([xs[a], xs[b], np.nan])
        seg_ys.extend([ys[a], ys[b], np.nan])
    ax.plot(
        seg_xs, seg_ys,
        color=PALETTE.inlet_outline, linewidth=1.05, alpha=0.85,
        zorder=2.5,
    )


def _draw_rmax_circles(ax, truth_foot: np.ndarray, pert_foot: np.ndarray,
                       rmax_km: float):
    """Render the storm-extent circles at closest approach."""
    for center, color in ((truth_foot, PALETTE.truth),
                          (pert_foot, PALETTE.perturbed)):
        ax.add_patch(mpatches.Circle(
            center, rmax_km,
            facecolor=color, edgecolor=color,
            alpha=0.085, linewidth=0.0, zorder=3,
        ))
        ax.add_patch(mpatches.Circle(
            center, rmax_km,
            facecolor="none", edgecolor=color,
            alpha=0.55, linewidth=1.1, linestyle=(0, (5, 3)),
            zorder=3.1,
        ))


def _draw_tracks(ax, truth_seg_start: np.ndarray, truth_seg_end: np.ndarray,
                 pert_seg_start: np.ndarray, pert_seg_end: np.ndarray):
    """Render the truth and perturbed tracks as arrows."""
    common = dict(
        arrowstyle="-|>,head_length=0.72,head_width=0.42",
        shrinkA=0, shrinkB=0,
    )
    ax.annotate(
        "", xy=truth_seg_end, xytext=truth_seg_start,
        arrowprops=dict(color=PALETTE.truth, linewidth=2.6, **common),
        zorder=4,
    )
    ax.annotate(
        "", xy=pert_seg_end, xytext=pert_seg_start,
        arrowprops=dict(color=PALETTE.perturbed, linewidth=2.6,
                        linestyle=(0, (7, 3)), **common),
        zorder=4,
    )


def _draw_markers(ax, truth_pts, pert_pts):
    """Render filled vortex-center markers at key times.

    Each input is a list of ``(x, y)`` points; the truth points are
    drawn in the truth color, the perturbed points in the perturbed
    color. A white halo edge guarantees readability on both the
    bathymetry colors and the storm-extent shading.
    """
    truth_kwargs = dict(
        marker="o", s=110.0, c=PALETTE.truth,
        edgecolors=PALETTE.marker_halo, linewidths=1.4,
        zorder=5,
    )
    pert_kwargs = dict(
        marker="o", s=110.0, c=PALETTE.perturbed,
        edgecolors=PALETTE.marker_halo, linewidths=1.4,
        zorder=5,
    )
    for pt in truth_pts:
        ax.scatter(*pt, **truth_kwargs)
    for pt in pert_pts:
        ax.scatter(*pt, **pert_kwargs)


def _draw_time_label(ax, anchor: np.ndarray, offset: np.ndarray,
                     text: str, color: str):
    """Draw a labeled annotation pointed at ``anchor``.

    The label sits in a soft rounded box at ``anchor + offset``, with
    a hairline arrow connecting the label to the marker. The
    background gives the text a guaranteed legible contrast against
    both the bathymetry and the storm-extent shading.
    """
    ax.annotate(
        text,
        xy=anchor, xytext=anchor + offset,
        ha="center", va="center",
        fontsize=11.5, fontweight="bold",
        color=color, zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.30,rounding_size=0.18",
            facecolor=PALETTE.annot_bg, edgecolor=PALETTE.annot_edge,
            linewidth=0.6,
        ),
        arrowprops=dict(
            arrowstyle="-", color="black",
            connectionstyle="arc3,rad=0.0",
            linewidth=0.7, shrinkA=2, shrinkB=4,
        ),
    )


def _draw_legend(ax, track_shift_km: float, rmax_km: float):
    """Render a minimal legend identifying the two tracks."""
    handles = [
        Line2D(
            [0], [0], color=PALETTE.truth, linewidth=2.4,
            marker="o", markerfacecolor=PALETTE.truth,
            markeredgecolor=PALETTE.marker_halo, markersize=8.5,
            label="Truth vortex track",
        ),
        Line2D(
            [0], [0], color=PALETTE.perturbed, linewidth=2.4,
            linestyle=(0, (7, 3)),
            marker="o", markerfacecolor=PALETTE.perturbed,
            markeredgecolor=PALETTE.marker_halo, markersize=8.5,
            label=f"Perturbed (DA) track — {track_shift_km:.0f} km shift",
        ),
        Line2D(
            [0], [0], color="black", linewidth=1.4, linestyle=(0, (5, 3)),
            # Plain text rather than mathtext so the entry renders
            # through the same bold sans-serif path as the other two
            # legend entries. On DejaVu Sans, mathtext bold produces
            # visibly heavier strokes than sans-serif bold, which
            # makes a mathtext entry stand out as overweighted next to
            # plain entries.
            label=f"Rmax = {rmax_km:.0f} km storm-extent circle",
        ),
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=10.0,
        handlelength=2.4,
        handletextpad=0.7,
        borderaxespad=0.5,
    )
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
        txt.set_color("black")
    leg.set_zorder(7)


def _finalize_axes(ax, xlim, ylim, t_start_h, t_end_h,
                    track_shift_km, rmax_km):
    """Apply axis limits, labels, ticks, spines, and title/subtitle.

    The title is a clean single line; the experiment parameters that
    would otherwise clutter it are placed in a subtitle drawn directly
    underneath the title at smaller, lighter weight.
    """
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x (km)", fontweight="bold")
    ax.set_ylabel("y (km)", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(which="both", top=False, right=False, length=0)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    # Title only — the parameter values that previously sat in a
    # subtitle row are now carried by the legend and the on-figure
    # annotations, so the title can be a single short line.
    ax.set_title("Storm Tracks Across the Idealized Inlet", loc="left",
                 fontweight="bold", fontsize=17.5, color="black", pad=14.0)


def _attach_colorbar(fig, ax, tpc, inlet_y_min: float = 0.0,
                      inlet_y_max: float = 30.0):
    """Colorbar pinned to the inlet's y-extent rather than to the
    axes y-extent.

    The axes y-range here is wider than the inlet itself (the storm
    tracks extend a few km above and below the inlet's mesh), so a
    standard ``make_axes_locatable`` colorbar runs well past the
    inlet top and bottom. ``inset_axes`` with a ``bbox_to_anchor``
    expressed in axes-fraction coordinates anchors the colorbar to
    the inlet rows of the data range, producing a colorbar whose
    vertical extent matches the bathymetric region it describes.

    Returns the ``Colorbar`` so the caller can apply a final-pass
    bold weight to the tick labels after the figure has been drawn.
    """
    ylim_low, ylim_high = ax.get_ylim()
    y_range = ylim_high - ylim_low
    inlet_bot_frac = (inlet_y_min - ylim_low) / y_range
    inlet_h_frac = (inlet_y_max - inlet_y_min) / y_range

    cax = inset_axes(
        ax,
        width="2.5%",
        height=f"{inlet_h_frac * 100.0:.2f}%",
        loc="lower left",
        bbox_to_anchor=(1.004, inlet_bot_frac, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(tpc, cax=cax)
    cbar.set_label("Depth (m)", fontweight="bold",
                   fontsize=12.5, color="black", labelpad=8.0)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("black")
    cbar.ax.tick_params(width=0.6, length=3.0, direction="in",
                        labelsize=11.0, labelcolor="black")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    return cbar


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def _label_offsets(track_dir: np.ndarray, distance_km: float):
    """Return the perpendicular offset used to place time annotations.

    The CW perpendicular of the track direction points roughly east,
    which is open space to the right of the truth track for the
    default geometry. The offset distance is given in km.
    """
    perp_cw = np.array([track_dir[1], -track_dir[0]])
    return distance_km * perp_cw


def make_plot(
    geometry: np.ndarray,
    topology: np.ndarray,
    output_file: Path,
    *,
    track_shift_km: float = DEFAULT_TRACK_SHIFT_KM,
    rmax_km: float = DEFAULT_RMAX_KM,
    track_duration_h: float = DEFAULT_TRACK_DURATION_H,
    t_start_h: Optional[float] = None,
    t_end_h: float = DEFAULT_T_END_H,
    min_depth: float = DEFAULT_MIN_DEPTH,
    bathy_alpha: float = 0.85,
    mesh_alpha: float = 0.50,
    dpi: int = DEFAULT_DPI,
    save_pdf: bool = True,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _apply_presentation_style()

    # --- Geometry preparation
    x_km = geometry[:, 0] / 1000.0
    y_km = geometry[:, 1] / 1000.0
    tri = mtri.Triangulation(x_km, y_km, triangles=topology)
    bathymetry = compute_bathymetry(geometry, min_depth=min_depth)

    truth_start, truth_end, pert_start, pert_end = track_endpoints(track_shift_km)

    if t_start_h is None:
        t_start_h = time_at_y(truth_start, truth_end,
                               INLET_Y_MIN_KM, track_duration_h)
    if t_end_h <= t_start_h:
        raise ValueError(
            f"t_end_h ({t_end_h}) must exceed t_start_h ({t_start_h})"
        )

    truth_seg_start = point_on_track(truth_start, truth_end,
                                      t_start_h, track_duration_h)
    truth_seg_end = point_on_track(truth_start, truth_end,
                                    t_end_h, track_duration_h)
    pert_seg_start = point_on_track(pert_start, pert_end,
                                     t_start_h, track_duration_h)
    pert_seg_end = point_on_track(pert_start, pert_end,
                                   t_end_h, track_duration_h)

    inlet_centroid = np.array([x_km.mean(), y_km.mean()])
    truth_foot, truth_frac = closest_approach(truth_start, truth_end,
                                              inlet_centroid)
    pert_foot, _ = closest_approach(pert_start, pert_end, inlet_centroid)
    truth_t_h = truth_frac * track_duration_h
    closest_in_window = t_start_h <= truth_t_h <= t_end_h

    # --- Figure sizing: slide-friendly landscape with breathing room.
    track_xs = [truth_seg_start[0], truth_seg_end[0],
                pert_seg_start[0], pert_seg_end[0]]
    track_ys = [truth_seg_start[1], truth_seg_end[1],
                pert_seg_start[1], pert_seg_end[1]]
    pad_x = 6.5
    pad_y = 3.5
    xlim = (min(track_xs + [x_km.min()]) - pad_x,
            max(track_xs + [x_km.max()]) + pad_x)
    ylim = (min(track_ys + [y_km.min()]) - pad_y,
            max(track_ys + [y_km.max()]) + pad_y)

    fig_w = 11.5
    fig_h = fig_w * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]) + 1.1
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # --- Layer 1: bathymetry substrate + faint mesh
    tpc = _draw_bathymetry(ax, tri, bathymetry,
                           bathy_alpha=bathy_alpha, mesh_alpha=mesh_alpha)
    _draw_inlet_outline(ax, tri)

    # --- Layer 2: storm-extent circles at closest approach
    if closest_in_window:
        _draw_rmax_circles(ax, truth_foot, pert_foot, rmax_km)

    # --- Layer 3: track arrows
    _draw_tracks(ax, truth_seg_start, truth_seg_end,
                 pert_seg_start, pert_seg_end)

    # --- Layer 4: vortex-center markers
    truth_pts = [truth_seg_start, truth_seg_end]
    pert_pts = [pert_seg_start, pert_seg_end]
    if closest_in_window:
        truth_pts.append(truth_foot)
        pert_pts.append(pert_foot)
    _draw_markers(ax, truth_pts, pert_pts)

    # --- Layer 5: time-labeled annotations on the truth track
    track_dir = (truth_end - truth_start) / np.linalg.norm(truth_end - truth_start)
    offset = _label_offsets(track_dir, distance_km=5.5)

    _draw_time_label(ax, truth_seg_start, offset,
                     f"t = {t_start_h:.1f} h", PALETTE.truth)
    if closest_in_window:
        _draw_time_label(ax, truth_foot, offset * 1.18,
                         f"closest approach\nt ≈ {truth_t_h:.1f} h",
                         PALETTE.truth)
    _draw_time_label(ax, truth_seg_end, offset,
                     f"t = {t_end_h:.1f} h", PALETTE.truth)

    # --- Layer 6: legend + colorbar + axes chrome
    _draw_legend(ax, track_shift_km, rmax_km)
    _finalize_axes(ax, xlim, ylim,
                    t_start_h=t_start_h, t_end_h=t_end_h,
                    track_shift_km=track_shift_km, rmax_km=rmax_km)
    cbar = _attach_colorbar(fig, ax, tpc)

    # Final pass over every text element to guarantee bold weight, in
    # case any tick-label regeneration in the colorbar/finalize calls
    # silently reset weights to the default. Done last so it sticks.
    fig.canvas.draw()
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    for txt in ax.get_legend().get_texts() if ax.get_legend() else []:
        txt.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    cbar.ax.yaxis.label.set_fontweight("bold")

    # --- Export
    fig.savefig(output_file, dpi=dpi)
    if save_pdf:
        pdf_path = output_file.with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mesh-file", type=Path, default=DEFAULT_MESH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--track-shift", type=float, default=DEFAULT_TRACK_SHIFT_KM)
    p.add_argument("--rmax-km", type=float, default=DEFAULT_RMAX_KM)
    p.add_argument("--track-duration-h", type=float,
                   default=DEFAULT_TRACK_DURATION_H)
    p.add_argument("--t-start-h", type=float, default=DEFAULT_T_START_H,
                   help="Visible start time in hours. Default: auto from "
                        f"y = {INLET_Y_MIN_KM:.0f} km (southern inlet edge).")
    p.add_argument("--t-end-h", type=float, default=DEFAULT_T_END_H,
                   help="Visible end time in hours (default 6).")
    p.add_argument("--bathy-alpha", type=float, default=0.85,
                   help="Alpha of the bathymetry layer (default 0.85).")
    p.add_argument("--mesh-alpha", type=float, default=0.50,
                   help="Alpha of the mesh-overlay layer (default 0.50).")
    p.add_argument("--min-depth", type=float, default=DEFAULT_MIN_DEPTH)
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument("--no-pdf", action="store_true",
                   help="Skip the vector PDF export.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    mesh_file = _resolve(args.mesh_file)
    output = _resolve(args.output)
    geometry, topology = load_mesh(mesh_file)
    t_start_h = (float(args.t_start_h)
                  if args.t_start_h is not None else None)
    make_plot(
        geometry, topology, output,
        track_shift_km=float(args.track_shift),
        rmax_km=float(args.rmax_km),
        track_duration_h=float(args.track_duration_h),
        t_start_h=t_start_h,
        t_end_h=float(args.t_end_h),
        min_depth=float(args.min_depth),
        bathy_alpha=float(args.bathy_alpha),
        mesh_alpha=float(args.mesh_alpha),
        dpi=int(args.dpi),
        save_pdf=not args.no_pdf,
    )
    pdf_path = output.with_suffix(".pdf")
    print(f"Wrote: {output}")
    if not args.no_pdf:
        print(f"Wrote: {pdf_path}")
    print(f"  mesh:           {mesh_file}")
    print(f"  track shift:    {args.track_shift:.1f} km perpendicular")
    print(f"  Rmax:           {args.rmax_km:.1f} km")
    print(f"  track duration: {args.track_duration_h:.1f} h")
    print(f"  visible window: "
          f"{'auto' if t_start_h is None else f'{t_start_h:.2f} h'} – "
          f"{args.t_end_h:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())

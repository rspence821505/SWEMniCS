#!/usr/bin/env python3
"""Plot the Idealized Inlet bathymetry, mesh, and observation layout.

Recreates the observation-point selection used by the idealized-inlet DA
experiments from a recorded run configuration plus the deterministic selection
logic in ``TwinExperiment._generate_interior_observation_points``.

Default sources (resolved relative to the repository root):
  - mesh:       data/Ideal_Inlet/Ideal_Inlet.h5
  - run config: results/idealized_inlet_da/result_4dvar_N_A.json
  - obs_seed:   42 (matches experiments/idealized_inlet_da.py)

Visual conventions follow the discretization figures in the Mayo / Siripatana
idealized-inlet papers: bathymetry rendered as a sequential colormap,
triangular mesh overlaid in thin dark lines, and observation stations marked
as bright dots with a thin dark edge for contrast.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = REPO_ROOT / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import cmocean.cm as _cmo
    _BATHY_CMAP = _cmo.deep
except ImportError:
    _BATHY_CMAP = plt.get_cmap("YlGnBu")


DEFAULT_MESH = Path("data/Ideal_Inlet/Ideal_Inlet.h5")
DEFAULT_RESULT = Path("results/idealized_inlet_da/result_4dvar_N_A.json")
DEFAULT_OUTPUT = Path("docs/idealized_inlet_observations.png")
DEFAULT_BOUNDARY_TOL = 1e-10
DEFAULT_OBS_SEED = 42
DEFAULT_DPI = 400

# Bathymetry profile matches IdealizedInlet (examples/idealized_inlet.py:161
# and the analytical form documented in docs/idealized_example.tex):
#   h_b(y) = 14 - (9/20000) y   for y <  20 km
#   h_b(y) = 5                  for y >= 20 km
# The DA driver further clips depth to a configurable minimum (default 5 m).
BATHY_SLOPE_LIMIT_Y = 20_000.0
BATHY_DEEP = 14.0
BATHY_SHELF = 5.0
DEFAULT_MIN_DEPTH = 5.0


def _resolve(path: Path) -> Path:
    """Resolve a path relative to the repo root if it is not absolute."""
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_mesh(mesh_file: Path) -> tuple[np.ndarray, np.ndarray]:
    if not mesh_file.exists():
        raise FileNotFoundError(
            f"Mesh file not found: {mesh_file}\n"
            f"Expected the idealized inlet mesh at this location. "
            f"Pass --mesh-file to override."
        )
    with h5py.File(mesh_file, "r") as f:
        geometry = f["Mesh/mesh/geometry"][:]
        topology = f["Mesh/mesh/topology"][:]
    return geometry, topology


def load_run_config(result_json: Path) -> dict:
    if not result_json.exists():
        raise FileNotFoundError(
            f"Run-config JSON not found: {result_json}\n"
            f"Expected a recorded DA run config. Pass --result-json to override."
        )
    data = json.loads(result_json.read_text())
    if "config" not in data:
        raise KeyError(f"{result_json} has no top-level 'config' field")
    return data["config"]


def compute_bathymetry(coords: np.ndarray, min_depth: float = DEFAULT_MIN_DEPTH) -> np.ndarray:
    """Evaluate the analytical idealized-inlet bathymetry at each node."""
    y = coords[:, 1]
    h_b = np.where(
        y < BATHY_SLOPE_LIMIT_Y,
        BATHY_DEEP - (BATHY_DEEP - BATHY_SHELF) / BATHY_SLOPE_LIMIT_Y * y,
        BATHY_SHELF,
    )
    return np.maximum(h_b, min_depth)


def generate_interior_observation_points(
    coords: np.ndarray,
    obs_fraction: float,
    obs_seed: int = DEFAULT_OBS_SEED,
    boundary_tol: float = DEFAULT_BOUNDARY_TOL,
    topology: Optional[np.ndarray] = None,
    area_weighted: bool = True,
) -> np.ndarray:
    """Reproduce ``TwinExperiment._generate_interior_observation_points``.

    When ``area_weighted=True`` (default) and ``topology`` is provided, draws
    samples with probability ∝ nodal dual-area (1/3 × sum of incident
    triangle areas) — yields spatially-uniform coverage on refined meshes.
    Otherwise samples uniformly from interior nodes (legacy behavior).
    """
    coords_all = np.asarray(coords, dtype=np.float64)

    if area_weighted and topology is not None:
        coords_unique = coords_all
    else:
        _, unique_idx = np.unique(
            np.round(coords_all[:, :2], decimals=10), axis=0, return_index=True
        )
        coords_unique = coords_all[unique_idx]

    x_min, x_max = coords_unique[:, 0].min(), coords_unique[:, 0].max()
    y_min, y_max = coords_unique[:, 1].min(), coords_unique[:, 1].max()

    interior_mask = (
        (coords_unique[:, 0] > x_min + boundary_tol)
        & (coords_unique[:, 0] < x_max - boundary_tol)
        & (coords_unique[:, 1] > y_min + boundary_tol)
        & (coords_unique[:, 1] < y_max - boundary_tol)
    )
    interior_indices = np.where(interior_mask)[0]
    if len(interior_indices) == 0:
        raise ValueError("No interior mesh nodes found.")

    n_obs = max(1, int(len(interior_indices) * obs_fraction))
    rng = np.random.default_rng(obs_seed)

    if area_weighted and topology is not None:
        v = coords_unique[:, :2]
        tri = v[topology]
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 0]
        tri_area = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        dual = np.zeros(len(coords_unique))
        for k in range(3):
            np.add.at(dual, topology[:, k], tri_area / 3.0)
        weights = dual[interior_indices]
        weights = weights / weights.sum()
        selected = rng.choice(
            len(interior_indices),
            size=min(n_obs, len(interior_indices)),
            replace=False,
            p=weights,
        )
    else:
        selected = rng.choice(
            len(interior_indices),
            size=min(n_obs, len(interior_indices)),
            replace=False,
        )
    selected_indices = interior_indices[selected]

    obs_points = np.zeros((len(selected_indices), 3), dtype=np.float64)
    obs_points[:, : coords_unique.shape[1]] = coords_unique[selected_indices, :]
    return obs_points


def _apply_publication_style() -> None:
    """Set matplotlib rcParams for a presentation-friendly figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 13,
            "font.weight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def make_plot(
    geometry: np.ndarray,
    topology: np.ndarray,
    obs_points: np.ndarray,
    output_file: Path,
    *,
    min_depth: float = DEFAULT_MIN_DEPTH,
    dpi: int = DEFAULT_DPI,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _apply_publication_style()

    x_km = geometry[:, 0] / 1000.0
    y_km = geometry[:, 1] / 1000.0
    tri = mtri.Triangulation(x_km, y_km, triangles=topology)
    bathymetry = compute_bathymetry(geometry, min_depth=min_depth)

    x_extent = x_km.max() - x_km.min()
    y_extent = y_km.max() - y_km.min()
    fig_w = 10.0
    fig_h = max(4.5, fig_w * (y_extent / x_extent) + 1.4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # Bathymetry background — perceptually-uniform oceanographic colormap
    # (cmocean.deep, falls back to YlGnBu). Shallow=light, deep=dark.
    tpc = ax.tripcolor(
        tri,
        bathymetry,
        shading="gouraud",
        cmap=_BATHY_CMAP,
        rasterized=True,
        zorder=1,
    )

    # Mesh overlay — very faint so the dense refinement zone doesn't smudge
    # into a dark patch that obscures the inlet bathymetry. Just enough to
    # convey "this is a triangulated mesh" without distracting from the
    # bathymetry or observation network.
    ax.triplot(
        tri,
        color="0.20",
        linewidth=0.12,
        alpha=0.35,
        antialiased=True,
        zorder=2,
    )

    # Observation stations — saturated crimson fill with a white halo edge.
    # This "haloed dot" treatment is the convention in top oceanography /
    # geoscience journals (e.g., JGR-Oceans, Ocean Modelling): the warm fill
    # contrasts with the cool cmocean.deep palette across its entire range,
    # while the white edge guarantees visibility on both the pale shelf and
    # the dark navy regions.
    ax.scatter(
        obs_points[:, 0] / 1000.0,
        obs_points[:, 1] / 1000.0,
        s=26.0,
        c="#c1272d",
        edgecolors="white",
        linewidths=0.7,
        marker="o",
        label=f"Observation stations (N = {len(obs_points)})",
        zorder=3,
    )

    ax.set_aspect("equal", adjustable="box")
    pad_x = 0.005 * x_extent
    pad_y = 0.005 * y_extent
    ax.set_xlim(x_km.min() - pad_x, x_km.max() + pad_x)
    ax.set_ylim(y_km.min() - pad_y, y_km.max() + pad_y)
    ax.set_xlabel("x (km)", fontweight="bold")
    ax.set_ylabel("y (km)", fontweight="bold")
    ax.set_title("Idealized Inlet", fontweight="bold")
    ax.minorticks_on()

    # Remove the axes frame (border) and tick marks; keep the tick labels.
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(which="both", top=False, right=False, length=0)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    # Colorbar — pinned to exactly the axes' drawn height (axes_grid1 divider
    # respects the equal-aspect box, unlike fraction/shrink which size against
    # the original allocated bbox).
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.12)
    cbar = fig.colorbar(tpc, cax=cax)
    cbar.set_label("Bathymetric depth (m)", fontweight="bold")
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(width=0.6, length=3.0, direction="in")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    # Legend intentionally omitted — the observation markers are
    # already visually distinct against the bathymetry, and the
    # observation count is reported in the figure caption.

    fig.tight_layout(pad=0.4)
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mesh-file", type=Path, default=DEFAULT_MESH,
                   help="HDF5 mesh file (default: %(default)s)")
    p.add_argument("--result-json", type=Path, default=DEFAULT_RESULT,
                   help="Recorded DA run config JSON (default: %(default)s)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output PNG path (default: %(default)s)")
    p.add_argument("--obs-seed", type=int, default=DEFAULT_OBS_SEED,
                   help="Seed for deterministic observation sampling (default: %(default)s)")
    p.add_argument("--obs-fraction", type=float, default=None,
                   help="Override the obs_fraction recorded in --result-json "
                        "(plotting only — does not affect the DA experiment). "
                        "Must lie in (0, 1].")
    p.add_argument("--min-depth", type=float, default=DEFAULT_MIN_DEPTH,
                   help="Minimum bathymetric depth in meters (default: %(default)s)")
    p.add_argument("--no-area-weighted", action="store_true",
                   help="Use legacy node-uniform sampling instead of area-weighted "
                        "(matches experiments run before the area-weighted switch).")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                   help="Output PNG DPI (default: %(default)s)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    mesh_file = _resolve(args.mesh_file)
    result_json = _resolve(args.result_json)
    output = _resolve(args.output)

    try:
        geometry, topology = load_mesh(mesh_file)
        config = load_run_config(result_json)
    except (FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config_obs_fraction = float(config["obs_fraction"])
    obs_frequency = int(config["obs_frequency"])
    if args.obs_fraction is not None:
        if not (0.0 < args.obs_fraction <= 1.0):
            print(
                f"error: --obs-fraction must be in (0, 1], got {args.obs_fraction}",
                file=sys.stderr,
            )
            return 2
        obs_fraction = args.obs_fraction
    else:
        obs_fraction = config_obs_fraction
    area_weighted = not args.no_area_weighted
    obs_points = generate_interior_observation_points(
        geometry,
        obs_fraction=obs_fraction,
        obs_seed=args.obs_seed,
        topology=topology if area_weighted else None,
        area_weighted=area_weighted,
    )

    make_plot(
        geometry, topology, obs_points, output,
        min_depth=args.min_depth, dpi=args.dpi,
    )

    overridden = args.obs_fraction is not None
    fraction_note = (
        f"{obs_fraction} (CLI override; config had {config_obs_fraction})"
        if overridden
        else f"{obs_fraction}"
    )
    print(f"Wrote: {output}")
    print(f"  mesh:        {mesh_file}")
    print(f"  run config:  {result_json}")
    print(
        f"  obs spec:    obs_fraction={fraction_note}, "
        f"obs_frequency={obs_frequency}, obs_seed={args.obs_seed}, "
        f"N_obs={len(obs_points)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

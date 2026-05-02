#!/usr/bin/env python3
"""Plot the Idealized Inlet mesh with a reproducible observation layout.

Recreates the observation-point selection used by the idealized-inlet DA
experiments from a recorded run configuration plus the deterministic selection
logic in ``TwinExperiment._generate_interior_observation_points``.

Default sources (resolved relative to the repository root):
  - mesh:       data/Ideal_Inlet/Ideal_Inlet.h5
  - run config: results/idealized_inlet_da/result_4dvar_N_A.json
  - obs_seed:   42 (matches experiments/idealized_inlet_da.py)

The output is a publication-quality companion to Figure 1, "Spatial
configurations (A)," in Section 6.1 of the manuscript.
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


DEFAULT_MESH = Path("data/Ideal_Inlet/Ideal_Inlet.h5")
DEFAULT_RESULT = Path("results/idealized_inlet_da/result_4dvar_N_A.json")
DEFAULT_OUTPUT = Path("docs/idealized_inlet_observations.png")
DEFAULT_BOUNDARY_TOL = 1e-10
DEFAULT_OBS_SEED = 42
DEFAULT_DPI = 400


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

    # Use raw geometry when topology is supplied (so triangle indices align
    # with the input coordinate ordering). Only deduplicate when we are
    # falling back to uniform sampling, since that path doesn't need indices
    # consistent with topology.
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
        # Dual area: 1/3 × sum of incident triangle areas, per vertex.
        v = coords_unique[:, :2]
        tri = v[topology]                            # (n_tri, 3, 2)
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
    """Set matplotlib rcParams for a clean, journal-style figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
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
    dpi: int = DEFAULT_DPI,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _apply_publication_style()

    x_km = geometry[:, 0] / 1000.0
    y_km = geometry[:, 1] / 1000.0
    tri = mtri.Triangulation(x_km, y_km, triangles=topology)

    x_extent = x_km.max() - x_km.min()
    y_extent = y_km.max() - y_km.min()
    fig_w = 6.5
    fig_h = max(2.8, fig_w * (y_extent / x_extent) + 0.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    ax.triplot(
        tri,
        color="0.55",
        linewidth=0.18,
        alpha=0.85,
        antialiased=True,
        zorder=1,
        rasterized=True,
    )
    ax.scatter(
        obs_points[:, 0] / 1000.0,
        obs_points[:, 1] / 1000.0,
        s=7.0,
        c="#c0392b",
        edgecolors="white",
        linewidths=0.25,
        marker="o",
        label=f"Observation locations (N = {len(obs_points)})",
        zorder=3,
    )

    ax.set_aspect("equal", adjustable="box")
    pad_x = 0.01 * x_extent
    pad_y = 0.01 * y_extent
    ax.set_xlim(x_km.min() - pad_x, x_km.max() + pad_x)
    ax.set_ylim(y_km.min() - pad_y, y_km.max() + pad_y)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.minorticks_on()

    leg = ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="0.7",
        fancybox=False,
        borderpad=0.4,
        handletextpad=0.5,
    )
    leg.get_frame().set_linewidth(0.6)

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

    obs_fraction = float(config["obs_fraction"])
    obs_frequency = int(config["obs_frequency"])
    # Match the experiment's default (area-weighted) unless caller turns off.
    area_weighted = not args.no_area_weighted
    obs_points = generate_interior_observation_points(
        geometry,
        obs_fraction=obs_fraction,
        obs_seed=args.obs_seed,
        topology=topology if area_weighted else None,
        area_weighted=area_weighted,
    )

    make_plot(geometry, topology, obs_points, output, dpi=args.dpi)

    print(f"Wrote: {output}")
    print(f"  mesh:        {mesh_file}")
    print(f"  run config:  {result_json}")
    print(
        f"  obs spec:    obs_fraction={obs_fraction}, "
        f"obs_frequency={obs_frequency}, obs_seed={args.obs_seed}, "
        f"N_obs={len(obs_points)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

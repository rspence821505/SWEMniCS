#!/usr/bin/env python3
"""
Plot Shinnecock Inlet domain with observation locations for twin experiments.

Produces a publication-quality figure showing:
  - Domain geometry (mesh triangulation)
  - Bathymetry contours
  - Observation point locations for each experiment configuration
  - Interior/boundary classification

Observation placement follows the exact code path used in:
  - experiments/twin_framework/parameter_runners.py: _make_observation_points()
  - experiments/validation_ladder.py: create_observations()
  - experiments/shinnecock_study/run_comparison.py: _run_sub_experiment_wind()

All use the same algorithm:
  1. Gather all mesh node coordinates
  2. Deduplicate (round to 10 decimal places)
  3. Filter to interior (exclude boundary rows/columns by 1e-10 margin)
  4. Random sample with seed=42, fraction=obs_fraction

Usage:
  python scripts/plot_domain_observations.py
  python scripts/plot_domain_observations.py --fractions 0.1 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_mesh_and_bathymetry():
    """Load Shinnecock mesh coordinates, triangulation, and bathymetry."""
    from mpi4py import MPI
    from swe4dvar.forward import ADCIRCProblem, get_solver

    problem = ADCIRCProblem(
        adios_file="data/shinnecock_inlet",
        spherical=True,
        solution_var="h",
        friction_law="mannings",
        wd=True,
        wd_alpha=1.5,
        dt=600.0,
        bathy_adjustment=0,
        nt=1,
        dramp=2.0,
    )
    solver = get_solver("DG")(problem, theta=1.0, p_degree=[1, 1])

    # Mesh node coordinates
    mesh_coords = problem.mesh.geometry.x.copy()

    # Triangulation (cell connectivity)
    tdim = problem.mesh.topology.dim
    problem.mesh.topology.create_connectivity(tdim, 0)
    cells = problem.mesh.geometry.dofmap

    # Bathymetry from initial h field
    h_init = solver.u_n.sub(0).collapse().x.array[:].copy()
    V_scalar = solver.V.sub(0).collapse()[0]
    scalar_coords = V_scalar.tabulate_dof_coordinates()[:, :2]

    return mesh_coords, cells, h_init, scalar_coords


def select_observation_points(mesh_coords, fraction, seed=42):
    """Replicate the exact observation selection from the twin experiments.

    This matches:
      - parameter_runners.py: _make_observation_points()
      - validation_ladder.py: create_observations()
      - run_comparison.py: _run_sub_experiment_wind()
    """
    # Deduplicate
    _, unique_idx = np.unique(
        np.round(mesh_coords[:, :2], decimals=10), axis=0, return_index=True,
    )
    unique_coords = mesh_coords[unique_idx]

    # Interior filter (exclude boundary)
    x_min, x_max = unique_coords[:, 0].min(), unique_coords[:, 0].max()
    y_min, y_max = unique_coords[:, 1].min(), unique_coords[:, 1].max()
    interior_mask = (
        (unique_coords[:, 0] > x_min + 1e-10) &
        (unique_coords[:, 0] < x_max - 1e-10) &
        (unique_coords[:, 1] > y_min + 1e-10) &
        (unique_coords[:, 1] < y_max - 1e-10)
    )
    interior = unique_coords[interior_mask]
    boundary = unique_coords[~interior_mask]

    # Random sample
    rng = np.random.default_rng(seed)
    n_obs = max(1, int(len(interior) * fraction))
    chosen = rng.choice(len(interior), size=min(n_obs, len(interior)), replace=False)
    obs_points = interior[chosen, :2]

    return obs_points, interior[:, :2], boundary[:, :2]


def plot_domain_with_observations(
    mesh_coords, cells, h_init, scalar_coords,
    obs_configs: list[dict],
    output_path: Path,
):
    """Create publication-quality domain + observation plot.

    Parameters
    ----------
    obs_configs : list of dict
        Each dict has: label, fraction, seed, color, marker
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import Normalize
    from scipy.spatial import cKDTree

    # Convert coordinates to km for readability
    x_offset = mesh_coords[:, 0].min()
    y_offset = mesh_coords[:, 1].min()
    x_km = (mesh_coords[:, 0] - x_offset) / 1000
    y_km = (mesh_coords[:, 1] - y_offset) / 1000

    # Build triangulation from mesh cells
    triangles = []
    for i in range(cells.shape[0]):
        cell_nodes = cells[i]
        triangles.append(cell_nodes[:3])  # First 3 nodes of each cell
    triangles = np.array(triangles)
    triang = mtri.Triangulation(x_km, y_km, triangles)

    # Map bathymetry from scalar DOFs to mesh nodes via nearest neighbor
    tree = cKDTree(scalar_coords)
    _, nearest = tree.query(mesh_coords[:, :2])
    bathy_at_nodes = h_init[nearest]

    n_configs = len(obs_configs)
    if n_configs <= 2:
        fig, axes = plt.subplots(1, n_configs, figsize=(7 * n_configs, 8),
                                  squeeze=False)
        axes = axes[0]
    else:
        ncols = min(3, n_configs)
        nrows = (n_configs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 8 * nrows),
                                  squeeze=False)
        axes = axes.flatten()

    for idx, (ax, cfg) in enumerate(zip(axes, obs_configs)):
        # Bathymetry fill
        levels = np.linspace(-3, 60, 20)
        cf = ax.tricontourf(triang, bathy_at_nodes, levels=levels,
                            cmap="terrain", alpha=0.6, extend="both")

        # Mesh edges (light)
        ax.triplot(triang, color="gray", linewidth=0.15, alpha=0.3)

        # Coastline (zero contour)
        ax.tricontour(triang, bathy_at_nodes, levels=[0], colors="black",
                       linewidths=1.0)

        # Observation points
        obs_pts, interior, boundary = select_observation_points(
            mesh_coords, cfg["fraction"], cfg.get("seed", 42),
        )
        obs_x = (obs_pts[:, 0] - x_offset) / 1000
        obs_y = (obs_pts[:, 1] - y_offset) / 1000

        ax.scatter(obs_x, obs_y,
                   c=cfg.get("color", "red"),
                   marker=cfg.get("marker", "o"),
                   s=cfg.get("size", 12),
                   alpha=0.8,
                   edgecolors="black",
                   linewidths=0.3,
                   zorder=5,
                   label=f'{len(obs_pts)} obs ({cfg["fraction"]*100:.0f}%)')

        ax.set_xlabel("x (km)", fontsize=12)
        ax.set_ylabel("y (km)", fontsize=12)
        ax.set_title(cfg["label"], fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=10)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(cf, cax=cbar_ax)
    cbar.set_label("Bathymetric depth (m)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.subplots_adjust(left=0.06, right=0.90, top=0.93, bottom=0.08,
                        wspace=0.25)
    fig.suptitle("Shinnecock Inlet: Observation Locations", fontsize=15,
                 fontweight="bold", y=0.97)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Shinnecock domain with observation locations")
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.1, 0.3],
                        help="Observation fractions to plot (default: 0.1 0.3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / "domain_observations.png"

    print("Loading Shinnecock mesh and bathymetry...")
    mesh_coords, cells, h_init, scalar_coords = load_mesh_and_bathymetry()
    print(f"  Mesh nodes: {mesh_coords.shape[0]}")
    print(f"  Scalar DOFs: {scalar_coords.shape[0]}")

    # Define experiment configurations
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    obs_configs = []
    for i, frac in enumerate(args.fractions):
        obs_configs.append({
            "label": f"obs_fraction = {frac}",
            "fraction": frac,
            "seed": 42,
            "color": colors[i % len(colors)],
            "marker": "o",
            "size": max(5, 20 - 5 * i),
        })

    print(f"Plotting {len(obs_configs)} observation configurations...")
    plot_domain_with_observations(
        mesh_coords, cells, h_init, scalar_coords,
        obs_configs, output_path,
    )

    # Print summary
    print("\nObservation placement summary:")
    for cfg in obs_configs:
        obs_pts, interior, boundary = select_observation_points(
            mesh_coords, cfg["fraction"], cfg.get("seed", 42),
        )
        print(f"  {cfg['label']}:")
        print(f"    Interior nodes: {len(interior)}")
        print(f"    Boundary nodes: {len(boundary)}")
        print(f"    Obs selected: {len(obs_pts)}")
        print(f"    Seed: {cfg.get('seed', 42)}")

    print(f"\nCode path: experiments/twin_framework/parameter_runners.py: "
          f"_make_observation_points()")
    print(f"Algorithm: interior filter → random sample (seed=42)")


if __name__ == "__main__":
    raise SystemExit(main())

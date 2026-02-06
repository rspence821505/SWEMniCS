#!/usr/bin/env python3
"""
Interactive Observation Point Selection Tool

This script provides an interactive matplotlib interface for selecting
observation points on a mesh. Selected points are saved to a JSON file
that can be used with shinnecock.py, tidal.py, or other DA experiments.

The tool supports both mesh node selection (for CG/SUPG) and DOF location
selection (for DG discretizations).

Usage:
    # For Shinnecock (ADIOS mesh) - mesh nodes (CG/SUPG)
    python scripts/select_observation_points.py --mesh-type shinnecock

    # For Tidal (rectangular mesh) - mesh nodes (CG/SUPG)
    python scripts/select_observation_points.py --mesh-type tidal --nx 40 --ny 10

    # For DG discretization - select at DOF locations
    python scripts/select_observation_points.py --mesh-type tidal --solver-type DG --p-degree 1

    # For Idealized Inlet (XDMF mesh)
    python scripts/select_observation_points.py --mesh-type inlet

    # Custom output file
    python scripts/select_observation_points.py --mesh-type tidal -o my_obs_points.json

Controls:
    - Left click: Select/deselect nearest point
    - Right click: Clear all selections
    - Enter/Return: Save and exit
    - Escape: Exit without saving
    - 'c': Clear all selections
    - 's': Save current selection

Author: SWE4DVar Development Team
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from scipy.spatial import cKDTree

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class ObservationPointSelector:
    """Interactive tool for selecting observation points on a mesh."""

    def __init__(
        self,
        coords,
        triangles=None,
        title="Select Observation Points",
        point_type="node",
    ):
        """
        Initialize the selector.

        Parameters
        ----------
        coords : np.ndarray
            Point coordinates, shape (n_points, 2) or (n_points, 3).
        triangles : np.ndarray, optional
            Triangle connectivity, shape (n_triangles, 3). Used for mesh overlay.
        title : str
            Window title.
        point_type : str
            Type of points: "node" for mesh nodes, "dof" for DOF locations.
        """
        self.coords = coords[:, :2] if coords.shape[1] > 2 else coords
        self.triangles = triangles
        self.title = title
        self.point_type = point_type

        # Build KD-tree for fast nearest neighbor lookup
        self.tree = cKDTree(self.coords)

        # Selected point indices
        self.selected_indices = set()

        # Setup figure with improved styling
        plt.style.use('seaborn-v0_8-whitegrid')
        self.fig, self.ax = plt.subplots(figsize=(14, 9))
        self.fig.canvas.manager.set_window_title(title)
        self.fig.patch.set_facecolor('#f8f9fa')
        self.ax.set_facecolor('#ffffff')

        # Add title with bold styling
        self.ax.set_title(
            title,
            fontsize=16,
            fontweight='bold',
            color='#2c3e50',
            pad=20
        )

        # Plot mesh
        self._plot_mesh()

        # Color palette
        node_color = '#74b9ff' if point_type == "dof" else '#b2bec3'
        selected_color = '#e74c3c'
        selected_edge = '#c0392b'

        # Plot points (all points as small dots)
        self.node_scatter = self.ax.scatter(
            self.coords[:, 0],
            self.coords[:, 1],
            c=node_color,
            s=20 if point_type == "dof" else 15,
            alpha=0.7,
            picker=True,
            pickradius=5,
            label=f'{point_type.upper()}s ({len(self.coords):,})',
            edgecolors='white',
            linewidths=0.3
        )

        # Selected points (highlighted)
        self.selected_scatter = self.ax.scatter(
            [], [],
            c=selected_color,
            s=150,
            marker='o',
            edgecolors=selected_edge,
            linewidths=2.5,
            label='Selected',
            zorder=10
        )

        # Status text with improved styling
        self.status_text = self.ax.text(
            0.02, 0.98, self._get_status_text(),
            transform=self.ax.transAxes,
            verticalalignment='top',
            fontsize=12,
            fontweight='bold',
            color='#2c3e50',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='#ffeaa7',
                edgecolor='#fdcb6e',
                linewidth=2,
                alpha=0.95
            )
        )

        # Instructions with improved styling
        instructions = (
            "⌨ Controls\n"
            "━━━━━━━━━━━━━━━━\n"
            f"  ● Left click: Select/deselect {point_type}\n"
            "  ● Right click: Clear all\n"
            "  ● Enter: Save & exit\n"
            "  ● Esc: Exit without saving\n"
            "  ● 'c': Clear all\n"
            "  ● 's': Save"
        )
        self.ax.text(
            0.98, 0.98, instructions,
            transform=self.ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            fontsize=10,
            fontfamily='monospace',
            color='#2c3e50',
            bbox=dict(
                boxstyle='round,pad=0.6',
                facecolor='#dfe6e9',
                edgecolor='#636e72',
                linewidth=1.5,
                alpha=0.95
            )
        )

        # Styled axis labels
        self.ax.set_xlabel('X Coordinate', fontsize=12, fontweight='bold', color='#2c3e50')
        self.ax.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold', color='#2c3e50')
        self.ax.tick_params(axis='both', labelsize=10, colors='#636e72')

        # Make tick labels bold
        for label in self.ax.get_xticklabels() + self.ax.get_yticklabels():
            label.set_fontweight('bold')

        self.ax.set_aspect('equal')

        # Enhanced legend
        legend = self.ax.legend(
            loc='lower right',
            fontsize=11,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95,
            edgecolor='#636e72'
        )
        legend.get_frame().set_facecolor('#ffffff')
        for text in legend.get_texts():
            text.set_fontweight('bold')
            text.set_color('#2c3e50')

        # Add subtle border to plot area
        for spine in self.ax.spines.values():
            spine.set_color('#636e72')
            spine.set_linewidth(1.5)

        plt.tight_layout()

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Result
        self.saved = False

    def _plot_mesh(self):
        """Plot mesh triangles if available."""
        if self.triangles is not None:
            # For DOF mode, we need mesh node coordinates for triplot
            # The triangles reference mesh nodes, not DOFs
            pass  # Mesh overlay handled separately for DOF mode

    def _get_status_text(self):
        """Get status text string."""
        count = len(self.selected_indices)
        return f"✓ Selected: {count:,} {self.point_type}{'s' if count != 1 else ''}"

    def _update_display(self):
        """Update the display after selection changes."""
        if len(self.selected_indices) > 0:
            selected_coords = self.coords[list(self.selected_indices)]
            self.selected_scatter.set_offsets(selected_coords)
        else:
            self.selected_scatter.set_offsets(np.empty((0, 2)))

        self.status_text.set_text(self._get_status_text())
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return

        # Check if toolbar is in zoom or pan mode - skip selection if so
        toolbar = self.fig.canvas.toolbar
        if toolbar is not None and toolbar.mode in ('zoom rect', 'pan/zoom'):
            return

        if event.button == 1:  # Left click - select/deselect
            # Find nearest point
            dist, idx = self.tree.query([event.xdata, event.ydata])

            # Compute a reasonable selection radius based on mesh spacing
            # Use ~10x the average nearest neighbor distance as threshold (generous for easy clicking)
            if not hasattr(self, '_selection_radius'):
                # Compute once and cache
                sample_size = min(100, len(self.coords))
                sample_dists, _ = self.tree.query(self.coords[:sample_size], k=2)
                avg_spacing = np.mean(sample_dists[:, 1])  # Distance to nearest neighbor
                self._selection_radius = avg_spacing * 10.0

            # Only select if click is close enough to a point
            if dist > self._selection_radius:
                return  # Click too far from any point

            # Toggle selection
            if idx in self.selected_indices:
                self.selected_indices.remove(idx)
                print(f"  ✗ Deselected {self.point_type} {idx}")
            else:
                self.selected_indices.add(idx)
                print(f"  ✓ Selected {self.point_type} {idx} at ({self.coords[idx, 0]:.2f}, {self.coords[idx, 1]:.2f})")

            self._update_display()

        elif event.button == 3:  # Right click - clear all
            self.selected_indices.clear()
            print("  ↻ Cleared all selections")
            self._update_display()

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'enter' or event.key == 'return':
            self.saved = True
            plt.close(self.fig)
        elif event.key == 'escape':
            self.saved = False
            plt.close(self.fig)
        elif event.key == 'c':
            self.selected_indices.clear()
            print("  ↻ Cleared all selections")
            self._update_display()
        elif event.key == 's':
            self.saved = True
            print(f"  ✓ Selection saved: {len(self.selected_indices):,} {self.point_type}s")

    def run(self):
        """Run the interactive selector."""
        plt.show()
        return self.saved

    def get_selected_points(self):
        """
        Get the selected observation points.

        Returns
        -------
        dict
            Dictionary with selected point information.
        """
        indices = sorted(list(self.selected_indices))
        coords = self.coords[indices].tolist() if len(indices) > 0 else []

        # Add z-coordinate as 0 for 3D compatibility
        coords_3d = [[c[0], c[1], 0.0] for c in coords]

        return {
            "n_points": len(indices),
            "point_type": self.point_type,
            "point_indices": indices,
            # Keep node_indices for backward compatibility
            "node_indices": indices,
            "coordinates": coords_3d,
        }


def get_dof_coordinates(mesh, solver_type, p_degree):
    """
    Get DOF coordinates for a given discretization.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh.
    solver_type : str
        Solver type: CG, SUPG, DG, DGNC, DGCG.
    p_degree : int
        Polynomial degree.

    Returns
    -------
    np.ndarray
        DOF coordinates, shape (n_dofs, 2) or (n_dofs, 3).
    """
    from dolfinx import fem
    import basix

    # Determine element type based on solver
    if solver_type.upper() in ["DG", "DGNC", "DGCG"]:
        # Discontinuous Galerkin
        element = basix.ufl.element(
            "DG", mesh.topology.cell_name(), p_degree, shape=(3,)
        )
    else:
        # Continuous Galerkin (CG, SUPG)
        element = basix.ufl.element(
            "Lagrange", mesh.topology.cell_name(), p_degree, shape=(3,)
        )

    V = fem.functionspace(mesh, element)

    # Get DOF coordinates
    # For mixed spaces, we get coordinates for each component
    dof_coords = V.tabulate_dof_coordinates()

    # Remove duplicate coordinates (DOFs at same location for different components)
    # For a (3,) vector space, every 3rd DOF is at the same location
    n_components = 3
    unique_coords = dof_coords[::n_components]

    return unique_coords


def load_shinnecock_mesh(adios_path="data/shinnecock_inlet", solver_type=None, p_degree=1):
    """Load Shinnecock mesh from ADIOS files."""
    try:
        print(f"  Loading Shinnecock mesh from {adios_path}...", flush=True)
        print(f"    → Importing adios4dolfinx...", flush=True)
        import adios4dolfinx
        print(f"    → Importing dolfinx...", flush=True)
        from dolfinx import mesh as dmesh
        from mpi4py import MPI

        mesh_file = f"{adios_path}_mesh.bp"
        print(f"    → Reading mesh file: {mesh_file}", flush=True)

        # Try BP4 engine first (more stable), then BP5
        # Note: adios4dolfinx API: filename first, then comm; use legacy=True for older files
        engines = ["BP4", "BP5"]
        mesh = None
        for engine in engines:
            try:
                print(f"    → Trying {engine} engine...", flush=True)
                mesh = adios4dolfinx.read_mesh(
                    mesh_file,
                    MPI.COMM_SELF,
                    engine=engine,
                    ghost_mode=dmesh.GhostMode.none,
                    legacy=True,
                )
                print(f"    → Success with {engine} engine!", flush=True)
                break
            except Exception as e:
                print(f"    → {engine} failed: {e}", flush=True)
                continue

        if mesh is None:
            raise RuntimeError(f"Failed to read mesh with any engine: {engines}")
        print(f"    → Mesh loaded, extracting topology...", flush=True)

        triangles = mesh.topology.connectivity(2, 0).array.reshape(-1, 3)

        # Get coordinates based on solver type
        if solver_type and solver_type.upper() in ["DG", "DGNC", "DGCG"]:
            print(f"    → Using DOF coordinates for {solver_type} (p={p_degree})...")
            coords = get_dof_coordinates(mesh, solver_type, p_degree)
            point_type = "dof"
            print(f"    → Loaded {len(coords):,} DOFs, {len(triangles):,} triangles")
        else:
            coords = mesh.geometry.x
            point_type = "node"
            print(f"    → Loaded {len(coords):,} nodes, {len(triangles):,} triangles")

        return coords, triangles, "shinnecock", point_type, mesh

    except Exception as e:
        print(f"Error loading Shinnecock mesh: {e}")
        print("Make sure ADIOS files exist in data/shinnecock_inlet_mesh.bp")
        sys.exit(1)


def load_inlet_mesh(xdmf_path="data/Ideal_Inlet/Ideal_Inlet.xdmf", solver_type=None, p_degree=1):
    """Load Idealized Inlet mesh from XDMF file."""
    try:
        from dolfinx import io, mesh as dmesh
        from mpi4py import MPI

        print(f"  Loading Inlet mesh from {xdmf_path}...")

        with io.XDMFFile(MPI.COMM_SELF, xdmf_path, "r") as xdmf:
            mesh = xdmf.read_mesh(ghost_mode=dmesh.GhostMode.none)

        triangles = mesh.topology.connectivity(2, 0).array.reshape(-1, 3)

        # Get coordinates based on solver type
        if solver_type and solver_type.upper() in ["DG", "DGNC", "DGCG"]:
            print(f"    → Using DOF coordinates for {solver_type} (p={p_degree})...")
            coords = get_dof_coordinates(mesh, solver_type, p_degree)
            point_type = "dof"
            print(f"    → Loaded {len(coords):,} DOFs, {len(triangles):,} triangles")
        else:
            coords = mesh.geometry.x
            point_type = "node"
            print(f"    → Loaded {len(coords):,} nodes, {len(triangles):,} triangles")

        return coords, triangles, "inlet", point_type, mesh

    except Exception as e:
        print(f"Error loading Inlet mesh: {e}")
        print(f"Make sure XDMF file exists at {xdmf_path}")
        sys.exit(1)


def create_tidal_mesh(nx=20, ny=5, x0=0.0, x1=10000.0, y0=0.0, y1=5000.0,
                      solver_type=None, p_degree=1):
    """Create a rectangular tidal mesh."""
    print(f"  Creating tidal mesh: {nx}x{ny} elements...")

    # Create DOLFINx mesh for DOF extraction
    from dolfinx import mesh as dmesh
    from mpi4py import MPI

    mesh = dmesh.create_rectangle(
        MPI.COMM_SELF,
        [np.array([x0, y0]), np.array([x1, y1])],
        [nx, ny],
        cell_type=dmesh.CellType.triangle,
        diagonal=dmesh.DiagonalType.crossed,
    )

    triangles = mesh.topology.connectivity(2, 0).array.reshape(-1, 3)

    # Get coordinates based on solver type
    if solver_type and solver_type.upper() in ["DG", "DGNC", "DGCG"]:
        print(f"    → Using DOF coordinates for {solver_type} (p={p_degree})...")
        coords = get_dof_coordinates(mesh, solver_type, p_degree)
        point_type = "dof"
        print(f"    → Created {len(coords):,} DOFs, {len(triangles):,} triangles")
    else:
        coords = mesh.geometry.x
        point_type = "node"
        print(f"    → Created {len(coords):,} nodes, {len(triangles):,} triangles")

    return coords, triangles, "tidal", point_type, mesh


def save_observation_points(output_file, mesh_type, points_data, mesh_params=None):
    """Save selected observation points to JSON file."""
    data = {
        "mesh_type": mesh_type,
        "mesh_params": mesh_params or {},
        **points_data
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n  ✓ Saved {points_data['n_points']:,} observation points to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive observation point selection tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--mesh-type",
        choices=["tidal", "shinnecock", "inlet"],
        default="tidal",
        help="Type of mesh to load (default: tidal)"
    )

    # Solver type for DOF-based selection
    parser.add_argument(
        "--solver-type",
        choices=["CG", "SUPG", "DG", "DGNC", "DGCG"],
        default=None,
        help="Solver type for DOF-based selection. If not specified, uses mesh nodes. "
             "For DG/DGNC/DGCG, shows DOF locations instead of mesh nodes."
    )
    parser.add_argument(
        "--p-degree",
        type=int,
        default=1,
        help="Polynomial degree for DOF-based selection (default: 1)"
    )

    # Tidal mesh parameters
    parser.add_argument("--nx", type=int, default=20, help="Tidal mesh: elements in x")
    parser.add_argument("--ny", type=int, default=5, help="Tidal mesh: elements in y")
    parser.add_argument("--x0", type=float, default=0.0, help="Tidal mesh: x min")
    parser.add_argument("--x1", type=float, default=10000.0, help="Tidal mesh: x max")
    parser.add_argument("--y0", type=float, default=0.0, help="Tidal mesh: y min")
    parser.add_argument("--y1", type=float, default=5000.0, help="Tidal mesh: y max")

    # ADIOS/XDMF paths
    parser.add_argument(
        "--adios-file",
        default="data/shinnecock_inlet",
        help="Path to Shinnecock ADIOS files (without extension)"
    )
    parser.add_argument(
        "--xdmf-file",
        default="data/Ideal_Inlet/Ideal_Inlet.xdmf",
        help="Path to Inlet XDMF file"
    )

    # Output
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON file (default: data/<mesh_type>_obs_points.json)"
    )

    args = parser.parse_args()

    # Load mesh based on type
    mesh_params = {}
    if args.mesh_type == "tidal":
        mesh_params = {
            "nx": args.nx, "ny": args.ny,
            "x0": args.x0, "x1": args.x1,
            "y0": args.y0, "y1": args.y1,
            "solver_type": args.solver_type,
            "p_degree": args.p_degree,
        }
        coords, triangles, mesh_name, point_type, mesh = create_tidal_mesh(
            nx=args.nx, ny=args.ny, x0=args.x0, x1=args.x1, y0=args.y0, y1=args.y1,
            solver_type=args.solver_type, p_degree=args.p_degree
        )
    elif args.mesh_type == "shinnecock":
        mesh_params = {
            "adios_file": args.adios_file,
            "solver_type": args.solver_type,
            "p_degree": args.p_degree,
        }
        coords, triangles, mesh_name, point_type, mesh = load_shinnecock_mesh(
            args.adios_file, solver_type=args.solver_type, p_degree=args.p_degree
        )
    elif args.mesh_type == "inlet":
        mesh_params = {
            "xdmf_file": args.xdmf_file,
            "solver_type": args.solver_type,
            "p_degree": args.p_degree,
        }
        coords, triangles, mesh_name, point_type, mesh = load_inlet_mesh(
            args.xdmf_file, solver_type=args.solver_type, p_degree=args.p_degree
        )

    # Default output file - include solver type if using DOF mode
    if args.output is None:
        if args.solver_type and args.solver_type.upper() in ["DG", "DGNC", "DGCG"]:
            output_file = f"data/{mesh_name}_{args.solver_type.lower()}_p{args.p_degree}_obs_points.json"
        else:
            output_file = f"data/{mesh_name}_obs_points.json"
    else:
        output_file = args.output

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Run interactive selector
    print("\n" + "━" * 60)
    print("  ✦ Interactive Observation Point Selection ✦")
    print("━" * 60)
    if point_type == "dof":
        print(f"  Mode:   DOF selection ({args.solver_type}, p={args.p_degree})")
        print(f"  Mesh:   {mesh_name} ({len(coords):,} DOFs)")
    else:
        print(f"  Mode:   Mesh node selection (CG/SUPG compatible)")
        print(f"  Mesh:   {mesh_name} ({len(coords):,} nodes)")
    print(f"  Output: {output_file}")
    print("━" * 60)

    # For DOF mode, overlay mesh using geometry coordinates
    mesh_coords_for_overlay = None
    if point_type == "dof":
        mesh_coords_for_overlay = mesh.geometry.x

    selector = ObservationPointSelector(
        coords,
        triangles=None,  # Don't overlay mesh in DOF mode (coords don't match triangles)
        title=f"Select Observation Points - {mesh_name.title()} ({point_type.upper()}s)",
        point_type=point_type,
    )

    # Add mesh overlay for DOF mode with improved styling
    if point_type == "dof" and mesh_coords_for_overlay is not None:
        mesh_x = mesh_coords_for_overlay[:, 0]
        mesh_y = mesh_coords_for_overlay[:, 1]
        selector.ax.triplot(
            mesh_x, mesh_y, triangles,
            color='#95a5a6',
            linewidth=0.4,
            alpha=0.4
        )

    saved = selector.run()

    if saved and len(selector.selected_indices) > 0:
        points_data = selector.get_selected_points()
        save_observation_points(output_file, mesh_name, points_data, mesh_params)
        print("\n  ✦ Done! Use this file with:")
        print(f"    python examples/shinnecock.py --da-mode 4dvar --obs-points-file {output_file}")
        print(f"    python experiments/serial_da/tidal_4dvar.py --obs-points-file {output_file}")
        print()
    elif saved:
        print("\n  ⚠ No points selected. File not saved.\n")
    else:
        print("\n  ✗ Exited without saving.\n")


if __name__ == "__main__":
    main()

"""Visualization utilities for solver results.

This module provides classes and functions for creating videos and plots
of shallow water equation solutions, including water surface elevation,
velocity fields, and bathymetry.
"""

from pathlib import Path
from dolfinx import fem as fe, io
from typing import Optional
import sys

try:
    import pyvista
    import dolfinx.plot
    import numpy as np

    HAVE_PYVISTA = True
except ImportError:
    HAVE_PYVISTA = False


class SolverVisualizer:
    """Manages visualization output for solver results.

    This class handles creation of time-series visualizations using VTX format
    for efficient parallel I/O, and optional interactive plotting with PyVista.
    """

    def __init__(
        self,
        domain,
        V_scalar,
        V_vel,
        problem,
        verbose: bool = False,
    ):
        """Initialize the visualizer.

        Args:
            domain: The mesh domain
            V_scalar: Scalar function space for depth/elevation
            V_vel: Vector function space for velocity
            problem: Problem object containing bathymetry
            verbose: Enable verbose logging
        """
        self.domain = domain
        self.V_scalar = V_scalar
        self.V_vel = V_vel
        self.problem = problem
        self.verbose = verbose

        # Writers (initialized by initialize_video)
        self.wse_writer: Optional[io.VTXWriter] = None
        self.h_writer: Optional[io.VTXWriter] = None
        self.vel_writer: Optional[io.VTXWriter] = None
        self.bathy_writer: Optional[io.VTXWriter] = None

        # Plot functions
        self.eta_plot: Optional[fe.Function] = None
        self.h_plot: Optional[fe.Function] = None
        self.vel_plot: Optional[fe.Function] = None
        self.bathy_plot: Optional[fe.Function] = None

        self.initialized = False

    def initialize_video(self, filename: str):
        """Initialize video writers for output.

        Creates output directory and initializes VTX writers for water surface
        elevation, depth, velocity, and bathymetry.

        Args:
            filename: Base path for output files (directory will be created)
        """
        # Create plot functions
        self.eta_plot = fe.Function(self.V_scalar)
        self.eta_plot.name = "eta"

        self.h_plot = fe.Function(self.V_scalar)
        self.h_plot.name = "depth"

        self.vel_plot = fe.Function(self.V_vel)
        self.vel_plot.name = "depth averaged velocity"

        self.bathy_plot = fe.Function(self.V_scalar)
        self.bathy_plot.name = "bathymetry"

        # Create output directory
        results_folder = Path(filename)
        results_folder.mkdir(exist_ok=True, parents=True)

        # Initialize writers
        self.wse_writer = io.VTXWriter(
            self.domain.comm,
            results_folder / "WSE.bp",
            self.eta_plot,
            engine="BP4",
        )

        self.h_writer = io.VTXWriter(
            self.domain.comm, results_folder / "h.bp", self.h_plot, engine="BP4"
        )

        self.vel_writer = io.VTXWriter(
            self.domain.comm,
            results_folder / "vel.bp",
            self.vel_plot,
            engine="BP4",
        )

        self.bathy_writer = io.VTXWriter(
            self.domain.comm,
            results_folder / "bathy.bp",
            self.bathy_plot,
            engine="BP4",
        )

        self.initialized = True

    def plot_frame(self, u, t: float):
        """Write a frame of the solution to video files.

        Args:
            u: Solution function containing [eta, u, v]
            t: Current time value
        """
        if not self.initialized:
            raise RuntimeError("Must call initialize_video() before plot_frame()")

        # Extract and write water surface elevation
        self.eta_expr = fe.Expression(
            u.sub(0).collapse() - self.problem.h_b,
            self.V_scalar.element.interpolation_points(),
        )
        self.eta_plot.interpolate(self.eta_expr)

        # Extract and write velocity
        self.v_expr = fe.Expression(
            u.sub(1).collapse(), self.V_vel.element.interpolation_points()
        )
        self.vel_plot.interpolate(self.v_expr)

        # Write depth
        self.h_plot.interpolate(u.sub(0).collapse())

        # Write current frame
        self.wse_writer.write(t)
        self.h_writer.write(t)
        self.vel_writer.write(t)

        # Write bathymetry only on first timestep
        if t == 0 or not hasattr(self, "_bathy_written"):
            if self.verbose:
                print("Writing bathymetry")
            self.bathy_plot.interpolate(
                fe.Expression(
                    self.problem.h_b, self.V_scalar.element.interpolation_points()
                )
            )
            self.bathy_writer.write(t)
            self._bathy_written = True

    def finalize_video(self):
        """Close all video writers and finalize output files."""
        if not self.initialized:
            return

        if self.wse_writer is not None:
            self.wse_writer.close()
        if self.h_writer is not None:
            self.h_writer.close()
        if self.vel_writer is not None:
            self.vel_writer.close()
        if self.bathy_writer is not None:
            self.bathy_writer.close()

    def plot_func_interactive(self, func, name: str = "eta"):
        """Create interactive plot of a function using PyVista.

        Args:
            func: Function to plot
            name: Name for the scalar field

        Raises:
            ValueError: If PyVista is not installed
        """
        if not HAVE_PYVISTA:
            raise ValueError("PyVista not installed! Cannot create interactive plots.")

        num_cells = self.domain.topology.index_map(self.domain.topology.dim).size_local
        cell_entities = np.arange(num_cells, dtype=np.int32)

        args = dolfinx.plot.create_vtk_mesh(
            self.domain, self.domain.topology.dim, cell_entities
        )
        grid = pyvista.UnstructuredGrid(*args)

        # Map cells to points
        from dolfinx import cpp

        cell_geom_entities = cpp.mesh.entities_to_geometry(
            self.domain, 2, cell_entities, False
        )
        point_cells = np.full(len(args[-1]), 0)
        for i, p in enumerate(cell_geom_entities):
            point_cells[p] = i

        # Evaluate function
        data = func.eval(self.domain.geometry.x, point_cells)
        print(f"Data range: [{data.min()}, {data.max()}]")
        print(f"Min location: {np.argmin(data)}")

        grid.point_data[name] = data
        grid.set_active_scalars(name)

        # Find and highlight minimum point
        bad_point = self.domain.geometry.x[np.argmin(data)]

        # Create plotter
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_scalar_bar=True, show_edges=True)
        plotter.add_points(bad_point[None, :], point_size=10.0, color="red")
        plotter.view_xy()
        plotter.set_focus(bad_point)
        print(f"Focus point: {bad_point}")
        plotter.show()

    def __repr__(self) -> str:
        status = "initialized" if self.initialized else "not initialized"
        return f"SolverVisualizer({status})"

    def __del__(self):
        """Destructor to ensure writers are closed."""
        if self.initialized:
            try:
                self.finalize_video()
            except:
                pass

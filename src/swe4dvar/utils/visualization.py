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

try:
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False


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

        # Get MPI rank for automatic rank checking
        self.rank = domain.comm.rank

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

    def _should_plot(self) -> bool:
        """Return True only on rank 0 for MPI-safe plotting.

        Returns:
            True if this is rank 0, False otherwise
        """
        return self.rank == 0

    def print_saved_files(self, *messages):
        """Print messages only on rank 0.

        This is a convenience method for printing informational messages
        about saved files in an MPI-aware way.

        Args:
            *messages: Messages to print (each message on a new line)
        """
        if self._should_plot():
            for msg in messages:
                print(msg)

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

    def plot_timeseries(
        self,
        solver_vals,
        dt: float,
        station_idx: int = 0,
        component: int = 0,
        component_name: str = "h",
        title: str = None,
        xlabel: str = "t (days)",
        ylabel: str = None,
        filename: str = None,
        reference_data: Optional[dict] = None,
        time_units: str = "days",
    ):
        """Plot time series at a station.

        This method is MPI-aware and only creates plots on rank 0.

        Args:
            solver_vals: Array of shape (nt+1, n_stations, 3) containing h, u, v
            dt: Time step size in seconds
            station_idx: Index of station to plot (default: 0)
            component: Component index (0=h, 1=u, 2=v)
            component_name: Name of component for legend
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            filename: Output filename (if None, displays interactively)
            reference_data: Optional dict with 'time' and 'values' arrays for comparison
            time_units: Units for time axis ('seconds', 'days', 'hours')
        """
        # Only plot on rank 0
        if not self._should_plot():
            return

        if not HAVE_MATPLOTLIB:
            raise ValueError("Matplotlib not installed! Cannot create plots.")

        nt = solver_vals.shape[0] - 1
        t_f = nt * dt

        # Convert time to requested units
        if time_units == "days":
            time_array = np.linspace(0, t_f / (60 * 60 * 24), nt + 1)
        elif time_units == "hours":
            time_array = np.linspace(0, t_f / (60 * 60), nt + 1)
        else:  # seconds
            time_array = np.linspace(0, t_f, nt + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(
            time_array,
            solver_vals[:, station_idx, component],
            "k-",
            linewidth=2,
            label=f"{component_name} (solver)",
        )

        # Add reference data if provided
        if reference_data is not None:
            plt.plot(
                reference_data["time"],
                reference_data["values"],
                "b--",
                linewidth=2,
                label=reference_data.get("label", "Reference"),
            )

        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel if ylabel else component_name)
        if title:
            plt.title(title)
        plt.legend()

        if filename:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_spatial_profile(
        self,
        solver_vals,
        x_coords,
        dt: float,
        timesteps: list,
        component: int = 0,
        component_name: str = "h",
        title: str = None,
        xlabel: str = "x (m)",
        ylabel: str = None,
        filename: str = None,
        offset: float = 0.0,
        analytical_solution_func=None,
    ):
        """Plot spatial profile at multiple timesteps.

        This method is MPI-aware and only creates plots on rank 0.

        Args:
            solver_vals: Array of shape (nt+1, n_stations, 3) containing h, u, v
            x_coords: X-coordinates of stations
            dt: Time step size in seconds
            timesteps: List of timestep indices to plot
            component: Component index (0=h, 1=u, 2=v)
            component_name: Name of component for legend
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            filename: Output filename (if None, displays interactively)
            offset: Offset to add to component values (e.g., for bathymetry)
            analytical_solution_func: Optional function(x, t) returning (h_analytic, u_analytic)
        """
        # Only plot on rank 0
        if not self._should_plot():
            return

        if not HAVE_MATPLOTLIB:
            raise ValueError("Matplotlib not installed! Cannot create plots.")

        plt.figure(figsize=(10, 6))

        for timestep in timesteps:
            if timestep >= solver_vals.shape[0]:
                continue

            t = timestep * dt

            # Plot analytical solution first (if available)
            if analytical_solution_func is not None and timestep != 0:
                h_analytic, u_analytic = analytical_solution_func(x_coords, t)
                analytic_values = h_analytic if component == 0 else u_analytic
                plt.plot(
                    x_coords,
                    analytic_values,
                    "-",
                    linewidth=1,
                    label=f"{component_name} exact at {t:.1f}s",
                )

            # Plot solver results
            plt.plot(
                x_coords,
                solver_vals[timestep, :, component] + offset,
                "--",
                linewidth=2,
                label=f"{component_name} at {t:.1f}s",
            )

        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel if ylabel else component_name)
        if title:
            plt.title(title)
        plt.legend()

        if filename:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_tidal_timeseries(
        self,
        solver_vals,
        dt: float,
        nt: int,
        station_idx: int = 0,
        scheme_name: str = "SUPG",
        output_dir: str = ".",
        plot_elevation: bool = True,
    ):
        """Plot tidal height, velocities, and elevation time series (tidal.py example).

        This method is MPI-aware and only creates plots on rank 0.

        Args:
            solver_vals: Array of shape (nt+1, n_stations, 3) containing h, u, v
            dt: Time step size in seconds
            nt: Number of timesteps
            station_idx: Index of station to plot
            scheme_name: Name of solver scheme for plot title
            output_dir: Directory for output files
            plot_elevation: If True, also plot water surface elevation (η = h + bathymetry)
        """
        # Only plot on rank 0
        if not self._should_plot():
            return

        if not HAVE_MATPLOTLIB:
            raise ValueError("Matplotlib not installed! Cannot create plots.")

        t_f = nt * dt

        # Plot water height (h)
        self.plot_timeseries(
            solver_vals=solver_vals,
            dt=dt,
            station_idx=station_idx,
            component=0,
            component_name="h",
            title=f"Tidal Water Depth for {scheme_name} Scheme",
            xlabel="t (days)",
            ylabel="Water depth (m)",
            filename=f"{output_dir}/tidal_height_{scheme_name}.png",
            time_units="days",
        )

        # Plot water surface elevation (η = h + bathymetry) if requested
        if plot_elevation and self.problem is not None:
            import numpy as np

            # Get bathymetry value at the station
            # For simple problems with constant bathymetry, evaluate at origin
            try:
                if hasattr(self.problem.h_b, 'x'):
                    # h_b is a Function
                    bathy_val = float(self.problem.h_b.x.array[0])
                elif callable(self.problem.h_b):
                    # h_b is a lambda/function - evaluate at origin
                    bathy_val = float(self.problem.h_b(np.zeros((1, 3)))[0])
                else:
                    # h_b is a constant
                    bathy_val = float(self.problem.h_b)
            except:
                # If we can't determine bathymetry, skip elevation plot
                bathy_val = None

            if bathy_val is not None:
                # Create modified solver_vals with elevation instead of depth
                elevation_vals = solver_vals.copy()
                elevation_vals[:, :, 0] = solver_vals[:, :, 0] + bathy_val

                self.plot_timeseries(
                    solver_vals=elevation_vals,
                    dt=dt,
                    station_idx=station_idx,
                    component=0,
                    component_name="η",
                    title=f"Tidal Water Surface Elevation for {scheme_name} Scheme",
                    xlabel="t (days)",
                    ylabel="Water surface elevation (m)",
                    filename=f"{output_dir}/tidal_elevation_{scheme_name}.png",
                    time_units="days",
                )

        # Plot x-velocity (u)
        self.plot_timeseries(
            solver_vals=solver_vals,
            dt=dt,
            station_idx=station_idx,
            component=1,
            component_name="u",
            title=f"Tidal X-Velocity for {scheme_name} Scheme",
            xlabel="t (days)",
            ylabel="X-velocity (m/s)",
            filename=f"{output_dir}/tidal_velocity_u_{scheme_name}.png",
            time_units="days",
        )

        # Plot y-velocity (v)
        self.plot_timeseries(
            solver_vals=solver_vals,
            dt=dt,
            station_idx=station_idx,
            component=2,
            component_name="v",
            title=f"Tidal Y-Velocity for {scheme_name} Scheme",
            xlabel="t (days)",
            ylabel="Y-velocity (m/s)",
            filename=f"{output_dir}/tidal_velocity_v_{scheme_name}.png",
            time_units="days",
        )

    def plot_dam_break(
        self,
        solver_vals,
        dt: float,
        nt: int,
        Lx: float,
        dam_height: float,
        timesteps: list,
        scheme_name: str = "SUPG",
        output_dir: str = ".",
        analytical_solution_func=None,
    ):
        """Plot dam break surface elevation and velocity profiles (dam_break.py example).

        This method is MPI-aware and only creates plots on rank 0.

        Args:
            solver_vals: Array of shape (nt+1, n_stations, 3) containing h, u, v
            dt: Time step size in seconds
            nt: Number of timesteps
            Lx: Domain length in x-direction
            dam_height: Dam height for offset
            timesteps: List of timestep indices to plot
            scheme_name: Name of solver scheme for plot title
            output_dir: Directory for output files
            analytical_solution_func: Optional function(x, t) returning (h_analytic, u_analytic)
        """
        # Only plot on rank 0
        if not self._should_plot():
            return

        if not HAVE_MATPLOTLIB:
            raise ValueError("Matplotlib not installed! Cannot create plots.")

        nx = solver_vals.shape[1]
        x = np.linspace(0, Lx, nx)

        # Plot surface elevation
        self.plot_spatial_profile(
            solver_vals=solver_vals,
            x_coords=x,
            dt=dt,
            timesteps=timesteps,
            component=0,
            component_name="h",
            title=f"Surface Elevation for {scheme_name} Scheme",
            xlabel="x (m)",
            ylabel="Surface elevation (m)",
            filename=f"{output_dir}/dam_height_{scheme_name}_order1_dt.png",
            offset=dam_height,
            analytical_solution_func=analytical_solution_func,
        )

        # Plot x-velocity
        self.plot_spatial_profile(
            solver_vals=solver_vals,
            x_coords=x,
            dt=dt,
            timesteps=timesteps,
            component=1,
            component_name="$u_x$",
            title=f"Velocity for {scheme_name} Scheme",
            xlabel="x (m)",
            ylabel="Velocity in x direction (m/s)",
            filename=f"{output_dir}/dam_velocity_{scheme_name}_order1_dt.png",
            offset=0.0,
            analytical_solution_func=analytical_solution_func,
        )

    def plot_inlet_comparison(
        self,
        solver_vals,
        dt: float,
        nt: int,
        station_idx: int = 0,
        adcirc_file: Optional[str] = None,
        dgswem_file: Optional[str] = None,
        scheme_name: str = "DG",
        output_dir: str = ".",
    ):
        """Plot idealized inlet results with ADCIRC/DGSWEM comparison (idealized_inlet.py example).

        This method is MPI-aware and only creates plots on rank 0.

        Args:
            solver_vals: Array of shape (nt+1, n_stations, 3) containing h, u, v
            dt: Time step size in seconds
            nt: Number of timesteps
            station_idx: Index of station to plot
            adcirc_file: Path to ADCIRC CSV file (columns: time, h, u, v)
            dgswem_file: Path to DGSWEM CSV file
            scheme_name: Name of solver scheme for plot title
            output_dir: Directory for output files
        """
        # Only plot on rank 0
        if not self._should_plot():
            return

        if not HAVE_MATPLOTLIB:
            raise ValueError("Matplotlib not installed! Cannot create plots.")

        t_f = nt * dt
        time_array = np.linspace(0, t_f, nt + 1)

        # Plot height comparison
        plt.figure(figsize=(10, 6))
        plt.plot(
            time_array,
            solver_vals[:nt + 1, station_idx, 0],
            "k-",
            linewidth=2,
            label=f"{scheme_name} solver",
        )

        if adcirc_file is not None:
            try:
                adcirc_dat = np.loadtxt(adcirc_file, delimiter=",", dtype=float)
                plt.plot(
                    adcirc_dat[:, 0],
                    adcirc_dat[:, 1],
                    "b--",
                    linewidth=2,
                    label="ADCIRC",
                )
            except FileNotFoundError:
                if self.verbose:
                    print(f"Warning: ADCIRC file not found: {adcirc_file}")

        if dgswem_file is not None:
            try:
                dgswem_dat = np.loadtxt(dgswem_file, delimiter=",", dtype=float)
                plt.plot(
                    dgswem_dat[:, 0],
                    dgswem_dat[:, 1],
                    "r--",
                    linewidth=2,
                    label="DGSWEM",
                )
            except FileNotFoundError:
                if self.verbose:
                    print(f"Warning: DGSWEM file not found: {dgswem_file}")

        plt.grid(True)
        plt.xlabel("t (s)")
        plt.title("Height for Ideal Inlet Case")
        plt.legend()
        plt.savefig(f"{output_dir}/inlet_height_{scheme_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot y-velocity comparison
        plt.figure(figsize=(10, 6))
        plt.plot(
            time_array,
            solver_vals[:nt + 1, station_idx, 2],
            "k-",
            linewidth=2,
            label=f"{scheme_name} solver",
        )

        if adcirc_file is not None:
            try:
                adcirc_dat = np.loadtxt(adcirc_file, delimiter=",", dtype=float)
                plt.plot(
                    adcirc_dat[:, 0],
                    adcirc_dat[:, 3],
                    "b--",
                    linewidth=2,
                    label="ADCIRC",
                )
            except FileNotFoundError:
                pass

        plt.grid(True)
        plt.xlabel("t (s)")
        plt.title("Velocity in y for Ideal Inlet Case")
        plt.legend()
        plt.savefig(f"{output_dir}/inlet_velocity_{scheme_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

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

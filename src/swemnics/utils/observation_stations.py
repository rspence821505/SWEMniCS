"""Observation station management for spatial monitoring and data collection.

This module provides utilities for managing observation stations, including
point location, data recording, wetting/drying detection, and parallel
gathering of station data.
"""

import numpy as np
from dolfinx import fem as fe, geometry, mesh
from mpi4py import MPI
from typing import Tuple, List, Optional


class StationManager:
    """Manages observation stations for recording time series data.

    This class handles the geometric location of observation points within
    the mesh, evaluates solution fields at these points, and provides
    utilities for wetting/drying detection and parallel data gathering.
    """

    def __init__(self, domain: mesh.Mesh, V_scalar, h_b, verbose: bool = False):
        """Initialize the station manager.

        Args:
            domain: The mesh domain
            V_scalar: Scalar function space for evaluations
            h_b: Bathymetry expression/function
            verbose: Enable verbose logging
        """
        self.domain = domain
        self.V_scalar = V_scalar
        self.h_b = h_b
        self.verbose = verbose

        # Station location data (populated by init_stations)
        self.cells: List[int] = []
        self.station_index: List[int] = []
        self.station_bathy: np.ndarray = np.array([], dtype=float)
        self.points_on_proc: np.ndarray = np.array([], dtype=float)

    def init_stations(self, points: List) -> np.ndarray:
        """Initialize recording stations and identify points on each processor.

        This method performs geometric searches to identify which cells contain
        each observation point, handling the distributed mesh across MPI ranks.

        Args:
            points: List of point coordinates [[x1,y1], [x2,y2], ...] or
                [[x1,y1,z1], [x2,y2,z2], ...]

        Returns:
            Array of points located on this processor
        """
        # Convert points to numpy array
        points = np.asarray(points, dtype=self.domain.geometry.x.dtype)

        # DOLFINx geometry functions always expect 3D points, even for 2D meshes
        # Reshape to (n_points, 3), padding with zeros if necessary
        if points.size == 0:
            points = points.reshape(-1, 3)
        else:
            if points.ndim == 1:
                points = points.reshape(1, -1)

            n_points = points.shape[0]
            if points.shape[1] < 3:
                # Pad with zeros for missing dimensions (e.g., 2D -> 3D)
                padding = np.zeros((n_points, 3 - points.shape[1]), dtype=points.dtype)
                points = np.hstack([points, padding])
            elif points.shape[1] > 3:
                # Truncate to 3 dimensions
                points = points[:, :3]

        if len(points) == 0:
            self.cells = []
            self.station_bathy = np.array([], dtype=float)
            self.points_on_proc = np.array([], dtype=float)
            return self.points_on_proc

        # Build bounding box tree for geometric searches
        try:
            # DOLFINx 0.6.0 style
            bb_tree = geometry.BoundingBoxTree(self.domain, self.domain.topology.dim)
        except:
            # DOLFINx 0.8.0+ style
            bb_tree = geometry.bb_tree(self.domain, self.domain.topology.dim)

        cells = []
        points_on_proc = []

        # Find cells whose bounding-box collide with the points
        try:
            # DOLFINx 0.6.0 style
            cell_candidates = geometry.compute_collisions(bb_tree, points)
        except:
            # DOLFINx 0.8.0+ style
            cell_candidates = geometry.compute_collisions_points(bb_tree, points)

        # Choose one of the cells that contains the point
        colliding_cells = geometry.compute_colliding_cells(
            self.domain, cell_candidates, points
        )

        self.station_index = []
        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])
                self.station_index.append(i)

        self.cells = cells

        # Evaluate bathymetry at station locations
        bathy_func = fe.Function(self.V_scalar)
        bathy_func.interpolate(
            fe.Expression(self.h_b, self.V_scalar.element.interpolation_points())
        )
        self.station_bathy = bathy_func.eval(points_on_proc, self.cells)

        self.points_on_proc = np.array(points_on_proc, dtype=np.float64)
        return self.points_on_proc

    def record_stations(self, u_sol, solution_var: str = "h") -> np.ndarray:
        """Record time series values at station locations.

        Args:
            u_sol: Solution function containing water depth and velocity
            solution_var: Type of solution variable ('h', 'eta', or 'flux')

        Returns:
            Array of shape (n_stations, 3) containing [h, u, v] at each station
        """
        # Evaluate water depth at stations
        h_values = u_sol.sub(0).eval(self.points_on_proc, self.cells)

        # Adjust for free surface elevation if needed
        if solution_var in ["h", "flux"]:
            h_values -= self.station_bathy

        # Evaluate velocity at stations
        vel_values = u_sol.sub(1).eval(self.points_on_proc, self.cells)

        # Ensure proper shapes for concatenation
        # h_values should be (n_stations,) and vel_values should be (n_stations, 2)
        if h_values.ndim == 1:
            h_values = h_values.reshape(-1, 1)  # Shape: (n_stations, 1)
        if vel_values.ndim == 1:
            # Single station case
            vel_values = vel_values.reshape(1, -1)  # Shape: (1, 2)

        # Combine into single array: [h, u, v]
        result = np.hstack([h_values, vel_values])
        return result

    def check_dry_nodes(
        self,
        solution,
        evaluation_points: np.ndarray,
        save_bathy: bool = False,
        save_true_bathy: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Identify and apply wetting/drying conditions at monitoring stations.

        This method identifies "dry" nodes where the water surface elevation
        is below the bathymetry, which is important for data assimilation
        applications where observations in dry regions should be treated
        differently.

        Args:
            solution: FEniCS solution object containing water depth (H) and velocity
            evaluation_points: Coordinates where solution should be evaluated
            save_bathy: If True, save bathymetry values to storage
            save_true_bathy: If True, save true bathymetry and dry indices

        Returns:
            Tuple of (water_height, dry_node_indices) where:
                - water_height: Array of water depths at evaluation points
                - dry_node_indices: Indices of points that are dry
        """
        # Extract water depth values at evaluation points
        water_height = solution.sub(0).eval(evaluation_points, self.cells)

        # Get bathymetry values
        # If h_b is a Constant, use its value; otherwise evaluate it
        if hasattr(self.h_b, 'eval'):
            bathy = self.h_b.eval(evaluation_points, self.cells)
        else:
            # For Constant objects, use the stored station bathymetry from init_stations
            bathy = self.station_bathy

        # Calculate water surface elevation relative to bathymetry
        water_surface_elevation = water_height - bathy

        # Identify dry nodes: where water surface is below the bottom
        is_dry = water_surface_elevation < -bathy
        dry_node_indices = np.where(is_dry)[0]

        return water_height, dry_node_indices

    def gather_station(
        self, root: int, local_stats: np.ndarray, local_vals: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Gather station data from all processors to the root processor.

        Args:
            root: Root MPI rank to gather to
            local_stats: Local station coordinates (unused, kept for compatibility)
            local_vals: Local station values to gather

        Returns:
            Tuple of (indices, values) on root rank, (None, None) on others:
                - indices: Global station indices
                - values: Gathered station values
        """
        comm = self.domain.comm
        rank = comm.Get_rank()

        gathered_vals = comm.gather(local_vals, root=root)
        gathered_inds = comm.gather(np.array(self.station_index, dtype=int), root=root)

        vals = None
        inds = None

        if rank == root:
            vals = np.concatenate(gathered_vals, axis=1)
            inds = np.concatenate(gathered_inds)

        return inds, vals

    def get_num_local_stations(self) -> int:
        """Get number of stations on this processor.

        Returns:
            Number of stations located on this MPI rank
        """
        return len(self.station_index)

    def get_station_coordinates(self) -> np.ndarray:
        """Get coordinates of stations on this processor.

        Returns:
            Array of station coordinates
        """
        return self.points_on_proc

    def __repr__(self) -> str:
        return (
            f"StationManager("
            f"local_stations={self.get_num_local_stations()}, "
            f"bathy_evaluated={len(self.station_bathy) > 0})"
        )

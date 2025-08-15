import numpy as np
from typing import Union, List, Dict, Tuple, Optional


class StationSelector:
    """
    Flexible station selection system for observation matrices.
    Supports multiple selection methods: indices, coordinates, frequency, and regions.
    """

    def __init__(self, prob, V):
        self.prob = prob
        self.V = V
        self.num_cells = len(prob.mesh.geometry.dofmap)
        self.all_cells = np.arange(self.num_cells)

        # Pre-compute cell coordinates for efficiency
        self._compute_cell_coordinates()

    def _compute_cell_coordinates(self):
        """Pre-compute coordinates for all cells"""
        V_collapsed, indices_into_V = self.V.sub(0).collapse()
        collapsed_dof_coords = V_collapsed.tabulate_dof_coordinates()

        self.cell_coordinates = []
        for i in range(self.num_cells):
            coords_for_cell = collapsed_dof_coords[V_collapsed.dofmap.cell_dofs(i)]
            cell_coord = coords_for_cell.mean(axis=0)  # centroid of cell
            self.cell_coordinates.append(cell_coord)

        self.cell_coordinates = np.array(self.cell_coordinates)
        self.V_collapsed = V_collapsed
        self.indices_into_V = np.array(indices_into_V)

    def select_by_indices(self, cell_indices: List[int]) -> np.ndarray:
        """Select stations by specific cell indices"""
        return np.array(cell_indices)

    def select_by_frequency(self, freq: int, offset: int = 0) -> np.ndarray:
        """Select stations by frequency (every freq-th cell)"""
        return self.all_cells[offset::freq]

    def select_by_coordinates(
        self, target_coords: List[Tuple], tolerance: float = None
    ) -> np.ndarray:
        """Select stations nearest to specified coordinates"""
        selected_cells = []

        for target_coord in target_coords:
            distances = np.linalg.norm(
                self.cell_coordinates - np.array(target_coord), axis=1
            )
            nearest_cell = np.argmin(distances)

            if tolerance is not None and distances[nearest_cell] > tolerance:
                print(
                    f"Warning: No cell found within tolerance {tolerance} for coordinate {target_coord}"
                )
                continue

            selected_cells.append(nearest_cell)

        return np.array(selected_cells)

    def select_by_region(
        self, bounds: Dict[str, Tuple], criteria: str = "center"
    ) -> np.ndarray:
        """
        Select stations within a rectangular region

        Args:
            bounds: Dict with keys 'x', 'y', 'z' (as needed) and values as (min, max) tuples
            criteria: 'center' (cell center in region) or 'any' (any part of cell in region)
        """
        selected_cells = []

        for i, coord in enumerate(self.cell_coordinates):
            in_region = True

            for dim_idx, (dim_name, (min_val, max_val)) in enumerate(bounds.items()):
                if dim_idx < len(coord):
                    if not (min_val <= coord[dim_idx] <= max_val):
                        in_region = False
                        break

            if in_region:
                selected_cells.append(i)

        return np.array(selected_cells)

    def select_by_pattern(self, pattern_config: Dict) -> np.ndarray:
        """
        Select stations using a pattern configuration

        Example patterns:
        - {'type': 'grid', 'nx': 5, 'ny': 5, 'bounds': {'x': (0, 1), 'y': (0, 1)}}
        """
        pattern_type = pattern_config["type"]

        if pattern_type == "grid":
            return self._select_grid_pattern(pattern_config)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

    def _select_grid_pattern(self, config):
        """Create a grid of stations"""
        nx, ny = config["nx"], config["ny"]
        bounds = config["bounds"]

        x_range = bounds["x"]
        y_range = bounds["y"]

        x_coords = np.linspace(x_range[0], x_range[1], nx)
        y_coords = np.linspace(y_range[0], y_range[1], ny)

        target_coords = [(x, y) for x in x_coords for y in y_coords]
        return self.select_by_coordinates(target_coords)


def build_observation_matrix(prob, V, station_config):
    """
    Build observation matrix with flexible station selection

    Args:
        prob: Problem object with mesh
        V: Function space
        station_config: Dictionary specifying how to select stations

    Station config examples:
        {'method': 'indices', 'params': [0, 5, 10, 15, 20]}
        {'method': 'frequency', 'params': {'freq': 3, 'offset': 1}}
        {'method': 'coordinates', 'params': [(0.1, 0.2), (0.5, 0.8), (0.9, 0.1)]}
        {'method': 'region', 'params': {'bounds': {'x': (0.2, 0.8), 'y': (0.3, 0.7)}}}
        {'method': 'pattern', 'params': {'type': 'grid', 'nx': 3, 'ny': 3, 'bounds': {'x': (0, 1), 'y': (0, 1)}}}
    """

    selector = StationSelector(prob, V)

    # Select stations based on configuration
    method = station_config["method"]
    params = station_config["params"]

    if method == "indices":
        station_cells = selector.select_by_indices(params)
    elif method == "frequency":
        station_cells = selector.select_by_frequency(**params)
    elif method == "coordinates":
        station_cells = selector.select_by_coordinates(params)
    elif method == "region":
        station_cells = selector.select_by_region(**params)
    elif method == "pattern":
        station_cells = selector.select_by_pattern(params)
    else:
        raise ValueError(f"Unknown selection method: {method}")

    # Remove duplicates and sort

    station_cells = np.array(
        list(dict.fromkeys(station_cells))
    )  # Remove duplicates, preserve order

    # Create observation matrix
    H = np.zeros((len(station_cells), V.dofmap.index_map.size_local))
    station_coords = []

    # Get the collapsed space and indices
    V_collapsed, indices_into_V = V.sub(0).collapse()
    collapsed_dof_coords = V_collapsed.tabulate_dof_coordinates()
    indices_into_V = np.array(indices_into_V)

    # Build the observation matrix
    for station, cell_idx in enumerate(station_cells):
        coords_for_cell = collapsed_dof_coords[V_collapsed.dofmap.cell_dofs(cell_idx)]
        dofs_in_orig_V = indices_into_V[V_collapsed.dofmap.cell_dofs(cell_idx)]
        H[station, dofs_in_orig_V] = 1 / 3
        station_coord = coords_for_cell.mean(axis=0)
        station_coords.append(station_coord)

    print(f"Selected {len(station_cells)} stations using method '{method}'")
    print(f"Station cells: {station_cells}")

    return H, np.array(station_coords), station_cells

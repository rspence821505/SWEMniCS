"""Classes for meteorological forcing.

Currently only OceanWeather-esque (gridded) forcing is supported, and must be provided in an HDF5 file.
"""

from dolfinx import fem as fe
from swe4dvar.physics.constants import R
import numpy as np


class GriddedForcing:
    """Supports meteorological forcing on a regular grid.

    For spherical meshes (Shinnecock, etc.), the grid is in lat/lon degrees
    and mesh coordinates are reverse-projected from meters to degrees.

    For Cartesian meshes (idealized inlet, etc.), set cartesian=True.
    The grid coordinates ("latitude"/"longitude" in the HDF5) are then
    interpreted directly as y/x coordinates in meters — no projection.
    """

    def __init__(self, filename, lat0=35, cartesian=False):
        """Initialize the forcing.

        Parameters
        ----------
        filename : str
            Path to HDF5 file with datasets: latitude, longitude, time,
            windx, windy, pressure.
        lat0 : float
            Reference latitude for spherical reverse-projection (ignored
            when cartesian=True).
        cartesian : bool
            If True, grid coordinates are in meters (x, y) and no
            spherical projection is applied.
        """
        try:
            import h5py
        except ModuleNotFoundError:
            raise ModuleNotFoundError("h5py is required for GriddedForcing")
        with h5py.File(filename, "r") as ds:
            self.lat = ds["latitude"][:]
            self.lon = ds["longitude"][:]
            self._windx = ds["windx"][:]
            self._windy = ds["windy"][:]
            self._pressure = ds["pressure"][:]
            self.time = ds["time"][:]

        # check dimensions
        nt, nlat, nlon = len(self.time), len(self.lat), len(self.lon)
        shape = (nt, nlat, nlon)
        print(self._windx.shape, self._windy.shape, self._pressure.shape)
        assert self._windx.shape == shape
        assert self._windy.shape == shape
        assert self._pressure.shape == shape
        self.lat0 = lat0
        self.cartesian = cartesian

    def set_V(self, V):
        """Set the FunctionSpace and precompute interpolation weights.

        For spherical meshes, reverse-projects mesh coords from projected
        meters to lat/lon degrees before interpolation.
        For Cartesian meshes (self.cartesian=True), uses mesh coords directly.
        """
        self._V = V
        self.windx = fe.Function(V)
        self.windy = fe.Function(V)
        self.pressure = fe.Function(V)
        self.coords = V.tabulate_dof_coordinates()[:, :2]

        if not self.cartesian:
            # Reverse spherical projection: projected meters → degrees
            self.coords = np.rad2deg(
                self.coords / np.array([[R * np.cos(np.deg2rad(self.lat0)), R]])
            )
        # else: coords are already in the same units as the grid (meters)

        # determine insertion indices
        # "lon" = x-axis grid, "lat" = y-axis grid (regardless of coordinate system)
        lon_inds = np.searchsorted(self.lon, self.coords[:, 0])
        lat_inds = np.searchsorted(self.lat, self.coords[:, 1])
        lower_lon_inds = lon_inds - 1
        lower_lat_inds = lat_inds - 1
        # ensure all indices are within bounds
        lon_inds = np.clip(lon_inds, 0, len(self.lon) - 1)
        lower_lon_inds = np.clip(lower_lon_inds, 0, len(self.lon) - 1)
        lat_inds = np.clip(lat_inds, 0, len(self.lat) - 1)
        lower_lat_inds = np.clip(lower_lat_inds, 0, len(self.lat) - 1)

        # assume regular spacing, for now
        lon_cellsize = self.lon[1] - self.lon[0]
        lat_cellsize = self.lat[1] - self.lat[0]
        # compute interpolation weights
        self.lon_pos = (self.coords[:, 0] - self.lon[lower_lon_inds]) / lon_cellsize
        self.lat_pos = (self.coords[:, 1] - self.lat[lower_lat_inds]) / lat_cellsize
        self.lon_inds, self.lower_lon_inds = lon_inds, lower_lon_inds
        self.lat_inds, self.lower_lat_inds = lat_inds, lower_lat_inds

    def evaluate(self, t):
        """Evaluate wind and pressure at a specified time t"""

        t_ind = np.searchsorted(self.time, t)
        time_resolution = self.time[1] - self.time[0]
        last_ind = t_ind - 1
        if last_ind < 0:
            last_ind = 0
        if t_ind >= len(self.time):
            t_ind = len(self.time) - 1
        t_pos = (t - self.time[last_ind]) / (time_resolution)

        # Linear temporal interpolation: windx(t) = α·windx[i₀] + (1-α)·windx[i₁]
        # where α = (t - t[i₀])/(t[i₁] - t[i₀]) is the normalized time position
        windx = t_pos * self._windx[last_ind] + (1 - t_pos) * self._windx[t_ind]

        # Linear temporal interpolation: windy(t) = α·windy[i₀] + (1-α)·windy[i₁]
        # where α = (t - t[i₀])/(t[i₁] - t[i₀]) is the normalized time position
        windy = t_pos * self._windy[last_ind] + (1 - t_pos) * self._windy[t_ind]

        # Linear temporal interpolation: pressure(t) = α·pressure[i₀] + (1-α)·pressure[i₁]
        # where α = (t - t[i₀])/(t[i₁] - t[i₀]) is the normalized time position
        pressure = (
            t_pos * self._pressure[last_ind] + (1 - t_pos) * self._pressure[t_ind]
        )

        # Bilinear spatial interpolation: windx(x,y) = Σᵢⱼ wᵢⱼ·windx[i,j]
        # where wᵢⱼ are the bilinear weights based on lon_pos and lat_pos
        self.windx.x.array[:] = (
            self.lon_pos
            * self.lat_pos
            * windx[self.lower_lat_inds, self.lower_lon_inds]
            + self.lon_pos
            * (1 - self.lat_pos)
            * windx[self.lat_inds, self.lower_lon_inds]
            + (1 - self.lon_pos)
            * self.lat_pos
            * windx[self.lower_lat_inds, self.lon_inds]
            + (1 - self.lon_pos)
            * (1 - self.lat_pos)
            * windx[self.lat_inds, self.lon_inds]
        )

        # Bilinear spatial interpolation: windy(x,y) = Σᵢⱼ wᵢⱼ·windy[i,j]
        # where wᵢⱼ are the bilinear weights based on lon_pos and lat_pos
        self.windy.x.array[:] = (
            self.lon_pos
            * self.lat_pos
            * windy[self.lower_lat_inds, self.lower_lon_inds]
            + self.lon_pos
            * (1 - self.lat_pos)
            * windy[self.lat_inds, self.lower_lon_inds]
            + (1 - self.lon_pos)
            * self.lat_pos
            * windy[self.lower_lat_inds, self.lon_inds]
            + (1 - self.lon_pos)
            * (1 - self.lat_pos)
            * windy[self.lat_inds, self.lon_inds]
        )

        # Bilinear spatial interpolation: pressure(x,y) = Σᵢⱼ wᵢⱼ·pressure[i,j]
        # where wᵢⱼ are the bilinear weights based on lon_pos and lat_pos
        # Convert from mBar to pascals (multiply by 100)
        self.pressure.x.array[:] = 100 * (
            self.lon_pos
            * self.lat_pos
            * pressure[self.lower_lat_inds, self.lower_lon_inds]
            + self.lon_pos
            * (1 - self.lat_pos)
            * pressure[self.lat_inds, self.lower_lon_inds]
            + (1 - self.lon_pos)
            * self.lat_pos
            * pressure[self.lower_lat_inds, self.lon_inds]
            + (1 - self.lon_pos)
            * (1 - self.lat_pos)
            * pressure[self.lat_inds, self.lon_inds]
        )


class ParametricWindForcing(GriddedForcing):
    """Low-dimensional parameterization of a reference gridded wind field.

    Parameters are interpreted as:
    - theta[0]: zonal track shift in km
    - theta[1]: multiplicative wind-intensity bias
    - theta[2]: temporal offset in seconds

    This stage-1 implementation deliberately keeps the mapping simple and
    explicit so the control-to-forcing dependence is fully traceable.
    """

    parameter_names = ("track_shift_km", "intensity_bias", "timing_offset_s")

    def __init__(self, filename, lat0=35, theta=None):
        super().__init__(filename, lat0=lat0)
        self.theta = np.zeros(3, dtype=float)
        if theta is not None:
            self.set_parameters(theta)

    def set_parameters(self, theta):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != 3:
            raise ValueError(
                f"ParametricWindForcing expects 3 parameters, got {theta.size}"
            )
        self.theta[:] = theta

    def evaluate(self, t):
        """Evaluate the transformed wind field at time ``t``."""
        track_shift_km, intensity_bias, timing_offset_s = self.theta

        effective_t = t + timing_offset_s
        t_ind = np.searchsorted(self.time, effective_t)
        time_resolution = self.time[1] - self.time[0]
        last_ind = t_ind - 1
        if last_ind < 0:
            last_ind = 0
        if t_ind >= len(self.time):
            t_ind = len(self.time) - 1
        t_pos = (effective_t - self.time[last_ind]) / time_resolution

        windx = t_pos * self._windx[last_ind] + (1 - t_pos) * self._windx[t_ind]
        windy = t_pos * self._windy[last_ind] + (1 - t_pos) * self._windy[t_ind]
        pressure = (
            t_pos * self._pressure[last_ind] + (1 - t_pos) * self._pressure[t_ind]
        )

        wind_scale = 1.0 + intensity_bias
        pressure_deficit_scale = wind_scale * wind_scale
        ambient_pressure = 1013.25
        windx = wind_scale * windx
        windy = wind_scale * windy
        pressure = ambient_pressure + pressure_deficit_scale * (pressure - ambient_pressure)

        shift_lon_deg = track_shift_km / (111.32 * np.cos(np.deg2rad(self.lat0)))
        shifted_lon = self.coords[:, 0] - shift_lon_deg
        shifted_lat = self.coords[:, 1]

        lon_inds = np.searchsorted(self.lon, shifted_lon)
        lat_inds = np.searchsorted(self.lat, shifted_lat)
        lower_lon_inds = np.clip(lon_inds - 1, 0, len(self.lon) - 1)
        lower_lat_inds = np.clip(lat_inds - 1, 0, len(self.lat) - 1)
        lon_inds = np.clip(lon_inds, 0, len(self.lon) - 1)
        lat_inds = np.clip(lat_inds, 0, len(self.lat) - 1)

        lon_cellsize = self.lon[1] - self.lon[0]
        lat_cellsize = self.lat[1] - self.lat[0]
        lon_pos = (shifted_lon - self.lon[lower_lon_inds]) / lon_cellsize
        lat_pos = (shifted_lat - self.lat[lower_lat_inds]) / lat_cellsize

        self.windx.x.array[:] = (
            lon_pos
            * lat_pos
            * windx[lower_lat_inds, lower_lon_inds]
            + lon_pos
            * (1 - lat_pos)
            * windx[lat_inds, lower_lon_inds]
            + (1 - lon_pos)
            * lat_pos
            * windx[lower_lat_inds, lon_inds]
            + (1 - lon_pos)
            * (1 - lat_pos)
            * windx[lat_inds, lon_inds]
        )

        self.windy.x.array[:] = (
            lon_pos
            * lat_pos
            * windy[lower_lat_inds, lower_lon_inds]
            + lon_pos
            * (1 - lat_pos)
            * windy[lat_inds, lower_lon_inds]
            + (1 - lon_pos)
            * lat_pos
            * windy[lower_lat_inds, lon_inds]
            + (1 - lon_pos)
            * (1 - lat_pos)
            * windy[lat_inds, lon_inds]
        )

        self.pressure.x.array[:] = 100 * (
            lon_pos
            * lat_pos
            * pressure[lower_lat_inds, lower_lon_inds]
            + lon_pos
            * (1 - lat_pos)
            * pressure[lat_inds, lower_lon_inds]
            + (1 - lon_pos)
            * lat_pos
            * pressure[lower_lat_inds, lon_inds]
            + (1 - lon_pos)
            * (1 - lat_pos)
            * pressure[lat_inds, lon_inds]
        )

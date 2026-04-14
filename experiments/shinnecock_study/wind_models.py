"""Synthetic wind field generators for Phase 7 wind-driven twin experiments.

Generates HDF5 files compatible with GriddedForcing (src/swe4dvar/physics/forcing.py).
Two models:
  1. Holland (1980) parametric hurricane with translating vortex
  2. Simple uniform wind ramp (for debugging / ablation)

Wind grid coordinates are geographic (lon, lat) — GriddedForcing reverse-projects
mesh DOF coordinates from projected space to (lon, lat) for interpolation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import copy


# Physical constants (match src/swe4dvar/physics/constants.py)
RHO_AIR = 1.293       # kg/m^3
OMEGA = 7.292124e-5    # Earth angular velocity (rad/s)
KM_PER_DEG_LAT = 111.32  # km per degree latitude


@dataclass
class WindGridConfig:
    """Configuration for the wind field grid.

    Grid covers the Shinnecock domain (lon≈[-72.9,-72.0], lat≈[40.4,41.0])
    with margin for storm approach/departure.
    """
    lon_min: float = -73.5
    lon_max: float = -71.5
    lat_min: float = 39.5
    lat_max: float = 41.5
    nlon: int = 41   # 0.05° spacing ~5.5 km (needed for pressure gradient resolution)
    nlat: int = 41

    @property
    def lon(self) -> np.ndarray:
        return np.linspace(self.lon_min, self.lon_max, self.nlon)

    @property
    def lat(self) -> np.ndarray:
        return np.linspace(self.lat_min, self.lat_max, self.nlat)


@dataclass
class HollandHurricaneConfig:
    """Holland (1980) parametric hurricane model.

    Track waypoints define storm center position over time.
    Wind field includes gradient wind balance + 20° inflow angle + translation velocity.

    p_central_mbar is derived from Vmax if not explicitly set, using:
        dp = Vmax^2 * rho_air / (B * exp(-1))
    This ensures the gradient wind at r=Rmax equals Vmax (ignoring Coriolis correction).
    """
    track_waypoints: List[Tuple[float, float, float]]  # [(time_s, lon_deg, lat_deg), ...]
    Vmax: float = 30.0              # Maximum sustained wind speed (m/s)
    Rmax_km: float = 30.0           # Radius of maximum wind (km)
    p_central_mbar: Optional[float] = None  # Central pressure; derived from Vmax if None
    p_ambient_mbar: float = 1013.25 # Ambient pressure (mbar)
    B: float = 1.5                  # Holland shape parameter (1.0-2.5)
    inflow_angle_deg: float = 20.0  # Boundary layer inflow angle (degrees toward center)
    f_coriolis: float = 2 * OMEGA * np.sin(np.deg2rad(40.7))  # At Shinnecock latitude

    def __post_init__(self):
        """Derive p_central from Vmax if not explicitly provided."""
        if self.p_central_mbar is None:
            # dp = Vmax^2 * rho_air / (B * exp(-1)) ensures V(Rmax) ≈ Vmax
            dp_pa = self.Vmax**2 * RHO_AIR / (self.B * np.exp(-1.0))
            self.p_central_mbar = self.p_ambient_mbar - dp_pa / 100.0


@dataclass
class RampWindConfig:
    """Simple uniform wind ramp model.

    Wind ramps linearly from 0 to Wmax over ramp_duration_s, then holds constant.
    Direction is meteorological convention (degrees CW from N, FROM direction).
    """
    Wmax: float = 20.0              # Maximum wind speed (m/s)
    direction_deg: float = 225.0    # Wind FROM direction (225 = from SW)
    ramp_duration_s: float = 7200.0 # Time to reach Wmax (seconds)
    p_ambient_mbar: float = 1013.25 # Uniform ambient pressure


# Default storm track: approaches from SW, closest approach ~20km W of Shinnecock
# at DA window midpoint (t≈90000s = 25h, after 24h ramp).
# Storm starts ~800km away (wind < 3 m/s at domain during early ramp),
# arriving near the domain only in the last ~6h of ramp.
DEFAULT_TRACK = [
    (0,      -78.0, 36.0),    # ~800km SW — negligible wind at domain during early ramp
    (43200,  -76.0, 38.0),    # 12h: still ~400km away (~4 m/s at domain)
    (72000,  -74.0, 39.5),    # 20h: ~200km away (~8 m/s at domain)
    (86400,  -73.0, 40.3),    # 24h: DA window starts, ~60km away (~15 m/s at domain)
    (93600,  -72.7, 40.55),   # 26h: closest approach (~25km W of domain center)
    (108000, -72.0, 41.2),    # 30h: receding NE
    (129600, -71.5, 41.5),    # 36h: far NE
]


def _interp_track(track_waypoints: list, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate storm center position from track waypoints.

    Returns:
        lon_c: center longitude at each time (degrees)
        lat_c: center latitude at each time (degrees)
    """
    wp_t = np.array([wp[0] for wp in track_waypoints])
    wp_lon = np.array([wp[1] for wp in track_waypoints])
    wp_lat = np.array([wp[2] for wp in track_waypoints])

    lon_c = np.interp(times, wp_t, wp_lon)
    lat_c = np.interp(times, wp_t, wp_lat)
    return lon_c, lat_c


def _compute_translation_velocity(track_waypoints: list, times: np.ndarray,
                                   lon_c: np.ndarray, lat_c: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute storm translation velocity (m/s) from track.

    Uses centered finite differences on the interpolated track.

    Returns:
        Vt_x: eastward translation velocity (m/s) at each time
        Vt_y: northward translation velocity (m/s) at each time
    """
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    lat_c_rad = np.deg2rad(lat_c)

    # Centered differences (forward/backward at endpoints)
    dlon = np.gradient(lon_c, dt)  # deg/s
    dlat = np.gradient(lat_c, dt)  # deg/s

    # Convert to m/s
    Vt_x = dlon * KM_PER_DEG_LAT * np.cos(lat_c_rad) * 1000.0
    Vt_y = dlat * KM_PER_DEG_LAT * 1000.0

    return Vt_x, Vt_y


def generate_holland_wind_field(
    config: HollandHurricaneConfig,
    grid: WindGridConfig,
    times: np.ndarray,
    wind_ramp_s: float = 43200.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Holland hurricane wind and pressure fields.

    Wind field = gradient wind vortex + inflow angle + storm translation velocity.
    Pressure field = Holland pressure profile (dynamically consistent with wind).

    A temporal ramp multiplier tanh(2t/wind_ramp_s) gradually introduces wind
    and pressure forcing, matching the tidal ramp behavior. This prevents
    shock-loading the solver at t=0 when the far-field wind is non-negligible.

    Parameters
    ----------
    config : HollandHurricaneConfig
    grid : WindGridConfig
    times : np.ndarray
        Time array in seconds.
    wind_ramp_s : float
        Wind ramp timescale (seconds). Wind multiplier = tanh(2*t/wind_ramp_s).
        Default 43200 = 12h (reaches 96% at 12h). Set to 0 for no ramp.

    Returns:
        windx: (nt, nlat, nlon) eastward wind component (m/s)
        windy: (nt, nlat, nlon) northward wind component (m/s)
        pressure: (nt, nlat, nlon) atmospheric pressure (mbar)
    """
    nt = len(times)
    lon = grid.lon
    lat = grid.lat
    nlat, nlon = len(lat), len(lon)

    # Meshgrid for spatial coordinates
    LON, LAT = np.meshgrid(lon, lat)  # (nlat, nlon)
    LAT_RAD = np.deg2rad(LAT)

    # Storm parameters
    Rmax_m = config.Rmax_km * 1000.0
    dp = (config.p_ambient_mbar - config.p_central_mbar) * 100.0  # Pa
    B = config.B
    f = config.f_coriolis

    # Interpolate storm track
    lon_c, lat_c = _interp_track(config.track_waypoints, times)
    Vt_x, Vt_y = _compute_translation_velocity(config.track_waypoints, times, lon_c, lat_c)

    # Allocate output
    windx = np.zeros((nt, nlat, nlon))
    windy = np.zeros((nt, nlat, nlon))
    pressure = np.zeros((nt, nlat, nlon))

    inflow_rad = np.deg2rad(config.inflow_angle_deg)

    for ti in range(nt):
        # Distance from each grid point to storm center (km → m)
        dx_km = (LON - lon_c[ti]) * KM_PER_DEG_LAT * np.cos(np.deg2rad(lat_c[ti]))
        dy_km = (LAT - lat_c[ti]) * KM_PER_DEG_LAT
        r_km = np.sqrt(dx_km**2 + dy_km**2)
        r_km = np.maximum(r_km, 1.0)  # Floor at 1 km to prevent NaN at center
        r_m = r_km * 1000.0

        # Angle from center to grid point (math convention: CCW from east)
        theta = np.arctan2(dy_km, dx_km)

        # Holland pressure profile: p(r) = p_c + dp * exp(-(Rmax/r)^B)
        rmax_over_r_B = (config.Rmax_km / r_km) ** B
        exp_term = np.exp(-rmax_over_r_B)
        p_at_r = config.p_central_mbar + (config.p_ambient_mbar - config.p_central_mbar) * exp_term
        pressure[ti] = p_at_r

        # Holland gradient wind speed
        # V(r) = sqrt(B * dp * exp(-(Rmax/r)^B) * (Rmax/r)^B / rho_air + r^2*f^2/4) - r*f/2
        term1 = B * dp * exp_term * rmax_over_r_B / RHO_AIR
        term2 = r_m**2 * f**2 / 4.0
        V = np.sqrt(np.maximum(term1 + term2, 0.0)) - r_m * f / 2.0
        V = np.maximum(V, 0.0)

        # Wind direction: cyclonic (CCW in NH) + inflow angle toward center
        # Tangential direction: perpendicular to radial, CCW
        # theta_tang = theta + pi/2 (CCW tangent)
        # With inflow: rotate toward center by inflow_angle
        wind_dir = theta + np.pi / 2.0 - inflow_rad  # CCW tangent minus inflow

        # Decompose into (east, north) components (vortex only)
        Wx_vortex = -V * np.sin(wind_dir - theta) * np.cos(theta) + V * np.cos(wind_dir - theta) * (-np.sin(theta))
        # Simpler: direct decomposition
        Wx_vortex = V * np.cos(wind_dir)
        Wy_vortex = V * np.sin(wind_dir)

        # Add storm translation velocity (creates right-of-track asymmetry)
        windx[ti] = Wx_vortex + Vt_x[ti]
        windy[ti] = Wy_vortex + Vt_y[ti]

    # Apply temporal ramp: tanh(2*t/wind_ramp_s)
    # This gradually introduces wind forcing, preventing shock-loading at t=0
    # when the Holland far-field wind is still non-negligible (~5-7 m/s at 800km)
    if wind_ramp_s > 0:
        ramp = np.tanh(2.0 * times / wind_ramp_s)  # 0→1, reaches 0.96 at t=wind_ramp_s
        ramp_3d = ramp[:, np.newaxis, np.newaxis]
        windx *= ramp_3d
        windy *= ramp_3d
        # Ramp pressure deficit too (p approaches ambient during ramp)
        dp_field = pressure - config.p_ambient_mbar  # negative (low pressure at center)
        pressure = config.p_ambient_mbar + dp_field * ramp_3d

    return windx, windy, pressure


def generate_ramp_wind_field(
    config: RampWindConfig,
    grid: WindGridConfig,
    times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate uniform wind ramp field.

    Wind ramps linearly from 0 to Wmax, then holds constant.
    Direction converts from meteorological convention (FROM direction)
    to Cartesian components.

    Returns:
        windx: (nt, nlat, nlon) eastward wind component (m/s)
        windy: (nt, nlat, nlon) northward wind component (m/s)
        pressure: (nt, nlat, nlon) atmospheric pressure (mbar)
    """
    nt = len(times)
    nlat, nlon = len(grid.lat), len(grid.lon)

    # Convert FROM direction to TO direction in math convention
    # Meteorological: 0=N, 90=E, 180=S, 270=W (wind FROM)
    # Math: 0=E, 90=N (wind TO)
    # FROM SW (225°) means wind blows TO NE
    to_dir_met = (config.direction_deg + 180.0) % 360.0
    to_dir_rad = np.deg2rad(90.0 - to_dir_met)  # Convert to math convention

    wx_unit = np.cos(to_dir_rad)
    wy_unit = np.sin(to_dir_rad)

    # Time-dependent magnitude
    ramp_frac = np.minimum(times / config.ramp_duration_s, 1.0)
    W_mag = config.Wmax * ramp_frac  # (nt,)

    # Broadcast to full grid (uniform in space)
    windx = W_mag[:, np.newaxis, np.newaxis] * np.full((1, nlat, nlon), wx_unit)
    windy = W_mag[:, np.newaxis, np.newaxis] * np.full((1, nlat, nlon), wy_unit)

    # Uniform pressure (no pressure gradient for uniform wind)
    pressure = np.full((nt, nlat, nlon), config.p_ambient_mbar)

    return windx, windy, pressure


def generate_zero_wind_field(
    grid: WindGridConfig,
    times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate zero-wind field (tidal-only control).

    Returns:
        windx, windy: all zeros
        pressure: uniform 1013.25 mbar
    """
    nt = len(times)
    nlat, nlon = len(grid.lat), len(grid.lon)
    windx = np.zeros((nt, nlat, nlon))
    windy = np.zeros((nt, nlat, nlon))
    pressure = np.full((nt, nlat, nlon), 1013.25)
    return windx, windy, pressure


def write_wind_hdf5(
    filepath: str,
    grid: WindGridConfig,
    times: np.ndarray,
    windx: np.ndarray,
    windy: np.ndarray,
    pressure: np.ndarray,
) -> None:
    """Write wind field to HDF5 compatible with GriddedForcing.

    Dataset names and conventions match GriddedForcing.__init__() exactly:
      - "latitude": ascending 1D array (degrees)
      - "longitude": ascending 1D array (degrees)
      - "time": 1D array (seconds)
      - "windx": 3D (nt, nlat, nlon) — eastward wind (m/s)
      - "windy": 3D (nt, nlat, nlon) — northward wind (m/s)
      - "pressure": 3D (nt, nlat, nlon) — atmospheric pressure (mbar)
    """
    import h5py

    lon = grid.lon
    lat = grid.lat
    nt, nlat, nlon = len(times), len(lat), len(lon)

    # Validate shapes
    assert windx.shape == (nt, nlat, nlon), f"windx shape {windx.shape} != ({nt}, {nlat}, {nlon})"
    assert windy.shape == (nt, nlat, nlon), f"windy shape {windy.shape} != ({nt}, {nlat}, {nlon})"
    assert pressure.shape == (nt, nlat, nlon), f"pressure shape {pressure.shape} != ({nt}, {nlat}, {nlon})"

    # Validate ascending order
    assert np.all(np.diff(lon) > 0), "longitude must be strictly ascending"
    assert np.all(np.diff(lat) > 0), "latitude must be strictly ascending"

    # Validate time coverage
    assert times[0] <= 0, f"time array must start at t≤0 (starts at {times[0]})"

    with h5py.File(filepath, "w") as f:
        f.create_dataset("latitude", data=lat)
        f.create_dataset("longitude", data=lon)
        f.create_dataset("time", data=times)
        f.create_dataset("windx", data=windx)
        f.create_dataset("windy", data=windy)
        f.create_dataset("pressure", data=pressure)

    print(f"  Wind HDF5 written: {filepath} ({nt} times, {nlat}×{nlon} grid)")


def generate_perturbed_config(
    base_config,
    perturbation_type: str,
    magnitude: float,
):
    """Create a perturbed copy of a wind configuration.

    Parameters
    ----------
    base_config : HollandHurricaneConfig or RampWindConfig
    perturbation_type : str
        "track_shift": shift hurricane track perpendicular by magnitude km
        "intensity_bias": scale Vmax/Wmax by (1 + magnitude)
        "timing_offset": shift storm timing by magnitude seconds
        "direction_shift": rotate ramp wind direction by magnitude degrees
    magnitude : float
        Perturbation magnitude (units depend on type)

    Returns
    -------
    Perturbed config of same type as base_config
    """
    config = copy.deepcopy(base_config)

    if perturbation_type == "track_shift":
        if not isinstance(config, HollandHurricaneConfig):
            raise ValueError("track_shift only applies to HollandHurricaneConfig")

        # Shift track perpendicular to track direction
        # Compute average track direction
        wps = config.track_waypoints
        avg_dir_rad = np.arctan2(
            wps[-1][2] - wps[0][2],  # dlat
            (wps[-1][1] - wps[0][1]) * np.cos(np.deg2rad(40.7))  # dlon (scaled)
        )
        # Perpendicular direction (to the right of track)
        perp_rad = avg_dir_rad - np.pi / 2.0

        # Shift in degrees
        shift_lon = magnitude / (KM_PER_DEG_LAT * np.cos(np.deg2rad(40.7))) * np.cos(perp_rad)
        shift_lat = magnitude / KM_PER_DEG_LAT * np.sin(perp_rad)

        config.track_waypoints = [
            (t, lon + shift_lon, lat + shift_lat) for t, lon, lat in wps
        ]

    elif perturbation_type == "intensity_bias":
        if isinstance(config, HollandHurricaneConfig):
            config.Vmax *= (1.0 + magnitude)
            # Scale pressure deficit consistently: dp ∝ Vmax^2
            dp_orig = config.p_ambient_mbar - config.p_central_mbar
            dp_new = dp_orig * (1.0 + magnitude) ** 2
            config.p_central_mbar = config.p_ambient_mbar - dp_new
        elif isinstance(config, RampWindConfig):
            config.Wmax *= (1.0 + magnitude)

    elif perturbation_type == "timing_offset":
        if isinstance(config, HollandHurricaneConfig):
            config.track_waypoints = [
                (t + magnitude, lon, lat) for t, lon, lat in config.track_waypoints
            ]
        else:
            raise ValueError("timing_offset only applies to HollandHurricaneConfig")

    elif perturbation_type == "direction_shift":
        if isinstance(config, RampWindConfig):
            config.direction_deg = (config.direction_deg + magnitude) % 360.0
        else:
            raise ValueError("direction_shift only applies to RampWindConfig")

    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")

    return config

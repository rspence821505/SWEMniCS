# Idealized Inlet Wind-Driven Twin Experiment: Feasibility Audit

## 1. Executive Summary

The idealized inlet **can** support wind-driven twin experiments with **modest implementation effort**. The core obstacle is a single function: `GriddedForcing.set_V()` hardcodes a reverse spherical projection (line 46-48 in `src/swe4dvar/physics/forcing.py`) that assumes mesh coordinates are in projected lat/lon. For the Cartesian idealized inlet, this produces garbage coordinates.

**Recommended path**: Create a `CartesianGriddedForcing` subclass (or add a `cartesian=True` flag to `GriddedForcing`) that skips the reverse projection and interprets the HDF5 grid coordinates directly as meters. This is a ~20-line change. Everything else in the forcing pipeline (wind stress computation, pressure gradient, drag coefficient, time interpolation) works identically in Cartesian and spherical — the physics is coordinate-system agnostic once the wind field is on the mesh.

---

## 2. Relevant Files and Components

| File | Role |
|---|---|
| `src/swe4dvar/physics/forcing.py` | `GriddedForcing`, `ParametricWindForcing` — wind field interpolation |
| `src/swe4dvar/forward/problems.py` | `IdealizedInlet`, `BaseProblem.make_Source()` — forcing entry point |
| `src/swe4dvar/forward/solvers/cg_implicit.py` | Weak form assembly — `init_weak_form()` calls `make_Source()` |
| `experiments/shinnecock_study/wind_models.py` | `HollandHurricaneConfig`, `generate_holland_wind_field()` |
| `examples/idealized_inlet.py` | Existing idealized inlet example |

---

## 3. Current Idealized Inlet Forcing Architecture

### 3.1 Forcing Entry Point

The `IdealizedInlet` inherits from `TidalProblem` which inherits from `BaseProblem`. The forcing attribute is defined at line 68 of `problems.py`:

```python
@dataclass
class BaseProblem:
    forcing: GriddedForcing = None
```

Currently, `IdealizedInlet` is instantiated with `forcing=None` (no wind). If a `GriddedForcing` object were passed, the following chain would execute:

1. `BaseProblem.init_V()` (line 132-134): calls `self.forcing.set_V(scalar_V)` and `self.forcing.evaluate(self.t)`
2. `BaseProblem.make_Source()` (line 528-576): adds wind stress and pressure gradient terms to the momentum equation
3. `TidalProblem.advance_time()` (line 880-884): calls `self.forcing.evaluate(self.t)` at each timestep

### 3.2 What Already Works

- **Wind stress physics**: `make_Source()` adds `-ρ_air/ρ_water * C_D * |W| * W` to momentum — purely physical, no coordinate assumptions
- **Pressure gradient**: `h * ∇p / ρ_water` — uses UFL derivatives, coordinate-agnostic
- **Drag coefficient**: `C_D = (0.75 + 0.067 * |W|) * 1e-3` — scalar formula, no coordinates
- **Time interpolation**: Linear in `evaluate()` — works for any grid
- **Temporal ramping**: `tanh(2t/ramp_s)` — independent of coordinates

### 3.3 What Breaks

**`GriddedForcing.set_V()`**, line 44-48:
```python
self.coords = V.tabulate_dof_coordinates()[:, :2]
self.coords = np.rad2deg(
    self.coords / np.array([[R * np.cos(np.deg2rad(self.lat0)), R]])
)
```

This reverse-projects from meters to degrees assuming a Mercator-like projection. For the idealized inlet (x ∈ [0, 50000] m, y ∈ [0, 30500] m), this produces coordinates of ~0.0003° — meaningless for any realistic wind grid.

### 3.4 Spherical Flag

`IdealizedInlet` inherits `spherical = False` from `BaseProblem`. This means the spherical metric scaling in `make_Source()` is skipped (correct for Cartesian). The `ADCIRCProblem` sets `spherical = True` for Shinnecock.

---

## 4. Current Shinnecock Wind-Forcing Architecture

### 4.1 Full Call Chain

1. **Wind generation**: `generate_holland_wind_field()` produces `(windx, windy, pressure)` arrays on a lat/lon grid
2. **HDF5 storage**: `write_wind_hdf5()` saves with datasets `latitude`, `longitude`, `time`, `windx`, `windy`, `pressure`
3. **Loading**: `GriddedForcing.__init__()` reads HDF5, stores lat/lon/time arrays and 3D wind/pressure data
4. **Mesh setup**: `GriddedForcing.set_V()` reverse-projects mesh coords to lat/lon, precomputes bilinear interpolation weights
5. **Per-timestep**: `GriddedForcing.evaluate(t)` does temporal + spatial bilinear interpolation, writes to FEM Functions
6. **PDE assembly**: `make_Source()` reads `self.forcing.windx`, `.windy`, `.pressure` FEM Functions and constructs UFL wind stress + pressure gradient terms
7. **Newton solve**: Wind forcing enters the assembled residual `R(u)` at every Newton iteration

### 4.2 Perturbation Mechanism

`ParametricWindForcing` extends `GriddedForcing` with 3 parameters:
- `track_shift_km`: Shifts spatial interpolation coordinates (15km default)
- `intensity_bias`: Scales wind magnitude
- `timing_offset_s`: Shifts temporal interpolation

The twin experiment creates model error by using truth wind for the truth run and perturbed wind (track-shifted) for the DA run.

---

## 5. Compatibility Gap Analysis

| Component | Shinnecock | Idealized Inlet | Gap |
|---|---|---|---|
| Mesh coordinates | Projected lat/lon (meters) | Cartesian (meters) | **`set_V()` reverse projection fails** |
| HDF5 grid coords | lat/lon degrees | Would need meters | **Wind grid format different** |
| Wind generation | Holland hurricane on lat/lon | Needs Cartesian equivalent | **No Cartesian wind model exists** |
| `make_Source()` | Works (spherical=True) | Works (spherical=False) | **Compatible** |
| Wind stress physics | Identical | Identical | **Compatible** |
| Time interpolation | Identical | Identical | **Compatible** |
| Perturbation mechanism | Track shift in lat/lon | Would need Cartesian equivalent | **Needs adaptation** |
| Forcing attribute | `forcing=GriddedForcing(...)` | `forcing=None` currently | **Easy to add** |

**Only two real gaps:**
1. `GriddedForcing.set_V()` assumes reverse spherical projection
2. No Cartesian wind generation/perturbation model exists

---

## 6. Feasibility of Each Design Option

### Option A: Cartesian Adaptation of Holland Hurricane

**Complexity**: Medium
**What assumes lat/lon**:
- `generate_holland_wind_field()`: Storm track in (time, lon, lat), distances computed as `dx_km = (lon - center_lon) * 111.32 * cos(lat)`. This is deeply lat/lon-coupled.
- `WindGridConfig`: Grid defined as lon/lat ranges
- `generate_perturbed_config("track_shift", 15.0)`: Shift in km converted to degrees

**What would need rewriting**: Nearly everything in the wind generation. The Holland vortex profile (pressure, gradient wind) is purely radial (distance-based) and could work in Cartesian, but the grid setup, track interpolation, and perturbation logic all assume geographic coordinates.

**Verdict**: Possible but significant effort. Not the cleanest path.

### Option B: Simpler Idealized Wind Forcing

**Complexity**: Low
**Approach**: A spatially smooth wind field defined directly in Cartesian coordinates. For example:
- A translating Gaussian wind vortex: `W(x,y,t) = Vmax * exp(-r²/Rmax²)` centered at `(x_c(t), y_c(t))`
- Perturbation: shift the track by `Δx` or `Δy` km
- No lat/lon conversion needed — everything in meters

**Scientific validity**: This creates the exact same type of model error as the Holland model — a wind field mismatch between truth and forecast due to storm position uncertainty. The Gaussian vortex captures the essential physics: strong winds near center, decay with distance, translating storm. It lacks the Holland pressure profile and Coriolis-dependent asymmetry, but for algorithm validation this is sufficient.

**Verdict**: Best option. Simple, scientifically valid, fast to implement, no coordinate conversion issues.

### Option C: Minimal Forcing Interface Extension

**Complexity**: Very Low (~20 lines)
**Approach**: Add a `cartesian` flag to `GriddedForcing` that skips the reverse projection:

```python
def set_V(self, V):
    self.coords = V.tabulate_dof_coordinates()[:, :2]
    if not self.cartesian:
        self.coords = np.rad2deg(
            self.coords / np.array([[R * np.cos(np.deg2rad(self.lat0)), R]])
        )
    # ... rest unchanged (searchsorted, weights)
```

Then the HDF5 "latitude"/"longitude" datasets would contain y/x coordinates in meters instead of degrees. The bilinear interpolation is coordinate-system agnostic.

**Verdict**: This is the infrastructure enabler that makes Option B trivial. Implement C first, then B builds on top of it.

---

## 7. Recommended Implementation Path

**Primary path: Option C + B combined**

### Step 1: Add `cartesian` flag to `GriddedForcing` (~20 lines)
- Skip reverse projection when `cartesian=True`
- HDF5 grid uses meters for "latitude"/"longitude" (or rename to "y"/"x")

### Step 2: Implement Cartesian wind vortex generator (~80 lines)
- Translating Gaussian or Rankine vortex in (x, y, t)
- Configurable: center track, Vmax, Rmax, translation velocity
- Output: same HDF5 format as Holland model
- Perturbation: shift center track by (Δx, Δy)

### Step 3: Wire into idealized inlet experiment (~50 lines)
- Create wind HDF5 files for truth and perturbed tracks
- Pass `CartesianGriddedForcing` to `IdealizedInlet(forcing=...)`
- Run twin experiment using existing DA infrastructure

**Total estimated effort**: ~150 lines of new code, 0 architectural changes.

---

## 8. Minimal Requirements for a Valid Wind-Driven Twin Experiment

### Truth forcing
- Wind vortex translating across the idealized inlet domain
- Vmax = 20-40 m/s, Rmax = 10-20 km
- Track: e.g., (x=40000, y=-20000) → (x=25000, y=40000) over 24h
- Strong enough to create ~0.5-2m storm surge at the coast (y=30000)

### Perturbed forcing
- Same vortex with track shifted by Δx = 5-15 km perpendicular to track direction
- This creates a spatially coherent wind field mismatch — stronger on one side, weaker on the other

### Mismatch parameterization
- Single parameter: track shift distance (km)
- Controlled, reproducible, physically meaningful

### Expected state error
- Wind mismatch → different surge heights along the coast
- Error should be O(0.1-1m) in elevation, O(0.1-1 m/s) in velocity
- Verify by running truth and perturbed forward models and computing RMSE

### Observations
- Point elevation observations at interior nodes (same as Shinnecock)
- obs_fraction = 0.1-0.3, obs_frequency = 6 timesteps
- Noise level: 0.01 relative

### Verification
- Run truth forward, run perturbed forward, compute difference
- If difference is O(noise level), wind perturbation is too weak
- If difference causes solver divergence, wind is too strong

---

## 9. Is the Idealized Inlet a Good Testbed?

**Yes, with Option B wind forcing.** Reasons:

1. **Domain geometry**: 50km × 30km with sloping bathymetry (14m→5m) creates spatially varying surge response — different from Shinnecock's single-throat bottleneck
2. **Multiple flow paths**: Wide open boundary means wind-driven flow enters along the entire southern edge, creating rich spatial structure
3. **Depth variation**: 14m at open boundary to 5m at back wall means friction and nonlinearity effects vary spatially — better for DC-WME eigenvalue spread
4. **Existing infrastructure**: Mesh, solver, time-stepping, Newton solver all work. Only forcing needs addition.
5. **Manageable size**: 11,768 nodes (~4× Shinnecock) — feasible on a laptop (~2h per experiment)

**Potential concern**: The idealized inlet has **no narrow throat** (unlike Shinnecock). This is actually an advantage for DC-WME — the wider geometry should produce richer L_wme eigenvalue structure because perturbations in different parts of the domain evolve differently.

---

## 10. Next Implementation Steps

1. **Add `cartesian` flag to `GriddedForcing.set_V()`** — skip reverse projection
2. **Write `CartesianWindVortex` generator** — Gaussian/Rankine vortex in (x, y, t) with configurable track
3. **Write `generate_perturbed_vortex_config()`** — track shift perturbation
4. **Write `write_cartesian_wind_hdf5()`** — same format as Holland but with meter-based grid
5. **Create `experiments/idealized_inlet_twin.py`** — twin experiment driver
6. **Run forward verification** — truth vs perturbed, check surge difference magnitude
7. **Run identifiability audit** — compute L_wme Gram matrix eigenvalue spread
8. **Run 4D-Var vs DC-WME comparison** — the key scientific test

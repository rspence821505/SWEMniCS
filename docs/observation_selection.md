# Interactive Observation Point Selection Tool

This document describes how to use the `scripts/select_observation_points.py` tool for interactively selecting observation points on a mesh for 4D-Var data assimilation experiments.

## Overview

The observation point selection tool provides a graphical interface for manually selecting which mesh locations should be observed during a twin experiment. This is useful when you want:

- **Specific observation networks**: Select points at known sensor locations
- **Strategic placement**: Choose points in regions of interest (e.g., inlet channels, near boundaries)
- **Reproducible experiments**: Save and reuse the same observation locations across runs
- **DOF-aware selection**: Select at actual DOF locations for DG discretizations

## Quick Start

```bash
# Basic usage - select mesh nodes for CG/SUPG
python scripts/select_observation_points.py --mesh-type tidal

# For DG discretization - select at DOF locations
python scripts/select_observation_points.py --mesh-type tidal --solver-type DG

# For Shinnecock mesh
python scripts/select_observation_points.py --mesh-type shinnecock
```

## Command-Line Arguments

### Mesh Selection

| Argument | Values | Description |
|----------|--------|-------------|
| `--mesh-type` | `tidal`, `shinnecock`, `inlet` | Type of mesh to load |

### Solver/Discretization (Optional)

| Argument | Values | Description |
|----------|--------|-------------|
| `--solver-type` | `CG`, `SUPG`, `DG`, `DGNC`, `DGCG` | Solver type for DOF-based selection |
| `--p-degree` | Integer (default: 1) | Polynomial degree |

When `--solver-type` is specified as `DG`, `DGNC`, or `DGCG`, the tool shows DOF locations instead of mesh nodes. For `CG` and `SUPG`, DOFs coincide with mesh nodes.

### Tidal Mesh Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--nx` | 20 | Number of elements in x direction |
| `--ny` | 5 | Number of elements in y direction |
| `--x0` | 0.0 | Domain x minimum |
| `--x1` | 10000.0 | Domain x maximum |
| `--y0` | 0.0 | Domain y minimum |
| `--y1` | 5000.0 | Domain y maximum |

### File Paths

| Argument | Default | Description |
|----------|---------|-------------|
| `--adios-file` | `data/shinnecock_inlet` | Path to Shinnecock ADIOS files (without extension) |
| `--xdmf-file` | `data/Ideal_Inlet/Ideal_Inlet.xdmf` | Path to Inlet XDMF file |
| `-o`, `--output` | Auto-generated | Output JSON file path |

## Interactive Controls

Once the graphical window opens:

| Control | Action |
|---------|--------|
| **Left click** | Select or deselect the nearest point |
| **Right click** | Clear all selections |
| **Enter/Return** | Save selections and exit |
| **Escape** | Exit without saving |
| **'c'** | Clear all selections |
| **'s'** | Save current selection (without exiting) |

## Visual Elements

- **Light gray points**: Mesh nodes (node mode)
- **Light blue points**: DOF locations (DOF mode)
- **Red circles**: Currently selected points
- **Gray mesh lines**: Element boundaries (background)

## Usage Examples

### Example 1: Tidal Problem with Mesh Nodes (CG/SUPG)

```bash
# Create mesh and select observation points
python scripts/select_observation_points.py \
    --mesh-type tidal \
    --nx 20 \
    --ny 5

# Output: data/tidal_obs_points.json
```

### Example 2: Tidal Problem with DG DOFs

```bash
# Select at DG DOF locations (p=1)
python scripts/select_observation_points.py \
    --mesh-type tidal \
    --solver-type DG \
    --p-degree 1 \
    --nx 20 \
    --ny 5

# Output: data/tidal_dg_p1_obs_points.json
```

### Example 3: Shinnecock Inlet

```bash
# Mesh nodes
python scripts/select_observation_points.py --mesh-type shinnecock

# DG DOFs
python scripts/select_observation_points.py \
    --mesh-type shinnecock \
    --solver-type DG \
    --p-degree 1

# Output: data/shinnecock_dg_p1_obs_points.json
```

### Example 4: Custom Output File

```bash
python scripts/select_observation_points.py \
    --mesh-type tidal \
    --solver-type DG \
    -o my_custom_obs_points.json
```

## Output File Format

The tool saves a JSON file with the following structure:

```json
{
  "mesh_type": "tidal",
  "mesh_params": {
    "nx": 20,
    "ny": 5,
    "x0": 0.0,
    "x1": 10000.0,
    "y0": 0.0,
    "y1": 5000.0,
    "solver_type": "DG",
    "p_degree": 1
  },
  "n_points": 5,
  "point_type": "dof",
  "point_indices": [42, 108, 256, 389, 512],
  "node_indices": [42, 108, 256, 389, 512],
  "coordinates": [
    [1250.0, 1250.0, 0.0],
    [2500.0, 2500.0, 0.0],
    [5000.0, 1250.0, 0.0],
    [7500.0, 3750.0, 0.0],
    [8750.0, 1250.0, 0.0]
  ]
}
```

| Field | Description |
|-------|-------------|
| `mesh_type` | Type of mesh used |
| `mesh_params` | Parameters used to create/load the mesh |
| `n_points` | Number of selected points |
| `point_type` | `"node"` or `"dof"` |
| `point_indices` | Indices in the point array |
| `node_indices` | Same as point_indices (for backward compatibility) |
| `coordinates` | 3D coordinates `[x, y, z]` of each point |

## Using Selected Points in DA Experiments

After selecting observation points, use them in your DA experiments:

### Shinnecock Example

```bash
python examples/shinnecock.py \
    --da-mode 4dvar \
    --obs-points-file data/shinnecock_obs_points.json \
    --T 12 \
    --verbose
```

### Tidal 4D-Var Experiment

```bash
python experiments/serial_da/tidal_4dvar.py \
    --obs-points-file data/tidal_obs_points.json \
    --nx 20 \
    --ny 5 \
    --final-time 86400
```

### Tidal DC-WME Experiment

```bash
python experiments/serial_da/tidal_dcwme.py \
    --obs-points-file data/tidal_dg_p1_obs_points.json \
    --nx 20 \
    --ny 5
```

### Parallel Experiments

```bash
mpirun -n 4 python experiments/parallel_da/tidal_4dvar_mpi.py \
    --obs-points-file data/tidal_obs_points.json
```

## Node vs DOF Selection

### When to Use Node Selection (Default)

- **CG or SUPG solvers**: DOFs are located at mesh nodes
- **General purpose**: Works for any discretization (interpolation is used)
- **Simpler visualization**: Fewer points to choose from

### When to Use DOF Selection

- **DG solvers**: DOFs are at element interiors, not mesh nodes
- **Exact observations**: Observe directly at DOF locations without interpolation
- **DG-specific experiments**: When you need observations at DG quadrature points

### Comparison

| Discretization | Mesh Nodes | DOFs (p=1) | DOFs (p=2) |
|----------------|------------|------------|------------|
| Tidal 20x5 | 126 | 2,400 | 6,000 |
| Shinnecock | ~5,000 | ~30,000 | ~90,000 |

For DG, the number of DOFs = (number of elements) × (DOFs per element) × (3 components).

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  OBSERVATION SELECTION WORKFLOW              │
└─────────────────────────────────────────────────────────────┘

Step 1: Run selection tool
        ┌─────────────────────────────────────┐
        │  python select_observation_points.py │
        │  --mesh-type tidal --solver-type DG  │
        └─────────────────┬───────────────────┘
                          │
                          ▼
Step 2: Interactive selection window opens
        ┌─────────────────────────────────────┐
        │  ┌───────────────────────────────┐  │
        │  │  · · · · · · · · · · · · · ·  │  │
        │  │  · · ● · · · · ● · · · · · ·  │  │
        │  │  · · · · · · · · · · ● · · ·  │  │
        │  │  · · · · ● · · · · · · · · ·  │  │
        │  │  · · · · · · · · · ● · · · ·  │  │
        │  └───────────────────────────────┘  │
        │  Left-click to select points        │
        │  Press Enter when done              │
        └─────────────────┬───────────────────┘
                          │
                          ▼
Step 3: Points saved to JSON file
        ┌─────────────────────────────────────┐
        │  data/tidal_dg_p1_obs_points.json   │
        │  {                                   │
        │    "n_points": 5,                    │
        │    "coordinates": [[...], ...]       │
        │  }                                   │
        └─────────────────┬───────────────────┘
                          │
                          ▼
Step 4: Use in DA experiment
        ┌─────────────────────────────────────┐
        │  python tidal_4dvar.py              │
        │  --obs-points-file <json_file>      │
        └─────────────────────────────────────┘
```

## Troubleshooting

### "No module named 'dolfinx'"

The tool requires DOLFINx for mesh loading. Ensure you have the swe4dvar environment activated:

```bash
conda activate swe4dvar
```

### "Error loading mesh"

Check that the data files exist:

```bash
# For Shinnecock
ls data/shinnecock_inlet_mesh.bp

# For Inlet
ls data/Ideal_Inlet/Ideal_Inlet.xdmf
```

### Window doesn't appear

If using SSH, ensure X11 forwarding is enabled:

```bash
ssh -X user@host
```

Or use a different matplotlib backend:

```bash
export MPLBACKEND=TkAgg
python scripts/select_observation_points.py --mesh-type tidal
```

### Too many DOFs to select from

For large DG meshes, consider:

1. Using a coarser mesh for point selection
2. Using node selection and letting the observation operator interpolate
3. Programmatically generating observation points instead

## See Also

- [Twin Experiment Workflow](twin_experiment_workflow.md) - Complete DA experiment documentation
- [API Reference](api_reference.md) - PointObservationOperator documentation

#!/usr/bin/env python3
"""Check observation locations vs mesh nodes."""

import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from da_experiment_utils import generate_observation_points
from swe4dvar.data_assimilation import PointObservationOperator


def check_obs_locations():
    """Check observation locations."""
    comm = MPI.COMM_WORLD

    nx, ny = 4, 2
    dt = 3600.0
    nt = 2

    problem = TidalProblem(nx=nx, ny=ny, dt=dt, nt=nt)
    solver = get_solver("CG")(problem, theta=0.5, p_degree=[1, 1])

    # Get mesh vertices
    mesh = problem.mesh
    geom = mesh.geometry.x
    print(f"Mesh vertices: {len(geom)}")
    print(f"  Coordinates shape: {geom.shape}")

    # Get observation points
    obs_points = generate_observation_points(mesh, fraction=0.5)
    print(f"\nObservation points: {len(obs_points)}")
    for i, pt in enumerate(obs_points):
        print(f"  Obs {i}: ({pt[0]:.4f}, {pt[1]:.4f})")

    # Check which mesh vertices are observation points
    print("\nMesh vertices:")
    for i, v in enumerate(geom):
        is_obs = any(np.allclose(v[:2], pt[:2], atol=1e-10) for pt in obs_points)
        status = "OBS" if is_obs else ""
        print(f"  Node {i}: ({v[0]:.4f}, {v[1]:.4f}) {status}")

    # Get function space DOF coordinates per sub-space
    V = solver.V
    print(f"\nFunction space: {V.num_sub_spaces} sub-spaces")

    # Try to understand DOF ordering
    # For a mixed space (P1, P1×2), DOFs should be ordered by node
    print("\nInferred DOF mapping (assuming node-wise interleaving):")
    n_nodes = 15  # P1 on 5×3 = 15 nodes
    n_comp = 3  # h, uh, vh
    for node in range(n_nodes):
        coords = geom[node] if node < len(geom) else [0, 0]
        is_obs = any(np.allclose(coords[:2], pt[:2], atol=1e-10) for pt in obs_points)
        obs_str = " [OBS]" if is_obs else ""
        dofs = [node * n_comp + c for c in range(n_comp)]
        print(f"  Node {node} at ({coords[0]:.4f}, {coords[1]:.4f}){obs_str}: DOFs {dofs}")


if __name__ == "__main__":
    check_obs_locations()

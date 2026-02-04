#!/usr/bin/env python3
"""Check DOF ordering for SWE function space."""

import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver


def check_dof_ordering():
    """Check DOF ordering."""
    comm = MPI.COMM_WORLD

    # Create problem
    nx, ny = 4, 2
    dt = 3600.0
    num_time_steps = 2

    problem = TidalProblem(nx=nx, ny=ny, dt=dt, nt=num_time_steps)
    solver = get_solver("CG")(problem, theta=0.5, p_degree=[1, 1])

    # Get function space info
    V = solver.V
    print(f"Function space: {V}")
    print(f"Number of DOFs: {V.dofmap.index_map.size_global}")

    # Get sub-spaces
    num_sub = V.num_sub_spaces
    print(f"Number of sub-spaces: {num_sub}")

    for i in range(num_sub):
        sub = V.sub(i)
        sub_collapsed, _ = sub.collapse()
        print(f"  Sub-space {i}: {sub_collapsed.dofmap.index_map.size_global} DOFs")

    # Check DOF ownership
    print(f"\nLocal DOF range: {V.dofmap.index_map.local_range}")

    # Get DOF coordinates
    mesh = problem.mesh
    dof_coords = V.tabulate_dof_coordinates()
    print(f"\nDOF coordinates shape: {dof_coords.shape}")

    # Show first 15 DOFs with their coordinates
    print(f"\nFirst 15 DOFs:")
    print(f"{'DOF':>4}  {'x':>10}  {'y':>10}  {'sub':>4}")
    for i in range(min(15, len(dof_coords))):
        # Determine which sub-space this DOF belongs to
        # For a mixed space, DOFs are interleaved
        sub_idx = i % num_sub
        print(f"{i:4d}  {dof_coords[i,0]:10.4f}  {dof_coords[i,1]:10.4f}  {sub_idx:4d}")


if __name__ == "__main__":
    check_dof_ordering()

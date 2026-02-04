#!/usr/bin/env python3
"""Check actual DOF structure for mixed space."""

import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver


def check_dof_structure():
    """Check DOF structure."""
    comm = MPI.COMM_WORLD

    nx, ny = 4, 2
    dt = 3600.0
    nt = 2

    problem = TidalProblem(nx=nx, ny=ny, dt=dt, nt=nt)
    solver = get_solver("CG")(problem, theta=1.0, p_degree=[1, 1])

    V = solver.V
    mesh = problem.mesh

    print("=" * 60)
    print("MIXED FUNCTION SPACE DOF STRUCTURE")
    print("=" * 60)
    print(f"Function space: {V}")
    print(f"Total DOFs: {V.dofmap.index_map.size_global}")
    print(f"Block size: {V.dofmap.index_map_bs}")

    # Get sub-spaces
    print(f"\nNumber of sub-spaces: {V.num_sub_spaces}")
    for i in range(V.num_sub_spaces):
        sub = V.sub(i)
        sub_collapsed, sub_to_parent = sub.collapse()
        print(f"\nSub-space {i}:")
        print(f"  DOFs: {sub_collapsed.dofmap.index_map.size_global}")
        print(f"  Block size: {sub_collapsed.dofmap.index_map_bs}")
        print(f"  Sub-to-parent map (first 20): {sub_to_parent[:20]}")

    # Check DOF coordinates
    print("\n" + "=" * 60)
    print("DOF COORDINATES BY SUB-SPACE")
    print("=" * 60)

    # Get h sub-space DOF coords
    sub0 = V.sub(0)
    sub0_collapsed, sub0_to_parent = sub0.collapse()
    coords0 = sub0_collapsed.tabulate_dof_coordinates()
    print(f"\nSub-space 0 (h) - {len(coords0)} DOFs:")
    for i, c in enumerate(coords0[:15]):
        parent_dof = sub0_to_parent[i]
        print(f"  h DOF {i} at ({c[0]:.0f}, {c[1]:.0f}) -> parent DOF {parent_dof}")

    # Get momentum sub-space DOF coords
    sub1 = V.sub(1)
    sub1_collapsed, sub1_to_parent = sub1.collapse()
    coords1 = sub1_collapsed.tabulate_dof_coordinates()
    print(f"\nSub-space 1 (uh,vh) - {len(coords1)} DOFs (block size {sub1_collapsed.dofmap.index_map_bs}):")
    for i, c in enumerate(coords1[:15]):
        parent_dof = sub1_to_parent[i]
        comp = i % sub1_collapsed.dofmap.index_map_bs if sub1_collapsed.dofmap.index_map_bs > 1 else 0
        var = "uh" if comp == 0 else "vh"
        print(f"  {var} DOF {i} at ({c[0]:.0f}, {c[1]:.0f}) -> parent DOF {parent_dof}")


if __name__ == "__main__":
    check_dof_structure()

"""Minimal dolfinx proof-of-life — tiny mesh, one Poisson solve, zero optimization.

Goal: confirm that the full stack (mpi4py -> petsc4py -> dolfinx) loads and
solves a trivial problem. ~1 second on 1 rank, ~1 second on 4 ranks.

Run:
    python test_dolfinx.py            # serial
    ibrun -n 4 python test_dolfinx.py # 4 ranks (works on login too — < 1 sec)
"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl

comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    print(f"dolfinx  {dolfinx.__version__}")
    print(f"petsc4py {PETSc.Sys.getVersion()}")
    print(f"mpi4py   ranks={comm.size}")

# Tiny 8x8 unit square
domain = mesh.create_unit_square(comm, 8, 8, mesh.CellType.triangle)

V = fem.functionspace(domain, ("Lagrange", 1))

# Dirichlet BC: u=0 on boundary
boundary_facets = mesh.locate_entities_boundary(
    domain, dim=domain.topology.dim - 1,
    marker=lambda x: np.full(x.shape[1], True, dtype=bool),
)
boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(1.0))
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem = LinearProblem(a, L, bcs=[bc],
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Small check: interior max should be positive, boundary zero.
local_max = uh.x.array.max()
global_max = comm.allreduce(local_max, op=MPI.MAX)

if rank == 0:
    print(f"Poisson solved. global max(u) = {global_max:.6f}")
    print("OK" if global_max > 0 else "FAIL (solution is zero everywhere)")

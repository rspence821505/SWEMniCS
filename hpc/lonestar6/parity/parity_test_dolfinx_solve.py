"""parity_test_dolfinx_solve.py — tiny deterministic Poisson solve.

8x8 unit square, Dirichlet u=0, f=1 forcing, LU direct solve. Single-rank
by default. Emits JSON with mesh stats and solution norms.

This probes the full dolfinx -> PETSc -> MPI -> BLAS chain. Numerical
parity here is the tightest we'll get; any mismatch at this level points
to a real ABI/linker drift rather than an experiment-level problem.

Version-agnostic: works on dolfinx 0.9 and 0.10.

Usage:
    python parity_test_dolfinx_solve.py > dolfinx.json
    mpiexec -n 2 python parity_test_dolfinx_solve.py    # also valid
"""
import json
import math

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, mesh
import ufl

try:
    from dolfinx.fem.petsc import LinearProblem
except ImportError:
    from dolfinx.fem import LinearProblem  # pre-0.8 fallback (unlikely)

comm = MPI.COMM_WORLD
rank = comm.rank

NX, NY = 8, 8
domain = mesh.create_unit_square(comm, NX, NY, mesh.CellType.triangle)

V = fem.functionspace(domain, ("Lagrange", 1))

tdim = domain.topology.dim
num_cells_local = domain.topology.index_map(tdim).size_local
num_cells_global = domain.topology.index_map(tdim).size_global
num_vertices_global = domain.topology.index_map(0).size_global
num_dofs_global = V.dofmap.index_map.size_global

# Dirichlet BC
boundary_facets = mesh.locate_entities_boundary(
    domain, dim=tdim - 1,
    marker=lambda x: np.full(x.shape[1], True, dtype=bool),
)
boundary_dofs = fem.locate_dofs_topological(V, tdim - 1, boundary_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(1.0))
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Build LinearProblem across versions (0.10 added petsc_options_prefix)
petsc_opts = {"ksp_type": "preonly", "pc_type": "lu"}
try:
    problem = LinearProblem(a, L, bcs=[bc],
                            petsc_options_prefix="parity_dolfinx_",
                            petsc_options=petsc_opts)
except TypeError:
    problem = LinearProblem(a, L, bcs=[bc], petsc_options=petsc_opts)

solved = problem.solve()
uh = solved[0] if isinstance(solved, tuple) else solved

arr = uh.x.array
local_l2 = float(np.sqrt(np.sum(arr * arr)))
local_linf = float(np.max(np.abs(arr)))
global_l2sq = comm.allreduce(local_l2 ** 2, op=MPI.SUM)
global_l2 = math.sqrt(global_l2sq)
global_linf = comm.allreduce(local_linf, op=MPI.MAX)

# L2 norm of residual r = A uh - b (matrix-free assemble)
A_mat = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
A_mat.assemble()
b_vec = fem.petsc.assemble_vector(fem.form(L))
# Apply lifting + bc
try:
    fem.petsc.apply_lifting(b_vec, [fem.form(a)], bcs=[[bc]])
    b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b_vec, [bc])
except Exception:
    pass

b_norm = b_vec.norm(PETSc.NormType.NORM_2)

x_vec = b_vec.duplicate()
uh.x.scatter_forward()
with uh.x.petsc_vec.localForm() as _loc, x_vec.localForm() as x_loc:
    x_loc.copy(_loc)
r = b_vec.duplicate()
A_mat.mult(x_vec, r)
r.axpy(-1.0, b_vec)
r_norm = r.norm(PETSc.NormType.NORM_2)

if rank == 0:
    out = {
        "dolfinx_version": dolfinx.__version__,
        "mesh_nx": NX,
        "mesh_ny": NY,
        "num_cells_global": int(num_cells_global),
        "num_vertices_global": int(num_vertices_global),
        "num_dofs_global": int(num_dofs_global),
        "mpi_size": comm.size,
        "u_l2_global": global_l2,
        "u_linf_global": global_linf,
        "b_l2": float(b_norm),
        "residual_l2": float(r_norm),
        "residual_rel": float(r_norm / b_norm) if b_norm > 0 else None,
    }
    print(json.dumps(out, indent=2, sort_keys=True))

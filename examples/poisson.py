from dolfinx import fem, mesh, la

from ufl import TrialFunction, TestFunction, dx, grad, inner, SpatialCoordinate
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem as fe
from basix.ufl import element
from dolfinx.fem import functionspace
from ufl import sin, pi  # Add this import at the top if not already
import dolfinx.fem.petsc as fe_petsc
import matplotlib.pyplot as plt
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Global parameters
alpha = 1e-4  # regularization parameter


def create_element(mesh: mesh.Mesh, family: str, degree: int, shape: tuple[int] = ()):
    """Compatible element creation for UFL and Basix.

    Args:
        mesh: _description_
        family: _description_
        degree: _description_
        shape: _description_. Defaults to ().

    Returns:
        _description_
    """
    try:
        from ufl import FiniteElement, VectorElement

        if shape == ():
            return FiniteElement(family, mesh.ufl_cell(), degree)
        else:
            assert len(shape) == 1
            return VectorElement(family, mesh.ufl_cell(), degree, dim=shape[0])
    except ImportError:
        from basix.ufl import element

        return element(family, mesh.basix_cell(), degree, shape=shape)


def solve_forward_poisson(u_control, V):
    """
    Solve -Δy = u in Ω with y = 0 on ∂Ω given a control u.
    Parameters:
    u_control : fem.Function - the control function (right-hand side)
    V : fem.FunctionSpace - function space for the state y
    Returns:
    y : fem.Function - solution to the state equation
    """
    # Trial and test functions
    y = TrialFunction(V)
    v = TestFunction(V)

    # Bilinear and linear forms
    a = inner(grad(y), grad(v)) * dx
    L = u_control * v * dx

    # Dirichlet BC: y = 0 on ∂Ω
    boundary_dofs = fem.locate_dofs_geometrical(V, lambda x: np.full(x.shape[1], True))
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)

    # Assemble and solve
    A = fe_petsc.assemble_matrix(fem.form(a), bcs=[bc])
    A.assemble()
    b = fe_petsc.assemble_vector(fem.form(L))
    fe_petsc.apply_lifting(b, [fem.form(a)], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fe_petsc.set_bc(b, [bc])

    # Solve linear system
    y_sol = fem.Function(V)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setFromOptions()
    solver.solve(b, y_sol.x.petsc_vec)
    y_sol.x.scatter_forward()

    return y_sol


def solve_adjoint(y: fem.Function, y_d: fem.Function, V):
    """Solve -Δp = y - y_d in Ω, p = 0 on ∂Ω."""
    p = TrialFunction(V)
    v = TestFunction(V)
    r = fem.Function(V)
    r.x.array[:] = y.x.array - y_d.x.array

    a = inner(grad(p), grad(v)) * dx
    L = inner(r, v) * dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = fe_petsc.assemble_matrix(a_form)
    A.assemble()
    b = fe_petsc.assemble_vector(L_form)

    dofs = fem.locate_dofs_geometrical(
        V,
        lambda x: np.isclose(x[0], 0)
        | np.isclose(x[0], 1)
        | np.isclose(x[1], 0)
        | np.isclose(x[1], 1),
    )
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    fe_petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fe_petsc.set_bc(b, [bc])

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setFromOptions()
    # create solution vector
    p_sol = fem.Function(V)
    solver.solve(b, p_sol.x.petsc_vec)
    p_sol.x.scatter_forward()
    return p_sol


def cost_functional(tao, u_vec):
    # Step 1: Update the FEniCSx Function u from u_vec
    u_array = u.x.petsc_vec  # Read-write view
    u_array[:] = u_vec.getArray(readonly=True)  # Safe copy
    u.x.petsc_vec.setArray(u_array)
    u.x.scatter_forward()

    # Step 2: Solve forward Poisson problem
    y = solve_forward_poisson(u, V)

    # Step 3: Compute cost functional
    diff = fem.Function(V)
    diff.interpolate(lambda x: y.x.array - y_d.x.array)

    J_data = fem.assemble_scalar(fem.form(0.5 * diff * diff * dx))
    J_reg = fem.assemble_scalar(fem.form(0.5 * alpha * u * u * dx))
    J_total = J_data + J_reg

    return J_total


def gradient_functional(tao, u_vec, g_vec):
    # u.x.array[:] = u_vec.array
    u.x.array[:] = u_vec.getArray(readonly=True)
    y_sol = solve_forward_poisson(u, V)
    p_sol = solve_adjoint(y_sol, y_d, V)
    g_vec.array[:] = -p_sol.x.array
    g_vec.assemble()


def main():
    global V, u, y_d

    domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)

    # Define scalar Lagrange element of degree 1
    f_element = element("Lagrange", domain.basix_cell(), 1)
    V = functionspace(domain, f_element)

    # Desired state: y_d = sin(pi x) sin(pi y)
    x = SpatialCoordinate(domain)

    y_d_expr = fem.Expression(
        sin(pi * x[0]) * sin(pi * x[1]), V.element.interpolation_points()
    )

    y_d = fem.Function(V)
    y_d.interpolate(y_d_expr)

    # Initial guess for control: zero
    u = fem.Function(V)
    u.x.array[:] = 0.0

    # Setup PETSc vector for optimization
    u_vec = u.x.petsc_vec.copy()
    # u_vec = u.vector.copy()
    u_vec.set(0.0)

    tao = PETSc.TAO().create(comm)
    tao.setType("lmvm")
    tao.setObjective(cost_functional)
    tao.setGradient(gradient_functional)
    tao.setSolution(u_vec)
    tao.setFromOptions()

    if comm.rank == 0:
        print("Solving PDE-constrained optimization using TAO...\n")
    tao.solve()

    u.x.array[:] = u_vec.array
    u.x.scatter_forward()

    # Solve and plot final state
    y_sol = solve_forward_poisson(u, V)
    coords = y_sol.function_space.mesh.geometry.x
    values = y_sol.x.array

    if comm.rank == 0:
        # Simple imshow plot
        from matplotlib.tri import Triangulation

        cells = V.mesh.topology.connectivity(2, 0).array.reshape(-1, 3)
        triang = Triangulation(coords[:, 0], coords[:, 1], cells)
        plt.figure(figsize=(6, 5))
        plt.tricontourf(triang, values, cmap="viridis")
        plt.colorbar(label="Optimal state y")
        plt.title("Optimal state from Poisson control")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

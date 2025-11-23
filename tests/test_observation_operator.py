"""
Unit tests for observation operator implementations.

Tests cover both Continuous Galerkin (CG) and Discontinuous Galerkin (DG)
function spaces with comprehensive validation of:
1. Point location in distributed meshes
2. Forward operator correctness
3. Adjoint consistency: ⟨Hu, w⟩ = ⟨u, H^Tw⟩
4. MPI determinism
5. Component selection for mixed spaces
6. DG-specific averaging behaviors

Run with:
    pytest test_observation.py -v
    mpirun -np 4 pytest test_observation.py -v  # Parallel tests
"""

import pytest
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
import dolfinx
import dolfinx.mesh
from dolfinx import fem
import ufl


# Import from the module we're testing
try:
    from swemnics.data_assimilation.observation_operator import (
        PointObservationOperator,
        is_discontinuous_space,
    )
except ImportError:
    # If running from examples/tests directory
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from swemnics.data_assimilation.observation_operator import (
        PointObservationOperator,
        is_discontinuous_space,
    )


@pytest.fixture
def comm():
    """MPI communicator fixture."""
    return MPI.COMM_WORLD


@pytest.fixture
def simple_mesh_2d(comm):
    """Create a simple 2D unit square mesh."""
    return dolfinx.mesh.create_unit_square(
        comm, nx=10, ny=10, cell_type=dolfinx.mesh.CellType.triangle
    )


@pytest.fixture
def simple_mesh_2d_distributed(comm):
    """Create a 2D mesh that will be distributed across ranks."""
    return dolfinx.mesh.create_unit_square(
        comm, nx=20, ny=20, cell_type=dolfinx.mesh.CellType.triangle
    )


@pytest.fixture
def observation_points_interior():
    """Observation points guaranteed to be in unit square."""
    return np.array(
        [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
            [0.5, 0.5],
        ]
    )


@pytest.fixture
def observation_points_grid():
    """Grid of observation points."""
    x = np.linspace(0.1, 0.9, 5)
    y = np.linspace(0.1, 0.9, 5)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


class TestSpaceDetection:
    """Tests for CG/DG space detection."""

    def test_detect_cg_space(self, simple_mesh_2d):
        """Test that CG spaces are correctly identified."""
        V_cg = fem.functionspace(simple_mesh_2d, ("Lagrange", 1))
        assert not is_discontinuous_space(V_cg)

    def test_detect_dg_space(self, simple_mesh_2d):
        """Test that DG spaces are correctly identified."""
        V_dg = fem.functionspace(simple_mesh_2d, ("Discontinuous Lagrange", 1))
        assert is_discontinuous_space(V_dg)

    def test_detect_dg_alternative_name(self, simple_mesh_2d):
        """Test DG detection with alternative family name."""
        # Some versions may use "DG" directly
        V_dg = fem.functionspace(simple_mesh_2d, ("DG", 1))
        assert is_discontinuous_space(V_dg)


class TestPointObservationOperatorCG:
    """Tests for PointObservationOperator with CG spaces."""

    def test_construction_cg_scalar(
        self, simple_mesh_2d, observation_points_interior, comm
    ):
        """Test construction with CG scalar function space."""
        V_cg = fem.functionspace(simple_mesh_2d, ("Lagrange", 1))

        obs_op = PointObservationOperator(V_cg, observation_points_interior, comm=comm)

        assert obs_op.get_num_observations() == len(observation_points_interior)
        assert obs_op.comm.size == comm.size
        assert not obs_op.is_dg  # Should be detected as CG

    def test_construction_cg_vector(
        self, simple_mesh_2d, observation_points_interior, comm
    ):
        """Test construction with CG vector function space."""
        V_cg_vec = fem.functionspace(simple_mesh_2d, ("Lagrange", 1, (2,)))

        obs_op = PointObservationOperator(
            V_cg_vec, observation_points_interior, component_indices=[0], comm=comm
        )

        assert obs_op.get_num_observations() == len(observation_points_interior)
        assert not obs_op.is_dg

    def test_forward_cg_constant(
        self, simple_mesh_2d, observation_points_interior, comm
    ):
        """Test CG forward operator with constant function."""
        V_cg = fem.functionspace(simple_mesh_2d, ("Lagrange", 1))
        obs_op = PointObservationOperator(V_cg, observation_points_interior, comm=comm)

        # Create constant function u = 2.5
        u = fem.Function(V_cg)
        u.x.array[:] = 2.5

        # Apply observation operator
        obs = obs_op.forward(u.x.petsc_vec)

        # Check that all observations equal 2.5
        obs_array = obs.getArray()
        assert np.allclose(obs_array, 2.5, rtol=1e-10)

    def test_forward_cg_linear(self, simple_mesh_2d, observation_points_interior, comm):
        """Test CG forward operator with linear function u = x + y."""
        V_cg = fem.functionspace(simple_mesh_2d, ("Lagrange", 1))
        obs_op = PointObservationOperator(V_cg, observation_points_interior, comm=comm)

        # Create linear function u = x + y
        u = fem.Function(V_cg)
        x_coords = ufl.SpatialCoordinate(simple_mesh_2d)
        u.interpolate(
            fem.Expression(
                x_coords[0] + x_coords[1], V_cg.element.interpolation_points()
            )
        )

        # Apply observation operator
        obs = obs_op.forward(u.x.petsc_vec)

        # Check values: u(x,y) = x + y
        obs_array = obs.getArray()
        expected = observation_points_interior[:, 0] + observation_points_interior[:, 1]
        assert np.allclose(obs_array, expected, rtol=1e-6)

    def test_adjoint_consistency_cg(
        self, simple_mesh_2d, observation_points_interior, comm
    ):
        """
        Test CG adjoint consistency: ⟨Hu, w⟩ = ⟨u, H^Tw⟩

        This is the critical test for adjoint correctness.
        """
        V_cg = fem.functionspace(simple_mesh_2d, ("Lagrange", 1))
        obs_op = PointObservationOperator(V_cg, observation_points_interior, comm=comm)

        # Create random state vector u
        u = fem.Function(V_cg)
        u.x.array[:] = np.random.randn(len(u.x.array))
        u.x.scatter_forward()

        # Create random observation vector w
        w = PETSc.Vec().createSeq(obs_op.n_obs, comm=PETSc.COMM_SELF)
        w.setArray(np.random.randn(obs_op.n_obs))
        w.assemble()

        # Compute LHS: ⟨Hu, w⟩
        Hu = obs_op.forward(u.x.petsc_vec)
        lhs = Hu.dot(w)

        # Compute RHS: ⟨u, H^Tw⟩
        HTw = obs_op.adjoint(w)
        rhs = u.x.petsc_vec.dot(HTw)

        # Check consistency
        rel_error = abs(lhs - rhs) / (abs(lhs) + 1e-16)

        if comm.rank == 0:
            print(f"\nCG Adjoint consistency test:")
            print(f"  LHS (⟨Hu, w⟩) = {lhs:.10e}")
            print(f"  RHS (⟨u, H^Tw⟩) = {rhs:.10e}")
            print(f"  Relative error = {rel_error:.10e}")

        assert (
            rel_error < 1e-10
        ), f"CG adjoint consistency failed: {rel_error:.2e} > 1e-10"


class TestPointObservationOperatorDG:
    """Tests for PointObservationOperator with DG spaces."""

    def test_construction_dg_scalar(
        self, simple_mesh_2d, observation_points_interior, comm
    ):
        """Test construction with DG scalar function space."""
        V_dg = fem.functionspace(simple_mesh_2d, ("Discontinuous Lagrange", 1))

        obs_op = PointObservationOperator(V_dg, observation_points_interior, comm=comm)

        assert obs_op.get_num_observations() == len(observation_points_interior)
        assert obs_op.is_dg  # Should be detected as DG

    def test_forward_dg_constant(
        self, simple_mesh_2d, observation_points_interior, comm
    ):
        """Test DG forward operator with constant function."""
        V_dg = fem.functionspace(simple_mesh_2d, ("Discontinuous Lagrange", 1))
        obs_op = PointObservationOperator(V_dg, observation_points_interior, comm=comm)

        # Create constant function u = 3.7
        u = fem.Function(V_dg)
        u.x.array[:] = 3.7

        # Apply observation operator
        obs = obs_op.forward(u.x.petsc_vec)

        # Check that all observations equal 3.7 (even with averaging)
        obs_array = obs.getArray()
        assert np.allclose(obs_array, 3.7, rtol=1e-10)

    def test_forward_dg_linear(self, simple_mesh_2d, observation_points_interior, comm):
        """Test DG forward operator with linear function."""
        V_dg = fem.functionspace(simple_mesh_2d, ("Discontinuous Lagrange", 1))
        obs_op = PointObservationOperator(V_dg, observation_points_interior, comm=comm)

        # Create linear function u = x + y
        u = fem.Function(V_dg)
        x_coords = ufl.SpatialCoordinate(simple_mesh_2d)
        u.interpolate(
            fem.Expression(
                x_coords[0] + x_coords[1], V_dg.element.interpolation_points()
            )
        )

        # Apply observation operator
        obs = obs_op.forward(u.x.petsc_vec)

        # Check values (with some tolerance for averaging at boundaries)
        obs_array = obs.getArray()
        expected = observation_points_interior[:, 0] + observation_points_interior[:, 1]
        assert np.allclose(obs_array, expected, rtol=1e-5)

    def test_adjoint_consistency_dg_arithmetic(
        self, simple_mesh_2d, observation_points_interior, comm
    ):
        """
        Test DG adjoint consistency with arithmetic averaging.

        Critical test: ⟨Hu, w⟩ = ⟨u, H^Tw⟩ must hold for DG.
        """
        V_dg = fem.functionspace(simple_mesh_2d, ("Discontinuous Lagrange", 1))
        obs_op = PointObservationOperator(
            V_dg, observation_points_interior, dg_averaging="arithmetic", comm=comm
        )

        # Create random state vector u
        u = fem.Function(V_dg)
        u.x.array[:] = np.random.randn(len(u.x.array))
        u.x.scatter_forward()

        # Create random observation vector w
        w = PETSc.Vec().createSeq(obs_op.n_obs, comm=PETSc.COMM_SELF)
        w.setArray(np.random.randn(obs_op.n_obs))
        w.assemble()

        # Compute LHS: ⟨Hu, w⟩
        Hu = obs_op.forward(u.x.petsc_vec)
        lhs = Hu.dot(w)

        # Compute RHS: ⟨u, H^Tw⟩
        HTw = obs_op.adjoint(w)
        rhs = u.x.petsc_vec.dot(HTw)

        # Check consistency
        rel_error = abs(lhs - rhs) / (abs(lhs) + 1e-16)

        if comm.rank == 0:
            print(f"\nDG Adjoint consistency test (arithmetic):")
            print(f"  LHS (⟨Hu, w⟩) = {lhs:.10e}")
            print(f"  RHS (⟨u, H^Tw⟩) = {rhs:.10e}")
            print(f"  Relative error = {rel_error:.10e}")

        assert (
            rel_error < 1e-9
        ), f"DG adjoint consistency failed: {rel_error:.2e} > 1e-9"

    def test_adjoint_consistency_dg_volume_weighted(
        self, simple_mesh_2d, observation_points_interior, comm
    ):
        """Test DG adjoint consistency with volume-weighted averaging."""
        V_dg = fem.functionspace(simple_mesh_2d, ("Discontinuous Lagrange", 1))
        obs_op = PointObservationOperator(
            V_dg, observation_points_interior, dg_averaging="volume_weighted", comm=comm
        )

        # Random test vectors
        u = fem.Function(V_dg)
        u.x.array[:] = np.random.randn(len(u.x.array))
        u.x.scatter_forward()

        w = PETSc.Vec().createSeq(obs_op.n_obs, comm=PETSc.COMM_SELF)
        w.setArray(np.random.randn(obs_op.n_obs))
        w.assemble()

        # Test consistency
        Hu = obs_op.forward(u.x.petsc_vec)
        HTw = obs_op.adjoint(w)

        lhs = Hu.dot(w)
        rhs = u.x.petsc_vec.dot(HTw)

        rel_error = abs(lhs - rhs) / (abs(lhs) + 1e-16)

        if comm.rank == 0:
            print(f"\nDG Adjoint consistency test (volume-weighted):")
            print(f"  Relative error = {rel_error:.10e}")

        assert rel_error < 1e-9

    def test_dg_averaging_modes(self, simple_mesh_2d, comm):
        """Test that different DG averaging modes produce reasonable results."""
        V_dg = fem.functionspace(simple_mesh_2d, ("Discontinuous Lagrange", 1))

        # Point on element boundary (will have multiple values in DG)
        boundary_point = np.array([[0.5, 0.5]])

        # Create discontinuous function
        u = fem.Function(V_dg)
        x_coords = ufl.SpatialCoordinate(simple_mesh_2d)
        # Function that varies: u = x^2 + y^2
        u.interpolate(
            fem.Expression(
                x_coords[0] ** 2 + x_coords[1] ** 2, V_dg.element.interpolation_points()
            )
        )

        # Test arithmetic averaging
        obs_op_arith = PointObservationOperator(
            V_dg, boundary_point, dg_averaging="arithmetic", comm=comm
        )
        y_arith = obs_op_arith.forward(u.x.petsc_vec).getArray()[0]

        # Test volume-weighted averaging
        obs_op_vol = PointObservationOperator(
            V_dg, boundary_point, dg_averaging="volume_weighted", comm=comm
        )
        y_vol = obs_op_vol.forward(u.x.petsc_vec).getArray()[0]

        # Both should be close to expected value at (0.5, 0.5): 0.25 + 0.25 = 0.5
        expected = 0.5

        if comm.rank == 0:
            print(f"\nDG averaging modes at boundary point (0.5, 0.5):")
            print(f"  Arithmetic:      {y_arith:.6f}")
            print(f"  Volume-weighted: {y_vol:.6f}")
            print(f"  Expected:        {expected:.6f}")

        assert np.isclose(y_arith, expected, rtol=1e-2)
        assert np.isclose(y_vol, expected, rtol=1e-2)


class TestCGvsDGComparison:
    """Direct comparison tests between CG and DG operators."""

    def test_smooth_function_cg_vs_dg(
        self, simple_mesh_2d, observation_points_interior, comm
    ):
        """
        For smooth functions, CG and DG should give similar results.

        DG may differ slightly due to averaging at boundaries.
        """
        V_cg = fem.functionspace(simple_mesh_2d, ("Lagrange", 1))
        V_dg = fem.functionspace(simple_mesh_2d, ("Discontinuous Lagrange", 1))

        obs_op_cg = PointObservationOperator(
            V_cg, observation_points_interior, comm=comm
        )
        obs_op_dg = PointObservationOperator(
            V_dg, observation_points_interior, comm=comm
        )

        # Create smooth function: u = sin(πx)sin(πy)
        x_coords = ufl.SpatialCoordinate(simple_mesh_2d)
        expr = ufl.sin(ufl.pi * x_coords[0]) * ufl.sin(ufl.pi * x_coords[1])

        u_cg = fem.Function(V_cg)
        u_cg.interpolate(fem.Expression(expr, V_cg.element.interpolation_points()))

        u_dg = fem.Function(V_dg)
        u_dg.interpolate(fem.Expression(expr, V_dg.element.interpolation_points()))

        # Apply operators
        y_cg = obs_op_cg.forward(u_cg.x.petsc_vec).getArray()
        y_dg = obs_op_dg.forward(u_dg.x.petsc_vec).getArray()

        # Should be very similar for smooth functions
        rel_diff = np.linalg.norm(y_cg - y_dg) / (np.linalg.norm(y_cg) + 1e-16)

        if comm.rank == 0:
            print(f"\nCG vs DG for smooth function:")
            print(f"  Relative difference: {rel_diff:.10e}")

        assert rel_diff < 0.05  # Within 5% for smooth functions

    def test_both_maintain_adjoint_consistency(
        self, simple_mesh_2d, observation_points_grid, comm
    ):
        """Verify both CG and DG maintain adjoint consistency."""
        V_cg = fem.functionspace(simple_mesh_2d, ("Lagrange", 1))
        V_dg = fem.functionspace(simple_mesh_2d, ("Discontinuous Lagrange", 1))

        for space_type, V in [("CG", V_cg), ("DG", V_dg)]:
            obs_op = PointObservationOperator(V, observation_points_grid, comm=comm)

            u = fem.Function(V)
            u.x.array[:] = np.random.randn(len(u.x.array))
            u.x.scatter_forward()

            w = PETSc.Vec().createSeq(obs_op.n_obs, comm=PETSc.COMM_SELF)
            w.setArray(np.random.randn(obs_op.n_obs))
            w.assemble()

            Hu = obs_op.forward(u.x.petsc_vec)
            HTw = obs_op.adjoint(w)

            lhs = Hu.dot(w)
            rhs = u.x.petsc_vec.dot(HTw)

            rel_error = abs(lhs - rhs) / (abs(lhs) + 1e-16)

            if comm.rank == 0:
                print(f"\n{space_type} adjoint consistency: {rel_error:.2e}")

            assert rel_error < 1e-9


class TestMixedDGCGSpace:
    """Tests for mixed DG-CG spaces (like SWEMniCS)."""

    def test_mixed_space_subspace_extraction(self, simple_mesh_2d, comm):
        """Test observation on subspaces of mixed DG-CG formulation."""
        # Create mixed space like SWEMniCS: (H_dg, u_cg, v_cg)
        from dolfinx.fem import functionspace

        # H: DG scalar
        V_H = functionspace(simple_mesh_2d, ("Discontinuous Lagrange", 1))

        # Velocity: CG vector
        V_vel = functionspace(simple_mesh_2d, ("Lagrange", 1, (2,)))

        points = np.array([[0.3, 0.3], [0.7, 0.7]])

        # Test H observation (DG)
        obs_op_H = PointObservationOperator(V_H, points, comm=comm)
        assert obs_op_H.is_dg

        H = fem.Function(V_H)
        H.x.array[:] = 5.0
        y_H = obs_op_H.forward(H.x.petsc_vec)
        assert np.allclose(y_H.getArray(), 5.0, rtol=1e-10)

        # Test velocity observation (CG)
        obs_op_u = PointObservationOperator(
            V_vel, points, component_indices=[0], comm=comm
        )
        assert not obs_op_u.is_dg

        vel = fem.Function(V_vel)
        vel.x.array[::2] = 2.0  # u-component
        vel.x.array[1::2] = 3.0  # v-component
        y_u = obs_op_u.forward(vel.x.petsc_vec)
        assert np.allclose(y_u.getArray(), 2.0, rtol=1e-6)


class TestLinearityProperties:
    """Test linearity of forward and adjoint operators."""

    @pytest.mark.parametrize("space_family", ["Lagrange", "Discontinuous Lagrange"])
    def test_forward_linearity(
        self, simple_mesh_2d, observation_points_interior, space_family, comm
    ):
        """Test that forward operator is linear: H(αu + βv) = αH(u) + βH(v)."""
        V = fem.functionspace(simple_mesh_2d, (space_family, 1))
        obs_op = PointObservationOperator(V, observation_points_interior, comm=comm)

        # Create two functions
        u = fem.Function(V)
        v = fem.Function(V)
        u.x.array[:] = np.random.randn(len(u.x.array))
        v.x.array[:] = np.random.randn(len(v.x.array))
        u.x.scatter_forward()
        v.x.scatter_forward()

        # Scalars
        alpha, beta = 2.5, -1.3

        # Compute H(αu + βv)
        w = u.x.petsc_vec.copy()
        w.scale(alpha)
        w.axpy(beta, v.x.petsc_vec)
        Hw = obs_op.forward(w)

        # Compute αH(u) + βH(v)
        Hu = obs_op.forward(u.x.petsc_vec)
        Hv = obs_op.forward(v.x.petsc_vec)
        expected = Hu.copy()
        expected.scale(alpha)
        expected.axpy(beta, Hv)

        # Compare
        Hw_array = Hw.getArray()
        expected_array = expected.getArray()
        assert np.allclose(Hw_array, expected_array, rtol=1e-12)

    @pytest.mark.parametrize("space_family", ["Lagrange", "Discontinuous Lagrange"])
    def test_adjoint_linearity(
        self, simple_mesh_2d, observation_points_interior, space_family, comm
    ):
        """Test that adjoint operator is linear: H^T(αw + βz) = αH^T(w) + βH^T(z)."""
        V = fem.functionspace(simple_mesh_2d, (space_family, 1))
        obs_op = PointObservationOperator(V, observation_points_interior, comm=comm)

        # Create two observation vectors
        w = PETSc.Vec().createSeq(obs_op.n_obs, comm=PETSc.COMM_SELF)
        z = PETSc.Vec().createSeq(obs_op.n_obs, comm=PETSc.COMM_SELF)
        w.setArray(np.random.randn(obs_op.n_obs))
        z.setArray(np.random.randn(obs_op.n_obs))
        w.assemble()
        z.assemble()

        # Scalars
        alpha, beta = 1.7, -2.1

        # Compute H^T(αw + βz)
        y = w.copy()
        y.scale(alpha)
        y.axpy(beta, z)
        HTy = obs_op.adjoint(y)

        # Compute αH^T(w) + βH^T(z)
        HTw = obs_op.adjoint(w)
        HTz = obs_op.adjoint(z)
        expected = HTw.copy()
        expected.scale(alpha)
        expected.axpy(beta, HTz)

        # Compare
        HTy.axpy(-1.0, expected)  # HTy -= expected
        error_norm = HTy.norm()
        expected_norm = expected.norm()

        rel_error = error_norm / (expected_norm + 1e-16)
        assert rel_error < 1e-12


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.parametrize("space_family", ["Lagrange", "Discontinuous Lagrange"])
    def test_single_observation_point(self, simple_mesh_2d, space_family, comm):
        """Test with single observation point."""
        V = fem.functionspace(simple_mesh_2d, (space_family, 1))
        obs_points = np.array([[0.5, 0.5]])

        obs_op = PointObservationOperator(V, obs_points, comm=comm)

        u = fem.Function(V)
        u.x.array[:] = 1.0

        obs = obs_op.forward(u.x.petsc_vec)
        assert obs.getSize() == 1
        assert np.allclose(obs.getArray(), [1.0])

    @pytest.mark.parametrize("space_family", ["Lagrange", "Discontinuous Lagrange"])
    def test_boundary_observation_points(self, simple_mesh_2d, space_family, comm):
        """Test with points on mesh boundary."""
        V = fem.functionspace(simple_mesh_2d, (space_family, 1))

        # Points on boundary of unit square
        boundary_points = np.array(
            [
                [0.0, 0.5],
                [1.0, 0.5],
                [0.5, 0.0],
                [0.5, 1.0],
            ]
        )

        obs_op = PointObservationOperator(V, boundary_points, comm=comm)

        # Should work without errors
        u = fem.Function(V)
        x_coords = ufl.SpatialCoordinate(simple_mesh_2d)
        u.interpolate(
            fem.Expression(x_coords[0] + x_coords[1], V.element.interpolation_points())
        )

        obs = obs_op.forward(u.x.petsc_vec)

        # Check boundary values
        obs_array = obs.getArray()
        expected = boundary_points[:, 0] + boundary_points[:, 1]
        assert np.allclose(obs_array, expected, rtol=1e-5)

    def test_points_outside_mesh_raises(self, simple_mesh_2d, comm):
        """Test that points outside mesh raise an error."""
        V = fem.functionspace(simple_mesh_2d, ("Lagrange", 1))

        bad_points = np.array(
            [
                [0.5, 0.5],  # Good point
                [1.5, 1.5],  # Outside mesh
            ]
        )

        with pytest.raises(RuntimeError, match="not found in mesh"):
            PointObservationOperator(V, bad_points, comm=comm)


class TestMPIDeterminism:
    """Test MPI determinism and parallel correctness."""

    @pytest.mark.parametrize("space_family", ["Lagrange", "Discontinuous Lagrange"])
    def test_forward_mpi_determinism(
        self, simple_mesh_2d_distributed, observation_points_grid, space_family, comm
    ):
        """Test that forward operator gives same result on different rank counts."""
        V = fem.functionspace(simple_mesh_2d_distributed, (space_family, 1))

        obs_op = PointObservationOperator(V, observation_points_grid, comm=comm)

        # Create deterministic function
        u = fem.Function(V)
        x_coords = ufl.SpatialCoordinate(simple_mesh_2d_distributed)
        u.interpolate(
            fem.Expression(
                ufl.sin(np.pi * x_coords[0]) * ufl.cos(np.pi * x_coords[1]),
                V.element.interpolation_points(),
            )
        )

        # Apply operator
        obs = obs_op.forward(u.x.petsc_vec)
        obs_array = obs.getArray()

        # Gather results on rank 0
        all_results = comm.gather(obs_array, root=0)

        if comm.rank == 0:
            # All ranks should give same result
            for i in range(1, len(all_results)):
                assert np.allclose(all_results[0], all_results[i], rtol=1e-12)


class TestPerformance:
    """Performance and scaling tests."""

    @pytest.mark.parametrize("space_family", ["Lagrange", "Discontinuous Lagrange"])
    def test_forward_operator_efficiency(
        self, simple_mesh_2d, observation_points_grid, space_family, comm, benchmark
    ):
        """Benchmark forward operator performance."""
        V = fem.functionspace(simple_mesh_2d, (space_family, 1))
        obs_op = PointObservationOperator(V, observation_points_grid, comm=comm)

        u = fem.Function(V)
        u.x.array[:] = np.random.randn(len(u.x.array))

        def run_forward():
            return obs_op.forward(u.x.petsc_vec)

        result = benchmark(run_forward)

        if comm.rank == 0:
            print(
                f"\nForward operator ({space_family}) with {len(observation_points_grid)} points"
            )


# Provide a no-op benchmark fixture if pytest-benchmark is unavailable
try:
    import pytest_benchmark.plugin  # type: ignore
except Exception:

    @pytest.fixture
    def benchmark():
        """Minimal stand-in for pytest-benchmark fixture."""

        def _run(func, *args, **kwargs):
            return func(*args, **kwargs)

        return _run


if __name__ == "__main__":
    # Run tests with: python test_observation.py
    pytest.main([__file__, "-v", "-s"])

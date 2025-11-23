"""Example usage of covariance matrix implementations.

This demonstrates how to use the different covariance matrix types
in data assimilation applications.
"""

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
import sys
from typing import Optional


from swemnics.data_assimilation.covariance import (
    DiagonalCovariance,
    DenseCovariance,
    ImplicitCovariance,
    PrecisionBasedCovariance,
    create_observation_covariance,
    create_background_covariance_from_ensemble,
    check_covariance_symmetry,
    check_inverse_consistency,
)


def _set_vec_from_global(vec: PETSc.Vec, global_values: np.ndarray) -> None:
    """Fill a distributed PETSc Vec with the global numpy array."""
    start, end = vec.getOwnershipRange()
    vec.setArray(np.array(global_values[start:end], dtype=float, copy=True))
    vec.assemble()


def _gather_vec(vec: PETSc.Vec, comm: MPI.Comm) -> Optional[np.ndarray]:
    """Gather a distributed PETSc Vec onto rank 0."""
    local = vec.getArray()
    gathered = comm.gather(local.copy(), root=0)
    if comm.rank == 0:
        return np.concatenate(gathered)
    return None


def example_observation_covariance():
    """Example 1: Create observation error covariance matrix."""
    print("\n" + "=" * 60)
    print("Example 1: Observation Error Covariance")
    print("=" * 60)

    comm = MPI.COMM_WORLD

    # Scenario: 20 tide gauge observations with σ = 0.15 m uncertainty
    n_gauges = 20
    gauge_std = 0.15  # meters

    R = create_observation_covariance(comm, n_gauges, gauge_std)

    print(f"Created observation covariance: {n_gauges} observations")
    print(f"Observation standard deviation: {gauge_std} m")
    print(f"Observation variance: {gauge_std**2} m²")

    # Test properties
    print(f"Matrix size: {R.size}")
    print(f"Symmetry test: {check_covariance_symmetry(R)}")
    print(f"Inverse consistency: {check_inverse_consistency(R)}")

    # Example usage: compute ⟨y-H(x), R⁻¹(y-H(x))⟩
    residual = R.create_vec()
    residual.setRandom()  # Simulated observation-prediction mismatch

    misfit = R.inner_product_inv(residual, residual)
    print(f"Weighted observation misfit: {misfit:.4f}")

    residual.destroy()


def example_background_covariance_diagonal():
    """Example 2: Simple diagonal background covariance."""
    print("\n" + "=" * 60)
    print("Example 2: Diagonal Background Covariance")
    print("=" * 60)

    comm = MPI.COMM_WORLD

    # Spatially varying background uncertainty
    n_dofs = 100

    # Larger uncertainty near boundaries, smaller in interior
    x = np.linspace(0, 1, n_dofs)
    spatial_variance = 0.5 + 0.5 * (x**2 + (1 - x) ** 2)  # U-shaped

    B = DiagonalCovariance(comm, size=n_dofs, diagonal=spatial_variance)

    print(f"Background covariance: {n_dofs} DoFs")
    print(
        f"Variance range: [{spatial_variance.min():.3f}, {spatial_variance.max():.3f}]"
    )

    # Sample from prior: x = x_b + B^(1/2)·ξ where ξ ~ N(0,I)
    xi = B.create_vec()
    xi.setRandom()  # Standard normal

    perturbation = B.sqrt_apply(xi)
    perturbation_norm = perturbation.norm()

    print(f"Prior sample perturbation norm: {perturbation_norm:.4f}")

    xi.destroy()
    perturbation.destroy()


def example_background_covariance_correlated():
    """Example 3: Background covariance with spatial correlation."""
    print("\n" + "=" * 60)
    print("Example 3: Correlated Background Covariance")
    print("=" * 60)

    comm = MPI.COMM_WORLD

    # Small example for clarity
    n_dofs = 10

    # Create correlation using exponential decay: ρ(i,j) = exp(-|i-j|/L)
    length_scale = 2.0
    correlation = np.zeros((n_dofs, n_dofs))
    for i in range(n_dofs):
        for j in range(n_dofs):
            correlation[i, j] = np.exp(-abs(i - j) / length_scale)

    # Uniform variance
    variances = np.ones(n_dofs) * 1.0

    B = DenseCovariance.from_correlation(comm, correlation, variances)

    print(f"Background covariance: {n_dofs} DoFs")
    print(f"Correlation length scale: {length_scale}")

    # Check correlation structure
    # Set v1 as impulse at center
    impulse = np.zeros(n_dofs)
    impulse[n_dofs // 2] = 1.0
    v1 = B.create_vec()
    _set_vec_from_global(v1, impulse)

    # Apply covariance to see correlation structure
    Bv1 = B.apply(v1)

    correlation_structure = _gather_vec(Bv1, comm)
    if comm.rank == 0 and correlation_structure is not None:
        print(f"Correlation at center: {correlation_structure[n_dofs//2]:.3f}")
        print(f"Correlation 2 points away: {correlation_structure[n_dofs//2 + 2]:.3f}")

    v1.destroy()
    Bv1.destroy()


def example_implicit_covariance_precision():
    """Example 4: Implicit covariance via precision matrix."""
    print("\n" + "=" * 60)
    print("Example 4: Implicit Covariance (Precision Matrix)")
    print("=" * 60)

    comm = MPI.COMM_WORLD

    # Create a simple 1D precision matrix: C⁻¹ = α·I + β·L
    # where L is the discrete Laplacian
    n_dofs = 20
    alpha = 1.0  # Identity scaling
    beta = 0.1  # Laplacian scaling

    # Build sparse precision matrix
    precision = PETSc.Mat().create(comm=comm)
    precision.setSizes((n_dofs, n_dofs))
    precision.setType(PETSc.Mat.Type.AIJ)
    precision.setPreallocationNNZ(3)  # Tridiagonal
    precision.setUp()

    if comm.rank == 0:
        # Main diagonal: α + 2β
        for i in range(n_dofs):
            precision.setValue(i, i, alpha + 2 * beta)

        # Off-diagonals: -β
        for i in range(n_dofs - 1):
            precision.setValue(i, i + 1, -beta)
            precision.setValue(i + 1, i, -beta)

    precision.assemblyBegin()
    precision.assemblyEnd()

    B = PrecisionBasedCovariance(comm, precision, inverse_is_explicit=True)

    print(f"Implicit background covariance: {n_dofs} DoFs")
    print(f"Precision matrix: α={alpha}, β={beta}")
    print(f"Effective correlation length: ~{np.sqrt(beta/alpha):.2f}")

    # Test inverse application (should be fast)
    v = B.create_vec()
    v.setRandom()

    # C⁻¹·v is just a sparse matrix-vector product
    C_inv_v = B.apply_inverse(v)

    print(f"Applied C⁻¹ (fast, explicit)")
    print(f"Result norm: {C_inv_v.norm():.4f}")

    v.destroy()
    C_inv_v.destroy()


def example_cost_functional_computation():
    """Example 5: Computing 4D-Var cost functional terms."""
    print("\n" + "=" * 60)
    print("Example 5: Computing Cost Functional Terms")
    print("=" * 60)

    comm = MPI.COMM_WORLD

    # Setup
    n_state = 50
    n_obs = 10

    # Background covariance (diagonal for simplicity)
    B = DiagonalCovariance(comm, size=n_state, variance=2.0)

    # Observation covariance
    R = create_observation_covariance(comm, n_obs, obs_std=0.5)

    # Simulated data
    m = B.create_vec()  # Current control (state estimate)
    m_b = B.create_vec()  # Background state
    y = R.create_vec()  # Observations
    H_m = R.create_vec()  # H(m) - predicted observations

    m.setRandom()
    m_b.setRandom()
    y.setRandom()
    H_m.setRandom()

    # Compute background term: J_b = ½⟨m - m_b, B⁻¹(m - m_b)⟩
    diff_b = m.duplicate()
    m.copy(diff_b)
    diff_b.axpy(-1.0, m_b)  # diff_b = m - m_b

    J_b = 0.5 * B.inner_product_inv(diff_b, diff_b)

    # Compute observation term: J_o = ½⟨H(m) - y, R⁻¹(H(m) - y)⟩
    diff_o = H_m.duplicate()
    H_m.copy(diff_o)
    diff_o.axpy(-1.0, y)  # diff_o = H(m) - y

    J_o = 0.5 * R.inner_product_inv(diff_o, diff_o)

    # Total cost
    J = J_b + J_o

    print(f"Background term (J_b): {J_b:.4f}")
    print(f"Observation term (J_o): {J_o:.4f}")
    print(f"Total cost (J): {J:.4f}")

    # Cleanup
    for vec in [m, m_b, y, H_m, diff_b, diff_o]:
        vec.destroy()


def example_ensemble_covariance():
    """Example 6: Background covariance from ensemble."""
    print("\n" + "=" * 60)
    print("Example 6: Ensemble-Based Background Covariance")
    print("=" * 60)

    comm = MPI.COMM_WORLD

    # Generate synthetic ensemble
    n_state = 30
    n_ensemble = 10

    print(f"State dimension: {n_state}")
    print(f"Ensemble size: {n_ensemble}")

    # Create ensemble members
    ensemble = []
    for i in range(n_ensemble):
        member = PETSc.Vec().create(comm=comm)
        member.setSizes((PETSc.DECIDE, n_state))
        member.setUp()

        # Add some structure: mean + random perturbation
        if comm.rank == 0:
            mean = np.sin(np.linspace(0, 2 * np.pi, n_state))
            perturbation = np.random.randn(n_state) * 0.3
            values = mean + perturbation
        else:
            values = None
        values = comm.bcast(values, root=0)
        _set_vec_from_global(member, values)

        ensemble.append(member)

    # Construct covariance with inflation
    inflation = 1.2
    B = create_background_covariance_from_ensemble(comm, ensemble, inflation)

    print(f"Inflation factor: {inflation}")
    print(f"Covariance constructed from {n_ensemble} members")

    # Test properties
    print(f"Symmetry: {check_covariance_symmetry(B)}")

    # Cleanup
    for member in ensemble:
        member.destroy()


def run_all_examples():
    """Run all examples."""
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        print("\n" + "=" * 60)
        print("COVARIANCE MATRIX EXAMPLES")
        print("=" * 60)
        print(f"Running on {comm.size} MPI rank(s)")

    example_observation_covariance()
    example_background_covariance_diagonal()
    example_background_covariance_correlated()
    example_implicit_covariance_precision()
    example_cost_functional_computation()
    example_ensemble_covariance()

    if comm.rank == 0:
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_examples()

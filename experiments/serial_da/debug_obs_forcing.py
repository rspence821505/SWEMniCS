#!/usr/bin/env python3
"""Debug observation forcing distribution."""

import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.utils import get_default_solver_params

from da_experiment_utils import (
    ForwardModelWrapper,
    generate_observation_points,
    generate_observations,
)
from swe4dvar.data_assimilation import (
    DiagonalCovariance,
    PointObservationOperator,
)


def debug_obs_forcing():
    """Debug observation forcing."""
    comm = MPI.COMM_WORLD

    nx, ny = 4, 2
    dt = 3600.0
    num_time_steps = 2

    problem = TidalProblem(nx=nx, ny=ny, dt=dt, nt=num_time_steps)
    solver = get_solver("CG")(problem, theta=1.0, p_degree=[1, 1])
    solver_params = get_default_solver_params()

    # Generate truth
    solver.time_loop(
        solver_parameters=solver_params,
        store_jacobians=True,
        save_state=True,
        enable_video=False,
    )

    truth_trajectory = [
        PETSc.Vec().createWithArray(s.copy(), comm=comm)
        for s in solver.storage.saved_states
    ]

    # Setup observations
    obs_points = generate_observation_points(problem.mesh, fraction=0.5)
    obs_op = PointObservationOperator(solver.V, obs_points, comm=comm)

    print("=" * 60)
    print("OBSERVATION OPERATOR STRUCTURE")
    print("=" * 60)
    print(f"Function space: {solver.V}")
    print(f"Is mixed: {obs_op.is_mixed}")
    print(f"Component indices: {obs_op.components}")
    print(f"Number of observations: {obs_op.n_obs}")

    # Get a state vector
    u = truth_trajectory[1]
    print(f"\nState vector size: {u.getSize()}")

    # Apply forward operator
    obs = obs_op.forward(u)
    print(f"Observation vector size: {obs.getSize()}")
    print(f"Observations: {obs.getArray()}")

    # Create a simple innovation vector
    innovation = obs.copy()
    innovation.setArray(np.ones(obs.getSize()))  # Unit innovation

    # Apply adjoint
    adj_state = obs_op.adjoint(innovation)
    print(f"\nAdjoint state size: {adj_state.getSize()}")

    # Check which DOFs are non-zero
    adj_array = adj_state.getArray()
    nonzero_dofs = np.nonzero(np.abs(adj_array) > 1e-14)[0]
    print(f"\nNon-zero DOFs in adjoint: {len(nonzero_dofs)} out of {len(adj_array)}")
    print(f"Non-zero DOF indices: {nonzero_dofs[:30]}...")

    # Group by sub-space (assuming h=0-14, uh/vh=15-44)
    h_dofs = [d for d in nonzero_dofs if d < 15]
    mom_dofs = [d for d in nonzero_dofs if d >= 15]
    print(f"\nNon-zero h DOFs: {len(h_dofs)}: {h_dofs}")
    print(f"Non-zero momentum DOFs: {len(mom_dofs)}: {mom_dofs}")

    # Print non-zero values
    print("\nAdjoint values at non-zero DOFs:")
    for d in nonzero_dofs:
        var = "h" if d < 15 else "uh/vh"
        print(f"  DOF {d} ({var}): {adj_array[d]:.6e}")


if __name__ == "__main__":
    debug_obs_forcing()

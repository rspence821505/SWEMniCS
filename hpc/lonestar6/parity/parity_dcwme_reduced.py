"""parity_dcwme_reduced.py — DC-WME cost+gradient parity test (reduced).

Identical structure to parity_4dvar_reduced.py but with DC-WME cost function
(static L_wme). Currently blocked by the same dolfinx 0.9 -> 0.10 API drifts
in the project code that block parity_4dvar_reduced.py on LS6
(interpolation_points and fem.petsc.create_vector).

Once the project source is ported to dolfinx 0.10, this test should run and
produce numerically comparable J_bg and ||grad|| between local and LS6.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

# Re-use everything from parity_4dvar_reduced, just swap the cost function.
sys.path.insert(0, str(Path(__file__).parent))
from parity_4dvar_reduced import (    # noqa: E402
    CONFIG, _dolfinx,                  # triggers the 0.10 shim
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serial_da"))

from swe4dvar.forward.problems import TidalProblem                      # noqa: E402
from swe4dvar.forward.solvers import get_solver                          # noqa: E402
from swe4dvar.data_assimilation import (                                 # noqa: E402
    PointObservationOperator, DiagonalCovariance,
)
from swe4dvar.data_assimilation.cost_functions import DCWMEFourDVarCost  # noqa: E402
from swe4dvar.utils import get_default_solver_params                      # noqa: E402
from da_experiment_utils import ForwardModelWrapper                      # type: ignore  # noqa: E402


def run_parity_dcwme(config=CONFIG):
    comm = MPI.COMM_WORLD
    rank = comm.rank

    problem = TidalProblem(
        nx=config["nx"], ny=config["ny"],
        x0=config["x0"], x1=config["x1"], y0=config["y0"], y1=config["y1"],
        nt=config["nt"], dt=config["dt"],
    )
    solver = get_solver("CG")(problem, theta=1.0, p_degree=[1, 1], verbose=False)
    solver_params = get_default_solver_params()
    solver_params["atol"] = config["solver_atol"]
    solver_params["max_it"] = config["solver_max_it"]
    forward_model = ForwardModelWrapper(solver, problem, solver_params)

    n_dofs_local = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs
    n_dofs_global = solver.V.dofmap.index_map.size_global * solver.V.dofmap.index_map_bs

    m_true = PETSc.Vec().createWithArray(solver.u.x.array.copy(), comm=comm)
    trajectory, _jac = forward_model.solve(m_true, store_jacobians=True)

    coords = problem.mesh.geometry.x
    interior_mask = (
        (coords[:, 0] > coords[:, 0].min() + 0.1) &
        (coords[:, 0] < coords[:, 0].max() - 0.1) &
        (coords[:, 1] > coords[:, 1].min() + 0.1) &
        (coords[:, 1] < coords[:, 1].max() - 0.1)
    )
    interior_coords = coords[interior_mask][: config["n_obs_points"]]
    obs_points = np.zeros((len(interior_coords), 3))
    obs_points[:, :2] = interior_coords[:, :2]

    obs_operator = PointObservationOperator(solver.V, obs_points, comm=comm)
    obs = obs_operator.forward(trajectory[-1])
    obs_arr = obs.getArray()
    rng = np.random.default_rng(seed=config["seed"])
    obs_arr = obs_arr + config["obs_rel_noise"] * np.abs(obs_arr) * rng.standard_normal(len(obs_arr))
    obs_perturbed = obs.duplicate()
    obs_perturbed.setArray(obs_arr)

    B = DiagonalCovariance(comm, n_dofs_local, variance=config["bg_variance"])
    n_obs_total = obs_operator.get_num_observations()
    R = DiagonalCovariance(comm, n_obs_total, variance=config["obs_variance"])

    cost_function = DCWMEFourDVarCost(
        forward_model=forward_model,
        observation_operator=obs_operator,
        background_cov=B,
        observation_cov=R,
        m_background=m_true.copy(),
        observations=[obs_perturbed],
        obs_times=[len(trajectory) - 1],
        comm=comm,
        # DC-WME-specific knobs: use static L_wme, no TLM build (parity test is small)
        l_wme_mode="static",
    )

    cost_function.clear_cache()
    J_bg = float(cost_function.value(m_true))
    cost_function.clear_cache()
    grad = cost_function.gradient(m_true)
    grad_arr = grad.getArray().copy()

    grad_l2_sq_local = float(np.sum(grad_arr * grad_arr))
    grad_l2_sq_global = comm.allreduce(grad_l2_sq_local, op=MPI.SUM)
    grad_l2_global = float(np.sqrt(grad_l2_sq_global))
    grad_linf_local = float(np.max(np.abs(grad_arr))) if grad_arr.size else 0.0
    grad_linf_global = comm.allreduce(grad_linf_local, op=MPI.MAX)

    out = {
        "test": "parity_dcwme_reduced",
        "mpi_size": comm.size,
        "n_dofs_global": int(n_dofs_global),
        "n_obs_total": int(n_obs_total),
        "J_bg": J_bg,
        "grad_l2_global": grad_l2_global,
        "grad_linf_global": grad_linf_global,
        "config": config,
    }
    return out


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    try:
        result = run_parity_dcwme()
    except Exception as e:
        if comm.rank == 0:
            err_out = {
                "test": "parity_dcwme_reduced",
                "FAILED": True,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
            print(json.dumps(err_out, indent=2, sort_keys=True))
        sys.exit(1)
    if comm.rank == 0:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))

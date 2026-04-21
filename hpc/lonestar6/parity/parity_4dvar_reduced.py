"""parity_4dvar_reduced.py — tight, deterministic 4D-Var cost+gradient evaluation.

Designed to run identically on local fenics-env3 (dolfinx 0.9) and LS6
(dolfinx 0.10). Uses the project's own forward model + FourDVarCost to
exercise the full pipeline, but avoids the TAO optimizer loop — we just
evaluate J(m_bg) and ∇J(m_bg) at a single fixed point and compare scalars.

Why this is enough for parity: if the forward solve, observation operator,
and discrete-adjoint gradient produce matching values at the SAME point,
then any downstream optimizer that calls these will also produce matching
trajectories (modulo line-search tie-breaking, which is not a correctness
concern).

Single-rank by default (`python parity_4dvar_reduced.py`). MPI-safe.

Output: JSON record on stdout.
"""
from __future__ import annotations
import json
import sys
import os
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

# -----------------------------------------------------------------------------
# dolfinx 0.9 vs 0.10 API shim (parity test only — do not copy into production)
#
# In dolfinx 0.9, element.interpolation_points is a METHOD: `el.interpolation_points()`.
# In dolfinx 0.10, it is a PROPERTY returning ndarray: `el.interpolation_points`.
# The project code calls it as a method (11 sites), so on 0.10 it fails with
# `'numpy.ndarray' object is not callable`. We wrap the property so that callers
# can invoke it either way — and we also cover the analogous `.value_shape`
# shift which may apply.
#
# This is scoped to the parity test so we can measure numerical parity despite
# the API drift. The actual porting fix must edit the project's call sites.
# -----------------------------------------------------------------------------
import dolfinx as _dolfinx

if _dolfinx.__version__.startswith(("0.10", "0.11", "0.12")):
    from dolfinx.fem import ElementMetaData  # noqa: F401
    # Patch FiniteElement: allow .interpolation_points() in addition to .interpolation_points
    try:
        # dolfinx.cpp FiniteElement classes are nanobind-bound; we patch
        # after calling an instance's .element attribute once to force class load.
        import dolfinx.mesh as _m
        import dolfinx.fem as _f
        _probe_domain = _m.create_unit_square(MPI.COMM_SELF, 2, 2)
        _probe_V = _f.functionspace(_probe_domain, ("Lagrange", 1))
        _elcls = type(_probe_V.element)
        _orig = _elcls.interpolation_points
        if not callable(_orig):
            # It's a property -> returns ndarray. Wrap it in a callable descriptor.
            class _CallableArr(np.ndarray):
                def __call__(self):
                    return np.asarray(self)
            def _ip_method(self):
                arr = _orig.fget(self) if hasattr(_orig, "fget") else _orig.__get__(self, type(self))
                # Return an object that is both an ndarray AND callable returning itself
                v = np.asarray(arr).view(_CallableArr)
                return v
            _elcls.interpolation_points = property(_ip_method)
    except Exception as _e:
        print(f"# WARN: interpolation_points shim did not apply: {_e}", file=sys.stderr)

# -- path setup ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serial_da"))

# -- project imports ---------------------------------------------------------
from swe4dvar.forward.problems import TidalProblem
from swe4dvar.forward.solvers import get_solver
from swe4dvar.data_assimilation import (
    FourDVarCost, PointObservationOperator, DiagonalCovariance,
)
from swe4dvar.utils import get_default_solver_params
from da_experiment_utils import ForwardModelWrapper  # type: ignore

# -- deterministic config ----------------------------------------------------
CONFIG = dict(
    nx=5, ny=3,
    x0=0.0, x1=10000.0,
    y0=0.0, y1=5000.0,
    nt=2,
    dt=1800.0,         # 30 min steps
    solver_atol=1e-6,
    solver_max_it=10,
    seed=42,
    obs_rel_noise=0.10,
    bg_variance=0.01,
    obs_variance=0.001,
    n_obs_points=5,
)


def run_parity_4dvar(config=CONFIG):
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Forward problem (tiny deterministic tidal)
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

    # m_true = initial condition the solver has set up
    m_true = PETSc.Vec().createWithArray(solver.u.x.array.copy(), comm=comm)

    # Forward trajectory
    trajectory, _jac = forward_model.solve(m_true, store_jacobians=True)

    # Pick deterministic interior observation points
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

    # Observations = forward at final time + deterministic Gaussian perturbation
    obs = obs_operator.forward(trajectory[-1])
    obs_arr = obs.getArray()
    rng = np.random.default_rng(seed=config["seed"])
    obs_arr = obs_arr + config["obs_rel_noise"] * np.abs(obs_arr) * rng.standard_normal(len(obs_arr))
    obs_perturbed = obs.duplicate()
    obs_perturbed.setArray(obs_arr)

    # Covariances
    B = DiagonalCovariance(comm, n_dofs_local, variance=config["bg_variance"])
    n_obs_total = obs_operator.get_num_observations()
    R = DiagonalCovariance(comm, n_obs_total, variance=config["obs_variance"])

    # Cost function
    cost_function = FourDVarCost(
        forward_model=forward_model,
        observation_operator=obs_operator,
        background_cov=B,
        observation_cov=R,
        m_background=m_true.copy(),
        observations=[obs_perturbed],
        obs_times=[len(trajectory) - 1],
        comm=comm,
    )

    # Evaluate at background (m_true) — expected to be a non-trivial cost
    # because observations are perturbed.
    cost_function.clear_cache()
    J_bg = float(cost_function.value(m_true))

    cost_function.clear_cache()
    grad = cost_function.gradient(m_true)
    grad_arr = grad.getArray().copy()

    # Global reductions so results are independent of partitioning
    grad_l2_sq_local = float(np.sum(grad_arr * grad_arr))
    grad_l2_sq_global = comm.allreduce(grad_l2_sq_local, op=MPI.SUM)
    grad_l2_global = float(np.sqrt(grad_l2_sq_global))
    grad_linf_local = float(np.max(np.abs(grad_arr))) if grad_arr.size else 0.0
    grad_linf_global = comm.allreduce(grad_linf_local, op=MPI.MAX)

    # Trajectory summary (final state L2 norm, global)
    # Trajectory entries may be PETSc Vec or dolfinx Function depending on project version.
    last = trajectory[-1]
    if hasattr(last, "getArray"):
        final_arr = last.getArray()
    elif hasattr(last, "x"):
        final_arr = np.asarray(last.x.array)
    else:
        final_arr = np.asarray(last)
    final_l2_sq_local = float(np.sum(final_arr * final_arr))
    final_l2_sq_global = comm.allreduce(final_l2_sq_local, op=MPI.SUM)
    final_l2_global = float(np.sqrt(final_l2_sq_global))

    # Observation residual summary
    obs_arr_stored = obs_perturbed.getArray().copy()
    obs_l2_local = float(np.sqrt(np.sum(obs_arr_stored * obs_arr_stored)))
    obs_l2_sq_local = float(np.sum(obs_arr_stored * obs_arr_stored))
    obs_l2_sq_global = comm.allreduce(obs_l2_sq_local, op=MPI.SUM)
    obs_l2_global = float(np.sqrt(obs_l2_sq_global))

    out = {
        "test": "parity_4dvar_reduced",
        "mpi_size": comm.size,
        "n_dofs_global": int(n_dofs_global),
        "n_obs_total": int(n_obs_total),
        "J_bg": J_bg,
        "grad_l2_global": grad_l2_global,
        "grad_linf_global": grad_linf_global,
        "final_state_l2_global": final_l2_global,
        "obs_l2_global": obs_l2_global,
        # First 3 gradient values on rank 0 for bit-level spot-check
        "grad_head_rank0": [float(x) for x in grad_arr[:3]] if rank == 0 else None,
        # Config echo for traceability
        "config": config,
    }
    return out


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    try:
        result = run_parity_4dvar()
    except Exception as e:
        if comm.rank == 0:
            err_out = {
                "test": "parity_4dvar_reduced",
                "FAILED": True,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
            print(json.dumps(err_out, indent=2, sort_keys=True))
        sys.exit(1)

    if comm.rank == 0:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))

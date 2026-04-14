#!/usr/bin/env python3
"""Minimal working example for augmented-control wind-parameter estimation.

This example demonstrates the stage-1 augmented-control path:
    c = theta = [track_shift_km, intensity_bias, timing_offset_s]

The hydrodynamic initial condition is held fixed through a reference state,
while the low-dimensional wind parameters are estimated with 4D-Var or
DC-WME over the packed control vector.

The script is intentionally short-window and serial-only. It is a working
architectural example, not a production Shinnecock sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from swe4dvar.control import ControlLayout
from swe4dvar.data_assimilation import (
    DiagonalCovariance,
    PointObservationOperator,
    create_cost_function,
)
from swe4dvar.forward import ADCIRCProblem, AugmentedForwardModelWrapper, LowDimWindController, get_solver
from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper
from swe4dvar.physics.forcing import ParametricWindForcing
from swe4dvar.utils import get_default_solver_params


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage-1 augmented-control wind parameter example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", choices=["4dvar", "wme"], default="4dvar")
    parser.add_argument("--adios-file", default="data/shinnecock_inlet")
    parser.add_argument("--wind-file", default="outputs/shinnecock_study/wind_fields/holland_V30_truth.h5")
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--nt", type=int, default=12)
    parser.add_argument("--obs-frequency", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=5)
    return parser.parse_args()


def _build_problem(adios_file: str, forcing, dt: float, nt: int):
    problem = ADCIRCProblem(
        adios_file=adios_file,
        spherical=True,
        solution_var="h",
        friction_law="mannings",
        wd=True,
        wd_alpha=1.5,
        dt=dt,
        bathy_adjustment=0,
        nt=nt,
        dramp=2.0,
        forcing=forcing,
    )
    solver = get_solver("DG")(problem, theta=1.0, p_degree=[1, 1])
    return problem, solver


def _select_observation_points(problem, fraction: float = 0.01):
    coords = problem.mesh.geometry.x
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    mask = (
        (coords[:, 0] > x_min + 1e-10)
        & (coords[:, 0] < x_max - 1e-10)
        & (coords[:, 1] > y_min + 1e-10)
        & (coords[:, 1] < y_max - 1e-10)
    )
    interior = coords[mask]
    n_obs = max(4, int(len(interior) * fraction))
    obs_points = np.zeros((n_obs, 3))
    obs_points[:, :2] = interior[:n_obs, :2]
    return obs_points


def _build_observations(wrapper, obs_operator, control_true, obs_times, noise_level):
    truth_trajectory, _ = wrapper.solve(control_true, store_jacobians=False)
    observations = []
    obs_noise_stds = []
    rng = np.random.default_rng(42)
    for t in obs_times:
        Hu = obs_operator.forward(truth_trajectory[t])
        arr = Hu.getArray()
        sigma = noise_level * (np.abs(arr).mean() + 1e-10)
        obs_noise_stds.append(sigma)
        noisy = arr + rng.normal(0.0, sigma, size=arr.shape)
        y = PETSc.Vec().createSeq(len(noisy), comm=PETSc.COMM_SELF)
        y.setArray(noisy)
        y.assemble()
        observations.append(y)
    return observations, np.asarray(obs_noise_stds)


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 1:
        raise SystemExit("The augmented-control wind example is serial-only in stage 1.")

    wind_file = Path(args.wind_file)
    if not wind_file.exists():
        raise FileNotFoundError(
            f"{wind_file} not found. Generate a base wind file first, for example via "
            "experiments/shinnecock_study/run_wind_sweep.py."
        )

    forcing = ParametricWindForcing(str(wind_file), lat0=35)
    problem, solver = _build_problem(args.adios_file, forcing, args.dt, args.nt)
    solver_params = get_default_solver_params(
        rtol=1e-5,
        atol=1e-6,
        max_it=10,
        relaxation_parameter=1.0,
        comm=comm,
        error_if_not_converged=False,
    )

    controller = LowDimWindController(forcing)
    layout = ControlLayout(state_size=0, theta_size=3, comm=PETSc.COMM_SELF)
    wrapper = AugmentedForwardModelWrapper(
        solver=solver,
        problem=problem,
        solver_params=solver_params,
        control_layout=layout,
        parameter_controller=controller,
        parameter_fd_steps=np.array([1.0, 0.01, 600.0]),
        reference_state=solver.u_n.x.array.copy(),
    )

    obs_points = _select_observation_points(problem)
    obs_operator = PointObservationOperator(solver.V, obs_points, comm=comm)
    obs_times = list(range(0, args.nt + 1, args.obs_frequency))

    theta_true = np.array([0.0, 0.0, 0.0], dtype=float)
    theta_background = np.array([10.0, -0.10, 1800.0], dtype=float)
    m_true = layout.create_petsc_vec(theta=theta_true)
    m_background = layout.create_petsc_vec(theta=theta_background)

    observations, obs_noise_stds = _build_observations(
        wrapper,
        obs_operator,
        m_true,
        obs_times,
        noise_level=0.02,
    )
    R = {
        t: DiagonalCovariance(comm, size=len(obs_points), variance=float(obs_noise_stds[i] ** 2))
        for i, t in enumerate(obs_times)
    }
    B_theta = DiagonalCovariance(comm, size=layout.theta_size, diagonal=np.array([100.0, 0.05, 3600.0 ** 2]))

    cost = create_cost_function(
        args.method,
        forward_model=wrapper,
        observation_operator=obs_operator,
        background_cov=B_theta,
        observation_cov=R,
        m_background=m_background,
        observations=observations,
        obs_times=obs_times,
        predictability_gamma=0.1,
    )

    lower = layout.create_petsc_vec(theta=[-30.0, -0.40, -6 * 3600.0])
    upper = layout.create_petsc_vec(theta=[30.0, 0.40, 6 * 3600.0])

    optimizer = PETScTAOWrapper(
        cost,
        tao_type="blmvm",
        lower_bounds=lower,
        upper_bounds=upper,
        options={
            "max_iterations": args.max_iter,
            "gradient_tolerance": 1e-4,
            "cost_tolerance": 1e-8,
            "verbose": True,
        },
    )
    analysis = optimizer.solve(m_background)
    theta_analysis = layout.unpack_vec(analysis).theta

    if comm.rank == 0:
        print("\nAugmented wind-parameter example")
        print(f"  method:      {args.method}")
        print(f"  theta_true:  {theta_true.tolist()}")
        print(f"  theta_bg:    {theta_background.tolist()}")
        print(f"  theta_an:    {theta_analysis.tolist()}")
        print(f"  iterations:  {optimizer.iteration}")
        print(f"  converged:   {optimizer.converged}")


if __name__ == "__main__":
    raise SystemExit(main())

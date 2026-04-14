"""Finite-difference parameter-estimation runners."""

from __future__ import annotations

import gc
import hashlib
import itertools
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .base import BaseTwinExperimentRunner
from .common import rank0_log, rmse, summarize_spectrum, write_json


class FiniteDifferenceParameterRunner(BaseTwinExperimentRunner):
    """Shared solver path for non-state inverse problems."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = self.method or "dcwme"
        self._simulation_cache: dict[str, dict[str, Any]] = {}
        self._objective_cache: dict[str, dict[str, Any]] = {}
        self._iteration_history: list[dict[str, Any]] = []
        self._linearization: dict[str, Any] | None = None

    def construct_problem_bundle(self) -> dict[str, Any]:
        self.log_stage("Problem Construction", "building parameter-space configuration")
        cfg = self.spec.forward_model_config
        return {
            "adios_file": cfg["adios_file"],
            "dt": float(cfg["dt"]),
            "nt_total": int(cfg["nt_total"]),
            "nt_ramp": int(cfg["nt_ramp"]),
            "obs_frequency": int(self.spec.observation_config["obs_frequency"]),
            "obs_fraction": float(self.spec.observation_config["obs_fraction"]),
            "obs_noise_level": float(self.spec.noise_model["obs_noise_level"]),
            "obs_seed": int(self.spec.noise_model["obs_seed"]),
            "max_iterations": int(cfg["max_iterations"]),
            "output_root": self.output_paths["root"],
        }

    def generate_truth(self, problem_bundle: dict[str, Any]) -> dict[str, Any]:
        self.log_stage("Forward Model", "running truth simulation")
        truth_theta = self.truth_parameters()
        truth_eval = self._simulate(theta=truth_theta, problem_bundle=problem_bundle, run_label="truth")
        return {
            "theta_true": truth_theta.tolist(),
            "truth_observations": truth_eval["obs_matrix"],
            "obs_points": truth_eval["obs_points"],
            "obs_times": truth_eval["obs_times"],
            "truth_summary": truth_eval["summary"],
        }

    def generate_observations(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        self.log_stage("Observation Generation", "sampling and perturbing synthetic observations")
        truth_obs = np.asarray(truth_bundle["truth_observations"], dtype=float)
        rng = np.random.default_rng(problem_bundle["obs_seed"])
        signal = np.mean(np.abs(truth_obs), axis=1) + 1e-10
        noise_stds = problem_bundle["obs_noise_level"] * signal
        noise = rng.normal(0.0, noise_stds[:, None], size=truth_obs.shape)
        noisy_obs = truth_obs + noise
        return {
            "truth_observations": truth_obs,
            "noisy_observations": noisy_obs,
            "noise_stds": noise_stds,
            "obs_points": truth_bundle["obs_points"],
            "obs_times": truth_bundle["obs_times"],
        }

    def solve_inverse(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        self.log_stage("Inverse Solve", f"running {self.method.upper()} in parameter space")
        self._latest_problem_bundle = problem_bundle
        self._latest_observation_bundle = observation_bundle
        theta_background = self.background_parameters()
        bounds = self.parameter_bounds()
        self._linearization = self._compute_linearization(
            theta_background,
            problem_bundle,
            observation_bundle,
        )

        initial = self._evaluate_objective(theta_background, problem_bundle, observation_bundle)
        self._record_iteration(theta_background, initial)

        optimization_start = time.time()
        result = minimize(
            fun=lambda x: self._objective_value(x, problem_bundle, observation_bundle),
            x0=theta_background,
            jac=lambda x: self._objective_gradient(x, problem_bundle, observation_bundle),
            method="L-BFGS-B",
            bounds=bounds,
            callback=lambda xk: self._callback(xk, problem_bundle, observation_bundle),
            options={"maxiter": problem_bundle["max_iterations"], "disp": False},
        )
        optimization_s = time.time() - optimization_start

        theta_analysis = np.asarray(result.x, dtype=float)
        final_eval = self._evaluate_objective(theta_analysis, problem_bundle, observation_bundle)
        if not self._iteration_history or not np.allclose(
            self._iteration_history[-1]["theta"],
            theta_analysis.tolist(),
        ):
            self._record_iteration(theta_analysis, final_eval)

        return {
            "theta_background": theta_background.tolist(),
            "theta_analysis": theta_analysis.tolist(),
            "scipy_result": {
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "nit": int(result.nit),
                "nfev": int(result.nfev),
                "njev": int(getattr(result, "njev", 0)),
            },
            "optimization_s": optimization_s,
            "final_eval": final_eval,
        }

    def collect_diagnostics(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
        solve_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        self.log_stage("Diagnostics", "assembling convergence and spectral summaries")
        return {
            "convergence": self._iteration_history,
            "linearization": self._linearization,
            "truth_summary": truth_bundle["truth_summary"],
        }

    def save_outputs(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
        solve_bundle: dict[str, Any],
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        self.log_stage("Results Save", "writing standardized output files")

        final_eval = solve_bundle["final_eval"]
        metrics = {
            "experiment": self.spec.name,
            "inverse_problem_type": self.spec.inverse_problem_type,
            "method": self.method,
            "background_parameter_error": float(
                rmse(np.asarray(solve_bundle["theta_background"]), self.truth_parameters())
            ),
            "analysis_parameter_error": float(
                rmse(np.asarray(solve_bundle["theta_analysis"]), self.truth_parameters())
            ),
            "observation_rmse": float(final_eval["observation_rmse"]),
            "cost": float(final_eval["total_cost"]),
            "converged": bool(solve_bundle["scipy_result"]["success"]),
            "num_iterations": int(solve_bundle["scipy_result"]["nit"]),
            "optimization_s": float(solve_bundle["optimization_s"]),
        }

        if self.rank == 0:
            write_json(self.output_paths["root"] / "metrics.json", metrics)
            write_json(
                self.output_paths["diagnostics"] / "convergence.json",
                diagnostics["convergence"],
            )
            write_json(
                self.output_paths["diagnostics"] / "spectral.json",
                diagnostics["linearization"],
            )
            write_json(
                self.output_paths["diagnostics"] / "stage_log.json",
                self.stage_log,
            )
            write_json(
                self.output_paths["trajectories"] / "parameter_history.json",
                {
                    "truth": truth_bundle["theta_true"],
                    "background": solve_bundle["theta_background"],
                    "analysis": solve_bundle["theta_analysis"],
                },
            )
            write_json(
                self.output_paths["trajectories"] / "truth_summary.json",
                diagnostics["truth_summary"],
            )

        return metrics

    def truth_parameters(self) -> np.ndarray:
        raise NotImplementedError

    def background_parameters(self) -> np.ndarray:
        raise NotImplementedError

    def parameter_stds(self) -> np.ndarray:
        raise NotImplementedError

    def fd_steps(self) -> np.ndarray:
        raise NotImplementedError

    def parameter_bounds(self):
        raise NotImplementedError

    def _background_weight(self) -> float:
        return 1.0

    def _simulate(
        self,
        *,
        theta: np.ndarray,
        problem_bundle: dict[str, Any],
        run_label: str,
        obs_points=None,
    ) -> dict[str, Any]:
        cache_key = self._theta_key(theta, prefix=run_label)
        if cache_key in self._simulation_cache:
            cached = self._simulation_cache[cache_key]
            if obs_points is None or cached["obs_points"] is obs_points:
                return cached

        sim_result = self._run_parameterized_forward(
            theta=np.asarray(theta, dtype=float),
            problem_bundle=problem_bundle,
            run_label=run_label,
            obs_points=obs_points,
        )
        self._simulation_cache[cache_key] = sim_result
        return sim_result

    def _evaluate_objective(
        self,
        theta: np.ndarray,
        problem_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        cache_key = self._theta_key(theta, prefix=f"objective:{self.method}")
        if cache_key in self._objective_cache:
            return self._objective_cache[cache_key]

        theta = np.asarray(theta, dtype=float)
        sim = self._simulate(
            theta=theta,
            problem_bundle=problem_bundle,
            run_label="analysis",
            obs_points=observation_bundle["obs_points"],
        )

        weighted_residuals = (
            (sim["obs_matrix"] - observation_bundle["noisy_observations"])
            / observation_bundle["noise_stds"][:, None]
        )
        background_delta = theta - self.background_parameters()
        background_term = 0.5 * self._background_weight() * np.sum(
            (background_delta / self.parameter_stds()) ** 2
        )
        observation_term = 0.5 * float(np.sum(weighted_residuals ** 2))
        q_theta = weighted_residuals.sum(axis=0) / np.sqrt(weighted_residuals.shape[0])
        wme_term = 0.5 * float(np.dot(q_theta, q_theta))
        predictability_term = 0.0
        if self.method == "dcwme":
            delta_q = q_theta - self._linearization["q_background"]
            L_inv = self._linearization["L_inv"]
            predictability_term = 0.5 * float(delta_q @ L_inv @ delta_q)
            total_cost = background_term + wme_term - predictability_term
        else:
            total_cost = background_term + observation_term

        payload = {
            "theta": theta.tolist(),
            "total_cost": float(total_cost),
            "background_term": float(background_term),
            "observation_term": float(observation_term),
            "wme_term": float(wme_term),
            "predictability_term": float(predictability_term),
            "observation_rmse": float(
                rmse(sim["obs_matrix"], observation_bundle["truth_observations"])
            ),
            "parameter_rmse": float(rmse(theta, self.truth_parameters())),
            "sim_summary": sim["summary"],
        }
        self._objective_cache[cache_key] = payload
        return payload

    def _objective_value(self, theta, problem_bundle, observation_bundle) -> float:
        return self._evaluate_objective(theta, problem_bundle, observation_bundle)["total_cost"]

    def _objective_gradient(self, theta, problem_bundle, observation_bundle) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        steps = self.fd_steps()
        grad = np.zeros_like(theta)
        for i, step in enumerate(steps):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += step
            theta_minus[i] -= step
            grad[i] = (
                self._objective_value(theta_plus, problem_bundle, observation_bundle)
                - self._objective_value(theta_minus, problem_bundle, observation_bundle)
            ) / (2.0 * step)
        return grad

    def _callback(self, theta, problem_bundle, observation_bundle) -> None:
        evaluation = self._evaluate_objective(theta, problem_bundle, observation_bundle)
        self._record_iteration(np.asarray(theta, dtype=float), evaluation)

    def _record_iteration(self, theta: np.ndarray, evaluation: dict[str, Any]) -> None:
        grad = self._objective_gradient(theta, self._latest_problem_bundle, self._latest_observation_bundle)
        self._iteration_history.append(
            {
                "iteration": len(self._iteration_history),
                "theta": theta.tolist(),
                "cost": float(evaluation["total_cost"]),
                "gradient_norm": float(np.linalg.norm(grad)),
                "background_term": float(evaluation["background_term"]),
                "observation_term": float(evaluation["observation_term"]),
                "wme_term": float(evaluation["wme_term"]),
                "predictability_term": float(evaluation["predictability_term"]),
                "observation_rmse": float(evaluation["observation_rmse"]),
                "parameter_rmse": float(evaluation["parameter_rmse"]),
            }
        )

    def _compute_linearization(
        self,
        theta_background: np.ndarray,
        problem_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        background_sim = self._simulate(
            theta=theta_background,
            problem_bundle=problem_bundle,
            run_label="background",
            obs_points=observation_bundle["obs_points"],
        )
        steps = self.fd_steps()
        base_pred = background_sim["obs_matrix"]
        n_times, n_obs = base_pred.shape
        p = len(theta_background)
        J_obs = np.zeros((n_times * n_obs, p))
        J_wme = np.zeros((n_obs, p))
        for i, step in enumerate(steps):
            theta_plus = theta_background.copy()
            theta_minus = theta_background.copy()
            theta_plus[i] += step
            theta_minus[i] -= step
            pred_plus = self._simulate(
                theta=theta_plus,
                problem_bundle=problem_bundle,
                run_label=f"lin_plus_{i}",
                obs_points=observation_bundle["obs_points"],
            )["obs_matrix"]
            pred_minus = self._simulate(
                theta=theta_minus,
                problem_bundle=problem_bundle,
                run_label=f"lin_minus_{i}",
                obs_points=observation_bundle["obs_points"],
            )["obs_matrix"]
            deriv = (pred_plus - pred_minus) / (2.0 * step)
            weighted_deriv = deriv / observation_bundle["noise_stds"][:, None]
            J_obs[:, i] = weighted_deriv.reshape(-1)
            J_wme[:, i] = weighted_deriv.sum(axis=0) / np.sqrt(n_times)

        B_diag = (self.parameter_stds() ** 2) / max(self._background_weight(), 1e-30)
        B = np.diag(B_diag)
        B_inv = np.diag(1.0 / B_diag)
        H_gn = B_inv + J_obs.T @ J_obs
        H_eigs = np.linalg.eigvalsh(H_gn)
        L = np.eye(n_obs) + J_wme @ B @ J_wme.T
        L_eigs = np.linalg.eigvalsh(L)
        q_background = (
            (
                (base_pred - observation_bundle["noisy_observations"])
                / observation_bundle["noise_stds"][:, None]
            ).sum(axis=0)
            / np.sqrt(n_times)
        )

        return {
            "parameter_hessian_spectrum": summarize_spectrum(H_eigs),
            "l_wme_spectrum": summarize_spectrum(L_eigs),
            "J_obs_shape": list(J_obs.shape),
            "J_wme_shape": list(J_wme.shape),
            "B_diagonal": B_diag.tolist(),
            "L_inv": np.linalg.inv(L),
            "q_background": q_background,
        }

    def _theta_key(self, theta: np.ndarray, prefix: str) -> str:
        arr = np.asarray(theta, dtype=float)
        return f"{prefix}:{hashlib.md5(arr.tobytes()).hexdigest()}"

    def _run_parameterized_forward(
        self,
        *,
        theta: np.ndarray,
        problem_bundle: dict[str, Any],
        run_label: str,
        obs_points=None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _make_observation_points(self, mesh, fraction: float, seed: int):
        local_coords = mesh.geometry.x
        all_coords = self.comm.gather(local_coords, root=0)
        if self.rank == 0:
            coords_all = np.vstack(all_coords)
            _, unique_idx = np.unique(
                np.round(coords_all[:, :2], decimals=10),
                axis=0,
                return_index=True,
            )
            coords = coords_all[unique_idx]
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            interior_mask = (
                (coords[:, 0] > x_min + 1e-10)
                & (coords[:, 0] < x_max - 1e-10)
                & (coords[:, 1] > y_min + 1e-10)
                & (coords[:, 1] < y_max - 1e-10)
            )
            interior = coords[interior_mask]
            rng = np.random.default_rng(seed)
            n_obs = max(1, int(len(interior) * fraction))
            chosen = rng.choice(len(interior), size=min(n_obs, len(interior)), replace=False)
            obs_points = np.zeros((len(chosen), 3))
            obs_points[:, :2] = interior[chosen, :2]
        else:
            obs_points = None
        return self.comm.bcast(obs_points, root=0)


class WindParameterRunner(FiniteDifferenceParameterRunner):
    def truth_parameters(self) -> np.ndarray:
        cfg = self.spec.forward_model_config
        return np.array(
            [
                cfg["truth_track_shift_km"],
                cfg["truth_intensity_bias"],
                cfg["truth_timing_offset_s"],
            ],
            dtype=float,
        )

    def background_parameters(self) -> np.ndarray:
        return np.asarray(self.spec.forward_model_config["background_parameters"], dtype=float)

    def parameter_stds(self) -> np.ndarray:
        return np.asarray(self.spec.forward_model_config["parameter_std"], dtype=float)

    def fd_steps(self) -> np.ndarray:
        return np.asarray(self.spec.forward_model_config["fd_steps"], dtype=float)

    def parameter_bounds(self):
        return [tuple(bound) for bound in self.spec.forward_model_config["bounds"]]

    def _run_parameterized_forward(
        self,
        *,
        theta: np.ndarray,
        problem_bundle: dict[str, Any],
        run_label: str,
        obs_points=None,
    ) -> dict[str, Any]:
        from dolfinx import la
        from swe4dvar.data_assimilation import PointObservationOperator
        from swe4dvar.forward.adcirc_problem import ADCIRCProblem
        from swe4dvar.forward.solvers import get_solver
        from swe4dvar.physics.forcing import GriddedForcing
        from swe4dvar.utils import get_default_solver_params

        from experiments.shinnecock_study.wind_models import (
            DEFAULT_TRACK,
            HollandHurricaneConfig,
            WindGridConfig,
            generate_holland_wind_field,
            write_wind_hdf5,
        )

        cfg = self.spec.forward_model_config
        dt = float(cfg["dt"])
        nt_total = int(cfg["nt_total"])
        times = np.arange(0.0, (nt_total + 1) * dt, dt)
        wind_dir = self.output_paths["diagnostics"] / "wind_fields"
        if self.rank == 0:
            wind_dir.mkdir(exist_ok=True)

        truth_vmax = float(cfg["truth_vmax"]) * (1.0 + float(theta[1]))
        base_track = [
            (float(t + theta[2]), lon, lat) for t, lon, lat in DEFAULT_TRACK
        ]
        if abs(theta[0]) > 0:
            from experiments.shinnecock_study.wind_models import generate_perturbed_config

            wind_config = HollandHurricaneConfig(track_waypoints=DEFAULT_TRACK, Vmax=truth_vmax)
            wind_config = generate_perturbed_config(wind_config, "timing_offset", float(theta[2]))
            wind_config = generate_perturbed_config(wind_config, "track_shift", float(theta[0]))
            wind_config.Vmax = truth_vmax
        else:
            wind_config = HollandHurricaneConfig(track_waypoints=base_track, Vmax=truth_vmax)

        wind_file = wind_dir / f"{self._theta_key(theta, run_label)}.h5"
        if self.rank == 0 and not wind_file.exists():
            grid = WindGridConfig()
            windx, windy, pressure = generate_holland_wind_field(
                wind_config,
                grid,
                times,
                wind_ramp_s=float(cfg["wind_ramp_s"]),
            )
            write_wind_hdf5(str(wind_file), grid, times, windx, windy, pressure)
        self.comm.Barrier()

        forcing = GriddedForcing(str(wind_file), lat0=35)
        problem = ADCIRCProblem(
            adios_file=cfg["adios_file"],
            spherical=True,
            solution_var="h",
            friction_law="mannings",
            wd=True,
            wd_alpha=1.5,
            dt=dt,
            bathy_adjustment=0,
            nt=nt_total,
            dramp=2.0,
            forcing=forcing,
        )
        solver = get_solver("DG")(problem, theta=1.0, p_degree=[1, 1])
        solver_params = get_default_solver_params(
            rtol=1e-5,
            atol=1e-6,
            max_it=20,
            relaxation_parameter=1.0,
            comm=self.comm,
            error_if_not_converged=False,
        )
        solver.time_loop(
            solver_parameters=solver_params,
            stations=[],
            plot_every=9999,
            save_state=True,
            store_jacobians=False,
            enable_video=False,
            monitor_progress=False,
        )

        if obs_points is None:
            obs_points = self._make_observation_points(
                problem.mesh,
                problem_bundle["obs_fraction"],
                problem_bundle["obs_seed"],
            )
        obs_operator = PointObservationOperator(solver.V, obs_points, comm=self.comm)
        obs_times = list(
            range(
                problem_bundle["nt_ramp"],
                problem_bundle["nt_total"] + 1,
                problem_bundle["obs_frequency"],
            )
        )
        u_owned_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs
        obs_matrix = np.zeros((len(obs_times), len(obs_points)))
        for i, obs_time in enumerate(obs_times):
            vec = la.create_petsc_vector(
                solver.V.dofmap.index_map,
                solver.V.dofmap.index_map_bs,
            )
            state_array = solver.storage.saved_states[obs_time]
            vec.setArray(state_array[:u_owned_size])
            vec.assemble()
            obs_vec = obs_operator.forward(vec)
            obs_matrix[i] = obs_vec.getArray().copy()
            obs_vec.destroy()
            vec.destroy()

        solver.storage.clear()
        del solver, problem, forcing
        gc.collect()

        return {
            "obs_matrix": obs_matrix,
            "obs_points": obs_points,
            "obs_times": obs_times,
            "summary": {
                "parameter_names": self.spec.control_variables,
                "parameter_values": theta.tolist(),
            },
        }


class ManningsNRunner(FiniteDifferenceParameterRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = self.method or "4dvar"
        self._augmented_iteration_history: list[dict[str, Any]] = []
        self._state_truth: np.ndarray | None = None
        self._theta_truth: np.ndarray | None = None
        self._control_layout = None
        self._cost_function = None

    def _profile_event(self, event: str, **payload) -> None:
        self.runtime_profiler.log_event(
            event,
            experiment=self.spec.name,
            method=self.method,
            **payload,
        )

    def truth_parameters(self) -> np.ndarray:
        return np.asarray(self.spec.forward_model_config["truth_coefficients"], dtype=float)

    def background_parameters(self) -> np.ndarray:
        return np.asarray(self.spec.forward_model_config["background_coefficients"], dtype=float)

    def parameter_stds(self) -> np.ndarray:
        return np.asarray(self.spec.forward_model_config["parameter_std"], dtype=float)

    def fd_steps(self) -> np.ndarray:
        return np.asarray(self.spec.forward_model_config["fd_steps"], dtype=float)

    def parameter_bounds(self):
        return [tuple(bound) for bound in self.spec.forward_model_config["bounds"]]

    def _background_weight(self) -> float:
        return float(self.spec.forward_model_config["regularization_weight"])

    def construct_problem_bundle(self) -> dict[str, Any]:
        bundle = super().construct_problem_bundle()
        bundle["background_seed"] = int(self.spec.noise_model.get("background_seed", 123))
        bundle["background_error_std"] = float(
            self.spec.noise_model.get("background_error_std", 0.02)
        )
        bundle["basis_shape"] = tuple(self.spec.forward_model_config["basis_shape"])
        bundle["basis_width_fraction"] = float(
            self.spec.forward_model_config["basis_width_fraction"]
        )
        bundle["n_bounds"] = tuple(self.spec.forward_model_config["n_clip"])
        if self.method not in {"4dvar", "dcwme", "dcwme_static", "dcwme_dynamic"}:
            raise ValueError(
                f"Unsupported method '{self.method}' for augmented Manning experiment"
            )
        return bundle

    def solve_inverse(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        self.log_stage(
            "Inverse Solve",
            f"running {self.method.upper()} on the packed [u0; theta_n] control",
        )
        self._profile_event(
            "mannings.solve_inverse.start",
            output_root=str(self.output_paths["root"]),
            nt_total=int(problem_bundle["nt_total"]),
            nt_ramp=int(problem_bundle["nt_ramp"]),
            obs_fraction=float(problem_bundle["obs_fraction"]),
            obs_frequency=int(problem_bundle["obs_frequency"]),
            max_iterations=int(problem_bundle["max_iterations"]),
        )
        from petsc4py import PETSc

        from swe4dvar.control import ControlLayout
        from swe4dvar.data_assimilation import (
            BlockDiagonalCovariance,
            DiagonalCovariance,
            PointObservationOperator,
            create_cost_function,
        )
        from swe4dvar.optimization.petsc_tao_wrapper import PETScTAOWrapper

        assimilation = self._build_augmented_assimilation_objects(problem_bundle)
        solver = assimilation["solver"]
        problem = assimilation["problem"]
        controller = assimilation["controller"]
        solver_params = assimilation["solver_params"]

        state_truth = solver.u_n.x.array.copy()
        theta_truth = self.truth_parameters()
        state_background = self._build_background_state(
            state_truth,
            problem_bundle["background_error_std"],
            problem_bundle["background_seed"],
        )
        theta_background = self.background_parameters()

        self._state_truth = state_truth.copy()
        self._theta_truth = theta_truth.copy()

        layout = ControlLayout(
            state_size=state_truth.size,
            theta_size=theta_background.size,
            comm=PETSc.COMM_SELF,
        )
        self._control_layout = layout

        from experiments.twin_experiment import ForwardModelWrapper

        forward_model = ForwardModelWrapper(
            solver,
            problem,
            solver_params,
            control_layout=layout,
            parameter_controller=controller,
            t_start=0.0,
        )

        m_background = layout.create_petsc_vec(
            u0=state_background,
            theta=theta_background,
        )
        m_truth = layout.create_petsc_vec(
            u0=state_truth,
            theta=theta_truth,
        )

        obs_operator = PointObservationOperator(
            solver.V,
            observation_bundle["obs_points"],
            comm=self.comm,
        )
        observations = self._create_observation_vectors(observation_bundle["noisy_observations"])
        observation_cov = self._create_observation_covariances(
            observation_bundle["obs_times"],
            observation_bundle["noise_stds"],
            observation_bundle["noisy_observations"].shape[1],
        )

        state_scale = np.maximum(np.abs(state_truth), 1e-3)
        state_variances = (problem_bundle["background_error_std"] * state_scale) ** 2
        theta_variances = (self.parameter_stds() ** 2) / max(self._background_weight(), 1e-30)
        B_state = DiagonalCovariance(
            PETSc.COMM_SELF,
            size=state_truth.size,
            diagonal=state_variances,
        )
        B_theta = DiagonalCovariance(
            PETSc.COMM_SELF,
            size=theta_background.size,
            diagonal=theta_variances,
        )
        background_cov = BlockDiagonalCovariance([B_state, B_theta], comm=PETSc.COMM_SELF)

        resolved_method = self._resolve_mannings_method(self.method)
        static_l_wme = None
        if resolved_method == "dcwme_static":
            from dolfinx import la

            static_start = time.perf_counter()
            state_template = la.create_petsc_vector(
                solver.V.dofmap.index_map,
                solver.V.dofmap.index_map_bs,
            )
            state_template.setArray(np.asarray(state_background, dtype=float).copy())
            state_template.assemble()
            static_l_wme = self._compute_static_l_wme(
                obs_operator=obs_operator,
                background_cov_state=B_state,
                m_template=state_template,
                n_obs_times=len(observation_bundle["obs_times"]),
                noise_stds=observation_bundle["noise_stds"],
            )
            state_template.destroy()
            self._profile_event(
                "mannings.solve_inverse.static_l_wme_ready",
                duration_s=float(time.perf_counter() - static_start),
            )

        cost_kwargs = {
            "forward_model": forward_model,
            "observation_operator": obs_operator,
            "background_cov": background_cov,
            "observation_cov": observation_cov,
            "m_background": m_background,
            "observations": observations,
            "obs_times": observation_bundle["obs_times"],
            "predictability_gamma": 0.1,
            "adaptive_gamma": True,
        }
        if resolved_method == "dcwme_static":
            cost_kwargs["predicted_cov_wme"] = static_l_wme
            cost_kwargs["auto_inflate_B"] = False

        cost_function = create_cost_function(
            "wme" if resolved_method.startswith("dcwme") else "4dvar",
            **cost_kwargs,
        )
        self._cost_function = cost_function
        cost_function.runtime_profiler = self.runtime_profiler
        self._profile_event(
            "mannings.solve_inverse.cost_function_ready",
            resolved_method=resolved_method,
            cost_function_type=type(cost_function).__name__,
        )

        lower_bounds, upper_bounds = self._create_augmented_bounds(layout, state_truth.size)
        optimizer = PETScTAOWrapper(
            cost_function,
            tao_type="blmvm",
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            options={
                "max_iterations": problem_bundle["max_iterations"],
                "gradient_tolerance": 1e-6,
                "cost_tolerance": 1e-8,
                "verbose": self.rank == 0,
                "line_search_initial_step": 1.0,
                "line_search_max_funcs": 50,
                "iteration_callback": self._iteration_callback,
                "runtime_profiler": self.runtime_profiler,
                "runtime_profile_label": f"tao.{resolved_method}",
            },
        )

        solve_start = time.perf_counter()
        m_analysis = optimizer.solve(m_background.copy())
        self._profile_event(
            "mannings.solve_inverse.optimizer_complete",
            duration_s=float(time.perf_counter() - solve_start),
            converged=bool(optimizer.converged),
            iterations=int(optimizer.iteration),
        )
        analysis_control = layout.unpack_vec(m_analysis)

        final_start = time.perf_counter()
        final_result = cost_function.value_detailed(m_analysis)
        self._profile_event(
            "mannings.solve_inverse.value_detailed_complete",
            duration_s=float(time.perf_counter() - final_start),
            value=float(final_result.value),
        )
        analysis_start = time.perf_counter()
        analysis_trajectory, _ = forward_model.solve(m_analysis, store_jacobians=False)
        self._profile_event(
            "mannings.solve_inverse.analysis_forward_complete",
            duration_s=float(time.perf_counter() - analysis_start),
            trajectory_len=len(analysis_trajectory),
        )
        sample_start = time.perf_counter()
        obs_prediction = self._sample_observation_matrix(
            analysis_trajectory,
            observation_bundle["obs_times"],
            obs_operator,
        )
        self._profile_event(
            "mannings.solve_inverse.sample_observations_complete",
            duration_s=float(time.perf_counter() - sample_start),
        )

        solve_bundle = {
            "resolved_method": resolved_method,
            "m_background": m_background,
            "m_truth": m_truth,
            "m_analysis": m_analysis,
            "background_control": {
                "u0": state_background.tolist(),
                "theta": theta_background.tolist(),
            },
            "analysis_control": {
                "u0": analysis_control.u0.tolist(),
                "theta": analysis_control.theta.tolist(),
            },
            "truth_control": {
                "u0": state_truth.tolist(),
                "theta": theta_truth.tolist(),
            },
            "cost_breakdown": {
                "value": float(final_result.value),
                "background_term": float(final_result.background_term),
                "observation_term": float(final_result.observation_term),
                "predictability_term": float(final_result.predictability_term),
            },
            "optimizer": {
                "converged": bool(optimizer.converged),
                "iterations": int(optimizer.iteration),
                "history": optimizer.convergence_history,
            },
            "analysis_observation_rmse": float(
                rmse(obs_prediction, observation_bundle["truth_observations"])
            ),
            "state_background_rmse": float(rmse(state_background, state_truth)),
            "state_analysis_rmse": float(rmse(analysis_control.u0, state_truth)),
            "parameter_background_rmse": float(rmse(theta_background, theta_truth)),
            "parameter_analysis_rmse": float(rmse(analysis_control.theta, theta_truth)),
            "analysis_mannings_summary": self._summarize_mannings_field(
                controller,
                analysis_control.theta,
            ),
            "background_mannings_summary": self._summarize_mannings_field(
                controller,
                theta_background,
            ),
            "truth_mannings_summary": self._summarize_mannings_field(
                controller,
                theta_truth,
            ),
        }

        return solve_bundle

    def collect_diagnostics(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
        solve_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        self.log_stage("Diagnostics", "assembling augmented-control summaries")
        diagnostics = {
            "convergence": self._augmented_iteration_history,
            "cost_breakdown": solve_bundle["cost_breakdown"],
            "background_vs_truth": {
                "state_rmse": solve_bundle["state_background_rmse"],
                "parameter_rmse": solve_bundle["parameter_background_rmse"],
            },
            "analysis_vs_truth": {
                "state_rmse": solve_bundle["state_analysis_rmse"],
                "parameter_rmse": solve_bundle["parameter_analysis_rmse"],
                "observation_rmse": solve_bundle["analysis_observation_rmse"],
            },
            "mannings_field": {
                "truth": solve_bundle["truth_mannings_summary"],
                "background": solve_bundle["background_mannings_summary"],
                "analysis": solve_bundle["analysis_mannings_summary"],
            },
        }
        diagnostics["spectral"] = self._collect_spectral_diagnostics()
        return diagnostics

    def save_outputs(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
        solve_bundle: dict[str, Any],
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        self.log_stage("Results Save", "writing augmented Manning outputs")
        metrics = {
            "experiment": self.spec.name,
            "inverse_problem_type": self.spec.inverse_problem_type,
            "method": self.method,
            "resolved_method": solve_bundle["resolved_method"],
            "control_variables": self.spec.control_variables,
            "state_background_rmse": solve_bundle["state_background_rmse"],
            "state_analysis_rmse": solve_bundle["state_analysis_rmse"],
            "parameter_background_rmse": solve_bundle["parameter_background_rmse"],
            "parameter_analysis_rmse": solve_bundle["parameter_analysis_rmse"],
            "observation_rmse": solve_bundle["analysis_observation_rmse"],
            "converged": solve_bundle["optimizer"]["converged"],
            "num_iterations": solve_bundle["optimizer"]["iterations"],
            "cost_breakdown": solve_bundle["cost_breakdown"],
            "spectral": diagnostics["spectral"],
        }

        if self.rank == 0:
            write_json(self.output_paths["root"] / "metrics.json", metrics)
            write_json(
                self.output_paths["diagnostics"] / "convergence.json",
                diagnostics["convergence"],
            )
            write_json(
                self.output_paths["diagnostics"] / "cost_breakdown.json",
                diagnostics["cost_breakdown"],
            )
            write_json(
                self.output_paths["diagnostics"] / "spectral.json",
                diagnostics["spectral"],
            )
            write_json(
                self.output_paths["diagnostics"] / "stage_log.json",
                self.stage_log,
            )
            write_json(
                self.output_paths["trajectories"] / "control_history.json",
                {
                    "truth": solve_bundle["truth_control"],
                    "background": solve_bundle["background_control"],
                    "analysis": solve_bundle["analysis_control"],
                },
            )
            write_json(
                self.output_paths["trajectories"] / "mannings_field_summary.json",
                diagnostics["mannings_field"],
            )

        return metrics

    def _build_augmented_assimilation_objects(self, problem_bundle: dict[str, Any]) -> dict[str, Any]:
        from swe4dvar.forward import ADCIRCProblem, ManningsBasisController, get_solver
        from swe4dvar.physics.forcing import GriddedForcing
        from swe4dvar.utils import get_default_solver_params

        cfg = self.spec.forward_model_config
        forcing = GriddedForcing(str(self._ensure_truth_wind_file(problem_bundle)), lat0=35)
        problem = ADCIRCProblem(
            adios_file=cfg["adios_file"],
            spherical=True,
            solution_var="h",
            friction_law="mannings",
            wd=True,
            wd_alpha=1.5,
            dt=float(cfg["dt"]),
            bathy_adjustment=0,
            nt=int(cfg["nt_total"]),
            dramp=2.0,
            forcing=forcing,
        )
        solver = get_solver(cfg.get("solver", "DG"))(problem, theta=1.0, p_degree=[1, 1])
        controller = ManningsBasisController(
            basis_shape=tuple(cfg["basis_shape"]),
            basis_width_fraction=float(cfg["basis_width_fraction"]),
            n_bounds=tuple(cfg["n_clip"]),
            reference_n=0.02,
        )
        controller.bind(problem, solver)
        solver_params = get_default_solver_params(
            rtol=1e-5,
            atol=1e-6,
            max_it=20,
            relaxation_parameter=1.0,
            comm=self.comm,
            error_if_not_converged=False,
        )
        return {
            "forcing": forcing,
            "problem": problem,
            "solver": solver,
            "controller": controller,
            "solver_params": solver_params,
        }

    def _ensure_truth_wind_file(self, problem_bundle: dict[str, Any]) -> Path:
        from experiments.shinnecock_study.wind_models import (
            DEFAULT_TRACK,
            HollandHurricaneConfig,
            WindGridConfig,
            generate_holland_wind_field,
            write_wind_hdf5,
        )

        cfg = self.spec.forward_model_config
        dt = float(cfg["dt"])
        nt_total = int(cfg["nt_total"])
        times = np.arange(0.0, (nt_total + 1) * dt, dt)
        wind_dir = self.output_paths["diagnostics"] / "wind_fields"
        if self.rank == 0:
            wind_dir.mkdir(exist_ok=True)
        wind_file = wind_dir / "mannings_truth_wind.h5"
        if self.rank == 0 and not wind_file.exists():
            grid = WindGridConfig()
            wind_cfg = HollandHurricaneConfig(
                track_waypoints=DEFAULT_TRACK,
                Vmax=float(cfg["truth_vmax"]),
            )
            windx, windy, pressure = generate_holland_wind_field(
                wind_cfg,
                grid,
                times,
                wind_ramp_s=float(cfg["wind_ramp_s"]),
            )
            write_wind_hdf5(str(wind_file), grid, times, windx, windy, pressure)
        self.comm.Barrier()
        return wind_file

    def _create_observation_vectors(self, obs_matrix: np.ndarray):
        from petsc4py import PETSc

        vectors = []
        for row in np.asarray(obs_matrix, dtype=float):
            vec = PETSc.Vec().createSeq(row.size, comm=PETSc.COMM_SELF)
            vec.setArray(row.copy())
            vec.assemble()
            vectors.append(vec)
        return vectors

    def _create_observation_covariances(self, obs_times, noise_stds, n_obs: int):
        from petsc4py import PETSc

        from swe4dvar.data_assimilation import DiagonalCovariance

        covariances = {}
        for time_index, sigma in zip(obs_times, noise_stds):
            covariances[time_index] = DiagonalCovariance(
                PETSc.COMM_SELF,
                size=n_obs,
                variance=float(sigma**2),
            )
        return covariances

    def _build_background_state(self, truth_state: np.ndarray, std: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        scale = np.maximum(np.abs(truth_state), 1e-3)
        return np.asarray(truth_state, dtype=float) + rng.normal(
            0.0,
            std * scale,
            size=truth_state.shape,
        )

    def _create_augmented_bounds(self, layout, state_size: int):
        lower = layout.create_petsc_vec(
            packed=np.concatenate(
                [
                    np.full(state_size, -1e20, dtype=float),
                    np.array([b[0] for b in self.parameter_bounds()], dtype=float),
                ]
            )
        )
        upper = layout.create_petsc_vec(
            packed=np.concatenate(
                [
                    np.full(state_size, 1e20, dtype=float),
                    np.array([b[1] for b in self.parameter_bounds()], dtype=float),
                ]
            )
        )
        return lower, upper

    def _sample_observation_matrix(self, trajectory, obs_times, obs_operator) -> np.ndarray:
        obs_matrix = np.zeros((len(obs_times), obs_operator.obs_points.shape[0]))
        for i, obs_time in enumerate(obs_times):
            obs_vec = obs_operator.forward(trajectory[obs_time])
            obs_matrix[i] = obs_vec.getArray(readonly=True).copy()
            obs_vec.destroy()
        return obs_matrix

    def _summarize_mannings_field(self, controller, theta: np.ndarray) -> dict[str, float]:
        field = controller.evaluate_field(theta)
        return {
            "min": float(np.min(field)),
            "max": float(np.max(field)),
            "mean": float(np.mean(field)),
        }

    def _iteration_callback(self, x, iteration: int, cost: float, grad_norm: float):
        if self._control_layout is None or self._state_truth is None or self._theta_truth is None:
            return None
        control = self._control_layout.unpack_vec(x)
        entry = {
            "state_rmse": float(rmse(control.u0, self._state_truth)),
            "parameter_rmse": float(rmse(control.theta, self._theta_truth)),
        }
        self._augmented_iteration_history.append(
            {
                "iteration": int(iteration),
                "cost": float(cost),
                "gradient_norm": float(grad_norm),
                **entry,
            }
        )
        self._profile_event(
            "mannings.iteration",
            iteration=int(iteration),
            cost=float(cost),
            gradient_norm=float(grad_norm),
            **entry,
        )
        return entry

    def _collect_spectral_diagnostics(self) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}
        if self._cost_function is None:
            return diagnostics
        if hasattr(self._cost_function, "B") and hasattr(self._cost_function.B, "blocks"):
            blocks = self._cost_function.B.blocks
            if len(blocks) >= 2:
                state_diag = blocks[0].diagonal.getArray(readonly=True).copy()
                theta_diag = blocks[1].diagonal.getArray(readonly=True).copy()
                diagnostics["state_background_covariance"] = summarize_spectrum(state_diag)
                diagnostics["parameter_background_covariance"] = summarize_spectrum(theta_diag)
        gram_eigs = getattr(self._cost_function, "_gram_eigenvalues", None)
        if gram_eigs is not None:
            diagnostics["gram_matrix"] = summarize_spectrum(gram_eigs)
        L_wme = getattr(self._cost_function, "_L_wme", None)
        if L_wme is not None and hasattr(L_wme, "min_eigenvalue") and hasattr(L_wme, "max_eigenvalue"):
            diagnostics["l_wme"] = {
                "lambda_min": float(L_wme.min_eigenvalue()),
                "lambda_max": float(L_wme.max_eigenvalue()),
                "condition_number": float(
                    L_wme.max_eigenvalue() / max(L_wme.min_eigenvalue(), 1e-30)
                ),
            }
        return diagnostics

    def _run_parameterized_forward(
        self,
        *,
        theta: np.ndarray,
        problem_bundle: dict[str, Any],
        run_label: str,
        obs_points=None,
    ) -> dict[str, Any]:
        from dolfinx import la
        from swe4dvar.data_assimilation import PointObservationOperator
        assimilation = self._build_augmented_assimilation_objects(problem_bundle)
        solver = assimilation["solver"]
        problem = assimilation["problem"]
        controller = assimilation["controller"]
        solver_params = assimilation["solver_params"]

        controller.apply(problem, solver, np.asarray(theta, dtype=float))
        solver.time_loop(
            solver_parameters=solver_params,
            stations=[],
            plot_every=9999,
            save_state=True,
            store_jacobians=False,
            enable_video=False,
            monitor_progress=False,
        )

        if obs_points is None:
            obs_points = self._make_observation_points(
                problem.mesh,
                problem_bundle["obs_fraction"],
                problem_bundle["obs_seed"],
            )
        obs_operator = PointObservationOperator(solver.V, obs_points, comm=self.comm)
        obs_times = list(
            range(
                int(problem_bundle["nt_ramp"]),
                int(problem_bundle["nt_total"]) + 1,
                int(problem_bundle["obs_frequency"]),
            )
        )
        u_owned_size = solver.V.dofmap.index_map.size_local * solver.V.dofmap.index_map_bs
        obs_matrix = np.zeros((len(obs_times), len(obs_points)))
        for i, obs_time in enumerate(obs_times):
            vec = la.create_petsc_vector(
                solver.V.dofmap.index_map,
                solver.V.dofmap.index_map_bs,
            )
            state_array = solver.storage.saved_states[obs_time]
            vec.setArray(state_array[:u_owned_size])
            vec.assemble()
            obs_vec = obs_operator.forward(vec)
            obs_matrix[i] = obs_vec.getArray().copy()
            obs_vec.destroy()
            vec.destroy()

        solver.storage.clear()
        field_values = controller.evaluate_field(np.asarray(theta, dtype=float))

        forcing = assimilation.get("forcing")
        del solver, problem, controller, forcing
        gc.collect()

        return {
            "obs_matrix": obs_matrix,
            "obs_points": obs_points,
            "obs_times": obs_times,
            "summary": {
                "parameter_names": self.spec.control_variables,
                "coefficient_values": theta.tolist(),
                "mannings_n_min": float(np.min(field_values)),
                "mannings_n_max": float(np.max(field_values)),
                "mannings_n_mean": float(np.mean(field_values)),
            },
        }

    def _build_gaussian_basis(self, coords: np.ndarray) -> np.ndarray:
        cfg = self.spec.forward_model_config
        nx, ny = cfg["basis_shape"]
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        centers_x = np.linspace(x_min, x_max, nx)
        centers_y = np.linspace(y_min, y_max, ny)
        centers = np.array([(x, y) for y in centers_y for x in centers_x], dtype=float)
        width = cfg["basis_width_fraction"] * max(x_max - x_min, y_max - y_min)
        basis = np.zeros((coords.shape[0], centers.shape[0]), dtype=float)
        for j, center in enumerate(centers):
            distances_sq = np.sum((coords - center) ** 2, axis=1)
            basis[:, j] = np.exp(-0.5 * distances_sq / max(width ** 2, 1e-30))
        row_sums = np.maximum(np.sum(basis, axis=1, keepdims=True), 1e-30)
        return basis / row_sums

    @staticmethod
    def _resolve_mannings_method(method: str) -> str:
        if method == "dcwme":
            return "dcwme_dynamic"
        return method

    def _compute_static_l_wme(
        self,
        *,
        obs_operator,
        background_cov_state,
        m_template,
        n_obs_times: int,
        noise_stds: np.ndarray,
    ):
        from experiments.shinnecock_study.run_comparison import _compute_static_L_wme

        noise_stds = np.asarray(noise_stds, dtype=float)
        if noise_stds.size == 0:
            raise ValueError("Cannot compute static L_wme with no observation noise entries")
        obs_variance = float(np.mean(noise_stds**2))
        static_l_wme, _ = _compute_static_L_wme(
            obs_operator=obs_operator,
            B=background_cov_state,
            n_obs_times=n_obs_times,
            obs_variance=obs_variance,
            m_template=m_template,
            predictability_gamma=0.1,
            adaptive_gamma=True,
            comm=self.comm,
            rank=self.rank,
        )
        return static_l_wme


class ManningsNSweepRunner(BaseTwinExperimentRunner):
    """Registry-driven sweep over the augmented state-plus-Manning experiment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = self.method or "all"

    def run(self) -> dict[str, Any]:
        self.log_stage("ExperimentSpec", "serializing Manning sweep configuration")
        self._write_config()

        problem_bundle = self.construct_problem_bundle()
        case_manifest = self._build_case_manifest(problem_bundle)

        if self.dry_run:
            payload = {
                "status": "dry_run",
                "experiment": self.spec.name,
                "method": self.method,
                "single_case": bool(problem_bundle.get("single_case", False)),
                "cheap_static_case": bool(problem_bundle.get("cheap_static_case", False)),
                "num_cases": len(case_manifest),
                "methods": problem_bundle["methods"],
                "regularization_weights": problem_bundle["regularization_weights"],
                "noise_levels": problem_bundle["noise_levels"],
                "window_lengths": problem_bundle["window_lengths"],
                "case_manifest": case_manifest,
                "stage_log": self.stage_log,
            }
            if self.rank == 0:
                write_json(self.output_paths["root"] / "metrics.json", payload)
                write_json(
                    self.output_paths["diagnostics"] / "case_manifest.json",
                    case_manifest,
                )
            return payload

        self.log_stage(
            "Inverse Solve",
            (
                "running a single resolved Manning case"
                if problem_bundle.get("single_case", False)
                else "running the full Manning sweep over regularization, noise, window length, and method"
            ),
        )
        raw_case_root = self.output_paths["diagnostics"] / "case_runs"
        if self.rank == 0:
            raw_case_root.mkdir(exist_ok=True)

        case_results = []
        for case in case_manifest:
            case_spec = self._build_case_spec(case)
            case_output_root = raw_case_root / case["case_id"]
            runner = ManningsNRunner(
                case_spec,
                method=case["method"],
                output_root=str(case_output_root),
                dry_run=False,
            )
            result = runner.run()
            case_results.append(
                {
                    **case,
                    "output_root": str(case_output_root),
                    "metrics": result,
                }
            )

        diagnostics = self.collect_diagnostics(problem_bundle, {}, {}, {"cases": case_results})
        return self.save_outputs(problem_bundle, {}, {}, {"cases": case_results}, diagnostics)

    def construct_problem_bundle(self) -> dict[str, Any]:
        self.log_stage("Problem Construction", "building Manning sweep grid from the registry")
        methods = ["4dvar", "dcwme_static", "dcwme_dynamic"] if self.method == "all" else [self.method]
        unknown_methods = [
            m for m in methods if m not in {"4dvar", "dcwme", "dcwme_static", "dcwme_dynamic"}
        ]
        if unknown_methods:
            raise ValueError(f"Unsupported Manning sweep method(s): {unknown_methods}")

        sweep = self.spec.sweep_parameters
        reg_weights = [float(v) for v in sweep.get("regularization_weight", [])]
        noise_levels = [float(v) for v in sweep.get("noise_levels", [])]
        window_lengths = [int(v) for v in sweep.get("window_lengths", [])]

        if not reg_weights:
            reg_weights = [float(self.spec.forward_model_config["regularization_weight"])]
        if not noise_levels:
            noise_levels = [float(self.spec.noise_model["obs_noise_level"])]
        if not window_lengths:
            default_window = int(
                self.spec.forward_model_config["nt_total"]
                - self.spec.forward_model_config["nt_ramp"]
            )
            window_lengths = [default_window]

        single_case = bool(self.case_overrides.get("single_case", False))
        cheap_static_case = bool(self.case_overrides.get("cheap_static_case", False))
        reg_override = self.case_overrides.get("regularization_weight")
        noise_override = self.case_overrides.get("obs_noise_level")
        window_override = self.case_overrides.get("window_length")
        obs_fraction_override = self.case_overrides.get("obs_fraction")
        obs_frequency_override = self.case_overrides.get("obs_frequency")
        max_iterations_override = self.case_overrides.get("max_iterations")

        if any(
            value is not None
            for value in (
                reg_override,
                noise_override,
                window_override,
                obs_fraction_override,
                obs_frequency_override,
                max_iterations_override,
            )
        ):
            single_case = True
        if cheap_static_case:
            single_case = True

        if reg_override is not None:
            reg_weights = [float(reg_override)]
        elif single_case:
            reg_weights = [float(self.spec.forward_model_config["regularization_weight"])]

        if noise_override is not None:
            noise_levels = [float(noise_override)]
        elif single_case:
            noise_levels = [float(self.spec.noise_model["obs_noise_level"])]

        if window_override is not None:
            window_lengths = [int(window_override)]
        elif cheap_static_case:
            window_lengths = [4]
        elif single_case:
            default_window = int(
                self.spec.forward_model_config["nt_total"]
                - self.spec.forward_model_config["nt_ramp"]
            )
            window_lengths = [default_window]

        return {
            "methods": methods,
            "regularization_weights": reg_weights,
            "noise_levels": noise_levels,
            "window_lengths": window_lengths,
            "single_case": single_case,
            "cheap_static_case": cheap_static_case,
            "obs_fraction_override": (
                float(obs_fraction_override) if obs_fraction_override is not None else None
            ),
            "obs_frequency_override": (
                int(obs_frequency_override) if obs_frequency_override is not None else None
            ),
            "max_iterations_override": (
                int(max_iterations_override) if max_iterations_override is not None else None
            ),
        }

    def _build_case_manifest(self, problem_bundle: dict[str, Any]) -> list[dict[str, Any]]:
        cases = []
        for reg_weight, noise_level, window_length, method in itertools.product(
            problem_bundle["regularization_weights"],
            problem_bundle["noise_levels"],
            problem_bundle["window_lengths"],
            problem_bundle["methods"],
        ):
            reg_label = self._format_float_label(reg_weight)
            noise_label = self._format_float_label(noise_level)
            case_id = (
                f"reg_{reg_label}"
                f"__noise_{noise_label}"
                f"__window_{int(window_length)}"
                f"__{method}"
            )
            cases.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "regularization_weight": float(reg_weight),
                    "obs_noise_level": float(noise_level),
                    "window_length": int(window_length),
                }
            )
        return cases

    def _build_case_spec(self, case: dict[str, Any]):
        forward_cfg = dict(self.spec.forward_model_config)
        forward_cfg["regularization_weight"] = float(case["regularization_weight"])
        forward_cfg["nt_total"] = int(forward_cfg["nt_ramp"]) + int(case["window_length"])
        if self.case_overrides.get("cheap_static_case", False):
            forward_cfg["max_iterations"] = 2
        if self.case_overrides.get("max_iterations") is not None:
            forward_cfg["max_iterations"] = int(self.case_overrides["max_iterations"])

        noise_cfg = dict(self.spec.noise_model)
        noise_cfg["obs_noise_level"] = float(case["obs_noise_level"])

        observation_cfg = dict(self.spec.observation_config)
        if self.case_overrides.get("cheap_static_case", False):
            observation_cfg["obs_fraction"] = 0.02
            observation_cfg["obs_frequency"] = 4
        if self.case_overrides.get("obs_fraction") is not None:
            observation_cfg["obs_fraction"] = float(self.case_overrides["obs_fraction"])
        if self.case_overrides.get("obs_frequency") is not None:
            observation_cfg["obs_frequency"] = int(self.case_overrides["obs_frequency"])

        return replace(
            self.spec,
            forward_model_config=forward_cfg,
            observation_config=observation_cfg,
            noise_model=noise_cfg,
        )

    def collect_diagnostics(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
        solve_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        self.log_stage("Diagnostics", "aggregating Manning sweep metrics across all grid points")
        summary_rows = []
        for case in solve_bundle["cases"]:
            metrics = case["metrics"]
            summary_rows.append(
                {
                    "case_id": case["case_id"],
                    "method": case["method"],
                    "resolved_method": case["metrics"].get("resolved_method"),
                    "regularization_weight": case["regularization_weight"],
                    "obs_noise_level": case["obs_noise_level"],
                    "window_length": case["window_length"],
                    "converged": metrics.get("converged"),
                    "num_iterations": metrics.get("num_iterations"),
                    "state_background_rmse": metrics.get("state_background_rmse"),
                    "state_analysis_rmse": metrics.get("state_analysis_rmse"),
                    "parameter_background_rmse": metrics.get("parameter_background_rmse"),
                    "parameter_analysis_rmse": metrics.get("parameter_analysis_rmse"),
                    "observation_rmse": metrics.get("observation_rmse"),
                    "cost_breakdown": metrics.get("cost_breakdown"),
                    "spectral": metrics.get("spectral"),
                }
            )

        return {
            "summary_rows": summary_rows,
            "case_manifest": [
                {
                    "case_id": case["case_id"],
                    "method": case["method"],
                    "resolved_method": case["metrics"].get("resolved_method"),
                    "regularization_weight": case["regularization_weight"],
                    "obs_noise_level": case["obs_noise_level"],
                    "window_length": case["window_length"],
                    "output_root": case["output_root"],
                }
                for case in solve_bundle["cases"]
            ],
        }

    def save_outputs(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
        solve_bundle: dict[str, Any],
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        self.log_stage("Results Save", "writing aggregated Manning sweep outputs")
        metrics = {
            "experiment": self.spec.name,
            "inverse_problem_type": self.spec.inverse_problem_type,
            "methods": problem_bundle["methods"],
            "regularization_weights": problem_bundle["regularization_weights"],
            "noise_levels": problem_bundle["noise_levels"],
            "window_lengths": problem_bundle["window_lengths"],
            "num_cases": len(solve_bundle["cases"]),
            "summary_rows": diagnostics["summary_rows"],
        }
        if self.rank == 0:
            write_json(self.output_paths["root"] / "metrics.json", metrics)
            write_json(
                self.output_paths["diagnostics"] / "summary_rows.json",
                diagnostics["summary_rows"],
            )
            write_json(
                self.output_paths["diagnostics"] / "case_manifest.json",
                diagnostics["case_manifest"],
            )
            write_json(
                self.output_paths["diagnostics"] / "stage_log.json",
                self.stage_log,
            )
        return metrics

    @staticmethod
    def _format_float_label(value: float) -> str:
        text = f"{float(value):.8g}"
        return text.replace("-", "m").replace(".", "p")

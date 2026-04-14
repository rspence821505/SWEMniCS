"""Restored WSE wind-ramp experiment runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseTwinExperimentRunner
from .common import rank0_log, write_json


WSE_METHOD_MAP = {
    "4dvar": ("4dvar", "a", "N/A"),
    "dcwme_dynamic": ("dcwme", "b", "dynamic"),
    "dcwme_static": ("dcwme", "c", "static"),
}


class WSEWindRampRunner(BaseTwinExperimentRunner):
    """Restores the original state-estimation meaning of the wind-ramp experiment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = self.method or "all"

    def construct_problem_bundle(self) -> dict[str, Any]:
        self.log_stage("Problem Construction", "restoring the original WSE state-estimation sweep")
        cfg = self.spec.forward_model_config
        methods = cfg["methods"] if self.method == "all" else [self.method]
        unknown_methods = [m for m in methods if m not in WSE_METHOD_MAP]
        if unknown_methods:
            raise ValueError(f"Unsupported WSE method(s): {unknown_methods}")
        return {
            "adios_file": cfg["adios_file"],
            "dt": float(cfg["dt"]),
            "nt_ramp": int(cfg["nt_ramp"]),
            "nt_da": int(cfg["nt_da"]),
            "n_windows": int(cfg["n_windows"]),
            "wind_ramp_s": float(cfg["wind_ramp_s"]),
            "track_shift_km": float(cfg["track_shift_km"]),
            "vmax_values": list(cfg["vmax_values"]),
            "methods": methods,
            "obs_fraction": float(self.spec.observation_config["obs_fraction"]),
            "obs_frequency": int(self.spec.observation_config["obs_frequency"]),
            "obs_noise_level": float(self.spec.noise_model["obs_noise_level"]),
            "background_error_std": float(self.spec.noise_model["background_error_std"]),
            "raw_output_dir": self.output_paths["diagnostics"] / "raw_state_runs",
        }

    def generate_truth(self, problem_bundle: dict[str, Any]) -> dict[str, Any]:
        self.log_stage("Forward Model", "building truth and perturbed wind files")
        from experiments.shinnecock_study.wind_models import (
            DEFAULT_TRACK,
            HollandHurricaneConfig,
            WindGridConfig,
            generate_holland_wind_field,
            generate_perturbed_config,
            generate_zero_wind_field,
            write_wind_hdf5,
        )

        wind_dir = self.output_paths["diagnostics"] / "wind_fields"
        if self.rank == 0:
            wind_dir.mkdir(exist_ok=True)

        times = np.arange(
            0.0,
            (problem_bundle["nt_ramp"] + problem_bundle["nt_da"] + 1) * problem_bundle["dt"],
            problem_bundle["dt"],
        )

        wind_files = {}
        for vmax in problem_bundle["vmax_values"]:
            truth_file = wind_dir / f"holland_V{int(vmax)}_truth.h5"
            pert_file = wind_dir / f"holland_V{int(vmax)}_pert{problem_bundle['track_shift_km']:.0f}km.h5"
            if self.rank == 0 and not truth_file.exists():
                grid = WindGridConfig()
                if vmax == 0:
                    windx, windy, pressure = generate_zero_wind_field(grid, times)
                else:
                    config = HollandHurricaneConfig(
                        track_waypoints=DEFAULT_TRACK,
                        Vmax=float(vmax),
                    )
                    windx, windy, pressure = generate_holland_wind_field(
                        config,
                        grid,
                        times,
                        wind_ramp_s=problem_bundle["wind_ramp_s"],
                    )
                write_wind_hdf5(str(truth_file), grid, times, windx, windy, pressure)
            if self.rank == 0 and not pert_file.exists():
                if vmax == 0:
                    import shutil

                    shutil.copy(truth_file, pert_file)
                else:
                    grid = WindGridConfig()
                    config = HollandHurricaneConfig(
                        track_waypoints=DEFAULT_TRACK,
                        Vmax=float(vmax),
                    )
                    config_pert = generate_perturbed_config(
                        config,
                        "track_shift",
                        problem_bundle["track_shift_km"],
                    )
                    windx, windy, pressure = generate_holland_wind_field(
                        config_pert,
                        grid,
                        times,
                        wind_ramp_s=problem_bundle["wind_ramp_s"],
                    )
                    write_wind_hdf5(str(pert_file), grid, times, windx, windy, pressure)
            wind_files[vmax] = {
                "truth": str(truth_file),
                "perturbed": str(pert_file),
            }
        self.comm.Barrier()
        return {"wind_files": wind_files}

    def generate_observations(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        self.log_stage(
            "Observation Generation",
            "delegated to the original TwinExperiment state-estimation worker with fixed seeds",
        )
        return {
            "obs_fraction": problem_bundle["obs_fraction"],
            "obs_frequency": problem_bundle["obs_frequency"],
            "obs_noise_level": problem_bundle["obs_noise_level"],
            "background_error_std": problem_bundle["background_error_std"],
        }

    def solve_inverse(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        self.log_stage("Inverse Solve", "running the restored WSE state-estimation sweep")
        from experiments.shinnecock_study.run_comparison import _run_in_subprocess

        script_path = str(
            (Path(__file__).resolve().parents[1] / "shinnecock_study" / "run_comparison.py").resolve()
        )
        raw_dir = problem_bundle["raw_output_dir"]
        if self.rank == 0:
            raw_dir.mkdir(exist_ok=True)
            (raw_dir / "data").mkdir(exist_ok=True)

        all_results = []
        mem_limit_mb = 12.0 * 1024.0
        for vmax in problem_bundle["vmax_values"]:
            point = {"Vmax": vmax}
            for method_name in problem_bundle["methods"]:
                method, sub_label, l_wme_mode = WSE_METHOD_MAP[method_name]
                result_file = raw_dir / "data" / f"phase7_wind{int(vmax)}_{sub_label}_results.json"
                if result_file.exists():
                    with open(result_file) as handle:
                        point[method_name] = json.load(handle)
                    continue

                run_config = {
                    "output_dir": str(raw_dir),
                    "adios_file": problem_bundle["adios_file"],
                    "sub_label": sub_label,
                    "method": method,
                    "wind_truth_file": truth_bundle["wind_files"][vmax]["truth"],
                    "wind_perturbed_file": truth_bundle["wind_files"][vmax]["perturbed"],
                    "nt_da": problem_bundle["nt_da"],
                    "nt_ramp": problem_bundle["nt_ramp"],
                    "phase_prefix": f"7_wind{int(vmax)}_",
                    "sweep_params": {
                        "dt": problem_bundle["dt"],
                        "obs_noise_level": observation_bundle["obs_noise_level"],
                        "obs_fraction": observation_bundle["obs_fraction"],
                        "obs_frequency": observation_bundle["obs_frequency"],
                        "background_error_std": observation_bundle["background_error_std"],
                    },
                    "l_wme_mode": l_wme_mode,
                    "n_windows": problem_bundle["n_windows"],
                    "result_file": str(result_file),
                    "mem_limit_gb": 12.0,
                    "worker_type": "wind",
                }
                point[method_name] = _run_in_subprocess(run_config, mem_limit_mb, script_path)
            all_results.append(point)

        return {"points": all_results}

    def collect_diagnostics(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
        solve_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        self.log_stage("Diagnostics", "summarizing restored WSE sweep results")
        summary_rows = []
        for point in solve_bundle["points"]:
            vmax = point["Vmax"]
            for method_name in problem_bundle["methods"]:
                result = point.get(method_name, {})
                row = {
                    "Vmax": vmax,
                    "method": method_name,
                    "status": result.get("status"),
                    "error_reduction": result.get("results", {}).get("error_reduction"),
                    "analysis_error": result.get("results", {}).get("analysis_error"),
                    "background_error": result.get("results", {}).get("background_error"),
                    "num_iterations": result.get("results", {}).get("num_iterations"),
                    "gradient_norm_history": result.get("convergence", {}).get("gradient_norm_history", []),
                    "analysis_state_rmse_history": result.get("convergence", {}).get(
                        "analysis_state_rmse_history",
                        [],
                    ),
                    "spectral_diagnostics": result.get("dcwme_diagnostics", {}),
                }
                summary_rows.append(row)
        return {"summary_rows": summary_rows}

    def save_outputs(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
        solve_bundle: dict[str, Any],
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        self.log_stage("Results Save", "writing standardized WSE outputs")
        metrics = {
            "experiment": self.spec.name,
            "inverse_problem_type": self.spec.inverse_problem_type,
            "methods": problem_bundle["methods"],
            "vmax_values": problem_bundle["vmax_values"],
            "summary_rows": diagnostics["summary_rows"],
        }
        if self.rank == 0:
            write_json(self.output_paths["root"] / "metrics.json", metrics)
            write_json(
                self.output_paths["diagnostics"] / "raw_results.json",
                solve_bundle["points"],
            )
            write_json(
                self.output_paths["diagnostics"] / "summary_rows.json",
                diagnostics["summary_rows"],
            )
            write_json(
                self.output_paths["diagnostics"] / "stage_log.json",
                self.stage_log,
            )
            write_json(
                self.output_paths["trajectories"] / "wind_file_index.json",
                truth_bundle["wind_files"],
            )
        return metrics

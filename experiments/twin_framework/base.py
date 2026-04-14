"""Base runner for registry-driven twin experiments."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .common import MPI, RealtimeProfiler, ensure_standard_output_tree, rank0_log, write_json


class BaseTwinExperimentRunner:
    """Enforces the standard ExperimentSpec -> results pipeline."""

    def __init__(
        self,
        spec,
        *,
        method: str | None = None,
        output_root: str | None = None,
        dry_run: bool = False,
        tag: str | None = None,
        case_overrides: dict[str, Any] | None = None,
    ):
        self.spec = spec
        self.method = method
        self.dry_run = dry_run
        self.tag = tag
        self.case_overrides = dict(case_overrides or {})
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        base_output = Path(output_root) if output_root else Path(spec.output_dir)
        if tag:
            base_output = base_output / tag
        self.output_paths = ensure_standard_output_tree(base_output)
        self.stage_log: list[dict[str, Any]] = []
        self.runtime_profiler = RealtimeProfiler(
            self.output_paths["diagnostics"] / "runtime_profile.jsonl",
            enabled=(self.rank == 0),
        )

    def log_stage(self, stage: str, detail: str) -> None:
        entry = {"stage": stage, "detail": detail}
        self.stage_log.append(entry)
        rank0_log(self.comm, f"[{self.spec.name}] {stage}: {detail}")
        self.runtime_profiler.log_event(
            "stage",
            experiment=self.spec.name,
            stage=stage,
            detail=detail,
            method=self.method,
            tag=self.tag,
        )

    def run(self) -> dict[str, Any]:
        self.log_stage("ExperimentSpec", "serializing registry configuration")
        self._write_config()

        if self.dry_run:
            payload = {
                "status": "dry_run",
                "experiment": self.spec.name,
                "method": self.method,
                "stage_log": self.stage_log,
            }
            write_json(self.output_paths["root"] / "metrics.json", payload)
            return payload

        problem_bundle = self.construct_problem_bundle()
        truth_bundle = self.generate_truth(problem_bundle)
        observation_bundle = self.generate_observations(problem_bundle, truth_bundle)
        solve_bundle = self.solve_inverse(problem_bundle, truth_bundle, observation_bundle)
        diagnostics = self.collect_diagnostics(
            problem_bundle,
            truth_bundle,
            observation_bundle,
            solve_bundle,
        )
        result = self.save_outputs(
            problem_bundle,
            truth_bundle,
            observation_bundle,
            solve_bundle,
            diagnostics,
        )
        return result

    def _write_config(self) -> None:
        config_payload = {
            "spec": asdict(self.spec),
            "method": self.method,
            "dry_run": self.dry_run,
            "tag": self.tag,
            "case_overrides": self.case_overrides,
        }
        if self.rank == 0:
            write_json(self.output_paths["root"] / "config.json", config_payload)

    def construct_problem_bundle(self) -> dict[str, Any]:
        raise NotImplementedError

    def generate_truth(self, problem_bundle: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def generate_observations(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def solve_inverse(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def collect_diagnostics(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
        solve_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def save_outputs(
        self,
        problem_bundle: dict[str, Any],
        truth_bundle: dict[str, Any],
        observation_bundle: dict[str, Any],
        solve_bundle: dict[str, Any],
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

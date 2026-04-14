"""Runner selection and CLI-facing experiment dispatch."""

from __future__ import annotations

from .parameter_runners import ManningsNRunner, ManningsNSweepRunner, WindParameterRunner
from .registry import EXPERIMENT_REGISTRY
from .wse_runner import WSEWindRampRunner


RUNNER_REGISTRY = {
    "wse_wind_ramp": WSEWindRampRunner,
    "wind_param": WindParameterRunner,
    "mannings_n": ManningsNSweepRunner,
}


def run_registered_experiment(
    experiment_name: str,
    *,
    method: str | None = None,
    output_root: str | None = None,
    dry_run: bool = False,
    tag: str | None = None,
    case_overrides: dict | None = None,
):
    if experiment_name not in EXPERIMENT_REGISTRY:
        raise KeyError(f"Unknown experiment '{experiment_name}'")
    spec = EXPERIMENT_REGISTRY[experiment_name]
    runner_cls = RUNNER_REGISTRY[experiment_name]
    runner = runner_cls(
        spec,
        method=method,
        output_root=output_root,
        dry_run=dry_run,
        tag=tag,
        case_overrides=case_overrides,
    )
    return runner.run()

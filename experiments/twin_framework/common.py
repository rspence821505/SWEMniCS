"""Shared utilities for the standardized twin-experiment framework."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
try:
    from mpi4py import MPI  # type: ignore
except ModuleNotFoundError:
    class _SerialComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def Barrier(self):
            return None

        def bcast(self, value, root=0):
            return value

        def gather(self, value, root=0):
            return [value]

    class _SerialMPI:
        COMM_WORLD = _SerialComm()

    MPI = _SerialMPI()  # type: ignore


def rank0_log(comm: MPI.Comm, *args, **kwargs) -> None:
    if comm.Get_rank() == 0:
        print(*args, **kwargs)


def ensure_standard_output_tree(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    trajectories_dir = output_dir / "trajectories"
    diagnostics_dir.mkdir(exist_ok=True)
    trajectories_dir.mkdir(exist_ok=True)
    return {
        "root": output_dir,
        "diagnostics": diagnostics_dir,
        "trajectories": trajectories_dir,
    }


def _json_default(value: Any):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


class RealtimeProfiler:
    """Append-only JSONL profiler with immediate flush for live monitoring."""

    def __init__(self, path: Path | str, *, enabled: bool = True):
        self.path = Path(path)
        self.enabled = bool(enabled)
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return
        record = {
            "ts_epoch": time.time(),
            "ts_iso": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default) + "\n")
            handle.flush()


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def summarize_spectrum(eigenvalues) -> dict[str, float | int | None]:
    eigvals = np.asarray(eigenvalues, dtype=float)
    if eigvals.size == 0:
        return {
            "count": 0,
            "lambda_min": None,
            "lambda_max": None,
            "lambda_mean": None,
            "condition_number": None,
            "spread_pct": None,
        }

    lam_min = float(np.min(eigvals))
    lam_max = float(np.max(eigvals))
    lam_mean = float(np.mean(eigvals))
    return {
        "count": int(eigvals.size),
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "lambda_mean": lam_mean,
        "condition_number": float(lam_max / max(lam_min, 1e-30)),
        "spread_pct": float(100.0 * (lam_max - lam_min) / max(abs(lam_mean), 1e-30)),
    }

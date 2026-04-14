#!/usr/bin/env python3
"""Standardized registry entrypoint for Shinnecock twin experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from experiments.twin_framework import EXPERIMENT_REGISTRY, run_registered_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a standardized Shinnecock twin experiment from the registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=sorted(EXPERIMENT_REGISTRY.keys()),
        help="Registered experiment name.",
    )
    parser.add_argument(
        "--method",
        default=None,
        help=(
            "Method override. Use 'all' for the restored WSE sweep, or one of "
            "'4dvar', 'dcwme', 'dcwme_static', 'dcwme_dynamic' for Manning. "
            "Leaving Manning unset/'all' runs the full 4dvar-vs-static-vs-dynamic grid."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional subdirectory tag under the experiment output root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Serialize the resolved experiment config without running the solve.",
    )
    parser.add_argument(
        "--single-case",
        action="store_true",
        help=(
            "For sweep-based experiments such as 'mannings_n', run a single resolved case "
            "instead of the full grid."
        ),
    )
    parser.add_argument(
        "--regularization-weight",
        type=float,
        default=None,
        help="Optional single-case Manning override for the parameter regularization weight.",
    )
    parser.add_argument(
        "--obs-noise-level",
        type=float,
        default=None,
        help="Optional single-case Manning override for observation noise level.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=None,
        help="Optional single-case Manning override for DA window length in timesteps.",
    )
    parser.add_argument(
        "--obs-fraction",
        type=float,
        default=None,
        help="Optional single-case Manning override for the observed mesh fraction.",
    )
    parser.add_argument(
        "--obs-frequency",
        type=int,
        default=None,
        help="Optional single-case Manning override for the observation stride in timesteps.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Optional single-case Manning override for the optimizer iteration cap.",
    )
    parser.add_argument(
        "--cheap-static-case",
        action="store_true",
        help=(
            "For Manning single-case runs, apply a lightweight pilot configuration "
            "(short window, sparse observations, low iteration cap) unless explicitly overridden."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    case_overrides = {
        key: value
        for key, value in {
            "single_case": args.single_case,
            "regularization_weight": args.regularization_weight,
            "obs_noise_level": args.obs_noise_level,
            "window_length": args.window_length,
            "obs_fraction": args.obs_fraction,
            "obs_frequency": args.obs_frequency,
            "max_iterations": args.max_iterations,
            "cheap_static_case": args.cheap_static_case,
        }.items()
        if value is not None and value is not False
    }
    run_registered_experiment(
        args.experiment,
        method=args.method,
        output_root=args.output_root,
        dry_run=args.dry_run,
        tag=args.tag,
        case_overrides=case_overrides,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Focused state-only DC-WME validation with wind ramp-up.

Runs a single 4D-Var vs DC-WME (static L_wme) comparison at Vmax=30
using the full authoritative pipeline from run_comparison.py.

This validates the 3 bug fixes to the static L_wme path:
  1. Missing +I noise term in L = I + (N/σ²) H B H^T
  2. Wrong σ² power in Eq 38 inflation formula
  3. Double-inflation when TLM-based Eq 38 is available

Usage:
  python experiments/run_state_dcwme_validation.py
  python experiments/run_state_dcwme_validation.py --vmax 30 --method dcwme_static
  python experiments/run_state_dcwme_validation.py --method 4dvar  # 4D-Var baseline
"""

import argparse
import json
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

os.environ.setdefault("CC", "/usr/bin/clang")


def main():
    parser = argparse.ArgumentParser(description="State-only DC-WME validation")
    parser.add_argument("--vmax", type=float, default=30.0)
    parser.add_argument("--method", default="dcwme_static",
                        choices=["4dvar", "dcwme_static", "dcwme_dynamic", "both"])
    parser.add_argument("--nt-ramp", type=int, default=144)
    parser.add_argument("--nt-da", type=int, default=12)
    parser.add_argument("--n-windows", type=int, default=1)
    parser.add_argument("--obs-fraction", type=float, default=0.1)
    parser.add_argument("--obs-frequency", type=int, default=6)
    parser.add_argument("--skip-eq38", action="store_true",
                        help="Skip TLM-based Eq 38 inflation (use B as-is, e.g. from env var overrides)")
    parser.add_argument("--mem-limit-gb", type=float, default=12.0)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "results" / "state_dcwme_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)  # child process writes results here

    from experiments.shinnecock_study.run_comparison import _run_in_subprocess

    script_path = str(PROJECT_ROOT / "experiments" / "shinnecock_study" / "run_comparison.py")

    # Generate wind files
    from experiments.shinnecock_study.wind_models import (
        DEFAULT_TRACK,
        HollandHurricaneConfig,
        WindGridConfig,
        generate_holland_wind_field,
        generate_perturbed_config,
        write_wind_hdf5,
    )
    import numpy as np

    nt_total = args.nt_ramp + args.nt_da
    dt = 600.0
    times = np.arange(0.0, (nt_total + 1) * dt, dt)
    grid = WindGridConfig()

    wind_dir = output_dir / "wind"
    wind_dir.mkdir(parents=True, exist_ok=True)

    truth_wind = wind_dir / f"truth_vmax{args.vmax:.0f}.h5"
    perturbed_wind = wind_dir / f"perturbed_vmax{args.vmax:.0f}.h5"

    if not truth_wind.exists():
        print(f"Generating truth wind (Vmax={args.vmax})...")
        truth_cfg = HollandHurricaneConfig(track_waypoints=DEFAULT_TRACK, Vmax=args.vmax)
        wx, wy, p = generate_holland_wind_field(truth_cfg, grid, times)
        write_wind_hdf5(str(truth_wind), grid, times, wx, wy, p)

    if not perturbed_wind.exists():
        print(f"Generating perturbed wind (15km track shift)...")
        truth_cfg = HollandHurricaneConfig(track_waypoints=DEFAULT_TRACK, Vmax=args.vmax)
        pert_cfg = generate_perturbed_config(truth_cfg, "track_shift", 15.0)
        wx, wy, p = generate_holland_wind_field(pert_cfg, grid, times)
        write_wind_hdf5(str(perturbed_wind), grid, times, wx, wy, p)

    # Define methods to run
    METHOD_MAP = {
        "4dvar": ("4dvar", "a", "N/A"),
        "dcwme_static": ("dcwme", "c", "static"),
        "dcwme_dynamic": ("dcwme", "b", "dynamic"),
    }

    if args.method == "both":
        methods = ["4dvar", "dcwme_static"]
    else:
        methods = [args.method]

    mem_limit_mb = args.mem_limit_gb * 1024

    for method_name in methods:
        method, sub_label, l_wme_mode = METHOD_MAP[method_name]
        result_file = output_dir / f"vmax{args.vmax:.0f}_{sub_label}_{method_name}_results.json"

        if result_file.exists():
            print(f"\n{method_name}: Result already exists at {result_file}, skipping")
            with open(result_file) as f:
                result = json.load(f)
            print(f"  RMSE: {result.get('state_rmse', '?')}")
            continue

        print(f"\n{'='*60}")
        print(f"Running {method_name.upper()} (Vmax={args.vmax}, {args.nt_da} DA steps, "
              f"{args.n_windows} window(s))")
        print(f"{'='*60}")

        run_config = {
            "output_dir": str(output_dir),
            "adios_file": "data/shinnecock_inlet",
            "sub_label": sub_label,
            "method": method,
            "wind_truth_file": str(truth_wind),
            "wind_perturbed_file": str(perturbed_wind),
            "nt_da": args.nt_da,
            "nt_ramp": args.nt_ramp,
            "phase_prefix": f"val_wind{int(args.vmax)}_",
            "sweep_params": {
                "dt": dt,
                "obs_noise_level": 0.01,
                "obs_fraction": args.obs_fraction,
                "obs_frequency": args.obs_frequency,
                "background_error_std": 0.02,
            },
            "l_wme_mode": l_wme_mode,
            "n_windows": args.n_windows,
            "skip_eq38": args.skip_eq38,
            "result_file": str(result_file),
            "mem_limit_gb": args.mem_limit_gb,
            "worker_type": "wind",
        }

        result = _run_in_subprocess(run_config, mem_limit_mb, script_path)

        print(f"\n{method_name} result:")
        print(f"  Status: {result.get('status', '?')}")
        print(f"  RMSE: {result.get('state_rmse', '?')}")
        if 'error' in result:
            print(f"  Error: {result['error']}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    raise SystemExit(main())

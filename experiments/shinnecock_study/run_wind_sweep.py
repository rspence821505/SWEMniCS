#!/usr/bin/env python3
"""
Phase 7 Wind Sweep Runner — configurable from command line.

Usage examples:
    # Original 2h sweep, dt=600, Vmax=0-40, all 3 methods
    python experiments/shinnecock_study/run_wind_sweep.py

    # dt=300, Vmax=20,30 only, 4dvar + static only
    python experiments/shinnecock_study/run_wind_sweep.py \
        --dt 300 --vmax 20 30 --methods 4dvar static

    # 3-window cycling, dt=300, obs_frequency=12
    python experiments/shinnecock_study/run_wind_sweep.py \
        --dt 300 --vmax 20 30 --methods 4dvar static \
        --n-windows 3 --obs-frequency 12

    # Custom obs noise and fraction
    python experiments/shinnecock_study/run_wind_sweep.py \
        --dt 300 --vmax 30 --methods 4dvar static \
        --obs-noise 0.5 --obs-fraction 0.3

    # Dry run (show configs without running)
    python experiments/shinnecock_study/run_wind_sweep.py --dry-run
"""

import argparse
import json
import os
import sys
import shutil
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 7 Wind Sweep Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Wind parameters
    parser.add_argument("--vmax", type=float, nargs="+", default=[0, 10, 20, 30, 40],
                        help="Wind intensities to sweep (m/s). Default: 0 10 20 30 40")
    parser.add_argument("--track-shift-km", type=float, default=15.0,
                        help="Track shift perturbation magnitude (km). Default: 15")
    parser.add_argument("--wind-ramp-s", type=float, default=43200.0,
                        help="Wind temporal ramp timescale (s). Default: 43200 (12h)")

    # DA methods
    parser.add_argument("--methods", nargs="+", default=["4dvar", "dynamic", "static"],
                        choices=["4dvar", "dynamic", "static"],
                        help="Methods to run. Default: 4dvar dynamic static")

    # Time stepping
    parser.add_argument("--dt", type=float, default=600.0,
                        help="Time step (s). Default: 600")
    parser.add_argument("--nt-da", type=int, default=None,
                        help="DA window timesteps. Default: auto (2h / dt)")
    parser.add_argument("--nt-ramp", type=int, default=None,
                        help="Ramp timesteps. Default: auto (24h / dt)")

    # Cycling
    parser.add_argument("--n-windows", type=int, default=1,
                        help="Number of cycling windows. Default: 1 (no cycling)")

    # Observation parameters
    parser.add_argument("--obs-fraction", type=float, default=0.1,
                        help="Observation fraction. Default: 0.1")
    parser.add_argument("--obs-frequency", type=int, default=6,
                        help="Observe every N timesteps. Default: 6")
    parser.add_argument("--obs-noise", type=float, default=0.01,
                        help="Observation noise level. Default: 0.01")

    # Background
    parser.add_argument("--bg-error-std", type=float, default=0.02,
                        help="Background error std. Default: 0.02")
    parser.add_argument("--predictability-gamma", type=float, default=0.1,
                        help="Eq 38 safety factor γ. Default: 0.1")

    # Infrastructure
    parser.add_argument("--mem-limit-gb", type=float, default=8.0,
                        help="Memory limit per subprocess (GB). Default: 8")
    parser.add_argument("--output-dir", type=str, default="outputs/shinnecock_study",
                        help="Output directory. Default: outputs/shinnecock_study")
    parser.add_argument("--adios-file", type=str, default="data/shinnecock_inlet",
                        help="ADCIRC mesh file prefix. Default: data/shinnecock_inlet")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag for result filenames (e.g. 'test1'). Default: auto-generated")

    # Control
    parser.add_argument("--dry-run", action="store_true",
                        help="Show configs without running")

    return parser.parse_args()


def build_tag(args):
    """Build a filename tag from the sweep parameters."""
    if args.tag:
        return args.tag

    parts = []
    if args.dt != 600:
        parts.append(f"dt{int(args.dt)}")
    if args.n_windows > 1:
        parts.append(f"{args.n_windows}w")
    if args.obs_fraction != 0.1:
        parts.append(f"obs{str(args.obs_fraction).replace('.', 'p')}")
    if args.obs_frequency != 6:
        parts.append(f"freq{args.obs_frequency}")
    if args.obs_noise != 0.01:
        parts.append(f"sig{str(args.obs_noise).replace('.', 'p')}")
    if args.predictability_gamma != 0.1:
        parts.append(f"gam{str(args.predictability_gamma).replace('.', 'p')}")

    return "_".join(parts) if parts else ""


def generate_wind_files(args, wind_dir, times):
    """Generate truth and perturbed wind HDF5 files for each Vmax."""
    from wind_models import (
        WindGridConfig, HollandHurricaneConfig,
        generate_holland_wind_field, generate_zero_wind_field,
        write_wind_hdf5, generate_perturbed_config, DEFAULT_TRACK,
    )

    grid = WindGridConfig()
    tag = build_tag(args)
    tag_suffix = f"_{tag}" if tag else ""

    files = {}
    for Vmax in args.vmax:
        truth_file = wind_dir / f"holland_V{int(Vmax)}{tag_suffix}_truth.h5"
        pert_file = wind_dir / f"holland_V{int(Vmax)}{tag_suffix}_pert{args.track_shift_km:.0f}km.h5"

        if not truth_file.exists():
            if Vmax == 0:
                wx, wy, p = generate_zero_wind_field(grid, times)
            else:
                config = HollandHurricaneConfig(
                    track_waypoints=DEFAULT_TRACK, Vmax=float(Vmax),
                )
                wx, wy, p = generate_holland_wind_field(
                    config, grid, times, wind_ramp_s=args.wind_ramp_s,
                )
            write_wind_hdf5(str(truth_file), grid, times, wx, wy, p)

        if not pert_file.exists():
            if Vmax == 0:
                shutil.copy(truth_file, pert_file)
                print(f"  Copied zero-wind: {pert_file.name}")
            else:
                config = HollandHurricaneConfig(
                    track_waypoints=DEFAULT_TRACK, Vmax=float(Vmax),
                )
                config_pert = generate_perturbed_config(
                    config, "track_shift", args.track_shift_km,
                )
                wx_p, wy_p, p_p = generate_holland_wind_field(
                    config_pert, grid, times, wind_ramp_s=args.wind_ramp_s,
                )
                write_wind_hdf5(str(pert_file), grid, times, wx_p, wy_p, p_p)

        files[Vmax] = (str(truth_file), str(pert_file))

    return files


METHOD_MAP = {
    "4dvar":   ("4dvar",  "a", "N/A"),
    "dynamic": ("dcwme",  "b", "dynamic"),
    "static":  ("dcwme",  "c", "static"),
}


def main():
    os.environ.setdefault("CC", "/usr/bin/clang")
    args = parse_args()

    # Compute timestep counts
    nt_ramp = args.nt_ramp or int(86400 / args.dt)  # 24h ramp
    nt_da_per_window = args.nt_da or int(7200 / args.dt)  # 2h per window
    nt_da_total = nt_da_per_window * args.n_windows

    tag = build_tag(args)
    tag_suffix = f"_{tag}" if tag else ""

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    wind_dir = output_dir / "wind_fields"
    data_dir.mkdir(parents=True, exist_ok=True)
    wind_dir.mkdir(parents=True, exist_ok=True)

    # Print config
    print("=" * 70)
    print("PHASE 7 WIND SWEEP")
    print("=" * 70)
    print(f"  Vmax values:      {args.vmax}")
    print(f"  Methods:          {args.methods}")
    print(f"  dt:               {args.dt}s")
    print(f"  nt_ramp:          {nt_ramp} ({nt_ramp * args.dt / 3600:.0f}h)")
    print(f"  nt_da per window: {nt_da_per_window} ({nt_da_per_window * args.dt / 3600:.1f}h)")
    print(f"  n_windows:        {args.n_windows}")
    print(f"  nt_da total:      {nt_da_total} ({nt_da_total * args.dt / 3600:.1f}h)")
    print(f"  obs_fraction:     {args.obs_fraction}")
    print(f"  obs_frequency:    {args.obs_frequency}")
    print(f"  obs_noise:        {args.obs_noise}")
    print(f"  bg_error_std:     {args.bg_error_std}")
    print(f"  pred. gamma:      {args.predictability_gamma}")
    print(f"  mem_limit:        {args.mem_limit_gb} GB")
    print(f"  tag:              '{tag}'")

    n_configs = len(args.vmax) * len(args.methods)
    print(f"  Total configs:    {n_configs}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would run these configs:")
        for Vmax in args.vmax:
            for method_name in args.methods:
                method, sub_label, l_wme_mode = METHOD_MAP[method_name]
                phase_prefix = f"7_wind{int(Vmax)}{tag_suffix}_"
                result_file = data_dir / f"phase{phase_prefix}{sub_label}_results.json"
                exists = "EXISTS" if result_file.exists() else "NEW"
                print(f"  V{int(Vmax):>3} {method_name:>8} → {result_file.name} [{exists}]")
        return 0

    # Generate wind files
    print("\n--- Generating wind fields ---")
    times = np.arange(0, (nt_ramp + nt_da_total + 1) * args.dt, args.dt)
    wind_files = generate_wind_files(args, wind_dir, times)

    # Run sweep
    from experiments.shinnecock_study.run_comparison import _run_in_subprocess
    script_path = str((Path(__file__).parent / "run_comparison.py").resolve())
    mem_limit_mb = args.mem_limit_gb * 1024

    all_results = []

    for Vmax in args.vmax:
        truth_file, pert_file = wind_files[Vmax]

        print(f"\n{'#' * 70}")
        print(f"# Vmax = {int(Vmax)} m/s")
        print(f"{'#' * 70}")

        point_results = {"Vmax": Vmax}

        for method_name in args.methods:
            method, sub_label, l_wme_mode = METHOD_MAP[method_name]
            phase_prefix = f"7_wind{int(Vmax)}{tag_suffix}_"
            result_file = data_dir / f"phase{phase_prefix}{sub_label}_results.json"

            # Skip/resume
            if result_file.exists():
                print(f"\n  Skipping {result_file.name} (exists)")
                with open(result_file) as f:
                    result = json.load(f)
                point_results[method_name] = result
                continue

            print(f"\n  Running: V{int(Vmax)} {method_name} [subprocess]", flush=True)

            sweep_params = {
                "dt": args.dt,
                "obs_noise_level": args.obs_noise,
                "obs_fraction": args.obs_fraction,
                "obs_frequency": args.obs_frequency,
                "background_error_std": args.bg_error_std,
                "nt_da": nt_da_total,
                "nt_ramp": nt_ramp,
            }

            run_config = {
                "output_dir": str(output_dir),
                "adios_file": args.adios_file,
                "sub_label": sub_label,
                "method": method,
                "wind_truth_file": truth_file,
                "wind_perturbed_file": pert_file,
                "nt_da": nt_da_total,
                "nt_ramp": nt_ramp,
                "phase_prefix": phase_prefix,
                "sweep_params": sweep_params,
                "l_wme_mode": l_wme_mode,
                "n_windows": args.n_windows,
                "result_file": str(result_file),
                "mem_limit_gb": args.mem_limit_gb,
                "worker_type": "wind",
            }

            result = _run_in_subprocess(run_config, mem_limit_mb, script_path)
            point_results[method_name] = result

            status = result.get("status", "unknown")
            if status == "failed":
                print(f"  FAILED: {result.get('error', '?')[:150]}")
            else:
                err = result.get("results", {}).get("error_reduction", "?")
                diag = result.get("dcwme_diagnostics", {})
                spread = diag.get("l_wme_spread_pct", None)
                wins = result.get("window_results", [])
                extras = []
                if spread is not None:
                    extras.append(f"spread={spread:.1f}%")
                if wins:
                    win_str = ", ".join([f"W{w['window']+1}={w['error_reduction']:.1f}%" for w in wins])
                    extras.append(f"[{win_str}]")
                extra_str = " " + " ".join(extras) if extras else ""
                print(f"  Done: err_red={err}{extra_str}")

        all_results.append(point_results)

    # Summary
    print(f"\n{'=' * 70}")
    print("SWEEP SUMMARY")
    print(f"{'=' * 70}")

    header_methods = [m for m in args.methods]
    header = f"{'Vmax':>6}" + "".join(f"{m:>10}" for m in header_methods)
    if "dynamic" in args.methods or "static" in args.methods:
        header += f"{'spread':>10}"
    print(header)
    print("-" * len(header))

    for point in all_results:
        Vmax = point["Vmax"]
        row = f"{int(Vmax):>6}"
        for m in header_methods:
            r = point.get(m, {})
            if r.get("status") == "failed":
                row += f"{'FAIL':>10}"
            else:
                err = r.get("results", {}).get("error_reduction", None)
                row += f"{err:>9.1f}%" if err is not None else f"{'?':>10}"

        if "dynamic" in args.methods:
            r = point.get("dynamic", {})
            spread = r.get("dcwme_diagnostics", {}).get("l_wme_spread_pct", None)
            row += f"{spread:>9.1f}%" if spread is not None else f"{'-':>10}"
        elif "static" in args.methods:
            r = point.get("static", {})
            spread = r.get("dcwme_diagnostics", {}).get("l_wme_spread_pct", None)
            row += f"{spread:>9.1f}%" if spread is not None else f"{'-':>10}"

        print(row)

    print(f"{'=' * 70}")

    # Save summary
    summary_file = data_dir / f"phase7{tag_suffix}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

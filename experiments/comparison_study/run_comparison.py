#!/usr/bin/env python3
"""
Entry point for the 4D-Var vs DC-WME comparison study.

This script runs systematic comparisons between standard 4D-Var and
Data-Consistent Weighted Mean Error (DC-WME) 4D-Var using twin experiments
with physics perturbation to avoid inverse crime.

Usage:
    # Run primary friction sweep (recommended first)
    python experiments/comparison_study/run_comparison.py --experiments friction

    # Run all experiments
    python experiments/comparison_study/run_comparison.py --experiments all

    # Run specific sweeps
    python experiments/comparison_study/run_comparison.py --experiments friction,obs_freq

    # Debug a single experiment
    python experiments/comparison_study/run_comparison.py \
        --single-experiment dcwme_friction_1.2 \
        --diagnostic-level verbose

Examples:
    # Step 1: Verify setup works
    python experiments/comparison_study/verify_setup.py

    # Step 2: Run friction sweep
    python experiments/comparison_study/run_comparison.py --experiments friction

    # Step 3: Run other sweeps (with model error)
    python experiments/comparison_study/run_comparison.py --experiments obs_freq,obs_fraction,noise

    # Step 4: Generate all plots
    python experiments/comparison_study/run_comparison.py --plot-only
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.comparison_study.config import (
    ComparisonStudyConfig,
    SweepConfig,
    AVAILABLE_EXPERIMENTS,
    get_experiment_description,
)
from experiments.comparison_study.runner import ComparisonRunner, run_single_experiment_verbose
from experiments.comparison_study.plotting import generate_all_plots


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 4D-Var vs DC-WME comparison study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available experiments:
  friction      - PRIMARY: Friction scale sweep (tests robustness to model error)
  obs_freq      - Observation frequency sweep
  obs_fraction  - Observation fraction sweep
  noise         - Observation noise sweep
  background    - Background error sweep
  bathymetry    - OPTIONAL: Bathymetry noise sweep
  all           - Run all experiments

Examples:
  python run_comparison.py --experiments friction
  python run_comparison.py --experiments friction,obs_freq,noise
  python run_comparison.py --experiments all
        """,
    )

    parser.add_argument(
        "--experiments",
        type=str,
        default="friction",
        help="Comma-separated list of experiments to run (or 'all')",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/comparison_study",
        help="Output directory for results and figures",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing results (start fresh)",
    )

    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing results (don't run experiments)",
    )

    parser.add_argument(
        "--single-experiment",
        type=str,
        help="Run a single experiment (e.g., 'dcwme_friction_1.2') with verbose diagnostics",
    )

    parser.add_argument(
        "--diagnostic-level",
        type=str,
        choices=["minimal", "standard", "verbose"],
        default="standard",
        help="Diagnostic level for experiments",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification checks before experiments",
    )

    # Problem configuration
    parser.add_argument(
        "--nx",
        type=int,
        default=20,
        help="Number of elements in x direction",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=10,
        help="Number of elements in y direction",
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=172800.0,
        help="Final simulation time (seconds)",
    )

    return parser.parse_args()


def run_verification():
    """Run verification checks before experiments."""
    print("\n" + "=" * 60)
    print("RUNNING VERIFICATION CHECKS")
    print("=" * 60)

    from experiments.comparison_study.verify_setup import main as verify_main

    return verify_main() == 0


def run_experiments(args):
    """Run the specified experiments."""
    # Parse experiments
    if args.experiments.lower() == "all":
        experiments = AVAILABLE_EXPERIMENTS[:-1]  # Exclude checkpointing for now
    else:
        experiments = [e.strip() for e in args.experiments.split(",")]
        for exp in experiments:
            if exp not in AVAILABLE_EXPERIMENTS:
                print(f"Error: Unknown experiment '{exp}'")
                print(f"Available: {', '.join(AVAILABLE_EXPERIMENTS)}")
                return 1

    # Create configuration
    base_config = ComparisonStudyConfig(
        nx=args.nx,
        ny=args.ny,
        final_time=args.final_time,
        output_dir=Path(args.output_dir),
        diagnostic_level=args.diagnostic_level,
    )
    sweep_config = SweepConfig()

    print("\n" + "=" * 60)
    print("COMPARISON STUDY CONFIGURATION")
    print("=" * 60)
    print(f"  Problem: TidalProblem (nx={base_config.nx}, ny={base_config.ny})")
    print(f"  Timesteps: {base_config.nt}")
    print(f"  Output: {base_config.output_dir}")
    print(f"  Experiments: {', '.join(experiments)}")
    print(f"  Resume: {not args.no_resume}")
    print(f"  Diagnostic level: {args.diagnostic_level}")

    # Create runner
    runner = ComparisonRunner(base_config, sweep_config)

    # Run experiments
    start_time = time.time()
    all_results = {}

    for exp in experiments:
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT: {exp.upper()}")
        print(f"Description: {get_experiment_description(exp)}")
        print("=" * 60)

        runner.results = {}  # Reset for each sweep
        results = runner.run_sweep(exp, resume=not args.no_resume)
        all_results[exp] = results

        # Generate plots for this sweep
        results_file = base_config.output_dir / "data" / f"{exp}_sweep.json"
        if results_file.exists():
            generate_all_plots(
                results_file,
                base_config.output_dir / "figures",
                exp,
            )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("COMPARISON STUDY COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed / 60:.1f} minutes")
    print(f"  Results: {base_config.output_dir / 'data'}")
    print(f"  Figures: {base_config.output_dir / 'figures'}")
    print(f"  Reports: {base_config.output_dir}")

    return 0


def run_single_experiment_debug(args):
    """Run a single experiment with verbose diagnostics."""
    exp_id = args.single_experiment

    # Parse experiment ID: method_sweep_value
    parts = exp_id.split("_")
    if len(parts) < 3:
        print(f"Error: Invalid experiment ID '{exp_id}'")
        print("Expected format: method_sweep_value (e.g., 'dcwme_friction_1.2')")
        return 1

    method = parts[0]
    sweep_param = parts[1]
    sweep_value = "_".join(parts[2:])  # Handle values with underscores

    # Try to parse value as float, otherwise keep as string
    try:
        sweep_value = float(sweep_value)
    except ValueError:
        pass

    print(f"\n{'=' * 60}")
    print(f"SINGLE EXPERIMENT DEBUG")
    print("=" * 60)
    print(f"  Method: {method}")
    print(f"  Sweep param: {sweep_param}")
    print(f"  Sweep value: {sweep_value}")
    print(f"  Diagnostic level: verbose")

    base_config = ComparisonStudyConfig(
        nx=args.nx,
        ny=args.ny,
        final_time=args.final_time,
        output_dir=Path(args.output_dir),
        diagnostic_level="verbose",
    )

    debug_dir = Path(args.output_dir) / "debug"
    result = run_single_experiment_verbose(
        method=method,
        sweep_param=sweep_param,
        sweep_value=sweep_value,
        base_config=base_config,
        save_intermediate=True,
        output_dir=debug_dir,
    )

    print(f"\nResult: {result.status}")
    if result.status == "failed":
        print(f"  Error: {result.error}")
        print(f"  Failure type: {result.failure_type}")
        diag = result.diagnostics or {}
        print(f"  Failed at iteration: {diag.get('failure_iteration', 'N/A')}")
        if diag.get("gradient_norm_history"):
            print(f"  Last gradient norm: {diag['gradient_norm_history'][-1]:.2e}")
    else:
        res = result.result or {}
        print(f"  Error reduction: {res.get('error_reduction', 0):.1f}%")
        print(f"  Iterations: {res.get('num_iterations', 0)}")
        print(f"  Wall time: {result.wall_time:.1f}s")

    return 0


def generate_plots_only(args):
    """Generate plots from existing results."""
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"

    if not data_dir.exists():
        print(f"Error: No data found in {data_dir}")
        return 1

    print("\n" + "=" * 60)
    print("GENERATING PLOTS FROM EXISTING RESULTS")
    print("=" * 60)

    for results_file in data_dir.glob("*_sweep.json"):
        sweep_name = results_file.stem.replace("_sweep", "")
        print(f"\nProcessing: {sweep_name}")
        generate_all_plots(results_file, figures_dir, sweep_name)

    print(f"\nPlots saved to: {figures_dir}")
    return 0


def main():
    """Main entry point."""
    args = parse_args()

    # Handle special modes
    if args.verify:
        if not run_verification():
            print("\nVerification failed. Please fix issues before running experiments.")
            return 1
        print("\nVerification passed. Ready to run experiments.")
        return 0

    if args.plot_only:
        return generate_plots_only(args)

    if args.single_experiment:
        return run_single_experiment_debug(args)

    # Run experiments
    return run_experiments(args)


if __name__ == "__main__":
    sys.exit(main())

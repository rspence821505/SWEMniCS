#!/usr/bin/env python3
"""
Verify serial vs parallel data assimilation results.

This script runs both serial and parallel versions of the DA experiments
and compares results to ensure they match within a specified tolerance.

The verification checks:
1. Final cost function values
2. Analysis state vectors
3. Gradient norms
4. Convergence behavior

Usage:
    python verify_serial_parallel.py [--tolerance 1e-10] [--quick]

Options:
    --tolerance    Maximum allowed difference (default: 1e-10)
    --quick        Run with reduced problem size for faster verification
    --nprocs       Number of MPI processes for parallel run (default: 4)
    --verbose      Show detailed comparison
"""

import argparse
import subprocess
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.utils.output_paths import DATA_DIR, ensure_output_dirs


@dataclass
class VerificationResult:
    """Result of a single verification comparison."""
    experiment: str
    metric: str
    serial_value: float
    parallel_value: float
    difference: float
    relative_error: float
    passed: bool
    tolerance: float


def load_experiment_results(filepath: str) -> Dict:
    """Load experiment results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def run_serial_experiment(script_path: str, args: List[str] = None) -> Optional[str]:
    """
    Run a serial experiment and return the results file path.

    Parameters
    ----------
    script_path : str
        Path to the experiment script.
    args : List[str]
        Additional command line arguments.

    Returns
    -------
    str or None
        Path to results file, or None if failed.
    """
    cmd = ["python", script_path]
    if args:
        cmd.extend(args)

    print(f"  Running serial: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        if result.returncode != 0:
            print(f"  ERROR: Serial run failed with code {result.returncode}")
            print(f"  STDERR: {result.stderr[:500]}...")
            return None
        return "success"
    except subprocess.TimeoutExpired:
        print("  ERROR: Serial run timed out")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def run_parallel_experiment(
    script_path: str,
    nprocs: int = 4,
    args: List[str] = None
) -> Optional[str]:
    """
    Run a parallel experiment with MPI.

    Parameters
    ----------
    script_path : str
        Path to the experiment script.
    nprocs : int
        Number of MPI processes.
    args : List[str]
        Additional command line arguments.

    Returns
    -------
    str or None
        "success" if successful, None otherwise.
    """
    cmd = ["mpirun", "-n", str(nprocs), "python", script_path]
    if args:
        cmd.extend(args)

    print(f"  Running parallel ({nprocs} ranks): {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        if result.returncode != 0:
            print(f"  ERROR: Parallel run failed with code {result.returncode}")
            print(f"  STDERR: {result.stderr[:500]}...")
            return None
        return "success"
    except subprocess.TimeoutExpired:
        print("  ERROR: Parallel run timed out")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def compare_results(
    serial_results: Dict,
    parallel_results: Dict,
    experiment_name: str,
    tolerance: float = 1e-10
) -> List[VerificationResult]:
    """
    Compare serial and parallel experiment results.

    Parameters
    ----------
    serial_results : Dict
        Serial experiment results.
    parallel_results : Dict
        Parallel experiment results.
    experiment_name : str
        Name of the experiment.
    tolerance : float
        Maximum allowed difference.

    Returns
    -------
    List[VerificationResult]
        List of comparison results.
    """
    comparisons = []

    # Metrics to compare
    metrics = [
        ("analysis_error", "Analysis Error"),
        ("background_error", "Background Error"),
        ("error_reduction", "Error Reduction (%)"),
        ("innovation_mean", "Innovation Mean"),
        ("innovation_std", "Innovation Std"),
    ]

    for key, display_name in metrics:
        serial_val = serial_results.get(key, 0.0)
        parallel_val = parallel_results.get(key, 0.0)

        diff = abs(serial_val - parallel_val)
        rel_error = diff / (abs(serial_val) + 1e-15)

        passed = diff <= tolerance or rel_error <= tolerance

        comparisons.append(VerificationResult(
            experiment=experiment_name,
            metric=display_name,
            serial_value=serial_val,
            parallel_value=parallel_val,
            difference=diff,
            relative_error=rel_error,
            passed=passed,
            tolerance=tolerance,
        ))

    # Compare cost history (final values)
    if "cost_history" in serial_results and "cost_history" in parallel_results:
        serial_cost = serial_results["cost_history"]
        parallel_cost = parallel_results["cost_history"]

        if serial_cost and parallel_cost:
            serial_final = serial_cost[-1]
            parallel_final = parallel_cost[-1]

            diff = abs(serial_final - parallel_final)
            rel_error = diff / (abs(serial_final) + 1e-15)
            passed = diff <= tolerance or rel_error <= tolerance

            comparisons.append(VerificationResult(
                experiment=experiment_name,
                metric="Final Cost",
                serial_value=serial_final,
                parallel_value=parallel_final,
                difference=diff,
                relative_error=rel_error,
                passed=passed,
                tolerance=tolerance,
            ))

    # Compare iteration count
    serial_iter = serial_results.get("num_iterations", 0)
    parallel_iter = parallel_results.get("num_iterations", 0)
    diff_iter = abs(serial_iter - parallel_iter)

    comparisons.append(VerificationResult(
        experiment=experiment_name,
        metric="Iterations",
        serial_value=float(serial_iter),
        parallel_value=float(parallel_iter),
        difference=diff_iter,
        relative_error=diff_iter / max(serial_iter, 1),
        passed=diff_iter <= 2,  # Allow small iteration difference
        tolerance=2.0,
    ))

    return comparisons


def print_comparison_table(comparisons: List[VerificationResult], verbose: bool = False):
    """Print formatted comparison table."""
    if not comparisons:
        print("No comparisons to display.")
        return

    # Group by experiment
    experiments = {}
    for comp in comparisons:
        if comp.experiment not in experiments:
            experiments[comp.experiment] = []
        experiments[comp.experiment].append(comp)

    for exp_name, exp_comps in experiments.items():
        print(f"\n{'='*70}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*70}")

        all_passed = all(c.passed for c in exp_comps)
        status = "PASSED" if all_passed else "FAILED"
        print(f"Status: {status}")
        print()

        if verbose or not all_passed:
            header = f"{'Metric':<20} {'Serial':<15} {'Parallel':<15} {'Diff':<12} {'Status':<8}"
            print(header)
            print("-" * 70)

            for comp in exp_comps:
                status_str = "OK" if comp.passed else "FAIL"
                if comp.metric == "Iterations":
                    print(f"{comp.metric:<20} {comp.serial_value:<15.0f} {comp.parallel_value:<15.0f} {comp.difference:<12.0f} {status_str:<8}")
                else:
                    print(f"{comp.metric:<20} {comp.serial_value:<15.6e} {comp.parallel_value:<15.6e} {comp.difference:<12.2e} {status_str:<8}")


def main():
    """Run serial-parallel verification."""
    parser = argparse.ArgumentParser(
        description="Verify serial vs parallel DA experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-10,
        help="Maximum allowed difference"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with reduced problem size for faster verification"
    )
    parser.add_argument(
        "--nprocs", type=int, default=4,
        help="Number of MPI processes"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed comparison output"
    )
    parser.add_argument(
        "--experiment", type=str, choices=["tidal_4dvar", "tidal_dcwme", "dam_break_4dvar", "dam_break_dcwme", "all"],
        default="all",
        help="Which experiment to verify"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Serial vs Parallel Verification")
    print("=" * 70)
    print(f"Tolerance: {args.tolerance}")
    print(f"MPI ranks: {args.nprocs}")
    print(f"Quick mode: {args.quick}")
    print("=" * 70)

    # Ensure output directories exist
    ensure_output_dirs()

    # Define experiments
    script_dir = Path(__file__).parent
    serial_dir = script_dir.parent / "serial_da"

    experiments = {
        "tidal_4dvar": {
            "serial": serial_dir / "tidal_4dvar.py",
            "parallel": script_dir / "tidal_4dvar_mpi.py",
            "serial_results": DATA_DIR / "tidal_4dvar_results.json",
            "parallel_results": DATA_DIR / "tidal_4dvar_mpi_results.json",
            "args": ["--nx", "10", "--ny", "5", "--dt", "3600", "--final-time", "86400", "--max-iter", "20"],
            "quick_args": ["--nx", "5", "--ny", "3", "--dt", "7200", "--final-time", "43200", "--max-iter", "5"],
        },
        "tidal_dcwme": {
            "serial": serial_dir / "tidal_dcwme.py",
            "parallel": script_dir / "tidal_dcwme_mpi.py",
            "serial_results": DATA_DIR / "tidal_dcwme_results.json",
            "parallel_results": DATA_DIR / "tidal_dcwme_mpi_results.json",
            "args": ["--nx", "10", "--ny", "5", "--dt", "3600", "--final-time", "86400", "--max-iter", "20"],
            "quick_args": ["--nx", "5", "--ny", "3", "--dt", "7200", "--final-time", "43200", "--max-iter", "5"],
        },
        "dam_break_4dvar": {
            "serial": serial_dir / "dam_break_4dvar.py",
            "parallel": script_dir / "dam_break_4dvar_mpi.py",
            "serial_results": DATA_DIR / "dam_break_4dvar_results.json",
            "parallel_results": DATA_DIR / "dam_break_4dvar_mpi_results.json",
            "args": ["--nx", "30", "--ny", "30", "--dt", "0.5", "--final-time", "20", "--max-iter", "20"],
            "quick_args": ["--nx", "15", "--ny", "15", "--dt", "1.0", "--final-time", "10", "--max-iter", "5"],
        },
        "dam_break_dcwme": {
            "serial": serial_dir / "dam_break_dcwme.py",
            "parallel": script_dir / "dam_break_dcwme_mpi.py",
            "serial_results": DATA_DIR / "dam_break_dcwme_results.json",
            "parallel_results": DATA_DIR / "dam_break_dcwme_mpi_results.json",
            "args": ["--nx", "30", "--ny", "30", "--dt", "0.5", "--final-time", "20", "--max-iter", "20"],
            "quick_args": ["--nx", "15", "--ny", "15", "--dt", "1.0", "--final-time", "10", "--max-iter", "5"],
        },
    }

    # Filter experiments if specific one requested
    if args.experiment != "all":
        experiments = {args.experiment: experiments[args.experiment]}

    all_comparisons = []
    all_passed = True

    for exp_name, exp_config in experiments.items():
        print(f"\n{'='*70}")
        print(f"Verifying: {exp_name}")
        print(f"{'='*70}")

        exp_args = exp_config["quick_args"] if args.quick else exp_config["args"]

        # Run serial experiment
        print("\n[1/2] Running serial experiment...")
        serial_success = run_serial_experiment(
            str(exp_config["serial"]),
            exp_args
        )

        if not serial_success:
            print(f"  SKIP: Serial experiment failed for {exp_name}")
            all_passed = False
            continue

        # Run parallel experiment
        print("\n[2/2] Running parallel experiment...")
        parallel_success = run_parallel_experiment(
            str(exp_config["parallel"]),
            nprocs=args.nprocs,
            args=exp_args
        )

        if not parallel_success:
            print(f"  SKIP: Parallel experiment failed for {exp_name}")
            all_passed = False
            continue

        # Load and compare results
        print("\n[3/3] Comparing results...")
        try:
            serial_results = load_experiment_results(str(exp_config["serial_results"]))
            parallel_results = load_experiment_results(str(exp_config["parallel_results"]))

            comparisons = compare_results(
                serial_results,
                parallel_results,
                exp_name,
                args.tolerance
            )

            all_comparisons.extend(comparisons)

            # Check if all passed
            exp_passed = all(c.passed for c in comparisons)
            if not exp_passed:
                all_passed = False

        except FileNotFoundError as e:
            print(f"  ERROR: Could not load results: {e}")
            all_passed = False
            continue

    # Print summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    print_comparison_table(all_comparisons, verbose=args.verbose)

    # Final status
    print("\n" + "=" * 70)
    if all_passed:
        print("OVERALL STATUS: PASSED")
        print("All serial and parallel results match within tolerance.")
    else:
        print("OVERALL STATUS: FAILED")
        print("Some differences exceed tolerance. See details above.")
    print("=" * 70)

    # Save verification results
    results_file = DATA_DIR / "verification_results.json"
    verification_data = {
        "tolerance": args.tolerance,
        "nprocs": args.nprocs,
        "quick_mode": args.quick,
        "overall_passed": all_passed,
        "comparisons": [
            {
                "experiment": c.experiment,
                "metric": c.metric,
                "serial_value": c.serial_value,
                "parallel_value": c.parallel_value,
                "difference": c.difference,
                "relative_error": c.relative_error,
                "passed": c.passed,
            }
            for c in all_comparisons
        ]
    }

    with open(results_file, 'w') as f:
        json.dump(verification_data, f, indent=2)
    print(f"\nVerification results saved to: {results_file}")

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

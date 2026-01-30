#!/usr/bin/env python3
"""
Scaling study for parallel data assimilation experiments.

This script measures wall-clock time for DA experiments with varying
numbers of MPI ranks and computes parallel efficiency metrics.

Metrics computed:
- Wall-clock time for each rank count
- Speedup: S(n) = T(1) / T(n)
- Parallel efficiency: E(n) = S(n) / n = T(1) / (n * T(n))

Usage:
    python scaling_study.py [--experiment tidal_4dvar] [--max-ranks 4]

Output:
    - outputs/data/scaling_results.json
    - outputs/figures/scaling_study.png
"""

import argparse
import subprocess
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swe4dvar.utils.output_paths import DATA_DIR, FIGURES_DIR, ensure_output_dirs


@dataclass
class ScalingResult:
    """Result from a single scaling run."""
    experiment: str
    num_ranks: int
    wall_time: float
    analysis_error: float
    num_iterations: int
    converged: bool


@dataclass
class ScalingStudyResults:
    """Complete scaling study results."""
    experiment: str
    rank_counts: List[int]
    wall_times: List[float]
    speedups: List[float]
    efficiencies: List[float]
    analysis_errors: List[float]
    target_efficiency: float
    efficiency_achieved: bool


def run_experiment(
    script_path: str,
    nprocs: int,
    args: List[str] = None,
    timeout: int = 1200
) -> Optional[Dict]:
    """
    Run an experiment and extract timing results.

    Parameters
    ----------
    script_path : str
        Path to the experiment script.
    nprocs : int
        Number of MPI processes.
    args : List[str]
        Additional command line arguments.
    timeout : int
        Timeout in seconds.

    Returns
    -------
    Dict or None
        Experiment results, or None if failed.
    """
    if nprocs == 1:
        cmd = ["python", script_path]
    else:
        cmd = ["mpirun", "-n", str(nprocs), "python", script_path]

    if args:
        cmd.extend(args)

    print(f"    Running with {nprocs} rank(s)...", end=" ", flush=True)

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"FAILED (code {result.returncode})")
            return None

        print(f"done ({elapsed:.2f}s)")

        # Parse wall time from output
        # Look for "Total time:" in output
        for line in result.stdout.split("\n"):
            if "Total time:" in line:
                try:
                    time_str = line.split(":")[-1].strip().replace("s", "").strip()
                    wall_time = float(time_str)
                    return {"wall_time": wall_time, "elapsed": elapsed}
                except ValueError:
                    pass

        # Fall back to measured elapsed time
        return {"wall_time": elapsed, "elapsed": elapsed}

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def load_latest_results(experiment: str, method: str = "mpi") -> Optional[Dict]:
    """Load the most recent experiment results."""
    results_file = DATA_DIR / f"{experiment}_{method}_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def compute_scaling_metrics(
    times: List[float],
    rank_counts: List[int]
) -> Tuple[List[float], List[float]]:
    """
    Compute speedup and parallel efficiency.

    Parameters
    ----------
    times : List[float]
        Wall-clock times for each rank count.
    rank_counts : List[int]
        Number of ranks for each measurement.

    Returns
    -------
    Tuple[List[float], List[float]]
        (speedups, efficiencies)
    """
    if not times or not rank_counts:
        return [], []

    # Find the single-rank time (or smallest rank count)
    min_ranks_idx = rank_counts.index(min(rank_counts))
    T1 = times[min_ranks_idx]
    n1 = rank_counts[min_ranks_idx]

    speedups = []
    efficiencies = []

    for i, (t, n) in enumerate(zip(times, rank_counts)):
        # Speedup relative to smallest rank count
        speedup = (T1 * n1) / (t * 1)  # Adjust for base rank count
        speedups.append(speedup)

        # Efficiency
        efficiency = speedup / n
        efficiencies.append(efficiency)

    return speedups, efficiencies


def create_scaling_plot(
    results: Dict[str, ScalingStudyResults],
    output_path: str
):
    """
    Create scaling study visualization.

    Parameters
    ----------
    results : Dict[str, ScalingStudyResults]
        Scaling results for each experiment.
    output_path : str
        Path to save the figure.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    # Plot 1: Wall-clock time vs ranks
    ax1 = axes[0]
    for i, (exp_name, exp_results) in enumerate(results.items()):
        ax1.plot(
            exp_results.rank_counts,
            exp_results.wall_times,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=exp_name.replace("_", " ").title(),
            linewidth=2,
            markersize=8
        )
    ax1.set_xlabel("Number of MPI Ranks", fontsize=12)
    ax1.set_ylabel("Wall-clock Time (s)", fontsize=12)
    ax1.set_title("Execution Time vs MPI Ranks", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(list(range(1, max(max(r.rank_counts) for r in results.values()) + 1)))

    # Plot 2: Speedup
    ax2 = axes[1]
    max_ranks = max(max(r.rank_counts) for r in results.values())
    ideal_ranks = list(range(1, max_ranks + 1))
    ax2.plot(ideal_ranks, ideal_ranks, 'k--', label='Ideal', linewidth=2, alpha=0.7)

    for i, (exp_name, exp_results) in enumerate(results.items()):
        ax2.plot(
            exp_results.rank_counts,
            exp_results.speedups,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=exp_name.replace("_", " ").title(),
            linewidth=2,
            markersize=8
        )
    ax2.set_xlabel("Number of MPI Ranks", fontsize=12)
    ax2.set_ylabel("Speedup", fontsize=12)
    ax2.set_title("Parallel Speedup", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(ideal_ranks)

    # Plot 3: Efficiency
    ax3 = axes[2]
    ax3.axhline(y=1.0, color='k', linestyle='--', label='Ideal (100%)', linewidth=2, alpha=0.7)
    ax3.axhline(y=0.7, color='r', linestyle=':', label='Target (70%)', linewidth=2, alpha=0.7)

    for i, (exp_name, exp_results) in enumerate(results.items()):
        ax3.plot(
            exp_results.rank_counts,
            exp_results.efficiencies,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=exp_name.replace("_", " ").title(),
            linewidth=2,
            markersize=8
        )
    ax3.set_xlabel("Number of MPI Ranks", fontsize=12)
    ax3.set_ylabel("Parallel Efficiency", fontsize=12)
    ax3.set_title("Parallel Efficiency", fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    ax3.set_xticks(ideal_ranks)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scaling plot saved to: {output_path}")


def main():
    """Run scaling study."""
    parser = argparse.ArgumentParser(
        description="Scaling study for parallel DA experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["tidal_4dvar", "tidal_dcwme", "dam_break_4dvar", "dam_break_dcwme", "all"],
        default="all",
        help="Which experiment to run scaling study on"
    )
    parser.add_argument(
        "--max-ranks",
        type=int,
        default=4,
        help="Maximum number of MPI ranks to test"
    )
    parser.add_argument(
        "--target-efficiency",
        type=float,
        default=0.7,
        help="Target parallel efficiency (0-1)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with reduced problem size"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Timeout per run in seconds"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Parallel Scaling Study")
    print("=" * 70)
    print(f"Maximum ranks: {args.max_ranks}")
    print(f"Target efficiency: {args.target_efficiency * 100:.0f}%")
    print(f"Quick mode: {args.quick}")
    print("=" * 70)

    # Ensure output directories exist
    ensure_output_dirs()

    script_dir = Path(__file__).parent

    # Define experiments
    experiments = {
        "tidal_4dvar": {
            "script": script_dir / "tidal_4dvar_mpi.py",
            "args": ["--nx", "10", "--ny", "5", "--dt", "3600", "--final-time", "86400", "--max-iter", "20"],
            "quick_args": ["--nx", "8", "--ny", "4", "--dt", "7200", "--final-time", "43200", "--max-iter", "10"],
        },
        "tidal_dcwme": {
            "script": script_dir / "tidal_dcwme_mpi.py",
            "args": ["--nx", "10", "--ny", "5", "--dt", "3600", "--final-time", "86400", "--max-iter", "20"],
            "quick_args": ["--nx", "8", "--ny", "4", "--dt", "7200", "--final-time", "43200", "--max-iter", "10"],
        },
        "dam_break_4dvar": {
            "script": script_dir / "dam_break_4dvar_mpi.py",
            "args": ["--nx", "30", "--ny", "30", "--dt", "0.5", "--final-time", "20", "--max-iter", "20"],
            "quick_args": ["--nx", "20", "--ny", "20", "--dt", "1.0", "--final-time", "10", "--max-iter", "10"],
        },
        "dam_break_dcwme": {
            "script": script_dir / "dam_break_dcwme_mpi.py",
            "args": ["--nx", "30", "--ny", "30", "--dt", "0.5", "--final-time", "20", "--max-iter", "20"],
            "quick_args": ["--nx", "20", "--ny", "20", "--dt", "1.0", "--final-time", "10", "--max-iter", "10"],
        },
    }

    # Filter experiments
    if args.experiment != "all":
        experiments = {args.experiment: experiments[args.experiment]}

    # Rank counts to test
    rank_counts = list(range(1, args.max_ranks + 1))

    all_results = {}

    for exp_name, exp_config in experiments.items():
        print(f"\n{'='*70}")
        print(f"Scaling Study: {exp_name}")
        print(f"{'='*70}")

        exp_args = exp_config["quick_args"] if args.quick else exp_config["args"]

        times = []
        errors = []
        iterations = []
        converged_list = []

        for nprocs in rank_counts:
            result = run_experiment(
                str(exp_config["script"]),
                nprocs,
                exp_args,
                timeout=args.timeout
            )

            if result:
                times.append(result["wall_time"])

                # Try to load detailed results
                method = "4dvar_mpi" if "4dvar" in exp_name else "dcwme_mpi"
                test_case = "tidal" if "tidal" in exp_name else "dam_break"
                detailed = load_latest_results(test_case, method)

                if detailed:
                    errors.append(detailed.get("analysis_error", 0.0))
                    iterations.append(detailed.get("num_iterations", 0))
                    converged_list.append(detailed.get("converged", False))
                else:
                    errors.append(0.0)
                    iterations.append(0)
                    converged_list.append(False)
            else:
                # If run failed, use a large time as placeholder
                times.append(float('inf'))
                errors.append(0.0)
                iterations.append(0)
                converged_list.append(False)

        # Compute scaling metrics
        valid_times = [t for t in times if t != float('inf')]
        if valid_times:
            speedups, efficiencies = compute_scaling_metrics(times, rank_counts)
        else:
            speedups = [0.0] * len(rank_counts)
            efficiencies = [0.0] * len(rank_counts)

        # Check if target efficiency achieved for max ranks
        max_rank_efficiency = efficiencies[-1] if efficiencies else 0.0
        efficiency_achieved = max_rank_efficiency >= args.target_efficiency

        all_results[exp_name] = ScalingStudyResults(
            experiment=exp_name,
            rank_counts=rank_counts,
            wall_times=times,
            speedups=speedups,
            efficiencies=efficiencies,
            analysis_errors=errors,
            target_efficiency=args.target_efficiency,
            efficiency_achieved=efficiency_achieved,
        )

        # Print summary for this experiment
        print(f"\n  {'Ranks':<8} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12}")
        print("  " + "-" * 42)
        for i, n in enumerate(rank_counts):
            time_str = f"{times[i]:.2f}" if times[i] != float('inf') else "N/A"
            speedup_str = f"{speedups[i]:.2f}" if speedups[i] else "N/A"
            eff_str = f"{efficiencies[i]*100:.1f}%" if efficiencies[i] else "N/A"
            print(f"  {n:<8} {time_str:<12} {speedup_str:<10} {eff_str:<12}")

        status = "PASSED" if efficiency_achieved else "FAILED"
        print(f"\n  Target efficiency ({args.target_efficiency*100:.0f}%) at {args.max_ranks} ranks: {status}")

    # Save results to JSON
    results_file = DATA_DIR / "scaling_results.json"
    output_data = {
        "metadata": {
            "max_ranks": args.max_ranks,
            "target_efficiency": args.target_efficiency,
            "quick_mode": args.quick,
        },
        "experiments": {
            name: asdict(result) for name, result in all_results.items()
        }
    }

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nScaling results saved to: {results_file}")

    # Generate plot
    figure_path = FIGURES_DIR / "scaling_study.png"
    create_scaling_plot(all_results, str(figure_path))

    # Final summary
    print("\n" + "=" * 70)
    print("SCALING STUDY SUMMARY")
    print("=" * 70)

    all_passed = all(r.efficiency_achieved for r in all_results.values())

    for name, result in all_results.items():
        status = "PASSED" if result.efficiency_achieved else "FAILED"
        max_eff = result.efficiencies[-1] * 100 if result.efficiencies else 0.0
        print(f"{name}: Efficiency at {args.max_ranks} ranks = {max_eff:.1f}% [{status}]")

    print(f"\nOverall: {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

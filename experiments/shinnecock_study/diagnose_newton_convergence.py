#!/usr/bin/env python3
"""
Diagnostic script to analyze Newton solver convergence issues.

Analyzes convergence patterns across timesteps to identify problematic regions.
Produces both text analysis and matplotlib visualizations.

Usage:
    python diagnose_newton_convergence.py <logfile> [--plot] [--save <output_dir>]

Examples:
    python diagnose_newton_convergence.py /tmp/phase1.log
    python diagnose_newton_convergence.py /tmp/phase1.log --plot
    python diagnose_newton_convergence.py /tmp/phase1.log --plot --save ./figures
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def parse_newton_log(logfile):
    """Parse Newton iteration details from log file."""
    timestep_data = defaultdict(list)
    current_timestep = None

    with open(logfile, 'r') as f:
        for line in f:
            # Match timestep markers
            if match := re.search(r'Timestep (\d+), t=([\d.e+-]+)', line):
                current_timestep = int(match.group(1))
                t_value = float(match.group(2))
                if current_timestep not in timestep_data:
                    timestep_data[current_timestep] = {
                        't': t_value,
                        'iterations': [],
                        'converged': False
                    }

            # Match Newton iterations
            if current_timestep and (match := re.search(
                r'Newton Iteration (\d+): Correction norm ([\d.e+-]+)', line
            )):
                iter_num = int(match.group(1))
                correction = float(match.group(2))
                timestep_data[current_timestep]['iterations'].append({
                    'iter': iter_num,
                    'correction': correction
                })

            # Match convergence
            if current_timestep and 'Newton solver converged' in line:
                timestep_data[current_timestep]['converged'] = True
                if match := re.search(r'converged in (\d+) iterations', line):
                    timestep_data[current_timestep]['num_iters'] = int(match.group(1))

    return dict(timestep_data)


def analyze_convergence_trends(timestep_data):
    """Analyze convergence patterns and identify problematic timesteps."""
    print("="*70)
    print("NEWTON SOLVER CONVERGENCE ANALYSIS")
    print("="*70)

    problematic = []
    slow_convergence = []

    for ts in sorted(timestep_data.keys()):
        data = timestep_data[ts]
        num_iters = len(data['iterations'])
        converged = data['converged']

        # Flag problematic timesteps
        if not converged:
            problematic.append((ts, num_iters, "Did not converge"))
        elif num_iters > 5:
            slow_convergence.append((ts, num_iters))

    print(f"\nTotal timesteps analyzed: {len(timestep_data)}")
    print(f"Converged: {sum(1 for d in timestep_data.values() if d['converged'])}")
    print(f"Failed to converge: {len(problematic)}")
    print(f"Slow convergence (>5 iters): {len(slow_convergence)}")

    if problematic:
        print("\n" + "="*70)
        print("PROBLEMATIC TIMESTEPS")
        print("="*70)
        for ts, iters, reason in problematic:
            t = timestep_data[ts]['t']
            print(f"  Timestep {ts} (t={t:.1f}s): {reason}")
            print(f"    Newton iterations attempted: {iters}")
            if timestep_data[ts]['iterations']:
                last_corr = timestep_data[ts]['iterations'][-1]['correction']
                print(f"    Last correction norm: {last_corr:.4e}")

    if slow_convergence:
        print("\n" + "="*70)
        print("SLOW CONVERGENCE TIMESTEPS")
        print("="*70)
        for ts, iters in sorted(slow_convergence, key=lambda x: x[1], reverse=True)[:10]:
            t = timestep_data[ts]['t']
            print(f"  Timestep {ts} (t={t:.1f}s): {iters} iterations")

    # Convergence rate analysis
    print("\n" + "="*70)
    print("CONVERGENCE RATE DISTRIBUTION")
    print("="*70)
    convergence_bins = defaultdict(int)
    for data in timestep_data.values():
        if data['converged']:
            num_iters = data.get('num_iters', len(data['iterations']))
            convergence_bins[num_iters] += 1

    for iters in sorted(convergence_bins.keys()):
        count = convergence_bins[iters]
        bar = '\u2588' * (count // 2)
        print(f"  {iters} iterations: {bar} ({count})")

    print("\n" + "="*70)


def plot_convergence(timestep_data, save_dir=None):
    """Create matplotlib visualizations of Newton convergence patterns.

    Produces a 3-panel figure:
      1. Newton iterations per timestep (bar chart)
      2. Final correction norm per timestep
      3. Convergence history for selected challenging timesteps

    Args:
        timestep_data: Dict from parse_newton_log()
        save_dir: If provided, save figures to this directory instead of showing
    """
    import matplotlib.pyplot as plt
    import numpy as np

    sorted_ts = sorted(timestep_data.keys())
    times = [timestep_data[ts]['t'] for ts in sorted_ts]
    times_hours = [t / 3600.0 for t in times]

    # --- Data extraction ---
    iters_per_step = []
    final_corrections = []
    converged_flags = []
    for ts in sorted_ts:
        data = timestep_data[ts]
        n_iters = data.get('num_iters', len(data['iterations']))
        iters_per_step.append(n_iters)
        converged_flags.append(data['converged'])

        if data['iterations']:
            final_corrections.append(data['iterations'][-1]['correction'])
        else:
            final_corrections.append(float('nan'))

    iters_per_step = np.array(iters_per_step)
    final_corrections = np.array(final_corrections)
    converged_flags = np.array(converged_flags)

    # --- Figure 1: Overview (2-panel) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: Iterations per timestep
    ax = axes[0]
    colors = ['#2ecc71' if c else '#e74c3c' for c in converged_flags]
    ax.bar(times_hours, iters_per_step, color=colors, width=0.08, edgecolor='none')
    ax.set_ylabel('Newton Iterations')
    ax.set_title('Newton Solver Convergence by Timestep')
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Typical threshold')

    # Mark failed timesteps
    failed_idx = ~converged_flags
    if failed_idx.any():
        failed_times = np.array(times_hours)[failed_idx]
        failed_iters = iters_per_step[failed_idx]
        ax.scatter(failed_times, failed_iters, color='red', marker='x',
                   s=100, zorder=5, label='Failed to converge')

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 2: Final correction norm
    ax = axes[1]
    valid = ~np.isnan(final_corrections)
    ax.semilogy(np.array(times_hours)[valid & converged_flags],
                final_corrections[valid & converged_flags],
                'o-', color='#2ecc71', markersize=3, label='Converged')
    if (valid & ~converged_flags).any():
        ax.semilogy(np.array(times_hours)[valid & ~converged_flags],
                    final_corrections[valid & ~converged_flags],
                    'x', color='#e74c3c', markersize=8, label='Failed')
    ax.set_ylabel('Final Correction Norm')
    ax.set_xlabel('Simulation Time (hours)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / 'newton_convergence_overview.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path / 'newton_convergence_overview.png'}")
    else:
        plt.show()

    # --- Figure 2: Convergence histories for challenging timesteps ---
    # Find timesteps with most iterations (top 5)
    challenging = sorted(
        [(ts, len(timestep_data[ts]['iterations'])) for ts in sorted_ts
         if len(timestep_data[ts]['iterations']) > 0],
        key=lambda x: x[1], reverse=True
    )[:5]

    if challenging:
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        for ts, _ in challenging:
            data = timestep_data[ts]
            corrections = [it['correction'] for it in data['iterations']]
            iters = list(range(1, len(corrections) + 1))
            t_hours = data['t'] / 3600.0
            label = f"ts={ts} (t={t_hours:.1f}h, {'OK' if data['converged'] else 'FAIL'})"
            linestyle = '-' if data['converged'] else '--'
            ax2.semilogy(iters, corrections, f'{linestyle}o', markersize=4, label=label)

        ax2.set_xlabel('Newton Iteration')
        ax2.set_ylabel('Correction Norm (relative)')
        ax2.set_title('Newton Convergence History - Challenging Timesteps')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig2.tight_layout()

        if save_dir:
            fig2.savefig(save_path / 'newton_convergence_histories.png', dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path / 'newton_convergence_histories.png'}")
        else:
            plt.show()

    # --- Figure 3: Iteration count histogram ---
    fig3, ax3 = plt.subplots(figsize=(8, 5))

    converged_iters = iters_per_step[converged_flags]
    if len(converged_iters) > 0:
        bins = range(0, int(converged_iters.max()) + 2)
        ax3.hist(converged_iters, bins=bins, color='#3498db', edgecolor='white',
                 alpha=0.8, align='left')
        ax3.set_xlabel('Newton Iterations to Convergence')
        ax3.set_ylabel('Number of Timesteps')
        ax3.set_title('Distribution of Newton Iterations')

        mean_iters = converged_iters.mean()
        ax3.axvline(x=mean_iters, color='red', linestyle='--',
                     label=f'Mean = {mean_iters:.1f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

    fig3.tight_layout()

    if save_dir:
        fig3.savefig(save_path / 'newton_iteration_histogram.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path / 'newton_iteration_histogram.png'}")
    else:
        plt.show()

    plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze Newton solver convergence from log files"
    )
    parser.add_argument("logfile", help="Path to Newton solver log file")
    parser.add_argument("--plot", action="store_true",
                        help="Generate matplotlib visualizations")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figures to this directory (implies --plot)")

    args = parser.parse_args()

    if not Path(args.logfile).exists():
        print(f"Error: Log file not found: {args.logfile}")
        sys.exit(1)

    timestep_data = parse_newton_log(args.logfile)
    analyze_convergence_trends(timestep_data)

    if args.plot or args.save:
        plot_convergence(timestep_data, save_dir=args.save)

#!/usr/bin/env python3
"""
HPC Wrapper for Shinnecock Inlet Data Assimilation Experiments on Frontera.

This script provides a high-level interface for running Shinnecock Inlet
simulations and DA experiments on TACC Frontera. It handles:

- Problem configuration with Frontera-optimized defaults
- Observation setup (50% spatial, configurable temporal)
- Memory monitoring and checkpointing
- Multi-node scaling configurations
- Results aggregation and reporting

Usage:
    # Direct execution (for testing)
    python run_experiment.py --mode forward --T 24

    # Via SLURM (recommended)
    sbatch job_submit.sh

    # Strong scaling study
    python run_experiment.py --mode scaling --min-nodes 1 --max-nodes 8

Example configurations:
    # 24-hour forward run
    python run_experiment.py --mode forward --T 24

    # 4D-Var with 50% observations, 6-hour interval
    python run_experiment.py --mode 4dvar --T 48 --obs-frequency 36

    # DC-WME on 4 nodes
    python run_experiment.py --mode dcwme --T 24 --nodes 4
"""

import argparse
import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))


@dataclass
class ShinnecockConfig:
    """Configuration for Shinnecock Inlet experiment on Frontera."""

    # Simulation parameters
    dt: float = 600.0  # Time step (seconds)
    T_hours: float = 24.0  # Simulation duration (hours)
    solver: str = "dg"  # Solver type: dg, supg, dgnc
    alpha: float = 1.5  # Wetting/drying parameter
    theta: float = 1.0  # Time stepping parameter
    dramp: float = 2.0  # Tidal ramp-up time (days)

    # Data file location
    adios_file: str = "data/shinnecock_inlet"

    # DA parameters
    da_mode: str = "none"  # none, 4dvar, dcwme
    obs_fraction: float = 0.5  # Spatial observation density
    obs_frequency: int = 6  # Temporal observation frequency (timesteps)
    obs_noise: float = 0.01  # Observation noise level
    background_error: float = 0.1  # Background error std
    max_da_iter: int = 50  # Max optimization iterations

    # HPC parameters
    nodes: int = 1  # Number of compute nodes
    tasks_per_node: int = 56  # MPI ranks per node (Frontera CLX)

    # Output configuration
    output_prefix: str = "shinnecock_frontera"
    output_dir: Optional[str] = None

    # Monitoring
    verbose: bool = True
    profile: bool = True
    checkpoint_interval: int = 100  # Checkpoint every N timesteps

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def total_ranks(self) -> int:
        """Total MPI ranks."""
        return self.nodes * self.tasks_per_node

    @property
    def nt(self) -> int:
        """Number of time steps."""
        return int(self.T_hours * 3600 / self.dt)

    @property
    def n_observations(self) -> int:
        """Number of observation times."""
        return len(range(self.obs_frequency, self.nt + 1, self.obs_frequency))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="HPC wrapper for Shinnecock Inlet experiments on Frontera",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["forward", "4dvar", "dcwme", "scaling", "generate-scripts"],
        default="forward",
        help="Experiment mode",
    )

    # Simulation parameters
    parser.add_argument("--dt", type=float, default=600, help="Time step (seconds)")
    parser.add_argument(
        "--T", type=float, default=24, help="Simulation duration (hours)"
    )
    parser.add_argument(
        "--solver",
        choices=["dg", "supg", "dgnc"],
        default="dg",
        help="Solver type",
    )

    # DA parameters
    parser.add_argument(
        "--obs-fraction", type=float, default=0.5, help="Observation spatial fraction"
    )
    parser.add_argument(
        "--obs-frequency",
        type=int,
        default=6,
        help="Observation frequency (timesteps)",
    )
    parser.add_argument(
        "--obs-noise", type=float, default=0.01, help="Observation noise level"
    )
    parser.add_argument(
        "--background-error", type=float, default=0.1, help="Background error std"
    )
    parser.add_argument(
        "--max-iter", type=int, default=50, help="Max DA iterations"
    )

    # HPC parameters
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument(
        "--tasks-per-node", type=int, default=56, help="Tasks per node"
    )

    # Scaling study parameters
    parser.add_argument(
        "--min-nodes", type=int, default=1, help="Min nodes for scaling study"
    )
    parser.add_argument(
        "--max-nodes", type=int, default=8, help="Max nodes for scaling study"
    )

    # Output
    parser.add_argument(
        "--output-prefix", type=str, default="shinnecock_frontera", help="Output prefix"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")

    return parser.parse_args()


def estimate_memory_usage(config: ShinnecockConfig) -> Dict[str, float]:
    """
    Estimate memory usage for Shinnecock simulation.

    Shinnecock mesh has approximately 5000 nodes with real bathymetry data.
    Memory estimates are based on DOLFINx function storage patterns.

    Returns memory estimates in GB.
    """
    # Shinnecock mesh characteristics (approximate)
    n_nodes = 5000  # Mesh nodes
    n_dofs_per_field = n_nodes * 2  # P1 on triangles (roughly)
    n_fields = 3  # h, u, v
    n_total_dofs = n_dofs_per_field * n_fields

    # Base memory per DOF (bytes): 8 (double) + overhead
    bytes_per_dof = 16

    # Memory components (in GB)
    estimates = {}

    # State vectors (current, old, increment)
    estimates["state_vectors"] = 3 * n_total_dofs * bytes_per_dof / 1e9

    # Jacobian matrix (sparse, ~50 nnz per row)
    nnz_per_row = 50
    estimates["jacobian"] = n_total_dofs * nnz_per_row * (bytes_per_dof + 4) / 1e9

    # Trajectory storage (if saving states)
    estimates["trajectory"] = config.nt * n_total_dofs * bytes_per_dof / 1e9

    # DA-specific: stored Jacobians for adjoint
    if config.da_mode != "none":
        # Store Jacobian at each timestep for adjoint computation
        estimates["jacobian_history"] = (
            config.nt * n_total_dofs * nnz_per_row * (bytes_per_dof + 4) / 1e9
        )
        # Optimization workspace
        estimates["optimization"] = 0.5  # GB for L-BFGS history

    # Total estimate
    estimates["total"] = sum(estimates.values())

    # Per-rank estimate (assuming uniform distribution)
    estimates["per_rank"] = estimates["total"] / config.total_ranks

    return estimates


def get_recommended_config(
    T_hours: float, da_mode: str
) -> ShinnecockConfig:
    """
    Get recommended configuration based on simulation parameters.

    Provides sensible defaults for Shinnecock Inlet on Frontera.
    """
    config = ShinnecockConfig(T_hours=T_hours, da_mode=da_mode)

    # Adjust nodes based on problem size and mode
    if da_mode != "none":
        # DA experiments need more memory for Jacobian storage
        if T_hours <= 24:
            config.nodes = 2
        elif T_hours <= 72:
            config.nodes = 4
        else:
            config.nodes = 8
    else:
        # Forward runs are less memory intensive
        if T_hours <= 48:
            config.nodes = 1
        elif T_hours <= 120:
            config.nodes = 2
        else:
            config.nodes = 4

    # Observation setup for DA
    if da_mode != "none":
        # 50% spatial coverage
        config.obs_fraction = 0.5
        # ~6-hour observation interval
        config.obs_frequency = int(6 * 3600 / config.dt)

    return config


def generate_slurm_script(config: ShinnecockConfig, output_path: Path) -> str:
    """Generate a SLURM script for the given configuration."""
    script = f"""#!/bin/bash
#SBATCH --job-name=shin_{config.da_mode}_{config.nodes}n
#SBATCH --output=shinnecock_%j.out
#SBATCH --error=shinnecock_%j.err
#SBATCH --partition=normal
#SBATCH --nodes={config.nodes}
#SBATCH --ntasks-per-node={config.tasks_per_node}
#SBATCH --time=04:00:00
#SBATCH --account=YOUR_ALLOCATION

# Auto-generated SLURM script for Shinnecock {config.da_mode.upper()}
# Generated: {datetime.now().isoformat()}

# Environment setup
source {SCRIPT_DIR}/environment_setup.sh

# Run simulation
cd {PROJECT_ROOT}

ibrun python examples/shinnecock.py \\
    --dt {config.dt} \\
    --T {config.T_hours} \\
    --solver {config.solver} \\
    --alpha {config.alpha} \\
    --da-mode {config.da_mode} \\
    --obs-fraction {config.obs_fraction} \\
    --obs-frequency {config.obs_frequency} \\
    --obs-noise {config.obs_noise} \\
    --background-error {config.background_error} \\
    --max-da-iter {config.max_da_iter} \\
    --output-prefix {config.output_prefix} \\
    --verbose \\
    --profile \\
    --newton-quiet \\
    --newton-store
"""
    output_path.write_text(script)
    return str(output_path)


def run_experiment(config: ShinnecockConfig, dry_run: bool = False) -> Dict[str, Any]:
    """
    Run a single experiment with the given configuration.

    Returns a dictionary with results and timing information.
    """
    # Build command
    cmd = [
        "ibrun" if os.environ.get("TACC_SYSTEM") else "mpirun",
    ]

    if not os.environ.get("TACC_SYSTEM"):
        cmd.extend(["-n", str(config.total_ranks)])

    cmd.extend([
        "python",
        str(PROJECT_ROOT / "examples" / "shinnecock.py"),
        "--dt", str(config.dt),
        "--T", str(config.T_hours),
        "--solver", config.solver,
        "--alpha", str(config.alpha),
        "--da-mode", config.da_mode,
        "--output-prefix", config.output_prefix,
    ])

    if config.da_mode != "none":
        cmd.extend([
            "--obs-fraction", str(config.obs_fraction),
            "--obs-frequency", str(config.obs_frequency),
            "--obs-noise", str(config.obs_noise),
            "--background-error", str(config.background_error),
            "--max-da-iter", str(config.max_da_iter),
        ])

    if config.verbose:
        cmd.append("--verbose")
    if config.profile:
        cmd.append("--profile")

    cmd.extend(["--newton-quiet", "--newton-store"])

    print(f"\nCommand: {' '.join(cmd)}")

    if dry_run:
        return {"status": "dry_run", "command": " ".join(cmd)}

    # Run the experiment
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=4 * 3600,  # 4 hour timeout
        )
        elapsed = time.time() - start_time

        return {
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "elapsed_time": elapsed,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "config": config.to_dict(),
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "config": config.to_dict(),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "config": config.to_dict(),
        }


def run_scaling_study(
    args, min_nodes: int = 1, max_nodes: int = 8
) -> List[Dict[str, Any]]:
    """
    Run a strong scaling study.

    Tests the same problem on increasing node counts.
    """
    results = []
    node_counts = [1, 2, 4, 8]
    node_counts = [n for n in node_counts if min_nodes <= n <= max_nodes]

    base_config = ShinnecockConfig(
        dt=args.dt,
        T_hours=args.T,
        solver=args.solver,
        da_mode=args.mode if args.mode in ["4dvar", "dcwme"] else "none",
        obs_fraction=args.obs_fraction,
        obs_frequency=args.obs_frequency,
        output_prefix=args.output_prefix,
        verbose=args.verbose,
        profile=True,  # Always profile for scaling studies
    )

    print("=" * 70)
    print("Shinnecock Inlet Strong Scaling Study")
    print("=" * 70)
    print(f"Configuration: T={args.T}h, dt={args.dt}s, solver={args.solver}")
    print(f"DA mode: {base_config.da_mode}")
    print(f"Node counts: {node_counts}")
    print("=" * 70)

    for nodes in node_counts:
        config = ShinnecockConfig(**base_config.to_dict())
        config.nodes = nodes
        config.output_prefix = f"{args.output_prefix}_n{nodes}"

        print(f"\nRunning with {nodes} nodes ({config.total_ranks} ranks)...")

        # Estimate memory
        mem = estimate_memory_usage(config)
        print(f"  Estimated memory: {mem['total']:.2f} GB total, {mem['per_rank']:.2f} GB/rank")

        result = run_experiment(config, dry_run=args.dry_run)
        result["nodes"] = nodes
        result["total_ranks"] = config.total_ranks
        results.append(result)

        if result["status"] == "success":
            print(f"  Completed in {result['elapsed_time']:.2f}s")
        else:
            print(f"  Status: {result['status']}")

    # Save scaling results
    output_file = Path(args.output_dir or ".") / f"scaling_study_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScaling results saved to: {output_file}")

    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Print header
    print("=" * 70)
    print("Shinnecock Inlet HPC Experiment Runner")
    print("TACC Frontera Configuration")
    print("=" * 70)

    if args.mode == "scaling":
        # Run scaling study
        results = run_scaling_study(args, args.min_nodes, args.max_nodes)

    elif args.mode == "generate-scripts":
        # Generate SLURM scripts for various configurations
        configs = [
            ("forward_24h", ShinnecockConfig(T_hours=24, da_mode="none")),
            ("forward_120h", ShinnecockConfig(T_hours=120, da_mode="none", nodes=2)),
            ("4dvar_24h", ShinnecockConfig(T_hours=24, da_mode="4dvar", nodes=2)),
            ("4dvar_48h", ShinnecockConfig(T_hours=48, da_mode="4dvar", nodes=4)),
            ("dcwme_24h", ShinnecockConfig(T_hours=24, da_mode="dcwme", nodes=2)),
        ]

        scripts_dir = SCRIPT_DIR / "generated_scripts"
        scripts_dir.mkdir(exist_ok=True)

        print(f"\nGenerating SLURM scripts in: {scripts_dir}")
        for name, config in configs:
            script_path = scripts_dir / f"submit_{name}.sh"
            generate_slurm_script(config, script_path)
            print(f"  Created: {script_path.name}")

    else:
        # Single experiment
        da_mode = "none" if args.mode == "forward" else args.mode

        config = ShinnecockConfig(
            dt=args.dt,
            T_hours=args.T,
            solver=args.solver,
            da_mode=da_mode,
            obs_fraction=args.obs_fraction,
            obs_frequency=args.obs_frequency,
            obs_noise=args.obs_noise,
            background_error=args.background_error,
            max_da_iter=args.max_iter,
            nodes=args.nodes,
            tasks_per_node=args.tasks_per_node,
            output_prefix=args.output_prefix,
            verbose=args.verbose,
            profile=args.profile,
        )

        # Print configuration
        print(f"\nConfiguration:")
        print(f"  Mode:          {config.da_mode}")
        print(f"  Duration:      {config.T_hours} hours ({config.nt} timesteps)")
        print(f"  Solver:        {config.solver}")
        print(f"  Nodes:         {config.nodes} ({config.total_ranks} ranks)")

        if config.da_mode != "none":
            print(f"  Observations:  {config.obs_fraction*100:.0f}% spatial, every {config.obs_frequency} steps")
            print(f"  Obs times:     {config.n_observations}")

        # Memory estimate
        mem = estimate_memory_usage(config)
        print(f"\nMemory Estimate:")
        print(f"  Total:         {mem['total']:.2f} GB")
        print(f"  Per rank:      {mem['per_rank']:.2f} GB")

        if mem["per_rank"] > 3.0:
            print("  WARNING: Memory per rank exceeds 3 GB - consider using more nodes")

        # Run experiment
        print("\n" + "=" * 70)
        result = run_experiment(config, dry_run=args.dry_run)

        if result["status"] == "success":
            print(f"\nExperiment completed successfully in {result['elapsed_time']:.2f}s")
        elif result["status"] == "dry_run":
            print("\nDry run - no execution performed")
        else:
            print(f"\nExperiment failed with status: {result['status']}")
            if "stderr" in result:
                print(f"Error output:\n{result['stderr'][:1000]}")


if __name__ == "__main__":
    main()

"""
Load balancing metrics and analysis for MPI-parallel computations.

Provides utilities to track and report:
- Per-rank cell counts
- Per-rank DOF counts
- Communication volume per rank
- Load imbalance percentages
- Performance bottleneck identification
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
import dolfinx


class LoadBalancingMetrics:
    """
    Tracks and analyzes load balancing metrics for distributed mesh computations.

    Provides detailed statistics on how work and data are distributed across MPI ranks,
    helping identify load imbalance issues that can affect parallel scalability.
    """

    def __init__(self, comm: MPI.Comm = None):
        """
        Initialize load balancing metrics tracker.

        Args:
            comm: MPI communicator (defaults to MPI.COMM_WORLD)
        """
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Storage for metrics
        self.metrics_history: List[Dict] = []

    def compute_mesh_metrics(self, mesh: dolfinx.mesh.Mesh) -> Dict:
        """
        Compute mesh distribution metrics across all ranks.

        Args:
            mesh: The distributed mesh

        Returns:
            Dictionary containing mesh distribution statistics
        """
        # Get local mesh information
        num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
        num_cells_ghost = mesh.topology.index_map(mesh.topology.dim).num_ghosts
        num_cells_total = num_cells_local + num_cells_ghost

        # Gather data from all ranks
        all_cells_local = self.comm.allgather(num_cells_local)
        all_cells_ghost = self.comm.allgather(num_cells_ghost)
        all_cells_total = self.comm.allgather(num_cells_total)

        # Compute statistics
        global_cells = sum(all_cells_local)
        avg_cells_local = global_cells / self.size
        max_cells_local = max(all_cells_local)
        min_cells_local = min(all_cells_local)

        # Load imbalance: (max - avg) / avg * 100
        imbalance = (max_cells_local - avg_cells_local) / avg_cells_local * 100 if avg_cells_local > 0 else 0.0

        metrics = {
            'global_cells': global_cells,
            'local_cells_per_rank': all_cells_local,
            'ghost_cells_per_rank': all_cells_ghost,
            'total_cells_per_rank': all_cells_total,
            'avg_local_cells': avg_cells_local,
            'max_local_cells': max_cells_local,
            'min_local_cells': min_cells_local,
            'imbalance_percent': imbalance,
            'efficiency_percent': 100.0 - imbalance
        }

        return metrics

    def compute_dof_metrics(self, V: dolfinx.fem.FunctionSpace) -> Dict:
        """
        Compute DOF distribution metrics across all ranks.

        Args:
            V: Function space

        Returns:
            Dictionary containing DOF distribution statistics
        """
        # Get local DOF information
        num_dofs_local = V.dofmap.index_map.size_local
        num_dofs_ghost = V.dofmap.index_map.num_ghosts
        num_dofs_global = V.dofmap.index_map.size_global

        # Gather data from all ranks
        all_dofs_local = self.comm.allgather(num_dofs_local)
        all_dofs_ghost = self.comm.allgather(num_dofs_ghost)

        # Compute statistics
        avg_dofs_local = num_dofs_global / self.size
        max_dofs_local = max(all_dofs_local)
        min_dofs_local = min(all_dofs_local)

        # Load imbalance
        imbalance = (max_dofs_local - avg_dofs_local) / avg_dofs_local * 100 if avg_dofs_local > 0 else 0.0

        metrics = {
            'global_dofs': num_dofs_global,
            'local_dofs_per_rank': all_dofs_local,
            'ghost_dofs_per_rank': all_dofs_ghost,
            'avg_local_dofs': avg_dofs_local,
            'max_local_dofs': max_dofs_local,
            'min_local_dofs': min_dofs_local,
            'imbalance_percent': imbalance,
            'efficiency_percent': 100.0 - imbalance
        }

        return metrics

    def compute_communication_metrics(self, mesh: dolfinx.mesh.Mesh, V: dolfinx.fem.FunctionSpace) -> Dict:
        """
        Estimate communication volume based on ghost cells and DOFs.

        Args:
            mesh: The distributed mesh
            V: Function space

        Returns:
            Dictionary containing communication volume estimates
        """
        # Ghost cell and DOF counts
        num_ghost_cells = mesh.topology.index_map(mesh.topology.dim).num_ghosts
        num_ghost_dofs = V.dofmap.index_map.num_ghosts

        # Gather from all ranks
        all_ghost_cells = self.comm.allgather(num_ghost_cells)
        all_ghost_dofs = self.comm.allgather(num_ghost_dofs)

        # Estimate communication volume (in units of DOFs to communicate)
        total_ghost_dofs = sum(all_ghost_dofs)
        avg_ghost_dofs = total_ghost_dofs / self.size
        max_ghost_dofs = max(all_ghost_dofs)
        min_ghost_dofs = min(all_ghost_dofs)

        # Communication imbalance
        comm_imbalance = (max_ghost_dofs - avg_ghost_dofs) / avg_ghost_dofs * 100 if avg_ghost_dofs > 0 else 0.0

        metrics = {
            'ghost_cells_per_rank': all_ghost_cells,
            'ghost_dofs_per_rank': all_ghost_dofs,
            'total_ghost_dofs': total_ghost_dofs,
            'avg_ghost_dofs': avg_ghost_dofs,
            'max_ghost_dofs': max_ghost_dofs,
            'min_ghost_dofs': min_ghost_dofs,
            'comm_imbalance_percent': comm_imbalance
        }

        return metrics

    def compute_comprehensive_metrics(
        self,
        mesh: dolfinx.mesh.Mesh,
        V: dolfinx.fem.FunctionSpace,
        name: str = "default"
    ) -> Dict:
        """
        Compute all load balancing metrics at once.

        Args:
            mesh: The distributed mesh
            V: Function space
            name: Name for this metric snapshot

        Returns:
            Dictionary containing all metrics
        """
        mesh_metrics = self.compute_mesh_metrics(mesh)
        dof_metrics = self.compute_dof_metrics(V)
        comm_metrics = self.compute_communication_metrics(mesh, V)

        all_metrics = {
            'name': name,
            'num_ranks': self.size,
            'mesh': mesh_metrics,
            'dofs': dof_metrics,
            'communication': comm_metrics
        }

        # Store in history
        self.metrics_history.append(all_metrics)

        return all_metrics

    def print_metrics(self, metrics: Optional[Dict] = None):
        """
        Print formatted metrics report.

        Args:
            metrics: Metrics dictionary (uses most recent if None)
        """
        if metrics is None:
            if not self.metrics_history:
                if self.rank == 0:
                    print("No metrics available. Call compute_comprehensive_metrics first.")
                return
            metrics = self.metrics_history[-1]

        if self.rank == 0:
            print("\n" + "="*70)
            print(f"Load Balancing Metrics Report: {metrics['name']}")
            print(f"Number of MPI Ranks: {metrics['num_ranks']}")
            print("="*70)

            # Mesh metrics
            print("\n--- Mesh Distribution ---")
            m = metrics['mesh']
            print(f"Global cells: {m['global_cells']}")
            print(f"Cells per rank (local): min={m['min_local_cells']}, "
                  f"avg={m['avg_local_cells']:.1f}, max={m['max_local_cells']}")
            print(f"Load imbalance: {m['imbalance_percent']:.2f}%")
            print(f"Load efficiency: {m['efficiency_percent']:.2f}%")

            # DOF metrics
            print("\n--- DOF Distribution ---")
            d = metrics['dofs']
            print(f"Global DOFs: {d['global_dofs']}")
            print(f"DOFs per rank (local): min={d['min_local_dofs']}, "
                  f"avg={d['avg_local_dofs']:.1f}, max={d['max_local_dofs']}")
            print(f"Load imbalance: {d['imbalance_percent']:.2f}%")
            print(f"Load efficiency: {d['efficiency_percent']:.2f}%")

            # Communication metrics
            print("\n--- Communication Volume ---")
            c = metrics['communication']
            print(f"Total ghost DOFs: {c['total_ghost_dofs']}")
            print(f"Ghost DOFs per rank: min={c['min_ghost_dofs']}, "
                  f"avg={c['avg_ghost_dofs']:.1f}, max={c['max_ghost_dofs']}")
            print(f"Communication imbalance: {c['comm_imbalance_percent']:.2f}%")

            # Overall assessment
            print("\n--- Overall Assessment ---")
            worst_imbalance = max(
                m['imbalance_percent'],
                d['imbalance_percent'],
                c['comm_imbalance_percent']
            )

            if worst_imbalance < 10.0:
                status = "EXCELLENT - Load is well balanced"
            elif worst_imbalance < 20.0:
                status = "GOOD - Acceptable load balance"
            elif worst_imbalance < 30.0:
                status = "FAIR - Consider repartitioning for better scalability"
            else:
                status = "POOR - Significant load imbalance detected"

            print(f"Worst imbalance: {worst_imbalance:.2f}%")
            print(f"Status: {status}")
            print("="*70 + "\n")

    def print_detailed_per_rank_metrics(self, metrics: Optional[Dict] = None):
        """
        Print detailed per-rank breakdown of metrics.

        Args:
            metrics: Metrics dictionary (uses most recent if None)
        """
        if metrics is None:
            if not self.metrics_history:
                if self.rank == 0:
                    print("No metrics available.")
                return
            metrics = self.metrics_history[-1]

        if self.rank == 0:
            print("\n" + "="*70)
            print(f"Detailed Per-Rank Metrics: {metrics['name']}")
            print("="*70)

            print("\nRank | Local Cells | Ghost Cells | Local DOFs | Ghost DOFs")
            print("-" * 70)

            for r in range(metrics['num_ranks']):
                local_cells = metrics['mesh']['local_cells_per_rank'][r]
                ghost_cells = metrics['mesh']['ghost_cells_per_rank'][r]
                local_dofs = metrics['dofs']['local_dofs_per_rank'][r]
                ghost_dofs = metrics['dofs']['ghost_dofs_per_rank'][r]

                print(f"{r:4d} | {local_cells:11d} | {ghost_cells:11d} | "
                      f"{local_dofs:10d} | {ghost_dofs:10d}")

            print("="*70 + "\n")

    def check_load_balance_quality(
        self,
        mesh: dolfinx.mesh.Mesh,
        V: dolfinx.fem.FunctionSpace,
        threshold: float = 20.0
    ) -> Tuple[bool, str]:
        """
        Check if load balance meets quality threshold.

        Args:
            mesh: The distributed mesh
            V: Function space
            threshold: Maximum acceptable imbalance percentage

        Returns:
            Tuple of (is_balanced, message)
        """
        metrics = self.compute_comprehensive_metrics(mesh, V, "quality_check")

        worst_imbalance = max(
            metrics['mesh']['imbalance_percent'],
            metrics['dofs']['imbalance_percent'],
            metrics['communication']['comm_imbalance_percent']
        )

        is_balanced = worst_imbalance <= threshold

        if is_balanced:
            message = f"Load balance is good: {worst_imbalance:.2f}% <= {threshold}% threshold"
        else:
            message = (f"Load imbalance detected: {worst_imbalance:.2f}% > {threshold}% threshold. "
                      f"Consider repartitioning the mesh.")

        return is_balanced, message


class CommunicationTracker:
    """
    Tracks MPI communication volume and patterns during execution.

    Helps identify communication bottlenecks in parallel algorithms.
    """

    def __init__(self, comm: MPI.Comm = None):
        """
        Initialize communication tracker.

        Args:
            comm: MPI communicator
        """
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Track communication operations
        self.comm_log: List[Dict] = []

    def record_scatter_forward(self, vec_size: int, context: str = ""):
        """
        Record a scatter_forward operation.

        Args:
            vec_size: Size of vector being scattered
            context: Description of the operation
        """
        self.comm_log.append({
            'type': 'scatter_forward',
            'size': vec_size,
            'context': context,
            'rank': self.rank
        })

    def record_scatter_reverse(self, vec_size: int, context: str = ""):
        """
        Record a scatter_reverse operation.

        Args:
            vec_size: Size of vector being scattered
            context: Description of the operation
        """
        self.comm_log.append({
            'type': 'scatter_reverse',
            'size': vec_size,
            'context': context,
            'rank': self.rank
        })

    def record_allreduce(self, data_size: int, context: str = ""):
        """
        Record an allreduce operation.

        Args:
            data_size: Amount of data in the reduction
            context: Description of the operation
        """
        self.comm_log.append({
            'type': 'allreduce',
            'size': data_size,
            'context': context,
            'rank': self.rank
        })

    def record_broadcast(self, data_size: int, root: int, context: str = ""):
        """
        Record a broadcast operation.

        Args:
            data_size: Amount of data being broadcast
            root: Root rank for broadcast
            context: Description of the operation
        """
        self.comm_log.append({
            'type': 'broadcast',
            'size': data_size,
            'root': root,
            'context': context,
            'rank': self.rank
        })

    def get_summary(self) -> Dict:
        """
        Get summary of communication operations.

        Returns:
            Dictionary with communication statistics
        """
        # Gather all logs to root
        all_logs = self.comm.gather(self.comm_log, root=0)

        summary = {
            'total_operations': 0,
            'operations_by_type': {},
            'total_volume': 0,
            'volume_by_type': {}
        }

        if self.rank == 0 and all_logs:
            # Flatten all logs
            all_operations = [op for log in all_logs for op in log]

            summary['total_operations'] = len(all_operations)

            # Count by type
            for op in all_operations:
                op_type = op['type']
                size = op.get('size', 0)

                if op_type not in summary['operations_by_type']:
                    summary['operations_by_type'][op_type] = 0
                    summary['volume_by_type'][op_type] = 0

                summary['operations_by_type'][op_type] += 1
                summary['volume_by_type'][op_type] += size
                summary['total_volume'] += size

        # Broadcast summary to all ranks
        summary = self.comm.bcast(summary, root=0)

        return summary

    def print_summary(self):
        """Print communication summary report."""
        summary = self.get_summary()

        if self.rank == 0:
            print("\n" + "="*70)
            print("Communication Tracking Summary")
            print("="*70)
            print(f"Total communication operations: {summary['total_operations']}")
            print(f"Total data volume: {summary['total_volume']} elements")
            print("\nOperations by type:")
            for op_type, count in summary['operations_by_type'].items():
                volume = summary['volume_by_type'][op_type]
                avg_size = volume / count if count > 0 else 0
                print(f"  {op_type}: {count} calls, {volume} elements, "
                      f"avg {avg_size:.1f} elements/call")
            print("="*70 + "\n")

    def clear(self):
        """Clear communication log."""
        self.comm_log.clear()

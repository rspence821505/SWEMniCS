"""
Parallel operations and utilities for MPI-based data assimilation.

Provides helper functions for distributed vector/matrix operations,
collective communications, and parallel I/O.
"""

from typing import List, Optional, Tuple
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI


class ParallelContext:
    """
    Manages MPI parallel context and provides utility operations.
    """

    def __init__(self, comm: MPI.Comm = None):
        """
        Initialize parallel context.

        Args:
            comm: MPI communicator (defaults to MPI.COMM_WORLD)
        """
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def is_root(self) -> bool:
        """Check if current process is root."""
        return self.rank == 0

    def barrier(self):
        """Synchronize all processes."""
        self.comm.Barrier()

    def broadcast(self, data, root: int = 0):
        """
        Broadcast data from root to all processes.

        Args:
            data: Data to broadcast (only meaningful on root)
            root: Root rank

        Returns:
            Broadcasted data on all ranks
        """
        return self.comm.bcast(data, root=root)

    def gather(self, data, root: int = 0):
        """
        Gather data from all processes to root.

        Args:
            data: Local data
            root: Root rank

        Returns:
            List of all data on root, None elsewhere
        """
        return self.comm.gather(data, root=root)

    def allgather(self, data):
        """
        Gather data from all processes to all processes.

        Args:
            data: Local data

        Returns:
            List of all data on all ranks
        """
        return self.comm.allgather(data)

    def reduce_sum(self, value: float, root: int = 0) -> float:
        """
        Sum value across all processes.

        Args:
            value: Local value
            root: Root rank for result

        Returns:
            Sum on root, undefined elsewhere
        """
        return self.comm.reduce(value, op=MPI.SUM, root=root)

    def allreduce_sum(self, value: float) -> float:
        """
        Sum value across all processes, result on all ranks.

        Args:
            value: Local value

        Returns:
            Sum on all ranks
        """
        return self.comm.allreduce(value, op=MPI.SUM)


class DistributedVectorOps:
    """
    Operations on distributed PETSc vectors.
    """

    @staticmethod
    def create_distributed_vec(size: int, comm: MPI.Comm = None) -> PETSc.Vec:
        """
        Create distributed vector with default partitioning.

        Args:
            size: Global size
            comm: MPI communicator

        Returns:
            Distributed PETSc vector
        """
        comm = comm or MPI.COMM_WORLD
        vec = PETSc.Vec().create(comm=comm)
        vec.setSizes(size)
        vec.setFromOptions()
        return vec

    @staticmethod
    def local_to_global(
        local_values: np.ndarray, vec: PETSc.Vec, local_indices: np.ndarray
    ):
        """
        Insert local values into global distributed vector.

        Args:
            local_values: Values owned by this process
            vec: Global vector
            local_indices: Global indices for local values
        """
        vec.setValues(local_indices, local_values, addv=PETSc.InsertMode.INSERT_VALUES)
        vec.assemblyBegin()
        vec.assemblyEnd()

    @staticmethod
    def global_to_local(vec: PETSc.Vec) -> np.ndarray:
        """
        Extract local portion of distributed vector.

        Args:
            vec: Distributed vector

        Returns:
            Local values as numpy array
        """
        return vec.getArray()

    @staticmethod
    def parallel_dot(v1: PETSc.Vec, v2: PETSc.Vec) -> float:
        """
        Compute parallel dot product.

        Handles communication internally.

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Global dot product
        """
        return v1.dot(v2)

    @staticmethod
    def parallel_norm(v: PETSc.Vec, norm_type: int = PETSc.NormType.NORM_2) -> float:
        """
        Compute parallel norm.

        Args:
            v: Vector
            norm_type: Type of norm (default: 2-norm)

        Returns:
            Global norm
        """
        return v.norm(norm_type)


class DistributedMatrixOps:
    """
    Operations on distributed PETSc matrices.
    """

    @staticmethod
    def create_distributed_mat(
        size: Tuple[int, int], nnz: int = 10, comm: MPI.Comm = None
    ) -> PETSc.Mat:
        """
        Create distributed sparse matrix.

        Args:
            size: (nrows, ncols) global size
            nnz: Expected nonzeros per row
            comm: MPI communicator

        Returns:
            Distributed sparse matrix
        """
        comm = comm or MPI.COMM_WORLD
        mat = PETSc.Mat().create(comm=comm)
        mat.setSizes(size)
        mat.setType(PETSc.Mat.Type.AIJ)
        # Preallocation must come BEFORE setUp for efficiency
        mat.setPreallocation(nnz=nnz)
        mat.setUp()
        return mat

    @staticmethod
    def parallel_matvec(mat: PETSc.Mat, vec: PETSc.Vec) -> PETSc.Vec:
        """
        Parallel matrix-vector product.

        Args:
            mat: Distributed matrix
            vec: Input vector

        Returns:
            Result vector
        """
        result = vec.duplicate()
        mat.mult(vec, result)
        return result


class ParallelIO:
    """
    Parallel I/O utilities for checkpointing and visualization.
    """

    def __init__(self, comm: MPI.Comm = None):
        """
        Initialize parallel I/O.

        Args:
            comm: MPI communicator
        """
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def save_vec(self, vec: PETSc.Vec, filename: str):
        """
        Save distributed vector to file.

        Uses PETSc binary format.

        Args:
            vec: Vector to save
            filename: Output filename
        """
        viewer = PETSc.Viewer().createBinary(filename, "w", comm=self.comm)
        vec.view(viewer)
        viewer.destroy()

    def load_vec(self, filename: str) -> PETSc.Vec:
        """
        Load distributed vector from file.

        Args:
            filename: Input filename

        Returns:
            Loaded vector
        """
        viewer = PETSc.Viewer().createBinary(filename, "r", comm=self.comm)
        vec = PETSc.Vec().create(comm=self.comm)
        vec.load(viewer)
        viewer.destroy()
        return vec

    def save_mat(self, mat: PETSc.Mat, filename: str):
        """
        Save distributed matrix to file.

        Args:
            mat: Matrix to save
            filename: Output filename
        """
        viewer = PETSc.Viewer().createBinary(filename, "w", comm=self.comm)
        mat.view(viewer)
        viewer.destroy()

    def load_mat(self, filename: str) -> PETSc.Mat:
        """
        Load distributed matrix from file.

        Args:
            filename: Input filename

        Returns:
            Loaded matrix
        """
        viewer = PETSc.Viewer().createBinary(filename, "r", comm=self.comm)
        mat = PETSc.Mat().create(comm=self.comm)
        mat.load(viewer)
        viewer.destroy()
        return mat


class LoadBalancer:
    """
    Load balancing utilities for distributed computations.
    """

    @staticmethod
    def distribute_observations(num_obs: int, num_ranks: int) -> List[Tuple[int, int]]:
        """
        Distribute observations across ranks.

        Args:
            num_obs: Total number of observations
            num_ranks: Number of MPI ranks

        Returns:
            List of (start, end) indices for each rank
        """
        obs_per_rank = num_obs // num_ranks
        remainder = num_obs % num_ranks

        ranges = []
        start = 0
        for rank in range(num_ranks):
            # Give extra observation to first 'remainder' ranks
            end = start + obs_per_rank + (1 if rank < remainder else 0)
            ranges.append((start, end))
            start = end

        return ranges

    @staticmethod
    def distribute_time_steps(num_steps: int, num_ranks: int) -> List[List[int]]:
        """
        Distribute time steps for parallel-in-time methods.

        Args:
            num_steps: Total time steps
            num_ranks: Number of ranks

        Returns:
            List of time step indices for each rank
        """
        steps_per_rank = num_steps // num_ranks
        remainder = num_steps % num_ranks

        assignments = []
        step = 0
        for rank in range(num_ranks):
            num_local = steps_per_rank + (1 if rank < remainder else 0)
            local_steps = list(range(step, step + num_local))
            assignments.append(local_steps)
            step += num_local

        return assignments


class ParallelTimer:
    """
    Parallel timer for performance profiling.
    """

    def __init__(self, comm: MPI.Comm = None):
        """
        Initialize parallel timer.

        Args:
            comm: MPI communicator
        """
        self.comm = comm or MPI.COMM_WORLD
        self.timers = {}

    def start(self, name: str):
        """
        Start timer for named region.

        Args:
            name: Timer name
        """
        self.comm.Barrier()
        if name not in self.timers:
            self.timers[name] = {"start": None, "total": 0.0, "count": 0}
        self.timers[name]["start"] = MPI.Wtime()

    def stop(self, name: str):
        """
        Stop timer for named region.

        Args:
            name: Timer name
        """
        self.comm.Barrier()
        elapsed = MPI.Wtime() - self.timers[name]["start"]
        self.timers[name]["total"] += elapsed
        self.timers[name]["count"] += 1

    def report(self, root: int = 0):
        """
        Report timing statistics.

        Args:
            root: Rank to print report
        """
        rank = self.comm.Get_rank()

        # Gather timings to root
        all_timers = self.comm.gather(self.timers, root=root)

        if rank == root:
            print("\n=== Parallel Timing Report ===")
            # Collect all timer names across all ranks to handle inconsistent timers
            all_timer_names = set()
            for t in all_timers:
                all_timer_names.update(t.keys())

            for name in all_timer_names:
                # Safely get times/counts, defaulting to 0 if timer doesn't exist on a rank
                times = [t.get(name, {}).get("total", 0.0) for t in all_timers]
                counts = [t.get(name, {}).get("count", 0) for t in all_timers]

                min_time = min(times)
                max_time = max(times)
                avg_time = sum(times) / len(times) if times else 0.0
                avg_count = sum(counts) / len(counts) if counts else 0.0

                print(f"{name}:")
                print(f"  Min: {min_time:.4f}s")
                print(f"  Max: {max_time:.4f}s")
                print(f"  Avg: {avg_time:.4f}s")
                print(f"  Count: {avg_count:.1f}")
                if avg_time > 0:
                    print(f"  Imbalance: {(max_time - min_time) / avg_time * 100:.1f}%")
                else:
                    print(f"  Imbalance: N/A")

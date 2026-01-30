"""
Non-blocking communication utilities for overlapping computation and communication.

Provides tools to use non-blocking MPI operations to improve parallel performance
by hiding communication latency behind computation.
"""

from typing import List, Optional, Callable, Any, Dict, Tuple
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.fem import Function


class NonBlockingScatter:
    """
    Non-blocking scatter operations for Function objects.

    Enables overlap of ghost cell updates with independent computation.
    """

    def __init__(self):
        """Initialize non-blocking scatter manager."""
        self.active_requests: List[MPI.Request] = []

    def scatter_forward_begin(self, u: Function) -> MPI.Request:
        """
        Begin non-blocking scatter forward operation.

        Note: DOLFINx/PETSc may not fully support non-blocking scatter yet.
        This is a placeholder for future implementation.

        Args:
            u: Function to scatter

        Returns:
            MPI request handle (placeholder)
        """
        # Start the scatter (blocking for now, but structured for future non-blocking)
        u.x.scatter_forward()

        # Return a dummy request (would be actual request in future)
        request = MPI.REQUEST_NULL
        self.active_requests.append(request)

        return request

    def scatter_forward_end(self, request: MPI.Request):
        """
        Complete non-blocking scatter forward operation.

        Args:
            request: Request handle from scatter_forward_begin
        """
        if request != MPI.REQUEST_NULL:
            request.Wait()

        if request in self.active_requests:
            self.active_requests.remove(request)

    def scatter_reverse_begin(self, u: Function, mode: PETSc.ScatterMode) -> MPI.Request:
        """
        Begin non-blocking scatter reverse operation.

        Args:
            u: Function to scatter
            mode: Scatter mode (ADD or INSERT)

        Returns:
            MPI request handle (placeholder)
        """
        # Start the scatter (blocking for now)
        u.x.scatter_reverse(mode)

        # Return a dummy request
        request = MPI.REQUEST_NULL
        self.active_requests.append(request)

        return request

    def scatter_reverse_end(self, request: MPI.Request):
        """
        Complete non-blocking scatter reverse operation.

        Args:
            request: Request handle from scatter_reverse_begin
        """
        if request != MPI.REQUEST_NULL:
            request.Wait()

        if request in self.active_requests:
            self.active_requests.remove(request)

    def wait_all(self):
        """Wait for all active non-blocking operations to complete."""
        for request in self.active_requests:
            if request != MPI.REQUEST_NULL:
                request.Wait()

        self.active_requests.clear()


class AsyncVectorOps:
    """
    Asynchronous vector operations with non-blocking communication.

    Provides non-blocking alternatives to common vector operations.
    """

    def __init__(self, comm: MPI.Comm = None):
        """
        Initialize async vector operations.

        Args:
            comm: MPI communicator
        """
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        # Track active operations
        self.active_ops: Dict[str, Any] = {}

    def allreduce_begin(
        self,
        send_data: np.ndarray,
        op: MPI.Op = MPI.SUM,
        tag: str = "default"
    ) -> Tuple[np.ndarray, MPI.Request]:
        """
        Begin non-blocking allreduce operation.

        Args:
            send_data: Local data to reduce
            op: MPI reduction operation
            tag: Tag to identify this operation

        Returns:
            Tuple of (recv_buffer, request)
        """
        recv_data = np.empty_like(send_data)
        request = self.comm.Iallreduce(send_data, recv_data, op=op)

        self.active_ops[tag] = {
            'type': 'allreduce',
            'recv_data': recv_data,
            'request': request
        }

        return recv_data, request

    def allreduce_end(self, tag: str = "default") -> np.ndarray:
        """
        Complete non-blocking allreduce operation.

        Args:
            tag: Tag identifying the operation

        Returns:
            Result of the reduction
        """
        if tag not in self.active_ops:
            raise ValueError(f"No active operation with tag '{tag}'")

        op_data = self.active_ops[tag]
        op_data['request'].Wait()

        result = op_data['recv_data']
        del self.active_ops[tag]

        return result

    def broadcast_begin(
        self,
        data: np.ndarray,
        root: int = 0,
        tag: str = "default"
    ) -> MPI.Request:
        """
        Begin non-blocking broadcast operation.

        Args:
            data: Data to broadcast (meaningful on root, buffer elsewhere)
            root: Root rank
            tag: Tag to identify this operation

        Returns:
            MPI request handle
        """
        request = self.comm.Ibcast(data, root=root)

        self.active_ops[tag] = {
            'type': 'broadcast',
            'data': data,
            'request': request
        }

        return request

    def broadcast_end(self, tag: str = "default") -> np.ndarray:
        """
        Complete non-blocking broadcast operation.

        Args:
            tag: Tag identifying the operation

        Returns:
            Broadcasted data
        """
        if tag not in self.active_ops:
            raise ValueError(f"No active operation with tag '{tag}'")

        op_data = self.active_ops[tag]
        op_data['request'].Wait()

        result = op_data['data']
        del self.active_ops[tag]

        return result

    def gather_begin(
        self,
        send_data: np.ndarray,
        root: int = 0,
        tag: str = "default"
    ) -> Tuple[Optional[np.ndarray], MPI.Request]:
        """
        Begin non-blocking gather operation.

        Args:
            send_data: Local data to gather
            root: Root rank
            tag: Tag to identify this operation

        Returns:
            Tuple of (recv_buffer on root, request)
        """
        if self.rank == root:
            recv_data = np.empty(
                (self.comm.Get_size(),) + send_data.shape,
                dtype=send_data.dtype
            )
        else:
            recv_data = None

        request = self.comm.Igather(send_data, recv_data, root=root)

        self.active_ops[tag] = {
            'type': 'gather',
            'recv_data': recv_data,
            'request': request
        }

        return recv_data, request

    def gather_end(self, tag: str = "default") -> Optional[np.ndarray]:
        """
        Complete non-blocking gather operation.

        Args:
            tag: Tag identifying the operation

        Returns:
            Gathered data on root, None elsewhere
        """
        if tag not in self.active_ops:
            raise ValueError(f"No active operation with tag '{tag}'")

        op_data = self.active_ops[tag]
        op_data['request'].Wait()

        result = op_data['recv_data']
        del self.active_ops[tag]

        return result

    def wait_all(self):
        """Wait for all active operations to complete."""
        for tag, op_data in list(self.active_ops.items()):
            op_data['request'].Wait()

        self.active_ops.clear()


class OverlapComputeComm:
    """
    Helper class for overlapping computation with communication.

    Provides patterns for common overlap scenarios in 4D-Var.
    """

    def __init__(self, comm: MPI.Comm = None):
        """
        Initialize overlap helper.

        Args:
            comm: MPI communicator
        """
        self.comm = comm or MPI.COMM_WORLD
        self.async_ops = AsyncVectorOps(comm)
        self.scatter_ops = NonBlockingScatter()

    def overlap_scatter_with_work(
        self,
        u: Function,
        independent_work: Callable[[], Any]
    ) -> Any:
        """
        Overlap ghost cell scatter with independent computation.

        Args:
            u: Function to scatter
            independent_work: Callable that performs work not requiring ghosts

        Returns:
            Result of independent_work

        Example:
            result = overlap.overlap_scatter_with_work(
                u,
                lambda: compute_something_local()
            )
        """
        # Start non-blocking scatter
        request = self.scatter_ops.scatter_forward_begin(u)

        # Do independent work while scatter is in progress
        result = independent_work()

        # Wait for scatter to complete
        self.scatter_ops.scatter_forward_end(request)

        return result

    def overlap_allreduce_with_work(
        self,
        local_value: float,
        independent_work: Callable[[], Any],
        op: MPI.Op = MPI.SUM
    ) -> Tuple[Any, float]:
        """
        Overlap global reduction with independent computation.

        Args:
            local_value: Local value to reduce
            independent_work: Callable that performs independent work
            op: MPI reduction operation

        Returns:
            Tuple of (work_result, reduced_value)

        Example:
            work_result, global_sum = overlap.overlap_allreduce_with_work(
                local_contribution,
                lambda: do_other_computation()
            )
        """
        # Start non-blocking allreduce
        send_data = np.array([local_value])
        recv_data, request = self.async_ops.allreduce_begin(send_data, op=op)

        # Do independent work
        work_result = independent_work()

        # Wait for allreduce
        reduced = self.async_ops.allreduce_end("default")

        return work_result, reduced[0]

    def pipeline_scatter_operations(
        self,
        functions: List[Function],
        operations: List[Callable[[Function], None]]
    ):
        """
        Pipeline multiple scatter operations with computation.

        Args:
            functions: List of functions to scatter
            operations: List of operations to perform on each function
                       (should not require ghost data from same function)

        Example:
            # Scatter u1, compute with u2, scatter u2, compute with u3, etc.
            pipeline_scatter_operations(
                [u1, u2, u3],
                [compute_op1, compute_op2, compute_op3]
            )
        """
        if len(functions) != len(operations):
            raise ValueError("Must have equal number of functions and operations")

        requests = []

        for i, (u, op) in enumerate(zip(functions, operations)):
            # Start scatter for current function
            request = self.scatter_ops.scatter_forward_begin(u)
            requests.append(request)

            # Perform operation on previous function (if any)
            # This overlaps communication for current with computation for previous
            if i > 0:
                self.scatter_ops.scatter_forward_end(requests[i-1])
                operations[i-1](functions[i-1])

        # Handle last function
        if functions:
            self.scatter_ops.scatter_forward_end(requests[-1])
            operations[-1](functions[-1])


class AsyncObservationOperator:
    """
    Asynchronous observation operator with overlapped communication.

    Demonstrates how to structure observation operators to overlap
    point evaluation with global gathering.
    """

    def __init__(self, comm: MPI.Comm = None):
        """
        Initialize async observation operator.

        Args:
            comm: MPI communicator
        """
        self.comm = comm or MPI.COMM_WORLD
        self.async_ops = AsyncVectorOps(comm)

    def apply_with_overlap(
        self,
        local_observations: np.ndarray,
        independent_work: Optional[Callable[[], Any]] = None
    ) -> np.ndarray:
        """
        Gather observations with optional overlapped work.

        Args:
            local_observations: Local observation values
            independent_work: Optional work to perform during gather

        Returns:
            Global observation vector
        """
        # Start non-blocking allgather
        global_obs = np.empty(
            self.comm.Get_size() * len(local_observations),
            dtype=local_observations.dtype
        )

        request = self.comm.Iallgather(local_observations, global_obs)

        # Do independent work if provided
        result = None
        if independent_work:
            result = independent_work()

        # Wait for gather to complete
        request.Wait()

        return global_obs


class BatchedCommunication:
    """
    Batches multiple small communications into fewer larger ones.

    Reduces latency overhead from many small messages.
    """

    def __init__(self, comm: MPI.Comm = None):
        """
        Initialize batched communication manager.

        Args:
            comm: MPI communicator
        """
        self.comm = comm or MPI.COMM_WORLD

        # Buffer for batching
        self.send_buffer: List[float] = []
        self.recv_buffer: List[float] = []

    def add_to_batch(self, value: float):
        """
        Add value to communication batch.

        Args:
            value: Value to add
        """
        self.send_buffer.append(value)

    def allreduce_batch(self, op: MPI.Op = MPI.SUM) -> np.ndarray:
        """
        Perform batched allreduce of all buffered values.

        Args:
            op: MPI reduction operation

        Returns:
            Array of reduced values
        """
        if not self.send_buffer:
            return np.array([])

        # Convert to numpy array
        send_data = np.array(self.send_buffer, dtype=np.float64)
        recv_data = np.empty_like(send_data)

        # Single allreduce for all values
        self.comm.Allreduce(send_data, recv_data, op=op)

        # Clear buffer
        self.send_buffer.clear()

        return recv_data

    def reset(self):
        """Clear communication buffers."""
        self.send_buffer.clear()
        self.recv_buffer.clear()


def demonstrate_overlap_pattern():
    """
    Demonstrate a typical overlap pattern for 4D-Var.

    Shows how to structure code to hide communication latency.
    """
    comm = MPI.COMM_WORLD
    overlap = OverlapComputeComm(comm)

    # Example: In cost function evaluation, overlap observation gather
    # with other computation

    # Pseudo-code pattern:
    # 1. Start gathering observations (non-blocking)
    # 2. Compute background term (independent work)
    # 3. Wait for observations and compute observation term

    # This is a template - actual implementation would use real Functions
    print(f"[Rank {comm.rank}] Overlap pattern template ready")


if __name__ == "__main__":
    # Demonstrate the overlap pattern when run directly
    demonstrate_overlap_pattern()

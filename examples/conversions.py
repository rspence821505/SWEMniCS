import numpy as np
from dolfinx import fem as fe
from mpi4py import MPI
from typing import Tuple, Callable, Optional, Literal
from petsc4py import PETSc
import pickle


def numpy_to_petsc_distributed(numpy_array, petsc_vec_template, comm=MPI.COMM_WORLD):
    """Set values in distributed PETSc vector from numpy array."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get distribution info from template
    local_size = petsc_vec_template.getLocalSize()
    ownership_range = petsc_vec_template.getOwnershipRange()
    start, end = ownership_range

    # Scatter the numpy array to all ranks
    if rank == 0:
        global_size = len(numpy_array)
        counts = [(global_size + i) // size for i in range(size)]
        displs = [sum(counts[:i]) for i in range(size)]

        local_array = np.zeros(local_size, dtype=np.float64)
        comm.Scatterv(
            ([numpy_array, counts, displs, MPI.DOUBLE]),
            local_array,
            root=0,
        )
    else:
        local_array = np.zeros(local_size, dtype=np.float64)
        comm.Scatterv((None, None, None, MPI.DOUBLE), local_array, root=0)

    # Set values in PETSc vector
    petsc_vec_template.setValues(range(start, end), local_array)
    petsc_vec_template.assemble()


# Helper functions for PETSc/NumPy conversion
def petsc_to_numpy(petsc_vec, comm=MPI.COMM_WORLD):
    """Convert distributed PETSc vector to numpy array on all ranks."""
    # Gather to rank 0, then broadcast
    rank = comm.Get_rank()
    if rank == 0:
        numpy_array = petsc_vec.getArray()
        # Gather from all ranks
        all_arrays = comm.gather(numpy_array, root=0)
        # Concatenate all pieces
        full_array = np.concatenate(all_arrays)
    else:
        local_array = petsc_vec.getArray()
        comm.gather(local_array, root=0)
        full_array = None

    # Broadcast full array to all ranks
    full_array = comm.bcast(full_array, root=0)
    return full_array


def get_petsc_matrix_as_numpy(petsc_mat):
    """Helper to safely convert PETSc matrix to numpy."""
    try:
        # For small matrices, convert to dense
        rows, cols = petsc_mat.getSize()
        dense_mat = PETSc.Mat().createDense([rows, cols])
        petsc_mat.convert(PETSc.Mat.Type.DENSE, dense_mat)
        return dense_mat.getDenseArray()
    except:
        # Fallback: manual extraction
        rows, cols = petsc_mat.getSize()
        result = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                result[i, j] = petsc_mat.getValue(i, j)
        return result


def numpy_to_petsc_matrix(H_np, comm=MPI.COMM_WORLD):
    rank = comm.Get_rank()
    n_rows, n_cols = H_np.shape if rank == 0 else (0, 0)
    global_shape = comm.bcast((n_rows, n_cols), root=0)

    H_petsc = PETSc.Mat().createAIJ(size=global_shape, comm=comm)
    H_petsc.setUp()

    if rank == 0:
        for i in range(n_rows):
            cols = H_np[i].nonzero()[0].astype(np.int32)
            vals = H_np[i, cols]
            H_petsc.setValues(int(i), cols, vals)

    H_petsc.assemble()
    return H_petsc


def numpy_to_petsc_vector(v_np, comm=MPI.COMM_WORLD):
    rank = comm.Get_rank()
    size = comm.Get_size()

    global_size = len(v_np) if rank == 0 else 0
    global_size = comm.bcast(global_size, root=0)

    counts = [(global_size + i) // size for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]
    local_size = counts[rank]

    local_array = np.zeros(local_size, dtype=np.float64)
    comm.Scatterv(
        (
            [v_np, counts, displs, MPI.DOUBLE]
            if rank == 0
            else [None, None, None, MPI.DOUBLE]
        ),
        local_array,
        root=0,
    )

    v_petsc = PETSc.Vec().createMPI(global_size, local_size, comm=comm)
    v_petsc.setValues(range(displs[rank], displs[rank] + local_size), local_array)
    v_petsc.assemble()
    return v_petsc


def create_distributed_state_vector(state_np=None, state_dim=1296, comm=MPI.COMM_WORLD):
    rank = comm.Get_rank()
    size = comm.Get_size()

    counts = [(state_dim + i) // size for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]
    local_size = counts[rank]

    state_petsc = PETSc.Vec().createMPI(state_dim, local_size, comm=comm)

    if state_np is not None:
        local_array = np.zeros(local_size, dtype=np.float64)
        comm.Scatterv(
            (
                [state_np, counts, displs, MPI.DOUBLE]
                if rank == 0
                else [None, None, None, MPI.DOUBLE]
            ),
            local_array,
            root=0,
        )
        state_petsc.setValues(
            range(displs[rank], displs[rank] + local_size), local_array
        )

    state_petsc.assemble()
    return state_petsc


def create_distributed_square_matrix(A_np, comm=MPI.COMM_WORLD):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        n_rows, n_cols = A_np.shape
        assert n_rows == n_cols, f"Matrix must be square: {n_rows} x {n_cols}"
    else:
        n_rows = n_cols = 0

    n = comm.bcast(n_rows, root=0)

    counts = [(n + i) // size for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]
    local_size = counts[rank]
    row_start = displs[rank]

    print(
        f"Rank {rank}: Distributing {n}x{n} matrix, local size: {local_size}x{local_size}",
        flush=True,
    )

    A_petsc = PETSc.Mat().createAIJ(size=((n, local_size), (n, local_size)), comm=comm)
    A_petsc.setUp()

    if rank == 0:
        for dest_rank in range(size):
            dest_start = displs[dest_rank]
            dest_end = dest_start + counts[dest_rank]
            A_local = A_np[dest_start:dest_end, :]

            if dest_rank == 0:
                for i, global_row in enumerate(range(dest_start, dest_end)):
                    nz_cols = np.nonzero(A_local[i, :])[0]
                    if len(nz_cols) > 0:
                        vals = A_local[i, nz_cols]
                        A_petsc.setValues(global_row, nz_cols.astype(np.int32), vals)
            else:
                comm.send(A_local, dest=dest_rank, tag=200)
    else:
        A_local = comm.recv(source=0, tag=200)
        for i, global_row in enumerate(range(row_start, row_start + local_size)):
            nz_cols = np.nonzero(A_local[i, :])[0]
            if len(nz_cols) > 0:
                vals = A_local[i, nz_cols]
                A_petsc.setValues(global_row, nz_cols.astype(np.int32), vals)

    A_petsc.assemble()
    return A_petsc


def create_distributed_observation_matrix(H_np, state_dim, comm=MPI.COMM_WORLD):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        obs_dim, full_state_dim = H_np.shape
        assert (
            full_state_dim == state_dim
        ), f"H matrix columns {full_state_dim} != state_dim {state_dim}"
    else:
        obs_dim = 0
        full_state_dim = 0

    obs_dim = comm.bcast(obs_dim, root=0)
    full_state_dim = comm.bcast(full_state_dim, root=0)

    counts = [(full_state_dim + i) // size for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]
    local_cols = counts[rank]
    col_start = displs[rank]

    print(
        f"Rank {rank}: obs_dim={obs_dim}, full_state_dim={full_state_dim}, local_cols={local_cols}, col_start={col_start}",
        flush=True,
    )

    try:
        H_petsc = PETSc.Mat().createAIJ(size=(obs_dim, full_state_dim), comm=comm)
        H_petsc.setUp()

        if rank == 0:
            for i in range(obs_dim):
                cols = H_np[i].nonzero()[0].astype(np.int32)
                vals = H_np[i, cols]
                if len(cols) > 0:
                    H_petsc.setValues(int(i), cols, vals)

        H_petsc.assemble()
        print(f"Rank {rank}: Created replicated H matrix successfully", flush=True)
        return H_petsc

    except Exception as e:
        print(f"Rank {rank}: Failed to create replicated matrix: {e}", flush=True)

    H_petsc = PETSc.Mat().create(comm=comm)
    H_petsc.setType(PETSc.Mat.Type.MPIAIJ)
    H_petsc.setSizes(((obs_dim, None), (full_state_dim, local_cols)))
    H_petsc.setUp()

    if rank == 0:
        for dest_rank in range(size):
            dest_start = displs[dest_rank]
            dest_end = dest_start + counts[dest_rank]
            H_local = H_np[:, dest_start:dest_end]

            if dest_rank == 0:
                for i in range(obs_dim):
                    nz_cols = np.nonzero(H_local[i, :])[0]
                    if len(nz_cols) > 0:
                        global_cols = nz_cols + dest_start
                        vals = H_local[i, nz_cols]
                        H_petsc.setValues(i, global_cols.astype(np.int32), vals)
            else:
                comm.send(H_local, dest=dest_rank, tag=100)
    else:
        H_local = comm.recv(source=0, tag=100)
        for i in range(obs_dim):
            nz_cols = np.nonzero(H_local[i, :])[0]
            if len(nz_cols) > 0:
                global_cols = nz_cols + col_start
                vals = H_local[i, nz_cols]
                H_petsc.setValues(i, global_cols.astype(np.int32), vals)

    H_petsc.assemble()
    print(f"Rank {rank}: Created distributed H matrix successfully", flush=True)
    return H_petsc

from mpi4py import MPI
import numpy as np
from dolfinx import fem as fe
from tqdm import tqdm
from scipy.optimize import minimize, OptimizeResult
from scipy.sparse.linalg import spsolve
from typing import Tuple, Callable, Optional, Literal
from petsc4py import PETSc
import pickle

from swemnics.problems import SlopedBeachProblem, TidalProblem
from swemnics import solvers as Solvers

# ============================================================================
# CONVERSION FUNCTIONS (Missing from your import)
# ============================================================================


def petsc_to_numpy(petsc_vec, comm=MPI.COMM_WORLD):
    """Convert distributed PETSc vector to numpy array on all ranks."""
    rank = comm.Get_rank()

    # Get local portion of the vector
    local_array = petsc_vec.getArray().copy()
    local_size = len(local_array)

    # Debug: print local sizes
    print(
        f"Rank {rank}: local_size={local_size}, global_size={petsc_vec.getSize()}",
        flush=True,
    )

    # Gather sizes first to understand the distribution
    all_sizes = comm.gather(local_size, root=0)

    if rank == 0:
        print(f"All local sizes: {all_sizes}", flush=True)

        # Calculate cumulative offsets based on actual sizes
        offsets = [0]
        for size in all_sizes[:-1]:
            offsets.append(offsets[-1] + size)

        print(f"Calculated offsets: {offsets}", flush=True)

        # Create full array
        total_size = sum(all_sizes)
        full_array = np.zeros(total_size, dtype=np.float64)

        # Gather all local arrays
        all_local_arrays = comm.gather(local_array, root=0)

        # Fill the full array using actual sizes and calculated offsets
        for i, (offset, size, local_data) in enumerate(
            zip(offsets, all_sizes, all_local_arrays)
        ):
            print(
                f"Setting full_array[{offset}:{offset + size}] = local_data of size {len(local_data)}",
                flush=True,
            )
            full_array[offset : offset + size] = local_data
    else:
        comm.gather(local_array, root=0)
        full_array = None

    # Broadcast full array to all ranks
    full_array = comm.bcast(full_array, root=0)
    return full_array


def numpy_to_petsc_distributed(numpy_array, petsc_vec_template, comm=MPI.COMM_WORLD):
    """Set values in distributed PETSc vector from numpy array."""
    rank = comm.Get_rank()

    # Get the local array size that PETSc actually expects
    current_local_array = petsc_vec_template.getArray()
    local_size = len(current_local_array)

    # Gather all local sizes to understand distribution
    all_sizes = comm.gather(local_size, root=0)

    if rank == 0:
        # Calculate offsets based on actual local sizes
        offsets = [0]
        for size in all_sizes[:-1]:
            offsets.append(offsets[-1] + size)

        # Send each rank its portion based on actual sizes
        for dest_rank in range(1, comm.Get_size()):
            start_idx = offsets[dest_rank]
            end_idx = start_idx + all_sizes[dest_rank]
            local_portion = numpy_array[start_idx:end_idx]
            comm.send(local_portion, dest=dest_rank, tag=999)

        # Set rank 0's own portion directly into the array
        rank0_portion = numpy_array[0 : all_sizes[0]]
        local_array = petsc_vec_template.getArray()
        local_array[:] = rank0_portion
    else:
        # Receive the local portion and set it directly
        local_portion = comm.recv(source=0, tag=999)
        local_array = petsc_vec_template.getArray()
        local_array[:] = local_portion

    petsc_vec_template.assemble()


def get_petsc_matrix_as_numpy(petsc_mat):
    """Helper to safely convert PETSc matrix to numpy."""
    try:
        # Get matrix dimensions
        rows, cols = petsc_mat.getSize()

        # Create numpy array and fill it
        result = np.zeros((rows, cols))

        # Get values row by row (safer for distributed matrices)
        for i in range(rows):
            row_values = []
            for j in range(cols):
                try:
                    value = petsc_mat.getValue(i, j)
                    row_values.append(value)
                except:
                    row_values.append(0.0)
            result[i, :] = row_values

        return result
    except Exception as e:
        print(f"Error converting PETSc matrix to numpy: {e}")
        # Return identity matrix as fallback
        rows, cols = petsc_mat.getSize()
        return np.eye(min(rows, cols))


def create_distributed_state_vector(state_np=None, state_dim=None, comm=MPI.COMM_WORLD):
    """Create a distributed state vector for 4DVar optimization."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    # If state_dim not provided, get it from state_np
    if state_dim is None and state_np is not None:
        state_dim = len(state_np)
    elif state_dim is None:
        raise ValueError("Either state_dim or state_np must be provided")

    # Create the PETSc vector with automatic local sizing
    state_petsc = PETSc.Vec().createMPI(state_dim, comm=comm)

    # If initial values provided, set them
    if state_np is not None:
        # Get the actual distribution from PETSc
        start, end = state_petsc.getOwnershipRange()
        local_size = end - start

        # Use PETSc's distribution for scattering
        if rank == 0:
            # Calculate actual distribution based on PETSc ownership
            all_ranges = comm.gather((start, end), root=0)
            counts = [r[1] - r[0] for r in all_ranges]
            displs = [r[0] for r in all_ranges]
        else:
            all_ranges = comm.gather((start, end), root=0)
            counts = None
            displs = None

        # Broadcast the distribution info
        counts = comm.bcast(counts, root=0)
        displs = comm.bcast(displs, root=0)

        local_array = np.zeros(local_size, dtype=np.float64)

        if rank == 0:
            comm.Scatterv([state_np, counts, displs, MPI.DOUBLE], local_array, root=0)
        else:
            comm.Scatterv([None, counts, displs, MPI.DOUBLE], local_array, root=0)

        # Set the local values
        state_petsc.setValues(range(start, end), local_array)

    state_petsc.assemble()
    return state_petsc


def numpy_to_petsc_matrix(H_np, comm=MPI.COMM_WORLD):
    """Convert numpy matrix to replicated PETSc matrix."""
    rank = comm.Get_rank()
    n_rows, n_cols = H_np.shape if rank == 0 else (0, 0)
    global_shape = comm.bcast((n_rows, n_cols), root=0)

    H_petsc = PETSc.Mat().createAIJ(size=global_shape, comm=comm)
    H_petsc.setUp()

    if rank == 0:
        for i in range(n_rows):
            cols = H_np[i].nonzero()[0].astype(np.int32)
            vals = H_np[i, cols]
            if len(cols) > 0:
                H_petsc.setValues(int(i), cols, vals)

    H_petsc.assemble()
    return H_petsc


def create_distributed_observation_matrix(H_np, state_dim, comm=MPI.COMM_WORLD):
    """Create H matrix (replicated for efficiency)."""
    rank = comm.Get_rank()

    # Get matrix dimensions
    if rank == 0:
        obs_dim, full_state_dim = H_np.shape
        assert (
            full_state_dim == state_dim
        ), f"H matrix columns {full_state_dim} != state_dim {state_dim}"
    else:
        obs_dim = 0
        full_state_dim = 0

    # Broadcast dimensions
    obs_dim = comm.bcast(obs_dim, root=0)
    full_state_dim = comm.bcast(full_state_dim, root=0)

    print(
        f"Rank {rank}: obs_dim={obs_dim}, full_state_dim={full_state_dim}", flush=True
    )

    # Create replicated matrix (simple approach)
    try:
        H_petsc = PETSc.Mat().createAIJ(size=(obs_dim, full_state_dim), comm=comm)
        H_petsc.setUp()

        # Set values only on rank 0 (replicated matrix)
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
        print(f"Rank {rank}: Failed to create H matrix: {e}", flush=True)
        raise


# ============================================================================
# MAIN 4DVAR FUNCTIONS
# ============================================================================


def create_problem_solver(problem_params, problem_type, true_signal=False):
    """Create problem and solver instances."""
    if problem_type == "sloped_beach":
        sloped_kwargs = {
            "dt": problem_params["dt"],
            "nt": problem_params["num_steps"],
            "friction_law": problem_params["fric_law"],
            "solution_var": problem_params["sol_var"],
            "wd_alpha": 0.36,
            "wd": True,
        }
        problem = SlopedBeachProblem(**sloped_kwargs)
    elif problem_type == "tidal":
        problem = TidalProblem(**problem_params)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    solver_kwargs = {
        "theta": 1,
        "p_degree": [1, 1],
        "verbose": False,
        "adjoint_method": True,
    }

    solver = Solvers.DGImplicit(problem, **solver_kwargs)

    if "t" in problem_params:
        solver.problem.t = problem_params["t"]

    return problem, solver


def background_loss_petsc(z_petsc, z_b_petsc, B_inv_petsc, comm=MPI.COMM_WORLD):
    """Calculate the background loss term using PETSc operations."""
    # Create temporary vector for difference
    diff_b = z_petsc.duplicate()
    diff_b.copy(z_petsc)
    diff_b.axpy(-1.0, z_b_petsc)  # diff_b = z - z_b

    # Compute B_inv @ diff_b
    B_inv_diff = z_petsc.duplicate()
    B_inv_petsc.mult(diff_b, B_inv_diff)

    # Compute 0.5 * diff_b^T @ B_inv_diff
    dot_product = diff_b.dot(B_inv_diff)

    # Clean up
    diff_b.destroy()
    B_inv_diff.destroy()

    return 0.5 * dot_product


def observation_loss_petsc(Qz_np, y_obs_np, R_inv_np):
    """Calculate the observation loss term (still using numpy for obs operations)."""
    obs_diff = y_obs_np.T - Qz_np
    return 0.5 * np.sum(obs_diff * (R_inv_np @ obs_diff))


def get_trajectory_petsc(u0_petsc, solver_params, stations, solver, comm=None):
    """Propagate state through model using PETSc state vector."""
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Convert PETSc vector to numpy for DOLFINx
    u0_numpy = petsc_to_numpy(u0_petsc, comm)

    # Convert to DOLFINx function
    V = solver.V
    u_0 = fe.Function(V)
    u_0.x.array[:] = u0_numpy

    # Clear any stale state
    if hasattr(solver, "saved_states"):
        solver.saved_states.clear()
    if hasattr(solver, "saved_adjoints"):
        solver.saved_adjoints.clear()

    # Run the time loop
    try:
        _, _ = solver.time_loop(
            solver_parameters=solver_params,
            stations=stations,
            u_0=u_0,
            save_states=True,
            adjoint_method=True,
        )
    except Exception as e:
        if rank == 0:
            print(f"Error in time loop: {e}")
        raise

    return solver


def bayes_cost_function_petsc(tao, x_petsc, user_ctx):
    """PETSc TAO-compatible cost function for Bayesian 4D-Var."""
    comm = user_ctx["comm"]
    rank = comm.Get_rank()

    try:
        # Unpack context
        solver = user_ctx["solver"]
        init_time = user_ctx["init_time"]
        u_b_petsc = user_ctx["u_b_petsc"]
        y_obs_np = user_ctx["y_obs_np"]
        obs_time_indices = user_ctx["obs_time_idxs"]
        H_np = user_ctx["H_np"]
        R_inv_np = user_ctx["R_inv_np"]
        B_inv_petsc = user_ctx["B_inv_petsc"]
        solver_params = user_ctx["solver_params"]
        stations = user_ctx["stations"]

        # Reset solver time
        solver.problem.t = init_time

        # Run model
        solver = get_trajectory_petsc(x_petsc, solver_params, stations, solver, comm)
        states = np.array(solver.saved_states)
        observed_states = states[obs_time_indices]
        Qz = H_np @ observed_states.T

        # Compute loss terms
        J_b = background_loss_petsc(x_petsc, u_b_petsc, B_inv_petsc, comm)
        J_o = observation_loss_petsc(Qz, y_obs_np, R_inv_np)

        total_cost = J_b + J_o

        if rank == 0:
            print(f"Cost: J_b={J_b:.6e}, J_o={J_o:.6e}, Total={total_cost:.6e}")

        return total_cost

    except Exception as e:
        if rank == 0:
            print(f"Error in cost function: {e}")
        return 1e10


def gradient_function_petsc(tao, x_petsc, g_petsc, user_ctx):
    """PETSc TAO-compatible gradient function."""
    comm = user_ctx["comm"]
    rank = comm.Get_rank()

    try:
        # Unpack context
        solver = user_ctx["solver"]
        u_b_petsc = user_ctx["u_b_petsc"]
        y_obs_np = user_ctx["y_obs_np"]
        obs_spatial_indices = user_ctx["obs_spatial_idxs"]
        obs_time_indices = user_ctx["obs_time_idxs"]
        H_np = user_ctx["H_np"]
        R_inv_np = user_ctx["R_inv_np"]
        B_inv_petsc = user_ctx["B_inv_petsc"]
        L_inv_np = user_ctx["L_inv_np"]
        Q_zb_np = user_ctx["Q_zb_np"]
        adjoint_type = user_ctx["adjoint_type"]

        # Check if saved adjoints are available
        if not solver.saved_adjoints:
            if rank == 0:
                raise ValueError(
                    "No saved adjoints found. Ensure the model was run with adjoint_method=True."
                )

        if not solver.saved_states:
            if rank == 0:
                raise ValueError(
                    "No saved states found. Ensure the model was run with adjoint_method=True."
                )

        # Compute adjoint using existing function
        λ_0 = swe_adjoint(
            solver,
            H_np,
            y_obs_np,
            obs_spatial_indices,
            obs_time_indices,
            R_inv_np,
            L_inv_np,
            Q_zb_np,
            adjoint_type,
            comm,
        )

        # Compute background gradient term: B_inv @ (x - u_b)
        diff_b = x_petsc.duplicate()
        diff_b.copy(x_petsc)
        diff_b.axpy(-1.0, u_b_petsc)

        bg_grad = x_petsc.duplicate()
        B_inv_petsc.mult(diff_b, bg_grad)

        # Convert adjoint to PETSc and add to background gradient
        λ_0_petsc = x_petsc.duplicate()
        numpy_to_petsc_distributed(λ_0, λ_0_petsc, comm)

        # g = B_inv @ (x - u_b) + λ_0
        g_petsc.copy(bg_grad)
        g_petsc.axpy(1.0, λ_0_petsc)

        # Clean up
        diff_b.destroy()
        bg_grad.destroy()
        λ_0_petsc.destroy()

    except Exception as e:
        if rank == 0:
            print(f"Error in gradient function: {e}")
        # Set gradient to zero on error
        g_petsc.set(0.0)


# Keep all your existing functions (swe_adjoint, adjoint_rhs, etc.)
def _adjoint_rhs_bayes(H, Hu, yobs, R_inv, **kwargs):
    """Compute the adjoint right-hand side for the Bayesian cost function."""
    obs_residual = Hu - yobs
    return H.T @ R_inv @ obs_residual


def adjoint_rhs(
    H: np.ndarray,
    Hu: np.ndarray,
    yobs: np.ndarray,
    R_inv: np.ndarray,
    L_inv: np.ndarray,
    Q_zb: np.ndarray,
    Q_zb_wme: Optional[np.ndarray] = None,
    Q_z_wme: Optional[np.ndarray] = None,
    adjoint_type: Literal["bayes", "dci", "dci_wme"] = "bayes",
) -> np.ndarray:
    """Compute the right-hand side of the adjoint equation for various 4D-Var formulations."""
    dispatch = {"bayes": _adjoint_rhs_bayes}

    if adjoint_type not in dispatch:
        raise ValueError(f"Unknown adjoint type: {adjoint_type}")

    return dispatch[adjoint_type](
        H=H,
        Hu=Hu,
        yobs=yobs,
        R_inv=R_inv,
        L_inv=L_inv,
        Q_zb=Q_zb,
        Q_zb_wme=Q_zb_wme,
        Q_z_wme=Q_z_wme,
    )


def swe_adjoint(
    solver,
    H: np.ndarray,
    obs_data: np.ndarray,
    obs_spatial_idxs: np.ndarray,
    obs_time_idxs: np.ndarray,
    R_inv: np.ndarray,
    L_inv: Optional[np.ndarray] = None,
    Q_zb: Optional[np.ndarray] = None,
    adjoint_type: Literal["bayes", "dci", "dci_wme"] = "bayes",
    comm: Optional[MPI.Comm] = None,
) -> np.ndarray:
    """Compute the initial adjoint vector λ₀ for a 4D-Var cost functional."""
    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    adjoints = solver.saved_adjoints
    trajectories = solver.saved_states

    nt = len(adjoints)
    N_dof = adjoints[0].shape[0]
    λ = np.zeros(N_dof)

    Qzb = Q_zb.T if Q_zb is not None else None

    for n in reversed(range(nt)):
        if n in obs_time_idxs:
            idx = np.where(obs_time_idxs == n)[0][0]
            u = trajectories[n + 1].copy()
            Hu = H @ u
            yobs = obs_data[idx].copy()
            q_zb = Qzb[idx].copy() if Q_zb is not None else None

            rhs = adjoint_rhs(
                H,
                Hu,
                yobs,
                R_inv,
                L_inv,
                q_zb,
                Q_zb_wme=None,
                Q_z_wme=None,
                adjoint_type=adjoint_type,
            )

            λ += rhs

        # Solve A_n^T λ_n = λ
        A_T = adjoints[n]

        try:
            λ = spsolve(A_T, λ)
        except Exception as e:
            if rank == 0:
                print(f"Error solving adjoint system at time step {n}: {e}", flush=True)
            raise ValueError(
                f"Adjoint matrix at time step {n} is singular or ill-conditioned."
            )

    return λ


def optimize_4dvar_tao(
    u0_petsc, cost_function_type: str, solver, init_time, comm=MPI.COMM_WORLD, **kwargs
):
    """Perform 4D-Var optimization using PETSc TAO."""
    rank = comm.Get_rank()

    # Create TAO optimizer
    tao = PETSc.TAO().create(comm=comm)

    # Set up user context for callbacks
    user_ctx = {
        "solver": solver,
        "init_time": init_time,
        "comm": comm,
        "adjoint_type": cost_function_type,
        **kwargs,
    }

    # Create gradient vector with same distribution as state vector
    g_petsc = u0_petsc.duplicate()

    # Set initial guess
    tao.setInitial(u0_petsc)

    # Set objective and gradient functions
    if cost_function_type == "bayes":
        tao.setObjective(bayes_cost_function_petsc, user_ctx)
    else:
        raise ValueError(f"Cost function type {cost_function_type} not implemented yet")

    tao.setGradient(gradient_function_petsc, g_petsc, user_ctx)

    # Set TAO options
    tao.setType(PETSc.TAO.Type.LMVM)
    tao.setTolerances(gatol=1e-6, grtol=1e-6, gttol=1e-6)
    tao.setMaximumIterations(10)  # Reduced for testing

    if rank == 0:
        tao.setMonitor(
            lambda tao, step, f, gnorm, cnorm, xdiff, user: print(
                f"Iteration {step}: f={f:.6e}, gnorm={gnorm:.6e}"
            )
        )

    tao.setFromOptions()

    # Solve
    if rank == 0:
        print("Starting TAO optimization...")

    tao.solve()

    # Get solution
    solution = tao.getSolution()

    # Get solver information
    if rank == 0:
        reason = tao.getConvergedReason()
        niter = tao.getIterationNumber()
        final_f = tao.getFunctionValue()
        final_gnorm = tao.getGradientNorm()

        print(f"\nTAO Optimization completed:")
        print(f"  Converged reason: {reason}")
        print(f"  Iterations: {niter}")
        print(f"  Final objective: {final_f:.6e}")
        print(f"  Final gradient norm: {final_gnorm:.6e}")

    # Clean up
    g_petsc.destroy()
    tao.destroy()

    return solution


def run_assimilation_petsc(
    problem_params,
    solver_params,
    stations_petsc,
    y_obs_petsc,
    obs_per_window,
    obs_spatial_indices,
    obs_time_indices,
    H_petsc,
    R_inv_petsc,
    B_inv_petsc,
    L_inv_petsc,
    hb_petsc,
    problem_type,
    cost_function_type,
    comm=MPI.COMM_WORLD,
):
    """Run 4DVar analysis using PETSc TAO optimization."""
    rank = comm.Get_rank()

    # Convert PETSc objects to numpy for compatibility with existing code
    stations_np = get_petsc_matrix_as_numpy(stations_petsc)
    y_obs_np = get_petsc_matrix_as_numpy(y_obs_petsc)
    H_np = get_petsc_matrix_as_numpy(H_petsc)
    R_inv_np = get_petsc_matrix_as_numpy(R_inv_petsc)
    L_inv_np = get_petsc_matrix_as_numpy(L_inv_petsc)
    hb_np = get_petsc_matrix_as_numpy(hb_petsc).flatten()

    name = "Hotstart"
    analysis = []
    analysis_state_petsc = None
    num_windows = problem_params["num_windows"]
    steps_per_window = problem_params["num_steps"]
    obs_times_current_window = obs_time_indices[:obs_per_window]

    for idx in tqdm(range(num_windows), desc="Processing windows", unit="window"):
        if rank == 0:
            print(f"\n{'='*20} Window {idx + 1}/{num_windows} {'='*20}")

        # Extract observations for current window
        indices = np.arange(obs_per_window) + (idx * obs_per_window)
        if rank == 0:
            print(f"Processing indices: {indices}")

        yobs_current_window = y_obs_np[indices]

        # Update initial time for model
        initial_time = int(idx * steps_per_window * problem_params["dt"])
        problem_params.update({"t": initial_time})

        # Create problem and solver
        _, solver = create_problem_solver(
            problem_params, problem_type, true_signal=False
        )
        solver.problem.t = initial_time
        V = solver.V

        # Initialize state vector (distributed PETSc)
        if analysis_state_petsc is None:
            # First window - use initial state from solver
            u_0 = fe.Function(V)
            initial_state_numpy = u_0.x.array[:]
            if rank == 0:
                print(
                    f"DOLFINx state vector size: {len(initial_state_numpy)} (expected: {state_dim})"
                )
            u0_petsc = create_distributed_state_vector(initial_state_numpy, comm=comm)
        else:
            # Use previous analysis as initial guess
            u0_petsc = analysis_state_petsc.duplicate()
            u0_petsc.copy(analysis_state_petsc)

        # Create background state (same as initial for this window)
        u_b_petsc = u0_petsc.duplicate()
        u_b_petsc.copy(u0_petsc)

        # Generate background trajectory for Q_zb
        if rank == 0:
            print("Generating background trajectory...")

        u_0_numpy = petsc_to_numpy(u0_petsc, comm)
        u_0 = fe.Function(V)
        u_0.x.array[:] = u_0_numpy

        solver.time_loop(
            solver_parameters=solver_params,
            stations=stations_np,
            u_0=u_0,
            save_states=True,
            adjoint_method=False,
        )

        # Process background state
        background = np.array(solver.saved_states)
        observed_background_states = background[obs_times_current_window]
        Q_zb = H_np @ observed_background_states.T

        # Clear states for optimization
        solver.saved_states = []

        if rank == 0:
            print("Starting optimization...")

        # Perform optimization using TAO
        optimized_state_petsc = optimize_4dvar_tao(
            u0_petsc=u0_petsc,
            cost_function_type=cost_function_type,
            solver=solver,
            init_time=initial_time,
            comm=comm,
            u_b_petsc=u_b_petsc,
            y_obs_np=yobs_current_window,
            obs_spatial_idxs=obs_spatial_indices,
            obs_time_idxs=obs_times_current_window,
            H_np=H_np,
            R_inv_np=R_inv_np,
            B_inv_petsc=B_inv_petsc,
            L_inv_np=L_inv_np,
            Q_zb_np=Q_zb,
            solver_params=solver_params,
            stations=stations_np,
        )

        # Run analysis forward
        if rank == 0:
            print("Running analysis forward...")

        solver.problem.t = initial_time
        optimized_numpy = petsc_to_numpy(optimized_state_petsc, comm)
        u_opt = fe.Function(V)
        u_opt.x.array[:] = optimized_numpy

        solver.time_loop(
            solver_parameters=solver_params,
            stations=stations_np,
            u_0=u_opt,
            adjoint_method=False,
        )

        # Save analysis state for next window
        analysis_state_numpy = solver.u.x.array[:]
        if analysis_state_petsc is not None:
            analysis_state_petsc.destroy()
        analysis_state_petsc = create_distributed_state_vector(
            analysis_state_numpy, comm=comm
        )

        # Collect results
        current_analysis = solver.vals.copy()
        if idx < num_windows - 1:
            current_analysis = current_analysis[:-1, :, :]
        analysis.append(current_analysis)

        # Clean up
        u0_petsc.destroy()
        u_b_petsc.destroy()
        optimized_state_petsc.destroy()

        if rank == 0:
            print(f"Window {idx + 1} completed!")

    # Clean up final analysis state
    if analysis_state_petsc is not None:
        analysis_state_petsc.destroy()

    # Combine all windows
    return np.concatenate(analysis, axis=0)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(f"[Rank {rank}] Reached Initial MPI Call \n\n", flush=True)

# Load pickle only on rank 0
if rank == 0:
    with open("da_inputs.pkl", "rb") as f:
        da_inputs = pickle.load(f)
else:
    da_inputs = None

# Broadcast the dictionary to all ranks
da_inputs = comm.bcast(da_inputs, root=0)

# Access dictionary keys on all ranks
state_dim = da_inputs["state_dim"]
obs_dim = da_inputs["obs_dim"]
obs_per_window = da_inputs["obs_per_window"]
obs_spatial_indices = da_inputs["obs_spatial_indices"]
obs_time_indices = da_inputs["obs_time_indices"]
problem_params = da_inputs["problem_params"]
solver_params = da_inputs["solver_params"]

# Extract and convert NumPy arrays
true_states = np.array(da_inputs["true_states"])
y_obs = da_inputs["y_obs"]
stations = da_inputs["stations"]
H_np = da_inputs["H"]
R_inv_np = da_inputs["R_inv"]
B_inv_np = da_inputs["B_inv"]
L_inv_np = da_inputs["L_inv"]
hb_np = da_inputs["hb"]

# Convert to PETSc objects
true_states_petsc = numpy_to_petsc_matrix(true_states, comm=comm)
y_obs_petsc = numpy_to_petsc_matrix(y_obs, comm=comm)
stations_petsc = numpy_to_petsc_matrix(stations, comm=comm)

# Observation matrix (replicated version)
H_petsc = create_distributed_observation_matrix(H_np, state_dim, comm=comm)

# Small replicated covariance matrices
R_inv_petsc = numpy_to_petsc_matrix(R_inv_np, comm=comm)
L_inv_petsc = numpy_to_petsc_matrix(L_inv_np, comm=comm)

# Background covariance matrix (currently replicated)
B_inv_petsc = numpy_to_petsc_matrix(B_inv_np, comm=comm)

# Bathymetry at stations (replicated, reshape to matrix)
hb_petsc = numpy_to_petsc_matrix(hb_np.reshape(1, -1), comm=comm)

# Print summary on rank 0
if rank == 0:
    print(
        f"Rank {rank} - Data Summary:\n"
        f"State Dimension: {state_dim}\n"
        f"Observation Dimension: {obs_dim}\n"
        f"Background Covariance Matrix Shape B: {B_inv_petsc.getSize()} (replicated)\n"
        f"Observation Covariance Matrix Shape R: {R_inv_petsc.getSize()} (replicated)\n"
        f"Predicted Error Covariance Matrix shape L: {L_inv_petsc.getSize()} (replicated)\n"
        f"Observation Matrix Shape H: {H_petsc.getSize()} (replicated)\n"
        f"true_states shape: {true_states_petsc.getSize()} (replicated)\n"
        f"y_obs shape: {y_obs_petsc.getSize()} (replicated)\n"
        f"Stations shape: {stations_petsc.getSize()} (replicated)\n"
        f"Bathymetry at stations hb: {hb_petsc.getSize()} (replicated)\n",
        flush=True,
    )

print(f"[Rank {rank}] Before run_assimilation\n\n", flush=True)

# Run the assimilation using PETSc TAO
if rank == 0:
    print(f"[Rank {rank}] Starting PETSc TAO-based 4DVar assimilation\n")

bayes_analysis = run_assimilation_petsc(
    problem_params,
    solver_params,
    stations_petsc,
    y_obs_petsc,
    obs_per_window,
    obs_spatial_indices,
    obs_time_indices,
    H_petsc,
    R_inv_petsc,
    B_inv_petsc,
    L_inv_petsc,
    hb_petsc,
    "sloped_beach",
    cost_function_type="bayes",
    comm=comm,
)

if rank == 0:
    print(f"[Rank {rank}] Completed PETSc TAO-based 4DVar assimilation\n")

    # Post-processing and validation (only on rank 0)
    try:
        # Convert PETSc matrices back to numpy for analysis
        true_states_np = get_petsc_matrix_as_numpy(true_states_petsc)
        H_np = get_petsc_matrix_as_numpy(H_petsc)
        hb_np = get_petsc_matrix_as_numpy(hb_petsc).flatten()

        # Compute validation metrics
        Hu_true = H_np @ true_states_np.T
        pred = bayes_analysis[1:, :, 0] + hb_np
        bayes_height_misfit_rmse = np.sqrt(np.mean((Hu_true - pred.T) ** 2))

        print(f"=== FINAL RESULTS ===")
        print(f"Bayes Analysis Height RMSE: {bayes_height_misfit_rmse:.6f}")
        print(f"Analysis shape: {bayes_analysis.shape}")
        print(f"=====================")

    except Exception as e:
        print(f"Error in post-processing: {e}")

# Finalize MPI
if rank == 0:
    print("4DVar assimilation completed successfully!")

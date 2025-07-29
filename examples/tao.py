from petsc4py import PETSc
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Global counters and timers
func_eval_counter = 0
grad_eval_counter = 0
grad_eval_time = 0.0
obj_eval_time = 0.0

# Global file handle
monitor_file = None
log_file = None


def extended_rosenbrock_objective(tao, x):
    global func_eval_counter, obj_eval_time
    func_eval_counter += 1
    t0 = time.perf_counter()

    x_arr = x.getArray(readonly=True)
    print(f"Rank {rank} evaluating objective at x = {type(x_arr)}")
    start, end = x.getOwnershipRange()
    f_local = 0.0

    for i in range(start, end - 1, 2):
        x1 = x_arr[i - start]
        x2 = x_arr[i + 1 - start]
        f_local += (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2

    f_total = comm.allreduce(f_local, op=MPI.SUM)
    obj_eval_time += time.perf_counter() - t0
    return f_total


def extended_rosenbrock_gradient(tao, x, g):
    global grad_eval_counter, grad_eval_time
    grad_eval_counter += 1
    t0 = time.perf_counter()

    x_arr = x.getArray(readonly=True)
    g_arr = g.getArray()
    start, end = x.getOwnershipRange()

    for i in range(end - start):
        g_arr[i] = 0.0

    for i in range(start, end - 1, 2):
        i0 = i - start
        i1 = i + 1 - start
        x1 = x_arr[i0]
        x2 = x_arr[i1]
        g_arr[i0] += -2 * (1 - x1) - 400 * x1 * (x2 - x1**2)
        g_arr[i1] += 200 * (x2 - x1**2)

    g.assemble()
    grad_eval_time += time.perf_counter() - t0


def monitor_callback(tao):
    sols = tao.getSolution().getArray(readonly=True)
    its = tao.getIterationNumber()
    obj = tao.getObjectiveValue()

    gradient = tao.getGradient()
    gvec = gradient[0] if isinstance(gradient, tuple) else gradient
    gnorm = gvec.norm()

    msg = (
        f"[Iteration {its:3d}] x = {sols},  f(x) = {obj:.6e},  ||grad|| = {gnorm:.3e}\n"
    )

    if monitor_file is not None:
        monitor_file.write(msg)
        monitor_file.flush()
    elif rank == 0:
        print(msg, end="")


def run_extended_rosenbrock(n=50000):
    global monitor_file, log_file

    if n % 2 != 0:
        raise ValueError("Extended Rosenbrock function requires even n.")

    if rank == 0:
        monitor_file = open("monitor_output.log", "w")
        log_file = open("diagnostics_output.log", "w")

    x = PETSc.Vec().create(comm=comm)
    x.setSizes((PETSc.DECIDE, n))
    x.setFromOptions()
    x.set(0.0)
    x.assemble()

    tao = PETSc.TAO().create(comm=comm)
    tao.setType("lmvm")
    tao.setObjective(extended_rosenbrock_objective)
    tao.setGradient(extended_rosenbrock_gradient)
    tao.setSolution(x)
    tao.setMonitor(monitor_callback)

    comm.barrier()
    t_start = time.perf_counter()
    tao.solve()
    comm.barrier()
    t_end = time.perf_counter()

    # Gather diagnostics
    total_func_evals = comm.reduce(func_eval_counter, op=MPI.SUM, root=0)
    total_grad_evals = comm.reduce(grad_eval_counter, op=MPI.SUM, root=0)
    total_grad_time = comm.reduce(grad_eval_time, op=MPI.SUM, root=0)
    total_obj_time = comm.reduce(obj_eval_time, op=MPI.SUM, root=0)

    local_total_time = t_end - t_start
    all_grad_times = comm.gather(grad_eval_time, root=0)
    all_obj_times = comm.gather(obj_eval_time, root=0)
    all_total_times = comm.gather(local_total_time, root=0)

    if rank == 0:
        # Load balance diagnostics
        def report_balance(label, times):
            avg = sum(times) / len(times)
            max_t = max(times)
            min_t = min(times)
            imbalance = 100.0 * (max_t - min_t) / avg if avg > 0 else 0.0
            return (
                f"{label} Load Balance:\n"
                f"  Min = {min_t:.6f} s, Max = {max_t:.6f} s, "
                f"Avg = {avg:.6f} s, Imbalance = {imbalance:.2f}%\n"
            )

        print(
            f"\n===== EXTENDED ROSENBROCK OPTIMIZATION Results (n = {n}) ===== \n",
            file=log_file,
        )
        print("PETSc TAO method     :", tao.getType(), file=log_file)
        print("MPI ranks            :", size, file=log_file)
        print("Optimal solution x*  :", x.getArray(), file=log_file)
        print("Optimal value f(x*)  :", tao.getObjectiveValue(), file=log_file)
        print("Iterations           :", tao.getIterationNumber(), file=log_file)
        print("Converged            :", tao.converged, file=log_file)
        print("Convergence reason   :", tao.getConvergedReason(), file=log_file)
        print("Total runtime (s)    :", sum(all_total_times) / size, file=log_file)
        print("Total function evals :", total_func_evals, file=log_file)
        print("Total gradient evals :", total_grad_evals, file=log_file)
        print("Total grad eval time :", total_grad_time, "seconds", file=log_file)
        print("Total obj  eval time :", total_obj_time, "seconds\n", file=log_file)
        print(f"\n===== LOAD BALANCE  Results (n = {n}) ===== \n", file=log_file)
        print(report_balance("Gradient", all_grad_times), file=log_file)
        print(report_balance("Objective", all_obj_times), file=log_file)
        print(report_balance("Total", all_total_times), file=log_file)

        monitor_file.close()
        log_file.close()

        print("Extended Rosenbrock optimization completed successfully.")


def main():
    run_extended_rosenbrock(n=10e3)  # Adjust n as needed for testing


if __name__ == "__main__":
    main()

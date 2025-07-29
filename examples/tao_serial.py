from petsc4py import PETSc
import time

# Global counters
func_eval_counter = 0
grad_eval_counter = 0
grad_eval_time = 0.0


def extended_rosenbrock_objective(tao, x):
    global func_eval_counter
    func_eval_counter += 1

    x_arr = x.getArray(readonly=True)
    f = 0.0
    for i in range(0, len(x_arr) - 1, 2):
        x1 = x_arr[i]
        x2 = x_arr[i + 1]
        f += (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2
    return f


def extended_rosenbrock_gradient(tao, x, g):
    global grad_eval_counter, grad_eval_time
    grad_eval_counter += 1
    t0 = time.perf_counter()

    x_arr = x.getArray(readonly=True)
    g_arr = g.getArray()

    for i in range(len(g_arr)):
        g_arr[i] = 0.0  # clear previous values

    for i in range(0, len(x_arr) - 1, 2):
        x1 = x_arr[i]
        x2 = x_arr[i + 1]
        g_arr[i] += -2 * (1 - x1) - 400 * x1 * (x2 - x1**2)
        g_arr[i + 1] += 200 * (x2 - x1**2)

    g.assemble()
    grad_eval_time += time.perf_counter() - t0


def run_serial_rosenbrock(n=50000):
    if n % 2 != 0:
        raise ValueError("Extended Rosenbrock function requires even n.")

    # Create a sequential vector
    x = PETSc.Vec().createSeq(n, comm=PETSc.COMM_SELF)
    x.set(0.0)
    x.assemble()

    tao = PETSc.TAO().create(comm=PETSc.COMM_SELF)
    tao.setType("lmvm")
    tao.setObjective(extended_rosenbrock_objective)
    tao.setGradient(extended_rosenbrock_gradient)
    tao.setSolution(x)

    t_start = time.time()
    tao.solve()
    t_end = time.time()

    print(f"\n===== SERIAL EXTENDED ROSENBROCK OPTIMIZATION (n = {n}) =====")
    print("PETSc TAO method     :", tao.getType())
    print("Optimal solution x* =", x.getArray())
    print("Optimal value f(x*)  :", tao.getObjectiveValue())
    print("Iterations           :", tao.getIterationNumber())
    print("Converged            :", tao.converged)
    print("Convergence reason   :", tao.getConvergedReason())
    print("Total runtime (s)    :", t_end - t_start)
    print("Total function evals :", func_eval_counter)
    print("Total gradient evals :", grad_eval_counter)
    print("Total grad eval time :", grad_eval_time, "seconds")


def main():
    run_serial_rosenbrock(n=50000)


if __name__ == "__main__":
    main()

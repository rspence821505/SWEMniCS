"""Repeated single-window eval lifecycle reproducer (memory-leak classifier).

Builds one DA window using the existing idealized-inlet setup, then calls
``cost_fn.value_gradient(m_background)`` ``N`` times in the same process at the
same control vector. Per-eval RSS + PETSc memory + (optional) malloc_trim
impact is logged via :mod:`swe4dvar.utils.eval_memory_diag`.

This is **not** a cycling script. It is a single-window, single-process,
repeated-identical-eval lifecycle probe. Use it to classify whether per-eval
memory growth is:

  * live PETSc-owned state (PETSc memory rises with RSS, never drops)
  * PETSc allocator pool retention (PETSc memory drops, RSS does not)
  * glibc allocator retention (RSS only drops after malloc_trim(0))

It re-uses ``run_single_method`` from ``experiments/idealized_inlet_da.py``,
gated by ``SWE4DVAR_REPEAT_EVALS=N`` which makes that function skip TAO and
do N repeated value_gradient calls instead.

Usage::

    SWE4DVAR_EVAL_MEM_DIAG=1 \\
    SWE4DVAR_EVAL_MEM_DIAG_CSV=results/eval_mem_repro_rank{rank}.csv \\
    SWE4DVAR_REPEAT_EVALS=15 \\
    SWE4DVAR_MALLOC_TRIM=1 \\
    mpirun -n 4 python experiments/repeated_eval_reproducer.py \\
        --vmax 10 --nt-ramp 24 --nt-da 6 \\
        --obs-fraction 0.05 --obs-frequency 5 \\
        --obs-noise-level 0.01 --background-error-std 0.02

This is a thin wrapper that defaults a few CLI flags and forwards everything
else to ``experiments/idealized_inlet_da.main``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    # Default to a small but realistic case if no extras given.
    # Users can override anything via the CLI.
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))

    # Heuristic default: at least 10 repeated evals.
    if "SWE4DVAR_REPEAT_EVALS" not in os.environ:
        os.environ["SWE4DVAR_REPEAT_EVALS"] = "12"
    # Memory diagnostic on by default — that is the whole point of this script.
    os.environ.setdefault("SWE4DVAR_EVAL_MEM_DIAG", "1")

    from experiments import idealized_inlet_da as _da
    from swe4dvar.utils import eval_memory_diag as _emd

    # Optional PETSc malloc-debug / memory-view (gated by env var)
    _emd.configure_petsc_malloc_logging()

    # Force --n-windows 1 unless caller already set it. We are testing a
    # single-window's repeated eval lifecycle, not multi-window cycling.
    if "--n-windows" not in sys.argv:
        sys.argv.extend(["--n-windows", "1"])

    return int(_da.main() or 0)


if __name__ == "__main__":
    sys.exit(main())

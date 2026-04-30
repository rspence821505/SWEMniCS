"""Eval-boundary memory diagnostics for cost-function lifecycle.

Records RSS, PETSc-reported memory, and (if enabled) malloc_trim impact at
key points around each cost-function evaluation, so we can distinguish:

  - PETSc memory rises with RSS, never drops    → live PETSc-owned state leak
  - PETSc memory drops, RSS does not            → PETSc allocator pool retention
  - RSS drops only after malloc_trim(0)         → glibc allocator retention

Gating:
  SWE4DVAR_EVAL_MEM_DIAG=1            → record at lifecycle points
  SWE4DVAR_EVAL_MEM_DIAG_CSV=PATH     → append CSV rows there (per-rank or rank-0 only)
  SWE4DVAR_EVAL_MEM_DIAG_RANK0=1      → only rank 0 writes (default: all ranks)
  SWE4DVAR_MALLOC_TRIM=1              → call libc malloc_trim(0) after cleanup
  SWE4DVAR_PETSC_MALLOC_DEBUG=1       → set -malloc_debug -memory_view at startup

CSV columns:
  wall_s, eval_id, rank, label, rss_mb, petsc_curr_mb, petsc_max_mb, trim_freed_mb
"""

from __future__ import annotations

import os
import time
from typing import Optional


_T0 = time.perf_counter()
_HEADER_WRITTEN: set = set()
_LIBC_HANDLE = None
_LIBC_HAS_TRIM: Optional[bool] = None


def is_enabled() -> bool:
    return os.environ.get("SWE4DVAR_EVAL_MEM_DIAG", "0") == "1"


def malloc_trim_enabled() -> bool:
    return os.environ.get("SWE4DVAR_MALLOC_TRIM", "0") == "1"


def _csv_path(rank: int = 0) -> Optional[str]:
    """Resolve CSV path; substitutes '{rank}' if present in the env var."""
    p = os.environ.get("SWE4DVAR_EVAL_MEM_DIAG_CSV", "").strip()
    if not p:
        return None
    if "{rank}" in p:
        return p.format(rank=rank)
    return p


def _rank0_only() -> bool:
    return os.environ.get("SWE4DVAR_EVAL_MEM_DIAG_RANK0", "0") == "1"


def _get_rss_mb() -> float:
    """Resident set size in MB for the current process. Linux/macOS."""
    # Prefer /proc/self/status on Linux (cheap, no deps)
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0  # kB → MB
    except Exception:
        pass
    # Fallback: psutil
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass
    # Fallback: resource (macOS reports bytes, Linux kB; rough)
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        ru = usage.ru_maxrss
        # Linux returns kB, macOS returns bytes
        if hasattr(os, "uname") and os.uname().sysname.lower() == "darwin":
            return ru / (1024.0 * 1024.0)
        return ru / 1024.0
    except Exception:
        return -1.0


def _get_petsc_mem_mb() -> tuple[float, float]:
    """(current, max) PETSc-reported memory in MB. Returns (-1, -1) on failure."""
    try:
        from petsc4py import PETSc
        cur = PETSc.Log.getCurrentEventPerfInfo  # noqa: F841 — touch to avoid lint
        mu = PETSc.Log.getMemoryUsage()  # bytes (current "high water" or current)
        return mu / (1024.0 * 1024.0), -1.0
    except Exception:
        return -1.0, -1.0


def _open_libc():
    """Return libc handle (lazy). None on platforms where it isn't loadable."""
    global _LIBC_HANDLE, _LIBC_HAS_TRIM
    if _LIBC_HANDLE is not None or _LIBC_HAS_TRIM is False:
        return _LIBC_HANDLE
    try:
        import ctypes
        import ctypes.util
        name = ctypes.util.find_library("c") or "libc.so.6"
        _LIBC_HANDLE = ctypes.CDLL(name, use_errno=True)
        _LIBC_HAS_TRIM = hasattr(_LIBC_HANDLE, "malloc_trim")
        if not _LIBC_HAS_TRIM:
            _LIBC_HANDLE = None
            _LIBC_HAS_TRIM = False
    except Exception:
        _LIBC_HANDLE = None
        _LIBC_HAS_TRIM = False
    return _LIBC_HANDLE


def maybe_malloc_trim() -> tuple[bool, float]:
    """If SWE4DVAR_MALLOC_TRIM=1, call libc malloc_trim(0) and return (called, rss_freed_mb).

    rss_freed_mb is the difference in RSS from immediately-before to immediately-after
    the trim call. Negative or zero values are normal on platforms where the allocator
    has nothing to release.

    Returns (False, 0.0) when disabled or unavailable.
    """
    if not malloc_trim_enabled():
        return False, 0.0
    libc = _open_libc()
    if libc is None or not _LIBC_HAS_TRIM:
        return False, 0.0
    rss_before = _get_rss_mb()
    try:
        libc.malloc_trim(0)
    except Exception:
        return False, 0.0
    rss_after = _get_rss_mb()
    return True, max(0.0, rss_before - rss_after)


def _ensure_header(path: str) -> None:
    if path in _HEADER_WRITTEN:
        return
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    if new_file:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        except Exception:
            pass
        try:
            with open(path, "a") as f:
                f.write(
                    "wall_s,eval_id,rank,label,"
                    "rss_mb,petsc_curr_mb,petsc_max_mb,trim_freed_mb\n"
                )
        except Exception:
            pass
    _HEADER_WRITTEN.add(path)


def record(
    label: str,
    eval_id: Optional[int] = None,
    comm=None,
    do_trim: bool = False,
) -> dict:
    """Record memory state at a lifecycle point.

    label: short string describing the point (e.g. "before_value_gradient",
           "after_forward", "after_adjoint", "after_cleanup", "after_trim").
    eval_id: optional integer eval index.
    comm: optional MPI comm; if provided and rank0_only, skip writes on other ranks.
    do_trim: if True (and SWE4DVAR_MALLOC_TRIM=1), call malloc_trim *before* sampling
             so the recorded RSS reflects the post-trim state. The trim_freed_mb
             column reports how much RSS dropped.
    """
    if not is_enabled():
        return {}
    rank = 0
    if comm is not None:
        try:
            rank = comm.Get_rank()
        except Exception:
            rank = 0
    if _rank0_only() and rank != 0:
        return {}

    trim_freed = 0.0
    if do_trim:
        _, trim_freed = maybe_malloc_trim()

    rss_mb = _get_rss_mb()
    petsc_cur_mb, petsc_max_mb = _get_petsc_mem_mb()
    wall = time.perf_counter() - _T0

    rec = {
        "wall_s": wall,
        "eval_id": -1 if eval_id is None else int(eval_id),
        "rank": int(rank),
        "label": label,
        "rss_mb": float(rss_mb),
        "petsc_curr_mb": float(petsc_cur_mb),
        "petsc_max_mb": float(petsc_max_mb),
        "trim_freed_mb": float(trim_freed),
    }

    path = _csv_path(rank=rank)
    if path:
        _ensure_header(path)
        try:
            with open(path, "a") as f:
                f.write(
                    f"{rec['wall_s']:.3f},{rec['eval_id']},{rec['rank']},"
                    f"{rec['label']},{rec['rss_mb']:.1f},"
                    f"{rec['petsc_curr_mb']:.1f},{rec['petsc_max_mb']:.1f},"
                    f"{rec['trim_freed_mb']:.1f}\n"
                )
        except Exception:
            pass
    else:
        # No CSV configured: emit a compact stderr line so it shows up in slurm logs.
        if rank == 0:
            print(
                f"[mem-diag] eval={rec['eval_id']} {label} "
                f"rss={rec['rss_mb']:.0f}MB petsc={rec['petsc_curr_mb']:.0f}MB "
                f"trim_freed={rec['trim_freed_mb']:.0f}MB t={rec['wall_s']:.1f}s",
                flush=True,
            )
    return rec


def configure_petsc_malloc_logging() -> None:
    """If SWE4DVAR_PETSC_MALLOC_DEBUG=1, enable PETSc -malloc_debug + -memory_view.

    Intended to be called at the very top of main(), before any PETSc objects are
    constructed. Safe to call when PETSc isn't initialized yet.
    """
    if os.environ.get("SWE4DVAR_PETSC_MALLOC_DEBUG", "0") != "1":
        return
    try:
        from petsc4py import PETSc
        PETSc.Options().setValue("-malloc_debug", "")
        PETSc.Options().setValue("-memory_view", "")
        PETSc.Options().setValue("-log_view_memory", "")
    except Exception:
        pass


# Auto-bump eval_id at each forward solve. The cost function will set its
# own eval id via cost_functions; this counter is a fallback for the reproducer.
_AUTO_EVAL_ID = 0


def bump_auto_eval_id() -> int:
    global _AUTO_EVAL_ID
    _AUTO_EVAL_ID += 1
    return _AUTO_EVAL_ID


def get_auto_eval_id() -> int:
    return _AUTO_EVAL_ID


# Thread-thin "current eval id" register so deep call sites (e.g. the adjoint
# transpose-solve loop in implicit_adjoint.py) can stamp records with the same
# eval_id the cost function set at value_gradient entry, without plumbing it
# through every signature.
_CURRENT_EVAL_ID: int = -1


def set_current_eval_id(eid: int) -> None:
    global _CURRENT_EVAL_ID
    _CURRENT_EVAL_ID = int(eid)


def get_current_eval_id() -> int:
    return _CURRENT_EVAL_ID

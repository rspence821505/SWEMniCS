"""Shim module to expose data assimilation covariance classes as ``covariance``.

`tests/test_covariance.py` expects ``import covariance`` to work.  The real
implementation lives in ``swemnics/data-assimilation/covariance.py`` (hyphenated
directory), which cannot be imported via the regular dotted module path, so we
load it manually and re-export its public symbols.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Iterable, List
import sys


def _load_covariance_module() -> ModuleType:
    """Load the actual covariance module from its filesystem path."""
    src_dir = Path(__file__).resolve().parent
    module_path = src_dir / "swemnics" / "data-assimilation" / "covariance.py"

    spec = importlib.util.spec_from_file_location(
        "swemnics.data_assimilation.covariance", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot locate covariance module at {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


_cov_module = _load_covariance_module()

if hasattr(_cov_module, "__all__"):
    _export_names: Iterable[str] = _cov_module.__all__  # type: ignore[attr-defined]
else:
    _export_names = [name for name in dir(_cov_module) if not name.startswith("_")]

globals().update({name: getattr(_cov_module, name) for name in _export_names})
__all__: List[str] = list(_export_names)

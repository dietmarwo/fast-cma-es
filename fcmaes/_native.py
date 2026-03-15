"""Import shim for the nanobind extension.

This prefers the installed extension module, but during local source-tree
development it can also load a built extension from ``build/*`` so pytest can
run without installing the wheel first.
"""

from __future__ import annotations

import importlib.machinery
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterator


def _candidate_extension_paths() -> Iterator[Path]:
    explicit = os.environ.get("FCMAES_EXT_PATH")
    if explicit:
        yield Path(explicit)

    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent
    build_dir = repo_root / "build"

    if not build_dir.exists():
        return

    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    seen = set()
    for suffix in suffixes:
        patterns = (
            f"_fcmaes_ext{suffix}",
            f"*/_fcmaes_ext{suffix}",
            f"*/fcmaes/_fcmaes_ext{suffix}",
        )
        for pattern in patterns:
            for candidate in sorted(build_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield resolved


def _load_local_extension() -> ModuleType:
    fullname = "fcmaes._fcmaes_ext"
    for candidate in _candidate_extension_paths():
        spec = importlib.util.spec_from_file_location(fullname, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
        parent = sys.modules.get("fcmaes")
        if parent is not None:
            setattr(parent, "_fcmaes_ext", module)
        return module
    raise ModuleNotFoundError(
        "No module named 'fcmaes._fcmaes_ext'. "
        "Build the extension first or set FCMAES_EXT_PATH to the compiled module."
    )


try:
    _fcmaes_ext = importlib.import_module("fcmaes._fcmaes_ext")
except ModuleNotFoundError:
    _fcmaes_ext = _load_local_extension()

__all__ = ["_fcmaes_ext"]

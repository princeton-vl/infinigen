# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Locate packaged kernels without compiling or changing process settings."""

import ctypes
import logging
from functools import lru_cache
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

logger = logging.getLogger(__name__)

# Every kernel the native path loads. Each is loaded only when generation reaches
# it, so they are all checked up front.
_REQUIRED = (
    "decorate_dispatch",
    "subdivide",
    "surface_grid",
    "deform",
    "stucci",
    "topology",
)

# stucci and deform are GPL-2.0-or-later and are built into the optional
# infinigen_gpl submodule; the rest are BSD-3-Clause and are built here.
_DIRECTORIES = (
    Path(__file__).parent / "native",
    Path(__file__).parents[5] / "infinigen_gpl" / "perf" / "pinecone",
)


def _find(name):
    for directory in _DIRECTORIES:
        for suffix in EXTENSION_SUFFIXES:
            path = directory / (name + suffix)
            if path.is_file():
                return path
    return None


@lru_cache(None)
def load_library(name):
    path = _find(name.removesuffix(".so"))
    if path is None:
        raise RuntimeError(
            f"Missing pinecone native library {name}. Build with "
            "INFINIGEN_INSTALL_PERF=True pip install -e ., or bind "
            "native_pinecone.enabled = False to use the default pinecone."
        )
    return ctypes.CDLL(str(path))


@lru_cache(None)
def kernels_available():
    """True when every kernel the native path needs is built; otherwise logs the missing
    ones, and the factory uses the default implementation."""
    missing = [name for name in _REQUIRED if _find(name) is None]
    if missing:
        logger.warning(
            "pinecone native kernels %s are not built; using the default "
            "implementation. Build with: INFINIGEN_INSTALL_PERF=True pip install -e .",
            ", ".join(missing),
        )
    return not missing

# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Plan a tile's erosion and snowfall as steps to skip or run in a pool worker.

Sampling the base heightmap consumes the global random state, so it runs in place;
erosion and snowfall only touch the tile folder, so skipping or deferring their
gin-resolved `(module, function, kwargs)` steps leaves the RNG stream unmoved.
"""

import contextvars
import importlib
from contextlib import contextmanager
from pathlib import Path

# Context rather than arguments: the gated entry points keep the default signatures.
_planning = contextvars.ContextVar("infinigen_land_process_planning", default=None)
_needed_process = contextvars.ContextVar("infinigen_land_process_needed", default=None)

# needed_process() when no scene said which land process it reads
NOT_GIVEN = object()

# the pool of an enabled land_process_pool() context, set by the pool module
_active_pool = None


@contextmanager
def planning(**context):
    """Inside this, land-process entry points return their step instead of running,
    with `context` (a `seed`) passed on to it."""
    token = _planning.set(dict(context))
    try:
        yield
    finally:
        _planning.reset(token)


def planning_context():
    """The active `planning()` context, or None when the entry points should run."""
    return _planning.get()


@contextmanager
def needed_processes(process):
    """Declare the land process the scene reads for tiles generated inside this."""
    token = _needed_process.set((process,))
    try:
        yield
    finally:
        _needed_process.reset(token)


def needed_process():
    """The process set by `needed_processes()`, or `NOT_GIVEN` outside it."""
    value = _needed_process.get()
    return NOT_GIVEN if value is None else value[0]


def run_steps(steps):
    """Run `steps` in this process, in order, skipping None entries."""
    for step in steps:
        if step is None:
            continue
        module_name, function_name, kwargs = step
        module = importlib.import_module(module_name)
        getattr(module, function_name)(**kwargs)


def run_land_process_steps(folder, steps):
    """Run `folder`'s steps now, or submit them to the active land-process pool."""
    if _active_pool is not None:
        _active_pool.submit(folder, steps)
    else:
        run_steps(steps)


def mark_finished(folder, marker):
    """Create `marker` once every step submitted for `folder` has run."""
    if _active_pool is not None and _active_pool.has_job(folder):
        _active_pool.add_finish_marker(folder, marker)
    else:
        Path(marker).touch()

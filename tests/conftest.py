# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Vineet Bansal

import logging
import sys

import bpy
import numpy as np
import pytest
from bpy.app.handlers import persistent
from procfunc.util.teardown import exit_skipping_teardown

_session_exit_code = 0
_render_calls = []


@persistent
def _record_render(*args: object) -> None:
    _render_calls.append(args)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    _render_calls.clear()
    bpy.app.handlers.render_init.append(_record_render)
    try:
        result = yield
    finally:
        bpy.app.handlers.render_init.remove(_record_render)
    if _render_calls and item.get_closest_marker("render_gpu") is None:
        pytest.fail(
            "test rendered without @pytest.mark.render_gpu; GitHub-hosted runners "
            "cannot render, so mark it to run only on self-hosted CI"
        )
    return result


# The --optimized selection for the session, restored after a test changes it.
_session_packages = False
_session_selection = None


def pytest_sessionfinish(session, exitstatus):
    global _session_exit_code
    _session_exit_code = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    sys.stdout.flush()
    sys.stderr.flush()
    logging.shutdown()
    exit_skipping_teardown(_session_exit_code)


def pytest_addoption(parser):
    parser.addoption(
        "--optimized",
        nargs="?",
        const="all",
        default=None,
        metavar="NAMES",
        help=(
            "select optimized implementations for this run: --optimized for "
            "every one, or --optimized=trees,leaves for those. Tests that "
            "exercise one skip themselves when it is not selected."
        ),
    )


def pytest_configure(config):
    logging.basicConfig(
        format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        force=True,  # Override any existing configuration
    )
    names = config.getoption("--optimized")
    if names is not None:
        # before collection: some test modules check the selection on import
        import infinigen  # keep-local: v1-only, absent in the v2 env

        global _session_packages, _session_selection
        _session_packages = (
            True
            if names == "all"
            else [n.strip() for n in names.split(",") if n.strip()]
        )
        _session_selection = infinigen.use_optimized(_session_packages)


@pytest.fixture(scope="function", autouse=True)
def cleanup():
    yield
    try:
        import gin  # keep-local: v1-only dep, absent in the v2 test env
    except ModuleNotFoundError:
        pass
    else:
        gin.clear_config()

    # The selection is process-wide; undo any change a test made.
    try:
        import infinigen  # keep-local: v1-only, absent in the v2 env
        from infinigen import _perf_selection
    except ModuleNotFoundError:
        pass
    else:
        if _perf_selection.optimized_packages() != (
            _session_selection or dict.fromkeys(_perf_selection.NAMES, False)
        ):
            infinigen.use_optimized(_session_packages)

    bpy.ops.wm.read_factory_settings(use_empty=True)


@pytest.fixture(scope="function")
def rng():
    return np.random.default_rng(seed=42)

# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Vineet Bansal

import logging
import sys

import bpy
import numpy as np
import pytest
from procfunc.util.teardown import exit_skipping_teardown

_session_exit_code = 0


def pytest_sessionfinish(session, exitstatus):
    global _session_exit_code
    _session_exit_code = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    sys.stdout.flush()
    sys.stderr.flush()
    logging.shutdown()
    exit_skipping_teardown(_session_exit_code)


def pytest_configure(config):
    logging.basicConfig(
        format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        force=True,  # Override any existing configuration
    )


@pytest.fixture(scope="function", autouse=True)
def cleanup():
    yield
    try:
        import gin  # keep-local: v1-only dep, absent in the v2 test env
    except ModuleNotFoundError:
        pass
    else:
        gin.clear_config()
    bpy.ops.wm.read_factory_settings(use_empty=True)


@pytest.fixture(scope="function")
def rng():
    return np.random.default_rng(seed=42)

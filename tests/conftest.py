# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Vineet Bansal

import logging

import bpy
import gin
import numpy as np
import pytest


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
    gin.clear_config()
    bpy.ops.wm.read_factory_settings(use_empty=True)


@pytest.fixture(scope="function")
def rng():
    return np.random.default_rng(seed=42)

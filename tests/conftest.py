# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Vineet Bansal

import bpy
import gin
import pytest


@pytest.fixture(scope="function", autouse=True)
def cleanup():
    yield
    gin.clear_config()
    bpy.ops.wm.read_factory_settings(use_empty=True)

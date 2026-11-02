# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import bpy

from infinigen.core.util import blender as butil
from infinigen.perf.core.util.isolation import IsolatedScene


def test_the_scratch_view_layer_is_never_rendered():
    butil.clear_scene()
    scene = bpy.context.scene
    stale = scene.view_layers.get(IsolatedScene.VL_NAME)
    if stale is None:
        stale = scene.view_layers.new(IsolatedScene.VL_NAME)
    stale.use = True  # a saved scene with the layer enabled
    with IsolatedScene(tag="test"):
        butil.spawn_empty("built")
    assert scene.view_layers[IsolatedScene.VL_NAME].use is False
    assert scene.view_layers["ViewLayer"].use is True

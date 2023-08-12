# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import bpy
import numpy as np


def test_exists_fire_in_scene():
    for obj in bpy.data.objects:
        if "Fluid" in obj.modifiers:
            return
    assert False


def test_some_density_in_frame():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in bpy.data.objects:
        if "Fluid" in obj.modifiers:
            object_eval = obj.evaluated_get(depsgraph)
            mod = object_eval.modifiers["Fluid"]
            if mod.fluid_type == "DOMAIN":
                density_grid = np.array(mod.domain_settings.density_grid)
                assert any(density_grid > 0)


def test_all_fluid_domains_have_density():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in bpy.data.objects:
        if "Fluid" in obj.modifiers:
            object_eval = obj.evaluated_get(depsgraph)
            mod = object_eval.modifiers["Fluid"]
            if mod.fluid_type == "DOMAIN":
                density_grid = np.array(mod.domain_settings.density_grid)
                assert any(density_grid > 0)


def test_depth_not_infinite_all_pixels():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 10

    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links
    rl = tree.nodes.new("CompositorNodeRLayers")

    # create output node
    v = tree.nodes.new("CompositorNodeViewer")
    v.use_alpha = False

    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True  ## Links
    links.new(rl.outputs["Depth"], v.inputs[0])  # link Z to output

    # render
    bpy.ops.render.render()

    pixels = bpy.data.images["Viewer Node"].pixels
    pixels = np.array(pixels)

    print(pixels[:100])

    for x in pixels:
        assert not np.isclose(x, 1e10)

    print(np.max(pixels))


def test_depth_infinity_portion():
    eps = 0.01
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 10

    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links
    rl = tree.nodes.new("CompositorNodeRLayers")

    # create output node
    v = tree.nodes.new("CompositorNodeViewer")
    v.use_alpha = False

    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True  ## Links
    links.new(rl.outputs["Depth"], v.inputs[0])  # link Z to output

    # render
    bpy.ops.render.render()

    pixels = bpy.data.images["Viewer Node"].pixels
    pixels = np.array(pixels)

    cnt = 0
    for x in pixels:
        if np.isclose(x, 1e10):
            cnt += 1
    cnt = cnt / 3
    n = len(pixels) / 4
    print(cnt / n)
    assert cnt / n < eps

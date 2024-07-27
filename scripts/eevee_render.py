# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import argparse
from pathlib import Path

import bpy
import mathutils

from infinigen.core.rendering.render import enable_gpu
from infinigen.core.util import blender as butil


def get_override(area_type, region_type):
    for area in bpy.context.screen.areas:
        if area.type == area_type:
            for region in area.regions:
                if region.type == region_type:
                    override = {"area": area, "region": region}
                    return override
    # error message if the area or region wasn't found
    raise RuntimeError(
        "Wasn't able to find",
        region_type,
        " in area ",
        area_type,
        "\n Make sure it's open while executing script.",
    )


def process(scene_folder: Path):
    butil.clear_scene()
    bpy.ops.wm.open_mainfile(filepath=str(scene_folder / "scene.blend"))

    for o in butil.get_collection("ceiling").objects:
        o.active_material.use_backface_culling = True
    for o in butil.get_collection("wall").objects:
        o.active_material.use_backface_culling = True

    bpy.ops.object.light_add(type="SUN")
    light = bpy.context.active_object
    light.rotation_euler = (-0.7, 0.1, 0.22)
    light.data.energy = 5

    room = next(o for o in butil.get_collection("floor").objects if not o.hide_render)

    cam = bpy.context.scene.camera
    t = mathutils.Matrix.Translation(room.location)
    s = mathutils.Matrix.Scale(1.5, 4)
    cam.matrix_world = (
        t
        @ s
        @ mathutils.Euler((0.42, 0, 0.2)).to_matrix().to_4x4()
        @ t.inverted()
        @ cam.matrix_world
    )

    bpy.context.scene.render.filepath = str(scene_folder / "Image_EEVEE")
    enable_gpu()
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.ops.render.render(write_still=True)

    butil.save_blend(scene_folder / "eevee.blend")


parser = argparse.ArgumentParser()
parser.add_argument("input_folder", type=Path)
args = parser.parse_args()

for p in args.input_folder.iterdir():
    if not (p / "scene.blend").exists():
        print(f"{p=} has no scene.blend")
        continue
    process(p)

# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


from pathlib import Path

import bpy
import gin
import os
import shutil
from infinigen.terrain.utils import random_int
from infinigen.core.util.blender import ViewportMode
from infinigen.core.util.logging import Timer
from infinigen.core.util.random import random_general as rg


spatial_size = 40
resolution = 64
buffered_frames = 10

@gin.configurable
def ocean_asset(
    folder,
    frame_start,
    frame_end,
    time_scale=0.04,
    wave_scale=("uniform", 2, 3),
    choppiness=("uniform", 0.5, 1),
    wave_alignment=("uniform", 0, 0.1),
    verbose=0,
    spectrum="PHILLIPS", #("choice", ["PHILLIPS", "PIERSON_MOSKOWITZ", "JONSWAP", "TEXEL_MARSEN_ARSLOE"], [0.5, 0.5/3, 0.5/3, 0.5/3]),
    bake_foam_fade=0.8,
    link_folder=None,
):
    tmp_start, tmp_end = bpy.context.scene.frame_start, bpy.context.scene.frame_end
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = frame_start, frame_end + buffered_frames
    spectrum = rg(spectrum)
    params={
        "random_seed": max(0, random_int()),
        "geometry_mode": 'DISPLACE',
        "spatial_size": spatial_size,
        "wave_scale": wave_scale if spectrum == "PHILLIPS" else 0.5,
        "resolution": resolution,
        "viewport_resolution": resolution,
        "choppiness": choppiness,
        "wave_alignment": wave_alignment,
        "use_foam": True,
        "foam_layer_name": "foam",
        "spectrum": spectrum,
        "bake_foam_fade": bake_foam_fade,
    }

    with Timer("build ocean", disable_timer=not verbose):
        for mod_key in params:
            params[mod_key] = rg(params[mod_key])
        Path(folder).mkdir(parents=True, exist_ok=True)
        bpy.ops.mesh.primitive_plane_add(size=spatial_size)
        obj = bpy.context.active_object
        obj.name = "ocean"
        with ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
            bpy.ops.mesh.subdivide(number_cuts=256)
            bpy.ops.mesh.subdivide(number_cuts=16)
        mod = obj.modifiers.new(name="ocean", type="OCEAN")
        for mod_key in params:
            setattr(mod, mod_key, params[mod_key])
        mod.frame_start = frame_start
        mod.frame_end = frame_end + buffered_frames
        if (folder / "cache").exists():
            shutil.rmtree(folder / "cache")
        (folder / "cache").mkdir(parents=True)
        mod.filepath = str(folder / "cache")
        for t, f in [(time_scale * frame_start, frame_start), (time_scale * (frame_end + buffered_frames), frame_end + buffered_frames)]:
            mod.time = t
            mod.keyframe_insert("time", frame=f)
        obj.animation_data.action.fcurves[0].keyframe_points[0].interpolation = 'LINEAR'
        obj.animation_data.action.fcurves[0].keyframe_points[1].interpolation = 'LINEAR'
    with Timer("bake ocean", disable_timer=not verbose):
        bpy.ops.object.ocean_bake(modifier="ocean")
        while True:
            if (folder / f"cache/foam_{frame_end + buffered_frames:04d}.exr").exists(): break
    
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = tmp_start, tmp_end

    if link_folder is not None:
        for i in range(frame_start, frame_end + 1):
            dst = link_folder / f"cache/disp_{i:04d}.exr"
            if not dst.exists():
                os.symlink(folder / f"cache/disp_{i + buffered_frames:04d}.exr", dst)
            dst = link_folder / f"cache/foam_{i:04d}.exr"
            if not dst.exists():
                os.symlink(folder / f"cache/foam_{i + buffered_frames:04d}.exr", dst)
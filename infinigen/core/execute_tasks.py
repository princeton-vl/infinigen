# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import logging
import os
import pickle
import shutil
import time
import typing
from collections import defaultdict
from pathlib import Path

# ruff: noqa: E402
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # This must be done BEFORE import cv2.
# See https://github.com/opencv/opencv/issues/21326#issuecomment-1008517425

import bpy
import gin

import infinigen.assets.scatters
from infinigen.core import init, surface
from infinigen.core.placement import camera as cam_util
from infinigen.core.rendering.render import render_image
from infinigen.core.rendering.resample import resample_scene
from infinigen.core.tagging import tag_system
from infinigen.core.util import blender as butil
from infinigen.core.util import exporting
from infinigen.core.util.logging import Timer, create_text_file, save_polycounts
from infinigen.core.util.math import int_hash
from infinigen.core.util.organization import Task
from infinigen.terrain import Terrain
from infinigen.tools.export import export_scene, triangulate_meshes

logger = logging.getLogger(__name__)


def get_scene_tag(name):
    try:
        o = next(o for o in bpy.data.objects if o.name.startswith(f"{name}="))
        return o.name.split("=")[-1].strip("'\"")
    except StopIteration:
        return None


@gin.configurable
def render(
    scene_seed,
    output_folder,
    camera,
    render_image_func=render_image,
    resample_idx=None,
    hide_water=False,
):
    if hide_water and "water_fine" in bpy.data.objects:
        logger.info("Hiding water fine")
        bpy.data.objects["water_fine"].hide_render = True
        bpy.data.objects["water_fine"].hide_viewport = True
    if resample_idx is not None and resample_idx != 0:
        resample_scene(int_hash((scene_seed, resample_idx)))
    with Timer("Render Frames"):
        render_image_func(frames_folder=Path(output_folder), camera=camera)


def is_static(obj):
    while True:
        if obj.name.startswith("scatter:"):
            return False
        if obj.users_collection[0].name.startswith("assets:"):
            return False
        if obj.constraints is not None and len(obj.constraints) > 0:
            return False
        if obj.animation_data is not None:
            return False
        for modifier in obj.modifiers:
            if modifier.type == "NODES":
                if modifier.node_group.animation_data is not None:
                    return False
            elif modifier.type == "ARMATURE":
                return False

        if obj.parent is None:
            break
        obj = obj.parent
    return True


@gin.configurable
def save_meshes(
    scene_seed: int,
    output_folder: Path,
    cameras: list[bpy.types.Object],
    frame_range,
    resample_idx=False,
    point_trajectory_src_frame=1,
):
    if resample_idx is not None and resample_idx > 0:
        resample_scene(int_hash((scene_seed, resample_idx)))

    triangulate_meshes()

    for col in bpy.data.collections:
        col.hide_viewport = col.hide_render

    previous_frame_mesh_id_mapping = dict()
    current_frame_mesh_id_mapping = defaultdict(dict)

    # save static meshes
    for obj in bpy.data.objects:
        obj.hide_viewport = not (not obj.hide_render and is_static(obj))
    frame_idx = point_trajectory_src_frame
    frame_info_folder = output_folder / f"frame_{frame_idx:04d}"
    frame_info_folder.mkdir(parents=True, exist_ok=True)

    logger.info("Working on static objects")
    exporting.save_obj_and_instances(
        frame_info_folder / "static_mesh",
        previous_frame_mesh_id_mapping,
        current_frame_mesh_id_mapping,
    )
    previous_frame_mesh_id_mapping = dict(current_frame_mesh_id_mapping)
    current_frame_mesh_id_mapping.clear()

    for obj in bpy.data.objects:
        obj.hide_viewport = not (not obj.hide_render and not is_static(obj))

    for frame_idx in set(
        [point_trajectory_src_frame]
        + list(range(int(frame_range[0]), int(frame_range[1] + 2)))
    ):
        bpy.context.scene.frame_set(frame_idx)
        bpy.context.view_layer.update()
        frame_info_folder = output_folder / f"frame_{frame_idx:04d}"
        frame_info_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"save_meshes processing {frame_idx=}")

        exporting.save_obj_and_instances(
            frame_info_folder / "mesh",
            previous_frame_mesh_id_mapping,
            current_frame_mesh_id_mapping,
        )
        for cam in cameras:
            cam_util.save_camera_parameters(
                camera_obj=cam,
                output_folder=frame_info_folder / "cameras",
                frame=frame_idx,
            )
        previous_frame_mesh_id_mapping = dict(current_frame_mesh_id_mapping)
        current_frame_mesh_id_mapping.clear()


def validate_version(scene_version):
    if (
        scene_version is None
        or scene_version.split(".")[:-1] != infinigen.__version__.split(".")[:-1]
    ):
        raise ValueError(
            f"infinigen_examples/generate_nature.py {infinigen.__version__=} attempted to load a scene created by version {scene_version=}"
        )
    if scene_version != infinigen.__version__:
        logger.warning(
            f"{infinigen.__version__=} has minor version mismatch with {scene_version=}"
        )


@gin.configurable
def group_collections(config):
    for config in config:  # Group collections before fine runs
        butil.group_in_collection(
            [o for o in bpy.data.objects if o.name.startswith(f'{config["name"]}:')],
            config["name"],
        )
        butil.group_toplevel_collections(
            config["name"],
            hide_viewport=config["hide_viewport"],
            hide_render=config["hide_render"],
        )


@gin.configurable
def execute_tasks(
    compose_scene_func: typing.Callable,
    populate_scene_func: typing.Callable,
    input_folder: Path,
    output_folder: Path,
    task: str,
    scene_seed: int,
    frame_range: tuple[int],
    camera_id: tuple[int],
    resample_idx: int = None,
    output_blend_name: str = "scene.blend",
    generate_resolution=(1280, 720),
    fps: int = 24,
    reset_assets=True,
    dryrun=False,
    optimize_terrain_diskusage=False,
    point_trajectory_src_frame=1,
):
    if input_folder != output_folder:
        if reset_assets:
            if os.path.islink(output_folder / "assets"):
                os.unlink(output_folder / "assets")
            elif (output_folder / "assets").exists():
                shutil.rmtree(output_folder / "assets")
        if (
            (not os.path.islink(output_folder / "assets"))
            and (not (output_folder / "assets").exists())
            and input_folder is not None
            and (input_folder / "assets").exists()
        ):
            os.symlink(input_folder / "assets", output_folder / "assets")
            # in this way, even coarse task can have input_folder to have pregenerated on-the-fly assets (e.g., in last run) to speed up developing

    if dryrun:
        time.sleep(15)
        return

    if Task.Coarse not in task and task != Task.FineTerrain:
        with Timer("Reading input blendfile"):
            bpy.ops.wm.open_mainfile(filepath=str(input_folder / "scene.blend"))
            tag_system.load_tag(path=str(input_folder / "MaskTag.json"))
        butil.approve_all_drivers()

    if frame_range[1] < frame_range[0]:
        raise ValueError(
            f"{frame_range=} is invalid, frame range must be nonempty. Blender end frame is INCLUSIVE"
        )

    logger.info(
        f"Processing frames {frame_range[0]} through {frame_range[1]} inclusive"
    )
    bpy.context.scene.frame_start = int(frame_range[0])
    bpy.context.scene.frame_end = int(frame_range[1])
    bpy.context.scene.frame_set(int(frame_range[0]))
    bpy.context.scene.render.fps = fps
    bpy.context.scene.render.resolution_x = generate_resolution[0]
    bpy.context.scene.render.resolution_y = generate_resolution[1]
    bpy.context.view_layer.update()

    surface.registry.initialize_from_gin()
    init.configure_blender()

    if Task.Coarse in task:
        butil.clear_scene(targets=[bpy.data.objects])
        butil.spawn_empty(f"{infinigen.__version__=}")
        info = compose_scene_func(output_folder, scene_seed)
        outpath = output_folder / "assets"
        outpath.mkdir(exist_ok=True)
        with open(outpath / "info.pickle", "wb") as f:
            pickle.dump(info, f, protocol=pickle.HIGHEST_PROTOCOL)

    camera_rigs = cam_util.get_camera_rigs()
    camrig_id, subcam_id = camera_id
    active_camera = camera_rigs[camrig_id].children[subcam_id]
    cam_util.set_active_camera(active_camera)

    group_collections()

    if Task.Populate in task and populate_scene_func is not None:
        populate_scene_func(output_folder, scene_seed, camera_rigs)

    need_terrain_processing = "atmosphere" in bpy.data.objects

    if Task.FineTerrain in task and need_terrain_processing:
        with open(output_folder / "assets" / "info.pickle", "rb") as f:
            info = pickle.load(f)
        terrain = Terrain(
            scene_seed,
            surface.registry,
            task=task,
            on_the_fly_asset_folder=output_folder / "assets",
            height_offset=info["height_offset"],
            whole_bbox=info["whole_bbox"],
        )

        terrain.fine_terrain(
            output_folder,
            cameras=[c for rig in camera_rigs for c in rig.children],
            optimize_terrain_diskusage=optimize_terrain_diskusage,
        )

    group_collections()

    if input_folder is not None and input_folder != output_folder:
        for mesh in os.listdir(input_folder):
            if (
                mesh.endswith(".glb") or mesh.endswith(".b_displacement.npy")
            ) and not os.path.islink(output_folder / mesh):
                os.symlink(input_folder / mesh, output_folder / mesh)
    if Task.Coarse in task or Task.Populate in task or Task.FineTerrain in task:
        with Timer("Writing output blendfile"):
            logging.info(
                f"Writing output blendfile to {output_folder / output_blend_name}"
            )
            if optimize_terrain_diskusage and task == [Task.FineTerrain]:
                os.symlink(
                    input_folder / output_blend_name, output_folder / output_blend_name
                )
            else:
                bpy.ops.wm.save_mainfile(
                    filepath=str(output_folder / output_blend_name)
                )

        tag_system.save_tag(path=str(output_folder / "MaskTag.json"))

        with (output_folder / "version.txt").open("w") as f:
            f.write(f"{infinigen.__version__}\n")

        with (output_folder / "polycounts.txt").open("w") as f:
            save_polycounts(f)

    for col in bpy.data.collections["unique_assets"].children:
        col.hide_viewport = False

    if need_terrain_processing and (
        Task.Render in task
        or Task.GroundTruth in task
        or Task.MeshSave in task
        or Task.Export in task
    ):
        terrain = Terrain(
            scene_seed,
            surface.registry,
            task=task,
            on_the_fly_asset_folder=output_folder / "assets",
        )
        if optimize_terrain_diskusage:
            terrain.load_glb(output_folder)

    if Task.Render in task or Task.GroundTruth in task:
        render(
            scene_seed,
            output_folder=output_folder,
            camera=active_camera,
            resample_idx=resample_idx,
        )

    if Task.Export in task:
        export_scene(input_folder / output_blend_name, output_folder)

    if Task.MeshSave in task:
        save_meshes(
            scene_seed,
            output_folder=output_folder,
            cameras=[c for rig in camera_rigs for c in rig.children],
            frame_range=frame_range,
            point_trajectory_src_frame=point_trajectory_src_frame,
        )


def main(input_folder, output_folder, scene_seed, task, task_uniqname, **kwargs):
    version_req = ["4.2.0"]
    assert bpy.app.version_string in version_req, (
        f"You are using blender={bpy.app.version_string} which is "
        f"not supported. Please use {version_req}"
    )
    logger.info(f"infinigen version {infinigen.__version__}")
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if input_folder is not None:
        input_folder = Path(input_folder).absolute()
    output_folder = Path(output_folder).absolute()
    output_folder.mkdir(exist_ok=True, parents=True)

    if task_uniqname is not None:
        create_text_file(filename=f"START_{task_uniqname}")

    with Timer("MAIN TOTAL"):
        execute_tasks(
            input_folder=input_folder,
            output_folder=output_folder,
            task=task,
            scene_seed=scene_seed,
            **kwargs,
        )

    if task_uniqname is not None:
        create_text_file(filename=f"FINISH_{task_uniqname}")
        create_text_file(
            filename=f"operative_gin_{task_uniqname}.txt",
            text=gin.operative_config_str(),
        )

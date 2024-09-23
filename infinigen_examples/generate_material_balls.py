# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this
# source tree.

# Authors:
# - Lingjie Mei
# - Alex Raistrick
# - Karhan Kayan - add fire option

import importlib
import logging
import os
from pathlib import Path

from numpy.random import uniform
from tqdm import tqdm

from infinigen.assets.materials import tile
from infinigen.assets.materials.ceramic import shader_ceramic

# ruff: noqa: E402
# NOTE: logging config has to be before imports that use logging
logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

import bpy
import gin
import numpy as np

from infinigen.assets.lighting import (
    hdri_lighting,
    holdout_lighting,
    sky_lighting,
    three_point_lighting,
)
from infinigen.assets.utils.decorate import read_base_co
from infinigen.assets.utils.misc import subclasses
from infinigen.core import init, surface
from infinigen.core.placement import factory

# noinspection PyUnresolvedReferences
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.test_utils import load_txt_list
from infinigen_examples.generate_individual_assets import (
    adjust_cam_distance,
    make_args,
    setup_camera,
)

logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,
)

scale = 0.4


def build_scene_surface(factory_name, idx):
    try:
        with gin.unlock_config():
            try:
                template = importlib.import_module(
                    f"infinigen.assets.materials.{factory_name}"
                )
            except ImportError:
                for subdir in os.listdir("infinigen/assets/materials"):
                    if not subdir.endswith(".py"):
                        with gin.unlock_config():
                            module = importlib.import_module(
                                f'infinigen.assets.materials.{subdir.split(".")[0]}'
                            )
                        if hasattr(module, factory_name):
                            template = getattr(module, factory_name)
                            break
                else:
                    raise Exception(f"{factory_name} not Found.")
            if type(template) is type:
                template = template(idx)

            bpy.ops.mesh.primitive_ico_sphere_add(radius=scale, subdivisions=7)
            asset = bpy.context.active_object
            asset.rotation_euler = (
                uniform(np.pi / 6, np.pi / 3),
                uniform(-np.pi / 12, np.pi / 12),
                uniform(-np.pi / 12, np.pi / 12),
            )

            with FixedSeed(idx):
                if "metal" in factory_name or "sofa_fabric" in factory_name:
                    template.apply(asset, scale=0.1)
                elif "hardwood" in factory_name:
                    template.apply(asset, rotation=(np.pi / 2, 0, 0))
                elif "brick" in factory_name:
                    template.apply(asset, height=uniform(0.25, 0.3))
                elif "tile" in factory_name:
                    template.apply(asset, alternating=idx % 4 in [0, 1])
                else:
                    template.apply(asset)
    except ModuleNotFoundError:
        raise Exception(f"{factory_name} not Found.")
    return asset


def build_scene(path, factory_names, args):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y = map(
        int, args.resolution.split("x")
    )
    scene.cycles.samples = args.samples
    butil.clear_scene()

    if not args.fire:
        bpy.context.scene.render.film_transparent = args.film_transparent
        bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value[
            -1
        ] = 0
    camera, center = setup_camera(args)

    assets = []
    with tqdm(total=len(factory_names)) as pbar:
        for idx, factory_name in enumerate(factory_names):
            asset = build_scene_surface(factory_name, idx)
            assets.append(asset)
            asset.name = factory_name
            pbar.update(1)
    margin = scale * 2.2
    size = 3
    for i in range(len(assets)):
        assets[i].location = (i // size) * margin, (i % size) * margin, scale

    bpy.ops.mesh.primitive_grid_add(size=1, x_subdivisions=400, y_subdivisions=400)
    asset = bpy.context.active_object
    asset.scale = [scale * len(assets) / size * 4] * 3
    asset.location = (len(assets) // size - 1) * margin / 2, size // 2 * margin * 0.8, 0
    tile.apply(asset, shader_func=shader_ceramic, alternating=True)

    with FixedSeed(args.lighting):
        if args.hdri:
            hdri_lighting.add_lighting()
        elif args.three_point:
            holdout_lighting.add_lighting()
            three_point_lighting.add_lighting(asset)
        else:
            sky_lighting.add_lighting(camera)
            nodes = bpy.data.worlds["World"].node_tree.nodes
            sky_texture = [n for n in nodes if n.name.startswith("Sky Texture")][-1]
            sky_texture.sun_elevation = np.deg2rad(args.elevation)
            sky_texture.sun_rotation = np.pi * 0.75

    if args.scale_reference:
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.3, depth=1.8, location=(4.9, 4.9, 1.8 / 2)
        )

    if args.cam_center > 0 and asset:
        co = read_base_co(asset) + asset.location
        center.location = (np.amin(co, 0) + np.amax(co, 0)) / 2
        center.location[-1] += args.cam_zoff

    if args.cam_dist <= 0 and asset:
        adjust_cam_distance(asset, camera, args.margin, 0.6)

    cam_info_ng = bpy.data.node_groups.get("nodegroup_active_cam_info")
    if cam_info_ng is not None:
        cam_info_ng.nodes["Object Info"].inputs["Object"].default_value = camera

    if args.save_blend:
        (path / "scenes").mkdir(exist_ok=True)
        butil.save_blend(f"{path}/scenes/scene_{idx:03d}.blend", autopack=True)


def main(args):
    bpy.context.window.workspace = bpy.data.workspaces["Geometry Nodes"]

    init.apply_gin_configs("infinigen_examples/configs_indoor", skip_unknown=True)
    surface.registry.initialize_from_gin()

    extras = "[%(filename)s:%(lineno)d] " if args.loglevel == logging.DEBUG else ""
    logging.basicConfig(
        format=f"[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] {extras}| %(message)s",
        level=args.loglevel,
        datefmt="%H:%M:%S",
    )
    logging.getLogger("infinigen").setLevel(args.loglevel)

    if ".txt" in args.factories[0]:
        name = args.factories[0].split(".")[-2].split("/")[-1]
    else:
        name = "_".join(args.factories)
    path = Path(os.getcwd()) / "outputs" / name
    path.mkdir(exist_ok=True)

    if args.gpu:
        init.configure_render_cycles()

    factories = list(args.factories)
    if "ALL_ASSETS" in factories:
        factories += [f.__name__ for f in subclasses(factory.AssetFactory)]
        factories.remove("ALL_ASSETS")
    elif "ALL_SCATTERS" in factories:
        factories += [f.stem for f in Path("surfaces/scatters").iterdir()]
        factories.remove("ALL_SCATTERS")
    elif "ALL_MATERIALS" in factories:
        factories += [f.stem for f in Path("infinigen/assets/materials").iterdir()]
        factories.remove("ALL_MATERIALS")
    elif ".txt" in factories[0]:
        factories = [
            f.split(".")[-1] for f in load_txt_list(factories[0], skip_sharp=False)
        ]
    elif "woods" in factories[0]:
        factories = (
            ["wood"] * 3
            + ["staggered_wood_tile"] * 3
            + ["square_wood_tile"] * 3
            + ["hexagon_wood_tile"] * 3
            + ["composite_wood_tile"] * 3
            + ["crossed_wood_tile"] * 3
        )

    try:
        build_scene(path, factories, args)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    args = make_args()
    args.no_mod = args.no_mod or args.fire
    args.film_transparent = args.film_transparent
    with FixedSeed(1):
        main(args)

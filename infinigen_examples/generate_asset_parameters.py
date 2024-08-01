# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this
# source tree.

# Authors:
# - Lingjie Mei
# - Alex Raistrick
# - Karhan Kayan - add fire option

import logging
import math
import os
import random
import subprocess
import traceback
from collections.abc import Callable
from itertools import product
from pathlib import Path

# ruff: noqa: E402
# NOTE: logging config has to be before imports that use logging
logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

import bpy
import numpy as np
from PIL import Image

from infinigen.assets.lighting import (
    CeilingLightFactory,
    hdri_lighting,
    holdout_lighting,
    sky_lighting,
    three_point_lighting,
)
from infinigen.assets.materials.woods import non_wood_tile, wood_tile
from infinigen.assets.utils.decorate import read_base_co, read_co, read_normal
from infinigen.assets.utils.misc import subclasses
from infinigen.assets.utils.object import center, new_cube, origin2lowest
from infinigen.core import init, surface
from infinigen.core.init import configure_cycles_devices
from infinigen.core.placement import factory
from infinigen.core.surface import write_attr_data
from infinigen.core.tagging import tag_system

# noinspection PyUnresolvedReferences
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.camera import points_inview
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.test_utils import load_txt_list
from infinigen_examples.asset_parameters import parameters
from infinigen_examples.generate_individual_assets import make_args, setup_camera

logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,
)


def build_scene_asset(args, factory_name, idx):
    params = parameters[factory_name]["globals"].copy()
    i = idx // parameters[factory_name]["repeat"]
    params.update(parameters[factory_name]["individuals"].copy()[i])
    factory = parameters[factory_name]["factories"][i]
    idx = parameters[factory_name]["indices"][i]
    with FixedSeed(idx):
        factory = factory(idx)
        for k, v in params.items():
            setattr(
                factory,
                k,
                v()
                if isinstance(v, Callable)
                and hasattr(v, "__name__")
                and v.__name__ == "<lambda>"
                else v,
            )
        with FixedSeed(idx):
            if hasattr(factory, "post_init"):
                factory.post_init()
        with FixedSeed(idx):
            try:
                if args.spawn_placeholder:
                    ph = factory.spawn_placeholder(idx, (0, 0, 0), (0, 0, 0))
                    asset = factory.spawn_asset(idx, placeholder=ph)
                else:
                    asset = factory.spawn_asset(idx)
            except Exception as e:
                traceback.print_exc()
                print(f"{factory}.spawn_asset({idx=}) FAILED!! {e}")
                raise e
        with FixedSeed(idx):
            factory.finalize_assets(asset)
        origin2lowest(asset, True)
        bpy.context.view_layer.objects.active = asset
        parent = asset
        if asset.type == "EMPTY":
            meshes = [o for o in asset.children_recursive if o.type == "MESH"]
            sizes = []
            for m in meshes:
                co = read_co(m)
                sizes.append((np.amax(co, 0) - np.amin(co, 0)).sum())
            i = np.argmax(np.array(sizes))
            asset = meshes[i]
        asset.location = -center(asset)
        asset.location[-1] = 0
        butil.apply_transform(asset, True)
    if not args.no_mod:
        if parent.animation_data is not None:
            drivers = parent.animation_data.drivers.values()
            for d in drivers:
                parent.driver_remove(d.data_path)
        if not args.no_ground:
            plane = new_cube()
            plane.scale = [2.5] * 3
            co = read_co(asset)
            plane.location = asset.location
            plane.location[-1] += np.min(co[:, -1]) + 2.5
            butil.apply_transform(plane, True)
            plane_ = deep_clone_obj(plane)
            plane_.location[-1] -= 0.1
            plane_.scale = (1.5,) * 3
            normal = read_normal(plane)
            write_attr_data(plane, "ground", normal[:, -1] < -0.5, "INT", "FACE")
            idx = parameters[factory_name]["scene_idx"]
            with FixedSeed(idx):
                wood_tile.apply(plane, selection="ground")
                non_wood_tile.apply(plane, selection="!ground", vertical=True)
                factory = CeilingLightFactory(0)
                factory.light_factory.params["Wattage"] = (
                    factory.light_factory.params["Wattage"] * 20
                )
                light = factory.spawn_asset(0)
                light.location[-1] = np.min(co[:, -1]) + 5 - 0.5

    return asset


def build_scene(path, idx, factory_name, args):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y = map(
        int, args.resolution.split("x")
    )
    scene.cycles.samples = args.samples
    configure_cycles_devices(True)
    t = idx / (args.frame_end / args.cycles)
    args.cam_angle = (
        args.cam_angle[0],
        args.cam_angle[1],
        (np.abs(t - np.round(t)) * 2) * 180,
    )
    if not args.fire:
        bpy.context.scene.render.film_transparent = args.film_transparent
        bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value[
            -1
        ] = 0

    if idx % parameters[factory_name]["repeat"] == 0:
        butil.clear_scene()
        camera, center = setup_camera(args)
        asset = build_scene_asset(args, factory_name, idx)

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

    else:
        camera, center = setup_camera(args)
        asset = list(
            o for o in bpy.data.objects if "Factory" in o.name and o.parent is None
        )[0]

    if args.scale_reference:
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.3, depth=1.8, location=(4.9, 4.9, 1.8 / 2)
        )

    if args.cam_center > 0 and asset:
        co = read_base_co(asset) + asset.location
        center.location = (np.amin(co, 0) + np.amax(co, 0)) / 2
        center.location[-1] += args.cam_zoff

    if args.cam_dist <= 0 and asset:
        if "Factory" in factory_name:
            adjust_cam_distance(asset, camera, args.margin)
        else:
            adjust_cam_distance(asset, camera, args.margin, 0.75)

    cam_info_ng = bpy.data.node_groups.get("nodegroup_active_cam_info")
    if cam_info_ng is not None:
        cam_info_ng.nodes["Object Info"].inputs["Object"].default_value = camera

    if args.save_blend:
        (path / "scenes").mkdir(exist_ok=True)
        butil.save_blend(f"{path}/scenes/scene_{idx:03d}.blend", autopack=True)
        tag_system.save_tag(f"{path}/MaskTag.json")

    if args.fire:
        bpy.data.worlds["World"].node_tree.nodes["Background.001"].inputs[
            1
        ].default_value = 0.04
        bpy.context.scene.view_settings.exposure = -2

    if args.render == "image":
        (path / "images").mkdir(exist_ok=True)
        imgpath = path / f"images/image_{idx:03d}.png"
        scene.render.filepath = str(imgpath)
        bpy.ops.render.render(write_still=True)
    elif args.render == "video":
        bpy.context.scene.frame_end = args.frame_end
        t = f"frame / {args.frame_end / args.cycles}"
        parent(asset).driver_add("rotation_euler")[
            -1
        ].driver.expression = f"(abs({t}-round({t}))*2-.5)*{np.pi}"
        (path / "frames" / f"scene_{idx:03d}").mkdir(parents=True, exist_ok=True)
        imgpath = path / f"frames/scene_{idx:03d}/frame_###.png"
        scene.render.filepath = str(imgpath)
        bpy.ops.render.render(animation=True)


def parent(obj):
    return obj if obj.parent is None else obj.parent


def adjust_cam_distance(asset, camera, margin, percent=0.999):
    co = read_base_co(asset) * asset.scale
    co += asset.location
    lowest = np.amin(co, 0)
    highest = np.amax(co, 0)
    interp = np.linspace(lowest, highest, 11)
    bbox = np.array(list(product(*zip(*interp))))
    for cam_dist in np.exp(np.linspace(-1.0, 5.5, 500)):
        camera.location[1] = -cam_dist
        bpy.context.view_layer.update()
        inview = points_inview(bbox, camera)
        if inview.sum() / inview.size >= percent:
            camera.location[1] *= 1 + margin
            bpy.context.view_layer.update()
            break
    else:
        camera.location[1] = -6


def make_grid(args, path, n):
    files = []
    for filename in sorted(os.listdir(f"{path}/images")):
        if filename.endswith(".png"):
            files.append(f"{path}/images/{filename}")
    files = files[:n]
    if len(files) == 0:
        print("No images found")
        return
    with Image.open(files[0]) as i:
        x, y = i.size
    for i, name in enumerate([path.stem, f"{path.stem}_"]):
        if args.zoom:
            img = Image.new("RGBA", (2 * x, y))
            sz = int(np.floor(np.sqrt(n - 0.9)))
            if i > 0:
                random.shuffle(files)
            with Image.open(files[0]) as i:
                img.paste(i, (0, 0))
            for idx in range(sz**2):
                with Image.open(files[min(idx + 1, len(files) - 1)]) as i:
                    img.paste(
                        i.resize((x // sz, y // sz)),
                        (x + (idx % sz) * (x // sz), idx // sz * (y // sz)),
                    )
            img.save(f"{path}/{name}.png")
        else:
            sz_x = list(
                sorted(
                    range(1, n + 1),
                    key=lambda x: abs(math.ceil(n / x) / x - args.best_ratio),
                )
            )[0]
            sz_y = math.ceil(n / sz_x)
            img = Image.new("RGBA", (sz_x * x, sz_y * y))
            if i > 0:
                random.shuffle(files)
            for idx, file in enumerate(files):
                with Image.open(file) as i:
                    img.paste(i, (idx % sz_x * x, idx // sz_x * y))
            img.save(f"{path}/{name}.png")


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

    factories = list(args.factories)
    if "ALL_ASSETS" in factories:
        factories += [f.__name__ for f in subclasses(factory.AssetFactory)]
        factories.remove("ALL_ASSETS")
    if "ALL_SCATTERS" in factories:
        factories += [f.stem for f in Path("surfaces/scatters").iterdir()]
        factories.remove("ALL_SCATTERS")
    if "ALL_MATERIALS" in factories:
        factories += [f.stem for f in Path("infinigen/assets/materials").iterdir()]
        factories.remove("ALL_MATERIALS")
    if ".txt" in factories[0]:
        factories = [
            f.split(".")[-1] for f in load_txt_list(factories[0], skip_sharp=False)
        ]

    for fac in factories:
        fac_path = path / fac
        if fac_path.exists() and args.skip_existing:
            continue
        fac_path.mkdir(exist_ok=True)
        n_images = args.n_images
        if not args.postprocessing_only:
            for idx in range(args.n_images):
                try:
                    build_scene(fac_path, idx, fac, args)
                except Exception as e:
                    print(e)
                    continue
        if args.render == "image":
            make_grid(args, fac_path, n_images)
        if args.render == "video":
            (fac_path / "videos").mkdir(exist_ok=True)
            for i in range(n_images):
                subprocess.run(
                    f'ffmpeg -y -r 24 -pattern_type glob -i "{fac_path}/frames/scene_{i:03d}/frame*.png" '
                    f"{fac_path}/videos/video_{i:03d}.mp4",
                    shell=True,
                )


if __name__ == "__main__":
    args = make_args()
    args.no_mod = args.no_mod or args.fire
    args.film_transparent = args.film_transparent and not args.hdri
    args.n_images = (
        len(parameters[args.factories[0]]["factories"])
        * parameters[args.factories[0]]["repeat"]
    )
    with FixedSeed(1):
        main(args)

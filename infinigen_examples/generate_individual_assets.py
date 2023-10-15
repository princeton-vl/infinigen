# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this
# source tree.

# Authors: 
# - Lingjie Mei
# - Alex Raistrick
# - Karhan Kayan - add fire option

import argparse
import importlib
import math
import os
import re
import subprocess
import sys
import traceback
from itertools import product
from pathlib import Path
import logging

logging.basicConfig(
    format='[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] | %(message)s',
    datefmt='%H:%M:%S',
    level=logging.WARNING
)

import bpy
import gin
import numpy as np
from PIL import Image

from infinigen.assets.fluid.fluid import set_obj_on_fire
from infinigen.assets.utils.decorate import assign_material, read_base_co
from infinigen.assets.utils.tag import tag_object, tag_nodegroup, tag_system
from infinigen.assets.lighting import sky_lighting

from infinigen.core import surface, init
from infinigen.core.placement import density, factory 
from infinigen.core.rendering.render import enable_gpu
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.camera import get_3x4_P_matrix_from_blender
from infinigen.core.util.logging import Suppress
from infinigen.core.util import blender as butil

from infinigen.tools.results import strip_alpha_background as strip_alpha_background

from . import generate_nature  # to load most/all factory.AssetFactory subclasses

def build_scene_asset(factory_name, idx):
    factory = None
    for subdir in os.listdir('infinigen/assets'):
        with gin.unlock_config():
            module = importlib.import_module(f'infinigen.assets.{subdir.split(".")[0]}')
        if hasattr(module, factory_name):
            factory = getattr(module, factory_name)
            break
    if factory is None:
        raise ModuleNotFoundError(f'{factory_name} not Found.')
    with FixedSeed(idx):
        factory = factory(idx)
        try:
            asset = factory.spawn_asset(idx)
        except Exception as e:
            traceback.print_exc()
            print(f'{factory}.spawn_asset({idx=}) FAILED!! {e}')
            raise e
        factory.finalize_assets(asset)
        if args.fire:
            set_obj_on_fire(asset,0,resolution = args.fire_res, simulation_duration = args.fire_duration, noise_scale=2, add_turbulence = True, adaptive_domain = False)
            bpy.context.scene.frame_set(args.fire_duration)
            bpy.context.scene.frame_end = args.fire_duration
            bpy.data.worlds['World'].node_tree.nodes["Background.001"].inputs[1].default_value = 0.04
            bpy.context.scene.view_settings.exposure = -1
        bpy.context.view_layer.objects.active = asset
        parent = asset
        if asset.type == 'EMPTY':
            meshes = [o for o in asset.children_recursive if o.type == 'MESH']
            sizes = []
            for m in meshes:
                co = read_base_co(m)
                sizes.append((np.amax(co, 0) - np.amin(co, 0)).sum())
            i = np.argmax(np.array(sizes))
            asset = meshes[i]
        if not args.fire:
            if parent.animation_data is not None:
                drivers = parent.animation_data.drivers.values()
                for d in drivers:
                    parent.driver_remove(d.data_path)
            co = read_base_co(asset)
            x_min, x_max = np.amin(co, 0), np.amax(co, 0)
            parent.location = -(x_min[0] + x_max[0]) / 2, -(x_min[1] + x_max[1]) / 2, 0
            butil.apply_transform(parent, loc=True)
            bpy.ops.mesh.primitive_grid_add(size=5, x_subdivisions=400, y_subdivisions=400)
            plane = bpy.context.active_object
            plane.location[-1] = x_min[-1]
            plane.is_shadow_catcher = True
            material = bpy.data.materials.new('plane')
            material.use_nodes = True
            material.node_tree.nodes['Principled BSDF'].inputs[0].default_value = .015, .009, .003, 1
            assign_material(plane, material)

    return asset


def build_scene_surface(factory_name, idx):
    try:
        with gin.unlock_config():
            scatter = importlib.import_module(f'infinigen.assets.scatters.{factory_name}')

            if not hasattr(scatter, 'apply'):
                raise ValueError(f'{scatter} has no apply()')

            bpy.ops.mesh.primitive_grid_add(size=10, x_subdivisions=400, y_subdivisions=400)
            plane = bpy.context.active_object

            material = bpy.data.materials.new('plane')
            material.use_nodes = True
            material.node_tree.nodes['Principled BSDF'].inputs[0].default_value = .015, .009, .003, 1
            assign_material(plane, material)

            scatter.apply(plane, selection=density.placement_mask(.15, .45))
            asset = plane
    except ModuleNotFoundError:
        try:
            with gin.unlock_config():
                template = importlib.import_module(f'infinigen.assets.materials.{factory_name}')
                bpy.ops.mesh.primitive_ico_sphere_add(radius=.8, subdivisions=9)
                asset = bpy.context.active_object
                template.apply(asset)
        except ModuleNotFoundError:
            raise Exception(f'{factory_name} not Found.')

    return asset


def build_scene(path, idx, factory_name, args):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x, scene.render.resolution_y = map(int, args.resolution.split('x'))
    scene.cycles.samples = args.samples
    butil.clear_scene()

    if not args.fire:
        bpy.context.scene.render.film_transparent = args.film_transparent
        bpy.context.scene.world.node_tree.nodes['Background'].inputs[0].default_value[-1] = 0
    camera, center = setup_camera(args)

    with FixedSeed(args.lighting):
        sky_lighting.add_lighting(camera)
        nodes = bpy.data.worlds['World'].node_tree.nodes
        sky_texture = [n for n in nodes if n.name.startswith('Sky Texture')][-1]
        sky_texture.sun_elevation = np.deg2rad(args.elevation)
        sky_texture.sun_rotation = np.pi * .75

    if 'Factory' in factory_name:
        asset = build_scene_asset(factory_name, idx)
    else:
        asset = build_scene_surface(factory_name, idx)

    if args.scale_reference:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=1.8, location=(4.9, 4.9, 1.8 / 2))

    if args.cam_center > 0 and asset:
        co = read_base_co(asset) + asset.location
        center.location = (np.amin(co, 0) + np.amax(co, 0)) / 2
        center.location[-1] += args.cam_zoff

    if args.cam_dist <= 0 and asset:
        adjust_cam_distance(asset, camera, args.margin)

    cam_info_ng = bpy.data.node_groups.get('nodegroup_active_cam_info')
    if cam_info_ng is not None:
        cam_info_ng.nodes['Object Info'].inputs['Object'].default_value = camera

    if args.save_blend:
        (path / 'scenes').mkdir(exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=f"{path}/scenes/scene_{idx:03d}.blend", filter_backup=True)
        tag_system.save_tag(f"{path}/MaskTag.json")

    if args.fire:
        bpy.data.worlds['World'].node_tree.nodes["Background.001"].inputs[1].default_value = 0.04
        bpy.context.scene.view_settings.exposure = -2

    if args.render == 'image':
        (path / 'images').mkdir(exist_ok=True)
        imgpath = path / f"images/image_{idx:03d}.png"
        scene.render.filepath = str(imgpath)
        bpy.ops.render.render(write_still=True)
    elif args.render == 'video':
        bpy.context.scene.frame_end = args.frame_end
        parent(asset).driver_add('rotation_euler')[
            -1].driver.expression = f"frame/{args.frame_end / (2 * np.pi * args.cycles)}"
        (path / 'frames' / f'scene_{idx:03d}').mkdir(parents=True, exist_ok=True)
        imgpath = path / f"frames/scene_{idx:03d}/frame_###.png"
        scene.render.filepath = str(imgpath)
        bpy.ops.render.render(animation=True)


def parent(obj):
    return obj if obj.parent is None else obj.parent


def adjust_cam_distance(asset, camera, margin):
    co = read_base_co(asset)
    co += asset.location
    lowest = np.amin(co, 0)
    highest = np.amax(co, 0)
    bbox = np.array(list(product(*zip(lowest, highest))))
    render = bpy.context.scene.render
    for cam_dist in np.exp(np.linspace(-.5, 3.5, 100)):
        camera.location[1] = -cam_dist
        bpy.context.view_layer.update()
        proj = np.array(get_3x4_P_matrix_from_blender(camera)[0])
        x, y, z = proj @ np.concatenate([bbox, np.ones((len(bbox), 1))], -1).T
        inview = (np.all(z > 0) and np.all(x >= 0) and np.all(z > 0) and np.all(
            x / z < render.resolution_x) and np.all(y / z < render.resolution_y))
        if inview:
            camera.location[1] *= 1 + margin
            bpy.context.view_layer.update()
            break
    else:
        camera.location[1] = -6


def make_grid(args, path, n):
    files = []
    for filename in sorted(os.listdir(f'{path}/images')):
        if filename.endswith('.png'):
            files.append(f'{path}/images/{filename}')
    files = files[:n]
    if len(files) == 0:
        print('No images found')
        return
    with Image.open(files[0]) as i:
        x, y = i.size
    sz_x = list(sorted(range(1, n + 1), key=lambda x: abs(math.ceil(n / x) / x - args.best_ratio)))[0]
    sz_y = math.ceil(n / sz_x)
    img = Image.new('RGBA', (sz_x * x, sz_y * y))
    for idx, file in enumerate(files):
        with Image.open(file) as i:
            img.paste(i, (idx % sz_x * x, idx // sz_x * y))
    img.save(f'{path}/grid.png')


def setup_camera(args):
    cam_dist = args.cam_dist if args.cam_dist > 0 else 6
    bpy.ops.object.camera_add(location=(0, -cam_dist, 0), rotation=(np.pi / 2, 0, 0))
    camera = bpy.context.active_object
    camera.parent = butil.spawn_empty('Camera parent')
    camera.parent.location = (0, 0, args.cam_zoff)
    camera.parent.rotation_euler = np.deg2rad(np.array(args.cam_angle))
    bpy.data.scenes['Scene'].camera = camera
    scene = bpy.context.scene
    camera.data.sensor_height = camera.data.sensor_width * scene.render.resolution_y / scene.render.resolution_x
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces.active.region_3d.view_perspective = 'CAMERA'
            break
    cam_info_ng = bpy.data.node_groups.get('nodegroup_active_cam_info')
    if cam_info_ng is not None:
        cam_info_ng.nodes['Object Info'].inputs['Object'].default_value = camera
    return camera, camera.parent

    


def subclasses(cls):
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in subclasses(c)])


def main(args):
    bpy.context.window.workspace = bpy.data.workspaces['Geometry Nodes']
    
    init.apply_gin_configs('infinigen_examples/configs')
    surface.registry.initialize_from_gin()

    extras = '[%(filename)s:%(lineno)d] ' if args.loglevel == logging.DEBUG else ''
    logging.basicConfig(
        format=f'[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] {extras}| %(message)s',
        level=args.loglevel,
        datefmt='%H:%M:%S'
    )
    logging.getLogger("infinigen").setLevel(args.loglevel)

    name = '_'.join(args.factories)
    path = Path(os.getcwd()) / 'outputs' / name
    path.mkdir(exist_ok=True)

    if args.gpu:
        enable_gpu()

    factories = list(args.factories)
    if 'ALL_ASSETS' in factories:
        factories += [f.__name__ for f in subclasses(factory.AssetFactory)]
        factories.remove('ALL_ASSETS')
    if 'ALL_SCATTERS' in factories:
        factories += [f.stem for f in Path('surfaces/scatters').iterdir()]
        factories.remove('ALL_SCATTERS')
    if 'ALL_MATERIALS' in factories:
        factories += [f.stem for f in Path('infinigen/assets/materials').iterdir()]
        factories.remove('ALL_MATERIALS')

    for fac in factories:
        fac_path = path / fac
        if fac_path.exists() and args.skip_existing:
            continue
        fac_path.mkdir(exist_ok=True)
        n_images = args.n_images
        if not args.postprocessing_only:
            for idx in range(n_images):
                if args.seed >= 0: idx = args.seed
                build_scene(fac_path, idx, fac, args)
                try:
                    pass
                except Exception as e:
                    print(e)
                    continue
        if args.render == 'image':
            make_grid(args, fac_path, n_images)
        if args.render == 'video':
            (fac_path / 'videos').mkdir(exist_ok=True)
            for i in range(n_images):
                subprocess.run(
                    f'ffmpeg -y -r 24 -pattern_type glob -i "{fac_path}/frames/scene_{i:03d}/frame*.png" '
                    f'{fac_path}/videos/video_{i:03d}.mp4', shell=True)


def snake_case(s):
    return '_'.join(
        re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s.replace('-', ' '))).split()).lower()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--factories', default=[], nargs='+',
                        help="List factories/surface scatters/surface materials you want to render")
    parser.add_argument('-n', '--n_images', default=4, type=int, help="Number of scenes to render")
    parser.add_argument("-m", '--margin', default=.1,
                        help="Margin between the asset the boundary of the image when automatically adjusting "
                             "the camera")
    parser.add_argument('-R', '--resolution', default='1024x1024', type=str,
                        help="Image resolution widthxheight")
    parser.add_argument('-p', '--samples', default=200, type=int, help="Blender cycles samples")
    parser.add_argument('-l', '--lighting', default=0, type=int, help="Lighting seed")
    parser.add_argument('-o', '--cam_zoff', '--z_offset', type=float, default=.0,
                        help="Additional offset on Z axis for camera look-at positions")
    parser.add_argument('-g', '--gpu', action='store_true', help="Whether to use gpu in rendering")
    parser.add_argument('-s', '--save_blend', action='store_true', help="Whether to save .blend file")
    parser.add_argument('-e', '--elevation', default=60, type=float, help="Elevation of the sun")
    parser.add_argument('--cam_dist', default=0, type=float,
                        help="Distance from the camera to the look-at position")
    parser.add_argument('-a', '--cam_angle', default=(-30, 0, 0), type=float, nargs='+',
                        help="Camera rotation in XYZ")
    parser.add_argument('-c', '--cam_center', default=1, type=int, help="Camera rotation in XYZ")
    parser.add_argument('-r', '--render', default='image', type=str,
                        help="Whether to render the scene in images or video")
    parser.add_argument('-b', '--best_ratio', default=9 / 16, type=float,
                        help="Best aspect ratio for compiling the images into asset grid")
    parser.add_argument('-F', '--fire', action = 'store_true')
    parser.add_argument('-I', '--fire_res', default = 100, type = int)
    parser.add_argument('-U', '--fire_duration', default = 30, type = int)
    parser.add_argument('-t', '--film_transparent', default=1, type=int,
                        help="Whether the background is transparent")
    parser.add_argument('-E', '--frame_end', type=int, default=120, help="End of frame in videos")
    parser.add_argument('-C', '--cycles', type=float, default=1, help="render video cycles")
    parser.add_argument('-A', '--scale_reference', action='store_true', help="Add the scale reference")
    parser.add_argument('-S', '--skip_existing', action='store_true', help="Skip existing scenes and renders")
    parser.add_argument('-P', '--postprocessing_only', action='store_true', help="Only run postprocessing")
    parser.add_argument('-D', '--seed', type=int, default=-1, help="Run a specific seed.")
    parser.add_argument('-d', '--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)

    return init.parse_args_blender(parser)

if __name__ == '__main__':
    args = make_args()
    with FixedSeed(1):
        main(args)

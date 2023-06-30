import argparse
import ast
import importlib
import math
import os
import re
import sys
import traceback
from itertools import product
from pathlib import Path

import bpy
import gin
import numpy as np

sys.path.insert(0, os.getcwd())
from PIL import Image
from lighting import lighting
from surfaces import surface
from placement import density
from assets.utils.decorate import assign_material, read_base_co

import generate  # to load most/all AssetFactory subclasses

def build_scene_surface(factory_name, idx):



def build_scene(path, idx, factory_name, args):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    (path / 'images').mkdir(exist_ok=True)
    imgpath = path / f"images/image_{idx:03d}.png"
    scene.render.filepath = str(imgpath)
    scene.render.resolution_x, scene.render.resolution_y = map(int, args.resolution.split('x'))
    scene.cycles.samples = args.samples
    butil.clear_scene()

    bpy.context.scene.render.film_transparent = args.film_transparent
    bpy.context.scene.world.node_tree.nodes['Background'].inputs[0].default_value[-1] = 0
    camera, center = setup_camera(args)

    with FixedSeed(args.lighting):
        lighting.add_lighting(camera)
        nodes = bpy.data.worlds['World'].node_tree.nodes
        sky_texture = [n for n in nodes if n.name.startswith('Sky Texture')][-1]
        sky_texture.sun_elevation = np.deg2rad(args.elevation)
        sky_texture.sun_rotation = np.pi * .75

    if 'Factory' in factory_name:
    else:

        bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=1.8, location=(4.9, 4.9, 1.8 / 2))

    if args.cam_center > 0 and asset:
        co = read_base_co(asset)
        center.location = (np.amin(co, 0) + np.amax(co, 0)) / 2
        center.location[-1] += args.cam_zoff
    if args.cam_dist <= 0 and asset:
        adjust_cam_distance(asset, camera, args.margin)

    cam_info_ng = bpy.data.node_groups.get('nodegroup_active_cam_info')
    if cam_info_ng is not None:
        cam_info_ng.nodes['Object Info'].inputs['Object'].default_value = camera

        (path / 'scenes').mkdir(exist_ok=True)
    if args.render:


def adjust_cam_distance(asset, camera, margin):
    co = read_base_co(asset)
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


def import_surface_registry():
    def find_config(g):
        as_scene_type = f'config/scene_types/{g}.gin'
        if os.path.exists(as_scene_type):
            return as_scene_type
        as_base = f'config/{g}.gin'
        if os.path.exists(as_base):
            return as_base
        raise ValueError(f'Couldn not locate {g} in either config/ or config/scene_types')

    def sanitize_gin_override(overrides: list):
        if len(overrides) > 0:
            print("Overriden parameters:", overrides)
        output = list(overrides)
        for i, o in enumerate(overrides):
            if ('=' in o) and not any((c in o) for c in "\"'[]"):
                k, v = o.split('=')
                try:
                    ast.literal_eval(v)
                except:
                    output[i] = f'{k}="{v}"'
        return output

    gin.parse_config_files_and_bindings(
        ['config/base.gin'] + [find_config(g) for g in ['base_surface_registry']],
        bindings=sanitize_gin_override([]), skip_unknown=True)
    surface.registry.initialize_from_gin()


def main(args):
    bpy.context.window.workspace = bpy.data.workspaces['Geometry Nodes']
    import_surface_registry()
    path = Path(os.getcwd()) / 'outputs' / name
    path.mkdir(exist_ok=True)

    if args.gpu:
        enable_gpu()


        fac_path = path / factory


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--factories', default=[], nargs='+',
                        help="List factories/surface scatters/surface materials you want to render")
    parser.add_argument('-n', '--n_images', default=4, type=int, help="Number of scenes to render")
    parser.add_argument("-m", '--margin', default=.1,
                        help="Margin between the asset the boundary of the image when automatically adjusting "
                             "the camera")
    parser.add_argument('-r', '--resolution', default='1024x1024', type=str,
                        help="Image resolution widthxheight")
    parser.add_argument('-p', '--samples', default=200, type=int, help="Blender cycles samples")
    parser.add_argument('-l', '--lighting', default=0, type=int, help="Lighting seed")
    parser.add_argument('-o', '--cam_zoff', '--z_offset', type=float, default=.0,
                        help="Additional offset on Z axis for camera look-at positions")
    parser.add_argument('-g', '--gpu', action='store_true', help="Whether to use gpu in rendering")
    parser.add_argument('-s', '--save_blend', action='store_true', help="Whether to save .blend file")
    parser.add_argument('-e', '--elevation', default=60, type=float, help="Elevation of the sun")
    parser.add_argument('-d', '--cam_dist', default=0, type=float,
                        help="Distance from the camera to the look-at position")
    parser.add_argument('-a', '--cam_angle', default=(-30, 0, 0), type=float, nargs='+',
                        help="Camera rotation in XYZ")
    parser.add_argument('-c', '--cam_center', default=1, type=int,
                        help="Whether the camera look-at is at the center of the asset")
    parser.add_argument('-x', '--render', action='store_false', help="Whether to render the scene")
    parser.add_argument('-b', '--best_ratio', default=9 / 16, type=float,
                        help="Best aspect ratio for compiling the images into asset grid")
    parser.add_argument('-t', '--film_transparent', default=1, type=int)
    parser.add_argument('--scale_reference', action='store_true', help="Add the scale reference")
    parser.add_argument('--skip_existing', action='store_true', help="Skip existing scenes and renders")
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    return args


if __name__ == '__main__':
    args = make_args()
    with FixedSeed(1):

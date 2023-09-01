# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import argparse
import os
import sys
from pathlib import Path
import logging
from copy import copy

logging.basicConfig(
    format='[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] | %(message)s',
    datefmt='%H:%M:%S',
    level=logging.WARNING
)

import bpy
from mathutils import Vector, Matrix, bvhtree
import gin
import numpy as np
from tqdm import tqdm, trange


from infinigen.terrain import Terrain
from infinigen.assets.small_plants.fern import FernFactory
from infinigen.assets.creatures.util.animation.run_cycle import follow_path
from infinigen.assets.lighting import sky_lighting
from infinigen.assets.weather import kole_clouds
from infinigen.assets.scatters import grass, pebbles, pine_needle, pinecone
from infinigen.assets.materials import (
    mountain, sand, water, atmosphere_light_haze, sandstone, cracked_ground, \
    soil, dirt, cobble_stone, chunkyrock, stone, lava, ice, mud, snow
)
from infinigen.core.placement import placement, density, camera as cam_util
from infinigen.core.placement.split_in_view import split_inview
from infinigen.core.util import blender as butil

from infinigen.core import execute_tasks, init, surface

def find_flat_location(
   mesh, 
   bvh: bvhtree.BVHTree, 
   rad: float, 
   alt: float,
   retries=100, margin_pct=0.2, 
   ang_samples=36
):

    for i in trange(retries):
        
        ground_loc = copy(np.random.choice(mesh.data.vertices).co)
        origin = ground_loc + Vector((0, 0, alt))

        *_, center_alt = bvh.ray_cast(origin, Vector((0, 0, -1)))
        if center_alt is None:
            continue

        for j, ang in enumerate(np.linspace(0, 2*np.pi, ang_samples)):
            sample_loc = origin + Matrix.Rotation(ang, 4, 'Z') @ Vector((rad, 0, 0))

            *_, dist = bvh.ray_cast(origin, sample_loc - origin)
            if dist is not None and dist < rad:
                break

            *_, sample_alt = bvh.ray_cast(sample_loc, Vector((0, 0, -1)))
            if sample_alt is None or abs(center_alt - sample_alt) > margin_pct * center_alt:
                break

        else: # triggered if no `break` statement 
            return ground_loc
        
    raise ValueError(f'Failed to find flat area {retries=}')

def circular_camera_path(camera_rig, target_obj, rad, alt, duration):
    
    bpy.ops.curve.primitive_bezier_circle_add(
        location=target_obj.location + Vector((0, 0, alt)),
    )
    circle = bpy.context.object
    circle.scale = (rad,) * 3

    follow_path(camera_rig, circle, duration=duration)
    circle.data.driver_add('eval_time').driver.expression = 'frame'

    butil.constrain_object(camera_rig, 'TRACK_TO', target=target_obj)

@gin.configurable
def compose_scene(
    output_folder: Path, 
    scene_seed: int,

    asset_factory=None, # provided via gin
    grid_rad=1.2, 
    grid_dim=3, #NxN grid
    background='grass',
    camera_circle_radius=8,
    camera_altitude=2,
    circle_duration_sec=25,
    fstop=2,

    asset_scale=(1,1,1),
    asset_offset=(0,0,0),

    **params
):

    sky_lighting.add_lighting()

    if params.get("fancy_clouds", 0):
        kole_clouds.add_kole_clouds()
    
    camera_rigs = cam_util.spawn_camera_rigs()
    cam = cam_util.get_camera(0, 0)

    # find a flat spot on the terrain to do the demo\
    terrain = Terrain(scene_seed, surface.registry, task='coarse', on_the_fly_asset_folder=output_folder/"assets")
    terrain_mesh = terrain.coarse_terrain()
    terrain_bvh = bvhtree.BVHTree.FromObject(terrain_mesh, bpy.context.evaluated_depsgraph_get())
    if asset_factory is not None:
        center = find_flat_location(
            terrain_mesh, 
            terrain_bvh, 
            rad=camera_circle_radius * 1.5, 
            alt=camera_altitude * 1.5
        )
    else:
        center = (0, 0, 0)
    # move camera in a circle around that location
    center_obj = butil.spawn_empty('center')
    center_obj.location = center
    circular_camera_path(camera_rigs[0], center_obj, 
                         camera_circle_radius, camera_altitude,
                         duration=circle_duration_sec*bpy.context.scene.render.fps)
    cam.data.dof.use_dof = True
    cam.data.dof.aperture_fstop = fstop
    cam.data.dof.focus_object = center_obj

    # make a grid of locations around the center point
    offs = np.linspace(-grid_rad, grid_rad, grid_dim)
    xs, ys = np.meshgrid(offs, offs)
    locs = np.stack([xs, ys, np.full_like(ys, camera_altitude)], axis=-1).reshape(-1, 3)
    locs += np.array(center)

    # snap all the locations the floor
    for i, l in enumerate(locs):
        floorloc, *_ = terrain_bvh.ray_cast(Vector(l), Vector((0, 0, -1)))
        if floorloc is None:
            raise ValueError('Found a hole in the terain')
        locs[i] = np.array(floorloc + Vector(asset_offset))

    if asset_factory is not None:
        # spawn assets on each location in the grid
        fac = asset_factory(scene_seed)
        col = placement.scatter_placeholders(locs, fac)
        objs, updated_pholders = placement.populate_collection(fac, col)

        for _, o in updated_pholders:
            o.scale = asset_scale

    # apply a procedural backdrop on all visible parts of the terrain
    terrain_inview, *_ = split_inview(terrain_mesh, cam=cam, dist_max=params['inview_distance'], vis_margin=2)
    if background is None:
        pass
    elif background == 'grass':
        grass.apply(terrain_inview)
        pebbles.apply(terrain_inview)
    elif background == 'pine_forest':
        pine_needle.apply(terrain_inview)
        pinecone.apply(terrain_inview)
        pebbles.apply(terrain_inview)
    elif background == 'TODO ADD MORE OPTIONS HERE':
        pass
    else:
        raise ValueError(f'Unrecognized {background=}')

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=Path, required=True)
    parser.add_argument('--input_folder', type=Path, default=None)
    parser.add_argument('-s', '--seed', default=None, help="The seed used to generate the scene")
    parser.add_argument('-t', '--task', nargs='+', default=['coarse'],
                        choices=['coarse', 'populate', 'fine_terrain', 'ground_truth', 'render', 'mesh_save'])
    parser.add_argument('-g', '--configs', nargs='+', default=['base'],
                        help='Set of config files for gin (separated by spaces) '
                             'e.g. --configs file1 file2 (exclude .gin from path)')
    parser.add_argument('-p', '--overrides', nargs='+', default=[],
                        help='Parameter settings that override config defaults '
                             'e.g. --overrides module_1.a=2 module_2.b=3')
    parser.add_argument('--task_uniqname', type=str, default=None)
    parser.add_argument('-d', '--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument( '-v', '--verbose', action="store_const", dest="loglevel", const=logging.INFO)

    args = init.parse_args_blender(parser)

    extras = '[%(filename)s:%(lineno)d] ' if args.loglevel == logging.DEBUG else ''
    logging.basicConfig(
        format=f'[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] {extras}| %(message)s',
        level=args.loglevel,
        datefmt='%H:%M:%S'
    )
    logging.getLogger("infinigen").setLevel(args.loglevel)

    scene_seed = init.apply_scene_seed(args.seed, task=args.task)
    init.apply_gin_configs(
        configs=args.configs,
        overrides=args.overrides,
        configs_folder='infinigen_examples/configs', 
        mandatory_folders=['infinigen_examples/configs/scene_types'], 
        skip_unknown=True
    )
    
    execute_tasks.main(
        compose_scene_func=compose_scene,
        input_folder=args.input_folder, 
        output_folder=args.output_folder, 
        task=args.task, 
        task_uniqname=args.task_uniqname, 
        scene_seed=scene_seed
    )

if __name__ == "__main__":
    main()
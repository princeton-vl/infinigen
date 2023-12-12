# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import argparse
import ast
import os
import random
import sys
import cProfile
import shutil
from pathlib import Path
import logging
from functools import partial
import pprint
import time
from collections import defaultdict

# ruff: noqa: F402
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # This must be done BEFORE import cv2. 
# See https://github.com/opencv/opencv/issues/21326#issuecomment-1008517425

import bpy
import mathutils
from mathutils import Vector
import gin
import numpy as np
from numpy.random import uniform, normal, randint
from tqdm import tqdm
from frozendict import frozendict

from infinigen.terrain import Terrain

from infinigen.core.placement import (
    particles, placement, density, 
    camera as cam_util, 
    split_in_view, 
    factory,
    animation_policy, 
    instance_scatter, 
    detail,
)

from infinigen.assets.scatters import (
    pebbles, grass, snow_layer, ground_leaves, ground_twigs, \
    chopped_trees, pinecone, fern, flowerplant, monocot, ground_mushroom, \
    slime_mold, moss, ivy, lichen, mushroom, decorative_plants, seashells, \
    pine_needle, seaweed, coral_reef, jellyfish, urchin
)

from infinigen.assets.materials import (
    mountain, sand, water, atmosphere_light_haze, sandstone, cracked_ground, \
    soil, dirt, cobble_stone, chunkyrock, stone, lava, ice, mud, snow
)

from infinigen.assets import (
    fluid, 
    cactus, 
    trees, 
    rocks, 
    creatures, 
    lighting,
    weather,
)

from infinigen.core.rendering.render import render_image
from infinigen.core.rendering.resample import resample_scene
from infinigen.assets.monocot import kelp
from infinigen.core import surface, init

from infinigen.core.util.organization import Task, Attributes, TerrainNames

from infinigen.core.placement.split_in_view import split_inview

import infinigen.assets.scatters
from infinigen.assets.scatters.utils.selection import scatter_lower, scatter_upward

from infinigen.core.util import (
    blender as butil,
    logging as logging_util,
    pipeline, 
    exporting
)

from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.core.util.logging import Timer, save_polycounts, create_text_file
from infinigen.core.util.pipeline import RandomStageExecutor
from infinigen.core.util.random import sample_registry
from infinigen.assets.utils.tag import tag_system
   
logger = logging.getLogger(__name__)

@gin.configurable
def populate_scene(
    output_folder, 
    scene_seed, 
    **params
):
    p = RandomStageExecutor(scene_seed, output_folder, params)
    camera = bpy.context.scene.camera

    season = p.run_stage('choose_season', trees.random_season, use_chance=False, default=[])

    fire_cache_system = fluid.FireCachingSystem() if params.get('cached_fire') else None

    populated = {}
    populated['trees'] = p.run_stage('populate_trees', use_chance=False, default=[],
        fn=lambda: placement.populate_all(trees.TreeFactory, camera, season=season, vis_cull=4))#,
                                        #meshing_camera=camera, adapt_mesh_method='subdivide', cam_meshing_max_dist=8)) 
    populated['boulders'] = p.run_stage('populate_boulders', use_chance=False, default=[],
        fn=lambda: placement.populate_all(rocks.BoulderFactory, camera, vis_cull=3))#,
                                        #meshing_camera=camera, adapt_mesh_method='subdivide', cam_meshing_max_dist=8))
    populated['bushes'] = p.run_stage('populate_bushes', use_chance=False,
        fn=lambda: placement.populate_all(trees.BushFactory, camera, vis_cull=1, adapt_mesh_method='subdivide'))
    p.run_stage('populate_kelp', use_chance=False,
        fn=lambda: placement.populate_all(kelp.KelpMonocotFactory, camera, vis_cull=5))
    populated['cactus'] = p.run_stage('populate_cactus', use_chance=False,
        fn=lambda: placement.populate_all(cactus.CactusFactory, camera, vis_cull=6))
    p.run_stage('populate_clouds', use_chance=False,
        fn=lambda: placement.populate_all(weather.CloudFactory, camera, dist_cull=None, vis_cull=None))
    p.run_stage('populate_glowing_rocks', use_chance=False,
        fn=lambda: placement.populate_all(lighting.GlowingRocksFactory, camera, dist_cull=None, vis_cull=None))
    
    populated['cached_fire_trees'] = p.run_stage('populate_cached_fire_trees', use_chance=False, default=[],
        fn=lambda: placement.populate_all(fluid.CachedTreeFactory, camera, season=season, vis_cull=4, dist_cull=70, cache_system=fire_cache_system))
    populated['cached_fire_boulders'] = p.run_stage('populate_cached_fire_boulders', use_chance=False, default=[],
        fn=lambda: placement.populate_all(fluid.CachedBoulderFactory, camera, vis_cull=3, dist_cull=70, cache_system=fire_cache_system))
    populated['cached_fire_bushes'] = p.run_stage('populate_cached_fire_bushes', use_chance=False,
        fn=lambda: placement.populate_all(fluid.CachedBushFactory, camera, vis_cull=1, adapt_mesh_method='subdivide', cache_system=fire_cache_system))
    populated['cached_fire_cactus'] = p.run_stage('populate_cached_fire_cactus', use_chance=False,
        fn=lambda: placement.populate_all(fluid.CachedCactusFactory, camera, vis_cull=6, cache_system=fire_cache_system))
    
    grime_selection_funcs = {
        'trees': scatter_lower,
        'boulders': scatter_upward,
    }
    grime_types = {
        'slime_mold': slime_mold.SlimeMold,
        'lichen': lichen.Lichen,
        'ivy': ivy.Ivy,
        'mushroom': ground_mushroom.Mushrooms,
        'moss': moss.MossCover
    }
    def apply_grime(grime_type, surface_cls):
        surface_fac = surface_cls()
        for target_type, results, in populated.items():
            selection_func = grime_selection_funcs.get(target_type, None)
            for fac_seed, fac_pholders, fac_assets in results:
                if len(fac_pholders) == 0:
                    continue
                for inst_seed, obj in fac_assets:
                    with FixedSeed(int_hash((grime_type, fac_seed, inst_seed))):
                        p_k = f'{grime_type}_on_{target_type}_per_instance_chance'
                        if uniform() > params.get(p_k, 0.4):
                            continue
                        logger.debug(f'Applying {surface_fac} on {obj}')
                        surface_fac.apply(obj, selection=selection_func)
    for grime_type, surface_cls in grime_types.items():
        p.run_stage(grime_type, lambda: apply_grime(grime_type, surface_cls))

    def apply_snow_layer(surface_cls):
        surface_fac = surface_cls()
        for target_type, results, in populated.items():
            selection_func = grime_selection_funcs.get(target_type, None)
            for fac_seed, fac_pholders, fac_assets in results:
                if len(fac_pholders) == 0:
                    continue
                for inst_seed, obj in fac_assets:
                    tmp = obj.users_collection[0].hide_viewport
                    obj.users_collection[0].hide_viewport = False
                    surface_fac.apply(obj, selection=selection_func)
                    obj.users_collection[0].hide_viewport = tmp
    p.run_stage("snow_layer", lambda: apply_snow_layer(snow_layer.Snowlayer))

    creature_facs = {
        'beetles': creatures.BeetleFactory, 
        'bird': creatures.BirdFactory, 
        'carnivore': creatures.CarnivoreFactory, 
        'crab': creatures.CrabFactory, 
        'crustacean': creatures.CrustaceanFactory,
        'dragonfly': creatures.DragonflyFactory,
        'fish': creatures.FishFactory, 
        'flyingbird': creatures.FlyingBirdFactory, 
        'herbivore': creatures.HerbivoreFactory,
        'snake': creatures.SnakeFactory,
    }
    for k, fac in creature_facs.items():
        p.run_stage(f'populate_{k}', use_chance=False,
            fn=lambda: placement.populate_all(fac, camera=None))
        
    
    fire_warmup = params.get('fire_warmup', 50)
    simulation_duration = bpy.context.scene.frame_end - bpy.context.scene.frame_start + fire_warmup
    
    def set_fire(assets):
        objs = [o for *_, a in assets for _, o in a]
        with butil.EnableParentCollections(objs):
            fluid.set_fire_to_assets(
                assets, 
                bpy.context.scene.frame_start-fire_warmup, 
                simulation_duration, 
                output_folder
            )

    p.run_stage('trees_fire_on_the_fly', set_fire, populated['trees'], prereq='populate_trees')
    p.run_stage('bushes_fire_on_the_fly', set_fire, populated['bushes'], prereq='populate_bushes')   
    p.run_stage('boulders_fire_on_the_fly', set_fire, populated['boulders'], prereq='populate_boulders')
    p.run_stage('cactus_fire_on_the_fly', set_fire, populated['cactus'], prereq='populate_cactus')

    p.save_results(output_folder/'pipeline_fine.csv')

def get_scene_tag(name):
    try:
        o = next(o for o in bpy.data.objects if o.name.startswith(f'{name}='))
        return o.name.split('=')[-1].strip('\'\"')
    except StopIteration:
        return None

@gin.configurable
def render(scene_seed, output_folder, camera_id, render_image_func=render_image, resample_idx=None, hide_water = False):
    if hide_water and "water_fine" in bpy.data.objects:
        logger.info("Hiding water fine")
        bpy.data.objects["water_fine"].hide_render = True
        bpy.data.objects['water_fine'].hide_viewport = True
    if resample_idx is not None and resample_idx != 0:
        resample_scene(int_hash((scene_seed, resample_idx)))
    with Timer('Render Frames'):
        render_image_func(frames_folder=Path(output_folder), camera_id=camera_id)

@gin.configurable
def save_meshes(scene_seed, output_folder, frame_range, resample_idx=False):
    
    if resample_idx is not None and resample_idx > 0:
        resample_scene(int_hash((scene_seed, resample_idx)))

    for obj in bpy.data.objects:
        obj.hide_viewport = obj.hide_render

    for col in bpy.data.collections:
        col.hide_viewport = col.hide_render

    previous_frame_mesh_id_mapping = frozendict()
    current_frame_mesh_id_mapping = defaultdict(dict)
    for frame_idx in range(int(frame_range[0]), int(frame_range[1]+2)):

        bpy.context.scene.frame_set(frame_idx)
        bpy.context.view_layer.update()
        frame_info_folder = Path(output_folder) / f"frame_{frame_idx:04d}"
        frame_info_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Working on frame {frame_idx}")

        exporting.save_obj_and_instances(
            frame_info_folder / "mesh", 
            previous_frame_mesh_id_mapping, 
            current_frame_mesh_id_mapping
        )
        cam_util.save_camera_parameters(
            camera_ids=cam_util.get_cameras_ids(),
            output_folder=frame_info_folder / "cameras", 
            frame=frame_idx
        )
        previous_frame_mesh_id_mapping = frozendict(current_frame_mesh_id_mapping)
        current_frame_mesh_id_mapping.clear()

def validate_version(scene_version):
    if scene_version is None or scene_version.split('.')[:-1] != infinigen.__version__.split('.')[:-1]:
        raise ValueError(
            f'infinigen_examples/generate_nature.py {infinigen.__version__=} attempted to load a scene created by version {scene_version=}')
    if scene_version != infinigen.__version__:
        logger.warning(f'{infinigen.__version__=} has minor version mismatch with {scene_version=}')

@gin.configurable
def group_collections(config):
    for config in config: # Group collections before fine runs
        butil.group_in_collection([o for o in bpy.data.objects if o.name.startswith(f'{config["name"]}:')], config["name"])
        butil.group_toplevel_collections(config['name'], hide_viewport=config['hide_viewport'], hide_render=config['hide_render'])

@gin.configurable
def execute_tasks(
    compose_scene_func,
    input_folder, output_folder,
    task, scene_seed,
    frame_range, camera_id,
    resample_idx=None,
    output_blend_name="scene.blend",
    generate_resolution=(1280,720),
    reset_assets=True,
    focal_length=None,
    dryrun=False,
    optimize_terrain_diskusage=False,
):
    if input_folder != output_folder:
        if reset_assets:
            if os.path.islink(output_folder/"assets"):
                os.unlink(output_folder/"assets")
            elif (output_folder/"assets").exists():
                shutil.rmtree(output_folder/"assets")
        if (not os.path.islink(output_folder/"assets")) and (not (output_folder/"assets").exists()) and input_folder is not None and (input_folder/"assets").exists():
            os.symlink(input_folder/"assets", output_folder/"assets")
            # in this way, even coarse task can have input_folder to have pregenerated on-the-fly assets (e.g., in last run) to speed up developing

    if dryrun:
        time.sleep(15)
        return

    if Task.Coarse not in task and task != Task.FineTerrain:
        with Timer('Reading input blendfile'):
            bpy.ops.wm.open_mainfile(filepath=str(input_folder / 'scene.blend'))
            tag_system.load_tag(path=str(input_folder / "MaskTag.json"))
        butil.approve_all_drivers()
    
    if frame_range[1] < frame_range[0]:
        raise ValueError(f'{frame_range=} is invalid, frame range must be nonempty. Blender end frame is INCLUSIVE')

    logger.info(f'Processing frames {frame_range[0]} through {frame_range[1]} inclusive')
    bpy.context.scene.frame_start = int(frame_range[0])
    bpy.context.scene.frame_end = int(frame_range[1])
    bpy.context.scene.frame_set(int(frame_range[0]))
    bpy.context.scene.render.resolution_x = generate_resolution[0]
    bpy.context.scene.render.resolution_y = generate_resolution[1]
    bpy.context.view_layer.update()

    surface.registry.initialize_from_gin()
    init.configure_blender()
    
    if Task.Coarse in task:
        butil.clear_scene(targets=[bpy.data.objects])
        butil.spawn_empty(f'{infinigen.__version__=}')
        compose_scene_func(output_folder, scene_seed)

    camera = cam_util.set_active_camera(*camera_id)
    if focal_length is not None:
        camera.data.lens = focal_length

    group_collections()

    if Task.Populate in task:
        populate_scene(output_folder, scene_seed)

    if Task.FineTerrain in task:
        terrain = Terrain(scene_seed, surface.registry, task=task, on_the_fly_asset_folder=output_folder/"assets")
        terrain.fine_terrain(output_folder, optimize_terrain_diskusage=optimize_terrain_diskusage)

    group_collections()

    if input_folder is not None and input_folder != output_folder:
        for mesh in os.listdir(input_folder):
            if (mesh.endswith(".glb") or mesh.endswith(".b_displacement.npy")) and not os.path.islink(output_folder / mesh):
                os.symlink(input_folder / mesh, output_folder / mesh)
    if Task.Coarse in task or Task.Populate in task or Task.FineTerrain in task:

        with Timer(f'Writing output blendfile'):
            logging.info(f'Writing output blendfile to {output_folder / output_blend_name}')
            if optimize_terrain_diskusage and task == [Task.FineTerrain]: 
                os.symlink(input_folder / output_blend_name, output_folder / output_blend_name)
            else: 
                bpy.ops.wm.save_mainfile(filepath=str(output_folder / output_blend_name))
        
        tag_system.save_tag(path=str(output_folder / "MaskTag.json"))

        with (output_folder/ "version.txt").open('w') as f:
            f.write(f"{infinigen.__version__}\n")

        with (output_folder/'polycounts.txt').open('w') as f:
            save_polycounts(f)

    for col in bpy.data.collections['unique_assets'].children:
        col.hide_viewport = False

    if Task.Render in task or Task.GroundTruth in task or Task.MeshSave in task:
        terrain = Terrain(
            scene_seed, 
            surface.registry, 
            task=task,
            on_the_fly_asset_folder=output_folder/"assets"
        )
        if optimize_terrain_diskusage:
            terrain.load_glb(output_folder)

    if Task.Render in task or Task.GroundTruth in task:
        render(
            scene_seed, 
            output_folder=output_folder, 
            camera_id=camera_id, 
            resample_idx=resample_idx
        )

    if Task.MeshSave in task:
        save_meshes(
            scene_seed, 
            output_folder=output_folder, 
            frame_range=frame_range, 
        )

def main(
    input_folder, 
    output_folder,
    scene_seed,
    task, 
    task_uniqname,
    **kwargs
):
    
    version_req = ['3.6.0']
    assert bpy.app.version_string in version_req, f'You are using blender={bpy.app.version_string} which is ' \
                                                  f'not supported. Please use {version_req}'
    logger.info(f'infinigen version {infinigen.__version__}')
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if input_folder is not None:
        input_folder = Path(input_folder).absolute()
    output_folder = Path(output_folder).absolute()
    output_folder.mkdir(exist_ok=True, parents=True)

    if task_uniqname is not None:
        create_text_file(filename=f"START_{task_uniqname}")

    with Timer('MAIN TOTAL'):
        execute_tasks(
            input_folder=input_folder, output_folder=output_folder,
            task=task, scene_seed=scene_seed, **kwargs
        )

    if task_uniqname is not None:
        create_text_file(filename=f"FINISH_{task_uniqname}")
        create_text_file(filename=f"operative_gin_{task_uniqname}.txt", text=gin.operative_config_str())

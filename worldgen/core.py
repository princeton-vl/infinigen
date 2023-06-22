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
from collections import defaultdict

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # This must be done BEFORE import cv2.
# See https://github.com/opencv/opencv/issues/21326#issuecomment-1008517425

sys.path.append(os.path.split(os.path.abspath(__file__))[0])

import bpy
import mathutils
from mathutils import Vector
import gin
import numpy as np
from numpy.random import uniform, normal, randint
from tqdm import tqdm
from frozendict import frozendict

from terrain import Terrain
from util.organization import Task, Attributes, TerrainNames

from placement import placement, density, camera as cam_util
from placement.split_in_view import split_inview
from lighting import lighting, kole_clouds

from assets.trees.generate import TreeFactory, BushFactory, random_season, random_leaf_collection
from assets import boulder
from assets.glowing_rocks import GlowingRocksFactory
from assets.creatures import CarnivoreFactory, HerbivoreFactory, FishFactory, FishSchoolFactory, \
    BeetleFactory, AntSwarmFactory, BirdFactory, SnakeFactory, \
    CrustaceanFactory, FlyingBirdFactory, CrabFactory, LobsterFactory, SpinyLobsterFactory
from assets.insects.assembled.dragonfly import DragonflyFactory
from assets.cloud.generate import CloudFactory
from assets.cactus import CactusFactory
from assets.creatures import boid_swarm

from placement import placement, camera as cam_util
from assets.utils.misc import log_uniform

from rendering.render import render_image
from rendering.resample import resample_scene
from assets.monocot import kelp
from surfaces import surface

import surfaces.scatters
from surfaces.scatters import ground_mushroom, slime_mold, moss, ivy, lichen, snow_layer
from surfaces.scatters.utils.selection import scatter_lower, scatter_upward


from placement.factory import make_asset_collection
from util import blender as butil
from util import exporting
from util.logging import Timer, save_polycounts, create_text_file
from util.math import FixedSeed, int_hash
from util.pipeline import RandomStageExecutor
from util.random import sample_registry

from assets.utils.tag import tag_system

VERSION = '1.0.0'

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
                if "@" not in v:
                    output[i] = f'{k}="{v}"'
    return output
   
@gin.configurable
def populate_scene(
    output_folder, terrain, scene_seed, **params
):
    p = RandomStageExecutor(scene_seed, output_folder, params)
    camera = bpy.context.scene.camera

    season = p.run_stage('choose_season', random_season, use_chance=False, default=[])

    populated = {}
    populated['trees'] = p.run_stage('populate_trees', use_chance=False, default=[],
        fn=lambda: placement.populate_all(TreeFactory, camera, season=season, vis_cull=4, dist_cull=70))#,
                                        #meshing_camera=camera, adapt_mesh_method='subdivide', cam_meshing_max_dist=8)) 
    populated['boulders'] = p.run_stage('populate_boulders', use_chance=False, default=[],
        fn=lambda: placement.populate_all(boulder.BoulderFactory, camera, vis_cull=3, dist_cull=70))#,
                                        #meshing_camera=camera, adapt_mesh_method='subdivide', cam_meshing_max_dist=8))
    p.run_stage('populate_bushes', use_chance=False,
        fn=lambda: placement.populate_all(BushFactory, camera, vis_cull=1, adapt_mesh_method='subdivide'))
    p.run_stage('populate_kelp', use_chance=False,
        fn=lambda: placement.populate_all(kelp.KelpMonocotFactory, camera, vis_cull=5))
    p.run_stage('populate_cactus', use_chance=False,
        fn=lambda: placement.populate_all(CactusFactory, camera, vis_cull=6))
    p.run_stage('populate_clouds', use_chance=False,
        fn=lambda: placement.populate_all(CloudFactory, camera, dist_cull=None, vis_cull=None))
    p.run_stage('populate_glowing_rocks', use_chance=False,
        fn=lambda: placement.populate_all(GlowingRocksFactory, camera, dist_cull=None, vis_cull=None))
    
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
                        logging.debug(f'Applying {surface_fac} on {obj}')
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
        'carnivore': CarnivoreFactory, 'herbivore': HerbivoreFactory,
        'bird': BirdFactory, 'fish': FishFactory, 'snake': SnakeFactory,
        'beetles': BeetleFactory, 
        'flyingbird': FlyingBirdFactory, 'dragonfly': DragonflyFactory,
        'crab': CrabFactory, 'crustacean': CrustaceanFactory
    }
    for k, fac in creature_facs.items():
        p.run_stage(f'populate_{k}', use_chance=False,
            fn=lambda: placement.populate_all(fac, camera=None))

    p.save_results(output_folder/'pipeline_fine.csv')

def get_scene_tag(name):
    try:
        o = next(o for o in bpy.data.objects if o.name.startswith(f'{name}='))
        return o.name.split('=')[-1].strip('\'\"')
    except StopIteration:
        return None

@gin.configurable
def render(scene_seed, output_folder, camera_id, render_image_func=render_image, resample_idx=None):
    if resample_idx is not None and resample_idx != 0:
        resample_scene(int_hash((scene_seed, resample_idx)))
    with Timer('Render Frames'):
        render_image_func(frames_folder=Path(output_folder), camera_id=camera_id)

@gin.configurable
def save_meshes(scene_seed, output_folder, frame_range, camera_id, resample_idx=False):
    
    if resample_idx is not None and resample_idx > 0:
        resample_scene(int_hash((scene_seed, resample_idx)))

    for obj in bpy.data.objects:
        obj.hide_viewport = obj.hide_render

    for col in bpy.data.collections:
        col.hide_viewport = col.hide_render

    camera_rig_id, subcam_id = camera_id
    previous_frame_mesh_id_mapping = frozendict()
    current_frame_mesh_id_mapping = defaultdict(dict)
    for frame_idx in range(int(frame_range[0]), int(frame_range[1]+2)):
        bpy.context.scene.frame_set(frame_idx)
        bpy.context.view_layer.update()
        frame_info_folder = Path(output_folder) / f"frame_{frame_idx:04d}"
        frame_info_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"Working on frame {frame_idx}")
        exporting.save_obj_and_instances(str(frame_info_folder /  "mesh"), previous_frame_mesh_id_mapping, current_frame_mesh_id_mapping)
        cam_util.save_camera_parameters(camera_rig_id, [subcam_id], str(frame_info_folder / "cameras"), frame_idx)
        previous_frame_mesh_id_mapping = frozendict(current_frame_mesh_id_mapping)
        current_frame_mesh_id_mapping.clear()

def validate_version(scene_version):
    if scene_version is None or scene_version.split('.')[:-1] != VERSION.split('.')[:-1]:
        raise ValueError(
            f'generate.py {VERSION=} attempted to load a scene created by version {scene_version=}')
    if scene_version != VERSION:
        logging.warning(f'Worldgen {VERSION=} has minor version mismatch with {scene_version=}')

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
    generate_resolution=(1920,1080),
    reset_assets=True,
    focal_length=None,
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
    if Task.Coarse not in task:
        with Timer('Reading input blendfile'):
            bpy.ops.wm.open_mainfile(filepath=str(input_folder / 'scene.blend'))
            tag_system.load_tag(path=str(input_folder / "MaskTag.json"))
        scene_version = get_scene_tag('VERSION')
        butil.approve_all_drivers()
    
    if frame_range[1] < frame_range[0]:
        raise ValueError(f'{frame_range=} is invalid, frame range must be nonempty. Blender end frame is INCLUSIVE')

    logging.info(f'Processing frames {frame_range[0]} through {frame_range[1]} inclusive')
    bpy.context.scene.frame_start = int(frame_range[0])
    bpy.context.scene.frame_end = int(frame_range[1])
    bpy.context.scene.frame_set(int(frame_range[0]))
    bpy.context.view_layer.update()

    surface.registry.initialize_from_gin()
    bpy.ops.preferences.addon_enable(module='ant_landscape')
    bpy.ops.preferences.addon_enable(module='real_snow')
    bpy.context.preferences.system.scrollback = 0 
    bpy.context.preferences.edit.undo_steps = 0
    bpy.context.scene.render.resolution_x = generate_resolution[0]
    bpy.context.scene.render.resolution_y = generate_resolution[1]
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'

    bpy.context.scene.cycles.volume_step_rate = 0.1
    bpy.context.scene.cycles.volume_preview_step_rate = 0.1
    bpy.context.scene.cycles.volume_max_steps = 32

    if Task.Coarse in task or Task.FineTerrain in task or Task.Fine in task or Task.Populate in task:
        terrain = Terrain(scene_seed, surface.registry, task=task, on_the_fly_asset_folder=output_folder/"assets")

    if Task.Coarse in task:
        butil.clear_scene(targets=[bpy.data.objects])
        butil.spawn_empty(f'{VERSION=}')
        compose_scene_func(output_folder, terrain, scene_seed)

    camera = cam_util.set_active_camera(*camera_id)
    if focal_length is not None:
        camera.data.lens = focal_length

    group_collections()

    if Task.Populate in task:
        populate_scene(output_folder, terrain, scene_seed)

    if Task.Fine in task:
        raise RuntimeError(f'{task=} contains deprecated {Task.Fine=}')

    if Task.FineTerrain in task:
        terrain.fine_terrain(output_folder)
    
    group_collections()

    if Task.Coarse in task or Task.Populate in task or Task.FineTerrain in task:
        bpy.context.preferences.system.scrollback = 100 
        bpy.context.preferences.edit.undo_steps = 100
        with Timer(f'Writing output blendfile to {output_folder / output_blend_name}'):
            bpy.ops.wm.save_mainfile(filepath=str(output_folder / output_blend_name))
            tag_system.save_tag(path=str(output_folder / "MaskTag.json"))

        with (output_folder/ "version.txt").open('w') as f:
            scene_version = get_scene_tag('VERSION')
            f.write(f"{scene_version}\n")

        with (output_folder/'polycounts.txt').open('w') as f:
            save_polycounts(f)

    for col in bpy.data.collections['unique_assets'].children:
        col.hide_viewport = False

    if Task.Render in task or Task.GroundTruth in task:
        render(scene_seed, output_folder=output_folder, camera_id=camera_id, resample_idx=resample_idx)

    if Task.MeshSave in task:
        save_meshes(scene_seed, output_folder=output_folder, frame_range=frame_range, camera_id=camera_id)


def determine_scene_seed(args):

    if args.seed is None:
        if Task.Coarse not in args.task:
            raise ValueError(
                f'Running tasks on an already generated scene, you need to specify --seed or results will'
                f' not be view-consistent')
        return randint(1e7), 'chosen at random'

    # WARNING: Do not add support for decimal numbers here, it will cause ambiguity, as some hex numbers are valid decimals

    try:
        return int(args.seed, 16), 'parsed as hexadecimal'
    except ValueError:
        pass

    return int_hash(args.seed), 'hashed string to integer'

def apply_scene_seed(args):
    scene_seed, reason = determine_scene_seed(args)
    logging.info(f'Converted {args.seed=} to {scene_seed=}, {reason}')
    gin.constant('OVERALL_SEED', scene_seed)
    del args.seed

    random.seed(scene_seed)
    np.random.seed(scene_seed)
    return scene_seed

def apply_gin_configs(args, scene_seed, skip_unknown=False):

    scene_types = [p.stem for p in Path('config/scene_types').iterdir()]
    scene_specified = any(s in scene_types or s.startswith("figure") for s in args.gin_config)

    weights = {
        "kelp_forest": 1,
        "coral_reef": 1,
        "forest": 1,
        "river": 1,
        "desert": 1,
        "coast": 1,
        "cave": 1,
        "mountain": 1,
        "canyon": 0.5,
        "plain": 0.5,
        "cliff": 0.5,
        "arctic": 0.25,
        "snowy_mountain": 0.25,
}
    assert all(k in scene_types for k in weights)

    scene_types = [s for s in scene_types if s in weights]
    weights = np.array([weights[k] for k in scene_types], dtype=float)
    weights /= weights.sum()

    if not scene_specified:
        scene_type = np.random.RandomState(scene_seed).choice(scene_types, p=weights)
        logging.warning(f'Randomly selected {scene_type=}. IF THIS IS NOT INTENDED THEN YOU ARE MISSING SCENE CONFIGS')
        if len(args.gin_config) > 0 and args.gin_config[0] == 'base':
            args.gin_config = [scene_type] + args.gin_config[1:]
        else:
            args.gin_config = [scene_type] + args.gin_config

    def find_config(g):
        as_figure_type = f'config/figure_types/{g}.gin'
        if os.path.exists(as_figure_type):
            return as_figure_type
        as_scene_type = f'config/scene_types/{g}.gin'
        if os.path.exists(as_scene_type):
            return as_scene_type
        as_base = f'config/{g}.gin'
        if os.path.exists(as_base):
            return as_base
        raise ValueError(f'Couldn not locate {g} in either config/ or config/scene_types')

    bindings = sanitize_gin_override(args.gin_param)
    confs = [find_config(g) for g in ['base'] + args.gin_config]
    gin.parse_config_files_and_bindings(confs, bindings=bindings, skip_unknown=skip_unknown)

def main(
    input_folder, 
    output_folder,
    scene_seed,
    task, 
    task_uniqname,
    **kwargs
):
    
    version_req = ['3.3.1']
    assert bpy.app.version_string in version_req, f'You are using blender={bpy.app.version_string} which is ' \
                                                  f'not supported. Please use {version_req}'
    logging.info(f'infinigen version {VERSION}')
    logging.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

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

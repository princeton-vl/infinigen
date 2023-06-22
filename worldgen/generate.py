# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import argparse
import ast
import os
import random
import sys
import cProfile
from pathlib import Path
import logging
from functools import partial
import pprint
from collections import defaultdict

import bpy
import mathutils
from mathutils import Vector
import gin
import numpy as np
from numpy.random import uniform, normal, randint
from tqdm import tqdm
from frozendict import frozendict

sys.path.append(os.getcwd())

from terrain import Terrain
from util.organization import Task, Attributes, Tags, ElementNames

from placement import placement, density, camera as cam_util
from placement.split_in_view import split_inview
from lighting import lighting, kole_clouds

from assets.trees.generate import TreeFactory, BushFactory, random_season, random_leaf_collection
from assets.glowing_rocks import GlowingRocksFactory
from assets.creatures import CarnivoreFactory, HerbivoreFactory, FishFactory, FishSchoolFactory, \
    BeetleFactory, AntSwarmFactory, BirdFactory, SnakeFactory, \
    CrustaceanFactory, FlyingBirdFactory, CrabFactory, LobsterFactory, SpinyLobsterFactory
from assets.insects.assembled.dragonfly import DragonflyFactory
from assets.cloud.generate import CloudFactory
from assets.cactus import CactusFactory
from assets.creatures import boid_swarm

import surfaces.scatters
from surfaces.scatters import rocks, grass, snow_layer, ground_leaves, ground_twigs, \
    chopped_trees, pinecone, fern, flowerplant, monocot, ground_mushroom, \
    slime_mold, moss, ivy, lichen, mushroom, decorative_plants, seashells
from surfaces.scatters.utils.selection import scatter_lower, scatter_upward
from surfaces.templates import mountain, sand, water, atmosphere_light_haze, sandstone, cracked_ground, \
    soil, dirt, cobble_stone, chunkyrock, stone, lava, ice, mud, snow

from placement import particles, placement, density, camera as cam_util, animation_policy, instance_scatter, detail
from assets import particles as particle_assets

from surfaces.scatters import pine_needle, seaweed, coral_reef, jellyfish, urchin
from assets import boulder, cactus, caustics_lamp
from assets.monocot import kelp
from surfaces import surface

import surfaces.scatters

from placement.factory import make_asset_collection
from util import blender as butil
from util.logging import Timer
from util.math import FixedSeed, int_hash
from util.pipeline import RandomStageExecutor
from util.random import sample_registry, random_general

import core as infinigen

@gin.configurable
def compose_scene(output_folder, terrain, scene_seed, **params):

    p = RandomStageExecutor(scene_seed, output_folder, params)

    p.run_stage('fancy_clouds', kole_clouds.add_kole_clouds)

    season = p.run_stage('season', random_season, use_chance=False)
    logging.info(f'{season=}')

    terrain_mesh = p.run_stage('terrain', terrain.coarse_terrain, use_chance=False)
    density.set_tag_dict(terrain.tag_dict)
    terrain_bvh = mathutils.bvhtree.BVHTree.FromObject(terrain_mesh, bpy.context.evaluated_depsgraph_get())

    land_domain = params.get('land_domain_tags')
    underwater_domain = params.get('underwater_domain_tags')
    nonliving_domain = params.get('nonliving_domain_tags')

    def choose_forest_params():
        # params to be shared between unique and instanced trees
        n_tree_species = randint(1, params.get("max_tree_species", 3) + 1)
        tree_params = lambda: {
            'density': params.get("tree_density", uniform(0.045, 0.15)) / n_tree_species,
            'distance_min': uniform(1, 2.5),
            'select_scale': uniform(0.03, 0.3)
        }
        return [tree_params() for _ in range(n_tree_species)]
    tree_species_params = p.run_stage('forest_params', choose_forest_params, use_chance=False)

    def add_trees(terrain_mesh):
        for i, params in enumerate(tree_species_params):
            fac = TreeFactory(np.random.randint(1e7), coarse=True)
            selection = density.placement_mask(params['select_scale'], tag=land_domain)
            placement.scatter_placeholders_mesh(terrain_mesh, fac, selection=selection, altitude=-0.1,
                overall_density=params['density'], distance_min=params['distance_min'])
    p.run_stage('trees', add_trees, terrain_mesh)

    def add_bushes(terrain_mesh):
        n_bush_species = randint(1, params.get("max_bush_species", 2) + 1)
        for i in range(n_bush_species):
            spec_density = params.get("bush_density", uniform(0.03, 0.12)) / n_bush_species
            fac = BushFactory(int_hash((scene_seed, i)), coarse=True)
            selection = density.placement_mask(uniform(0.015, 0.2), normal_thresh=0.3, 
                select_thresh=uniform(0.5, 0.6), tag=land_domain)
            placement.scatter_placeholders_mesh(terrain_mesh, fac, altitude=-0.05,
                overall_density=spec_density, distance_min=uniform(0.05, 0.3),
                selection=selection)
    p.run_stage('bushes', add_bushes, terrain_mesh)

    def add_clouds(terrain_mesh):
        cloud_factory = CloudFactory(int_hash((scene_seed, 0)), coarse=True, terrain_mesh=terrain_mesh)
        placement.scatter_placeholders(cloud_factory.spawn_locations(), cloud_factory)
    p.run_stage('clouds', add_clouds, terrain_mesh)

    def add_boulders(terrain_mesh):
        n_boulder_species = randint(1, params.get("max_boulder_species", 5))
        for i in range(n_boulder_species):
            selection = density.placement_mask(0.05, tag=nonliving_domain, select_thresh=uniform(0.55, 0.6))
            fac = boulder.BoulderFactory(int_hash((scene_seed, i)), coarse=True)
            placement.scatter_placeholders_mesh(terrain_mesh, fac, 
                overall_density=params.get("boulder_density", uniform(.02, .05)) / n_boulder_species,
                selection=selection, altitude=-0.25)
    p.run_stage('boulders', add_boulders, terrain_mesh)

    def add_glowing_rocks(terrain_mesh):
        selection = density.placement_mask(uniform(0.03, 0.3), normal_thresh=-1.1, select_thresh=0, tag=Tags.Cave)
        fac = GlowingRocksFactory(int_hash((scene_seed, 0)), coarse=True)
        placement.scatter_placeholders_mesh(terrain_mesh, fac,
            overall_density=params.get("glow_rock_density", 0.025), selection=selection)
    p.run_stage('glowing_rocks', add_glowing_rocks, terrain_mesh)

    def add_kelp(terrain_mesh):
        fac = kelp.KelpMonocotFactory(int_hash((scene_seed, 0)), coarse=True)
        selection = density.placement_mask(scale=0.05, tag=underwater_domain)
        placement.scatter_placeholders_mesh(terrain_mesh, fac, altitude=-0.05,
            overall_density=params.get('kelp_density', uniform(.2, 1)),
            selection=selection, distance_min=3)
    p.run_stage('kelp', add_kelp, terrain_mesh)

    def add_cactus(terrain_mesh):
        n_cactus_species = randint(2, params.get("max_cactus_species", 4))
        for i in range(n_cactus_species):
            fac = cactus.CactusFactory(int_hash((scene_seed, i)), coarse=True)
            selection = density.placement_mask(scale=.05, tag=land_domain, select_thresh=0.57)
            placement.scatter_placeholders_mesh(terrain_mesh, fac, altitude=-0.05,
                overall_density=params.get('cactus_density', uniform(.02, .1) / n_cactus_species),
                selection=selection, distance_min=1)
    p.run_stage('cactus', add_cactus, terrain_mesh)

    def camera_preprocess():
        camera_rigs = cam_util.spawn_camera_rigs()
        scene_bvhtrees = cam_util.camera_selection_preprocessing(terrain)   
        return camera_rigs, scene_bvhtrees 
    camera_rigs, scene_bvhtrees = p.run_stage('camera_preprocess', camera_preprocess, use_chance=False)
    p.run_stage('pose_cameras', lambda: cam_util.configure_cameras(
        camera_rigs, scene_bvhtrees, terrain), use_chance=False)
    cam = cam_util.get_camera(0, 0)
    
    p.run_stage('lighting', lighting.add_lighting, cam, use_chance=False)
    # determine a small area of the terrain for the creatures to run around on
    # must happen before camera is animated, as camera may want to follow them around
    terrain_center, *_ = split_inview(terrain_mesh, cam=cam, 
            start=0, end=0, outofview=False, vis_margin=5, dist_max=params["center_distance"],
            hide_render=True, suffix='center')
    deps = bpy.context.evaluated_depsgraph_get()
    terrain_center_bvh = mathutils.bvhtree.BVHTree.FromObject(terrain_center, deps)
    
    pois = [] # objects / points of interest, for the camera to look at

    def add_ground_creatures(target):
        fac_class = sample_registry(params['ground_creature_registry'])
        fac = fac_class(int_hash((scene_seed, 0)), bvh=terrain_bvh, animation_mode='idle')
        n = params.get('max_ground_creatures', randint(1, 4))
        selection = density.placement_mask(select_thresh=0, tag='beach', altitude_range=(-0.5, 0.5)) if fac_class is CrabFactory else 1
        col = placement.scatter_placeholders_mesh(target, fac, num_placeholders=n, overall_density=1, selection=selection, altitude=0.2)
        return list(col.objects)
    pois += p.run_stage('ground_creatures', add_ground_creatures, target=terrain_center, default=[])

    def flying_creatures():
        fac_class = sample_registry(params['flying_creature_registry'])
        fac = fac_class(randint(1e7), bvh=terrain_bvh, animation_mode='idle')
        n = params.get('max_flying_creatures', randint(2, 7))
        col = placement.scatter_placeholders_mesh(terrain_center, fac, num_placeholders=n, overall_density=1, altitude=0.2)
        return list(col.objects)
    pois += p.run_stage('flying_creatures', flying_creatures, default=[])

    p.run_stage('animate_cameras', lambda: cam_util.animate_cameras(
        camera_rigs, scene_bvhtrees, pois=pois), use_chance=False)

    with Timer('Compute coarse terrain frustrums'):
        terrain_inview, *_ = split_inview(terrain_mesh, verbose=True, outofview=False, print_areas=True,
            cam=cam, vis_margin=2, dist_max=params['inview_distance'], hide_render=True, suffix='inview')
        terrain_near, *_ = split_inview(terrain_mesh, verbose=True, outofview=False, print_areas=True,
            cam=cam, vis_margin=2, dist_max=params['near_distance'], hide_render=True, suffix='near')

        collider = butil.modify_mesh(butil.deep_clone_obj(terrain_near), 'COLLISION', apply=False, show_viewport=True)
        collider.name = collider.name + '.collider'
        collider.collision.use_culling = False
        collider_col = butil.get_collection('colliders')
        butil.put_in_collection(collider, collider_col)

        butil.modify_mesh(terrain_near, 'SUBSURF', levels=2, apply=True)

        deps = bpy.context.evaluated_depsgraph_get()
        terrain_inview_bvh = mathutils.bvhtree.BVHTree.FromObject(terrain_inview, deps)

    p.run_stage('caustics', lambda: caustics_lamp.add_caustics(terrain_near))

    def add_fish_school():
        n = random_general(params.get("max_fish_schools", 3))
        for i in range(n):
            selection = density.placement_mask(0.1, select_thresh=0, tag=underwater_domain)
            fac = FishSchoolFactory(randint(1e7), bvh=terrain_inview_bvh)
            col = placement.scatter_placeholders_mesh(terrain_near, fac, selection=selection,
                overall_density=1, num_placeholders=1, altitude=2)
            placement.populate_collection(fac, col)
    p.run_stage('fish_school', add_fish_school, default=[])

    def add_bug_swarm():
        n = randint(1, params.get("max_bug_swarms", 3) + 1)
        selection = density.placement_mask(0.1, select_thresh=0, tag=land_domain)
        fac = AntSwarmFactory(randint(1e7), bvh=terrain_inview_bvh, coarse=True)
        col = placement.scatter_placeholders_mesh(terrain_inview, fac, 
            selection=selection, overall_density=1, num_placeholders=n, altitude=2)
        placement.populate_collection(fac, col)
    p.run_stage('bug_swarm', add_bug_swarm)

    def add_rocks(target):
        selection = density.placement_mask(scale=0.15, select_thresh=0.5,
            normal_thresh=0.7, return_scalar=True, tag=nonliving_domain)
        _, rock_col = surfaces.scatters.rocks.apply(target, selection=selection)
        return rock_col
    p.run_stage('rocks', add_rocks, terrain_inview)

    def add_ground_leaves(target):
        selection = density.placement_mask(scale=0.1, select_thresh=0.52, normal_thresh=0.7, return_scalar=True, tag=land_domain)
        surfaces.scatters.ground_leaves.apply(target, selection=selection, season=season)
    p.run_stage('ground_leaves', add_ground_leaves, terrain_near, prereq='trees')
                
    def add_ground_twigs(target):
        use_leaves = uniform() < 0.5
        selection = density.placement_mask(scale=0.15, select_thresh=0.55, normal_thresh=0.7, return_scalar=True, tag=nonliving_domain)
        surfaces.scatters.ground_twigs.apply(target, selection=selection, use_leaves=use_leaves)
    p.run_stage('ground_twigs', add_ground_twigs, terrain_near)

    def add_chopped_trees(target):
        selection = density.placement_mask(scale=0.15, select_thresh=uniform(0.55, 0.6), 
                                           normal_thresh=0.7, return_scalar=True, tag=nonliving_domain)
        surfaces.scatters.chopped_trees.apply(target, selection=selection)
    p.run_stage('chopped_trees', add_chopped_trees, terrain_inview)

    def add_grass(target):
        select_max = params.get('grass_select_max', 0.5)
        selection = density.placement_mask(
            normal_dir=(0, 0, 1), scale=0.1, tag=land_domain,
            return_scalar=True, select_thresh=uniform(select_max/2, select_max))
        surfaces.scatters.grass.apply(target, selection=selection)
    p.run_stage('grass', add_grass, terrain_inview)

    def add_monocots(target):
        selection = density.placement_mask(
            normal_dir=(0, 0, 1), scale=0.2, tag=land_domain)
        surfaces.scatters.monocot.apply(terrain_inview, grass=True, selection=selection)
        selection = density.placement_mask(
            normal_dir=(0, 0, 1), scale=0.2, select_thresh=0.55,
            tag=params.get("grass_habitats", None))
        surfaces.scatters.monocot.apply(target, grass=False, selection=selection)
    p.run_stage('monocots', add_monocots, terrain_inview)

    def add_ferns(target):
        selection = density.placement_mask(normal_dir=(0, 0, 1), scale=0.1, 
                    select_thresh=0.6, return_scalar=True, tag=land_domain)
        surfaces.scatters.fern.apply(target, selection=selection)
    p.run_stage('ferns', add_ferns, terrain_inview)

    def add_flowers(target):
        selection = density.placement_mask(normal_dir=(0, 0, 1), scale=0.01,
            select_thresh=0.6, return_scalar=True, tag=land_domain)
        surfaces.scatters.flowerplant.apply(target, selection=selection)
    p.run_stage('flowers', add_flowers, terrain_inview)

    def add_corals(target):
        vertical_faces = density.placement_mask(scale=0.15, select_thresh=uniform(.44, .48))
        coral_reef.apply(target, selection=vertical_faces, tag=underwater_domain,
                         density=params.get('coral_density', 2.5))
        horizontal_faces = density.placement_mask(scale=.15, normal_thresh=-.4, normal_thresh_high=.4)
        coral_reef.apply(target, selection=horizontal_faces, n=5, horizontal=True, tag=underwater_domain,
                         density=params.get('horizontal_coral_density', 2.5))
    p.run_stage('corals', add_corals, terrain_inview)

    p.run_stage('mushroom', lambda: surfaces.scatters.ground_mushroom.Mushrooms().apply(terrain_near,
        selection=density.placement_mask(scale=.1, select_thresh=.65, return_scalar=True, tag=land_domain),
        density=params.get('mushroom_density', 2)))

    p.run_stage('seaweed', lambda: seaweed.apply(terrain_inview, 
        selection=density.placement_mask(scale=0.05, select_thresh=.5, normal_thresh=0.4, tag=underwater_domain)))
    p.run_stage('urchin', lambda: urchin.apply(terrain_inview,
        selection=density.placement_mask(scale=0.05, select_thresh=.5, tag=underwater_domain)))
    p.run_stage('jellyfish', lambda: jellyfish.apply(terrain_inview,
        selection=density.placement_mask(scale=0.05, select_thresh=.5, tag=underwater_domain)))
    
    p.run_stage('seashells', lambda: surfaces.scatters.seashells.apply(terrain_near,
        selection=density.placement_mask(scale=0.05, select_thresh=.5, tag='landscape,', return_scalar=True)))
    p.run_stage('pinecone', lambda: surfaces.scatters.pinecone.apply(terrain_near,
        selection=density.placement_mask(scale=.1, select_thresh=.63, tag=land_domain)))
    p.run_stage('pine_needle', lambda: pine_needle.apply(terrain_near,
        selection=density.placement_mask(scale=uniform(0.05, 0.2), select_thresh=uniform(0.4, 0.55), tag=land_domain, return_scalar=True)))
    p.run_stage('decorative_plants', lambda: surfaces.scatters.decorative_plants.apply(terrain_near,
        selection=density.placement_mask(scale=uniform(0.05, 0.2), select_thresh=uniform(0.5, 0.65), tag=land_domain, return_scalar=True)))

    p.run_stage('wind', particle_assets.wind_effector)
    p.run_stage('turbulence', particle_assets.turbulence_effector)
    emitter_off = Vector((0, 0, 5)) # to allow space to fall into frame from off screen

    def add_leaf_particles():
        return particles.particle_system(
            emitter=butil.spawn_plane(location=emitter_off, size=60),
            subject=random_leaf_collection(n=5, season=season),
            settings=particles.falling_leaf_settings())
    def add_rain_particles():
        return particles.particle_system(
            emitter=butil.spawn_plane(location=emitter_off, size=30),
            subject=make_asset_collection(particle_assets.RaindropFactory(scene_seed), 5),
            settings=particles.rain_settings())
    def add_dust_particles():
        return particles.particle_system(
            emitter=butil.spawn_cube(location=Vector(), size=30),
            subject=make_asset_collection(particle_assets.DustMoteFactory(scene_seed), 5),
            settings=particles.floating_dust_settings())
    def add_marine_snow_particles():
        return particles.particle_system(
            emitter=butil.spawn_cube(location=Vector(), size=30),
            subject=make_asset_collection(particle_assets.DustMoteFactory(scene_seed), 5),
            settings=particles.marine_snow_setting())
    def add_snow_particles():
        return particles.particle_system(
            emitter=butil.spawn_plane(location=emitter_off, size=60),
            subject=make_asset_collection(particle_assets.SnowflakeFactory(scene_seed), 5),
            settings=particles.snow_settings())

    particle_systems = [
        p.run_stage('leaf_particles', add_leaf_particles, prereq='trees'),
        p.run_stage('rain_particles', add_rain_particles),
        p.run_stage('dust_particles', add_dust_particles),
        p.run_stage('marine_snow_particles', add_marine_snow_particles),
        p.run_stage('snow_particles', add_snow_particles),
    ]

    for emitter, system in filter(lambda s: s is not None, particle_systems):
        with Timer(f"Baking particle system"):
            butil.constrain_object(emitter, "COPY_LOCATION", use_offset=True, target=cam.parent)
            particles.bake(emitter, system)
        butil.put_in_collection(emitter, butil.get_collection('particles'))

    p.save_results(output_folder/'pipeline_coarse.csv')

    return terrain

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=Path)
    parser.add_argument('--input_folder', type=Path, default=None)
    parser.add_argument('-s', '--seed', default=None, help="The seed used to generate the scene")
    parser.add_argument('-t', '--task', nargs='+', default=['coarse'],
                        choices=['coarse', 'populate', 'fine_terrain', 'ground_truth', 'render', 'mesh_save'])
    parser.add_argument('-g', '--gin_config', nargs='+', default=['base'],
                        help='Set of config files for gin (separated by spaces) '
                             'e.g. --gin_config file1 file2 (exclude .gin from path)')
    parser.add_argument('-p', '--gin_param', nargs='+', default=[],
                        help='Parameter settings that override config defaults '
                             'e.g. --gin_param module_1.a=2 module_2.b=3')
    parser.add_argument('--task_uniqname', type=str, default=None)
    parser.add_argument('-d', '--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument( '-v', '--verbose', action="store_const", dest="loglevel", const=logging.INFO)

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    extras = '[%(filename)s:%(lineno)d] ' if args.loglevel == logging.DEBUG else ''
    logging.basicConfig(
        format=f'[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] {extras}| %(message)s',
        level=args.loglevel,
        datefmt='%H:%M:%S'
    )

    scene_seed = infinigen.apply_scene_seed(args)
    infinigen.apply_gin_configs(args, scene_seed)
    
    infinigen.main(
        compose_scene_func=compose_scene,
        input_folder=args.input_folder, 
        output_folder=args.output_folder, 
        task=args.task, 
        task_uniqname=args.task_uniqname, 
        scene_seed=scene_seed
    )

if __name__ == "__main__":
    main()

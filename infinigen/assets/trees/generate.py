# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Yiming Zuo, Alejandro Newell


import pdb
import logging

import gin
import numpy as np
from numpy.random import uniform, normal

import bpy

from infinigen.assets.trees import tree, treeconfigs, branch
from infinigen.assets.leaves import leaf, leaf_v2, leaf_pine, leaf_ginko, leaf_broadleaf, leaf_maple
from infinigen.assets.fruits import apple, blackberry, coconutgreen, durian, starfruit, strawberry, compositional_fruit
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from . import tree_flower

from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util import camera as camera_util

from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.placement import detail
from infinigen.core.placement.split_in_view import split_inview

from infinigen.core import surface

from infinigen.assets.weather.cloud.generate import CloudFactory
from ..utils.decorate import write_attribute

from infinigen.assets.utils.tag import tag_object, tag_nodegroup

logger = logging.getLogger(__name__)

@gin.configurable
class GenericTreeFactory(AssetFactory):

    scale = 0.35 # trees are defined in weird units currently, need converting to meters

    def __init__(
        self, 
        factory_seed, 
        genome: tree.TreeParams, 
        child_col, 
        trunk_surface, 
        realize=False, 
        meshing_camera=None, 
        cam_meshing_max_dist=1e7,
        coarse_mesh_placeholder=False,
        adapt_mesh_method='remesh', 
        decimate_placeholder_levels=0, 
        min_dist=None,
        coarse=False
    ):

        super(GenericTreeFactory, self).__init__(factory_seed, coarse=coarse)

        self.genome = genome
        self.child_col = child_col
        self.trunk_surface = trunk_surface
        self.realize = realize

        self.camera = meshing_camera
        self.cam_meshing_max_dist = cam_meshing_max_dist
        self.adapt_mesh_method = adapt_mesh_method
        self.decimate_placeholder_levels = decimate_placeholder_levels
        self.coarse_mesh_placeholder = coarse_mesh_placeholder

        self.min_dist = min_dist

    def create_placeholder(self, i, loc, rot):

        logger.debug(f'generating tree skeleton')
        skeleton_obj = tree.tree_skeleton(
            self.genome.skeleton, self.genome.trunk_spacecol, self.genome.roots_spacecol, init_pos=(0, 0, 0), scale=self.scale)
        
        if self.coarse_mesh_placeholder:
            pholder =  self._create_coarse_mesh(skeleton_obj)
        else:
            pholder = butil.spawn_cube(size=4)

        butil.parent_to(skeleton_obj, pholder, no_inverse=True)
        return pholder
            
    
    def _create_coarse_mesh(self, skeleton_obj):
        logger.debug('generating skinned mesh')
        coarse_mesh = deep_clone_obj(skeleton_obj)
        surface.add_geomod(coarse_mesh, tree.skin_tree, input_kwargs={'params': self.genome.skinning}, apply=True)

        if self.decimate_placeholder_levels > 0:
            butil.modify_mesh(coarse_mesh, 'DECIMATE', decimate_type='UNSUBDIV', iterations=self.decimate_placeholder_levels)

        return coarse_mesh

    def finalize_placeholders(self, placeholders):
        if not self.coarse_mesh_placeholder:
            return
        with FixedSeed(self.factory_seed):
            logger.debug(f'adding {self.trunk_surface} to {len(placeholders)=}')
            self.trunk_surface.apply(placeholders)

    def asset_parameters(self, distance: float, vis_distance: float) -> dict:
        if self.min_dist is not None and distance < self.min_dist:
            logger.warn(f'{self} recieved {distance=} which violates {self.min_dist=}. Ignoring')
            distance = self.min_dist
        return dict(face_size=detail.target_face_size(distance), distance=distance)

    def create_asset(self, placeholder, face_size, distance, **kwargs) -> bpy.types.Object:

        skeleton_obj = placeholder.children[0]

        if not self.coarse_mesh_placeholder:
            placeholder = self._create_coarse_mesh(skeleton_obj)
            self.trunk_surface.apply(placeholder)
            butil.parent_to(skeleton_obj, placeholder, no_inverse=True)
            placeholder.hide_render = True

        if self.child_col is not None:
            assert self.genome.child_placement is not None

            max_needed_child_fs = detail.target_face_size(self.min_dist, global_multiplier=1) if self.min_dist is not None else None

            logger.debug(f'adding tree children using {self.child_col=}')
            butil.select_none()
            surface.add_geomod(skeleton_obj, tree.add_tree_children, input_kwargs=dict(
                child_col=self.child_col, params=self.genome.child_placement, 
                realize=self.realize, merge_dist=max_needed_child_fs
            ))

        if self.camera is not None and distance < self.cam_meshing_max_dist:
            assert self.adapt_mesh_method != 'remesh'
            skin_obj, outofview, vert_dists, _ = split_inview(placeholder, cam=self.camera, vis_margin=0.15)
            butil.parent_to(outofview, skin_obj, no_inverse=True, no_transform=True)
            face_size = detail.target_face_size(vert_dists.min())
        else:
            skin_obj = deep_clone_obj(placeholder, keep_modifiers=True, keep_materials=True)

        skin_obj.hide_render = False

        if self.adapt_mesh_method == 'remesh':
            butil.modify_mesh(skin_obj, 'SUBSURF', levels=self.decimate_placeholder_levels + 1) # one extra level to smooth things out or remesh is jaggedy

        with butil.DisableModifiers(skin_obj):
            detail.adapt_mesh_resolution(skin_obj, face_size, method=self.adapt_mesh_method, apply=True)

        butil.parent_to(skin_obj, placeholder, no_inverse=True, no_transform=True)

        if self.realize:
            logger.debug(f'realizing tree children')
            butil.apply_modifiers(skin_obj)
            butil.apply_modifiers(skeleton_obj)
            with butil.SelectObjects([skin_obj, skeleton_obj], active=0):
                bpy.ops.object.join()
        else:
            butil.parent_to(skeleton_obj, skin_obj, no_inverse=True)

        tag_object(skin_obj, 'tree')
        return skin_obj
        

@gin.configurable
def random_season(weights=None):
    options = ['autumn', 'summer', 'spring', 'winter']
    
    if weights is not None:
        weights = np.array([weights[k] for k in options])
    else:
        weights = np.array([0.25, 0.3, 0.4, 0.1])
    return np.random.choice(options, p=weights/weights.sum())

@gin.configurable
def random_species(season='summer', pine_chance=0.):
    tree_species_code = np.random.rand(32)

    if season is None:
        season = random_season()

    if tree_species_code[-1] < pine_chance:
        return treeconfigs.pine_tree(), 'leaf_pine'
    # elif tree_species_code < 0.2:
    #     tree_args = treeconfigs.palm_tree()
    # elif tree_species_code < 0.3:
    #     tree_args = treeconfigs.baobab_tree()
    else:
        return treeconfigs.random_tree(tree_species_code, season), None

def random_tree_child_factory(seed, leaf_params, leaf_type, season, **kwargs):

    if season is None:
        season = random_season()

    fruit_scale = 0.2

    if leaf_type is None:
        return None, None
    elif leaf_type == 'leaf':
        return leaf.LeafFactory(seed, leaf_params, **kwargs), surface.registry('greenery')
    elif leaf_type == 'leaf_pine':
        return leaf_pine.LeafFactoryPine(seed, season, **kwargs), None
    elif leaf_type == 'leaf_ginko':
        return leaf_ginko.LeafFactoryGinko(seed, season, **kwargs), None
    elif leaf_type == 'leaf_maple':
        return leaf_maple.LeafFactoryMaple(seed, season, **kwargs), None
    elif leaf_type == 'leaf_broadleaf':
        return leaf_broadleaf.LeafFactoryBroadleaf(seed, season, **kwargs), None
    elif leaf_type == 'leaf_v2':
        return leaf_v2.LeafFactoryV2(seed, **kwargs), None
    elif leaf_type == 'berry':
        return leaf.BerryFactory(seed, leaf_params, **kwargs), None
    elif leaf_type == 'apple':
        return apple.FruitFactoryApple(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == 'blackberry':
        return blackberry.FruitFactoryBlackberry(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == 'coconutgreen':
        return coconutgreen.FruitFactoryCoconutgreen(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == 'durian':
        return durian.FruitFactoryDurian(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == 'starfruit':
        return starfruit.FruitFactoryStarfruit(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == 'strawberry':
        return strawberry.FruitFactoryStrawberry(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == 'compositional_fruit':
        return compositional_fruit.FruitFactoryCompositional(seed, scale=fruit_scale, **kwargs), None
    elif leaf_type == 'flower':
        return tree_flower.TreeFlowerFactory(seed, rad=uniform(0.15, 0.25), **kwargs), None
    elif leaf_type == 'cloud':
        return CloudFactory(seed), None
    else:
        raise ValueError(f'Unrecognized {leaf_type=}')   

def make_leaf_collection(seed, 
        leaf_params, n_leaf, leaf_types, decimate_rate=0.0,
        season=None):

    logger.debug(f'Starting make_leaf_collection({seed=}, {n_leaf=} ...)')

    if season is None:
        season = random_season()

    weights = []

    if not isinstance(leaf_types, list):
        leaf_types = [leaf_types]

    child_factories = []
    for leaf_type in leaf_types:
        if leaf_type is not None:
            leaf_factory, _ = random_tree_child_factory(seed, leaf_params, leaf_type=leaf_type, season=season)
            child_factories.append(leaf_factory)
            weights.append(1.0)

    weights = np.array(weights)
    weights /= np.sum(weights) # normalize to 1       

    col = make_asset_collection(child_factories, n_leaf, verbose=True, weights=weights)
    # if leaf_surface is not None:
    #     leaf_surface.apply(list(col.objects))
    for obj in col.objects:
        if decimate_rate > 0:
            butil.modify_mesh(obj, 'DECIMATE', ratio=1.0-decimate_rate, apply=True)
        butil.apply_transform(obj, rot=True, scale=True)
        butil.apply_modifiers(obj)
    return col

def random_leaf_collection(season, n=5):
    (_, _, leaf_params), leaf_type = random_species(season=season)
    return make_leaf_collection(np.random.randint(1e5), leaf_params, n_leaf=n, leaf_types=leaf_type or 'leaf_v2', decimate_rate=0.97)

def make_twig_collection(
    seed, 
    twig_params, leaf_params, 
    trunk_surface, 
    n_leaf, n_twig, 
    leaf_types,
    season=None, 
    twig_valid_dist=6
):

    logger.debug(f'Starting make_twig_collection({seed=}, {n_leaf=}, {n_twig=}...)')

    if season is None:
        season = random_season()

    if leaf_types is not None:
        child_col = make_leaf_collection(seed, leaf_params, n_leaf, leaf_types, season=season, decimate_rate=0.97)
    else:
        child_col = None

    twig_factory = GenericTreeFactory(seed, twig_params, child_col, trunk_surface=trunk_surface, realize=True)
    col = make_asset_collection(twig_factory, n_twig, verbose=False, distance=twig_valid_dist)

    if child_col is not None:
        child_col.hide_viewport = False
        butil.delete(list(child_col.objects))
    return col

def make_branch_collection(seed, twig_col, fruit_col, n_branch, coarse=False):

    logger.debug(f'Starting make_branch_collection({seed=}, ...)')

    branch_factory = branch.BranchFactory(seed, twig_col=twig_col, fruit_col=fruit_col, coarse=coarse)
    col = make_asset_collection(branch_factory, n_branch, verbose=False)

    return col

@gin.configurable
class TreeFactory(GenericTreeFactory):

    n_leaf = 5
    n_twig = 2

    @staticmethod
    def get_leaf_type(season):
        # return np.random.choice(['leaf', 'leaf_v2', 'flower', 'berry', 'leaf_ginko'], p=[0, 0.70, 0.15, 0, 0.15])
        # return 
        # return 'leaf_maple'
        leaf_type = np.random.choice(['leaf', 'leaf_v2', 'leaf_broadleaf', 'leaf_ginko', 'leaf_maple'], p=[0, 0.0, 0.70, 0.15, 0.15])
        flower_type = np.random.choice(['flower', 'berry', None], p=[1.0, 0.0, 0.0])
        if season == "spring":
            return [flower_type]
        else:
            return [leaf_type]
        # return [leaf_type, flower_type]
        # return ['leaf_broadleaf', 'leaf_maple', 'leaf_ginko', 'flower']

    @staticmethod
    def get_fruit_type():
        # return np.random.choice(['leaf', 'leaf_v2', 'flower', 'berry', 'leaf_ginko'], p=[0, 0.70, 0.15, 0, 0.15])
        # return 
        # return 'leaf_maple'
        fruit_type = np.random.choice(['apple', 'blackberry', 'coconutgreen', 
            'durian', 'starfruit', 'strawberry', 'compositional_fruit'], 
             p=[0.2, 0.0, 0.2, 0.2, 0.2, 0.0, 0.2])

        return fruit_type

    def __init__(self, seed, season=None, coarse=False, fruit_chance=1.0, **kwargs):

        with FixedSeed(seed):
            if season is None:
                season = np.random.choice(['summer', 'winter', 'autumn', 'spring'])

        with FixedSeed(seed):
            (tree_params, twig_params, leaf_params), leaf_type = random_species(season)

            leaf_type = leaf_type or self.get_leaf_type(season)
            if not isinstance(leaf_type, list):
                leaf_type = [leaf_type]

            trunk_surface = surface.registry('bark')

            if uniform() < fruit_chance:
                fruit_type = self.get_fruit_type()
            else:
                fruit_type = None
        
        super(TreeFactory, self).__init__(seed, tree_params, child_col=None, trunk_surface=trunk_surface, coarse=coarse, **kwargs)

        with FixedSeed(seed):
            colname = f'assets:{self}.twigs'
            use_cached = colname in bpy.data.collections
            if use_cached == coarse:
                logger.warning(f'In {self}, encountered {use_cached=} yet {coarse=}, unexpected since twigs are typically generated only in coarse')

            if colname not in bpy.data.collections:
                twig_col = make_twig_collection(seed, twig_params, leaf_params, trunk_surface, self.n_leaf, self.n_twig, leaf_type, season=season) 
                if fruit_type is not None:
                    fruit_col = make_leaf_collection(seed, leaf_params, self.n_leaf, fruit_type, season=season, decimate_rate=0.0)
                else:
                    fruit_col = butil.get_collection('Empty', reuse=True)

                self.child_col = make_branch_collection(seed, twig_col, fruit_col, n_branch=self.n_twig) 
                self.child_col.name = colname

                assert self.child_col.name == colname, f'Blender truncated {colname} to {self.child_col.name}'
            else:
                self.child_col = bpy.data.collections[colname]

@gin.configurable
class BushFactory(GenericTreeFactory):

    n_leaf = 3
    n_twig = 3
    max_distance = 50

    def __init__(self, seed, coarse=False, **kwargs):

        with FixedSeed(seed):
            shrub_shape = np.random.randint(2)
            trunk_surface = surface.registry('bark')
            tree_params, twig_params, leaf_params = treeconfigs.shrub(shrub_shape=shrub_shape)

        super(BushFactory, self).__init__(seed, tree_params, child_col=None, trunk_surface=trunk_surface, coarse=coarse, **kwargs)

        with FixedSeed(seed):

            leaf_type = np.random.choice(['leaf', 'leaf_v2', 'flower', 'berry'], p=[0.1, 0.4, 0.5, 0])

            colname = f'assets:{self}.twigs'
            use_cached = colname in bpy.data.collections
            if use_cached == coarse:
                logger.warning(f'In {self}, encountered {use_cached=} yet {coarse=}, unexpected since twigs are typically generated only in coarse')

            if colname not in bpy.data.collections:
                self.child_col = make_twig_collection(seed, twig_params, leaf_params, trunk_surface, self.n_leaf, self.n_twig, leaf_type) 
                self.child_col.name = colname
                assert self.child_col.name == colname, f'Blender truncated {colname} to {self.child_col.name}'
            else:
                self.child_col = bpy.data.collections[colname]
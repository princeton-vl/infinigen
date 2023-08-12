import bpy
import mathutils
from mathutils import Vector
import gin
import numpy as np
from numpy.random import uniform, normal, randint

from infinigen.core.util.pipeline import RandomStageExecutor
from infinigen.core.placement import placement, density

from infinigen.assets.fluid.cached_factory_wrappers import (
    CachedTreeFactory,
    CachedCreatureFactory, 
    CachedBoulderFactory, 
    CachedBushFactory, 
    CachedCactusFactory
)
from infinigen.assets.fluid.asset_cache import FireCachingSystem
from infinigen.assets.fluid.fluid import is_fire_in_scene
from infinigen.assets.fluid.flip_fluid import create_flip_fluid_domain, set_flip_fluid_domain, create_flip_fluid_inflow, set_flip_fluid_obstacle, get_objs_inside_domain, make_beach, make_river, make_tilted_river

def cached_fire_scenecomp_options(p: RandomStageExecutor, terrain_mesh, params, tree_species_params):

    land_domain = params.get('land_domain_tags')
    underwater_domain = params.get('underwater_domain_tags')
    nonliving_domain = params.get('nonliving_domain_tags')

    if params.get('cached_fire'):
        fire_cache_system = FireCachingSystem()

    def add_cached_fire_trees(terrain_mesh):
        params = tree_species_params[0]
        species = fire_cache_system.get_cached_species(CachedTreeFactory)
        ind = np.random.choice(len(species))
        s = species[ind]
        fac = CachedTreeFactory(s, coarse=True)
        selection = density.placement_mask(params['select_scale'], tag=land_domain)
        placement.scatter_placeholders_mesh(terrain_mesh, fac, selection=selection, altitude=-0.1,
                overall_density=params['density'], distance_min=params['distance_min'])
    p.run_stage('cached_fire_trees', add_cached_fire_trees, terrain_mesh)
    
    def add_cached_fire_bushes(terrain_mesh):
        n_bush_species = randint(1, params.get("max_bush_species", 2) + 1)
        spec_density = params.get("bush_density", uniform(0.03, 0.12)) / n_bush_species
        species = fire_cache_system.get_cached_species(CachedBushFactory)
        ind = np.random.choice(len(species))
        s = species[ind]
        fac = CachedBushFactory(s, coarse=True)
        selection = density.placement_mask(uniform(0.015, 0.2), normal_thresh=0.3, 
                select_thresh=uniform(0.5, 0.6), tag=land_domain)
        placement.scatter_placeholders_mesh(terrain_mesh, fac, altitude=-0.05,
                overall_density=spec_density, distance_min=uniform(0.05, 0.3),
                selection=selection)
    p.run_stage('cached_fire_bushes', add_cached_fire_bushes, terrain_mesh)

    def add_cached_fire_boulders(terrain_mesh):
        n_boulder_species = randint(1, params.get("max_boulder_species", 5))
        species = fire_cache_system.get_cached_species(CachedBoulderFactory)
        ind = np.random.choice(len(species))
        s = species[ind]
        fac = CachedBoulderFactory(s, coarse=True)
        selection = density.placement_mask(0.05, tag=nonliving_domain, select_thresh=uniform(0.55, 0.6))
        placement.scatter_placeholders_mesh(terrain_mesh, fac, 
                overall_density=params.get("boulder_density", uniform(.02, .05)) / n_boulder_species,
                selection=selection, altitude=-0.25)
    p.run_stage('cached_fire_boulders', add_cached_fire_boulders, terrain_mesh)

    def add_cached_fire_cactus(terrain_mesh):
        n_cactus_species = randint(2, params.get("max_cactus_species", 4))
        species = fire_cache_system.get_cached_species(CachedCactusFactory)
        ind = np.random.choice(len(species))
        s = species[ind]
        fac = CachedCactusFactory(s, coarse=True)
        selection = density.placement_mask(scale=.05, tag=land_domain, select_thresh=0.57)
        placement.scatter_placeholders_mesh(terrain_mesh, fac, altitude=-0.05,
                overall_density=params.get('cactus_density', uniform(.02, .1) / n_cactus_species),
                selection=selection, distance_min=1)
    p.run_stage('cached_fire_cactus', add_cached_fire_cactus, terrain_mesh)

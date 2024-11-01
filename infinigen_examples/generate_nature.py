# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import argparse
import itertools
import logging
from pathlib import Path

import bpy
import gin
import mathutils
import numpy as np
from mathutils import Vector
from numpy.random import randint, uniform

# ruff: noqa: E402
# NOTE: logging config has to be before imports that use logging
logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# unused imports required for gin to find modules currently, # TODO remove
# ruff: noqa: F401
from infinigen.assets import fluid, lighting, weather
from infinigen.assets.materials import (
    atmosphere_light_haze,
    chunkyrock,
    cobble_stone,
    cracked_ground,
    dirt,
    ice,
    lava,
    mountain,
    mud,
    sand,
    sandstone,
    snow,
    soil,
    stone,
    water,
)
from infinigen.assets.objects import (
    cactus,
    cloud,
    creatures,
    leaves,
    monocot,
    particles,
    rocks,
    trees,
)
from infinigen.assets.scatters import (
    chopped_trees,
    coral_reef,
    decorative_plants,
    fern,
    flowerplant,
    grass,
    ground_leaves,
    ground_mushroom,
    ground_twigs,
    ivy,
    jellyfish,
    lichen,
    monocots,
    moss,
    pebbles,
    pine_needle,
    pinecone,
    seashells,
    seaweed,
    slime_mold,
    snow_layer,
    urchin,
)
from infinigen.assets.scatters.utils.selection import scatter_lower, scatter_upward
from infinigen.core import execute_tasks, init, surface
from infinigen.core.placement import camera as cam_util
from infinigen.core.placement import density, placement, split_in_view
from infinigen.core.util import blender as butil
from infinigen.core.util import logging as logging_util
from infinigen.core.util import pipeline
from infinigen.core.util.imu import save_imu_tum_files
from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.core.util.organization import Tags, Task
from infinigen.core.util.pipeline import RandomStageExecutor
from infinigen.core.util.random import random_general, sample_registry
from infinigen.terrain import Terrain

logger = logging.getLogger(__name__)


@gin.configurable
def compose_nature(output_folder, scene_seed, **params):
    p = pipeline.RandomStageExecutor(scene_seed, output_folder, params)

    def add_coarse_terrain():
        terrain = Terrain(
            scene_seed,
            surface.registry,
            task="coarse",
            on_the_fly_asset_folder=output_folder / "assets",
        )
        terrain_mesh = terrain.coarse_terrain()
        density.set_tag_dict(terrain.tag_dict)
        return terrain, terrain_mesh

    terrain, terrain_mesh = p.run_stage(
        "terrain", add_coarse_terrain, use_chance=False, default=(None, None)
    )

    if terrain_mesh is None:
        terrain_mesh = butil.create_noise_plane()
        density.set_tag_dict({})

    scene_bvh = mathutils.bvhtree.BVHTree.FromObject(
        terrain_mesh, bpy.context.evaluated_depsgraph_get()
    )

    land_domain = params.get("land_domain_tags")
    underwater_domain = params.get("underwater_domain_tags")
    nonliving_domain = params.get("nonliving_domain_tags")

    p.run_stage("fancy_clouds", weather.kole_clouds.add_kole_clouds)

    season = p.run_stage("season", trees.random_season, use_chance=False)
    logging.info(f"{season=}")

    def choose_forest_params():
        # params to be shared between unique and instanced trees
        n_tree_species = randint(1, params.get("max_tree_species", 3) + 1)

        def tree_params():
            return {
                "density": params.get("tree_density", uniform(0.045, 0.15))
                / n_tree_species,
                "distance_min": uniform(1, 2.5),
                "select_scale": uniform(0.03, 0.3),
            }

        return [tree_params() for _ in range(n_tree_species)]

    tree_species_params = p.run_stage(
        "forest_params", choose_forest_params, use_chance=False
    )

    def add_trees(terrain_mesh):
        for i, params in enumerate(tree_species_params):
            fac = trees.TreeFactory(np.random.randint(1e7), coarse=True)
            selection = density.placement_mask(params["select_scale"], tag=land_domain)
            placement.scatter_placeholders_mesh(
                terrain_mesh,
                fac,
                selection=selection,
                altitude=-0.1,
                overall_density=params["density"],
                distance_min=params["distance_min"],
            )

    p.run_stage("trees", add_trees, terrain_mesh)

    def add_bushes(terrain_mesh):
        n_bush_species = randint(1, params.get("max_bush_species", 2) + 1)
        for i in range(n_bush_species):
            spec_density = (
                params.get("bush_density", uniform(0.03, 0.12)) / n_bush_species
            )
            fac = trees.BushFactory(int_hash((scene_seed, i)), coarse=True)
            selection = density.placement_mask(
                uniform(0.015, 0.2),
                normal_thresh=0.3,
                select_thresh=uniform(0.5, 0.6),
                tag=land_domain,
            )
            placement.scatter_placeholders_mesh(
                terrain_mesh,
                fac,
                altitude=-0.05,
                overall_density=spec_density,
                distance_min=uniform(0.05, 0.3),
                selection=selection,
            )

    p.run_stage("bushes", add_bushes, terrain_mesh)

    def add_clouds(terrain_mesh):
        cloud_factory = cloud.CloudFactory(
            int_hash((scene_seed, 0)), coarse=True, terrain_mesh=terrain_mesh
        )
        placement.scatter_placeholders(cloud_factory.spawn_locations(), cloud_factory)

    p.run_stage("clouds", add_clouds, terrain_mesh)

    def add_boulders(terrain_mesh):
        n_boulder_species = randint(1, params.get("max_boulder_species", 5))
        for i in range(n_boulder_species):
            selection = density.placement_mask(
                0.05, tag=nonliving_domain, select_thresh=uniform(0.55, 0.6)
            )
            fac = rocks.BoulderFactory(int_hash((scene_seed, i)), coarse=True)
            placement.scatter_placeholders_mesh(
                terrain_mesh,
                fac,
                overall_density=params.get("boulder_density", uniform(0.02, 0.05))
                / n_boulder_species,
                selection=selection,
                altitude=-0.25,
            )

    p.run_stage("boulders", add_boulders, terrain_mesh)

    fluid.cached_fire_scenecomp_options(p, terrain_mesh, params, tree_species_params)

    def add_glowing_rocks(terrain_mesh):
        selection = density.placement_mask(
            uniform(0.03, 0.3), normal_thresh=-1.1, select_thresh=0, tag=Tags.Cave
        )
        fac = rocks.GlowingRocksFactory(int_hash((scene_seed, 0)), coarse=True)
        placement.scatter_placeholders_mesh(
            terrain_mesh,
            fac,
            overall_density=params.get("glow_rock_density", 0.025),
            selection=selection,
        )

    p.run_stage("glowing_rocks", add_glowing_rocks, terrain_mesh)

    def add_kelp(terrain_mesh):
        fac = monocot.KelpMonocotFactory(int_hash((scene_seed, 0)), coarse=True)
        selection = density.placement_mask(scale=0.05, tag=underwater_domain)
        placement.scatter_placeholders_mesh(
            terrain_mesh,
            fac,
            altitude=-0.05,
            overall_density=params.get("kelp_density", uniform(0.2, 1)),
            selection=selection,
            distance_min=3,
        )

    p.run_stage("kelp", add_kelp, terrain_mesh)

    def add_cactus(terrain_mesh):
        n_cactus_species = randint(2, params.get("max_cactus_species", 4))
        for i in range(n_cactus_species):
            fac = cactus.CactusFactory(int_hash((scene_seed, i)), coarse=True)
            selection = density.placement_mask(
                scale=0.05, tag=land_domain, select_thresh=0.57
            )
            placement.scatter_placeholders_mesh(
                terrain_mesh,
                fac,
                altitude=-0.05,
                overall_density=params.get(
                    "cactus_density", uniform(0.02, 0.1) / n_cactus_species
                ),
                selection=selection,
                distance_min=1,
            )

    p.run_stage("cactus", add_cactus, terrain_mesh)

    def camera_preprocess():
        camera_rigs = cam_util.spawn_camera_rigs()
        scene_preprocessed = cam_util.camera_selection_preprocessing(
            terrain,
            terrain_mesh,
            tags_ratio=params.get("camera_selection_tags_ratio"),
            ranges_ratio=params.get("camera_selection_ranges_ratio"),
            anim_criterion_keys=params.get(
                "camera_selection_anim_criterion_keys", False
            ),
        )
        return camera_rigs, scene_preprocessed

    camera_rigs, scene_preprocessed = p.run_stage(
        "camera_preprocess", camera_preprocess, use_chance=False
    )

    bbox = (
        terrain.get_bounding_box()
        if terrain is not None
        else butil.bounds(terrain_mesh)
    )
    p.run_stage(
        "pose_cameras",
        lambda: cam_util.configure_cameras(
            camera_rigs,
            scene_preprocessed,
            init_bounding_box=bbox,
            terrain_mesh=terrain_mesh,
        ),
        use_chance=False,
    )
    primary_cams = [rig.children[0] for rig in camera_rigs]

    p.run_stage(
        "lighting",
        lighting.sky_lighting.add_lighting,
        primary_cams[0],
        use_chance=False,
    )

    # determine a small area of the terrain for the creatures to run around on
    # must happen before camera is animated, as camera may want to follow them around
    terrain_center, *_ = split_in_view.split_inview(
        terrain_mesh,
        primary_cams,
        dist_max=params["center_distance"],
        vis_margin=5,
        frame_start=0,
        frame_end=0,
        outofview=False,
        hide_render=True,
        suffix="center",
    )
    deps = bpy.context.evaluated_depsgraph_get()
    mathutils.bvhtree.BVHTree.FromObject(terrain_center, deps)

    pois = []  # objects / points of interest, for the camera to look at

    def add_ground_creatures(target):
        fac_class = sample_registry(params["ground_creature_registry"])
        fac = fac_class(int_hash((scene_seed, 0)), bvh=scene_bvh, animation_mode="idle")
        n = params.get("max_ground_creatures", randint(1, 4))
        selection = (
            density.placement_mask(
                select_thresh=0, tag="beach", altitude_range=(-0.5, 0.5)
            )
            if fac_class is creatures.CrabFactory
            else 1
        )
        col = placement.scatter_placeholders_mesh(
            target,
            fac,
            num_placeholders=n,
            overall_density=1,
            selection=selection,
            altitude=0.2,
        )
        return list(col.objects)

    pois += p.run_stage(
        "ground_creatures", add_ground_creatures, target=terrain_center, default=[]
    )

    def flying_creatures():
        fac_class = sample_registry(params["flying_creature_registry"])
        fac = fac_class(randint(1e7), bvh=scene_bvh, animation_mode="idle")
        n = params.get("max_flying_creatures", randint(2, 7))
        col = placement.scatter_placeholders_mesh(
            terrain_center, fac, num_placeholders=n, overall_density=1, altitude=0.2
        )
        return list(col.objects)

    pois += p.run_stage("flying_creatures", flying_creatures, default=[])

    def animate_cameras():
        cam_util.animate_cameras(camera_rigs, bbox, scene_preprocessed, pois=pois)

        frames_folder = output_folder.parent / "frames"
        animated_cams = [cam for cam in camera_rigs if cam.animation_data is not None]
        save_imu_tum_files(frames_folder / "imu_tum", animated_cams)

    p.run_stage(
        "animate_cameras",
        animate_cameras,
        use_chance=False,
    )

    with logging_util.Timer("Compute coarse terrain frustrums"):
        terrain_inview, *_ = split_in_view.split_inview(
            terrain_mesh,
            primary_cams,
            verbose=True,
            outofview=False,
            vis_margin=2,
            dist_max=params["inview_distance"],
            hide_render=True,
            suffix="inview",
        )
        terrain_near, *_ = split_in_view.split_inview(
            terrain_mesh,
            primary_cams,
            verbose=True,
            outofview=False,
            vis_margin=2,
            dist_max=params["near_distance"],
            hide_render=True,
            suffix="near",
        )

        collider = butil.modify_mesh(
            butil.deep_clone_obj(terrain_near),
            "COLLISION",
            apply=False,
            show_viewport=True,
        )
        collider.name = collider.name + ".collider"
        collider.collision.use_culling = False
        collider_col = butil.get_collection("colliders")
        butil.put_in_collection(collider, collider_col)

        butil.modify_mesh(terrain_near, "SUBSURF", levels=2, apply=True)

        deps = bpy.context.evaluated_depsgraph_get()
        terrain_inview_bvh = mathutils.bvhtree.BVHTree.FromObject(terrain_inview, deps)

    p.run_stage("caustics", lambda: lighting.caustics_lamp.add_caustics(terrain_near))

    def add_fish_school():
        n = random_general(params.get("max_fish_schools", 3))
        for i in range(n):
            selection = density.placement_mask(
                0.1, select_thresh=0, tag=underwater_domain
            )
            fac = creatures.FishSchoolFactory(randint(1e7), bvh=terrain_inview_bvh)
            col = placement.scatter_placeholders_mesh(
                terrain_near,
                fac,
                selection=selection,
                overall_density=1,
                num_placeholders=1,
                altitude=2,
            )
            placement.populate_collection(fac, col)

    p.run_stage("fish_school", add_fish_school, default=[])

    def add_bug_swarm():
        n = randint(1, params.get("max_bug_swarms", 3) + 1)
        selection = density.placement_mask(0.1, select_thresh=0, tag=land_domain)
        fac = creatures.AntSwarmFactory(
            randint(1e7), bvh=terrain_inview_bvh, coarse=True
        )
        col = placement.scatter_placeholders_mesh(
            terrain_inview,
            fac,
            selection=selection,
            overall_density=1,
            num_placeholders=n,
            altitude=2,
        )
        placement.populate_collection(fac, col)

    p.run_stage("bug_swarm", add_bug_swarm)

    def add_rocks(target):
        selection = density.placement_mask(
            scale=0.15,
            select_thresh=0.5,
            normal_thresh=0.7,
            return_scalar=True,
            tag=nonliving_domain,
        )
        _, rock_col = pebbles.apply(target, selection=selection)
        return rock_col

    p.run_stage("rocks", add_rocks, terrain_inview)

    def add_ground_leaves(target):
        selection = density.placement_mask(
            scale=0.1,
            select_thresh=0.52,
            normal_thresh=0.7,
            return_scalar=True,
            tag=land_domain,
        )
        ground_leaves.apply(target, selection=selection, season=season)

    p.run_stage("ground_leaves", add_ground_leaves, terrain_near, prereq="trees")

    def add_ground_twigs(target):
        use_leaves = uniform() < 0.5
        selection = density.placement_mask(
            scale=0.15,
            select_thresh=0.55,
            normal_thresh=0.7,
            return_scalar=True,
            tag=nonliving_domain,
        )
        ground_twigs.apply(target, selection=selection, use_leaves=use_leaves)

    p.run_stage("ground_twigs", add_ground_twigs, terrain_near)

    def add_chopped_trees(target):
        selection = density.placement_mask(
            scale=0.15,
            select_thresh=uniform(0.55, 0.6),
            normal_thresh=0.7,
            return_scalar=True,
            tag=nonliving_domain,
        )
        chopped_trees.apply(target, selection=selection)

    p.run_stage("chopped_trees", add_chopped_trees, terrain_inview)

    def add_grass(target):
        select_max = params.get("grass_select_max", 0.5)
        selection = density.placement_mask(
            normal_dir=(0, 0, 1),
            scale=0.1,
            tag=land_domain,
            return_scalar=True,
            select_thresh=uniform(select_max / 2, select_max),
        )
        grass.apply(target, selection=selection)

    p.run_stage("grass", add_grass, terrain_inview)

    def add_monocots(target):
        selection = density.placement_mask(
            normal_dir=(0, 0, 1), scale=0.2, tag=land_domain
        )
        monocots.apply(terrain_inview, grass=True, selection=selection)
        selection = density.placement_mask(
            normal_dir=(0, 0, 1),
            scale=0.2,
            select_thresh=0.55,
            tag=params.get("grass_habitats", None),
        )
        monocots.apply(target, grass=False, selection=selection)

    p.run_stage("monocots", add_monocots, terrain_inview)

    def add_ferns(target):
        selection = density.placement_mask(
            normal_dir=(0, 0, 1),
            scale=0.1,
            select_thresh=0.6,
            return_scalar=True,
            tag=land_domain,
        )
        fern.apply(target, selection=selection)

    p.run_stage("ferns", add_ferns, terrain_inview)

    def add_flowers(target):
        selection = density.placement_mask(
            normal_dir=(0, 0, 1),
            scale=0.01,
            select_thresh=0.6,
            return_scalar=True,
            tag=land_domain,
        )
        flowerplant.apply(target, selection=selection)

    p.run_stage("flowers", add_flowers, terrain_inview)

    def add_corals(target):
        vertical_faces = density.placement_mask(
            scale=0.15, select_thresh=uniform(0.44, 0.48)
        )
        coral_reef.apply(
            target,
            selection=vertical_faces,
            tag=underwater_domain,
            density=params.get("coral_density", 2.5),
        )
        horizontal_faces = density.placement_mask(
            scale=0.15, normal_thresh=-0.4, normal_thresh_high=0.4
        )
        coral_reef.apply(
            target,
            selection=horizontal_faces,
            n=5,
            horizontal=True,
            tag=underwater_domain,
            density=params.get("horizontal_coral_density", 2.5),
        )

    p.run_stage("corals", add_corals, terrain_inview)

    p.run_stage(
        "mushroom",
        lambda: ground_mushroom.Mushrooms().apply(
            terrain_near,
            selection=density.placement_mask(
                scale=0.1, select_thresh=0.65, return_scalar=True, tag=land_domain
            ),
            density=params.get("mushroom_density", 2),
        ),
    )

    p.run_stage(
        "seaweed",
        lambda: seaweed.apply(
            terrain_inview,
            selection=density.placement_mask(
                scale=0.05, select_thresh=0.5, normal_thresh=0.4, tag=underwater_domain
            ),
        ),
    )
    p.run_stage(
        "urchin",
        lambda: urchin.apply(
            terrain_inview,
            selection=density.placement_mask(
                scale=0.05, select_thresh=0.5, tag=underwater_domain
            ),
        ),
    )
    p.run_stage(
        "jellyfish",
        lambda: jellyfish.apply(
            terrain_inview,
            selection=density.placement_mask(
                scale=0.05, select_thresh=0.5, tag=underwater_domain
            ),
        ),
    )

    p.run_stage(
        "seashells",
        lambda: seashells.apply(
            terrain_near,
            selection=density.placement_mask(
                scale=0.05, select_thresh=0.5, tag="landscape,", return_scalar=True
            ),
        ),
    )
    p.run_stage(
        "pinecone",
        lambda: pinecone.apply(
            terrain_near,
            selection=density.placement_mask(
                scale=0.1, select_thresh=0.63, tag=land_domain
            ),
        ),
    )
    p.run_stage(
        "pine_needle",
        lambda: pine_needle.apply(
            terrain_near,
            selection=density.placement_mask(
                scale=uniform(0.05, 0.2),
                select_thresh=uniform(0.4, 0.55),
                tag=land_domain,
                return_scalar=True,
            ),
        ),
    )
    p.run_stage(
        "decorative_plants",
        lambda: decorative_plants.apply(
            terrain_near,
            selection=density.placement_mask(
                scale=uniform(0.05, 0.2),
                select_thresh=uniform(0.5, 0.65),
                tag=land_domain,
                return_scalar=True,
            ),
        ),
    )

    p.run_stage("wind", lambda: weather.WindEffector(randint(1e7))(0))
    p.run_stage("turbulence", lambda: weather.TurbulenceEffector(randint(1e7))(0))

    overhead_emitter = weather.spawn_emitter(
        camera_rigs[0], "plane", offset=Vector((0, 0, 5)), size=60
    )
    cube_emitter = weather.spawn_emitter(
        camera_rigs[0], "cube", offset=Vector(), size=30
    )

    def leaf_particles():
        gen = weather.FallingParticles(
            leaves.LeafFactoryV2(randint(1e7)),
            distribution=weather.falling_leaf_param_distribution,
        )
        return gen(overhead_emitter)

    p.run_stage("leaf_particles", leaf_particles)

    def rain_particles():
        gen = weather.FallingParticles(
            particles.RaindropFactory(randint(1e7)),
            distribution=weather.rain_param_distribution,
        )
        return gen(overhead_emitter)

    p.run_stage("rain_particles", rain_particles)

    def dust_particles():
        gen = weather.FallingParticles(
            particles.DustMoteFactory(randint(1e7)),
            distribution=weather.floating_dust_param_distribution,
        )
        return gen(cube_emitter)

    p.run_stage("dust_particles", dust_particles)

    def marine_snow_particles():
        gen = weather.FallingParticles(
            particles.MarineSnowFactory(randint(1e7)),
            distribution=weather.marine_snow_param_distribution,
        )
        return gen(cube_emitter)

    p.run_stage("marine_snow_particles", marine_snow_particles)

    def snow_particles():
        gen = weather.FallingParticles(
            particles.SnowflakeFactory(randint(1e7)),
            distribution=weather.snow_param_distribution,
        )
        return gen(overhead_emitter)

    p.run_stage("snow_particles", snow_particles)

    placeholders = list(
        itertools.chain.from_iterable(
            c.all_objects
            for c in bpy.data.collections
            if c.name.startswith("placeholders:")
        )
    )

    def add_simulated_river():
        return fluid.make_river(terrain_mesh, placeholders, output_folder=output_folder)

    p.run_stage("simulated_river", add_simulated_river, use_chance=False)

    def add_tilted_river():
        return fluid.make_tilted_river(
            terrain_mesh, placeholders, output_folder=output_folder
        )

    p.run_stage("tilted_river", add_tilted_river, use_chance=False)

    p.save_results(output_folder / "pipeline_coarse.csv")
    return {
        "height_offset": 0,
        "whole_bbox": None,
    }


@gin.configurable
def populate_scene(
    output_folder: Path, scene_seed: int, camera_rigs: list[bpy.types.Object], **params
):
    p = RandomStageExecutor(scene_seed, output_folder, params)

    primary_cams = [rig.children[0] for rig in camera_rigs]

    season = p.run_stage(
        "choose_season", trees.random_season, use_chance=False, default=[]
    )

    fire_cache_system = fluid.FireCachingSystem() if params.get("cached_fire") else None

    populated = {}
    populated["trees"] = p.run_stage(
        "populate_trees",
        use_chance=False,
        default=[],
        fn=lambda: placement.populate_all(
            trees.TreeFactory, primary_cams, season=season, vis_cull=4
        ),
    )  # ,
    # meshing_camera=camera, adapt_mesh_method='subdivide', cam_meshing_max_dist=8))
    populated["boulders"] = p.run_stage(
        "populate_boulders",
        use_chance=False,
        default=[],
        fn=lambda: placement.populate_all(
            rocks.BoulderFactory, primary_cams, vis_cull=3
        ),
    )  # ,
    # meshing_camera=camera, adapt_mesh_method='subdivide', cam_meshing_max_dist=8))
    populated["bushes"] = p.run_stage(
        "populate_bushes",
        use_chance=False,
        fn=lambda: placement.populate_all(
            trees.BushFactory, primary_cams, vis_cull=1, adapt_mesh_method="subdivide"
        ),
    )
    p.run_stage(
        "populate_kelp",
        use_chance=False,
        fn=lambda: placement.populate_all(
            monocot.KelpMonocotFactory, primary_cams, vis_cull=5
        ),
    )
    populated["cactus"] = p.run_stage(
        "populate_cactus",
        use_chance=False,
        fn=lambda: placement.populate_all(
            cactus.CactusFactory, primary_cams, vis_cull=6
        ),
    )
    p.run_stage(
        "populate_clouds",
        use_chance=False,
        fn=lambda: placement.populate_all(
            cloud.CloudFactory, primary_cams, dist_cull=None, vis_cull=None
        ),
    )
    p.run_stage(
        "populate_glowing_rocks",
        use_chance=False,
        fn=lambda: placement.populate_all(
            rocks.GlowingRocksFactory, primary_cams, dist_cull=None, vis_cull=None
        ),
    )

    populated["cached_fire_trees"] = p.run_stage(
        "populate_cached_fire_trees",
        use_chance=False,
        default=[],
        fn=lambda: placement.populate_all(
            fluid.CachedTreeFactory,
            primary_cams,
            season=season,
            vis_cull=4,
            dist_cull=70,
            cache_system=fire_cache_system,
        ),
    )
    populated["cached_fire_boulders"] = p.run_stage(
        "populate_cached_fire_boulders",
        use_chance=False,
        default=[],
        fn=lambda: placement.populate_all(
            fluid.CachedBoulderFactory,
            primary_cams,
            vis_cull=3,
            dist_cull=70,
            cache_system=fire_cache_system,
        ),
    )
    populated["cached_fire_bushes"] = p.run_stage(
        "populate_cached_fire_bushes",
        use_chance=False,
        fn=lambda: placement.populate_all(
            fluid.CachedBushFactory,
            primary_cams,
            vis_cull=1,
            adapt_mesh_method="subdivide",
            cache_system=fire_cache_system,
        ),
    )
    populated["cached_fire_cactus"] = p.run_stage(
        "populate_cached_fire_cactus",
        use_chance=False,
        fn=lambda: placement.populate_all(
            fluid.CachedCactusFactory,
            primary_cams,
            vis_cull=6,
            cache_system=fire_cache_system,
        ),
    )

    grime_selection_funcs = {
        "trees": scatter_lower,
        "boulders": scatter_upward,
    }
    grime_types = {
        "slime_mold": slime_mold.SlimeMold,
        "lichen": lichen.Lichen,
        "ivy": ivy.Ivy,
        "mushroom": ground_mushroom.Mushrooms,
        "moss": moss.MossCover,
    }

    def apply_grime(grime_type, surface_cls):
        surface_fac = surface_cls()
        for (
            target_type,
            results,
        ) in populated.items():
            selection_func = grime_selection_funcs.get(target_type, None)
            for fac_seed, fac_pholders, fac_assets in results:
                if len(fac_pholders) == 0:
                    continue
                for inst_seed, obj in fac_assets:
                    with FixedSeed(int_hash((grime_type, fac_seed, inst_seed))):
                        p_k = f"{grime_type}_on_{target_type}_per_instance_chance"
                        if uniform() > params.get(p_k, 0.4):
                            continue
                        logger.debug("Applying {surface_fac} on {obj}")
                        surface_fac.apply(obj, selection=selection_func)

    for grime_type, surface_cls in grime_types.items():
        p.run_stage(grime_type, lambda: apply_grime(grime_type, surface_cls))

    def apply_snow_layer(surface_cls):
        surface_fac = surface_cls()
        for (
            target_type,
            results,
        ) in populated.items():
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
        "beetles": creatures.BeetleFactory,
        "bird": creatures.BirdFactory,
        "carnivore": creatures.CarnivoreFactory,
        "crab": creatures.CrabFactory,
        "crustacean": creatures.CrustaceanFactory,
        "dragonfly": creatures.DragonflyFactory,
        "fish": creatures.FishFactory,
        "flyingbird": creatures.FlyingBirdFactory,
        "herbivore": creatures.HerbivoreFactory,
        "snake": creatures.SnakeFactory,
    }
    for k, fac in creature_facs.items():
        p.run_stage(
            f"populate_{k}",
            use_chance=False,
            fn=lambda: placement.populate_all(fac, cameras=None),
        )

    fire_warmup = params.get("fire_warmup", 50)
    simulation_duration = (
        bpy.context.scene.frame_end - bpy.context.scene.frame_start + fire_warmup
    )

    def set_fire(assets):
        objs = [o for *_, a in assets for _, o in a]
        with butil.EnableParentCollections(objs):
            fluid.set_fire_to_assets(
                assets,
                bpy.context.scene.frame_start - fire_warmup,
                simulation_duration,
                output_folder,
            )

    p.run_stage(
        "trees_fire_on_the_fly", set_fire, populated["trees"], prereq="populate_trees"
    )
    p.run_stage(
        "bushes_fire_on_the_fly",
        set_fire,
        populated["bushes"],
        prereq="populate_bushes",
    )
    p.run_stage(
        "boulders_fire_on_the_fly",
        set_fire,
        populated["boulders"],
        prereq="populate_boulders",
    )
    p.run_stage(
        "cactus_fire_on_the_fly",
        set_fire,
        populated["cactus"],
        prereq="populate_cactus",
    )

    p.save_results(output_folder / "pipeline_fine.csv")


def main(args):
    scene_seed = init.apply_scene_seed(args.seed)
    mandatory_exclusive = [Path("infinigen_examples/configs_nature/scene_types")]
    init.apply_gin_configs(
        configs=["base_nature.gin"] + args.configs,
        overrides=args.overrides,
        config_folders="infinigen_examples/configs_nature",
        mandatory_folders=mandatory_exclusive,
        mutually_exclusive_folders=mandatory_exclusive,
    )

    execute_tasks.main(
        compose_scene_func=compose_nature,
        populate_scene_func=populate_scene,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        task=args.task,
        task_uniqname=args.task_uniqname,
        scene_seed=scene_seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=Path)
    parser.add_argument("--input_folder", type=Path, default=None)
    parser.add_argument(
        "-s", "--seed", default=None, help="The seed used to generate the scene"
    )
    parser.add_argument(
        "-t",
        "--task",
        nargs="+",
        default=["coarse"],
        choices=[
            "coarse",
            "populate",
            "fine_terrain",
            "ground_truth",
            "render",
            "mesh_save",
            "export",
        ],
    )
    parser.add_argument(
        "-g",
        "--configs",
        nargs="+",
        default=["base"],
        help="Set of config files for gin (separated by spaces) "
        "e.g. --gin_config file1 file2 (exclude .gin from path)",
    )
    parser.add_argument(
        "-p",
        "--overrides",
        nargs="+",
        default=[],
        help="Parameter settings that override config defaults "
        "e.g. --gin_param module_1.a=2 module_2.b=3",
    )
    parser.add_argument("--task_uniqname", type=str, default=None)
    parser.add_argument("-d", "--debug", type=str, nargs="*", default=None)

    args = init.parse_args_blender(parser)

    logging.getLogger("infinigen").setLevel(logging.INFO)
    logging.getLogger("infinigen.core.nodes.node_wrangler").setLevel(logging.CRITICAL)

    if args.debug is not None:
        for name in logging.root.manager.loggerDict:
            if not name.startswith("infinigen"):
                continue
            if len(args.debug) == 0 or any(name.endswith(x) for x in args.debug):
                logging.getLogger(name).setLevel(logging.DEBUG)

    main(args)

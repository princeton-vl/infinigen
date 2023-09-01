# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import os
import sys
from itertools import chain
from pathlib import Path
from numpy.random import uniform, normal, randint
from mathutils import Vector
import logging
from infinigen.core.util.math import clip_gaussian

from infinigen.core.nodes.node_wrangler import (
    Nodes,
    NodeWrangler,
    infer_input_socket,
    infer_output_socket,
)
import bpy
import numpy as np


from infinigen.assets.materials import water, lava

from infinigen.assets.fluid import duplication_geomod
from infinigen.assets.materials import blackbody_shader, waterfall_material, smoke_material
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.logging import Timer

import gin

from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)

FLUID_INITIALIZED = False

def check_initalize_fluids():
    if FLUID_INITIALIZED:
        return
    bpy.ops.flip_fluid_operators.complete_installation()

# find next available number for fluid cache folder
def find_available_cache(cache_folder):
    Path(cache_folder).mkdir(parents=True, exist_ok=True)
    contents = os.listdir(cache_folder)
    cache_int = 1
    while True:
        cache_int_str = str(cache_int)
        if cache_int_str not in contents:
            return cache_int_str
        cache_int += 1


# whitewater particles for mantaflow
@gin.configurable
def create_spray_particles(num_particles=7):
    objs = []
    for i in range(num_particles):
        x_rot = np.random.uniform(0, 2 * np.pi)
        y_rot = np.random.uniform(0, 2 * np.pi)
        z_rot = np.random.uniform(0, 2 * np.pi)
        rot = (x_rot, y_rot, z_rot)

        radius = 0.5
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
        z = np.random.uniform(-radius, radius)
        loc = (x, y, z)

        mean_size = 0.13
        size = np.random.uniform(mean_size - 0.05, mean_size + 0.05)
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=1, scale=(size, size, size), rotation=rot, location=loc
        )
        objs.append(bpy.context.object)

    return butil.group_in_collection(objs, "spray_particles")


# mantaflow liquid domain
@gin.configurable
def create_liquid_domain(
    location,
    start_frame,
    fluid_type="water",
    size=1,
    resolution=64,
    simulation_duration=100,
    waterfall=False,
    dimensions=None,
    output_folder=None,
):
    check_initalize_fluids()
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.object
    if dimensions:
        obj.dimensions = dimensions
    mod = obj.modifiers.new("Fluid", type="FLUID")
    mod.fluid_type = "DOMAIN"
    settings = mod.domain_settings
    settings.resolution_max = resolution
    settings.domain_type = "LIQUID"
    settings.use_mesh = True
    settings.cache_type = "ALL"
    settings.use_diffusion = True
    if output_folder:
        cache_folder = os.path.join(output_folder, "cache")
        settings.cache_directory = os.path.join(
            cache_folder, find_available_cache(cache_folder)
        )
    else:
        cache_folder = os.path.join(os.getcwd(), "cache")
        settings.cache_directory = os.path.join(
            cache_folder, find_available_cache(cache_folder)
        )
    settings.cache_frame_end = start_frame + simulation_duration
    settings.cache_frame_start = start_frame

    # absorb in the borders
    settings.use_collision_border_back = False
    settings.use_collision_border_bottom = False
    settings.use_collision_border_front = False
    settings.use_collision_border_left = False
    settings.use_collision_border_right = False
    settings.use_collision_border_top = False


    if fluid_type == "water":
        if waterfall:
            waterfall_material.apply(obj)
            settings.use_spray_particles = True
            settings.use_foam_particles = True
            settings.use_bubble_particles = True
            spray_collection = create_spray_particles()
            foam_settings = obj.particle_systems["Foam"].particles.data.settings
            bubble_settings = obj.particle_systems["Bubbles"].particles.data.settings
            spray_settings = obj.particle_systems["Spray"].particles.data.settings
            foam_settings.render_type = "COLLECTION"
            bubble_settings.render_type = "COLLECTION"
            spray_settings.render_type = "COLLECTION"
            foam_settings.instance_collection = spray_collection
            bubble_settings.instance_collection = spray_collection
            spray_settings.instance_collection = spray_collection
            foam_settings.use_collection_pick_random = True
            bubble_settings.use_collection_pick_random = True
            spray_settings.use_collection_pick_random = True

        else:
            water.apply(obj)
    elif fluid_type == "lava":
        lava.apply(obj)
        settings.use_viscosity = True
        settings.viscosity_exponent = 1
        settings.surface_tension = 0.250

    return obj


@gin.configurable
def create_liquid_flow(
    location,
    fluid_type="water",
    size=0.1,
    flow_behavior="INFLOW",
    is_planar=False,
    z_velocity=4,
):
    check_initalize_fluids()
    if is_planar:
        pass
    else:
        bpy.ops.mesh.primitive_ico_sphere_add(location=location, radius=size)
    obj = bpy.context.object
    mod = obj.modifiers.new("Fluid", type="FLUID")
    mod.fluid_type = "FLOW"
    settings = mod.flow_settings
    settings.flow_behavior = flow_behavior
    settings.flow_type = "LIQUID"

    settings.use_initial_velocity = True
    settings.velocity_coord[2] = z_velocity

    return obj


@gin.configurable
def make_liquid_effector(obj):
    check_initalize_fluids()
    mod = obj.modifiers.new("Fluid", type="FLUID")
    mod.fluid_type = "EFFECTOR"
    settings = mod.effector_settings
    settings.use_plane_init = True


@gin.configurable
def add_field(location, noise=None, strength=None):
    check_initalize_fluids()
    logger.info("adding field")
    bpy.ops.object.select_all(action="DESELECT")
    field = bpy.data.objects.new("turbulence", None)

    obj = field
    bpy.context.collection.objects.link(obj)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    obj.location = location

    bpy.ops.object.forcefield_toggle()

    if noise:
        obj.field.noise = noise
    else:
        obj.field.noise = uniform(0, 0.2)
    if strength:
        obj.field.strength = strength
    else:
        obj.field.strength = uniform(0, 0.2)
    logger.info(f"field noise: {obj.field.noise}, field strength: {obj.field.strength}")

    return obj


@gin.configurable
def create_gas_domain(
    location,
    start_frame,
    fluid_type="fire_and_smoke",
    size=1,
    resolution=64,
    simulation_duration=100,
    adaptive_domain = True,
    output_folder=None,
    noise_scale=3,
    dissolve_speed=0,
    flame_vorticity = None,
    vorticity = None
):
    check_initalize_fluids()
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.object
    set_gas_domain_settings(obj, start_frame, fluid_type, resolution, simulation_duration, adaptive_domain, output_folder, noise_scale, dissolve_speed, flame_vorticity, vorticity)
    return obj


# intended to be used with quick smoke
@gin.configurable
def set_gas_domain_settings(
    obj,
    start_frame,
    fluid_type="fire_and_smoke",
    resolution=32,
    simulation_duration=30,
    adaptive_domain=True,
    output_folder=None,
    noise_scale=2,
    noise_strength = None,
    dissolve_speed=0,
    flame_vorticity = None,
    vorticity = None,
):
    check_initalize_fluids()
    if "Fluid" not in obj.modifiers:
        mod = obj.modifiers.new("Fluid", type="FLUID")
    else:
        mod = obj.modifiers["Fluid"]
    mod.fluid_type = "DOMAIN"
    settings = mod.domain_settings
    settings.resolution_max = resolution
    settings.domain_type = "GAS"
    settings.cache_type = "ALL"
    settings.use_noise = True
    settings.noise_scale = noise_scale
    if noise_strength:
        settings.noise_strength = noise_strength
    else:
        settings.noise_strength = uniform(0.5, 1)
    if vorticity:
        settings.vorticity = vorticity
    else:
        settings.vorticity = uniform(0, 0.1)
    settings.use_adaptive_domain = adaptive_domain
    settings.cache_frame_end = start_frame + simulation_duration
    settings.cache_frame_start = start_frame
    if output_folder:
        cache_folder = os.path.join(output_folder, "cache")
        settings.cache_directory = os.path.join(
            cache_folder, find_available_cache(cache_folder)
        )
    else:
        cache_folder = os.path.join(os.getcwd(), "cache")
        settings.cache_directory = os.path.join(
            cache_folder, find_available_cache(cache_folder)
        )

    settings.use_collision_border_back = False
    settings.use_collision_border_bottom = False
    settings.use_collision_border_front = False
    settings.use_collision_border_left = False
    settings.use_collision_border_right = False
    settings.use_collision_border_top = False

    if dissolve_speed > 0:
        settings.use_dissolve_smoke = True
        settings.dissolve_speed = dissolve_speed

    if fluid_type == "fire_and_smoke":
        if flame_vorticity:
            settings.flame_vorticity = flame_vorticity
        else:
            settings.flame_vorticity = uniform(0.45, 0.55)
        settings.burning_rate = uniform(0.50, 0.80)
        blackbody_shader.apply(obj)
    elif fluid_type == "smoke":
        smoke_material.apply(obj)
        settings.alpha = 1.0
    elif fluid_type == "fire":
        if flame_vorticity:
            settings.flame_vorticity = flame_vorticity
        else:
            settings.flame_vorticity = uniform(0.45, 0.55)
        settings.burning_rate = uniform(0.50, 0.80)
        blackbody_shader.apply(obj)
    else:
        raise ValueError

    return obj


@gin.configurable
def create_gas_flow(location, fluid_type="fire_and_smoke", size=0.1, fuel_amount=None):

    check_initalize_fluids()

    bpy.ops.mesh.primitive_ico_sphere_add(radius=size, location=location)
    obj = bpy.context.object
    mod = obj.modifiers.new("Fluid", type="FLUID")
    mod.fluid_type = "FLOW"
    settings = mod.flow_settings
    settings.flow_behavior = "INFLOW"
    if fluid_type == "fire_and_smoke":
        if fuel_amount:
            settings.fuel_amount = fuel_amount
        else:
            settings.fuel_amount = 4 * np.random.rand()
        settings.flow_type = "BOTH"
    elif fluid_type == "smoke":
        settings.flow_type = "SMOKE"
    elif fluid_type == "fire":
        if fuel_amount:
            settings.fuel_amount = fuel_amount
        else:
            settings.fuel_amount = 4 * np.random.rand()
        settings.flow_type = "FIRE"
    else:
        raise ValueError

    return obj


@gin.configurable
def set_gas_flow_settings(obj, fluid_type="fire_and_smoke", fuel_amount=None):

    check_initalize_fluids()

    if "Fluid" not in obj.modifiers:
        mod = obj.modifiers.new("Fluid", type="FLUID")
    else:
        mod = obj.modifiers["Fluid"]
    mod.fluid_type = "FLOW"
    settings = mod.flow_settings
    settings.flow_behavior = "INFLOW"
    settings.surface_distance = uniform(0.5, 1.0)
    if fluid_type == "fire_and_smoke":
        if fuel_amount:
            settings.fuel_amount = fuel_amount
        else:
            settings.fuel_amount = uniform(0.8, 1.2)
        settings.flow_type = "BOTH"
    elif fluid_type == "smoke":
        settings.flow_type = "SMOKE"
    elif fluid_type == "fire":
        if fuel_amount:
            settings.fuel_amount = fuel_amount
        else:
            settings.fuel_amount = uniform(0.8, 1.2)
        settings.flow_type = "FIRE"
    else:
        raise ValueError

    return obj


def obj_bb_minmax(obj):
    min_co = Vector((np.inf, np.inf, np.inf))
    max_co = Vector((-np.inf, -np.inf, -np.inf))
    for i in range(0, 8):
        bb_vec = obj.matrix_world @ Vector(obj.bound_box[i])

        min_co[0] = min(bb_vec[0], min_co[0])
        min_co[1] = min(bb_vec[1], min_co[1])
        min_co[2] = min(bb_vec[2], min_co[2])
        max_co[0] = max(bb_vec[0], max_co[0])
        max_co[1] = max(bb_vec[1], max_co[1])
        max_co[2] = max(bb_vec[2], max_co[2])
    return min_co, max_co


def get_instanced_part(obj):
    for child in obj.children:
        if len(child.data.polygons) == 0:
            return child
    return None


def fire_dfs(obj, fluid_type):
    if obj.type == "MESH":
        obj.select_set(True)
        set_gas_flow_settings(obj, fluid_type=fluid_type)

    for child in obj.children:
        fire_dfs(child, fluid_type)


def identify_instance_parent_col(instance_obj):
    mod = None
    for m in instance_obj.modifiers:
        if m.type == "NODES":
            mod = m
    bpy.ops.object.modifier_set_active(modifier=mod.name)
    col = mod.node_group.nodes["Group"].inputs[2].default_value
    return col


def decimate_and_realize_instances(instance_obj, parent_col):
    objs_to_copy = parent_col.objects
    copied_objs = []

    with Timer("Cloning"):
        # crashes the normal way of writing this, no idea why
        for i in range(len(objs_to_copy)):
            obj = objs_to_copy[i]
            copied_objs.append(
                deep_clone_obj(obj, keep_modifiers=True, keep_materials=True)
            )

        parent_col_copy = butil.group_in_collection(copied_objs, "cloned_parent_col")

    with Timer("Decimating"):
        # decimate
        for o in parent_col_copy.objects:
            _, mod = butil.modify_mesh(o, "DECIMATE", return_mod=True, apply=False)
            mod.decimate_type = "COLLAPSE"
            mod.ratio = 0
            _, mod = butil.modify_mesh(o, "DECIMATE", return_mod=True, apply=False)
            mod.decimate_type = "DISSOLVE"
            mod.use_dissolve_boundaries = True
            mod.angle_limit = 3.1  # 2.6
            # mod.ratio = 0
            # bpy.ops.object.modifier_apply(modifier=mod.name)

    with Timer("Copying Instance"):
        # copy instance
        instance_obj_clone = deep_clone_obj(
            instance_obj, keep_modifiers=True, keep_materials=True
        )
        bpy.context.collection.objects.unlink(instance_obj_clone)
        bpy.data.scenes["Scene"].collection.objects.link(instance_obj_clone)
        # not very general, works on trees
        butil.select_none()
        butil.select(instance_obj_clone)
        bpy.context.view_layer.objects.active = instance_obj_clone
        mod = None
        for m in instance_obj_clone.modifiers:
            if m.type == "NODES":
                mod = m
        bpy.ops.object.modifier_set_active(modifier=mod.name)
        bpy.ops.object.geometry_node_tree_copy_assign()
        mod.node_group.nodes["Group"].inputs[2].default_value = parent_col_copy

    # mod = None
    # for m in instance_obj_clone.modifiers:
    #     if m.type == 'NODES':
    #         mod = m

    with Timer("Realizing Instances"):
        # realize instances
        ng = mod.node_group
        nw = NodeWrangler(ng)
        output_node = nw.nodes["Group Output"]
        input_node = output_node.inputs["Geometry"].links[0].from_socket.node
        realize_node = nw.new_node(
            Nodes.RealizeInstances,
            input_kwargs={"Geometry": input_node.outputs["Geometry"]},
        )

        input_socket = infer_input_socket(output_node, "Geometry")
        output_socket = infer_output_socket(realize_node)
        nw.connect_input(input_socket, output_socket)

    # we need to refresh for some reason
    instance_obj_clone.hide_viewport = True
    instance_obj_clone.hide_viewport = False

    bpy.ops.object.modifier_apply(modifier=mod.name)
    logger.debug(len(instance_obj_clone.data.polygons))

    instance_obj_clone.hide_render = True
    parent_col_copy.hide_render = True
    parent_col_copy.hide_viewport = True

    return instance_obj_clone


@gin.configurable
def set_obj_on_fire(
    obj,
    start_frame,
    resolution=300,
    simulation_duration=30,
    fluid_type="fire_and_smoke",
    adaptive_domain=True,
    add_turbulence=False,
    dimensions=None,
    output_folder=None,
    noise_scale=2,
    estimate_domain=False,
    turbulence_strength=None,
    turbulence_noise=None,
    dissolve_speed=25,
    dom_scale=1,
):
    check_initalize_fluids()

    dissolve_speed += np.random.randint(-3, 4)
    if add_turbulence:
        add_field(
            (
                obj.matrix_world.translation[0],
                obj.matrix_world.translation[1],
                obj.matrix_world.translation[2] + uniform(2, 3),
            ),
            turbulence_noise, 
            turbulence_strength
        )

    butil.select_none()
    fire_dfs(obj, fluid_type)
    
    # disabled for now
    # with Timer('Decimating and Realizing Instance'):
    #     instanced_obj = get_instanced_part(obj)
    #     instance_obj_clone = None
    #     if instanced_obj:
    #         parent_col = identify_instance_parent_col(instanced_obj)
    #         instance_obj_clone = decimate_and_realize_instances(instanced_obj, parent_col)
    #         instance_obj_clone.select_set(True)
    #         set_gas_flow_settings(instance_obj_clone, fluid_type= fluid_type)

    with Timer("Baking Fire"):
        # this changes fluid type
        if estimate_domain:
            dimensions = estimate_smoke_domain(obj, start_frame, simulation_duration)
            bpy.ops.mesh.primitive_cube_add()
            dom = bpy.context.object
            dom.dimensions = dimensions
        else:
            bpy.ops.object.quick_smoke()
            fire_dfs(obj, fluid_type)
            dom = bpy.context.object

        dom.scale *= dom_scale
        min_co, max_co = obj_bb_minmax(dom)
        dom.matrix_world.translation[2] += -min_co[2] - 1
        set_gas_domain_settings(
            dom,
            start_frame,
            resolution=resolution,
            simulation_duration=simulation_duration,
            fluid_type=fluid_type,
            adaptive_domain=adaptive_domain,
            output_folder=output_folder,
            noise_scale=noise_scale,
            dissolve_speed=dissolve_speed,
        )
        bpy.context.view_layer.objects.active = dom
        if dimensions:
            dom.dimensions = dimensions
        bpy.ops.fluid.bake_all()

    # if instance_obj_clone:
    #     butil.delete(instance_obj_clone)

    dom["fire_system_type"] = "domain"
    obj["fire_system_type"] = "obj"
    dom["fire_obj"] = obj
    obj["fire_domain"] = dom

    return dom


@gin.configurable
def generate_waterfall(
    output_folder,
    start_frame,
    resolution=300,
    simulation_duration=30,
    fluid_type="water",
):

    check_initalize_fluids()

    seed = np.random.randint(10000)

    bpy.ops.mesh.landscape_add(
        ant_terrain_name="Landscape",
        land_material="",
        water_material="",
        texture_block="",
        at_cursor=True,
        smooth_mesh=True,
        tri_face=False,
        sphere_mesh=False,
        subdivision_x=128,
        subdivision_y=128,
        mesh_size=2,
        mesh_size_x=2,
        mesh_size_y=2,
        random_seed=seed,
        noise_offset_x=0,
        noise_offset_y=-0.88,
        noise_offset_z=3.72529e-09,
        noise_size_x=2,
        noise_size_y=2,
        noise_size_z=1,
        noise_size=1,
        noise_type="marble_noise",
        basis_type="VORONOI_F2F1",
        vl_basis_type="BLENDER",
        distortion=0.5,
        hard_noise="0",
        noise_depth=7,
        amplitude=0.5,
        frequency=2,
        dimension=1,
        lacunarity=2,
        offset=1,
        gain=1,
        marble_bias="0",
        marble_sharp="0",
        marble_shape="6",
        height=1.8,
        height_invert=False,
        height_offset=-0.15,
        fx_mixfactor=0,
        fx_mix_mode="0",
        fx_type="0",
        fx_bias="0",
        fx_turb=0,
        fx_depth=0,
        fx_amplitude=0.5,
        fx_frequency=2,
        fx_size=1,
        fx_loc_x=0,
        fx_loc_y=0,
        fx_height=0.5,
        fx_invert=False,
        fx_offset=0,
        edge_falloff="0",
        falloff_x=25,
        falloff_y=25,
        edge_level=0,
        maximum=1.25,
        minimum=0,
        vert_group="",
        strata=11,
        strata_type="0",
        water_plane=False,
        water_level=0.01,
        remove_double=False,
        show_main_settings=True,
        show_noise_settings=True,
        show_displace_settings=True,
        refresh=True,
        auto_refresh=True,
    )
    terrain = bpy.context.object
    terrain.scale = (5, 5, 5)

    make_liquid_effector(terrain)

    obj = create_liquid_flow(
        location=(0, 2.4, 6.5), fluid_type=fluid_type, size=0.5, flow_behavior="INFLOW"
    )
    dom = create_liquid_domain(
        location=(0, 0, 4),
        fluid_type=fluid_type,
        size=10,
        resolution=resolution,
        cache_frame_end=simulation_duration,
    )
    if fluid_type == "lava":
        set_fluid_to_smoke(
            dom, start_frame, resolution=300, simulation_duration=simulation_duration
        )

    bpy.ops.fluid.bake_all()
    bpy.ops.wm.save_mainfile(filepath=output_folder)


@gin.configurable
def import_obj_simulate(
    output_folder,
    obj_filepath,
    source_size,
    source_relative_pos,
    domain_size=10,
    liquid_type="water",
    resolution=300,
    simulation_duration=50,
):

    check_initalize_fluids()

    # assuming we are importing to the origin
    bpy.ops.import_scene.obj(filepath=obj_filepath)
    terrain = bpy.context.selected_objects[0]
    print(terrain, terrain.name)
    make_liquid_effector(terrain)
    obj = create_liquid_flow(
        location=source_relative_pos,
        fluid_type=liquid_type,
        size=source_size,
        flow_behavior="INFLOW",
    )
    dom = create_liquid_domain(
        location=(0, 0, 0),
        fluid_type=liquid_type,
        size=domain_size,
        resolution=resolution,
        cache_frame_end=simulation_duration,
    )
    bpy.ops.fluid.bake_all()
    bpy.ops.wm.save_mainfile(filepath=output_folder)


def find_root(node):
    if node.parent == None:
        return node
    return find_root(node.parent)


@gin.configurable
def set_fire_to_assets(assets, start_frame, simulation_duration, output_folder=None, max_fire_assets=1):
    
    check_initalize_fluids()

    if len(assets) == 0:
        return

    # sort by distance
    obj_dist = []
    obj_vis_dist = []
    for _, (fac_seed, pholders, new_assets) in enumerate(assets):
        for j, (inst_seed, obj) in enumerate(new_assets):
            obj_dist.append((abs(obj["dist"]), obj))
            obj_vis_dist.append((abs(obj["vis_dist"]), obj))

    obj_dist.sort(key=lambda x: x[0])
    obj_vis_dist.sort(key=lambda x: x[0])

    if len(obj_dist) == 0:
        return

    for i in range(max_fire_assets):

        closest = obj_dist[i]
        obj = closest[1]
        logger.info(f"Setting fire to {i=} {obj.name=}")

        set_obj_on_fire(
            obj,
            start_frame,
            simulation_duration=simulation_duration,
            noise_scale=2,
            add_turbulence=True,
            adaptive_domain=True,
            output_folder=output_folder,
            estimate_domain=False,
            dom_scale=1.1,
        )

def duplicate_fluid_obj(obj):
    bpy.ops.mesh.primitive_plane_add()
    new_obj = bpy.context.object
    duplication_geomod.apply(new_obj=new_obj, old_obj=obj)
    return new_obj

@gin.configurable
def estimate_smoke_domain(obj, start_frame, simulation_duration):

    check_initalize_fluids()

    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.quick_smoke()
    dom = bpy.context.object
    dom.scale *= 4
    set_gas_domain_settings(
        dom,
        start_frame,
        resolution=128,
        simulation_duration=simulation_duration,
        fluid_type="smoke",
        adaptive_domain=True,
    )
    bpy.ops.fluid.bake_all()
    bpy.context.scene.frame_set(start_frame + simulation_duration)
    dimensions = dom.dimensions.copy()
    bpy.data.objects.remove(dom, do_unlink=True)

    return dimensions


@gin.configurable
def estimate_liquid_domain(
    location, start_frame, simulation_duration, fluid_type="water"
):

    check_initalize_fluids()

    source = create_liquid_flow(
        location,
        fluid_type=fluid_type,
        size=0.3,
        flow_behavior="INFLOW",
        is_planar=False,
        z_velocity=4,
    )
    dom = create_liquid_domain(
        location,
        start_frame,
        fluid_type=fluid_type,
        size=10,
        resolution=60,
        simulation_duration=simulation_duration,
        waterfall=False,
    )

    bpy.ops.fluid.bake_all()
    bpy.context.scene.frame_set(start_frame + simulation_duration)

    dimensions = dom.dimensions.copy()
    bpy.data.objects.remove(dom, do_unlink=True)
    bpy.data.objects.remove(source, do_unlink=True)
    return dimensions


@gin.configurable
def set_fluid_to_smoke(obj, start_frame, resolution=300, simulation_duration=30):
    
    check_initalize_fluids()

    new_obj = duplicate_fluid_obj(obj)
    set_obj_on_fire(
        new_obj,
        start_frame,
        resolution=resolution,
        simulation_duration=simulation_duration,
        fluid_type="smoke",
        adaptive_domain=False,
    )
    new_obj.hide_viewport = True
    new_obj.hide_render = True


def set_instanced_on_fire(instanced_obj):
    parent_col = identify_instance_parent_col(instanced_obj)
    instance_obj_clone = decimate_and_realize_instances(instanced_obj, parent_col)


def fire_smoke_ground_truth(domain):
    bpy.context.view_layer.update()
    translation = domain.matrix_world @ Vector(domain.bound_box[0])
    # assumes modifier name
    cache_dir = bpy.path.abspath(
        domain.modifiers["Fluid"].domain_settings.cache_directory
    )
    data_dir = os.path.join(cache_dir, "data")
    contents = [f for f in os.listdir(data_dir)]
    filepath = os.path.join(data_dir, contents[0])
    files = [{"name": f, "name": f} for f in contents]
    bpy.ops.object.volume_import(filepath=filepath, directory=data_dir, files=files)
    vol = bpy.context.object
    vol.location += translation
    vol.rotation_euler = domain.rotation_euler
    bpy.ops.mesh.primitive_plane_add()
    gt_mesh = bpy.context.object
    gt_mesh.name = "fire_gt_mesh"
    mod = gt_mesh.modifiers.new("volume_to_mesh", type="VOLUME_TO_MESH")
    mod.object = vol
    mod.use_smooth_shade = True

    gt_mesh["fire_system_type"] = "gt_mesh"
    gt_mesh["fire_parent"] = domain
    gt_mesh["fire_vol"] = vol

    vol["fire_system_type"] = "volume"
    vol["fire_gt"] = gt_mesh
    vol["fire_parent"] = domain

    domain["fire_system_type"] = "domain"
    domain["fire_gt"] = gt_mesh
    domain["fire_vol"] = vol

    vol.hide_render = True
    vol.hide_viewport = True
    gt_mesh.hide_render = True

    return gt_mesh, vol


def is_fire_in_scene():
    for obj in bpy.data.objects:
        if "Fluid" in obj.modifiers:
            mod = obj.modifiers["Fluid"]
            if mod.fluid_type == "DOMAIN" and mod.domain_settings.domain_type == "GAS":
                return True
    return False

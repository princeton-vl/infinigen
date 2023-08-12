# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import os
import sys
from pathlib import Path
from numpy.random import uniform
from mathutils import Vector

import bpy

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # This must be done BEFORE import cv2.

from infinigen.assets.materials import water, lava, river_water, new_whitewater
from infinigen.assets.materials import blackbody_shader, waterfall_material, smoke_material
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.logging import Timer

from infinigen.core.util import blender as butil
from infinigen.core.util.organization import AssetFile, Materials, Process  
from infinigen.core.util.blender import object_from_trimesh, SelectObjects
from infinigen.terrain.utils import Mesh
import cv2
import subprocess

from infinigen.assets.fluid.fluid import find_available_cache, obj_bb_minmax
import infinigen.assets.fluid.liquid_particle_material as liquid_particle_material
from numpy.random import normal as N

import gin
from infinigen.core.util.organization import Assets, LandTile


def get_objs_inside_domain(dom, objects):
    ls = []
    min_co = Vector((1e9, 1e9, 1e9))
    max_co = Vector((-1e9, -1e9, -1e9))
    min_co, max_co = obj_bb_minmax(dom)
    print(min_co, max_co)
    for obj in objects:
        if (
            obj.matrix_world.translation[0] < max_co[0]
            and obj.matrix_world.translation[1] < max_co[1]
            and obj.matrix_world.translation[2] < max_co[2]
            and obj.matrix_world.translation[0] > min_co[0]
            and obj.matrix_world.translation[1] > min_co[1]
            and obj.matrix_world.translation[2] > min_co[2]
        ):
            ls.append(obj)

    return ls


@gin.configurable
def create_flip_fluid_domain(
    location,
    start_frame=0,
    fluid_type="water",
    size=1,
    resolution=64,
    simulation_duration=100,
    dimensions=None,
    output_folder=None,
    particle_size=0.0045,
):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.object
    if dimensions:
        obj.dimensions = dimensions

    obj.name = "flip_fluid_domain"

    # CHANGE to choose auto domain
    set_flip_fluid_domain(
        obj,
        start_frame,
        resolution=resolution,
        simulation_duration=simulation_duration,
        output_folder=output_folder,
        particle_size=particle_size,
        fluid_type=fluid_type,
    )

    return obj


@gin.configurable
def set_flip_fluid_domain(
    dom,
    start_frame,
    fluid_type="water",
    size=1,
    resolution=64,
    simulation_duration=100,
    output_folder=None,
    particle_size=0.0045,
    wavecrest_emission_rate=225,
    turbulence_emission_rate=225,
    spray_emission_speed=1.2,
    max_whitewater_energy=2.2,
    subdivisions=2,
):
    print("fluid resolution", resolution)
    butil.select(dom)
    bpy.context.view_layer.objects.active = dom
    bpy.ops.flip_fluid_operators.flip_fluid_add()
    dom = bpy.context.object
    dom.flip_fluid.object_type = "TYPE_DOMAIN"
    dom.flip_fluid.domain.whitewater.enable_whitewater_simulation = True
    dom.flip_fluid.domain.whitewater.enable_foam = True
    dom.flip_fluid.domain.whitewater.enable_bubbles = True
    dom.flip_fluid.domain.whitewater.enable_spray = True
    dom.flip_fluid.domain.simulation.resolution = resolution
    dom.flip_fluid.domain.whitewater.wavecrest_emission_rate = wavecrest_emission_rate
    dom.flip_fluid.domain.whitewater.turbulence_emission_rate = turbulence_emission_rate
    dom.flip_fluid.domain.whitewater.spray_emission_speed = spray_emission_speed
    dom.flip_fluid.domain.whitewater.min_max_whitewater_energy_speed.value_max = (
        max_whitewater_energy + N()
    )
    dom.flip_fluid.domain.render.viewport_whitewater_pct = 100
    dom.flip_fluid.domain.render.only_display_whitewater_in_render = False
    dom.flip_fluid.domain.render.whitewater_particle_scale = particle_size
    dom.flip_fluid.domain.simulation.frame_range_mode = "FRAME_RANGE_CUSTOM"
    dom.flip_fluid.domain.simulation.frame_range_custom.value_min = start_frame
    dom.flip_fluid.domain.simulation.frame_range_custom.value_max = (
        start_frame + simulation_duration
    )
    dom.flip_fluid.domain.surface.subdivisions = subdivisions
    dom.flip_fluid.domain.world.enable_surface_tension = True
    dom.flip_fluid.domain.world.enable_sheet_seeding = True

    if output_folder:
        cache_folder = os.path.join(output_folder, "cache")
        dom.flip_fluid.domain.cache.cache_directory = os.path.join(
            cache_folder, find_available_cache(cache_folder)
        )
    else:
        cache_folder = os.path.join(os.getcwd(), "cache")
        dom.flip_fluid.domain.cache.cache_directory = os.path.join(
            cache_folder, find_available_cache(cache_folder)
        )

    print(f"cache folder = {cache_folder}")

    bpy.ops.flip_fluid_operators.helper_select_surface()
    surface = bpy.context.object
    if fluid_type == "water":
        water.apply(surface)
    elif fluid_type == "river_water":
        river_water.apply(surface)
    bpy.ops.flip_fluid_operators.helper_select_spray()
    spray = bpy.context.object
    new_whitewater.apply(spray)
    bpy.ops.flip_fluid_operators.helper_select_foam()
    foam = bpy.context.object
    new_whitewater.apply(foam)
    bpy.ops.flip_fluid_operators.helper_select_bubble()
    bubble = bpy.context.object
    new_whitewater.apply(bubble)

    if bpy.context.scene.frame_end < start_frame + simulation_duration:
        bpy.context.scene.frame_end = start_frame + simulation_duration


@gin.configurable
def create_flip_fluid_inflow(
    location, size=0.1, initial_velo=(0, 0, 2), fluid_type="water"
):
    bpy.ops.mesh.primitive_ico_sphere_add(location=location, radius=size)
    obj = bpy.context.object
    set_flip_fluid_inflow(obj, initial_velo=initial_velo)

    return obj


def set_flip_fluid_obstacle(obj, is_planar=True, thickness=1, friction=0.3):
    butil.select(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.flip_fluid_operators.flip_fluid_add()
    obj.flip_fluid.object_type = "TYPE_OBSTACLE"
    obj.flip_fluid.obstacle.friction = friction

    if is_planar:
        mod = obj.modifiers.new("Solidify", type="SOLIDIFY")
        mod.thickness = thickness


@gin.configurable
def set_flip_fluid_inflow(obj, initial_velo=(0, 0, 0), contrain_fluid_velo=True):
    butil.select(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.flip_fluid_operators.flip_fluid_add()
    bpy.context.object.flip_fluid.object_type = "TYPE_INFLOW"
    bpy.context.object.flip_fluid.inflow.inflow_velocity = initial_velo
    obj.flip_fluid.inflow.constrain_fluid_velocity = contrain_fluid_velo


def set_flip_fluid_outflow(obj):
    butil.select(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.flip_fluid_operators.flip_fluid_add()
    bpy.context.object.flip_fluid.object_type = "TYPE_OUTFLOW"


# deprecated
@gin.configurable
def obj_simulate(
    output_folder,
    obj_name,
    source_size,
    source_relative_pos,
    domain_size,
    domain_location,
    liquid_type="water",
    initial_velo=0,
    resolution=300,
    simulation_duration=50,
):
    cache_directory = str(Path(output_folder) / "cache")
    # assuming we are importing to the origin
    # bpy.ops.import_scene.obj(filepath=obj_filepath)
    terrain = bpy.data.objects[obj_name]
    print(terrain, terrain.name)
    set_flip_fluid_obstacle(terrain)
    obj = create_flip_fluid_inflow(
        location=source_relative_pos,
        fluid_type=liquid_type,
        size=source_size,
        initial_velo=initial_velo,
    )
    dom = create_flip_fluid_domain(
        location=domain_location,
        start_frame=0,
        output_folder=cache_directory,
        fluid_type=liquid_type,
        size=domain_size,
        resolution=resolution,
        simulation_duration=simulation_duration,
    )

    print(f"resolution: {resolution}")
    with Timer("baking"):
        bpy.ops.flip_fluid_operators.bake_fluid_simulation_cmd()
    Path(output_folder).mkdir(exist_ok=True)
    bpy.ops.wm.save_mainfile(filepath=str(Path(output_folder) / "fluid.blend"))
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.objects.remove(dom, do_unlink=True)


# deprecated
def asset_heightmap_simulate(
    folder, resolution=300, simulation_duration=50, AntLandscapeTileSize=10
):
    # now only do for eroded one
    prefix = f"{Process.Erosion}."
    input_heightmap_path = f"{folder}/{prefix}{AssetFile.Heightmap}.exr"
    image = cv2.imread(input_heightmap_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    obj = Mesh(heightmap=image, downsample=16, L=50).save_blender("terrain")
    domain_size = AntLandscapeTileSize
    land_tile_type = str(folder).split("/")[-2]
    if land_tile_type == LandTile.Cliff:
        source_size = AntLandscapeTileSize / 20
        source_relative_pos = (0, 24, 33)
        liquid_type = "water"  # Materials.Water
        domain_location = (0, 0, 4 * AntLandscapeTileSize / 10)
        initial_velo = (0, -5, 0)
    elif land_tile_type == LandTile.Volcano:
        N = image.shape[0]
        center_height = image[N // 2, N // 2]
        source_size = AntLandscapeTileSize / 50
        source_relative_pos = (0, 0, center_height + source_size * 2)
        liquid_type = Materials.Lava
        domain_location = (0, 0, 4 * AntLandscapeTileSize / 10)
        initial_velo = (0, 0, 7.5)
    elif land_tile_type == LandTile.River:
        source_size = AntLandscapeTileSize / 50
        source_relative_pos = (0, 24, 2)
        liquid_type = "water"  # Materials.Water
        domain_location = (0, 0, 4 * AntLandscapeTileSize / 10)
        initial_velo = (0, -10, 0)

    simulation_folder = str(folder / "fluid_simulation")
    subprocess.call(f"rm -rf '{simulation_folder}'", shell=True)
    obj_simulate(
        simulation_folder,
        obj.name,
        source_size,
        source_relative_pos,
        domain_size,
        domain_location,
        liquid_type,
        initial_velo,
        resolution=resolution,
        simulation_duration=simulation_duration,
    )
    bpy.data.objects.remove(obj, do_unlink=True)


@gin.configurable
def make_still_water(start_mesh, dom, terrain, initial_velocity=(0, 0, 0)):
    butil.modify_mesh(
        start_mesh, "BOOLEAN", apply=True, operation="INTERSECT", object=dom
    )
    butil.modify_mesh(
        start_mesh, "BOOLEAN", apply=True, operation="DIFFERENCE", object=terrain
    )
    butil.select(start_mesh)
    bpy.context.view_layer.objects.active = start_mesh
    bpy.ops.flip_fluid_operators.flip_fluid_add()
    start_mesh.flip_fluid.object_type = "TYPE_FLUID"
    start_mesh.flip_fluid.fluid.initial_velocity = initial_velocity


@gin.configurable
def add_wave_pusher(location, amplitude, half_period=30, simulation_duration=250):
    bpy.ops.mesh.primitive_cylinder_add(
        radius=1,
        depth=2,
        enter_editmode=False,
        align="WORLD",
        location=location,
        scale=(1, 1, 1),
    )
    pusher = bpy.context.object
    pusher.dimensions[2] = 10
    pusher.rotation_euler[0] = 1.5708
    set_flip_fluid_obstacle(pusher, is_planar=False)
    pusher.location[2] = location[2]
    keyframe = 1
    ind = 0
    while keyframe < simulation_duration:
        pusher.keyframe_insert(data_path="location", frame=keyframe)
        keyframe += half_period
        ind += 1
        pusher.location[2] = location[2] if ind % 2 == 0 else location[2] + amplitude
    return pusher


@gin.configurable
def obstacle_terrain(terrain, dom, friction=0.3, is_planar=True):
    smaller_terrain = butil.deep_clone_obj(terrain)
    # butil.modify_mesh(smaller_terrain, "BOOLEAN", apply = True, operation = "INTERSECT", object = dom)
    set_flip_fluid_obstacle(
        smaller_terrain, is_planar=is_planar, thickness=0.5, friction=friction
    )
    smaller_terrain.hide_render = True


def obstacle_dfs(obj, friction=0.3):
    if obj.type == "MESH":
        set_flip_fluid_obstacle(obj, is_planar=False, friction=friction)
    for child in obj.children:
        obstacle_dfs(child)


@gin.configurable
def make_beach(terrain, obstacles, location=(0, 0, 0), output_folder=None):
    bpy.data.objects["water"].hide_render = True
    bpy.data.objects["water"].hide_viewport = True
    dom = create_flip_fluid_domain(
        (location[0] - 2.5, location[1], location[2] + 0.6),
        start_frame=0,
        resolution=400,
        simulation_duration=250,
        dimensions=(15, 10, 2.3),
        output_folder=output_folder,
        particle_size=0.0035,
    )
    bpy.ops.mesh.primitive_cube_add(location=(-5.5, 0, -1))
    start_mesh = bpy.context.object
    start_mesh.dimensions = (9, 10, 2)
    make_still_water(start_mesh, dom, terrain)
    add_wave_pusher(location=(-9.5, 0, -1.4), amplitude=0.6)
    obstacle_terrain(terrain, dom, is_planar=False)
    obstacles_restricted = get_objs_inside_domain(dom, obstacles)
    for obst in obstacles_restricted:
        print(obst)
        obstacle_dfs(obst)
    with Timer("baking"):
        bpy.ops.flip_fluid_operators.bake_fluid_simulation_cmd()


@gin.configurable
def make_river(
    terrain,
    obstacles,
    dom_location=(0, -0.2, 0.3),
    output_folder=None,
    caustics=False,
    flow_velocity=3.2,
    start_frame=0,
    resolution=400,
    simulation_duration=250,
    dom_dimensions=(10.5, 24.5, 3.7),
    particle_size=0.004,
    liquid_type="river_water",
    inflow_location=(-0.52415, -11.58, -0.97735),
    inflow_dimensions=(6, 1.21, 2),
    outflow_location=(0, 11.5841, 0),
    outflow_dimensions=(6, 1.21, 2),
    terrain_friction=0.3,
):
    flow_velocity = flow_velocity + N()
    cloned_water = butil.deep_clone_obj(
        bpy.data.objects["liquid"], keep_modifiers=False, keep_materials=False
    )
    bpy.data.objects["liquid"].hide_render = True
    bpy.data.objects["liquid"].hide_viewport = True
    dom = create_flip_fluid_domain(
        dom_location,
        start_frame=start_frame,
        resolution=resolution,
        simulation_duration=simulation_duration,
        dimensions=dom_dimensions,
        output_folder=output_folder,
        particle_size=particle_size,
        fluid_type=liquid_type,
    )
    for i in range(6):
        dom.flip_fluid.domain.simulation.fluid_boundary_collisions[i] = False
    make_still_water(cloned_water, dom, terrain, initial_velocity=(0, flow_velocity, 0))

    bpy.ops.mesh.primitive_cube_add(location=inflow_location)
    inflow = bpy.context.object
    inflow.dimensions = inflow_dimensions
    butil.modify_mesh(inflow, "BOOLEAN", apply=True, operation="INTERSECT", object=dom)
    butil.modify_mesh(
        inflow, "BOOLEAN", apply=True, operation="DIFFERENCE", object=terrain
    )
    set_flip_fluid_inflow(inflow, initial_velo=(0, flow_velocity, 0))

    bpy.ops.mesh.primitive_cube_add(location=outflow_location)
    outflow = bpy.context.object
    outflow.dimensions = outflow_dimensions
    butil.modify_mesh(outflow, "BOOLEAN", apply=True, operation="INTERSECT", object=dom)
    butil.modify_mesh(
        outflow, "BOOLEAN", apply=True, operation="DIFFERENCE", object=terrain
    )
    set_flip_fluid_outflow(outflow)

    obstacle_terrain(terrain, dom, is_planar=False, friction=terrain_friction)
    obstacles_restricted = get_objs_inside_domain(dom, obstacles)
    for obst in obstacles_restricted:
        print("Setting obstacle: ", obst)
        obstacle_dfs(obst, friction=terrain_friction)

    if "scatter" in bpy.data.collections:
        print("scatter exists")
        for obj in bpy.data.collections["scatter"].objects:
            print("Setting scatter obstacle: ", obj)
            obstacle_dfs(obj, friction=terrain_friction)

    with Timer("baking"):
        bpy.ops.flip_fluid_operators.bake_fluid_simulation_cmd()


@gin.configurable
def make_tilted_river(
    terrain,
    obstacles,
    output_folder=None,
    dom_location=(0, 0, 1.76),
    start_frame=0,
    resolution=400,
    simulation_duration=250,
    dom_dimensions=(21.4, 47.7, 26.9),
    particle_size=0.007,
    liquid_type="river_water",
    inflow_location=(0, 21.77, 10.71),
    inflow_dimensions=(12.1, 2, 3),
    flow_velocity=-4,
    terrain_friction=0.1,
):
    bpy.data.objects["liquid"].hide_render = True
    bpy.data.objects["liquid"].hide_viewport = True
    dom = create_flip_fluid_domain(
        dom_location,
        start_frame=start_frame,
        resolution=resolution,
        simulation_duration=simulation_duration,
        dimensions=dom_dimensions,
        output_folder=output_folder,
        particle_size=particle_size,
        fluid_type=liquid_type,
    )
    dom.flip_fluid.domain.simulation.time_scale = 0.5
    for i in range(6):
        dom.flip_fluid.domain.simulation.fluid_boundary_collisions[i] = False

    bpy.ops.mesh.primitive_cube_add(location=inflow_location)
    inflow = bpy.context.object
    inflow.dimensions = inflow_dimensions
    butil.modify_mesh(inflow, "BOOLEAN", apply=True, operation="INTERSECT", object=dom)
    butil.modify_mesh(
        inflow, "BOOLEAN", apply=True, operation="DIFFERENCE", object=terrain
    )
    set_flip_fluid_inflow(inflow, initial_velo=(0, flow_velocity + N(), 0))

    obstacle_terrain(terrain, dom, friction=terrain_friction, is_planar=False)
    obstacles_restricted = get_objs_inside_domain(dom, obstacles)
    for obst in obstacles_restricted:
        print(obst)
        obstacle_dfs(obst, friction=terrain_friction)

    if "scatter" in bpy.data.collections:
        print("scatter exists")
        for obj in bpy.data.collections["scatter"].objects:
            print(obj)
            obstacle_dfs(obj, friction=terrain_friction)

    with Timer("baking"):
        bpy.ops.flip_fluid_operators.bake_fluid_simulation_cmd()


if __name__ == "__main__":
    butil.clear_scene(targets=[bpy.data.objects])
    ASSET_ENV_VAR = "INFINIGEN_ASSET_FOLDER"
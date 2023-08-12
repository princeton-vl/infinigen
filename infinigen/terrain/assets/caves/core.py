# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson, Zeyu Ma


from pathlib import Path

import bpy
import gin
import numpy as np
from numpy import ascontiguousarray as AC

import infinigen.terrain.mesh_to_sdf as mesh_to_sdf
from infinigen.terrain.utils import Mesh
from infinigen.core.util.blender import SelectObjects, ViewportMode
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.organization import AssetFile

from .geometry_utils import increment_step, pitch_up, yaw_clockwise
from .pcfg import generate_string


def get_all_verts():
    if bpy.ops.mesh.select_all.poll():
        bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    return bpy.context.active_object.data.vertices


def select_vert(idx: int):
    assert idx >= 0
    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    # assert False, [len(bpy.context.active_object.data.vertices), len(get_all_verts())]
    assert idx < len(
        obj.data.vertices), f"There are only {len(obj.data.vertices)} {len(get_all_verts())} verts, cannot select {idx}"
    obj.data.vertices[idx].select = True
    bpy.ops.object.mode_set(mode='EDIT')


def move_forward(current_dir):
    assert type(current_dir) == np.ndarray and current_dir.size == 3
    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": current_dir})


def trace_string(symbols, num_verts=1, current_idx=-1, current_dir=(0.5, 0., 0.)):
    bpy.ops.object.mode_set(mode='EDIT')
    current_dir = np.array(current_dir).flatten()
    angle_magnitude = 15
    while len(symbols) > 0:
        symbol = symbols.pop(0)
        if symbol == 'f':
            move_forward(current_dir)
            num_verts += 1
            current_idx = num_verts - 1
        elif symbol == 'r':
            current_dir = yaw_clockwise(current_dir, angle_magnitude)
        elif symbol == 'l':
            current_dir = yaw_clockwise(current_dir, -angle_magnitude)
        elif symbol == 'u':
            current_dir = pitch_up(current_dir, angle_magnitude)
        elif symbol == 'd':
            current_dir = pitch_up(current_dir, -angle_magnitude)
        elif symbol == 'o':
            angle_magnitude += 15
        elif symbol == 'a':
            angle_magnitude -= 15
        elif symbol == 'b':
            current_dir = increment_step(current_dir, 1)
        elif symbol == 's':
            current_dir = increment_step(current_dir, -1)
        elif symbol == 'n':  # do nothing
            pass
        elif symbol == '[':
            num_verts = trace_string(
                symbols, num_verts, current_idx, current_dir)
            select_vert(current_idx)
        elif symbol == ']':
            return num_verts
        else:
            raise Exception(f"Symbol not defined: {symbol}")

        if symbol in list('rlud'):
            angle_magnitude = 15


class Cave:

    def scale_verts(self, random_scaling_factor=0.0):  # 0.8 is a good number
        assert random_scaling_factor >= 0.0
        vertices = get_all_verts()
        bpy.ops.object.mode_set(mode='OBJECT')
        obj = bpy.context.active_object
        radii = 2*np.ones((len(vertices), 2))
        urn = np.random.rand(*radii.shape)*2 - 1
        radii *= np.exp(urn * random_scaling_factor)
        assert radii.min() > 0.05
        obj.data.skin_vertices[0].data.foreach_set('radius', radii.flatten())

    def add_subdivision(self, name: str, levels: int):
        assert name not in self.modifier_stack
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.modifier_add(type='SUBSURF')
        bpy.context.object.modifiers["Subdivision"].name = name
        bpy.context.object.modifiers[name].levels = levels
        self.modifier_stack.append(name)

    def remesh(self, name: str, voxel_size: float):
        assert name not in self.modifier_stack
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.modifier_add(type='REMESH')
        bpy.context.object.modifiers["Remesh"].name = name
        bpy.context.object.modifiers[name].voxel_size = voxel_size
        self.modifier_stack.append(name)

    def add_skin(self):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.object.modifier_add(type='SKIN')
        self.modifier_stack.append("Skin")

    def apply_modifiers(self):
        while len(self.modifier_stack) > 0:
            bpy.ops.object.modifier_apply(modifier=self.modifier_stack.pop(0))

    def add_to_collection(self, name):
        if bpy.data.collections.get(name) is None:
            bpy.data.collections.new(name=name)
            bpy.context.scene.collection.children.link(
                bpy.data.collections[name])
        obj = bpy.context.active_object
        bpy.ops.collection.objects_remove_all()
        bpy.data.collections[name].objects.link(obj)

    def add_lights(self, num_lights, power=100):

        np.random.shuffle(self.path_verts)
        for single_vert in self.path_verts[:num_lights]:
            # print("LIGHT AT", single_vert)
            bpy.ops.object.light_add(
                type='POINT', align='WORLD', location=single_vert, scale=(1, 1, 1))
            bpy.context.object.data.energy = power
            self.add_to_collection("AllPointLights")

            # bpy.ops.outliner.collection_drop()

    def make_active(self):
        bpy.context.view_layer.objects.active = bpy.data.objects[self.name]

    def __init__(self, name="Cave") -> None:
        self.modifier_stack = []
        bpy.ops.mesh.primitive_vert_add()
        bpy.context.active_object.name = name
        generated_string = generate_string(max_len=5000)
        # print(f"Using String", ''.join(generated_string))
        trace_string(['f']*2 + generated_string)


def add_cave(
    rescale,
    cave_z
):
    cave = Cave("Cave")
    cave.add_skin()
    cave.scale_verts(0.1)
    cave.add_subdivision("firstsub", 2)
    cave.remesh("remesh", 0.5)
    cave.apply_modifiers()
    with SelectObjects(bpy.data.objects["Cave"]):
        bpy.data.objects["Cave"].scale = (rescale, rescale, rescale)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        bpy.data.objects["Cave"].location = (0, 0, cave_z)
        bpy.ops.object.transform_apply(scale=True)
    cave.path_verts = np.array([v.co for v in get_all_verts()])
    # cave.add_lights(n_cave_light)
    return cave

@gin.configurable
def caves_asset(
    folder,
    N=128,
):
    folder.mkdir(parents=True, exist_ok=True)
    add_cave(rescale=1, cave_z=0)
    name = "Cave"
    obj = bpy.data.objects[name]
    with ViewportMode(obj, "EDIT"):
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bounding_box = np.array([v[:] for v in obj.bound_box])
    min_gen, max_gen = [0, 0, 0], [0, 0, 0]
    for j in range(3):
        min_gen[j], max_gen[j] = bounding_box[:, j].min(), bounding_box[:,j].max()
    bounding_box = np.array([min_gen, max_gen])
    dim = (bounding_box[1] - bounding_box[0])
    bounding_box[0] -= dim / 4
    bounding_box[1] += dim / 4
    cave_mesh = Mesh(obj=obj).to_trimesh()
    bpy.data.objects.remove(obj, do_unlink=True)
    x = np.linspace(bounding_box[0, 0], bounding_box[1, 0], N)
    y = np.linspace(bounding_box[0, 1], bounding_box[1, 1], N)
    z = np.linspace(bounding_box[0, 2], bounding_box[1, 2], N)
    query_points = np.zeros((N, N, N, 3))
    for j in range(N):
        query_points[j, :, :, 0] = x[j]
        query_points[:, j, :, 1] = y[j]
        query_points[:, :, j, 2] = z[j]
    query_points = query_points.reshape(-1, 3)
    voxels = mesh_to_sdf.mesh_to_sdf(cave_mesh, query_points, surface_point_method='sample').reshape((N, N, N))
    np.save(folder/"occupancy.npy", voxels)
    np.save(folder/"boundingbox.npy", bounding_box)
    (folder / AssetFile.Finish).touch()

def assets_to_data(folder):
    data = {}
    occupancies = np.load(folder/"occupancy.npy")
    N = occupancies.shape[0]
    data["occupancy"] = AC(occupancies.reshape(-1))
    data["bounding_box"] = AC(np.load(folder/"boundingbox.npy").reshape(-1))
    return N, data
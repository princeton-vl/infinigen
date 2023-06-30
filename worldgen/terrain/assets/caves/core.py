from pathlib import Path
import bpy
import gin
import mesh_to_sdf
import numpy as np
from numpy import ascontiguousarray as AC
from terrain.utils import Mesh
from util.blender import SelectObjects, ViewportMode
from util.math import FixedSeed
from util.organization import AssetFile

from .geometry_utils import increment_step, pitch_up, yaw_clockwise
from .pcfg import generate_string

        bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    # assert False, [len(bpy.context.active_object.data.vertices), len(get_all_verts())]
    assert idx < len(
        obj.data.vertices), f"There are only {len(obj.data.vertices)} {len(get_all_verts())} verts, cannot select {idx}"
    bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": current_dir})

def trace_string(symbols, num_verts=1, current_idx=-1, current_dir=(0.5, 0., 0.)):
    bpy.ops.object.mode_set(mode='EDIT')
        elif symbol == 'n':  # do nothing
            num_verts = trace_string(
                symbols, num_verts, current_idx, current_dir)
    def scale_verts(self, random_scaling_factor=0.0):  # 0.8 is a good number
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

            bpy.context.scene.collection.children.link(
                bpy.data.collections[name])

            # print("LIGHT AT", single_vert)
            bpy.ops.object.light_add(
                type='POINT', align='WORLD', location=single_vert, scale=(1, 1, 1))


    def __init__(self, name="Cave") -> None:
        generated_string = generate_string(max_len=5000)

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
import bpy

import numpy as np
import mathutils

from tqdm import trange, tqdm
from numpy.random import uniform, normal
from nodes.node_wrangler import Nodes, NodeWrangler
from nodes import node_utils
from surfaces import surface


from lighting import lighting

from scipy.ndimage import distance_transform_edt
from skimage import measure



def set_curves(curve, points):
    curve.points[0].location = points[0]
    curve.points[1].location = points[-1]
    for point in points[1:-1]:
        curve.points.new(*point)


class Cumulus(object):
    VORONOI_SCALE_RANGE = [2.0, 6.0]

    PLACEHOLDER_DENSITY = 32.0

    PLANE_SCALES = [16, 16, 4]

    def __init__(self, name, ref_cloud):
        super().__init__()
        self.ref_cloud = ref_cloud
        self.geo_params, self.shader_params = self.get_node_params()

    def get_scale(self):
        scale_z = np.random.uniform(16.0, 32.0)
        scale_x = np.random.uniform(scale_z * 1.2, scale_z * 2.0)
        scale_y = np.random.uniform(0.5, 2.0) * scale_x
        return [scale_x, scale_y, scale_z]

    def get_curve_func(self):
        curve_pts = self.sample_curves()
        def curve_func(curves):
            curve = curves[2]
            set_curves(curve, curve_pts)
        return curve_func

    def get_params(self):
        cls = type(self)

        # Params
        voronoi_scale = np.random.uniform(*cls.VORONOI_SCALE_RANGE)

        return {
            'density': density,
            'anisotropy': anisotropy,
            'noise_scale': noise_scale,
            'noise_detail': noise_detail,
            'voronoi_scale': voronoi_scale,
            'mix_factor': mix_factor,
            'rotate_angle': rotate_angle,

    def update_geo_params(self, geo_params):
        return geo_params

    def update_shader_params(self, shader_params):
        return shader_params

    def get_node_params(self):
        curve_func = self.get_curve_func()

        params = self.get_params()

        shader_params = self.update_shader_params(dict(params))
        return geo_params, shader_params

    def sample_curves(self):
        second_pt_x = 0.0
        second_pt_y = np.random.uniform(0.80, 0.85)
        cloud = bpy.data.objects.new(self.name, self.ref_cloud.copy())
        link_object(cloud)

        shader_params = self.shader_params
        geo_params['material'] = mat

        if not marching_cubes:
            cloud.dimensions = geo_params['scale']
            return cloud

        name = cloud.name

        # Marching cubes

        min_pts = points.min(axis=0)
        max_pts = points.max(axis=0)


        dists /= dists.max()
        dists[dists < 0.01] = 0

        verts, faces, normals, values = measure.marching_cubes(dists, 0.08)

        max_v = verts.max()
        min_v = verts.min()
        verts = (verts - min_v) / (max_v - min_v) * (max_pts - min_pts) + min_pts

        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], faces)
        mesh.update()

        old_mesh = cloud.data
        bpy.data.objects.remove(cloud)
        bpy.data.meshes.remove(old_mesh)

        cloud = bpy.data.objects.new(name, mesh)
        cloud.active_material = mat
        cloud.dimensions = geo_params['scale']

        link_object(cloud)

        with util.logging.Suppress():
            # Set origin
            butil.select(cloud)
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

            # Fix normals
            bpy.context.view_layer.objects.active = cloud
            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.remove_doubles(threshold=0.0001)
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.editmode_toggle()

            # bpy.ops.object.modifier_apply(modifier="EdgeSplit")

            # bpy.ops.object.modifier_apply(modifier="Displace")
        return cloud


class Cumulonimbus(Cumulus):
    EMISSION_RANGE = [0.01, 0.01]

    PLACEHOLDER_DENSITY = 8.0

    PLANE_SCALES = [16, 16, 32]
    def __init__(self, name, ref_cloud):
        super().__init__(name, ref_cloud)

    def sample_curves(self):
        second_pt_x = 0.0
        second_pt_y = np.random.uniform(0.50, 0.70)

    def get_scale(self):
        scale_y = np.random.uniform(0.5, 2.0) * scale_x
        return scales

    def update_shader_params(self, shader_params):
        return shader_params


class Stratocumulus(Cumulus):
    ANGLE_ROTATE_RANGE = [0.0, np.pi / 4]

    def update_shader_params(self, shader_params):
        return shader_params

    def get_scale(self):
        scale_z = np.random.uniform(16.0, 32.0)
        scale_x = np.random.uniform(128.0, 256.0)
        scale_y = np.random.uniform(0.5, 2.0) * scale_x
        return [scale_x, scale_y, scale_z]

    def get_curve_func(self):
        y_pts, z_pts = self.sample_curves()
        def curve_func(curves):
            set_curves(curves[1], y_pts)
            set_curves(curves[2], z_pts)
        return curve_func

    def sample_y_curves(self):
        n = np.random.randint(2, 6)

        num_pts = n + n - 1

        xs = np.linspace(-1, 1, num_pts + 2)
        ys = [-1]
        for i in range(len(xs[1:-1])):
            if i % 2 == 0:
                y = np.random.uniform(0.8, 1.0)
            else:
                y = np.random.uniform(-1.0, -0.8)
            ys.append(y)
        ys.append(-1)
        ys = np.array(ys)
        return np.stack((xs, ys), axis=1)

    def sample_curves(self):
        z_pts = super().sample_curves()
        y_pts = self.sample_y_curves()
        return [y_pts, z_pts]


class Altocumulus(Cumulus):
    SCATTER_VORONOI_SCALE_RANGE = [1.0, 4.0]
    SCATTER_GRID_VERTICES_RANGE = [4, 12]

    NUM_SUBCLOUDS = 8
    def get_scale(self):
        scale_z = np.random.uniform(16.0, 32.0)
        scale_x = np.random.uniform(scale_z * 1.2, scale_z * 2.0)
        scale_y = np.random.uniform(0.5, 2.0) * scale_x
        return [s * 4 for s in scales]

    def get_params(self):
        cls = type(self)

        # Params
        voronoi_scales = np.random.uniform(*cls.VORONOI_SCALE_RANGE, size=cls.NUM_SUBCLOUDS)

        # Scatter Params

        return {
            'densities': densities,
            'anisotropies': anisotropies,
            'noise_scales': noise_scales,
            'noise_details': noise_details,
            'voronoi_scales': voronoi_scales,
            'mix_factors': mix_factors,
            'rotate_angles': rotate_angles,
            'emission_strengths': emissions,

    def update_shader_params(self, shader_params):

        shader_params = [{
            'density': np.random.uniform(0.05, 0.25),
            'anisotropy': param[0],
            'noise_scale': param[1],
            'noise_detail': param[2],
            'voronoi_scale': param[3],
            'mix_factor': param[4],
            'rotate_angle': param[5],
        return shader_params

    def get_node_params(self):
        cls = type(self)

        scale = self.get_scale()
        curve_funcs = [self.get_curve_func() for _ in range(cls.NUM_SUBCLOUDS)]

        params = self.get_params()

        shader_params = self.update_shader_params(dict(params))
        return geo_params, shader_params

        resolution = min(resolution, 64)
        cloud = bpy.data.objects.new(self.name, self.ref_cloud.copy())
        link_object(cloud)

        shader_params = self.shader_params
        geo_params['materials'] = mats

        # TODO: fix this and check if scales is still needed
        cloud.dimensions = geo_params['scale']
        return cloud


def points_to_voxel(points, voxel_k):
    voxel = np.zeros((voxel_k, voxel_k, voxel_k), dtype=bool)

    points += 1
    points /= 2
    points *= voxel_k

    voxel[points[:, 0], points[:, 1], points[:, 2]] = True
    return voxel


def link_object(obj):


class LinkObject(object):
    def __init__(self, obj):
        super(LinkObject, self).__init__()
        self.obj = obj

    def __enter__(self):

    def __exit__(self, exc_type, exc_val, exc_tb):


def clean():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.object.select_all(action="DESELECT")

    for mesh in bpy.data.meshes.values():
        bpy.data.meshes.remove(mesh)

    for material in bpy.data.materials.values():
        bpy.data.materials.remove(material)

    for node_group in bpy.data.node_groups.values():
        bpy.data.node_groups.remove(node_group)


def remove_collection(collection_name):
    collection = bpy.data.collections.get(collection_name)
    if collection is not None:
        bpy.data.collections.remove(collection)


def create_3d_grid(steps=64):
    xs = np.linspace(-1.0, 1.0, steps)
    ys = np.linspace(-1.0, 1.0, steps)
    zs = np.linspace(-1.0, 1.0, steps)
    xs, ys, zs = np.meshgrid(xs, ys, zs, indexing='ij')
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    zs = zs.reshape(-1)
    return np.stack((xs, ys, zs), axis=1)


def create_cube(name, steps=128, collection='Clouds'):
    grid = create_3d_grid(steps=steps)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(grid, [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.data.collections[collection].objects.link(obj)
    return obj


def initialize(collection):
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.preview_samples = 32
    bpy.context.scene.cycles.samples = 128
    xs = np.linspace(-256, 256, 5)
    ys = np.linspace(-256, 256, 5)
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    size = 128

    resolution = 256

    ref_cloud = bpy.data.meshes.new('ref_cloud')
    ref_cloud.from_pydata(ref_grid, [], [])
    ref_cloud.update()

    clouds += [
    for cloud in clouds:
        bpy.data.collections[collection].objects.link(cloud)
    bpy.context.view_layer.update()

    for i, cloud in enumerate(clouds):
        dimensions = cloud.dimensions

        cloud.location = [xs[i], ys[i], 0]

        if max_dim < size:
            continue

        scale = size / max_dim
        dimensions = dimensions * scale
        cloud.dimensions = dimensions

    lighting.add_lighting()
    return clouds


def create_collection(name):
    clouds_collection = bpy.data.collections.new(name)


def main():
    clean()
    collection_name = 'Clouds'
    remove_collection(collection_name)
    create_collection(collection_name)
    clouds = initialize(collection_name)


def single():
    clean()

    collection_name = 'Clouds'
    remove_collection(collection_name)
    create_collection(collection_name)

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.preview_samples = 32
    bpy.context.scene.cycles.samples = 128

    xs, ys = np.meshgrid(xs, ys)
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    ref_cloud = bpy.data.meshes.new('ref_cloud')
    ref_cloud.from_pydata(ref_grid, [], [])
    ref_cloud.update()

    for cloud in clouds:
        bpy.data.collections[collection_name].objects.link(cloud)
    bpy.context.view_layer.update()

    for i, cloud in enumerate(clouds):
        dimensions = cloud.dimensions

        cloud.location = [xs[i], ys[i], 0]

    lighting.add_lighting()
    return clouds


if __name__ == "__main__":
    main()

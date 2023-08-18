# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hei Law


import bpy

import numpy as np
import mathutils

from tqdm import trange, tqdm
from numpy.random import uniform, normal
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core import surface

from infinigen.assets.lighting import sky_lighting
from infinigen.assets.weather.cloud.node import geometry_func, shader_material
from infinigen.assets.weather.cloud.node import scatter_func

from scipy.ndimage import distance_transform_edt
from skimage import measure

from infinigen.core.util.logging import Suppress
from infinigen.core.util import blender as butil


def set_curves(curve, points):
    curve.points[0].location = points[0]
    curve.points[1].location = points[-1]
    for point in points[1:-1]:
        curve.points.new(*point)


class Cumulus(object):
    DENSITY_RANGE = [1.0, 1.0]
    ANISOTROPY_RANGE = [-0.5, 0.5]
    NOISE_SCALE_RANGE = [8.0, 16.0]
    NOISE_DETAIL_RANGE = [1.0, 16.0]
    VORONOI_SCALE_RANGE = [2.0, 6.0]
    MIX_FACTOR_RANGE = [0.3, 0.8]
    ANGLE_ROTATE_RANGE = [0.0, np.pi / 4]
    EMISSION_RANGE = [0.0, 0.0]

    PLACEHOLDER_DENSITY = 32.0
    MAX_EXPECTED_SCALE = 128.0

    PLANE_SCALES = [16, 16, 4]

    def __init__(self, name, ref_cloud):
        super().__init__()
        self.name = name
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
        density = np.random.uniform(*cls.DENSITY_RANGE)
        anisotropy = np.random.uniform(*cls.ANISOTROPY_RANGE)
        noise_scale = np.random.uniform(*cls.NOISE_SCALE_RANGE)
        noise_detail = np.random.uniform(*cls.NOISE_DETAIL_RANGE)
        voronoi_scale = np.random.uniform(*cls.VORONOI_SCALE_RANGE)
        mix_factor = np.random.uniform(*cls.MIX_FACTOR_RANGE)
        emission = np.random.uniform(*cls.EMISSION_RANGE)
        rotate_angle = np.random.uniform(*cls.ANGLE_ROTATE_RANGE)

        return {
            'density': density,
            'anisotropy': anisotropy,
            'noise_scale': noise_scale,
            'noise_detail': noise_detail,
            'voronoi_scale': voronoi_scale,
            'mix_factor': mix_factor,
            'rotate_angle': rotate_angle,
            'emission_strength': emission, }

    def update_geo_params(self, geo_params):
        return geo_params

    def update_shader_params(self, shader_params):
        shader_params.update({'density': np.random.uniform(0.05, 0.25), })
        return shader_params

    def get_node_params(self):
        scale = self.get_scale()
        curve_func = self.get_curve_func()

        params = self.get_params()
        params.update({'scale': scale, 'curve_func': curve_func, })

        geo_params = self.update_geo_params(dict(params))
        shader_params = self.update_shader_params(dict(params))
        return geo_params, shader_params

    def sample_curves(self):
        first_pt_x = np.random.uniform(-0.95, -0.80)
        first_pt_y = -1.0
        second_pt_x = 0.0
        second_pt_y = np.random.uniform(0.80, 0.85)
        third_pt_x = np.random.uniform(0.25, 0.75)
        third_pt_y = np.random.uniform(0.75, 0.90)
        forth_pt_x = 1.0
        forth_pt_y = np.random.uniform(0.90, 1.00)

        return [[first_pt_x, first_pt_y], [second_pt_x, second_pt_y], [third_pt_x, third_pt_y],
            [forth_pt_x, forth_pt_y], ]

    def make_cloud(self, marching_cubes=False, resolution=128, selection=None, ):
        cloud = bpy.data.objects.new(self.name, self.ref_cloud.copy())
        link_object(cloud)

        geo_params = self.geo_params
        shader_params = self.shader_params
        points_only = marching_cubes

        mat = surface.add_material(cloud, shader_material, selection=selection, input_kwargs=shader_params, )

        geo_params['material'] = mat
        surface.add_geomod(cloud, geometry_func(points_only=points_only, resolution=resolution, ),
            selection=selection, input_kwargs=geo_params, apply=True, )

        if not marching_cubes:
            cloud.dimensions = geo_params['scale']
            return cloud

        name = cloud.name

        # Marching cubes
        points = np.array([v.co for v in cloud.data.vertices])

        min_pts = points.min(axis=0)
        max_pts = points.max(axis=0)

        voxel = points_to_voxel(points, resolution)
        dists = distance_transform_edt(voxel)

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

        with Suppress():
            # Set origin
            butil.select(cloud)
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

            # Fix normals
            bpy.context.view_layer.objects.active = cloud
            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.remove_doubles(threshold=0.0001)
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.editmode_toggle()

            # modifier = cloud.modifiers.new("EdgeSplit", "EDGE_SPLIT")  # modifier.split_angle = 1.22173  #
            # bpy.ops.object.modifier_apply(modifier="EdgeSplit")

            # modifier = cloud.modifiers.new("Displace", "DISPLACE")  # modifier.strength = 0.00001  #
            # bpy.ops.object.modifier_apply(modifier="Displace")
        return cloud


class Cumulonimbus(Cumulus):
    DENSITY_RANGE = [1.0, 1.0]
    EMISSION_RANGE = [0.01, 0.01]

    PLACEHOLDER_DENSITY = 8.0
    MAX_EXPECTED_SCALE = 2048.0

    PLANE_SCALES = [16, 16, 32]

    def __init__(self, name, ref_cloud):
        super().__init__(name, ref_cloud)

    def sample_curves(self):
        first_pt_x = np.random.uniform(-0.65, -0.50)
        first_pt_y = -1.0
        second_pt_x = 0.0
        second_pt_y = np.random.uniform(0.50, 0.70)
        third_pt_x = np.random.uniform(0.25, 0.75)
        third_pt_y = np.random.uniform(0.80, 0.95)
        forth_pt_x = 1.0
        forth_pt_y = np.random.uniform(-1.0, 0.50)

        return [[first_pt_x, first_pt_y], [second_pt_x, second_pt_y], [third_pt_x, third_pt_y],
            [forth_pt_x, forth_pt_y], ]

    def make_cloud(self, marching_cubes=False, resolution=128, selection=None, ):
        return super().make_cloud(marching_cubes=marching_cubes, resolution=resolution * 2,
            selection=selection, )

    def get_scale(self):
        scale_x = np.random.uniform(512.0, 1024.0)
        scale_y = np.random.uniform(0.5, 2.0) * scale_x
        scale_z = np.random.uniform(256.0, 512.0)
        scales = [scale_x, scale_y, scale_z]
        return scales

    def update_shader_params(self, shader_params):
        shader_params.update({'density': np.random.uniform(0.1, 0.3), })
        return shader_params


class Stratocumulus(Cumulus):
    ANGLE_ROTATE_RANGE = [0.0, np.pi / 4]

    def update_shader_params(self, shader_params):
        shader_params.update({'density': np.random.uniform(0.01, 0.10), })
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
        scales = [scale_x, scale_y, scale_z]
        return [s * 4 for s in scales]

    def get_params(self):
        cls = type(self)

        # Params
        densities = np.random.uniform(*cls.DENSITY_RANGE, size=cls.NUM_SUBCLOUDS)
        anisotropies = np.random.uniform(*cls.ANISOTROPY_RANGE, size=cls.NUM_SUBCLOUDS)
        noise_scales = np.random.uniform(*cls.NOISE_SCALE_RANGE, size=cls.NUM_SUBCLOUDS)
        noise_details = np.random.uniform(*cls.NOISE_DETAIL_RANGE, size=cls.NUM_SUBCLOUDS)
        voronoi_scales = np.random.uniform(*cls.VORONOI_SCALE_RANGE, size=cls.NUM_SUBCLOUDS)
        mix_factors = np.random.uniform(*cls.MIX_FACTOR_RANGE, size=cls.NUM_SUBCLOUDS)
        emissions = np.random.uniform(*cls.EMISSION_RANGE, size=cls.NUM_SUBCLOUDS)
        rotate_angles = np.random.uniform(*cls.ANGLE_ROTATE_RANGE, size=cls.NUM_SUBCLOUDS)

        # Scatter Params
        voronoi_scale = np.random.uniform(*cls.SCATTER_VORONOI_SCALE_RANGE)
        vertices_x = np.random.randint(*cls.SCATTER_GRID_VERTICES_RANGE)
        vertices_y = np.random.randint(*cls.SCATTER_GRID_VERTICES_RANGE)
        scatter_params = {'voronoi_scale': voronoi_scale, 'vertices_x': vertices_x, 'vertices_y': vertices_y, }

        return {
            'densities': densities,
            'anisotropies': anisotropies,
            'noise_scales': noise_scales,
            'noise_details': noise_details,
            'voronoi_scales': voronoi_scales,
            'mix_factors': mix_factors,
            'rotate_angles': rotate_angles,
            'emission_strengths': emissions,
            'scatter_params': scatter_params, }

    def update_shader_params(self, shader_params):
        params = zip(shader_params['anisotropies'], shader_params['noise_scales'],
            shader_params['noise_details'], shader_params['voronoi_scales'], shader_params['mix_factors'],
            shader_params['rotate_angles'], shader_params['emission_strengths'], )

        shader_params = [{
            'density': np.random.uniform(0.05, 0.25),
            'anisotropy': param[0],
            'noise_scale': param[1],
            'noise_detail': param[2],
            'voronoi_scale': param[3],
            'mix_factor': param[4],
            'rotate_angle': param[5],
            'emission_strength': param[6], } for param in params]
        return shader_params

    def get_node_params(self):
        cls = type(self)

        scale = self.get_scale()
        curve_funcs = [self.get_curve_func() for _ in range(cls.NUM_SUBCLOUDS)]

        params = self.get_params()
        params.update({'scale': scale, 'curve_funcs': curve_funcs, })

        geo_params = self.update_geo_params(dict(params))
        shader_params = self.update_shader_params(dict(params))
        return geo_params, shader_params

    def make_cloud(self, marching_cubes=False, resolution=128, selection=None, ):
        resolution = min(resolution, 64)
        cloud = bpy.data.objects.new(self.name, self.ref_cloud.copy())
        link_object(cloud)

        geo_params = self.geo_params
        shader_params = self.shader_params
        points_only = apply = marching_cubes

        mats = [surface.shaderfunc_to_material(shader_material, **shader_param) for shader_param in
            shader_params]

        geo_params['materials'] = mats
        surface.add_geomod(cloud, scatter_func(points_only=points_only, resolution=resolution, ),
            selection=selection, input_kwargs=geo_params, apply=True, )

        # TODO: fix this and check if scales is still needed
        cloud.dimensions = geo_params['scale']
        return cloud


def points_to_voxel(points, voxel_k):
    voxel = np.zeros((voxel_k, voxel_k, voxel_k), dtype=bool)

    points += 1
    points /= 2
    points *= voxel_k
    points = points.astype(int)  # n x 3, (x, y, z)
    points = np.clip(points, 1, voxel_k - 1)

    voxel[points[:, 0], points[:, 1], points[:, 2]] = True
    return voxel


def link_object(obj):
    bpy.context.scene.collection.objects.link(obj)


class LinkObject(object):
    def __init__(self, obj):
        super(LinkObject, self).__init__()
        self.obj = obj

    def __enter__(self):
        bpy.context.scene.collection.objects.link(self.obj)

    def __exit__(self, exc_type, exc_val, exc_tb):
        bpy.context.scene.collection.objects.unlink(self.obj)


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

    ref_grid = create_3d_grid(steps=128)
    ref_cloud = bpy.data.meshes.new('ref_cloud')
    ref_cloud.from_pydata(ref_grid, [], [])
    ref_cloud.update()

    clouds = []
    clouds += [Cumulus(f'Cumulus_{i:03d}', ref_cloud).make_cloud(marching_cubes=False, resolution=resolution, )
        for i in range(6)]
    clouds += [Cumulonimbus(f'Cumulonimbus{i:03d}', ref_cloud).make_cloud(marching_cubes=False,
        resolution=resolution, ) for i in range(6)]
    clouds += [Stratocumulus(f'Stratocumulus{i:03d}', ref_cloud).make_cloud(marching_cubes=False,
        resolution=resolution, ) for i in range(6)]
    clouds += [
        Altocumulus(f'Altocumulus{i:03d}', ref_cloud).make_cloud(marching_cubes=False, resolution=resolution, )
        for i in range(7)]
    for cloud in clouds:
        bpy.data.collections[collection].objects.link(cloud)
    bpy.context.view_layer.update()

    for i, cloud in enumerate(clouds):
        dimensions = cloud.dimensions
        max_dim = max(dimensions[:2])

        cloud.location = [xs[i], ys[i], 0]

        if max_dim < size:
            continue

        scale = size / max_dim
        dimensions = dimensions * scale
        cloud.dimensions = dimensions

    sky_lighting.add_lighting()
    return clouds


def create_collection(name):
    clouds_collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(clouds_collection)


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

    xs = np.zeros((1,), dtype=np.float32)
    ys = np.zeros((1,), dtype=np.float32)
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    ref_grid = create_3d_grid(steps=256)
    ref_cloud = bpy.data.meshes.new('ref_cloud')
    ref_cloud.from_pydata(ref_grid, [], [])
    ref_cloud.update()

    clouds = [Cumulonimbus(f'Cumulonimbus_{i:03d}', ref_cloud).make_cloud(marching_cubes=True, resolution=128, )
        for i in range(1)]
    for cloud in clouds:
        bpy.data.collections[collection_name].objects.link(cloud)
    bpy.context.view_layer.update()

    for i, cloud in enumerate(clouds):
        dimensions = cloud.dimensions
        max_dim = max(dimensions[:2])

        cloud.location = [xs[i], ys[i], 0]

    sky_lighting.add_lighting()
    return clouds


if __name__ == "__main__":
    main()

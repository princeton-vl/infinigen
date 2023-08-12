# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hei Law


import gin
import bpy
import numpy as np

from infinigen.assets.utils.object import new_cube
from infinigen.core.placement.factory import AssetFactory

from infinigen.assets.weather.cloud.cloud import Cumulus, Cumulonimbus, Stratocumulus, Altocumulus
from infinigen.assets.weather.cloud.cloud import create_3d_grid

from infinigen.core import surface
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import random_general as rg

from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.assets.utils.tag import tag_object, tag_nodegroup


@gin.configurable
class CloudFactory(AssetFactory):
    def __init__(
        self,
        factory_seed,
        coarse=False,
        terrain_mesh=None,
        max_distance=300,
        steps=128,
        cloudy=("bool", 0.01),
    ):
        super(CloudFactory, self).__init__(factory_seed, coarse=coarse)

        self.max_distance = max_distance

        self.ref_cloud = bpy.data.meshes.new('ref_cloud')
        self.ref_cloud.from_pydata(create_3d_grid(steps=steps), [], [])
        self.ref_cloud.update()

        with FixedSeed(factory_seed):
            self.cloudy = rg(cloudy)

        self.cloud_types = [Cumulonimbus, ] if self.cloudy else [Cumulus, Stratocumulus, Altocumulus, ]

        self.resolutions = {
            Cumulonimbus: [16, 128],
            Cumulus: [16, 128],
            Stratocumulus: [32, 256],
            Altocumulus: [16, 64], }
        scale_resolution = 4
        self.resolutions = {k: (scale_resolution * u, scale_resolution * v) for k, (u, v) in
            self.resolutions.items()}

        self.min_distance = 256 if self.cloudy else 64
        self.dome_radius = 1024 if self.cloudy else 256
        self.dome_threshold = 32 if self.cloudy else 0
        self.density_range = [1e-5, 1e-4] if self.cloudy else [1e-4, 2e-4]

        self.max_scale = max([t.MAX_EXPECTED_SCALE for t in self.cloud_types])
        self.density = max([t.PLACEHOLDER_DENSITY for t in self.cloud_types])

    def spawn_locations(self):
        obj = new_cube()
        surface.add_geomod(obj, self.geo_dome, apply=True,
                           input_args=[self.dome_radius, self.dome_threshold, self.density_range,
                               self.min_distance])

        locations = np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
        butil.delete(obj)
        return locations

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return butil.spawn_empty('placeholder', disp_type='CUBE', s=self.max_scale)

    def create_asset(self, distance, **kwargs):
        cloud_type = np.random.choice(self.cloud_types)
        resolution_min, resolution_max = self.resolutions[cloud_type]
        resolution = max(1 - distance / self.max_distance, 0)
        resolution = resolution * (resolution_max - resolution_min) + resolution_min
        resolution = int(resolution)
        new_cloud = cloud_type("Cloud", self.ref_cloud)
        new_cloud = new_cloud.make_cloud(marching_cubes=False, resolution=resolution, )
        tag_object(new_cloud, 'cloud')
        return new_cloud

    @staticmethod
    def geo_dome(nw, dome_radius, dome_threshold, density_range, min_distance, ):
        ico_sphere = nw.new_node('GeometryNodeMeshIcoSphere',
                                 input_kwargs={'Radius': dome_radius, 'Subdivisions': 8, }, )

        transform = nw.new_node(Nodes.Transform,
                                input_kwargs={'Geometry': ico_sphere, 'Scale': (1.2, 1.4, 1.0), }, )

        position = nw.new_node(Nodes.InputPosition)
        separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': position, }, )

        less_than = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Z"], 1: dome_threshold, },
                                attrs={'operation': 'LESS_THAN', }, )

        delete_geometry = nw.new_node('GeometryNodeDeleteGeometry',
                                      input_kwargs={'Geometry': transform, 'Selection': less_than, }, )

        distribute_points_on_faces = nw.new_node(Nodes.DistributePointsOnFaces, input_kwargs={
            'Mesh': delete_geometry,
            'Distance Min': min_distance,
            'Density Max': np.random.uniform(*density_range),
            'Seed': np.random.randint(1e5), }, attrs={'distribute_method': 'POISSON', }, )

        combine_xyz = nw.new_node(Nodes.CombineXYZ,
                                  input_kwargs={'Z': nw.uniform(32, np.random.randint(64, 1e5)), }, )

        set_position = nw.new_node(Nodes.SetPosition, input_kwargs={
            'Geometry': distribute_points_on_faces.outputs["Points"],
            'Offset': combine_xyz, }, )

        verts = nw.new_node(Nodes.PointsToVertices, input_kwargs={'Points': set_position, }, )

        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': verts, }, )


class CumulonimbusFactory(CloudFactory):
    def __init__(self, factory_seed, coarse=False, max_distance=300, steps=128, ):
        self.cloud_types = [Cumulonimbus]
        super(CumulonimbusFactory, self).__init__(factory_seed, coarse, max_distance, steps)
        self.cloud_types = [Cumulonimbus]


class CumulusFactory(CloudFactory):
    def __init__(self, factory_seed, coarse=False, max_distance=300, steps=128, ):
        self.cloud_types = [Cumulus]
        super(CumulusFactory, self).__init__(factory_seed, coarse, max_distance, steps)
        self.cloud_types = [Cumulus]


class StratocumulusFactory(CloudFactory):
    def __init__(self, factory_seed, coarse=False, max_distance=300, steps=128, ):
        self.cloud_types = [Stratocumulus]
        super(StratocumulusFactory, self).__init__(factory_seed, coarse, max_distance, steps)
        self.cloud_types = [Stratocumulus]


class AltocumulusFactory(CloudFactory):
    def __init__(self, factory_seed, coarse=False, max_distance=300, steps=128, ):
        self.cloud_types = [Altocumulus]
        super(AltocumulusFactory, self).__init__(factory_seed, coarse, max_distance, steps)
        self.cloud_types = [Altocumulus]

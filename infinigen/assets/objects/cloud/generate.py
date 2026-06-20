# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hei Law

from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
import gin
import numpy as np
from numpy.random import uniform
from pydantic import Field

from infinigen.assets.utils.object import new_cube
from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.random import random_general as rg

from .cloud import (
    Altocumulus,
    Cumulonimbus,
    Cumulus,
    Stratocumulus,
    create_3d_grid,
)


class CloudParameters(AssetParameters):
    cloudy: bool = Field(default=False, json_schema_extra={"editable": False})


class CumulusParameters(CloudParameters):
    anisotropy: Annotated[
        float, Field(ge=-0.5, le=0.5, json_schema_extra={"editable": True})
    ] = 0.0
    first_pt_x: Annotated[
        float, Field(ge=-0.95, le=-0.8, json_schema_extra={"editable": True})
    ] = -0.875
    forth_pt_y: Annotated[
        float, Field(ge=0.9, le=1.0, json_schema_extra={"editable": True})
    ] = 0.95
    mix_factor: Annotated[
        float, Field(ge=0.3, le=0.8, json_schema_extra={"editable": True})
    ] = 0.55
    noise_detail: Annotated[
        float, Field(ge=1.0, le=16.0, json_schema_extra={"editable": True})
    ] = 8.5
    noise_scale: Annotated[
        float, Field(ge=8.0, le=16.0, json_schema_extra={"editable": True})
    ] = 12.0
    rotate_angle: Annotated[
        float, Field(ge=0.0, le=0.785398, json_schema_extra={"editable": True})
    ] = 0.0
    scale_x: Annotated[
        float, Field(ge=28.818331, le=63.379149, json_schema_extra={"editable": True})
    ] = 46.0
    scale_y: Annotated[
        float, Field(ge=0.5, le=2.0, json_schema_extra={"editable": True})
    ] = 1.0
    scale_z: Annotated[
        float, Field(ge=16.0, le=32.0, json_schema_extra={"editable": True})
    ] = 24.0
    second_pt_y: Annotated[
        float, Field(ge=0.8, le=0.85, json_schema_extra={"editable": True})
    ] = 0.825
    third_pt_x: Annotated[
        float, Field(ge=0.25, le=0.75, json_schema_extra={"editable": True})
    ] = 0.5
    third_pt_y: Annotated[
        float, Field(ge=0.75, le=0.9, json_schema_extra={"editable": True})
    ] = 0.825
    voronoi_scale: Annotated[
        float, Field(ge=2.0, le=6.0, json_schema_extra={"editable": True})
    ] = 4.0


@gin.configurable
class CloudFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CloudParameters

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
        self._cloudy_gin = cloudy
        self.ref_cloud = bpy.data.meshes.new("ref_cloud")
        self.ref_cloud.from_pydata(create_3d_grid(steps=steps), [], [])
        self.ref_cloud.update()
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> CloudParameters:
        return CloudParameters(seed=seed, cloudy=rg(self._cloudy_gin))

    def apply_parameters(
        self, params: CloudParameters, *, spawn_scope: bool = True
    ) -> None:
        self.cloudy = params.cloudy
        self.cloud_types = (
            [Cumulonimbus]
            if self.cloudy
            else [Cumulus, Stratocumulus, Altocumulus]
        )
        self.resolutions = {
            Cumulonimbus: [16, 128],
            Cumulus: [16, 128],
            Stratocumulus: [32, 256],
            Altocumulus: [16, 64],
        }
        scale_resolution = 4
        self.resolutions = {
            k: (scale_resolution * u, scale_resolution * v)
            for k, (u, v) in self.resolutions.items()
        }
        self.min_distance = 256 if self.cloudy else 64
        self.dome_radius = 1024 if self.cloudy else 256
        self.dome_threshold = 32 if self.cloudy else 0
        self.density_range = [1e-5, 1e-4] if self.cloudy else [1e-4, 2e-4]
        self.max_scale = max(t.MAX_EXPECTED_SCALE for t in self.cloud_types)
        self.density = max(t.PLACEHOLDER_DENSITY for t in self.cloud_types)
        self._use_fixed_spawn_draws = spawn_scope

    def spawn_locations(self):
        obj = new_cube()
        surface.add_geomod(
            obj,
            self.geo_dome,
            apply=True,
            input_args=[
                self.dome_radius,
                self.dome_threshold,
                self.density_range,
                self.min_distance,
            ],
        )

        locations = np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
        butil.delete(obj)
        return locations

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return butil.spawn_empty("placeholder", disp_type="CUBE", s=self.max_scale)

    def create_asset(self, distance, **kwargs):
        cloud_type = np.random.choice(self.cloud_types)

        resolution_min, resolution_max = self.resolutions[cloud_type]
        resolution = max(1 - distance / self.max_distance, 0)
        resolution = resolution * (resolution_max - resolution_min) + resolution_min
        resolution = int(resolution)

        new_cloud = cloud_type("Cloud", self.ref_cloud)
        new_cloud = new_cloud.make_cloud(
            marching_cubes=False,
            resolution=resolution,
        )
        butil.apply_transform(new_cloud)

        tag_object(new_cloud, "cloud")
        return new_cloud

    @staticmethod
    def geo_dome(
        nw,
        dome_radius,
        dome_threshold,
        density_range,
        min_distance,
    ):
        ico_sphere = nw.new_node(
            "GeometryNodeMeshIcoSphere",
            input_kwargs={
                "Radius": dome_radius,
                "Subdivisions": 8,
            },
        )

        transform = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": ico_sphere,
                "Scale": (1.2, 1.4, 1.0),
            },
        )

        position = nw.new_node(Nodes.InputPosition)
        separate_xyz = nw.new_node(
            Nodes.SeparateXYZ,
            input_kwargs={
                "Vector": position,
            },
        )

        less_than = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: separate_xyz.outputs["Z"],
                1: dome_threshold,
            },
            attrs={
                "operation": "LESS_THAN",
            },
        )

        delete_geometry = nw.new_node(
            "GeometryNodeDeleteGeometry",
            input_kwargs={
                "Geometry": transform,
                "Selection": less_than,
            },
        )

        distribute_points_on_faces = nw.new_node(
            Nodes.DistributePointsOnFaces,
            input_kwargs={
                "Mesh": delete_geometry,
                "Distance Min": min_distance,
                "Density Max": np.random.uniform(*density_range),
                "Seed": np.random.randint(1e5),
            },
            attrs={
                "distribute_method": "POISSON",
            },
        )

        combine_xyz = nw.new_node(
            Nodes.CombineXYZ,
            input_kwargs={
                "Z": nw.uniform(32, np.random.randint(64, 1e5)),
            },
        )

        set_position = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": distribute_points_on_faces.outputs["Points"],
                "Offset": combine_xyz,
            },
        )

        verts = nw.new_node(
            Nodes.PointsToVertices,
            input_kwargs={
                "Points": set_position,
            },
        )

        nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={
                "Geometry": verts,
            },
        )


class CumulonimbusFactory(CloudFactory):
    def __init__(
        self,
        factory_seed,
        coarse=False,
        max_distance=300,
        steps=128,
    ):
        self.cloud_types = [Cumulonimbus]
        super(CumulonimbusFactory, self).__init__(
            factory_seed, coarse, max_distance, steps
        )
        self.cloud_types = [Cumulonimbus]


class CumulusFactory(CloudFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CumulusParameters

    def __init__(
        self,
        factory_seed,
        coarse=False,
        max_distance=300,
        steps=128,
    ):
        self.cloud_types = [Cumulus]
        super(CumulusFactory, self).__init__(factory_seed, coarse, max_distance, steps)
        self.cloud_types = [Cumulus]

    def _sample_init_parameters(self, seed: int) -> CumulusParameters:
        params = super()._sample_init_parameters(seed)
        return CumulusParameters(**params.model_dump())

    def _sample_spawn_parameters(
        self, params: CumulusParameters, seed: int, i: int
    ) -> CumulusParameters:
        scale_z = uniform(16.0, 32.0)
        return params.model_copy(
            update={
                "anisotropy": uniform(-0.5, 0.5),
                "first_pt_x": uniform(-0.95, -0.8),
                "forth_pt_y": uniform(0.9, 1.0),
                "mix_factor": uniform(0.3, 0.8),
                "noise_detail": uniform(1.0, 16.0),
                "noise_scale": uniform(8.0, 16.0),
                "rotate_angle": uniform(0.0, np.pi / 4),
                "scale_z": scale_z,
                "scale_x": uniform(28.818331, 63.379149),
                "scale_y": uniform(0.5, 2.0),
                "second_pt_y": uniform(0.8, 0.85),
                "third_pt_x": uniform(0.25, 0.75),
                "third_pt_y": uniform(0.75, 0.9),
                "voronoi_scale": uniform(2.0, 6.0),
            }
        )

    def apply_parameters(
        self, params: CumulusParameters, *, spawn_scope: bool = True
    ) -> None:
        super().apply_parameters(params, spawn_scope=spawn_scope)
        if spawn_scope:
            self._cumulus_params = params

    def create_asset(self, distance, **kwargs):
        cloud_type = Cumulus
        resolution_min, resolution_max = self.resolutions[cloud_type]
        resolution = max(1 - distance / self.max_distance, 0)
        resolution = resolution * (resolution_max - resolution_min) + resolution_min
        resolution = int(resolution)

        if self._use_fixed_spawn_draws:
            p = self._cumulus_params
            curve_pts = [
                [p.first_pt_x, -1.0],
                [0.0, p.second_pt_y],
                [p.third_pt_x, p.third_pt_y],
                [1.0, p.forth_pt_y],
            ]
            geo_params = {
                "density": 1.0,
                "anisotropy": p.anisotropy,
                "noise_scale": p.noise_scale,
                "noise_detail": p.noise_detail,
                "voronoi_scale": p.voronoi_scale,
                "mix_factor": p.mix_factor,
                "rotate_angle": p.rotate_angle,
                "emission_strength": 0.0,
                "scale": [p.scale_x, p.scale_y, p.scale_z],
            }
        else:
            curve_pts = None
            geo_params = None

        new_cloud = _ParameterizedCumulus(
            "Cloud",
            self.ref_cloud,
            curve_pts=curve_pts,
            geo_params=geo_params,
        )
        new_cloud = new_cloud.make_cloud(
            marching_cubes=False,
            resolution=resolution,
        )
        butil.apply_transform(new_cloud)
        tag_object(new_cloud, "cloud")
        return new_cloud


class _ParameterizedCumulus(Cumulus):
    def __init__(
        self,
        name,
        ref_cloud,
        curve_pts=None,
        geo_params=None,
    ):
        self._curve_pts = curve_pts
        self._geo_params_override = geo_params
        super().__init__(name, ref_cloud)

    def get_scale(self):
        if self._geo_params_override is not None:
            return self._geo_params_override["scale"]
        return super().get_scale()

    def get_params(self):
        if self._geo_params_override is not None:
            params = dict(self._geo_params_override)
            params.pop("scale")
            return params
        return super().get_params()

    def sample_curves(self):
        if self._curve_pts is not None:
            return self._curve_pts
        return super().sample_curves()


class StratocumulusFactory(CloudFactory):
    def __init__(
        self,
        factory_seed,
        coarse=False,
        max_distance=300,
        steps=128,
    ):
        self.cloud_types = [Stratocumulus]
        super(StratocumulusFactory, self).__init__(
            factory_seed, coarse, max_distance, steps
        )
        self.cloud_types = [Stratocumulus]


class AltocumulusFactory(CloudFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CloudParameters

    def __init__(
        self,
        factory_seed,
        coarse=False,
        max_distance=300,
        steps=128,
    ):
        self.cloud_types = [Altocumulus]
        super(AltocumulusFactory, self).__init__(
            factory_seed, coarse, max_distance=max_distance, steps=steps
        )
        self.cloud_types = [Altocumulus]

    def apply_parameters(
        self, params: CloudParameters, *, spawn_scope: bool = True
    ) -> None:
        super().apply_parameters(params, spawn_scope=spawn_scope)
        self.cloud_types = [Altocumulus]

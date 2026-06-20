# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo

from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
import numpy as np
from numpy.random import uniform as U
from pydantic import Field

import infinigen.core.util.blender as butil
from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.table_decorations.utils import nodegroup_lofting_poly
from infinigen.assets.objects.tables.table_utils import nodegroup_n_gon_profile
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util.random import weighted_sample


class RangeHoodParameters(AssetParameters):
    Height_1: Annotated[float, Field(ge=0.05, le=0.07, json_schema_extra={"editable": True})]
    Height_2: Annotated[float, Field(ge=0.1, le=0.3, json_schema_extra={"editable": True})]
    Scale_2: Annotated[float, Field(ge=0.25, le=0.4, json_schema_extra={"editable": True})]
    Height_total: float = Field(json_schema_extra={"editable": False})
    Width: float = Field(json_schema_extra={"editable": False})
    Depth: float = Field(json_schema_extra={"editable": False})
    surface_material_gen: Any = Field(json_schema_extra={"editable": False})
    scratch: Any | None = Field(default=None, json_schema_extra={"editable": False})
    edge_wear: Any | None = Field(default=None, json_schema_extra={"editable": False})


class RangeHoodFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = RangeHoodParameters

    def __init__(self, factory_seed, coarse=False, dimensions=None):
        super().__init__(factory_seed, coarse=coarse)
        self.dimensions = dimensions
        self.init_legacy_parameters()

    @staticmethod
    def _default_dimensions() -> tuple[float, float, float]:
        return 0.55, 0.75, 1.0

    @staticmethod
    def sample_geometry_parameters(
        dimensions: tuple[float, float, float] | None = None,
    ) -> dict[str, float]:
        if dimensions is None:
            dimensions = RangeHoodFactory._default_dimensions()
        x, y, z = dimensions
        return {
            "Height_total": z,
            "Width": y,
            "Depth": x,
            "Height_1": U(0.05, 0.07),
            "Scale_2": U(0.25, 0.4),
            "Height_2": U(0.1, 0.3),
        }

    @staticmethod
    def sample_parameters(dimensions):
        return RangeHoodFactory.sample_geometry_parameters(dimensions)

    def _sample_materials(self) -> tuple[Any, Any | None, Any | None]:
        surface_gen_class = weighted_sample(material_assignments.metals)
        surface_material_gen = surface_gen_class()
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        scratch = None if U() > scratch_prob else scratch_fn()
        edge_wear = None if U() > edge_wear_prob else edge_wear_fn()
        return surface_material_gen, scratch, edge_wear

    def _sample_init_parameters(self, seed: int) -> RangeHoodParameters:
        geometry = self.sample_geometry_parameters(self.dimensions)
        surface_material_gen, scratch, edge_wear = self._sample_materials()
        return RangeHoodParameters(
            seed=seed,
            **geometry,
            surface_material_gen=surface_material_gen,
            scratch=scratch,
            edge_wear=edge_wear,
        )

    def apply_parameters(
        self, params: RangeHoodParameters, *, spawn_scope: bool = True
    ) -> None:
        self.params = params.model_dump(
            exclude={"seed", "surface_material_gen", "scratch", "edge_wear"},
            by_alias=False,
        )
        self.surface_material_gen = params.surface_material_gen
        self.surface = params.surface_material_gen
        self.scratch = params.scratch
        self.edge_wear = params.edge_wear
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object

        surface.add_geomod(
            obj, geometry_generate_hood, apply=True, input_kwargs=self.params
        )
        butil.modify_mesh(obj, "SOLIDIFY", apply=True, thickness=0.002)
        butil.modify_mesh(obj, "SUBSURF", apply=True, levels=1, render_levels=1)

        return obj

    def finalize_assets(self, assets):
        surface.assign_material(assets, self.surface())
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)


def geometry_generate_hood(nw: NodeWrangler, **kwargs):
    generatetabletop = nw.new_node(
        geometry_range_hood().name,
        input_kwargs={
            "Resolution": 64,
            "Height_total": kwargs["Height_total"],
            "Width": kwargs["Width"],
            "Depth": kwargs["Depth"],
            "Height_1": kwargs["Height_1"],
            "Scale_2": kwargs["Scale_2"],
            "Height_2": kwargs["Height_2"],
        },
    )

    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": generatetabletop},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "geometry_range_hood", singleton=False, type="GeometryNodeTree"
)
def geometry_range_hood(nw: NodeWrangler):
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketInt", "Resolution", 128),
            ("NodeSocketFloat", "Height_total", 0.0000),
            ("NodeSocketFloat", "Width", 0.0000),
            ("NodeSocketFloat", "Depth", 0.0000),
            ("NodeSocketFloat", "Profile Fillet Ratio", 0.0100),
            ("NodeSocketFloat", "Height_1", 0.0000),
            ("NodeSocketFloat", "Scale_2", 0.0000),
            ("NodeSocketFloat", "Height_2", 0.3000),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: 1.4140},
        attrs={"operation": "MULTIPLY"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: group_input.outputs["Width"]},
        attrs={"operation": "DIVIDE"},
    )

    ngonprofile = nw.new_node(
        nodegroup_n_gon_profile().name,
        input_kwargs={
            "Profile Width": multiply,
            "Profile Aspect Ratio": divide,
            "Profile Fillet Ratio": group_input.outputs["Profile Fillet Ratio"],
        },
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={"Curve": ngonprofile, "Count": group_input.outputs["Resolution"]},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_1})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": resample_curve, "Translation": combine_xyz},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Height_1"]}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry, "Translation": combine_xyz_1},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Height_2"]}
    )

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry,
            "Translation": combine_xyz_2,
            "Scale": group_input.outputs["Scale_2"],
        },
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Height_total"],
            1: group_input.outputs["Height_2"],
        },
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": subtract})

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform_geometry_2, "Translation": combine_xyz_3},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                transform_geometry_3,
                transform_geometry_2,
                transform_geometry_1,
                transform_geometry,
            ]
        },
    )

    lofting_poly = nw.new_node(
        nodegroup_lofting_poly().name,
        input_kwargs={
            "Profile Curves": join_geometry,
            "U Resolution": group_input.outputs["Resolution"],
            "V Resolution": group_input.outputs["Resolution"],
        },
    )

    delete_geometry = nw.new_node(
        Nodes.DeleteGeometry,
        input_kwargs={
            "Geometry": lofting_poly.outputs["Geometry"],
            "Selection": lofting_poly.outputs["Top"],
        },
    )

    grid = nw.new_node(
        Nodes.MeshGrid,
        input_kwargs={
            "Size X": group_input.outputs["Width"],
            "Size Y": group_input.outputs["Depth"],
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
        },
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_2})

    transform_geometry_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": grid.outputs["Mesh"],
            "Translation": combine_xyz_4,
            "Rotation": (-0.0698, 0.0000, 0.0000),
            "Scale": (0.9800, 0.9800, 1.0000),
        },
    )

    transform_geometry_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": transform_geometry_4,
            "Rotation": (0.1047, 0.0000, 0.0000),
            "Scale": (0.9500, 0.9700, 1.0000),
        },
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [delete_geometry, transform_geometry_5]},
    )

    transform_geometry_6 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Rotation": (0.0, 0.0000, -np.pi / 2),
        },
    )

    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_6},
        attrs={"is_active_output": True},
    )

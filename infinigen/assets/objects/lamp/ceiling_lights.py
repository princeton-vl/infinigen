# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Hongyu Wen
# - Alexander Raistrick: add point light

from __future__ import annotations

from typing import Annotated, Any, ClassVar

from numpy.random import randint as RI
from numpy.random import uniform as U
from pydantic import Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.lighting.indoor_lights import PointLampFactory
from infinigen.assets.materials.lamp_shaders import (
    shader_black,
    shader_lamp_bulb_nonemissive,
)
from infinigen.assets.utils.autobevel import BevelSharp
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import clip_gaussian


class CeilingLightParameters(AssetParameters):
    Radius: Annotated[float, Field(ge=0.1, le=0.25, json_schema_extra={"editable": True})]
    Thickness: Annotated[
        float, Field(ge=0.005, le=0.05, json_schema_extra={"editable": True})
    ]
    InnerRadius: Annotated[
        float, Field(ge=0.4, le=0.9, json_schema_extra={"editable": True})
    ]
    Height: Annotated[float, Field(ge=0.049, le=0.105, json_schema_extra={"editable": True})]
    InnerHeight: Annotated[
        float, Field(ge=0.5, le=1.1, json_schema_extra={"editable": True})
    ]
    Curvature: Annotated[float, Field(ge=0.1, le=0.5, json_schema_extra={"editable": True})]
    beveler_mult: Annotated[
        float, Field(ge=1.0, le=3.0, json_schema_extra={"editable": True})
    ]
    BlackMaterial: Any = Field(json_schema_extra={"editable": False})
    WhiteMaterial: Any = Field(json_schema_extra={"editable": False})
    scratch: Any | None = Field(default=None, json_schema_extra={"editable": False})
    edge_wear: Any | None = Field(default=None, json_schema_extra={"editable": False})
    light_factory: Any = Field(json_schema_extra={"editable": False})


class CeilingLightFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CeilingLightParameters

    def __init__(self, factory_seed, coarse=False, dimensions=[1.0, 1.0, 1.0]):
        super(CeilingLightFactory, self).__init__(factory_seed, coarse=coarse)
        self.dimensions = dimensions
        self.ceiling_light_default_params = [
            {
                "Radius": 0.2,
                "Thickness": 0.001,
                "InnerRadius": 0.2,
                "Height": 0.1,
                "InnerHeight": 0.1,
                "Curvature": 0.1,
            },
            {
                "Radius": 0.18,
                "Thickness": 0.05,
                "InnerRadius": 0.18,
                "Height": 0.1,
                "InnerHeight": 0.1,
                "Curvature": 0.25,
            },
            {
                "Radius": 0.2,
                "Thickness": 0.005,
                "InnerRadius": 0.18,
                "Height": 0.1,
                "InnerHeight": 0.03,
                "Curvature": 0.4,
            },
        ]
        self.init_legacy_parameters()

    @staticmethod
    def _sample_geometry(use_default: bool, defaults: list[dict]) -> dict[str, float]:
        if use_default:
            d = defaults[RI(0, len(defaults))]
            return {
                "Radius": d["Radius"],
                "Thickness": d["Thickness"],
                "InnerRadius": d["InnerRadius"] / d["Radius"],
                "Height": d["Height"],
                "InnerHeight": d["InnerHeight"] / d["Height"],
                "Curvature": d["Curvature"],
            }
        radius = clip_gaussian(0.12, 0.04, 0.1, 0.25)
        thickness = U(0.005, 0.05)
        height = 0.7 * clip_gaussian(0.09, 0.03, 0.07, 0.15)
        inner_height_ratio = U(0.5, 1.1)
        return {
            "Radius": radius,
            "Thickness": thickness,
            "InnerRadius": U(0.4, 0.9),
            "Height": height,
            "InnerHeight": inner_height_ratio,
            "Curvature": U(0.1, 0.5),
        }

    def sample_parameters(self, dimensions, use_default=False):
        return self._sample_geometry(use_default, self.ceiling_light_default_params)

    def _sample_materials(self) -> tuple[dict[str, Any], Any | None, Any | None]:
        wrapped_params = {
            "BlackMaterial": surface.shaderfunc_to_material(shader_black),
            "WhiteMaterial": surface.shaderfunc_to_material(shader_lamp_bulb_nonemissive),
        }
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch_fn, edge_wear_fn = material_assignments.wear_tear
        scratch = None if U() > scratch_prob else scratch_fn()
        edge_wear = None if U() > edge_wear_prob else edge_wear_fn()
        return wrapped_params, scratch, edge_wear

    def _sample_init_parameters(self, seed: int) -> CeilingLightParameters:
        geometry = self._sample_geometry(False, self.ceiling_light_default_params)
        materials, scratch, edge_wear = self._sample_materials()
        return CeilingLightParameters(
            seed=seed,
            **geometry,
            beveler_mult=U(1, 3),
            **materials,
            scratch=scratch,
            edge_wear=edge_wear,
            light_factory=PointLampFactory(seed),
        )

    def apply_parameters(
        self, params: CeilingLightParameters, *, spawn_scope: bool = True
    ) -> None:
        self.params = {
            "Radius": params.Radius,
            "Thickness": params.Thickness,
            "InnerRadius": params.Radius * params.InnerRadius,
            "Height": params.Height,
            "InnerHeight": params.Height * params.InnerHeight,
            "Curvature": params.Curvature,
            "BlackMaterial": params.BlackMaterial,
            "WhiteMaterial": params.WhiteMaterial,
        }
        self.scratch = params.scratch
        self.edge_wear = params.edge_wear
        self.light_factory = params.light_factory
        self.beveler = BevelSharp(mult=params.beveler_mult)
        self._use_fixed_spawn_draws = spawn_scope

    def get_material_params(self):
        return self._sample_materials()

    def create_placeholder(self, i, **params):
        obj = butil.spawn_cube()
        butil.modify_mesh(
            obj,
            "NODES",
            node_group=nodegroup_ceiling_light_geometry(),
            ng_inputs=self.params,
            apply=True,
        )
        return obj

    def create_asset(self, i, placeholder, **params):
        obj = butil.copy(placeholder, keep_materials=True)
        self.beveler(obj)

        lamp = self.light_factory.spawn_asset(i, loc=(0, 0, 0), rot=(0, 0, 0))

        butil.parent_to(lamp, obj, no_transform=True, no_inverse=True)
        lamp.location.z -= 0.03

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)


@node_utils.to_nodegroup(
    "nodegroup_ceiling_light_geometry", singleton=True, type="GeometryNodeTree"
)
def nodegroup_ceiling_light_geometry(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Radius", 0.2000),
            ("NodeSocketFloat", "Thickness", 0.0050),
            ("NodeSocketFloat", "InnerRadius", 0.1800),
            ("NodeSocketFloat", "Height", 0.1000),
            ("NodeSocketFloat", "InnerHeight", 0.0300),
            ("NodeSocketFloat", "Curvature", 0.4000),
            ("NodeSocketMaterial", "BlackMaterial", None),
            ("NodeSocketMaterial", "WhiteMaterial", None),
        ],
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz})

    curve_circle = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={"Resolution": 512, "Radius": group_input.outputs["Radius"]},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": curve_circle.outputs["Curve"],
        },
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": curve_to_mesh,
            "Offset Scale": group_input.outputs["Thickness"],
            "Individual": False,
        },
    )

    flip_faces = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": curve_to_mesh})

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [extrude_mesh.outputs["Mesh"], flip_faces]},
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth,
        input_kwargs={"Geometry": join_geometry, "Shade Smooth": False},
    )

    mesh_circle = nw.new_node(
        Nodes.MeshCircle,
        input_kwargs={"Radius": group_input.outputs["Radius"]},
        attrs={"fill_type": "NGON"},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_shade_smooth, mesh_circle]}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry_1,
            "Material": group_input.outputs["BlackMaterial"],
        },
    )

    ico_sphere_1 = nw.new_node(
        Nodes.MeshIcoSphere,
        input_kwargs={"Radius": group_input.outputs["InnerRadius"], "Subdivisions": 5},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": ico_sphere_1.outputs["Mesh"],
            "Name": "UVMap",
            3: ico_sphere_1.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    position_2 = nw.new_node(Nodes.InputPosition)

    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position_2})

    less_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: 0.0010},
        attrs={"operation": "LESS_THAN"},
    )

    separate_geometry_1 = nw.new_node(
        Nodes.SeparateGeometry,
        input_kwargs={"Geometry": store_named_attribute, "Selection": less_than},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["InnerHeight"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_1})

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": 1.0000, "Y": 1.0000, "Z": group_input.outputs["Curvature"]},
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": separate_geometry_1.outputs["Selection"],
            "Translation": combine_xyz_2,
            "Scale": combine_xyz_3,
        },
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_1})

    curve_line_1 = nw.new_node(
        Nodes.CurveLine,
        input_kwargs={"Start": (0.0000, 0.0000, -0.0010), "End": combine_xyz_1},
    )

    curve_circle_1 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Radius": group_input.outputs["InnerRadius"]}
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line_1,
            "Profile Curve": curve_circle_1.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform, curve_to_mesh_1]}
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": join_geometry_2,
            "Material": group_input.outputs["WhiteMaterial"],
        },
    )

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, set_material_1]}
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": join_geometry_3}
    )

    vector = nw.new_node(Nodes.Vector)
    vector.vector = (0.0000, 0.0000, 0.0000)

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": join_geometry_3,
            "Bounding Box": bounding_box.outputs["Bounding Box"],
            "LightPosition": vector,
        },
        attrs={"is_active_output": True},
    )

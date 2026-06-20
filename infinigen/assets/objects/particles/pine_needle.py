# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Lingjie Mei

from __future__ import annotations

from typing import Annotated, ClassVar

from numpy.random import normal as N
from pydantic import Field

from infinigen.assets import colors
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil


def shader_material(nw: NodeWrangler):
    object_info = nw.new_node(Nodes.ObjectInfo_Shader)

    colorramp = nw.new_node(
        Nodes.ColorRamp, input_kwargs={"Fac": object_info.outputs["Random"]}
    )
    colorramp.color_ramp.elements[0].position = 0.0000
    colorramp.color_ramp.elements[0].color = colors.hsv2rgba(colors.pine_needle_hsv())
    colorramp.color_ramp.elements[1].position = 1.0000
    colorramp.color_ramp.elements[1].color = colors.hsv2rgba(colors.pine_needle_hsv())

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF, input_kwargs={"Base Color": colorramp}
    )

    nw.new_node(
        Nodes.MaterialOutput, input_kwargs={"Surface": principled_bsdf}
    )


@node_utils.to_nodegroup(
    "nodegroup_pine_needle", singleton=False, type="GeometryNodeTree"
)
def nodegroup_pine_needle(nw: NodeWrangler):
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Scale", 0.0400),
            ("NodeSocketFloat", "Bend", 0.0300),
            ("NodeSocketFloat", "Radius", 0.0010),
        ],
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: (-1.0000, 0.0000, 0.0000),
            "Scale": group_input.outputs["Scale"],
        },
        attrs={"operation": "SCALE"},
    )

    scale_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: (0.0000, 1.0000, 0.0000),
            "Scale": group_input.outputs["Bend"],
        },
        attrs={"operation": "SCALE"},
    )

    scale_2 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: (1.0000, 0.0000, 0.0000),
            "Scale": group_input.outputs["Scale"],
        },
        attrs={"operation": "SCALE"},
    )

    quadratic_bezier = nw.new_node(
        Nodes.QuadraticBezier,
        input_kwargs={
            "Resolution": 5,
            "Start": scale.outputs["Vector"],
            "Middle": scale_1.outputs["Vector"],
            "End": scale_2.outputs["Vector"],
        },
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={"Resolution": 6, "Radius": group_input.outputs["Radius"]},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": quadratic_bezier,
            "Profile Curve": curve_circle.outputs["Curve"],
        },
    )

    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": curve_to_mesh},
        attrs={"is_active_output": True},
    )


class PineNeedleParameters(AssetParameters):
    s: Annotated[float, Field(ge=0.4, le=1.6, json_schema_extra={"editable": True})]
    Bend: Annotated[float, Field(ge=0.4, le=1.6, json_schema_extra={"editable": True})]
    Radius: Annotated[
        float, Field(ge=0.4, le=1.6, json_schema_extra={"editable": True})
    ]


class PineNeedleFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = PineNeedleParameters

    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    @staticmethod
    def _sample_shape_params() -> dict[str, float]:
        s = N(1, 0.2)
        return {
            "s": s,
            "Bend": N(1, 0.2),
            "Radius": N(1, 0.2),
        }

    def sample_params(self) -> dict[str, float]:
        params = self._sample_shape_params()
        s = params["s"]
        return {
            "Scale": 0.04 * s,
            "Bend": 0.03 * s * params["Bend"],
            "Radius": 0.001 * s * params["Radius"],
        }

    def _sample_init_parameters(self, seed: int) -> PineNeedleParameters:
        return PineNeedleParameters(seed=seed, **self._sample_shape_params())

    def apply_parameters(
        self, params: PineNeedleParameters, *, spawn_scope: bool = True
    ) -> None:
        self.s = params.s
        self.Bend = params.Bend
        self.Radius = params.Radius
        self._use_fixed_spawn_draws = spawn_scope

    def _ng_inputs(self) -> dict[str, float]:
        return {
            "Scale": 0.04 * self.s,
            "Bend": 0.03 * self.s * self.Bend,
            "Radius": 0.001 * self.s * self.Radius,
        }

    def create_asset(self, **_):
        obj = butil.spawn_vert("pine_needle")
        butil.modify_mesh(
            obj,
            "NODES",
            apply=True,
            node_group=nodegroup_pine_needle(),
            ng_inputs=self._ng_inputs(),
        )
        tag_object(obj, "pine_needle")
        return obj

    def finalize_assets(self, objs):
        surface.add_material(objs, shader_material)

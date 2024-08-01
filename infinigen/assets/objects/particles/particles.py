# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hei Law, Alexander Raistrick


import bpy
from numpy.random import normal as N

from infinigen.assets.materials import dirt
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_object
from infinigen.infinigen_gpl.surfaces import snow


def shader_raindrop(nw):
    glass_bsdf = nw.new_node(
        "ShaderNodeBsdfGlass",
        input_kwargs={
            "IOR": 1.33,
        },
    )
    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={
            "Surface": glass_bsdf,
        },
    )


def geo_raindrop(nw):
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            (
                "NodeSocketGeometry",
                "Geometry",
                None,
            )
        ],
    )

    position = nw.new_node(Nodes.InputPosition)

    vector_curves = nw.new_node(
        Nodes.VectorCurve,
        input_kwargs={
            "Vector": position,
        },
    )
    node_utils.assign_curve(
        vector_curves.mapping.curves[0],
        [(-1.0, -1.0), (1.0, 1.0)],
    )
    node_utils.assign_curve(
        vector_curves.mapping.curves[1],
        [(-1.0, -1.0), (1.0, 1.0)],
    )
    node_utils.assign_curve(
        vector_curves.mapping.curves[2],
        [(-1.0, -0.15 * N(1, 0.15)), (-0.6091, -0.0938), (1.0, 1.0)],
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": group_input.outputs["Geometry"],
            "Position": vector_curves,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Geometry": set_position,
        },
    )


class RaindropFactory(AssetFactory):
    def create_asset(self, **kwargs):
        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=1,
            enter_editmode=False,
            subdivisions=5,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )

        sphere = bpy.context.object

        surface.add_geomod(sphere, geo_raindrop, apply=True)
        tag_object(sphere, "raindrop")
        return sphere

    def finalize_assets(self, assets):
        surface.add_material(assets, shader_raindrop)


class DustMoteFactory(AssetFactory):
    def create_asset(self, **kwargs):
        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=1,
            subdivisions=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        tag_object(bpy.context.object, "dustmote")
        return bpy.context.object

    def finalize_assets(self, assets):
        dirt.apply(assets)


class SnowflakeFactory(AssetFactory):
    def create_asset(self, **params) -> bpy.types.Object:
        bpy.ops.mesh.primitive_circle_add(
            vertices=6,
            fill_type="TRIFAN",
        )
        tag_object(bpy.context.object, "snowflake")
        return bpy.context.object

    def finalize_assets(self, assets):
        snow.apply(assets, subsurface=0)

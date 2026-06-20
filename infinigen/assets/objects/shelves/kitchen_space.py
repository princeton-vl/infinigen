# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo, Stamatis Alexandropoulos

from __future__ import annotations

from typing import Annotated, ClassVar

import bpy
from mathutils import Vector
from numpy.random import choice, uniform
from pydantic import Field

from infinigen.assets.materials.ceramic.marble import shader_marble
from infinigen.assets.objects.shelves.kitchen_cabinet import KitchenCabinetFactory
from infinigen.assets.objects.tables.table_top import nodegroup_generate_table_top
from infinigen.assets.objects.wall_decorations.range_hood import RangeHoodFactory
from infinigen.assets.utils.object import new_bbox
from infinigen.core import surface, tagging
from infinigen.core import tags as t
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


def nodegroup_tag_cube(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    index = nw.new_node(Nodes.Index)

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: 5},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    cube = tagging.tag_nodegroup(
        nw, group_input.outputs["Geometry"], t.Subpart.SupportSurface, selection=equal
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": cube},
        attrs={"is_active_output": True},
    )


def geometry_nodes_add_cabinet_top(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0500

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Max"]}
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Min"]}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: separate_xyz.outputs["X"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: 1.4140},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: separate_xyz.outputs["Y"]},
        attrs={"operation": "SUBTRACT"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: subtract},
        attrs={"operation": "DIVIDE"},
    )

    generatetabletop = nw.new_node(
        nodegroup_generate_table_top().name,
        input_kwargs={
            "Thickness": value,
            "N-gon": 4,
            "Profile Width": multiply,
            "Aspect Ratio": divide,
            "Fillet Ratio": 0.0100,
            "Fillet Radius Vertical": 0.0100,
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": generatetabletop,
            "Material": surface.shaderfunc_to_material(shader_marble),
        },
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
    )

    divide_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Max"]}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": divide_1, "Z": separate_xyz_2.outputs["Z"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_material, "Translation": combine_xyz},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [group_input.outputs["Geometry"], transform_geometry]
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


def geometry_node_to_tagged_bbox(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler
    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": bounding_box, "Scale": (0.9700, 0.9700, 1.000)},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry},
        attrs={"is_active_output": True},
    )


def geometry_node_to_bbox(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler
    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": bounding_box, "Scale": (0.9700, 0.9700, 1.000)},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry},
        attrs={"is_active_output": True},
    )


class KitchenSpaceParameters(AssetParameters):
    kitchen_space_211: Annotated[
        float, Field(ge=1.0, le=2.0, json_schema_extra={"editable": True})
    ] = 1.0
    dimension_x: Annotated[float, Field(ge=0.7, le=1.0, json_schema_extra={"editable": True})]
    dimension_y: Annotated[float, Field(ge=1.7, le=5.0, json_schema_extra={"editable": True})]
    dimension_z: Annotated[float, Field(ge=2.3, le=2.5, json_schema_extra={"editable": True})]
    cabinet_bottom_height: Annotated[
        float, Field(ge=0.8, le=1.0, json_schema_extra={"editable": True})
    ]
    cabinet_top_height: Annotated[
        float, Field(ge=0.8, le=1.0, json_schema_extra={"editable": True})
    ]


class KitchenSpaceFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = KitchenSpaceParameters

    def __init__(self, factory_seed, coarse=False, dimensions=None, island=False):
        self._init_dimensions = dimensions
        self._init_island = island
        super(KitchenSpaceFactory, self).__init__(factory_seed, coarse=coarse)
        self.init_legacy_parameters()

    def sample_geometry_parameters(self, dimensions):
        self.cabinet_bottom_height = uniform(0.8, 1.0)
        self.cabinet_top_height = uniform(0.8, 1.0)

    def _sample_init_parameters(self, seed: int) -> KitchenSpaceParameters:
        if self._init_dimensions is None:
            dimension_x = uniform(0.7, 1)
            dimension_y = uniform(1.7, 5)
            dimension_z = uniform(2.3, 2.5)
        else:
            dimension_x = self._init_dimensions.x
            dimension_y = self._init_dimensions.y
            dimension_z = self._init_dimensions.z
        kitchen_space_211 = uniform(1.5, 2.0) if self._init_island else 1.0
        return KitchenSpaceParameters(
            seed=seed,
            kitchen_space_211=kitchen_space_211,
            dimension_x=dimension_x,
            dimension_y=dimension_y,
            dimension_z=dimension_z,
            cabinet_bottom_height=uniform(0.8, 1.0),
            cabinet_top_height=uniform(0.8, 1.0),
        )

    def apply_parameters(
        self, params: KitchenSpaceParameters, *, spawn_scope: bool = True
    ) -> None:
        self.island = self._init_island
        self.dimensions = Vector(
            (
                params.dimension_x * params.kitchen_space_211,
                params.dimension_y,
                params.dimension_z,
            )
        )
        self.cabinet_bottom_height = params.cabinet_bottom_height
        self.cabinet_top_height = params.cabinet_top_height
        self.params = {}
        self._use_fixed_spawn_draws = spawn_scope

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        x, y, z = self.dimensions
        box = new_bbox(
            -x / 2 * 1.08, x / 2 * 1.08, 0, y, 0, self.cabinet_bottom_height + 0.13
        )
        surface.add_geomod(box, nodegroup_tag_cube, apply=True)

        if not self.island:
            box_top = new_bbox(
                -x / 2, x * 0.16, 0, y, z - self.cabinet_top_height - 0.1, z
            )
            box = butil.join_objects([box, box_top])

        return box

    def create_asset(self, **params):
        x, y, z = self.dimensions
        parts = []

        cabinet_bottom_height = self.cabinet_bottom_height
        cabinet_top_height = self.cabinet_top_height

        cabinet_bottom_factory = KitchenCabinetFactory(
            self.factory_seed,
            dimensions=(x, y - 0.15, cabinet_bottom_height),
            drawer_only=True,
        )
        cabinet_bottom = cabinet_bottom_factory(i=0)
        parts.append(cabinet_bottom)

        surface.add_geomod(cabinet_bottom, geometry_nodes_add_cabinet_top, apply=True)

        if not self.island:
            # top
            top_mid_width = uniform(1.0, 1.3)
            cabinet_top_width = (y - top_mid_width) / 2.0 - 0.05

            cabinet_top_factory = KitchenCabinetFactory(
                self.factory_seed,
                dimensions=(x / 2.0, cabinet_top_width, cabinet_top_height),
                drawer_only=False,
            )
            cabinet_top_left = cabinet_top_factory(i=0)
            cabinet_top_right = cabinet_top_factory(i=1)

            cabinet_top_left.location = (-x / 4.0, 0.0, z - cabinet_top_height)
            cabinet_top_right.location = (
                -x / 4.0,
                y - cabinet_top_width,
                z - cabinet_top_height,
            )

            # hood / cab
            # mid_style = choice(['range_hood', 'cabinet'])
            # mid_style = 'range_hood'
            mid_style = choice(["cabinet"])
            if mid_style == "range_hood":
                range_hood_factory = RangeHoodFactory(
                    self.factory_seed,
                    dimensions=(x * 0.66, top_mid_width + 0.15, cabinet_top_height),
                )
                top_mid = range_hood_factory(i=0)
                top_mid.location = (-x * 0.5, y / 2.0, z - cabinet_top_height + 0.05)

            elif mid_style == "cabinet":
                cabinet_top_mid_factory = KitchenCabinetFactory(
                    self.factory_seed,
                    dimensions=(x * 0.66, top_mid_width, cabinet_top_height * 0.8),
                    drawer_only=False,
                )
                top_mid = cabinet_top_mid_factory(i=0)
                top_mid.location = (
                    -x / 6.0,
                    y / 2.0 - top_mid_width / 2.0,
                    z - (cabinet_top_height * 0.8),
                )

            else:
                raise NotImplementedError

            # parts += [sink, cabinet_top_left, cabinet_top_right, top_mid]
            parts += [cabinet_top_left, cabinet_top_right, top_mid]

        kitchen_space = butil.join_objects(
            parts
        )  # [cabinet_bottom, sink, cabinet_top_left, cabinet_top_right, top_mid])

        if not self.island:
            kitchen_space.dimensions = self.dimensions
        butil.apply_transform(kitchen_space)

        tagging.tag_system.relabel_obj(kitchen_space)

        return kitchen_space


class KitchenIslandParameters(AssetParameters):
    kitchen_space_211: Annotated[
        float, Field(ge=1.5, le=2.0, json_schema_extra={"editable": True})
    ]
    dimension_x: Annotated[float, Field(ge=0.7, le=1.0, json_schema_extra={"editable": True})]
    dimension_y: Annotated[float, Field(ge=1.7, le=5.0, json_schema_extra={"editable": True})]
    dimension_z: Annotated[float, Field(ge=2.3, le=2.5, json_schema_extra={"editable": True})]
    cabinet_bottom_height: Annotated[
        float, Field(ge=0.8, le=1.0, json_schema_extra={"editable": True})
    ]
    cabinet_top_height: Annotated[
        float, Field(ge=0.8, le=1.0, json_schema_extra={"editable": True})
    ]


class KitchenIslandFactory(KitchenSpaceFactory):
    parameters_model: ClassVar[type[AssetParameters]] = KitchenIslandParameters

    def __init__(self, factory_seed, coarse=False):
        self._init_dimensions = None
        self._init_island = True
        AssetFactory.__init__(self, factory_seed, coarse=coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> KitchenIslandParameters:
        return KitchenIslandParameters(
            seed=seed,
            kitchen_space_211=uniform(1.5, 2.0),
            dimension_x=uniform(0.7, 1),
            dimension_y=uniform(1.7, 5),
            dimension_z=uniform(2.3, 2.5),
            cabinet_bottom_height=uniform(0.8, 1.0),
            cabinet_top_height=uniform(0.8, 1.0),
        )

    def apply_parameters(
        self, params: KitchenIslandParameters, *, spawn_scope: bool = True
    ) -> None:
        self.island = True
        self.dimensions = Vector(
            (
                params.dimension_x * params.kitchen_space_211,
                params.dimension_y,
                params.dimension_z,
            )
        )
        self.cabinet_bottom_height = params.cabinet_bottom_height
        self.cabinet_top_height = params.cabinet_top_height
        self.params = {}
        self._use_fixed_spawn_draws = spawn_scope

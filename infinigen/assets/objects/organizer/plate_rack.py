# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

import bpy
from numpy.random import randint, uniform

from infinigen.assets.materials import shader_wood
from infinigen.assets.materials.plastics.plastic_rough import shader_rough_plastic
from infinigen.core import surface, tagging
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil


@node_utils.to_nodegroup(
    "nodegroup_plate_rack_connect", singleton=False, type="GeometryNodeTree"
)
def nodegroup_plate_rack_connect(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Radius", 1.0000),
            ("NodeSocketFloat", "Value1", 0.5000),
            ("NodeSocketFloat", "Value", 0.5000),
        ],
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Value1"], 1: 2.0000, 2: -0.0020},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": group_input.outputs["Radius"], "Depth": multiply_add},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Name": "uv_map",
            3: cylinder.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "CORNER"},
    )

    multiply_add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Value"], 2: -uniform(0.02, 0.045)},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_add_1})

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Translation": combine_xyz,
            "Rotation": (1.5708, 0.0000, 0.0000),
        },
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": transform, "Scale": (-1.0000, 1.0000, 1.0000)},
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_2, transform]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_2},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_rack_cyn", singleton=False, type="GeometryNodeTree")
def nodegroup_rack_cyn(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Radius", 1.0000),
            ("NodeSocketFloat", "Value", 0.5000),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Value"], 1: 0.0000}
    )

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": group_input.outputs["Radius"], "Depth": add},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Name": "uv_map",
            3: cylinder.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "CORNER"},
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add, 2: 0.0010},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_add})

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute, "Translation": combine_xyz_4},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_2},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_rack_base", singleton=False, type="GeometryNodeTree"
)
def nodegroup_rack_base(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Instance", None),
            ("NodeSocketFloat", "Value1", 0.5000),
            ("NodeSocketFloat", "Value2", 0.5000),
            ("NodeSocketFloat", "Value3", 0.5000),
            ("NodeSocketInt", "Count", 10),
        ],
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Value1"], 1: 0.0000}
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Value2"], 1: 0.0000}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add_1, "Z": add_1}
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            3: cube.outputs["UV Map"],
        },
        attrs={"data_type": "FLOAT_VECTOR", "domain": "CORNER"},
    )

    add_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Value3"], 1: 0.0000}
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": add_2})

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute, "Translation": combine_xyz_1},
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add, 2: -0.0150},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_add, "Y": add_2}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_add, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": add_2}
    )

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={
            "Count": group_input.outputs["Count"],
            "Start Location": combine_xyz_2,
            "Offset": combine_xyz_3,
        },
        attrs={"mode": "END_POINTS"},
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": mesh_line, "Instance": group_input.outputs["Instance"]},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": instance_on_points}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Base": transform, "Racks": realize_instances},
        attrs={"is_active_output": True},
    )


def rack_geometry_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.5 of the node_transpiler

    rack_radius = nw.new_node(Nodes.Value, label="rack_radius")
    rack_radius.outputs[0].default_value = kwargs["rack_radius"]

    rack_height = nw.new_node(Nodes.Value, label="rack_height")
    rack_height.outputs[0].default_value = kwargs["rack_height"]

    rack_cyn = nw.new_node(
        nodegroup_rack_cyn().name,
        input_kwargs={"Radius": rack_radius, "Value": rack_height},
    )

    base_length = nw.new_node(Nodes.Value, label="base_length")
    base_length.outputs[0].default_value = kwargs["base_length"]

    base_width = nw.new_node(Nodes.Value, label="base_width")
    base_width.outputs[0].default_value = kwargs["base_width"]

    base_gap = nw.new_node(Nodes.Value, label="base_gap")
    base_gap.outputs[0].default_value = kwargs["base_gap"]

    integer = nw.new_node(Nodes.Integer)
    integer.integer = kwargs["num_rack"]

    rack_base = nw.new_node(
        nodegroup_rack_base().name,
        input_kwargs={
            "Instance": rack_cyn,
            "Value1": base_length,
            "Value2": base_width,
            "Value3": base_gap,
            "Count": integer,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [rack_base.outputs["Base"], rack_base.outputs["Racks"]]
        },
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry, "Scale": (1.0000, -1.0000, 1.0000)},
    )

    plate_rack_connect = nw.new_node(
        nodegroup_plate_rack_connect().name,
        input_kwargs={"Radius": rack_radius, "Value1": base_gap, "Value": base_length},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform_1, join_geometry, plate_rack_connect]},
    )

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: base_width}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": combine_xyz},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": transform}
    )

    triangulate = nw.new_node(
        "GeometryNodeTriangulate", input_kwargs={"Mesh": realize_instances}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": triangulate,
            "Material": surface.shaderfunc_to_material(shader_wood),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_material},
        attrs={"is_active_output": True},
    )


def plate_geometry_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.5 of the node_transpiler

    radius = nw.new_node(Nodes.Value, label="radius")
    radius.outputs[0].default_value = kwargs["radius"]

    thickness = nw.new_node(Nodes.Value, label="thickness")
    thickness.outputs[0].default_value = kwargs["thickness"]

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Vertices": 64, "Radius": radius, "Depth": thickness},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": radius})

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    triangulate = nw.new_node(
        "GeometryNodeTriangulate", input_kwargs={"Mesh": transform_geometry}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": triangulate,
            "Material": surface.shaderfunc_to_material(shader_rough_plastic),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_material},
        attrs={"is_active_output": True},
    )


class PlateRackBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(PlateRackBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.params = params

    def sample_params(self):
        return self.params.copy()

    def get_place_points(self, params):
        # compute the lowest point in the bezier curve
        xs = []
        for i in range(params["num_rack"] - 1):
            l = params["base_length"]
            d = (l - 0.03) / (params["num_rack"] - 1)
            x = -l / 2.0 + 0.015 + (i + 0.5) * d
            xs.append(x)

        y = 0
        z = params["base_width"]

        place_points = []
        for x in xs:
            place_points.append((x, y, z))

        return place_points

    def get_asset_params(self, i=0):
        params = self.sample_params()
        if params.get("num_rack", None) is None:
            params["num_rack"] = randint(3, 7)
        if params.get("rack_radius", None) is None:
            params["rack_radius"] = uniform(0.0025, 0.006)
        if params.get("rack_height", None) is None:
            params["rack_height"] = uniform(0.08, 0.15)
        if params.get("base_length", None) is None:
            params["base_length"] = (params["num_rack"] - 1) * uniform(
                0.03, 0.06
            ) + 0.03
        if params.get("base_gap", None) is None:
            params["base_gap"] = uniform(0.05, 0.08)
        if params.get("base_width", None) is None:
            params["base_width"] = uniform(0.015, 0.03)

        return params

    def create_asset(self, i=0, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=1,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object

        obj_params = self.get_asset_params(i)
        surface.add_geomod(
            obj, rack_geometry_nodes, attributes=[], apply=True, input_kwargs=obj_params
        )
        tagging.tag_system.relabel_obj(obj)

        place_points = self.get_place_points(obj_params)

        return obj, place_points


class PlateBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(PlateBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.params = params

    def sample_params(self):
        return self.params.copy()

    def get_asset_params(self, i=0):
        params = self.sample_params()
        if params.get("radius", None) is None:
            params["radius"] = uniform(0.15, 0.25)
        if params.get("thickness", None) is None:
            params["thickness"] = uniform(0.01, 0.025)

        return params

    def create_asset(self, i=0, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=1,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object

        obj_params = self.get_asset_params(i)
        surface.add_geomod(
            obj,
            plate_geometry_nodes,
            attributes=[],
            apply=True,
            input_kwargs=obj_params,
        )
        tagging.tag_system.relabel_obj(obj)

        return obj


class PlateOnRackBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(PlateOnRackBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.params = params

        self.rack_fac = PlateRackBaseFactory(factory_seed, params=params)
        self.plate_fac = PlateBaseFactory(factory_seed, params=params)

    def get_asset_params(self, i):
        if self.params.get("base_gap", None) is None:
            d = uniform(0.05, 0.08)
            self.rack_fac.params["base_gap"] = d
            self.plate_fac.params["radius"] = d + uniform(0.025, 0.06)

    def create_asset(self, i, **params):
        self.get_asset_params(i)
        rack, place_points = self.rack_fac.create_asset(i)
        plate = self.plate_fac.create_asset(i)

        plate.location = place_points[0]
        butil.apply_transform(plate, loc=True)

        return plate

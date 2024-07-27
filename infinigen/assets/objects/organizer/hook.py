# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

import bpy
import numpy as np
from numpy.random import randint, uniform

from infinigen.assets.materials import shader_brushed_metal, shader_rough_plastic
from infinigen.core import surface, tagging
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil


def hook_geometry_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.5 of the node_transpiler

    hook_num = nw.new_node(Nodes.Integer, label="hook_num")
    hook_num.integer = kwargs["num_hook"]

    add = nw.new_node(Nodes.Math, input_kwargs={0: hook_num, 1: -1.0000})

    hook_gap = nw.new_node(Nodes.Value, label="hook_gap")
    hook_gap.outputs[0].default_value = kwargs["hook_gap"]

    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: hook_gap, 1: add}, attrs={"operation": "MULTIPLY"}
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply}, attrs={"operation": "MULTIPLY"}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_2})

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_1})

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={
            "Count": add,
            "Start Location": combine_xyz_2,
            "Offset": combine_xyz_1,
        },
        attrs={"mode": "END_POINTS"},
    )

    bezier_segment = nw.new_node(
        Nodes.CurveBezierSegment,
        input_kwargs={
            "Start": (0.0000, 0.0000, 0.0000),
            "Start Handle": (0.0000, 0.0000, kwargs["init_handle"]),
            "End Handle": kwargs["curve_handle"],
            "End": kwargs["curve_end_point"],
        },
    )

    curve_line = nw.new_node(Nodes.CurveLine)

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [bezier_segment, curve_line]}
    )

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Factor": spline_parameter.outputs["Factor"]}
    )
    node_utils.assign_curve(
        float_curve.mapping.curves[0], [(0.0000, 0.8), (0.5, 0.8), (1.0000, 0.8)]
    )

    raduis = nw.new_node(Nodes.Value, label="raduis")
    raduis.outputs[0].default_value = kwargs["hook_radius"]

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: float_curve, 1: raduis},
        attrs={"operation": "MULTIPLY"},
    )

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius,
        input_kwargs={"Curve": join_geometry_3, "Radius": multiply_3},
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={
            "Resolution": kwargs["hook_resolution"],
            "Point 1": (1.0000, 0.0000, 0.0000),
            "Point 3": (-1.0000, 0.0000, 0.0000),
        },
        attrs={"mode": "POINTS"},
    )

    hook_reshape = nw.new_node(Nodes.Vector, label="hook_reshape")
    hook_reshape.vector = (1.0000, 1.0000, 1.0000)

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": curve_circle.outputs["Curve"], "Scale": hook_reshape},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": transform_geometry_2,
            "Fill Caps": True,
        },
    )

    hook_size = nw.new_node(Nodes.Value, label="hook_size")
    hook_size.outputs[0].default_value = kwargs["hook_size"]

    transform_geometry = nw.new_node(
        Nodes.Transform, input_kwargs={"Geometry": curve_to_mesh, "Scale": hook_size}
    )

    realize_instances_1 = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": transform_geometry}
    )

    merge_by_distance_1 = nw.new_node(
        Nodes.MergeByDistance, input_kwargs={"Geometry": realize_instances_1}
    )

    instance_on_points = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={"Points": mesh_line, "Instance": merge_by_distance_1},
    )

    scale_instances = nw.new_node(
        Nodes.ScaleInstances, input_kwargs={"Instances": instance_on_points}
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": scale_instances,
            "Material": surface.shaderfunc_to_material(shader_brushed_metal),
        },
    )

    board_side_gap = nw.new_node(Nodes.Value, label="board_side_gap")
    board_side_gap.outputs[0].default_value = kwargs["board_side_gap"]

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply, 1: board_side_gap})

    board_thickness = nw.new_node(Nodes.Value, label="board_thickness")
    board_thickness.outputs[0].default_value = kwargs["board_thickness"]

    board_height = nw.new_node(Nodes.Value, label="board_height")
    board_height.outputs[0].default_value = kwargs["board_height"]

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": add_1, "Y": board_thickness, "Z": board_height},
    )

    cube = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz})

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: board_thickness, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math, input_kwargs={0: board_height}, attrs={"operation": "MULTIPLY"}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: hook_size, 1: multiply_5},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_4, "Z": subtract}
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": cube.outputs["Mesh"], "Translation": combine_xyz_3},
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_1,
            "Material": surface.shaderfunc_to_material(shader_rough_plastic),
        },
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material, set_material_1]}
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry_2}
    )

    triangulate = nw.new_node(
        "GeometryNodeTriangulate", input_kwargs={"Mesh": realize_instances}
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": triangulate, "Rotation": (0.0000, 0.0000, -1.5708)},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry_3},
        attrs={"is_active_output": True},
    )


def spatula_geometry_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.5 of the node_transpiler

    handle_length = nw.new_node(Nodes.Value, label="handle_length")
    handle_length.outputs[0].default_value = kwargs["handle_length"]

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": handle_length})

    mesh_line = nw.new_node(
        Nodes.MeshLine,
        input_kwargs={"Count": 64, "Offset": combine_xyz},
        attrs={"mode": "END_POINTS"},
    )

    mesh_to_curve = nw.new_node(Nodes.MeshToCurve, input_kwargs={"Mesh": mesh_line})

    handle_radius = nw.new_node(Nodes.Value, label="handle_radius")
    handle_radius.outputs[0].default_value = kwargs["handle_radius"]

    spline_parameter = nw.new_node(Nodes.SplineParameter)

    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": spline_parameter.outputs["Factor"]}
    )
    node_utils.assign_curve(
        float_curve.mapping.curves[0], kwargs["handle_control_points"]
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: handle_radius, 1: float_curve},
        attrs={"operation": "MULTIPLY"},
    )

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius, input_kwargs={"Curve": mesh_to_curve, "Radius": multiply}
    )

    curve_circle = nw.new_node(Nodes.CurveCircle)

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": curve_to_mesh,
            "Scale": (kwargs["handle_ratio"], 1.0, 1.0),
        },
    )

    hole_radius = nw.new_node(Nodes.Value, label="hole_radius")
    hole_radius.outputs[0].default_value = kwargs["hole_radius"]

    cylinder = nw.new_node(
        "GeometryNodeMeshCylinder",
        input_kwargs={"Radius": hole_radius, "Depth": 0.1000},
    )

    hole_place_ratio = nw.new_node(Nodes.Value, label="hole_placement")
    hole_place_ratio.outputs[0].default_value = kwargs["hole_placement"]

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: handle_length, 1: hole_place_ratio},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_1})

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cylinder.outputs["Mesh"],
            "Translation": combine_xyz_1,
            "Rotation": (0.0000, 1.5708, 0.0000),
            "Scale": (kwargs["hole_ratio"], 1.0000, 1.0000),
        },
    )

    difference = nw.new_node(
        Nodes.MeshBoolean,
        input_kwargs={"Mesh 1": transform_geometry, "Mesh 2": transform_geometry_1},
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": (
                kwargs["plate_thickness"],
                kwargs["plate_width"],
                kwargs["plate_length"],
            ),
            "Vertices X": 4,
            "Vertices Y": 4,
            "Vertices Z": 4,
        },
    )

    transform_geometry_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Translation": (0.0000, 0.0000, -kwargs["plate_length"] / 2.0),
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [difference.outputs["Mesh"], transform_geometry_3]},
    )

    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry}
    )

    triangulate = nw.new_node(
        "GeometryNodeTriangulate", input_kwargs={"Mesh": realize_instances}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply_2})

    transform_geometry_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": triangulate, "Translation": combine_xyz_2},
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_geometry_2,
            "Material": surface.shaderfunc_to_material(shader_rough_plastic),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_material},
        attrs={"is_active_output": True},
    )


class HookBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(HookBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.params = params

    def sample_params(self):
        return self.params.copy()

    def get_hang_points(self, params):
        # compute the lowest point in the bezier curve
        x = params["init_handle"]
        y = params["curve_handle"][2] - params["init_handle"]
        z = params["curve_end_point"][2] - params["curve_handle"][2]

        t1 = (x - y + np.sqrt(y**2 - x * z)) / (x + z - 2 * y)
        t2 = (x - y - np.sqrt(y**2 - x * z)) / (x + z - 2 * y)

        t = 0
        if t1 >= 0 and t1 <= 1:
            t = max(t1, t)
        if t2 >= 0 and t2 <= 1:
            t = max(t2, t)
        if t == 0:
            t = 0.5

        # get x, z coordinate
        alpha1 = 3 * ((1 - t) ** 2) * t
        alpha2 = 3 * (1 - t) * (t**2)
        alpha3 = t**3

        z = (
            alpha1 * params["init_handle"]
            + alpha2 * params["curve_handle"][-1]
            + alpha3 * params["curve_end_point"][-1]
        )
        x = alpha2 * params["curve_handle"][-2] + alpha3 * params["curve_end_point"][-2]

        ys = []
        total_length = (
            params["board_side_gap"] + (params["num_hook"] - 1) * params["hook_gap"]
        )
        for i in range(params["num_hook"]):
            y = (
                -total_length / 2.0
                + params["board_side_gap"] / 2.0
                + i * params["hook_gap"]
            )
            ys.append(y)

        hang_points = []
        for y in ys:
            hang_points.append((x * params["hook_size"], y, z * params["hook_size"]))

        return hang_points

    def get_asset_params(self, i=0):
        params = self.sample_params()
        if params.get("num_hook", None) is None:
            params["num_hook"] = randint(3, 6)
        if params.get("hook_size", None) is None:
            params["hook_size"] = uniform(0.05, 0.1)
        if params.get("hook_radius", None) is None:
            params["hook_radius"] = uniform(0.002, 0.004) / params["hook_size"]
        else:
            params["hook_radius"] = params["hook_radius"] / params["hook_size"]

        if params.get("hook_resolution", None) is None:
            params["hook_resolution"] = np.random.choice([4, 32], p=[0.5, 0.5])

        if params.get("hook_gap", None) is None:
            params["hook_gap"] = uniform(0.04, 0.08)
        if params.get("board_height", None) is None:
            params["board_height"] = params["hook_size"] + uniform(-0.02, 0.01)
        if params.get("board_thickness", None) is None:
            params["board_thickness"] = uniform(0.005, 0.015)
        if params.get("board_side_gap", None) is None:
            params["board_side_gap"] = uniform(0.03, 0.05)

        params["init_handle"] = uniform(-0.15, -0.25)
        params["curve_handle"] = (0, uniform(0.15, 0.35), uniform(-0.15, -0.35))
        params["curve_end_point"] = (0, uniform(0.35, 0.55), uniform(-0.05, 0.15))

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
            obj, hook_geometry_nodes, attributes=[], apply=True, input_kwargs=obj_params
        )
        tagging.tag_system.relabel_obj(obj)

        hang_points = self.get_hang_points(obj_params)

        return obj, hang_points


class SpatulaBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(SpatulaBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.params = params

    def sample_params(self):
        return self.params.copy()

    def get_asset_params(self, i=0):
        params = self.sample_params()

        if params.get("hole_radius", None) is None:
            params["hole_radius"] = uniform(0.003, 0.008)
        if params.get("hole_placement", None) is None:
            params["hole_placement"] = uniform(0.75, 0.9)
        if params.get("hole_ratio", None) is None:
            params["hole_ratio"] = uniform(0.8, 2.0)

        if params.get("handle_length", None) is None:
            params["handle_length"] = uniform(0.15, 0.25)

        if params.get("handle_ratio", None) is None:
            params["handle_ratio"] = uniform(0.1, 0.4)
        if params.get("handle_control_points", None) is None:
            params["handle_control_points"] = [
                (0, 0.5),
                (0.5, uniform(0.45, 0.65)),
                (1.0, uniform(0.4, 0.6)),
            ]
        if params.get("handle_radius", None) is None:
            params["handle_radius"] = (
                params["hole_radius"] / params["handle_control_points"][0][1]
            ) / uniform(0.6, 0.8)

        if params.get("plate_thickness", None) is None:
            params["plate_thickness"] = uniform(0.005, 0.01)
        if params.get("plate_width", None) is None:
            params["plate_width"] = uniform(0.04, 0.06)
        if params.get("plate_length", None) is None:
            params["plate_length"] = uniform(0.05, 0.08)

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
            spatula_geometry_nodes,
            attributes=[],
            apply=True,
            input_kwargs=obj_params,
        )
        tagging.tag_system.relabel_obj(obj)

        return obj


class SpatulaOnHookBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(SpatulaOnHookBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.params = params

        self.hook_fac = HookBaseFactory(factory_seed, params=params)
        self.spatula_fac = SpatulaBaseFactory(factory_seed, params=params)

    def get_asset_params(self, i):
        if self.params.get("hook_radius", None) is None:
            r = uniform(0.002, 0.0035)
            self.hook_fac.params["hook_radius"] = r
            self.spatula_fac.params["hole_radius"] = r / uniform(0.3, 0.6)

    def create_asset(self, i, **params):
        self.get_asset_params(i)
        hook, hang_points = self.hook_fac.create_asset(i)
        spatula = self.spatula_fac.create_asset(i)

        spatula.location = hang_points[0]
        butil.apply_transform(spatula, loc=True)

        return hook

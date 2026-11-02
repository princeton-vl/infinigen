# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Beining Han: original Infinigen fern (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/small_plants/fern.py)
# - David Yan, Jia Deng: optimized implementation

import numpy as np
from numpy.random import normal, randint, uniform

from infinigen.assets.objects.small_plants.fern import (
    FernFactory as _DefaultFernFactory,
)
from infinigen.assets.objects.small_plants.fern import (
    nodegroup_pinnae_level1_instance_rotation,
    nodegroup_pinnae_level1_stein,
    nodegroup_pinnae_level1_xaxis_rotation,
    nodegroup_pinnae_level2_stein,
    random_pinnae_level2_curvature,
)
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler

_templates = None


class _TemplateScope:
    """While active, node groups built by the builders below are templated."""

    def __enter__(self):
        global _templates
        self.outer = _templates
        _templates = {}
        return self

    def __exit__(self, *_):
        global _templates
        _templates = self.outer
        return False


def _instantiate(key, build, apply):
    """The node group `build()` makes. In a template scope the first call's group is
    the template, and later calls copy it and re-parametrize the copy with
    `apply(group)`; builders without parameters (`_no_params`) share the template."""
    if _templates is None:
        return build()
    template = _templates.get(key)
    if template is None:
        group = build()
        _templates[key] = group
        return group
    if apply is _no_params:
        return template
    group = template.copy()
    apply(group)
    return group


def _float_curves(group):
    return [n for n in group.nodes if n.bl_idname == Nodes.FloatCurve]


def _set_curve(node, points):
    curve = node.mapping.curves[0]
    for i, p in enumerate(points):
        curve.points[i].location = p


def _repoint(group, sub_groups):
    nodes = [n for n in group.nodes if n.bl_idname == "GeometryNodeGroup"]
    assert len(nodes) == len(sub_groups), (group.name, len(nodes), len(sub_groups))
    for node, sub in zip(nodes, sub_groups):
        if node.node_tree != sub:  # shared sub-groups are already in place
            node.node_tree = sub


def _no_params(_group):
    pass


@node_utils.to_nodegroup(
    "nodegroup_pinnae_level1_yaxis_rotation", singleton=False, type="GeometryNodeTree"
)
def nodegroup_pinnae_level1_yaxis_rotation(nw: NodeWrangler, points):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "From Max", 1.0),
            ("NodeSocketFloat", "Value", 1.0),
        ],
    )
    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input.outputs["Value"],
            2: group_input.outputs["From Max"],
        },
    )
    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": map_range.outputs["Result"]}
    )
    node_utils.assign_curve(float_curve.mapping.curves[0], points)
    add = nw.new_node(
        Nodes.Math, input_kwargs={0: float_curve, 1: -0.5}, attrs={"operation": "ADD"}
    )
    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 1.0}, attrs={"operation": "MULTIPLY"}
    )
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Value": multiply})


def _yaxis_points():
    curvature = np.clip(normal(0, 0.3, 1), a_min=-0.4, a_max=0.4)
    return [
        (0.0, 0.5),
        (0.1, curvature / 5.0 + 0.5),
        (0.25, curvature / 2.5 + 0.5),
        (0.45, curvature / 1.5 + 0.5),
        (0.6, curvature / 1.2 + 0.5),
        (1.0, curvature + 0.5),
    ]


def level1_yaxis_rotation(points):
    return _instantiate(
        "level1_yaxis_rotation",
        lambda: nodegroup_pinnae_level1_yaxis_rotation(points),
        lambda g: _set_curve(_float_curves(g)[0], points),
    )


@node_utils.to_nodegroup(
    "nodegroup_pinnae_level1_zaxis_rotation", singleton=False, type="GeometryNodeTree"
)
def nodegroup_pinnae_level1_zaxis_rotation(nw: NodeWrangler, points):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "From Max", 1.0),
            ("NodeSocketFloat", "Value", 1.0),
        ],
    )
    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input.outputs["Value"],
            2: group_input.outputs["From Max"],
        },
    )
    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": map_range.outputs["Result"]}
    )
    node_utils.assign_curve(float_curve.mapping.curves[0], points)
    add = nw.new_node(
        Nodes.Math, input_kwargs={0: float_curve, 1: -0.5}, attrs={"operation": "ADD"}
    )
    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 1.0}, attrs={"operation": "MULTIPLY"}
    )
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Value": multiply})


def _zaxis_points():
    curvature = normal(0, 0.2, 1)
    return [
        (0.0, 0.5),
        (0.1, curvature / 5.0 + 0.5),
        (0.25, curvature / 2.5 + 0.5),
        (0.45, curvature / 1.5 + 0.5),
        (0.6, curvature / 1.2 + 0.5),
        (1.0, curvature + 0.5),
    ]


def level1_zaxis_rotation(points):
    return _instantiate(
        "level1_zaxis_rotation",
        lambda: nodegroup_pinnae_level1_zaxis_rotation(points),
        lambda g: _set_curve(_float_curves(g)[0], points),
    )


@node_utils.to_nodegroup(
    "nodegroup_pinnae_level1_gravity_rotation", singleton=False, type="GeometryNodeTree"
)
def nodegroup_pinnae_level1_gravity_rotation(nw: NodeWrangler, points):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "From Max", 1.0),
            ("NodeSocketFloat", "Value", 1.0),
        ],
    )
    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input.outputs["Value"],
            2: group_input.outputs["From Max"],
        },
    )
    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": map_range.outputs["Result"]}
    )
    node_utils.assign_curve(float_curve.mapping.curves[0], points)
    add = nw.new_node(
        Nodes.Math, input_kwargs={0: float_curve, 1: -0.5}, attrs={"operation": "ADD"}
    )
    multiply = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 1}, attrs={"operation": "MULTIPLY"}
    )
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Value": multiply})


def _gravity_points(gravity_rotation=1.0):
    curvature = uniform(0.25, 0.42, size=(1,))[0] * gravity_rotation
    return [
        (0.0, 0.5),
        (0.1, curvature / 5.0 + 0.5),
        (0.25, curvature / 2.5 + 0.5),
        (0.45, curvature / 1.67 + 0.5),
        (0.6, curvature / 1.25 + 0.5),
        (1.0, curvature + 0.5),
    ]


def level1_gravity_rotation(points):
    return _instantiate(
        "level1_gravity_rotation",
        lambda: nodegroup_pinnae_level1_gravity_rotation(points),
        lambda g: _set_curve(_float_curves(g)[0], points),
    )


def level1_xaxis_rotation():
    return _instantiate(
        "level1_xaxis_rotation", nodegroup_pinnae_level1_xaxis_rotation, _no_params
    )


def level1_stein():
    return _instantiate("level1_stein", nodegroup_pinnae_level1_stein, _no_params)


@node_utils.to_nodegroup(
    "nodegroup_pinnae_level1_scale", singleton=False, type="GeometryNodeTree"
)
def nodegroup_pinnae_level1_scale(nw: NodeWrangler, points):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Value1", 1.0),
            ("NodeSocketFloat", "Value2", 1.0),
        ],
    )
    pinnae_contour_float_curve = nw.new_node(
        Nodes.FloatCurve,
        input_kwargs={"Value": group_input.outputs["Value1"]},
        label="PinnaeContourFloatCurve",
    )
    node_utils.assign_curve(pinnae_contour_float_curve.mapping.curves[0], points)
    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": group_input.outputs["Value2"], 3: 1.0, 4: 3.0},
    )
    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: pinnae_contour_float_curve, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Value": multiply})


def _contour_points(pinnae_contour):
    return [
        (0.0, pinnae_contour[0]),
        (0.2, pinnae_contour[1]),
        (0.4, pinnae_contour[2]),
        (0.55, pinnae_contour[3]),
        (0.7, pinnae_contour[4]),
        (0.8, pinnae_contour[5]),
        (0.9, pinnae_contour[6]),
        (1.0, pinnae_contour[7]),
    ]


def level1_scale(pinnae_contour):
    points = _contour_points(pinnae_contour)
    return _instantiate(
        "level1_scale",
        lambda: nodegroup_pinnae_level1_scale(points),
        lambda g: _set_curve(_float_curves(g)[0], points),
    )


def level1_instance_rotation():
    return _instantiate(
        "level1_instance_rotation",
        nodegroup_pinnae_level1_instance_rotation,
        _no_params,
    )


@node_utils.to_nodegroup(
    "nodegroup_pinnae_level1_rotation", singleton=False, type="GeometryNodeTree"
)
def nodegroup_pinnae_level1_rotation(nw: NodeWrangler, params):
    # Code generated using version 2.4.3 of the node_transpiler

    position = nw.new_node(Nodes.InputPosition)
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketFloat", "Value1", 1.0),
            ("NodeSocketFloat", "Value2", 0.5),
        ],
    )
    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )
    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: (0.0, 0.0, 1.0)},
        attrs={"operation": "MULTIPLY"},
    )
    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Value2"], 1: 0.0}
    )
    pinnae_index = nw.new_node(Nodes.Index, label="PinnaeIndex")
    pinnaelevel1xaxisrotation = nw.new_node(
        level1_xaxis_rotation().name,
        input_kwargs={
            "From Max": add,
            1: pinnae_index,
            2: group_input.outputs["Value1"],
        },
    )
    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={
            "Vector": position,
            "Center": (0, 0, 0),
            "Angle": pinnaelevel1xaxisrotation,
        },
        attrs={"rotation_type": "X_AXIS"},
    )
    pinnaelevel1gravityrotation = nw.new_node(
        level1_gravity_rotation(params["gravity"]).name,
        input_kwargs={"From Max": add, "Value": pinnae_index},
    )
    vector_rotate_1 = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={
            "Vector": vector_rotate,
            "Center": (0, 0, 0),
            "Angle": pinnaelevel1gravityrotation,
        },
        attrs={"rotation_type": "X_AXIS"},
    )
    pinnaelevel1zaxisrotation = nw.new_node(
        level1_zaxis_rotation(params["zaxis"]).name,
        input_kwargs={"From Max": add, "Value": pinnae_index},
    )
    vector_rotate_2 = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={
            "Vector": vector_rotate_1,
            "Center": multiply.outputs["Vector"],
            "Angle": pinnaelevel1zaxisrotation,
        },
        attrs={"rotation_type": "Z_AXIS"},
    )
    pinnaelevel1yaxisrotation = nw.new_node(
        level1_yaxis_rotation(params["yaxis"]).name,
        input_kwargs={"From Max": add, "Value": pinnae_index},
    )
    vector_rotate_3 = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={
            "Vector": vector_rotate_2,
            "Center": multiply.outputs["Vector"],
            "Angle": pinnaelevel1yaxisrotation,
        },
        attrs={"rotation_type": "Y_AXIS"},
    )
    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Vector": vector_rotate_3, "Value": pinnaelevel1xaxisrotation},
    )


def _level1_rotation_params(gravity_rotation=1):
    # the random draws in the order a build makes them: xaxis (none), gravity,
    # zaxis, yaxis
    gravity = _gravity_points(gravity_rotation)
    zaxis = _zaxis_points()
    yaxis = _yaxis_points()
    return {"gravity": gravity, "zaxis": zaxis, "yaxis": yaxis}


def level1_rotation(params):
    def apply(group):
        _repoint(
            group,
            [
                level1_xaxis_rotation(),
                level1_gravity_rotation(params["gravity"]),
                level1_zaxis_rotation(params["zaxis"]),
                level1_yaxis_rotation(params["yaxis"]),
            ],
        )

    return _instantiate(
        "level1_rotation", lambda: nodegroup_pinnae_level1_rotation(params), apply
    )


@node_utils.to_nodegroup(
    "nodegroup_pinnae_level1_instance_position",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_pinnae_level1_instance_position(nw: NodeWrangler, points):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Value1", 1.0),
            ("NodeSocketFloat", "From Max", 1.0),
            ("NodeSocketFloat", "Value2", 1.0),
        ],
    )
    map_range_3 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input.outputs["Value1"],
            2: group_input.outputs["From Max"],
            3: 1.0,
            4: 0.0,
        },
    )
    float_curve_2 = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": map_range_3.outputs["Result"]}
    )
    node_utils.assign_curve(float_curve_2.mapping.curves[0], points)
    accumulate_field_1 = nw.new_node(Nodes.AccumulateField, [float_curve_2])
    # pinnae scale w.r.t fern age
    map_range_5 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": group_input.outputs["Value2"], 3: 0.3, 4: 4.5},
    )
    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: accumulate_field_1.outputs["Trailing"],
            1: map_range_5.outputs["Result"],
        },
        attrs={"operation": "MULTIPLY"},
    )
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})
    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Vector": combine_xyz_1, "Result": map_range_3.outputs["Result"]},
    )


def level1_instance_position(pinnae_contour):
    points = _contour_points(pinnae_contour)
    return _instantiate(
        "level1_instance_position",
        lambda: nodegroup_pinnae_level1_instance_position(points),
        lambda g: _set_curve(_float_curves(g)[0], points),
    )


@node_utils.to_nodegroup(
    "nodegroup_pinnae_level2_rotation", singleton=False, type="GeometryNodeTree"
)
def nodegroup_pinnae_level2_rotation(nw: NodeWrangler, z_points, y_points, x_points):
    # Code generated using version 2.4.3 of the node_transpiler

    position_1 = nw.new_node(Nodes.InputPosition)
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketFloat", "Value1", 1.0),
            ("NodeSocketFloat", "Value2", 0.5),
            ("NodeSocketFloat", "Value3", 0.5),
        ],
    )
    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Value2"], 1: 0.0}
    )
    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Value3"], 1: 0.0}
    )
    map_range_2 = nw.new_node(
        Nodes.MapRange, input_kwargs={"Value": add, "From Max": add_1}
    )
    float_curve_1 = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": map_range_2.outputs["Result"]}
    )
    node_utils.assign_curve(float_curve_1.mapping.curves[0], z_points)
    add_2 = nw.new_node(Nodes.Math, input_kwargs={0: float_curve_1, 1: -0.25})

    # pinna z-axis curvature w.r.t the fern age
    map_range_7 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": group_input.outputs["Value1"], 3: 1.2, 4: 0.0},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_2, 1: map_range_7.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )
    vector_rotate_1 = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": position_1, "Center": (0, 0, 0), "Angle": multiply_1},
        attrs={"rotation_type": "Z_AXIS"},
    )
    map_range = nw.new_node(Nodes.MapRange, input_kwargs={"Value": add, 2: add_1})
    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": map_range.outputs["Result"]}
    )
    node_utils.assign_curve(float_curve.mapping.curves[0], y_points)
    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: float_curve, 1: -0.5})
    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_3, 1: 1.0}, attrs={"operation": "MULTIPLY"}
    )
    vector_rotate = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": vector_rotate_1, "Angle": multiply_2},
        attrs={"rotation_type": "Y_AXIS"},
    )
    map_range_1 = nw.new_node(Nodes.MapRange, input_kwargs={"Value": add, 2: add_1})
    float_curve_2 = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": map_range_1.outputs["Result"]}
    )
    node_utils.assign_curve(float_curve_2.mapping.curves[0], x_points)
    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: float_curve_2, 1: 1.0},
        attrs={"operation": "MULTIPLY"},
    )
    vector_rotate_2 = nw.new_node(
        Nodes.VectorRotate,
        input_kwargs={"Vector": vector_rotate, "Angle": multiply_3},
        attrs={"rotation_type": "X_AXIS"},
    )
    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Vector": vector_rotate_2}
    )


def _rotation_points(values):
    return [
        (0.0, values[0]),
        (0.1, values[1]),
        (0.25, values[2]),
        (0.45, values[3]),
        (0.6, values[4]),
        (1.0, values[5]),
    ]


def level2_rotation(z_axis_rotate, y_axis_rotate, x_axis_rotate):
    z_points = _rotation_points(z_axis_rotate)
    y_points = _rotation_points(y_axis_rotate)
    x_points = _rotation_points(x_axis_rotate)

    def apply(group):
        # float curves in creation order: z, y, x
        curves = _float_curves(group)
        _set_curve(curves[0], z_points)
        _set_curve(curves[1], y_points)
        _set_curve(curves[2], x_points)

    return _instantiate(
        "level2_rotation",
        lambda: nodegroup_pinnae_level2_rotation(z_points, y_points, x_points),
        apply,
    )


@node_utils.to_nodegroup(
    "nodegroup_pinnae_level2_set_point", singleton=False, type="GeometryNodeTree"
)
def nodegroup_pinnae_level2_set_point(nw: NodeWrangler, points):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Value1", 1.0),
            ("NodeSocketFloat", "From Max", 1.0),
            ("NodeSocketFloat", "Value2", 1.0),
        ],
    )
    map_range_4 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            "Value": group_input.outputs["Value1"],
            2: group_input.outputs["From Max"],
            3: 1.0,
            4: 0.0,
        },
    )
    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": map_range_4.outputs["Result"]}
    )
    node_utils.assign_curve(float_curve.mapping.curves[0], points)
    accumulate_field_2 = nw.new_node(Nodes.AccumulateField, [float_curve])
    # pinna scale w.r.t fern age
    map_range_6 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": group_input.outputs["Value2"], 3: 0.5, 4: 2.0},
    )
    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: accumulate_field_2.outputs["Trailing"],
            1: map_range_6.outputs["Result"],
        },
        attrs={"operation": "MULTIPLY"},
    )
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply})
    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={
            "Vector": combine_xyz_2,
            "Value": float_curve,
            "Result": map_range_4.outputs["Result"],
        },
    )


def _pinna_points(pinna_contour):
    return [
        (0.0, pinna_contour[0]),
        (0.38, pinna_contour[1]),
        (0.55, pinna_contour[2]),
        (0.75, pinna_contour[3]),
        (0.9, pinna_contour[4]),
        (1.0, pinna_contour[5]),
    ]


def level2_set_point(pinna_contour):
    points = _pinna_points(pinna_contour)
    return _instantiate(
        "level2_set_point",
        lambda: nodegroup_pinnae_level2_set_point(points),
        lambda g: _set_curve(_float_curves(g)[0], points),
    )


@node_utils.to_nodegroup(
    "nodegroup_pinnae_level2_instance_on_points",
    singleton=False,
    type="GeometryNodeTree",
)
def nodegroup_pinnae_level2_instance_on_points(nw: NodeWrangler, leaf, points):
    # Code generated using version 2.4.3 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Points", None),
            ("NodeSocketFloat", "Value1", 1.0),
            ("NodeSocketFloat", "Value2", 0.5),
            ("NodeSocketFloat", "Value3", 1.0),
        ],
    )
    index = nw.new_node(Nodes.Index)
    object_info_2 = nw.new_node(Nodes.ObjectInfo, input_kwargs={"Object": leaf})
    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": object_info_2.outputs["Geometry"],
            "Scale": (1.2, -1.0, 1.0),
        },
    )
    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": object_info_2.outputs["Geometry"],
            "Scale": (1.2, 1.0, 1.0),
        },
    )
    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform, transform_2]}
    )
    add = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["Value2"], 1: -0.3}
    )
    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": 1.57, "Z": add})
    float_curve_6 = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": group_input.outputs["Value1"]}
    )
    node_utils.assign_curve(float_curve_6.mapping.curves[0], points)
    # pinna leaf size w.r.t the fern age
    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": group_input.outputs["Value3"], 3: 6, 4: 8},
    )
    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: float_curve_6, 1: map_range.outputs["Result"]},
        attrs={"operation": "MULTIPLY"},
    )
    instance_on_points_2 = nw.new_node(
        Nodes.InstanceOnPoints,
        input_kwargs={
            "Points": group_input.outputs["Points"],
            "Selection": index,
            "Instance": join_geometry,
            "Rotation": combine_xyz_3,
            "Scale": multiply.outputs["Vector"],
        },
    )
    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Instances": instance_on_points_2}
    )


def level2_instance_on_points(leaf, pinna_contour):
    points = _pinna_points(pinna_contour)

    def apply(group):
        _set_curve(_float_curves(group)[0], points)
        info = next(n for n in group.nodes if n.bl_idname == Nodes.ObjectInfo)
        if info.inputs["Object"].default_value != leaf:
            info.inputs["Object"].default_value = leaf

    return _instantiate(
        "level2_instance_on_points",
        lambda: nodegroup_pinnae_level2_instance_on_points(leaf, points),
        apply,
    )


def level2_stein():
    return _instantiate("level2_stein", nodegroup_pinnae_level2_stein, _no_params)


def _pinnae_params(leaf_num_param, version_num_param, gravity_rotation):
    """Every random draw of one pinnae group in build order, plus derived values."""
    # Generate Random Pinnae Contour, Two Modes: Linear+Noise, StepwiseLinear+Noise
    mode_random_bit = randint(0, 2, size=(1,))[0]
    if mode_random_bit:
        pinnae_contour = [0, 0.2, 0.6, 1.4, 3.0, 4.0, 5.0, 6.0]
        for i in range(8):
            pinnae_contour[i] = (pinnae_contour[i] + normal(0, 0.04 * i, (1,))[0]) / 6.0
    else:
        pinnae_contour = [0, 0.2, 0.6, 1.4, 3.0, 4.0, 5.0, 4.2]
        for i in range(8):
            pinnae_contour[i] = (pinnae_contour[i] + normal(0, 0.04 * i, (1,))[0]) / 6.0
    bit_left = randint(2, size=(1,))[0]
    bit_right = randint(2, size=(1,))[0]
    versions, rotation = [], None
    for side in ("left", "right"):
        for i in range(version_num_param):
            # Define the Pinna Contour of each Version
            pinna_contour = []
            k = uniform(0.5, 0.58, size=(1,))[0]
            for j in range(6):
                pinna_contour.append(
                    k
                    * np.clip(j * (1.0 + normal(0, 0.1, (1,))[0]) / 5.0 + 0.08, 0, 0.7)
                )
            # Define the Num Leaf of each Version
            leaf_num = leaf_num_param + randint(-1, 2, (1,))[0]
            if side == "left" and i == 0:
                rotation = _level1_rotation_params(gravity_rotation)
            curvature = random_pinnae_level2_curvature()
            versions.append((pinna_contour, leaf_num, curvature))
    return {
        "pinnae_contour": pinnae_contour,
        "bit_left": bit_left,
        "bit_right": bit_right,
        "versions": versions,
        "rotation": rotation,
    }


@node_utils.to_nodegroup("nodegroup_pinnae", singleton=False, type="GeometryNodeTree")
def nodegroup_pinnae(
    nw: NodeWrangler,
    leaf,
    params,
    leaf_num_param,
    age_param,
    pinna_num_param,
    version_num_param,
    all_toggles=False,
):
    # Code generated using version 2.4.3 of the node_transpiler
    # all_toggles builds every version's "1 - modulo" toggle regardless of the
    # random bits, for the template; its copies drop the ones the bits rule out

    # Define Input Node
    leaf_index = nw.new_node(Nodes.Index, label="LeafIndex")
    pinna_index = nw.new_node(Nodes.Index, label="PinnaIndex")
    pinna_num = nw.new_node(Nodes.Integer, label="PinnaNum", attrs={"integer": 10})
    pinna_num.integer = pinna_num_param
    age = nw.new_node(Nodes.Value, label="Age")
    age.outputs[0].default_value = age_param

    mesh_lines_left, selections_left = [], []
    mesh_lines_right, selections_right = [], []

    pinnae_contour = params["pinnae_contour"]

    # Common Components
    pinnaelevel1instanceposition = nw.new_node(
        level1_instance_position(pinnae_contour).name,
        input_kwargs={0: pinna_index, "From Max": pinna_num, 2: age},
    )
    left_noise, right_noise = (
        nw.new_node(Nodes.WhiteNoiseTexture),
        nw.new_node(Nodes.WhiteNoiseTexture),
    )
    pinnaelevel1scale = nw.new_node(
        level1_scale(pinnae_contour).name,
        input_kwargs={0: pinnaelevel1instanceposition.outputs["Result"], 1: age},
    )

    # Left & Right Instance Point Selections for each Version
    random_bit = params["bit_left"] or all_toggles
    for i in range(version_num_param):
        index = nw.new_node(Nodes.Index)
        greater_equal = nw.new_node(
            Nodes.Compare,
            input_kwargs={0: left_noise.outputs["Value"], 1: i / version_num_param},
            attrs={"operation": "GREATER_EQUAL"},
        )
        less_equal = nw.new_node(
            Nodes.Compare,
            input_kwargs={
                0: left_noise.outputs["Value"],
                1: (i + 1) / version_num_param,
            },
            attrs={"operation": "LESS_EQUAL"},
        )
        op_and = nw.new_node(
            Nodes.BooleanMath, input_kwargs={0: greater_equal, 1: less_equal}
        )

        greater_than = nw.new_node(
            Nodes.Math,
            input_kwargs={0: index, 1: 2.0},
            attrs={"operation": "GREATER_THAN"},
        )
        modulo = nw.new_node(
            Nodes.Math, input_kwargs={0: index, 1: 2.0}, attrs={"operation": "MODULO"}
        )
        if random_bit:
            modulo = nw.new_node(
                Nodes.Math,
                input_kwargs={0: 1, 1: modulo},
                attrs={"operation": "SUBTRACT"},
                label=f"ToggleLeft{i}" if all_toggles else None,
            )
        op_and_1 = nw.new_node(
            Nodes.BooleanMath, input_kwargs={0: greater_than, 1: modulo}
        )
        op_and_2 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_and, 1: op_and_1})
        selections_left.append(op_and_2)

    random_bit = params["bit_right"] or all_toggles
    for i in range(version_num_param):
        greater_equal = nw.new_node(
            Nodes.Compare,
            input_kwargs={0: right_noise.outputs["Value"], 1: i / version_num_param},
            attrs={"operation": "GREATER_EQUAL"},
        )
        less_equal = nw.new_node(
            Nodes.Compare,
            input_kwargs={
                0: right_noise.outputs["Value"],
                1: (i + 1) / version_num_param,
            },
            attrs={"operation": "LESS_EQUAL"},
        )
        op_and = nw.new_node(
            Nodes.BooleanMath, input_kwargs={0: greater_equal, 1: less_equal}
        )
        index = nw.new_node(Nodes.Index)
        greater_than = nw.new_node(
            Nodes.Math,
            input_kwargs={0: index, 1: 2.0},
            attrs={"operation": "GREATER_THAN"},
        )
        modulo = nw.new_node(
            Nodes.Math, input_kwargs={0: index, 1: 2.0}, attrs={"operation": "MODULO"}
        )
        if random_bit:
            modulo = nw.new_node(
                Nodes.Math,
                input_kwargs={0: 1, 1: modulo},
                attrs={"operation": "SUBTRACT"},
                label=f"ToggleRight{i}" if all_toggles else None,
            )
        op_and_1 = nw.new_node(
            Nodes.BooleanMath, input_kwargs={0: greater_than, 1: modulo}
        )
        op_and_2 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_and, 1: op_and_1})
        selections_right.append(op_and_2)

    versions = params["versions"]
    # Each Pinna Version
    rotation, pinnaelevel1rotation = True, None
    for i in range(version_num_param):
        pinna_contour, leaf_num, curvature = versions[i]
        # Define the Num Leaf of each Version
        integer_2 = nw.new_node(Nodes.Integer, attrs={"integer": 10})
        integer_2.integer = leaf_num
        mesh_line_pinna = nw.new_node(
            Nodes.MeshLine, input_kwargs={"Count": pinna_num, "Offset": (0.0, 0.0, 0.0)}
        )
        set_position_pinna = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": mesh_line_pinna,
                "Position": pinnaelevel1instanceposition.outputs["Vector"],
            },
        )
        if rotation:
            pinnaelevel1rotation = nw.new_node(
                level1_rotation(params["rotation"]).name,
                input_kwargs={"Geometry": set_position_pinna, 1: age, 2: pinna_num},
            )
            rotation = False
        pinnaelevel1instancerotation = nw.new_node(
            level1_instance_rotation().name,
            input_kwargs={0: pinnaelevel1rotation.outputs["Value"], 1: age},
        )
        set_rotation_pinna = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": set_position_pinna,
                "Position": pinnaelevel1rotation.outputs["Vector"],
            },
        )
        mesh_line_leaf = nw.new_node(
            Nodes.MeshLine, input_kwargs={"Count": integer_2, "Offset": (0.0, 0.0, 0.0)}
        )
        pinnaelevel2setpoint = nw.new_node(
            level2_set_point(pinna_contour).name,
            input_kwargs={0: leaf_index, "From Max": integer_2, 2: age},
        )
        set_position_leaf = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": mesh_line_leaf,
                "Position": pinnaelevel2setpoint.outputs["Vector"],
            },
        )
        x_curvature, y_curvature, z_curvature = curvature
        pinnaelevel2rotation = nw.new_node(
            level2_rotation(z_curvature, y_curvature, x_curvature).name,
            input_kwargs={
                "Geometry": set_position_leaf,
                1: age,
                2: leaf_index,
                3: integer_2,
            },
        )
        set_rotation_leaf = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": set_position_leaf,
                "Position": pinnaelevel2rotation,
            },
        )
        pinna_on_pinnae = nw.new_node(
            Nodes.InstanceOnPoints,
            input_kwargs={
                "Points": set_rotation_pinna,
                "Selection": selections_left[i],
                "Instance": set_rotation_leaf,
                "Rotation": pinnaelevel1instancerotation,
                "Scale": pinnaelevel1scale,
            },
        )
        rotate_instances = nw.new_node(
            Nodes.RotateInstances,
            input_kwargs={
                "Instances": pinna_on_pinnae,
                "Rotation": (-0.1571, 0.0, 0.0),
            },
        )
        scale_instances = nw.new_node(
            Nodes.ScaleInstances,
            input_kwargs={"Instances": rotate_instances, "Scale": (-1.0, 1.0, 1.0)},
        )
        pinnaelevel2stein = nw.new_node(
            level2_stein().name,
            input_kwargs={
                0: pinnaelevel2setpoint.outputs["Result"],
                "Mesh": scale_instances,
            },
        )
        pinnaelevel2instanceonpoints = nw.new_node(
            level2_instance_on_points(leaf, pinna_contour).name,
            input_kwargs={
                "Points": scale_instances,
                1: pinnaelevel2setpoint.outputs["Result"],
                2: 0.0,
                3: age,
            },
        )
        join_geometry = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={
                "Geometry": [pinnaelevel2stein, pinnaelevel2instanceonpoints]
            },
        )
        mesh_lines_left.append(join_geometry)
        if i == version_num_param - 1:
            pinnaelevel1stein = nw.new_node(
                level1_stein().name,
                input_kwargs={
                    "Mesh": set_rotation_pinna,
                    1: age,
                    2: pinnaelevel1instanceposition.outputs["Result"],
                },
            )
            mesh_lines_left.append(pinnaelevel1stein)

    for i in range(version_num_param):
        pinna_contour, leaf_num, curvature = versions[version_num_param + i]
        # Define the Num Leaf of each Version
        integer_2 = nw.new_node(Nodes.Integer, attrs={"integer": 10})
        integer_2.integer = leaf_num
        mesh_line_pinna = nw.new_node(
            Nodes.MeshLine, input_kwargs={"Count": pinna_num, "Offset": (0.0, 0.0, 0.0)}
        )
        set_position_pinna = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": mesh_line_pinna,
                "Position": pinnaelevel1instanceposition.outputs["Vector"],
            },
        )
        pinnaelevel1instancerotation = nw.new_node(
            level1_instance_rotation().name,
            input_kwargs={0: pinnaelevel1rotation.outputs["Value"], 1: age},
        )
        set_rotation_pinna = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": set_position_pinna,
                "Position": pinnaelevel1rotation.outputs["Vector"],
            },
        )
        mesh_line_leaf = nw.new_node(
            Nodes.MeshLine, input_kwargs={"Count": integer_2, "Offset": (0.0, 0.0, 0.0)}
        )
        pinnaelevel2setpoint = nw.new_node(
            level2_set_point(pinna_contour).name,
            input_kwargs={0: leaf_index, "From Max": integer_2, 2: age},
        )
        set_position_leaf = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": mesh_line_leaf,
                "Position": pinnaelevel2setpoint.outputs["Vector"],
            },
        )
        x_curvature, y_curvature, z_curvature = curvature
        pinnaelevel2rotation = nw.new_node(
            level2_rotation(z_curvature, y_curvature, x_curvature).name,
            input_kwargs={
                "Geometry": set_position_leaf,
                1: age,
                2: leaf_index,
                3: integer_2,
            },
        )
        set_rotation_leaf = nw.new_node(
            Nodes.SetPosition,
            input_kwargs={
                "Geometry": set_position_leaf,
                "Position": pinnaelevel2rotation,
            },
        )
        pinna_on_pinnae = nw.new_node(
            Nodes.InstanceOnPoints,
            input_kwargs={
                "Points": set_rotation_pinna,
                "Selection": selections_right[i],
                "Instance": set_rotation_leaf,
                "Scale": pinnaelevel1scale,
                "Rotation": pinnaelevel1instancerotation,
            },
        )
        rotate_instances = nw.new_node(
            Nodes.RotateInstances,
            input_kwargs={
                "Instances": pinna_on_pinnae,
                "Rotation": (-0.1571, 0.0, 0.0),
            },
        )
        scale_instances = nw.new_node(
            Nodes.ScaleInstances,
            input_kwargs={"Instances": rotate_instances, "Scale": (1.0, 1.0, 1.0)},
        )
        pinnaelevel2stein = nw.new_node(
            level2_stein().name,
            input_kwargs={
                0: pinnaelevel2setpoint.outputs["Result"],
                "Mesh": scale_instances,
            },
        )
        pinnaelevel2instanceonpoints = nw.new_node(
            level2_instance_on_points(leaf, pinna_contour).name,
            input_kwargs={
                "Points": scale_instances,
                1: pinnaelevel2setpoint.outputs["Result"],
                2: 0.0,
                3: age,
            },
        )
        join_geometry = nw.new_node(
            Nodes.JoinGeometry,
            input_kwargs={
                "Geometry": [pinnaelevel2stein, pinnaelevel2instanceonpoints]
            },
        )
        mesh_lines_right.append(join_geometry)

    join_geometry_whole = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": mesh_lines_left + mesh_lines_right},
    )
    realize_instances = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": join_geometry_whole}
    )
    noise_texture = nw.new_node(
        Nodes.NoiseTexture, input_kwargs={"Scale": 0.4, "Roughness": 0.2}
    )
    set_positions = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": realize_instances,
            "Offset": noise_texture.outputs["Color"],
        },
    )
    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Geometry": set_positions}
    )


def geometry_pinnae_nodes(
    leaf,
    leaf_num_param=18,
    age_param=0.4,
    pinna_num_param=40,
    version_num_param=4,
    gravity_rotation=1,
):
    params = _pinnae_params(leaf_num_param, version_num_param, gravity_rotation)
    versions = params["versions"]

    def build(all_toggles):
        return nodegroup_pinnae(
            leaf,
            params,
            leaf_num_param,
            age_param,
            pinna_num_param,
            version_num_param,
            all_toggles=all_toggles,
        )

    def drop_toggle(group, name):
        # as built without the toggle: the modulo feeds the boolean AND directly
        toggle = group.nodes[name]
        modulo_socket = toggle.inputs[1].links[0].from_socket
        to_socket = toggle.outputs[0].links[0].to_socket
        group.nodes.remove(toggle)
        group.links.new(modulo_socket, to_socket)

    def apply(group):
        for side, bit in (("Left", params["bit_left"]), ("Right", params["bit_right"])):
            if not bit:
                for i in range(version_num_param):
                    drop_toggle(group, f"Toggle{side}{i}")
        # integer nodes in creation order: PinnaNum, then one per version
        integers = [n for n in group.nodes if n.bl_idname == Nodes.Integer]
        integers[0].integer = pinna_num_param
        for node, (_, leaf_num, _) in zip(integers[1:], versions):
            node.integer = leaf_num
        group.nodes["Age"].outputs[0].default_value = age_param
        # sub-groups in the order the build creates them
        pinnae_contour = params["pinnae_contour"]
        subs = [
            level1_instance_position(pinnae_contour),
            level1_scale(pinnae_contour),
        ]
        for i in range(version_num_param):
            pinna_contour, _, (x_curvature, y_curvature, z_curvature) = versions[i]
            if i == 0:
                subs.append(level1_rotation(params["rotation"]))
            subs += [
                level1_instance_rotation(),
                level2_set_point(pinna_contour),
                level2_rotation(z_curvature, y_curvature, x_curvature),
                level2_stein(),
                level2_instance_on_points(leaf, pinna_contour),
            ]
            if i == version_num_param - 1:
                subs.append(level1_stein())
        for i in range(version_num_param):
            pinna_contour, _, (x_curvature, y_curvature, z_curvature) = versions[
                version_num_param + i
            ]
            subs += [
                level1_instance_rotation(),
                level2_set_point(pinna_contour),
                level2_rotation(z_curvature, y_curvature, x_curvature),
                level2_stein(),
                level2_instance_on_points(leaf, pinna_contour),
            ]
        _repoint(group, subs)

    if _templates is None:
        return build(all_toggles=False)
    key = ("pinnae", version_num_param)
    template = _templates.get(key)
    if template is None:
        # the template has every toggle and is never used itself; every
        # instance, the first included, is a copy with the toggles its bits call for
        template = _templates[key] = build(all_toggles=True)
    group = template.copy()
    apply(group)
    return group


class FernFactory(_DefaultFernFactory):
    """Default `FernFactory` with templated node groups: within one asset, each
    builder above builds its group once, then copies and re-parametrizes it."""

    def create_asset(self, **params):
        with _TemplateScope():
            return _DefaultFernFactory.create_asset.default(self, **params)

    def debug_asset(self, **params):
        with _TemplateScope():
            return _DefaultFernFactory.debug_asset.default(self, **params)

# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alejandro Newell: original Infinigen tree geometry nodes (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/trees/utils/geometrynodes.py)
# - David Yan, Jia Deng: optimized implementation

import bpy
import numpy as np

from infinigen.assets.objects.trees.utils import mesh
from infinigen.assets.objects.trees.utils.geometrynodes import add_node_modifier
from infinigen.assets.objects.trees.utils.materials import new_link
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.perf.core.util.isolation import IsolatedScene

C = bpy.context
D = bpy.data


def sample_points_and_normals(obj, max_density=3, surface_dist=1, max_points=10000):
    # Need to instantiate point distribute
    m = add_node_modifier(obj)
    ng = m.node_group
    inp = ng.nodes.get("Group Input")
    out = ng.nodes.get("Group Output")
    dist = ng.nodes.new(type="GeometryNodeDistributePointsOnFaces")
    pos = ng.nodes.new("GeometryNodeInputPosition")
    scale_factor = ng.nodes.new("ShaderNodeValue")
    mult_normal = ng.nodes.new("ShaderNodeVectorMath")
    add_pos = ng.nodes.new("ShaderNodeVectorMath")
    set_pos = ng.nodes.new("GeometryNodeSetPosition")
    to_vtx = ng.nodes.new("GeometryNodePointsToVertices")

    new_link(ng, inp, "Geometry", dist, "Mesh")
    new_link(ng, dist, "Normal", mult_normal, 0)
    new_link(ng, scale_factor, 0, mult_normal, 1)
    new_link(ng, pos, 0, add_pos, 0)
    new_link(ng, mult_normal, 0, add_pos, 1)
    new_link(ng, dist, "Points", set_pos, "Geometry")
    new_link(ng, add_pos, 0, set_pos, "Position")
    new_link(ng, set_pos, "Geometry", to_vtx, "Points")
    new_link(ng, to_vtx, "Mesh", out, "Geometry")

    mult_normal.operation = "MULTIPLY"
    scale_factor.outputs[0].default_value = surface_dist
    dist.distribute_method = "POISSON"
    dist.inputs.get("Density Max").default_value = max_density

    # Get point coordinates
    IsolatedScene.ensure_evaluable(obj)
    dgraph = C.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(dgraph)
    vtx = mesh.vtx2cds(obj_eval.data.vertices, obj_eval.matrix_world)

    # Get normals
    scale_factor.outputs[0].default_value = 1
    for l in ng.links:
        if l.from_node == pos:
            ng.links.remove(l)

    IsolatedScene.ensure_evaluable(obj)
    dgraph = C.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(dgraph)
    normals = mesh.vtx2cds(obj_eval.data.vertices, np.eye(4))

    obj.modifiers.remove(obj.modifiers[-1])
    D.node_groups.remove(ng)

    idxs = mesh.subsample_vertices(vtx, max_num=max_points)
    return vtx[idxs], normals[idxs]


@node_utils.to_nodegroup("SetTreeRadius", singleton=False, type="GeometryNodeTree")
def set_tree_radius_unmerged(nw):
    """`SetTreeRadius` without its Merge by Distance, so every stem stays its own tube
    for the native tree meshing to retessellate one at a time."""
    # Code generated using version 2.3.1 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketBool", "Selection", True),
            ("NodeSocketFloat", "Reverse depth", 0.5),
            ("NodeSocketFloat", "Scaling", 0.2),
            ("NodeSocketFloat", "Exponent", 1.5),
            ("NodeSocketFloat", "Min radius", 0.02),
            ("NodeSocketFloat", "Max radius", 5.0),
            ("NodeSocketInt", "Profile res", 20),
            ("NodeSocketFloat", "Merge dist", 0.001),
        ],
    )

    mesh_to_curve = nw.new_node(
        Nodes.MeshToCurve,
        input_kwargs={
            "Mesh": group_input.outputs["Geometry"],
            "Selection": group_input.outputs["Selection"],
        },
    )

    set_spline_type = nw.new_node(
        Nodes.CurveSplineType,
        input_kwargs={"Curve": mesh_to_curve},
        attrs={"spline_type": "BEZIER"},
    )

    set_handle_type = nw.new_node(
        Nodes.SetHandleType, input_kwargs={"Curve": set_spline_type}
    )

    position = nw.new_node(Nodes.InputPosition)

    noise_texture = nw.new_node(
        Nodes.NoiseTexture, input_kwargs={"Vector": position, "Scale": 1.0}
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: noise_texture.outputs["Color"], "Scale": 0.02},
        attrs={"operation": "SCALE"},
    )

    set_handle_positions = nw.new_node(
        Nodes.SetHandlePositions,
        input_kwargs={"Curve": set_handle_type, "Offset": scale.outputs["Vector"]},
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={0: True, 1: mesh_to_curve, 2: set_handle_positions},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Reverse depth"],
            1: group_input.outputs["Scaling"],
        },
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply, 1: 0.1}, attrs={"operation": "MULTIPLY"}
    )

    power = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: group_input.outputs["Exponent"]},
        attrs={"operation": "POWER"},
    )

    maximum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: power, 1: group_input.outputs["Min radius"]},
        attrs={"operation": "MAXIMUM"},
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: maximum, 1: group_input.outputs["Max radius"]},
        attrs={"operation": "MINIMUM"},
    )

    set_curve_radius = nw.new_node(
        Nodes.SetCurveRadius,
        input_kwargs={"Curve": switch, "Radius": minimum},
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle,
        input_kwargs={"Resolution": group_input.outputs["Profile res"]},
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": set_curve_radius,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth,
        input_kwargs={"Geometry": curve_to_mesh, "Shade Smooth": False},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Geometry": set_shade_smooth}
    )

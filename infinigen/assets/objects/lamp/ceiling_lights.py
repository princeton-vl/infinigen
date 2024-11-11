# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# -
# - Alexander Raistrick: add point light


import numpy as np
from numpy.random import randint as RI
from numpy.random import uniform as U

from infinigen.assets.lighting.indoor_lights import PointLampFactory
from infinigen.assets.material_assignments import AssetList
from infinigen.assets.utils.autobevel import BevelSharp
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, clip_gaussian


class CeilingLightFactory(AssetFactory):
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
        with FixedSeed(factory_seed):
            self.light_factory = PointLampFactory(factory_seed)
            self.params = self.sample_parameters(dimensions)
            self.material_params, self.scratch, self.edge_wear = (
                self.get_material_params()
            )

        self.params.update(self.material_params)
        self.beveler = BevelSharp(mult=U(1, 3))

    def get_material_params(self):
        material_assignments = AssetList["CeilingLightFactory"]()
        black_material = material_assignments["black_material"].assign_material()
        white_material = material_assignments["white_material"].assign_material()

        wrapped_params = {
            "BlackMaterial": surface.shaderfunc_to_material(black_material),
            "WhiteMaterial": surface.shaderfunc_to_material(white_material),
        }
        scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
        scratch, edge_wear = material_assignments["wear_tear"]

        is_scratch = np.random.uniform() < scratch_prob
        is_edge_wear = np.random.uniform() < edge_wear_prob
        if not is_scratch:
            scratch = None

        if not is_edge_wear:
            edge_wear = None

        return wrapped_params, scratch, edge_wear

    def sample_parameters(self, dimensions, use_default=False):
        if use_default:
            return self.ceiling_light_default_params[
                RI(0, len(self.ceiling_light_default_params))
            ]
        else:
            Radius = clip_gaussian(0.12, 0.04, 0.1, 0.25)
            Thickness = U(0.005, 0.05)
            InnerRadius = Radius * U(0.4, 0.9)
            Height = 0.7 * clip_gaussian(0.09, 0.03, 0.07, 0.15)
            InnerHeight = Height * U(0.5, 1.1)
            Curvature = U(0.1, 0.5)
            params = {
                "Radius": Radius,
                "Thickness": Thickness,
                "InnerRadius": InnerRadius,
                "Height": Height,
                "InnerHeight": InnerHeight,
                "Curvature": Curvature,
            }
            return params

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

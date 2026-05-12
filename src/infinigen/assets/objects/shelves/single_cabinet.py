# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

import bpy
import numpy as np
from numpy.random import normal, randint, uniform

from infinigen.assets.objects.shelves.doors import CabinetDoorBaseFactory
from infinigen.assets.objects.shelves.large_shelf import LargeShelfBaseFactory
from infinigen.assets.utils.object import new_bbox
from infinigen.core import surface, tagging
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


def geometry_cabinet_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler
    right_door_info = nw.new_node(
        Nodes.ObjectInfo, input_kwargs={"Object": kwargs["door"][0]}
    )
    left_door_info = nw.new_node(
        Nodes.ObjectInfo, input_kwargs={"Object": kwargs["door"][1]}
    )
    shelf_info = nw.new_node(Nodes.ObjectInfo, input_kwargs={"Object": kwargs["shelf"]})

    doors = []
    transform_r = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": right_door_info.outputs["Geometry"],
            "Translation": kwargs["door_hinge_pos"][0],
            "Rotation": (0, 0, kwargs["door_open_angle"]),
        },
    )
    doors.append(transform_r)
    if len(kwargs["door_hinge_pos"]) > 1:
        transform_l = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": left_door_info.outputs["Geometry"],
                "Translation": kwargs["door_hinge_pos"][1],
                "Rotation": (0, 0, kwargs["door_open_angle"]),
            },
        )
        doors.append(transform_l)

    attaches = []
    for pos in kwargs["attach_pos"]:
        cube = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (0.0006, 0.0200, 0.04500)}
        )

        combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": -0.0100})

        transform = nw.new_node(
            Nodes.Transform, input_kwargs={"Geometry": cube, "Translation": combine_xyz}
        )

        cube_1 = nw.new_node(
            Nodes.MeshCube, input_kwargs={"Size": (0.0005, 0.0340, 0.0200)}
        )

        join_geometry = nw.new_node(
            Nodes.JoinGeometry, input_kwargs={"Geometry": [transform, cube_1]}
        )

        transform_1 = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": join_geometry,
                "Translation": (0.0000, -0.0170, 0.0000),
            },
        )

        transform_2 = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": transform_1,
                "Rotation": (0.0000, 0.0000, -1.5708),
            },
        )

        transform_3 = nw.new_node(
            Nodes.Transform, input_kwargs={"Geometry": transform_2, "Translation": pos}
        )

        attaches.append(transform_3)

    join_geometry_a = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": attaches}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": doors + [join_geometry_a]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


class SingleCabinetBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(SingleCabinetBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.shelf_params = {}
        self.door_params = {}
        self.mat_params = {}
        self.shelf_fac = LargeShelfBaseFactory(factory_seed)
        self.door_fac = CabinetDoorBaseFactory(factory_seed)
        with FixedSeed(factory_seed):
            self.params = self.sample_params()

    def sample_params(self):
        # Update fac params
        pass

    def get_material_params(self):
        with FixedSeed(self.factory_seed):
            params = self.mat_params.copy()
            if params.get("frame_material", None) is None:
                params["frame_material"] = np.random.choice(
                    ["white", "black_wood", "wood"], p=[0.5, 0.2, 0.3]
                )
            return params

    def get_shelf_params(self, i=0):
        params = self.shelf_params.copy()
        if params.get("shelf_cell_width", None) is None:
            params["shelf_cell_width"] = [
                np.random.choice([0.76, 0.36], p=[0.5, 0.5])
                * np.clip(normal(1.0, 0.1), 0.75, 1.25)
            ]
        if params.get("shelf_cell_height", None) is None:
            num_v_cells = randint(3, 7)
            shelf_cell_height = []
            for i in range(num_v_cells):
                shelf_cell_height.append(0.3 * np.clip(normal(1.0, 0.06), 0.75, 1.25))
            params["shelf_cell_height"] = shelf_cell_height
        if params.get("frame_material", None) is None:
            params["frame_material"] = self.mat_params["frame_material"]

        return params

    def get_door_params(self, i=0):
        params = self.door_params.copy()

        # get door params
        shelf_width = (
            self.shelf_params["shelf_width"]
            + self.shelf_params["side_board_thickness"] * 2
        )
        if params.get("door_width", None) is None:
            if shelf_width < 0.55:
                params["door_width"] = shelf_width
                params["num_door"] = 1
            else:
                params["door_width"] = shelf_width / 2.0 - 0.0005
                params["num_door"] = 2
        if params.get("door_height", None) is None:
            params["door_height"] = (
                self.shelf_params["division_board_z_translation"][-1]
                - self.shelf_params["division_board_z_translation"][0]
                + self.shelf_params["division_board_thickness"]
            )
            if len(
                self.shelf_params["division_board_z_translation"]
            ) > 5 and np.random.choice([True, False], p=[0.5, 0.5]):
                params["door_height"] = (
                    self.shelf_params["division_board_z_translation"][3]
                    - self.shelf_params["division_board_z_translation"][0]
                    + self.shelf_params["division_board_thickness"]
                )
        if params.get("frame_material", None) is None:
            params["frame_material"] = self.mat_params["frame_material"]

        return params

    def get_cabinet_params(self, i=0):
        params = dict()

        shelf_width = (
            self.shelf_params["shelf_width"]
            + self.shelf_params["side_board_thickness"] * 2
        )
        if self.door_params["num_door"] == 1:
            params["door_hinge_pos"] = [
                (
                    self.shelf_params["shelf_depth"] / 2.0 + 0.0025,
                    -shelf_width / 2.0,
                    self.shelf_params["bottom_board_height"],
                )
            ]
            params["door_open_angle"] = 0
            params["attach_pos"] = [
                (
                    self.shelf_params["shelf_depth"] / 2.0,
                    -self.shelf_params["shelf_width"] / 2.0,
                    self.shelf_params["bottom_board_height"] + z,
                )
                for z in self.door_params["attach_height"]
            ]
        elif self.door_params["num_door"] == 2:
            params["door_hinge_pos"] = [
                (
                    self.shelf_params["shelf_depth"] / 2.0 + 0.008,
                    -shelf_width / 2.0,
                    self.shelf_params["bottom_board_height"],
                ),
                (
                    self.shelf_params["shelf_depth"] / 2.0 + 0.008,
                    shelf_width / 2.0,
                    self.shelf_params["bottom_board_height"],
                ),
            ]
            params["door_open_angle"] = 0
            params["attach_pos"] = [
                (
                    self.shelf_params["shelf_depth"] / 2.0,
                    -self.shelf_params["shelf_width"] / 2.0,
                    self.shelf_params["bottom_board_height"] + z,
                )
                for z in self.door_params["attach_height"]
            ] + [
                (
                    self.shelf_params["shelf_depth"] / 2.0,
                    self.shelf_params["shelf_width"] / 2.0,
                    self.shelf_params["bottom_board_height"] + z,
                )
                for z in self.door_params["attach_height"]
            ]
        else:
            raise NotImplementedError

        return params

    def get_cabinet_components(self, i):
        # update material params
        self.mat_params = self.get_material_params()

        # create shelf
        shelf_params = self.get_shelf_params(i=i)
        self.shelf_fac.params = shelf_params
        shelf, shelf_params = self.shelf_fac.create_asset(i=i, ret_params=True)
        shelf.name = "cabinet_frame"
        self.shelf_params = shelf_params

        # create doors
        door_params = self.get_door_params(i=i)
        self.door_fac.params = door_params
        self.door_fac.params["door_left_hinge"] = False
        right_door, door_obj_params = self.door_fac.create_asset(i=i, ret_params=True)
        right_door.name = "cabinet_right_door"
        self.door_fac.params = door_obj_params
        self.door_fac.params["door_left_hinge"] = True
        left_door, _ = self.door_fac.create_asset(i=i, ret_params=True)
        left_door.name = "cabinet_left_door"
        self.door_params = door_obj_params

        return shelf, right_door, left_door

    def create_asset(self, i=0, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=1,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object

        shelf, right_door, left_door = self.get_cabinet_components(i=i)

        # create cabinet
        cabinet_params = self.get_cabinet_params(i=i)
        surface.add_geomod(
            obj,
            geometry_cabinet_nodes,
            attributes=[],
            apply=True,
            input_kwargs={
                "door": [right_door, left_door],
                "shelf": shelf,
                "door_hinge_pos": cabinet_params["door_hinge_pos"],
                "door_open_angle": cabinet_params["door_open_angle"],
                "attach_pos": cabinet_params["attach_pos"],
            },
        )
        butil.delete([left_door, right_door])
        obj = butil.join_objects([shelf, obj])

        tagging.tag_system.relabel_obj(obj)
        return obj


class SingleCabinetFactory(SingleCabinetBaseFactory):
    def sample_params(self):
        params = dict()
        params["Dimensions"] = (
            uniform(0.25, 0.35),
            uniform(0.3, 0.7),
            uniform(0.9, 1.8),
        )

        params["bottom_board_height"] = 0.083
        params["shelf_depth"] = params["Dimensions"][0] - 0.01
        num_h = int((params["Dimensions"][2] - 0.083) / 0.3)
        params["shelf_cell_height"] = [
            (params["Dimensions"][2] - 0.083) / num_h for _ in range(num_h)
        ]
        params["shelf_cell_width"] = [params["Dimensions"][1]]
        self.shelf_params = params
        self.dims = params["Dimensions"]

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        x, y, z = self.dims
        return new_bbox(
            -x / 2 * 1.2, x / 2 * 1.2, -y / 2 * 1.2, y / 2 * 1.2, 0, (z + 0.083) * 1.02
        )

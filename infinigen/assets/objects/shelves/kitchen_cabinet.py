# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.materials.shelf_shaders import (
    shader_shelves_black_wood,
    shader_shelves_black_wood_sampler,
    shader_shelves_white,
    shader_shelves_white_sampler,
    shader_shelves_wood,
    shader_shelves_wood_sampler,
)
from infinigen.assets.objects.shelves.doors import CabinetDoorBaseFactory
from infinigen.assets.objects.shelves.drawers import CabinetDrawerBaseFactory
from infinigen.assets.objects.shelves.large_shelf import LargeShelfBaseFactory
from infinigen.assets.utils.object import new_bbox
from infinigen.core import surface, tagging
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


def geometry_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler
    cabinets = []
    for i, component in enumerate(kwargs["components"]):
        frame_info = nw.new_node(
            Nodes.ObjectInfo, input_kwargs={"Object": component[0]}
        )

        attachments = []
        if component[1] == "door":
            right_door_info = nw.new_node(
                Nodes.ObjectInfo, input_kwargs={"Object": component[2][0]}
            )
            left_door_info = nw.new_node(
                Nodes.ObjectInfo, input_kwargs={"Object": component[2][1]}
            )

            transform_r = nw.new_node(
                Nodes.Transform,
                input_kwargs={
                    "Geometry": right_door_info.outputs["Geometry"],
                    "Translation": component[2][2]["door_hinge_pos"][0],
                    "Rotation": (0, 0, component[2][2]["door_open_angle"]),
                },
            )
            attachments.append(transform_r)
            if len(component[2][2]["door_hinge_pos"]) > 1:
                transform_l = nw.new_node(
                    Nodes.Transform,
                    input_kwargs={
                        "Geometry": left_door_info.outputs["Geometry"],
                        "Translation": component[2][2]["door_hinge_pos"][1],
                        "Rotation": (0, 0, component[2][2]["door_open_angle"]),
                    },
                )
                attachments.append(transform_l)
        elif component[1] == "drawer":
            for j, drawer in enumerate(component[2]):
                drawer_info = nw.new_node(
                    Nodes.ObjectInfo, input_kwargs={"Object": drawer[0]}
                )
                transform = nw.new_node(
                    Nodes.Transform,
                    input_kwargs={
                        "Geometry": drawer_info.outputs["Geometry"],
                        "Translation": drawer[1]["drawer_hinge_pos"],
                    },
                )
                attachments.append(transform)
        else:
            continue

        join_geometry = nw.new_node(
            Nodes.JoinGeometry, input_kwargs={"Geometry": attachments}
        )
        # [frame_info.outputs['Geometry']]})

        transform = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": join_geometry,
                "Translation": (0, kwargs["y_translations"][i], 0),
            },
        )
        cabinets.append(transform)

    try:
        join_geometry_1 = nw.new_node(
            Nodes.JoinGeometry, input_kwargs={"Geometry": cabinets}
        )
    except TypeError:
        import pdb

        pdb.set_trace()
    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1},
        attrs={"is_active_output": True},
    )


class KitchenCabinetBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(KitchenCabinetBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.frame_params = {}
        self.material_params = {}
        self.cabinet_widths = []
        self.frame_fac = LargeShelfBaseFactory(factory_seed)
        self.door_fac = CabinetDoorBaseFactory(factory_seed)
        self.drawer_fac = CabinetDrawerBaseFactory(factory_seed)
        self.drawer_only = False
        with FixedSeed(factory_seed):
            self.params = self.sample_params()

    def sample_params(self):
        pass

    def get_material_params(self):
        with FixedSeed(self.factory_seed):
            params = self.material_params.copy()
            if params.get("frame_material", None) is None:
                with FixedSeed(self.factory_seed):
                    params["frame_material"] = np.random.choice(
                        ["white", "black_wood", "wood"], p=[0.4, 0.3, 0.3]
                    )
            params["board_material"] = params["frame_material"]
            return self.get_material_func(params, randomness=True)

    def get_material_func(self, params, randomness=True):
        with FixedSeed(self.factory_seed):
            white_wood_params = shader_shelves_white_sampler()
            black_wood_params = shader_shelves_black_wood_sampler()
            normal_wood_params = shader_shelves_wood_sampler()
            if params["frame_material"] == "white":
                if randomness:
                    params["frame_material"] = lambda x: shader_shelves_white(
                        x, **white_wood_params
                    )
                else:
                    params["frame_material"] = shader_shelves_white
            elif params["frame_material"] == "black_wood":
                if randomness:
                    params["frame_material"] = lambda x: shader_shelves_black_wood(
                        x, **black_wood_params, z_axis_texture=True
                    )
                else:
                    params["frame_material"] = lambda x: shader_shelves_black_wood(
                        x, z_axis_texture=True
                    )
            elif params["frame_material"] == "wood":
                if randomness:
                    params["frame_material"] = lambda x: shader_shelves_wood(
                        x, **normal_wood_params, z_axis_texture=True
                    )
                else:
                    params["frame_material"] = lambda x: shader_shelves_wood(
                        x, z_axis_texture=True
                    )

            if params["board_material"] == "white":
                if randomness:
                    params["board_material"] = lambda x: shader_shelves_white(
                        x, **white_wood_params
                    )
                else:
                    params["board_material"] = shader_shelves_white
            elif params["board_material"] == "black_wood":
                if randomness:
                    params["board_material"] = lambda x: shader_shelves_black_wood(
                        x, **black_wood_params
                    )
                else:
                    params["board_material"] = shader_shelves_black_wood
            elif params["board_material"] == "wood":
                if randomness:
                    params["board_material"] = lambda x: shader_shelves_wood(
                        x, **normal_wood_params
                    )
                else:
                    params["board_material"] = shader_shelves_wood

            params["panel_meterial"] = params["frame_material"]
            params["knob_material"] = params["frame_material"]
            return params

    def get_frame_params(self, width, i=0):
        params = self.frame_params.copy()
        params["shelf_cell_width"] = [width]
        params.update(self.material_params.copy())
        return params

    def get_attach_params(self, attach_type, i=0):
        param_sets = []
        if attach_type == "none":
            pass
        elif attach_type == "door":
            params = dict()
            shelf_width = (
                self.frame_params["shelf_width"]
                + self.frame_params["side_board_thickness"] * 2
            )
            if shelf_width <= 0.6:
                params["door_width"] = shelf_width
                params["has_mid_ramp"] = False
                params["edge_thickness_1"] = 0.01
                params["door_hinge_pos"] = [
                    (
                        self.frame_params["shelf_depth"] / 2.0 + 0.0025,
                        -shelf_width / 2.0,
                        self.frame_params["bottom_board_height"],
                    )
                ]
                params["door_open_angle"] = 0
            else:
                params["door_width"] = shelf_width / 2.0 - 0.0005
                params["has_mid_ramp"] = False
                params["edge_thickness_1"] = 0.01
                params["door_hinge_pos"] = [
                    (
                        self.frame_params["shelf_depth"] / 2.0 + 0.008,
                        -shelf_width / 2.0,
                        self.frame_params["bottom_board_height"],
                    ),
                    (
                        self.frame_params["shelf_depth"] / 2.0 + 0.008,
                        shelf_width / 2.0,
                        self.frame_params["bottom_board_height"],
                    ),
                ]
                params["door_open_angle"] = 0

            params["door_height"] = (
                self.frame_params["division_board_z_translation"][-1]
                - self.frame_params["division_board_z_translation"][0]
                + self.frame_params["division_board_thickness"]
            )
            params.update(self.material_params.copy())
            param_sets.append(params)
        elif attach_type == "drawer":
            for i, h in enumerate(self.frame_params["shelf_cell_height"]):
                params = dict()
                drawer_h = (
                    self.frame_params["division_board_z_translation"][i + 1]
                    - self.frame_params["division_board_z_translation"][i]
                    - self.frame_params["division_board_thickness"]
                )
                drawer_depth = self.frame_params["shelf_depth"]
                params["drawer_board_width"] = self.frame_params["shelf_width"]
                params["drawer_board_height"] = drawer_h
                params["drawer_depth"] = drawer_depth
                params["drawer_hinge_pos"] = (
                    self.frame_params["shelf_depth"] / 2.0,
                    0,
                    (
                        self.frame_params["division_board_thickness"] / 2.0
                        + self.frame_params["division_board_z_translation"][i]
                    ),
                )
                params.update(self.material_params.copy())
                param_sets.append(params)
        else:
            raise NotImplementedError

        return param_sets

    def get_cabinet_params(self, i=0):
        x_translations = []
        accum_w, thickness = (
            0,
            self.frame_params.get("side_board_thickness", 0.005),
        )  # instructed by Beining
        for w in self.cabinet_widths:
            accum_w += thickness + w / 2.0
            x_translations.append(accum_w)
            accum_w += thickness + w / 2.0 + 0.0005
        return x_translations

    def create_cabinet_components(self, i, drawer_only=False):
        # update material params
        self.material_params = self.get_material_params()

        components = []
        for k, w in enumerate(self.cabinet_widths):
            # create frame
            frame_params = self.get_frame_params(w, i=i)
            self.frame_fac.params = frame_params
            frame, frame_params = self.frame_fac.create_asset(i=i, ret_params=True)
            frame.name = f"cabinet_frame_{k}"
            self.frame_params = frame_params

            # create attach
            if drawer_only:
                attach_type = np.random.choice(["drawer", "door"], p=[0.5, 0.5])
            else:
                attach_type = np.random.choice(
                    ["drawer", "door", "none"], p=[0.4, 0.4, 0.2]
                )

            attach_params = self.get_attach_params(attach_type, i=i)
            if attach_type == "door":
                self.door_fac.params = attach_params[0]
                self.door_fac.params["door_left_hinge"] = False
                right_door, door_obj_params = self.door_fac.create_asset(
                    i=i, ret_params=True
                )
                right_door.name = f"cabinet_right_door_{k}"
                self.door_fac.params = door_obj_params
                self.door_fac.params["door_left_hinge"] = True
                left_door, _ = self.door_fac.create_asset(i=i, ret_params=True)
                left_door.name = f"cabinet_left_door_{k}"
                components.append(
                    [frame, "door", [right_door, left_door, attach_params[0]]]
                )

            elif attach_type == "drawer":
                drawers = []
                for j, p in enumerate(attach_params):
                    self.drawer_fac.params = p
                    drawer = self.drawer_fac.create_asset(i=i)
                    drawer.name = f"drawer_{k}_layer{j}"
                    drawers.append([drawer, p])
                components.append([frame, "drawer", drawers])

            elif attach_type == "none":
                components.append([frame, "none"])

            else:
                raise NotImplementedError

        return components

    def create_asset(self, i=0, **params):
        components = self.create_cabinet_components(i=i, drawer_only=self.drawer_only)
        cabinet_params = self.get_cabinet_params(i=i)
        join_objs = []

        contain_attach = False
        for com in components:
            if com[1] == "none":
                continue
            else:
                contain_attach = True

        if contain_attach:
            bpy.ops.mesh.primitive_plane_add(
                size=1,
                enter_editmode=False,
                align="WORLD",
                location=(0, 0, 0),
                scale=(1, 1, 1),
            )
            obj = bpy.context.active_object
            surface.add_geomod(
                obj,
                geometry_nodes,
                attributes=[],
                input_kwargs={
                    "components": components,
                    "y_translations": cabinet_params,
                },
                apply=True,
            )

            join_objs += [obj]

        for i, c in enumerate(components):
            if c[1] == "door":
                butil.delete(c[2][:-1])
            elif c[1] == "drawer":
                butil.delete([x[0] for x in c[2]])
            c[0].location = (0, cabinet_params[i], 0)
            butil.apply_transform(c[0], loc=True)
            join_objs.append(c[0])

            # butil.delete(c[:1])
        obj = butil.join_objects(join_objs)
        tagging.tag_system.relabel_obj(obj)

        return obj


class KitchenCabinetFactory(KitchenCabinetBaseFactory):
    def __init__(
        self, factory_seed, params={}, coarse=False, dimensions=None, drawer_only=False
    ):
        self.dimensions = dimensions
        super().__init__(factory_seed, params, coarse)
        self.drawer_only = drawer_only

    def sample_params(self):
        params = dict()
        if self.dimensions is None:
            dimensions = (uniform(0.25, 0.35), uniform(1.0, 4.0), uniform(0.5, 1.3))
            self.dimensions = dimensions
        else:
            dimensions = self.dimensions
        params["Dimensions"] = dimensions

        params["bottom_board_height"] = 0.06
        params["shelf_depth"] = params["Dimensions"][0] - 0.01
        num_h = int((params["Dimensions"][2] - 0.06) / 0.3)
        params["shelf_cell_height"] = [
            (params["Dimensions"][2] - 0.06) / num_h for _ in range(num_h)
        ]

        self.frame_params = params

        n_cells = max(int(params["Dimensions"][1] / 0.45), 1)
        intervals = np.random.uniform(0.55, 1.0, size=(n_cells,))
        intervals = intervals / intervals.sum() * params["Dimensions"][1]
        self.cabinet_widths = intervals.tolist()

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        x, y, z = self.dimensions
        return new_bbox(-x / 2 * 1.2, x / 2 * 1.2, 0, y * 1.1, 0, (z + 0.06) * 1.03)

# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei: primary author
# - Yiming Zuo: updates for sim integration
# - Abhishek Joshi: updates for sim integration


import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets import colors
from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.autobevel import BevelSharp
from infinigen.assets.utils.decorate import read_co, write_attribute, write_co
from infinigen.assets.utils.draw import spin
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import (
    data2mesh,
    join_objects,
    mesh2obj,
    new_cube,
    new_line,
)
from infinigen.core import surface
from infinigen.core.constraints.constraint_language.constants import RoomConstants
from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.bevelling import add_bevel, get_bevel_edges
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.paths import blueprint_path_completion
from infinigen.core.util.random import log_uniform, weighted_sample

from .bar_handle import nodegroup_push_bar_handle
from .joint_utils import (
    nodegroup_add_jointed_geometry_metadata,
    nodegroup_arc_on_door_warper,
    nodegroup_door_frame_warper,
    nodegroup_hinge_joint,
    nodegroup_symmetry_along_y,
)


def geometry_node_join(
    nw,
    door_obj,
    handle_obj,
    handle_info_dict,
    door_frame,
    door_frame_style,
    door_arc,
    handle_offset,
    handle_height,
    handle_joint,
    door_width,
    door_height,
    door_depth,
    center_x_offset,
    center_z_offset,
    is_double_door,
    door_orientation,
):
    if (not is_double_door) and door_orientation == "left":
        single_door_flip_lr = True
    else:
        single_door_flip_lr = False

    door_frame = nw.new_node(Nodes.ObjectInfo, input_kwargs={"Object": door_frame})

    door_frame = nw.new_node(
        nodegroup_add_jointed_geometry_metadata().name,
        input_kwargs={
            "Geometry": door_frame.outputs["Geometry"],
            "Label": "door_frame",
        },
    )

    if single_door_flip_lr and door_frame_style == "single_column":
        door_frame = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": door_frame.outputs["Geometry"],
                "Translation": (-door_width, 0.0, 0.0),
            },
        )

    handle_height_value = nw.new_node(Nodes.Value)
    handle_height_value.outputs[0].default_value = handle_height

    # duplicate the handle on both side of the door
    handle_type = handle_info_dict["handle_type"]
    if handle_type == "bar":
        handle_material = handle_info_dict["shader"].generate()
        handle_object_info = nw.new_node(
            nodegroup_push_bar_handle().name,
            input_kwargs={
                "total_length": handle_info_dict["bar_length"],
                "thickness": handle_info_dict["bar_thickness"],
                "bar_aspect_ratio": handle_info_dict["bar_aspect_ratio"],
                "bar_height_ratio": handle_info_dict["bar_height_ratio"],
                "bar_length_ratio": handle_info_dict["bar_length_ratio"],
                "end_length_ratio": handle_info_dict["bar_end_length_ratio"],
                "end_height_ratio": handle_info_dict["bar_end_height_ratio"],
                "overall_x": -door_width * 0.95,
                "overall_y": door_depth / 2,
                "overall_z": handle_info_dict["bar_overall_z_offset"],
            },
        )

        handle_object_info = nw.new_node(
            Nodes.SetMaterial,
            input_kwargs={"Geometry": handle_object_info, "Material": handle_material},
        )

        handle_object_info = nw.new_node(
            nodegroup_add_jointed_geometry_metadata().name,
            input_kwargs={
                "Geometry": handle_object_info.outputs["Geometry"],
                "Label": "handle",
                "Value": 1,
            },
        )

    elif handle_type == "none":
        handle_object_info = None

    else:
        handle_object_info = nw.new_node(
            Nodes.ObjectInfo, input_kwargs={"Object": handle_obj}
        )
        handle_object_info = nw.new_node(
            nodegroup_symmetry_along_y().name,
            input_kwargs={"Geometry": handle_object_info.outputs["Geometry"]},
        )

    # if door_arc is not None:
    #     door_arc_info = nw.new_node(Nodes.ObjectInfo, input_kwargs={'Object': door_arc})
    # else:
    #     door_arc_info = None
    door_info = nw.new_node(Nodes.ObjectInfo, input_kwargs={"Object": door_obj})
    door_arc_info = nw.new_node(Nodes.ObjectInfo, input_kwargs={"Object": door_arc})

    if handle_joint == "hinge":

        def make_single_door(flip_lr=False):
            if flip_lr:
                handle_rotate_axis = (0, -1, 0)
            else:
                handle_rotate_axis = (0, 1, 0)

            bounding_box = nw.new_node(
                Nodes.BoundingBox,
                input_kwargs={"Geometry": door_info.outputs["Geometry"]},
            )

            add = nw.new_node(
                Nodes.VectorMath,
                input_kwargs={
                    0: bounding_box.outputs["Min"],
                    1: bounding_box.outputs["Max"],
                },
            )

            value_1 = nw.new_node(Nodes.Value)
            value_1.outputs[0].default_value = -0.5000

            multiply = nw.new_node(
                Nodes.VectorMath,
                input_kwargs={0: add.outputs["Vector"], 1: value_1},
                attrs={"operation": "MULTIPLY"},
            )

            handle_transformed = nw.new_node(
                Nodes.Transform,
                input_kwargs={
                    "Geometry": handle_object_info.outputs["Geometry"],
                    "Translation": multiply.outputs["Vector"],
                },
            )

            bounding_box_2 = nw.new_node(
                Nodes.BoundingBox,
                input_kwargs={"Geometry": handle_object_info.outputs["Geometry"]},
            )

            add_1 = nw.new_node(
                Nodes.VectorMath,
                input_kwargs={
                    0: bounding_box_2.outputs["Max"],
                    1: bounding_box_2.outputs["Min"],
                },
            )

            value_2 = nw.new_node(Nodes.Value)
            value_2.outputs[0].default_value = -0.5000

            multiply_1 = nw.new_node(
                Nodes.VectorMath,
                input_kwargs={0: add_1.outputs["Vector"], 1: value_2},
                attrs={"operation": "MULTIPLY"},
            )

            separate_xyz = nw.new_node(
                Nodes.SeparateXYZ, input_kwargs={"Vector": multiply_1.outputs["Vector"]}
            )

            separate_xyz_1 = nw.new_node(
                Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Min"]}
            )

            add_2 = nw.new_node(
                Nodes.Math,
                input_kwargs={
                    0: separate_xyz.outputs["X"],
                    1: separate_xyz_1.outputs["X"],
                },
            )

            add_4 = nw.new_node(
                Nodes.Math,
                input_kwargs={0: separate_xyz.outputs["Z"], 1: handle_height_value},
            )

            value_3 = nw.new_node(Nodes.Value)
            value_3.outputs[0].default_value = handle_offset

            add_3 = nw.new_node(Nodes.Math, input_kwargs={0: add_2, 1: value_3})

            door_handle_pos = nw.new_node(
                Nodes.CombineXYZ, input_kwargs={"X": add_3, "Z": add_4}
            )

            if flip_lr:
                handle_transformed = nw.new_node(
                    Nodes.Transform,
                    input_kwargs={
                        "Geometry": handle_transformed.outputs["Geometry"],
                        "Scale": (-1.0000, 1.0000, 1.0000),
                    },
                )

                handle_transformed = nw.new_node(
                    Nodes.FlipFaces, input_kwargs={"Mesh": handle_transformed}
                )

                door_handle_pos = nw.new_node(
                    Nodes.VectorMath,
                    input_kwargs={0: door_handle_pos.outputs["Vector"], 1: (-1, -1, 1)},
                    attrs={"operation": "MULTIPLY"},
                )

            if flip_lr:
                door_body = nw.new_node(
                    Nodes.Transform,
                    input_kwargs={
                        "Geometry": door_info.outputs["Geometry"],
                        "Scale": (-1.0000, 1.0000, 1.0000),
                    },
                )

                door_body = nw.new_node(
                    Nodes.FlipFaces, input_kwargs={"Mesh": door_body}
                )

                door_body = nw.new_node(
                    nodegroup_add_jointed_geometry_metadata().name,
                    input_kwargs={"Geometry": door_body, "Label": "door", "Value": 2},
                )

            else:
                door_body = door_info

                door_body = nw.new_node(
                    nodegroup_add_jointed_geometry_metadata().name,
                    input_kwargs={
                        "Geometry": door_body.outputs["Geometry"],
                        "Label": "door",
                        "Value": 1,
                    },
                )

            door = nw.new_node(
                nodegroup_hinge_joint().name,
                input_kwargs={
                    "Joint Label": "door_handle",
                    "Parent": door_body,
                    "Child": handle_transformed,
                    "Position": door_handle_pos,
                    "Axis": handle_rotate_axis,
                    "Value": 0.0,
                    "Min": 0.0,
                    "Max": 1.0,
                },
            )

            # combine_xyz = nw.new_node(Nodes.CombineXYZ,
            #     input_kwargs={'X': center_x_offset, 'Z': door_frame.outputs["center_z_offset"]})

            if door_arc_info is not None:
                if flip_lr:
                    door_arc = nw.new_node(
                        Nodes.Transform,
                        input_kwargs={
                            "Geometry": door_arc_info.outputs["Geometry"],
                            "Scale": (-1.0000, 1.0000, 1.0000),
                        },
                    )

                    door_arc = nw.new_node(
                        Nodes.FlipFaces, input_kwargs={"Mesh": door_arc}
                    )

                    door_arc = nw.new_node(
                        nodegroup_add_jointed_geometry_metadata().name,
                        input_kwargs={
                            "Geometry": door_arc,
                            "Label": "door",
                            "Value": 2,
                        },
                    )

                else:
                    door_arc = door_arc_info

                    door_arc = nw.new_node(
                        nodegroup_add_jointed_geometry_metadata().name,
                        input_kwargs={
                            "Geometry": door_arc.outputs["Geometry"],
                            "Label": "door",
                            "Value": 1,
                        },
                    )

                door = nw.new_node(
                    Nodes.JoinGeometry,
                    input_kwargs={
                        "Geometry": [
                            door_arc,
                            door.outputs["Geometry"],
                        ]
                    },
                )

            return door

    elif handle_joint == "rigid" or handle_joint == "slide":

        def make_single_door(flip_lr=False):
            # combine_xyz = nw.new_node(Nodes.CombineXYZ,
            #     input_kwargs={'X': door_frame.outputs["center_x_offset"], 'Z': door_frame.outputs["center_z_offset"]})

            if flip_lr:
                door_body = nw.new_node(
                    nodegroup_add_jointed_geometry_metadata().name,
                    input_kwargs={
                        "Geometry": door_info.outputs["Geometry"],
                        "Label": "door",
                        "Value": 2,
                    },
                )
            else:
                door_body = nw.new_node(
                    nodegroup_add_jointed_geometry_metadata().name,
                    input_kwargs={
                        "Geometry": door_info.outputs["Geometry"],
                        "Label": "door",
                        "Value": 1,
                    },
                )

            door = nw.new_node(
                Nodes.JoinGeometry,
                input_kwargs={
                    "Geometry": [
                        door_body.outputs["Geometry"],
                        handle_object_info.outputs["Geometry"],
                    ]
                },
            )

            if door_arc_info is not None:
                if flip_lr:
                    door_arc = nw.new_node(
                        nodegroup_add_jointed_geometry_metadata().name,
                        input_kwargs={
                            "Geometry": door_arc_info.outputs["Geometry"],
                            "Label": "door",
                            "Value": 2,
                        },
                    )
                else:
                    door_arc = nw.new_node(
                        nodegroup_add_jointed_geometry_metadata().name,
                        input_kwargs={
                            "Geometry": door_arc_info.outputs["Geometry"],
                            "Label": "door",
                            "Value": 1,
                        },
                    )

                door = nw.new_node(
                    Nodes.JoinGeometry,
                    input_kwargs={
                        "Geometry": [
                            door_arc.outputs["Geometry"],
                            door.outputs["Geometry"],
                        ]
                    },
                )

            if flip_lr:
                door = nw.new_node(
                    Nodes.Transform,
                    input_kwargs={
                        "Geometry": door.outputs["Geometry"],
                        "Scale": (-1.0000, 1.0000, 1.0000),
                    },
                )

                door = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": door})

            return door

        # door = make_single_door(flip_lr=single_door_flip_lr)
        # final_door = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': door, 'Translation': (center_x_offset, 0.0, center_z_offset)})

    elif handle_joint == "none":

        def make_single_door(flip_lr=False):
            # combine_xyz = nw.new_node(Nodes.CombineXYZ,
            #     input_kwargs={'X': door_frame.outputs["center_x_offset"], 'Z': door_frame.outputs["center_z_offset"]})

            if flip_lr:
                door_body = nw.new_node(
                    nodegroup_add_jointed_geometry_metadata().name,
                    input_kwargs={
                        "Geometry": door_info.outputs["Geometry"],
                        "Label": "door",
                        "Value": 2,
                    },
                )
            else:
                door_body = nw.new_node(
                    nodegroup_add_jointed_geometry_metadata().name,
                    input_kwargs={
                        "Geometry": door_info.outputs["Geometry"],
                        "Label": "door",
                        "Value": 1,
                    },
                )

            # no handle
            door = door_body

            if door_arc_info is not None:
                if flip_lr:
                    door_arc = nw.new_node(
                        nodegroup_add_jointed_geometry_metadata().name,
                        input_kwargs={
                            "Geometry": door_arc_info.outputs["Geometry"],
                            "Label": "door",
                            "Value": 2,
                        },
                    )
                else:
                    door_arc = nw.new_node(
                        nodegroup_add_jointed_geometry_metadata().name,
                        input_kwargs={
                            "Geometry": door_arc_info.outputs["Geometry"],
                            "Label": "door",
                            "Value": 1,
                        },
                    )

                door = nw.new_node(
                    Nodes.JoinGeometry,
                    input_kwargs={
                        "Geometry": [
                            door_arc.outputs["Geometry"],
                            door.outputs["Geometry"],
                        ]
                    },
                )

            if flip_lr:
                door = nw.new_node(
                    Nodes.Transform,
                    input_kwargs={
                        "Geometry": door.outputs["Geometry"],
                        "Scale": (-1.0000, 1.0000, 1.0000),
                    },
                )

                door = nw.new_node(Nodes.FlipFaces, input_kwargs={"Mesh": door})

            return door

    else:
        raise NotImplementedError

    door = make_single_door(flip_lr=single_door_flip_lr)
    if single_door_flip_lr:
        final_door = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": door,
                "Translation": (-center_x_offset, 0.0, center_z_offset),
            },
        )
    else:
        final_door = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": door,
                "Translation": (center_x_offset, 0.0, center_z_offset),
            },
        )

    # # compensate for the extra offset in y axis caused by handle
    bounding_box_1 = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": door_info.outputs["Geometry"]}
    )

    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: bounding_box_1.outputs["Min"],
            1: bounding_box_1.outputs["Max"],
        },
    )

    multiply = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add.outputs["Vector"], 1: (0.5000, 0.5000, 0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": door})

    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Min"], 1: bounding_box.outputs["Max"]},
    )

    multiply_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_1.outputs["Vector"], 1: (0.5000, 0.5000, 0.5000)},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: multiply.outputs["Vector"], 1: multiply_1.outputs["Vector"]},
        attrs={"operation": "SUBTRACT"},
    )

    vector = nw.new_node(Nodes.Vector)
    if single_door_flip_lr:
        vector.vector = (-door_width / 2.0, -door_depth / 2.0, 0.0000)
    else:
        vector.vector = (door_width / 2.0, -door_depth / 2.0, 0.0000)

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": separate_xyz.outputs["Y"]}
    )

    add_2 = nw.new_node(
        Nodes.VectorMath, input_kwargs={0: combine_xyz.outputs["Vector"], 1: vector}
    )

    if single_door_flip_lr:
        axis = (0.0000, 0.0000, -1.0000)
    else:
        axis = (0.0000, 0.0000, 1.0000)

    hinge_joint = nw.new_node(
        nodegroup_hinge_joint().name,
        input_kwargs={
            "Joint Label": "door_hinge",
            "Parent": door_frame.outputs["Geometry"],
            "Child": final_door,
            "Position": add_2,
            "Value": 0.0,
            "Axis": axis,
            "Min": 0.0,
            "Max": 2.0,
        },
    )

    if is_double_door:
        # Add one more door by flipping. This is safe as the flip is perpendicular to join axis
        # door_other = make_single_door((0.0000, -1.0000, 0.0000))

        # door_other = nw.new_node(Nodes.Transform,
        #     input_kwargs={'Geometry': door_other.outputs["Geometry"], 'Translation': (-center_x_offset, 0.0, center_z_offset), 'Scale': (-1.0000, 1.0000, 1.0000)})

        # door_other = nw.new_node(Nodes.FlipFaces, input_kwargs={'Mesh': door_other})

        # door_other = nw.new_node(Nodes.Transform,
        #     input_kwargs={'Geometry': door_other.outputs["Geometry"], 'Translation': (-center_x_offset, 0.0, center_z_offset), 'Scale': (-1.0000, -1.0000, 1.0000)})

        door_other = make_single_door(flip_lr=True)
        door_other = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": door_other,
                "Translation": (-center_x_offset, 0.0, center_z_offset),
            },
        )

        vector_other = nw.new_node(Nodes.Vector)
        vector_other.vector = (-door_width / 2.0, -door_depth / 2.0, 0.0000)

        add_other = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={0: combine_xyz.outputs["Vector"], 1: vector_other},
        )

        # another hinge joint
        hinge_joint = nw.new_node(
            nodegroup_hinge_joint().name,
            input_kwargs={
                "Joint Label": "door_hinge",
                "Parent": hinge_joint,
                "Child": door_other,
                "Position": add_other,
                "Value": 0.0,
                "Axis": (0.0000, 0.0000, -1.0000),  # flipped
                "Min": 0.0,
                "Max": 2.0,
            },
        )

    group_output = nw.new_node(
        Nodes.GroupOutput, input_kwargs={"Geometry": hinge_joint}
    )


class BaseDoorFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, constants=None):
        super(BaseDoorFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            if constants is None:
                constants = RoomConstants()

            self.width = constants.door_width
            self.height = constants.door_size
            self.constants = constants
            self.depth = constants.wall_thickness * log_uniform(0.25, 0.5)
            self.panel_margin = log_uniform(0.08, 0.12)
            self.bevel_width = uniform(0.005, 0.01)
            self.out_bevel = uniform() < 0.7
            self.shrink_width = log_uniform(0.005, 0.06)

            self.surface = weighted_sample(material_assignments.hard_materials)()
            self.has_glass = uniform(0, 1.0) < 0.5
            self.glass_surface = weighted_sample(material_assignments.glasses)()
            self.louver_surface = weighted_sample(material_assignments.hard_materials)()
            self.handle_surface = weighted_sample(material_assignments.metal_neutral)()
            self.has_louver = True

            self.handle_type = np.random.choice(
                ["knob", "lever", "pull", "bar", "none"]
            )

            self.door_frame_style = np.random.choice(
                [
                    "single_column",
                    "full_frame_square",
                    "full_frame_dome",
                    "full_frame_double_door",
                ]
            )
            self.door_frame_width = uniform(0.02, 0.06)

            self.door_orientation = np.random.choice(
                ["left", "right"]
            )  # handle on left/right for push

            self.handle_offset = self.panel_margin * 0.5
            self.handle_height = self.height * uniform(0.45, 0.5)

            self.door_arc_surface = np.random.choice(["door", "glass"])

            if self.handle_type in ["knob", "lever"]:
                self.handle_joint = "hinge"
            elif self.handle_type == "bar":
                self.handle_joint = "slide"
            elif self.handle_type == "pull":
                self.handle_joint = "rigid"
            else:  # none
                self.handle_joint = "none"

            if self.handle_type == "bar":
                self.handle_info_dict = {
                    "handle_type": self.handle_type,
                    "bar_length": uniform(0.7, 0.9) * self.width,
                    "bar_thickness": uniform(0.025, 0.045) * self.height,
                    "bar_aspect_ratio": uniform(0.4, 0.6),
                    "bar_height_ratio": uniform(0.7, 0.9),
                    "bar_length_ratio": uniform(0.5, 0.8),
                    "bar_end_length_ratio": uniform(0.1, 0.15),
                    "bar_end_height_ratio": uniform(1.8, 3.0),
                    "bar_overall_z_offset": -uniform(0.0, 0.1) * self.height,
                    "shader": weighted_sample(material_assignments.metals)(),
                    "color": colors.hsv2rgba(colors.metal_natural_hsv()),
                }

            else:
                self.handle_info_dict = {"handle_type": self.handle_type}

            self.knob_radius = uniform(0.03, 0.04)
            base_radius = uniform(1.1, 1.2)
            mid_radius = uniform(0.4, 0.5)
            self.knob_radius_mid = (
                base_radius,
                base_radius,
                mid_radius,
                mid_radius,
                1,
                uniform(0.6, 0.8),
                0,
            )
            self.knob_depth = uniform(0.08, 0.1)
            self.knob_depth_mid = [
                0,
                uniform(0.1, 0.15),
                uniform(0.25, 0.3),
                uniform(0.35, 0.45),
                uniform(0.6, 0.8),
                1,
                1 + 1e-3,
            ]

            self.lever_radius = uniform(0.03, 0.04)
            self.lever_mid_radius = uniform(0.01, 0.02)
            self.lever_depth = uniform(0.05, 0.08)
            self.lever_mid_depth = uniform(0.15, 0.25)
            self.lever_length = log_uniform(0.15, 0.2)
            self.level_type = np.random.choice(["wave", "cylinder", "bent"])

            self.pull_size = log_uniform(0.1, 0.4)
            self.pull_depth = uniform(0.05, 0.08)
            self.pull_width = log_uniform(0.08, 0.15)
            self.pull_extension = uniform(0.05, 0.15)
            self.to_pull_bevel = uniform() < 0.5
            self.pull_bevel_width = uniform(0.02, 0.04)
            self.pull_radius = uniform(0.01, 0.02)
            self.pull_type = np.random.choice(["u", "tee", "zed"])
            self.is_pull_circular = uniform() < 0.5 or self.pull_type == "zed"
            self.panel_surface = weighted_sample(material_assignments.frame)()
            self.auto_bevel = BevelSharp()
            self.auto_bevel.amount = 0.001
            self.side_bevel = log_uniform(0.005, 0.015)

            self.metal_color_hsv = colors.metal_hsv()

    def create_asset(self, **params) -> bpy.types.Object:
        for _ in range(100):
            obj = self._create_asset(**params)
            if max(obj.dimensions) < 5:
                return obj
            else:
                butil.delete(obj)
        else:
            raise ValueError("Bad door booleaning")

    def _create_asset(self, **params):
        # create handle

        if self.handle_type == "bar" or self.handle_type == "none":
            handle = None

        else:
            handles = []
            match self.handle_type:
                case "knob":
                    handles.extend(self.make_knobs())
                case "lever":
                    handles.extend(self.make_levers())
                case "pull":
                    handles.extend(self.make_pulls())
            for handle in handles:
                self.auto_bevel(handle)

            handle = join_objects(handles)
            self.handle_surface.apply(handle)

            handle.location = -self.width, -self.depth / 2, -self.height / 2
            butil.apply_transform(handle, True)
            handle = add_bevel(handle, get_bevel_edges(handle), offset=self.side_bevel)

        door = new_cube(location=(1, 1, 1))
        butil.apply_transform(door, loc=True)
        door.scale = self.width / 2, self.depth / 2, self.height / 2
        butil.apply_transform(door)
        panels = self.make_panels()
        extras = []
        for panel in panels:
            extras.extend(panel["func"](door, panel))
            butil.select_none()

        # compute the center offset so that door is at the very center of the frame
        if self.door_frame_style == "single_column":
            center_z_offset = 0.0
            center_x_offset = -0.5 * self.door_frame_width
            door_frame_width = self.width  # dummy
            full_frame = False
            top_dome = False
            is_double_door = False

        elif self.door_frame_style == "full_frame_square":
            center_x_offset = 0.5 * self.width
            center_z_offset = 0.0
            door_frame_width = self.width
            full_frame = True
            top_dome = False
            is_double_door = False

        elif self.door_frame_style == "full_frame_dome":
            center_x_offset = 0.5 * self.width
            center_z_offset = -0.25 * self.width
            door_frame_width = self.width
            full_frame = True
            top_dome = True
            is_double_door = False

        elif self.door_frame_style == "full_frame_double_door":
            center_x_offset = self.width
            center_z_offset = 0.0
            door_frame_width = 2 * self.width
            full_frame = True
            top_dome = False
            is_double_door = True

        else:
            raise NotImplementedError

        door = join_objects([door] + extras)
        self.auto_bevel(door)

        # create frame obj
        bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        door_frame = bpy.context.active_object

        surface.add_geomod(
            door_frame,
            nodegroup_door_frame_warper,
            apply=True,
            input_kwargs={
                "full_frame": full_frame,
                "top_dome": top_dome,
                "door_width": door_frame_width,
                "door_height": self.height,
                "door_depth": self.depth,
                "frame_width": self.door_frame_width,
            },
        )

        self.auto_bevel(door_frame)
        door_frame = add_bevel(
            door_frame, get_bevel_edges(door), offset=self.side_bevel
        )

        # arc for dome frame
        if self.door_frame_style == "full_frame_dome":
            bpy.ops.mesh.primitive_plane_add(
                size=2,
                enter_editmode=False,
                align="WORLD",
                location=(0, 0, 0),
                scale=(1, 1, 1),
            )
            door_arc = bpy.context.active_object

            surface.add_geomod(
                door_arc,
                nodegroup_arc_on_door_warper,
                apply=True,
                input_kwargs={
                    "door_width": self.width,
                    "door_depth": self.depth * 0.8,
                },
            )
            self.auto_bevel(door_arc)

            door_arc.location = -self.width / 2, 0.0, self.height / 2
            butil.apply_transform(door_arc, True)
            door_arc = add_bevel(
                door_arc, get_bevel_edges(door_arc), offset=self.side_bevel
            )

        else:
            bpy.ops.mesh.primitive_cube_add(
                size=0.001,
                enter_editmode=False,
                align="WORLD",
                location=(0, 0, 0),
                scale=(1, 1, 1),
            )
            door_arc = bpy.context.active_object
            door_arc.location = -self.width / 2, 0.0, 0.0
            butil.apply_transform(door_arc, True)
            # door_arc = bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
            # door_arc = bpy.context.active_object
            # door_arc.location = -self.width / 2, 0.0, 0.0
            # butil.apply_transform(door_arc, True)

        if self.door_frame_style == "full_frame_dome":
            if self.door_arc_surface == "door":
                self.surface.apply([door_frame, door, door_arc])
            elif self.door_arc_surface == "glass":
                self.surface.apply([door_frame, door])
                self.glass_surface.apply(door_arc, clear=True)
            else:
                breakpoint()
                raise NotImplementedError
        else:
            self.surface.apply([door_frame, door])

        if self.has_glass:
            self.glass_surface.apply(door, selection="glass", clear=True)
        if self.has_louver:
            self.louver_surface.apply(door, selection="louver")

        door.location = -self.width, -self.depth / 2, -self.height / 2
        butil.apply_transform(door, True)
        door = add_bevel(door, get_bevel_edges(door), offset=self.side_bevel)

        # create geometry node and join
        col = butil.group_in_collection(
            [door, handle, door_frame, door_arc], name="door_base_elements", reuse=False
        )

        # col = butil.group_in_collection(
        #     [door, handle], name="door_base_elements", reuse=False
        # )
        col.hide_viewport = True
        col.hide_render = True

        # fake obj
        bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        door_joined = bpy.context.active_object

        surface.add_geomod(
            door_joined,
            geometry_node_join,
            input_kwargs={
                "door_obj": door,
                "handle_obj": handle,
                "handle_info_dict": self.handle_info_dict,
                "door_frame": door_frame,
                "door_frame_style": self.door_frame_style,
                "door_arc": door_arc,
                "handle_offset": self.handle_offset,
                "handle_height": self.handle_height - self.height / 2,
                "handle_joint": self.handle_joint,
                "door_width": self.width,
                "door_height": self.height,
                "door_depth": self.depth,
                "center_x_offset": center_x_offset,
                "center_z_offset": center_z_offset,
                "is_double_door": is_double_door,
                "door_orientation": self.door_orientation,
            },
        )

        return door_joined

    def make_panels(self):
        return []

    def make_knobs(self):
        x_anchors = np.array(self.knob_radius_mid) * self.knob_radius
        y_anchors = np.array(self.knob_depth_mid) * self.knob_depth
        obj = spin([x_anchors, y_anchors, 0], [0, 2, 3], axis=(0, 1, 0))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.edge_face_add()
        return self.make_handles(obj)

    def make_handles(self, obj):
        # making handle two-sided is moved to nodegroup_join
        write_attribute(obj, 1, "handle", "FACE")
        obj.location = self.handle_offset, self.depth, self.handle_height
        butil.apply_transform(obj, loc=True)

        return [
            obj,
        ]

    def make_levers(self):
        x_anchors = (
            self.lever_radius,
            self.lever_radius,
            self.lever_mid_radius,
            self.lever_mid_radius,
            0,
        )
        y_anchors = (
            np.array([0, self.lever_mid_depth, self.lever_mid_depth, 1, 1 + 1e-3])
            * self.lever_depth
        )
        obj = spin([x_anchors, y_anchors, 0], [0, 1, 2, 3], axis=(0, 1, 0))
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.fill()
        lever = new_line(4)
        if self.level_type == "wave":
            co = read_co(lever)
            co[1, -1] = -uniform(0.2, 0.3)
            co[3, -1] = uniform(0.1, 0.15)
            write_co(lever, co)
        elif self.level_type == "bent":
            co = read_co(lever)
            co[4, 1] = -uniform(0.2, 0.3)
            write_co(lever, co)
        lever.scale = [self.lever_length] * 3
        butil.apply_transform(lever)
        butil.select_none()
        with butil.ViewportMode(lever, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={"value": (0, 0, self.lever_mid_radius * 2)}
            )
        butil.modify_mesh(
            lever, "SOLIDIFY", lever, thickness=self.lever_mid_radius, offset=0
        )
        butil.modify_mesh(lever, "SUBSURF", render_levels=1, levels=1)
        lever.location = (
            -self.lever_mid_radius,
            self.lever_depth,
            -self.lever_mid_radius,
        )
        butil.apply_transform(lever, loc=True)
        obj = join_objects([obj, lever])
        return self.make_handles(obj)

    def make_pulls(self):
        if self.pull_type == "u":
            vertices = (
                (0, 0, self.pull_size),
                (0, self.pull_depth, self.pull_size),
                (0, self.pull_depth, 0),
            )
            edges = (0, 1), (1, 2)
        elif self.pull_type == "tee":
            vertices = (
                (0, 0, self.pull_size),
                (0, self.pull_depth, self.pull_size),
                (0, self.pull_depth, 0),
                (0, self.pull_depth, self.pull_size + self.pull_extension),
            )
            edges = (0, 1), (1, 2), (1, 3)
        else:
            vertices = (
                (0, 0, self.pull_size),
                (0, self.pull_depth, self.pull_size),
                (self.pull_width, self.pull_depth, self.pull_size),
                (self.pull_width, self.pull_depth, 0),
            )
            edges = (0, 1), (1, 2), (2, 3)
        obj = mesh2obj(data2mesh(vertices, edges))
        butil.modify_mesh(obj, "MIRROR", use_axis=(False, False, True))
        if self.to_pull_bevel:
            butil.modify_mesh(
                obj, "BEVEL", width=self.pull_bevel_width, segments=4, affect="VERTICES"
            )
        if self.is_pull_circular:
            surface.add_geomod(
                obj,
                geo_radius,
                apply=True,
                input_args=[self.pull_radius, 32],
                input_kwargs={"to_align_tilt": False},
            )
        else:
            with butil.ViewportMode(obj, "EDIT"):
                bpy.ops.mesh.select_mode(type="EDGE")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.extrude_edges_move(
                    TRANSFORM_OT_translate={"value": (self.pull_radius * 2, 0, 0)}
                )
                bpy.ops.mesh.select_all(action="SELECT")
            obj.location = -self.pull_radius, -self.pull_radius, -self.pull_radius
            butil.apply_transform(obj, loc=True)
            with butil.ViewportMode(obj, "EDIT"):
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.normals_make_consistent(inside=False)
            butil.modify_mesh(obj, "SOLIDIFY", thickness=self.pull_radius * 2, offset=0)
        return self.make_handles(obj)

    @property
    def casing_factory(self):
        from infinigen.assets.objects.elements import DoorCasingFactory

        factory = DoorCasingFactory(self.factory_seed, self.coarse, self.constants)
        factory.surface = self.surface
        factory.metal_color = self.metal_color_hsv
        return factory

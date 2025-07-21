# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Hongyu Wen: primary author
# - Abhishek Joshi: updates for sim integration
# Acknowledgment: This file draws inspiration
# from https://www.youtube.com/watch?v=o50FE2W1m8Y
# by Open Class

import functools
import json
import math
import random
import shutil
import string

import numpy as np
from numpy.random import normal, randint, uniform
import gin

from infinigen.assets.composition import material_assignments
from infinigen.assets.materials import metal, plastic
from infinigen.assets.sim_objects.singlefridge import (
    nodegroup_multi_drawer_top,
    nodegroup_singlefridge,
)
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.paths import blueprint_path_completion
from infinigen.core.util.random import weighted_sample


def get_all_metal_shaders(color):
    metal_shaders_list = [
        metal.brushed_metal.shader_brushed_metal,
        metal.galvanized_metal.shader_galvanized_metal,
        metal.grained_and_polished_metal.shader_grained_metal,
        metal.hammered_metal.shader_hammered_metal,
    ]
    new_shaders = [
        functools.partial(shader, base_color=color) for shader in metal_shaders_list
    ]
    for idx, ns in enumerate(new_shaders):
        # fix taken from: https://github.com/elastic/apm-agent-python/issues/293
        ns.__name__ = metal_shaders_list[idx].__name__

    return new_shaders


def sample_gold():
    """Generate a gold color variation"""
    # Gold colors are generally in yellow-orange hue range
    # 36/360 to 56/360 converted to 0-1 scale
    h = np.random.uniform(0.1, 0.155)  # Gold hue range
    s = np.random.uniform(0.65, 0.9)  # Moderate to high saturation
    v = np.random.uniform(0.75, 1.0)  # Bright

    return (h, s, v)


def sample_silver():
    """Generate a silver color variation"""
    # Silver colors are desaturated with high brightness
    h = np.random.uniform(0, 1)  # Hue doesn't matter much due to low saturation
    s = np.random.uniform(0, 0.1)  # Very low saturation
    v = np.random.uniform(0.75, 0.9)  # High but not maximum brightness

    return (h, s, v)


def sample_light_exterior():
    """Generate a light color for the lamp shade exterior"""
    # Light pastel colors - high value, moderate-low saturation
    h = np.random.uniform(0, 1)  # Any hue is possible
    s = np.random.uniform(0.1, 0.3)  # Low-moderate saturation for pastel effect
    v = np.random.uniform(0.8, 0.95)  # High value but not too bright

    return (h, s, v)


class MultifridgeFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)
        self.sim_blueprint = blueprint_path_completion("multifridge.json")
        self.material_params = self.get_material_params()

    def get_material_params(self):
        body_material = weighted_sample(material_assignments.kitchen_appliance_hard)()()
        inner_material = weighted_sample(
            material_assignments.kitchen_appliance_hard
        )()()
        glass_material = weighted_sample(
            material_assignments.appliance_front_maybeglass
        )()()

        r = np.random.rand()
        if r < 0.5:
            body_material = metal.MetalBasic()(color_hsv=sample_light_exterior())

        def sample_handle_mat():
            gold = sample_gold()
            silver = sample_silver()

            shader = weighted_sample(
                [
                    (metal.MetalBasic, 0.7),
                    (plastic.Plastic, 0.3),
                ]
            )()
            r = np.random.rand()
            if r < 1 / 3:
                return shader(color_hsv=gold, color_rgba=hsv2rgba(gold))
            elif r < 2 / 3:
                return shader(color_hsv=silver, color_rgba=hsv2rgba(silver))
            else:
                return shader()

        handle_material = sample_handle_mat()

        transparent_door = uniform(0.0, 1.0) < 0.5
        transparent_shelf = uniform(0.0, 1.0) < 0.5
        params = {
            "BodyMaterial": body_material,
            "HandleMaterial": handle_material,
            "DoorShelfMaterial": inner_material,
            "DoorGlassMaterial": glass_material if transparent_door else body_material,
            "ShelfMaterial": inner_material,
            "ShelfGlassMaterial": glass_material
            if transparent_shelf
            else inner_material,
            "DrawerMaterial": inner_material,
            "DrawerHandleMaterial": inner_material,
        }

        return params

    def sample_heights(self, level_num):
        if level_num == 1:
            return [uniform(1.0, 1.9)]
        elif level_num == 2:
            if uniform() < 1:  # 1 drawer + 1 fridge
                heights = [uniform(1.0, 1.5), uniform(0.3, 0.5)]
                return heights
            else:
                return [uniform(0.6, 1), uniform(0.6, 1)]
        elif level_num == 3:
            if uniform() < 1:  # 2 drawers + 1 fridge
                heights = [uniform(1.0, 1.5), uniform(0.3, 0.5), uniform(0.3, 0.5)]
                return heights
            else:  # 1 drawers + 2 fridges
                heights = [uniform(0.6, 0.9), uniform(0.6, 0.9), uniform(0.2, 0.3)]
                return heights

    def sample_handle_parameters(self, bent_handle, handle_length):
        if bent_handle:
            handle_top_size = (handle_length, uniform(0.05, 0.12), uniform(0.05, 0.12))
            handle_top_thickness = uniform(0.008, 0.015)
            handle_top_roundness = uniform(0.5, 1.0)
            handle_support_size = (
                handle_length * uniform(0.05, 0.1),
                handle_top_size[1] * uniform(0.5, 0.8),
                uniform(0.01, 0.02),
            )
        else:
            handle_top_size = (
                handle_length,
                uniform(0.03, 0.08),
                0,
            )
            handle_top_thickness = handle_top_size[1]
            handle_top_roundness = uniform(0.8, 1.0)
            handle_support_size = (
                handle_length * uniform(0.05, 0.1),
                handle_top_size[1] * uniform(0.5, 0.8),
                uniform(0.05, 0.08),
            )
        handle_support_margin = handle_length * uniform(0.05, 0.1)
        return (
            handle_top_size,
            handle_top_thickness,
            handle_top_roundness,
            handle_support_size,
            handle_support_margin,
        )

    def sample_parameters(self):
        # add code here to randomly sample from parameters
        fullsize = (uniform(0.8, 1.2), uniform(0.8, 1.0), 0)
        level_num = np.random.choice([1, 2, 3], p=[0.65, 0.25, 0.1])

        heights = self.sample_heights(level_num)
        fullsize = (fullsize[0], fullsize[1], sum(heights))

        wall_thickness = max(normal(0.04, 0.01), 0.01)
        door_on_right = uniform(0.0, 1.0) < 0.5
        two_doors = uniform(0.0, 1.0) < 0.4
        door_handle_margin = max(normal(0.1, 0.01), 0.05)

        body_outer_roundness = uniform(0.0, 0.05)
        body_inner_roundness = uniform(0.0, 0.05)
        bent_handle = uniform(0.0, 1.0) < 0.5

        self.params = []
        remain_height = fullsize[2]
        for index in range(level_num):
            size = (fullsize[0], fullsize[1], heights[index])
            is_fridge = size[2] > 0.5

            if is_fridge:
                door_shelf_height = uniform(0.1, 0.2)
                door_shelf_margin = size[2] * uniform(0.2, 0.5)
                door_shelf_num = min(
                    4,
                    math.floor(
                        (size[2] - door_shelf_margin)
                        * uniform(0.85, 0.9)
                        / door_shelf_height
                    ),
                )
                door_shelf_size = (
                    door_shelf_height * uniform(0.5, 1.0),
                    size[1] * uniform(0.85, 0.9),
                    uniform(0.1, 0.15),
                )
                door_shelf_thickness = max(normal(0.01, 0.005), 0.005)

                door_handle_length = size[2] * uniform(0.5, 0.9)
                (
                    door_handle_top_size,
                    door_handle_top_thickness,
                    door_handle_top_roundness,
                    door_handle_support_size,
                    door_handle_support_margin,
                ) = self.sample_handle_parameters(bent_handle, door_handle_length)

                door_margin = max(
                    uniform(0.02, 0.05),
                    (size[1] - door_shelf_size[1]) / 2 + door_shelf_thickness * 2,
                )
                door_l_margin = door_margin
                door_r_margin = door_margin
                door_u_margin = door_margin
                door_b_margin = door_margin
                if door_on_right:
                    door_l_margin += door_handle_margin + 0.5 * door_handle_top_size[1]
                else:
                    door_r_margin += door_handle_margin + 0.5 * door_handle_top_size[1]

                shelf_margin = door_shelf_size[2] + size[0] * uniform(0.1, 0.2)
                shelf_layer_num = max(
                    (size[2] - wall_thickness * 2) / uniform(0.3, 0.5) - 2, 1
                )

                drawer_on_bottom = True
                drawer_height = uniform(0.2, 0.3)
                drawer_num = randint(1, 3)
                drawer_handle_length = (
                    (size[1] - 2 * wall_thickness) / drawer_num * uniform(0.3, 0.9)
                )

                (
                    drawer_handle_top_size,
                    drawer_handle_top_thickness,
                    drawer_handle_top_roundness,
                    drawer_handle_support_size,
                    drawer_handle_support_margin,
                ) = self.sample_handle_parameters(bent_handle, drawer_handle_length)

                params = {
                    # Body Parameters
                    "Size": size,
                    "WallThickness": wall_thickness,
                    "BodyOuterRoundness": body_outer_roundness,
                    "BodyInnerRoundness": body_inner_roundness,
                    # Door Parameters
                    "DoorOnRight": door_on_right,
                    "TwoDoors": two_doors,
                    "DoorHandleMargin": door_handle_margin,
                    "DoorShelfSize": door_shelf_size,
                    "DoorShelfThickness": door_shelf_thickness,
                    "DoorShelfNum": door_shelf_num,
                    "DoorShelfMargin": door_shelf_margin,
                    "DoorHandleTopSize": door_handle_top_size,
                    "DoorHandleTopThickness": door_handle_top_thickness,
                    "DoorHandleTopRoundness": door_handle_top_roundness,
                    "DoorHandleSupportSize": door_handle_support_size,
                    "DoorHandleSupportMargin": door_handle_support_margin,
                    "DoorLMargin": door_l_margin,
                    "DoorRMargin": door_r_margin,
                    "DoorUMargin": door_u_margin,
                    "DoorBMargin": door_b_margin,
                    "DoorHingeJointValue": 2.0,
                    # Shelf Parameters
                    "ShelfMargin": shelf_margin,
                    "ShelfLayerNum": shelf_layer_num,
                    "ShelfThickness": max(normal(0.01, 0.005), 0.005),
                    "ShelfBoardMargin": uniform(0.05, 0.1),
                    "ShelfNetFBNum": randint(5, 20),
                    "ShelfNetLRNum": randint(5, 20),
                    "ShelfNettedShelf": uniform(0.0, 1.0) < 0.5,
                    # Drawer Parameters
                    "DrawerOnBottom": drawer_on_bottom,
                    "DrawerNum": drawer_num,
                    "DrawerHeight": uniform(0.2, 0.4),
                    "DrawerWallThickness": max(normal(0.01, 0.005), 0.005),
                    "DrawerHandleMargin": uniform(0.05, 0.07),
                    "DrawerHandleTopSize": drawer_handle_top_size,
                    "DrawerHandleTopThickness": drawer_handle_top_thickness,
                    "DrawerHandleTopRoundness": drawer_handle_top_roundness,
                    "DrawerHandleSupportSize": drawer_handle_support_size,
                    "DrawerHandleSupportMargin": drawer_handle_length
                    * uniform(0.05, 0.1),
                    "DrawerBodyRoundness": uniform(0.0, 0.1),
                    "DrawerSlideRoundness": uniform(0.0, 0.1),
                    "DrawernnerRoundness": uniform(0.0, 0.1),
                    "DrawerSlidingJointValue": 0.2,
                    "Value": index,
                }
                params.update(self.material_params)
            else:
                handle_length = size[1] * uniform(0.5, 0.9)
                drawer_num = randint(1, 2)
                (
                    handle_top_size,
                    handle_top_thickness,
                    handle_top_roundness,
                    handle_support_size,
                    handle_support_margin,
                ) = self.sample_handle_parameters(bent_handle, handle_length)
                params = {
                    "Size": size,
                    "WallThickness": wall_thickness,
                    "HandleMargin": door_handle_margin,
                    "HandleTopSize": handle_top_size,
                    "HandleTopThickness": handle_top_thickness,
                    "HandleTopRoundness": handle_top_roundness,
                    "HandleSupportSize": handle_support_size,
                    "HandleSupportMargin": handle_support_margin,
                    "SlidingJointValue": 0.2,
                    "BodyRoundness": body_outer_roundness,
                    "SlideRoundness": body_inner_roundness,
                    "InnerRoundness": body_inner_roundness,
                    "InnerMaterial": self.material_params["BodyMaterial"],
                    "OuterMaterial": self.material_params["BodyMaterial"],
                    "DrawerNum": drawer_num,
                    "Value": index,
                }
            self.params.append(params)
        return self.params, level_num

    def create_asset(self, export=True, exporter="mjcf", asset_params=None, **kwargs):
        self.params, level_num = self.sample_parameters()

        random_string = "".join(
            random.choices(string.ascii_letters + string.digits, k=10)
        )
        tmp_buleprint_path = blueprint_path_completion(
            f"multifridge_tmp_{random_string}.json"
        )
        shutil.copy(self.sim_blueprint, tmp_buleprint_path)

        self.sim_blueprint = tmp_buleprint_path

        blueprint_content = json.load(open(tmp_buleprint_path, "r"))
        blueprint_content["graph"]["jointr"]["children"] = {}  # clear all children

        objs = []
        upper_height = 0
        for i in range(level_num):
            obj = butil.spawn_vert()
            if self.params[i]["Size"][2] > 0.5:  # fridge
                node_group = nodegroup_singlefridge
                blueprint_content["graph"]["jointr"]["children"].update(
                    {str(i): "joint0"}
                )
            else:
                node_group = nodegroup_multi_drawer_top
                blueprint_content["graph"]["jointr"]["children"].update(
                    {str(i): "duplicate0"}
                )
                obj.location.x += self.params[i]["WallThickness"] / 2

            butil.modify_mesh(
                obj,
                "NODES",
                apply=False,
                node_group=node_group(),
                ng_inputs=self.params[i],
            )

            upper_height += self.params[i]["Size"][2] / 2
            obj.location.z -= upper_height
            objs.append(obj)
            upper_height += self.params[i]["Size"][2] / 2

        json.dump(blueprint_content, open(tmp_buleprint_path, "w"), indent=4)

        obj = butil.join_objects(objs)

        return obj

    @classmethod
    @gin.configurable(module='MultifridgeFactory')
    def sample_joint_parameters(
        cls,
        door_hinge_stiffness_min: float       = 0.0,
        door_hinge_stiffness_max: float       = 0.0,
        door_hinge_damping_min: float         = 50.0,
        door_hinge_damping_max: float         = 200.0,
        internal_drawer_stiffness_min: float  = 0.0,
        internal_drawer_stiffness_max: float  = 0.0,
        internal_drawer_damping_min: float    = 50.0,
        internal_drawer_damping_max: float    = 200.0,
        freezer_drawer_stiffness_min: float   = 0.0,
        freezer_drawer_stiffness_max: float   = 0.0,
        freezer_drawer_damping_min: float     = 50.0,
        freezer_drawer_damping_max: float     = 200.0,
    ):
        return {
            "door_hinge": {
                "stiffness": uniform(
                    door_hinge_stiffness_min,
                    door_hinge_stiffness_max
                ),
                "damping": uniform(
                    door_hinge_damping_min,
                    door_hinge_damping_max
                ),
            },
            "internal_drawer": {
                "stiffness": uniform(
                    internal_drawer_stiffness_min,
                    internal_drawer_stiffness_max
                ),
                "damping": uniform(
                    internal_drawer_damping_min,
                    internal_drawer_damping_max
                ),
            },
            "freezer_drawer": {
                "stiffness": uniform(
                    freezer_drawer_stiffness_min,
                    freezer_drawer_stiffness_max
                ),
                "damping": uniform(
                    freezer_drawer_damping_min,
                    freezer_drawer_damping_max
                ),
            },
        }


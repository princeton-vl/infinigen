import json
import random
import string
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.paths import blueprint_path_completion

class MultiDoublefridgeFactory(AssetFactory):
    def __init__(self, factory_seed=None, coarse=False):
        super().__init__(factory_seed=factory_seed, coarse=False)
        self.material_params, self.scratch, self.edge_wear = self.get_material_params()

    def get_material_params(self):
        material_assignments = AssetList["SinglefridgeFactory"]()
        body_material = material_assignments["body"].assign_material()
        inner_material = material_assignments["inner"].assign_material()
        glass_material = material_assignments["glass"].assign_material()

        transparent_door = uniform(0.0, 1.0) < 0.5
        transparent_shelf = uniform(0.0, 1.0) < 0.5
        params = {
            "BodyMaterial": body_material,
            "DoorShelfMaterial": inner_material,
            "DoorGlassMaterial": glass_material if transparent_door else body_material,
            "ShelfMaterial": inner_material,
            "DrawerMaterial": inner_material,
            "DrawerHandleMaterial": inner_material,
        }

        wrapped_params = {
            k: surface.shaderfunc_to_material(v) for k, v in params.items()
        }

        scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
        scratch, edge_wear = material_assignments["wear_tear"]

        is_scratch = uniform() < scratch_prob
        is_edge_wear = uniform() < edge_wear_prob
        if not is_scratch:
            scratch = None

        if not is_edge_wear:
            edge_wear = None

        return wrapped_params, scratch, edge_wear

    def sample_heights(self, level_num):
        if level_num == 1:
        elif level_num == 2:
                return heights
            else:
                return [uniform(0.6, 1), uniform(0.6, 1)]
        elif level_num == 3:
                return heights
                heights = [uniform(0.6, 0.9), uniform(0.6, 0.9), uniform(0.2, 0.3)]
                return heights

    def sample_handle_parameters(self, bent_handle, handle_length):
        if bent_handle:
            handle_top_thickness = uniform(0.008, 0.015)
            handle_top_roundness = uniform(0.5, 1.0)
            handle_support_size = (
                handle_length * uniform(0.05, 0.1),
                handle_top_size[1] * uniform(0.5, 0.8),
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
            )
        handle_support_margin = handle_length * uniform(0.05, 0.1)

    def sample_parameters(self):
        # add code here to randomly sample from parameters

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
                door_shelf_margin = size[2] * uniform(0.1, 0.2)
                door_shelf_size = (
                    door_shelf_height,
                    size[1] / 2 * uniform(0.85, 0.9),
                )
                door_shelf_thickness = max(normal(0.01, 0.005), 0.005)

                door_handle_length = size[2] * uniform(0.5, 0.9)
                door_l_margin = door_margin
                door_r_margin = door_margin
                door_u_margin = door_margin
                door_b_margin = door_margin
                if door_on_right:
                    door_l_margin += door_handle_margin + 0.5 * door_handle_top_size[1]
                else:
                    door_r_margin += door_handle_margin + 0.5 * door_handle_top_size[1]
                shelf_margin = door_shelf_size[2] + size[0] * uniform(0.1, 0.2)

                drawer_on_bottom = True
                drawer_height = uniform(0.2, 0.3)
                drawer_num = randint(1, 3)


                params = {
                    # Body Parameters
                    "Size": size,
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

        self.params, level_num = self.sample_parameters()

        shutil.copy(self.sim_blueprint, tmp_buleprint_path)


        objs = []
        upper_height = 0
        for i in range(level_num):
            obj = butil.spawn_vert()
                node_group = nodegroup_doublefridge
            else:
                node_group = nodegroup_multi_drawer_top
                obj.location.x += self.params[i]["WallThickness"] / 2

            butil.modify_mesh(
            )

            upper_height += self.params[i]["Size"][2] / 2
            obj.location.z -= upper_height
            objs.append(obj)
            upper_height += self.params[i]["Size"][2] / 2

        obj = butil.join_objects(objs)



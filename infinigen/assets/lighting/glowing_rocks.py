# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


import bpy
import gin
import numpy as np
from infinigen.core.util import blender as butil

from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.assets.rocks.blender_rock import BlenderRockFactory
from infinigen.core import surface
from infinigen.core.util.color import color_category
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

def shader_glowrock(nw: NodeWrangler, transparent_for_bounce=True):
    object_info = nw.new_node(Nodes.ObjectInfo_Shader)
    white_noise = nw.new_node(Nodes.WhiteNoiseTexture, attrs={"noise_dimensions": "4D"},
                                input_kwargs={"Vector": (object_info, "Random")})
    mix_rgb = nw.new_node(Nodes.MixRGB, [0.6, (white_noise, "Color"), tuple(color_category("gem"))])
    translucent_bsdf = nw.new_node(Nodes.TranslucentBSDF, [mix_rgb])
    transparent_bsdf = nw.new_node(Nodes.TransparentBSDF, [mix_rgb])
    is_camera_ray = nw.new_node(Nodes.LightPath) if transparent_for_bounce else 1
    mix_shader = nw.new_node(Nodes.MixShader, [is_camera_ray, transparent_bsdf, translucent_bsdf])
    nw.new_node(Nodes.MaterialOutput, [mix_shader])

@gin.configurable
class GlowingRocksFactory(AssetFactory):

    def quickly_resample(obj):
        assert obj.type == "EMPTY", obj.type
        obj.rotation_euler[:] = np.random.uniform(-np.pi, np.pi, size=(3,))

    def __init__(self, factory_seed, coarse=False, transparent_for_bounce=True, watt_power_range=(400, 800), **kwargs):
        super().__init__(factory_seed, coarse=coarse)
        if coarse:
            return
        self.watt_power_range = watt_power_range
        self.rock_collection = make_asset_collection(BlenderRockFactory(np.random.randint(1e5), detail=1),
                                                     name="glow_rock_base", n=5)
        
        for o in self.rock_collection.objects:
            butil.modify_mesh(o, 'SUBSURF', levels=2)

        self.material = surface.shaderfunc_to_material(shader_glowrock)

    def create_placeholder(self, i, loc, rot):
        placeholder = butil.spawn_empty('placeholder', disp_type='SPHERE', s=0.1)
        return placeholder
        
    def create_asset(self, *args, **kwargs) -> bpy.types.Object:
        src_obj = np.random.choice(list(self.rock_collection.objects))

        new_obj = src_obj.copy()
        new_obj.data = src_obj.data
        new_obj.animation_data_clear()
        bpy.context.collection.objects.link(new_obj)

        new_obj.rotation_euler = np.random.uniform(-np.pi, np.pi, 3)
        new_obj.scale = np.random.uniform(0.7, 1.5, 3) * 0.5
        new_obj.active_material = self.material
        bbox = np.asarray(new_obj.bound_box[:])  # 8 3
        min_side_length = (bbox.max(axis=0) - bbox.min(axis=0)).min()

        # Diameter is set to half the shortest edge of the bbox
        bpy.ops.object.light_add(type='POINT', radius=min_side_length * 1.0, align='WORLD', location=(0, 0, 0),
                                 rotation=(0, 0, 0), scale=(1, 1, 1))
        point_light = bpy.context.selected_objects[0]
        point_light.data.energy = round(np.random.uniform(*self.watt_power_range))
        point_light.parent = new_obj
        tag_object(new_obj, 'glowing_rocks')
        return new_obj

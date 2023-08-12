# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import gin
import bpy
from infinigen.core.placement.factory import AssetFactory


from infinigen.core.util import blender as butil

from infinigen.assets.fluid.fluid import (
    create_liquid_domain,
    create_liquid_flow,
    create_gas_domain,
    create_gas_flow,
    add_field,
)
from infinigen.assets.fluid.flip_fluid import (
    create_flip_fluid_domain,
    set_flip_fluid_domain,
    create_flip_fluid_inflow,
    set_flip_fluid_obstacle,
    get_objs_inside_domain,
)

from infinigen.core.util.logging import Timer


@gin.configurable
class FluidFactory(AssetFactory):
    # should fix the fluid type problem shouldnt specify here
    def __init__(self, factory_seed, terrain=None, fluid_type=None):
        super(FluidFactory, self).__init__(factory_seed)
        if factory_seed % 2:
            self.fluid_type = "water"
        else:
            self.fluid_type = "water"
        self.factory_seed = factory_seed

        self.terrain = terrain
        self.max_distance = 40
        self.max_expected_radius = 1
        self.fluid_collection = []
        self.called_once = False

    def cull(self, distance: float, vis_distance: float):
        return (
            distance > self.max_distance or vis_distance > self.max_expected_radius * 2
        )

    def finalize_assets(self, assets):
        cache_dirs = dict()

        for i, emp in enumerate(self.fluid_collection):
            dom = None
            flow = None
            for obj in emp.children:
                print(obj, self.fluid_type)
                bpy.context.collection.objects.link(obj)
                if "Fluid" not in obj.modifiers:
                    continue
                if obj.modifiers["Fluid"].fluid_type == "DOMAIN":
                    dom = obj
                else:
                    flow = obj
            assert dom != None and flow != None

            bpy.context.view_layer.objects.active = dom
            print(self.fluid_type)
            bpy.ops.fluid.bake_all()
            bpy.context.collection.objects.unlink(dom)
            bpy.context.collection.objects.unlink(flow)

            mod = dom.modifiers["Fluid"]
            settings = mod.domain_settings
            cache_dir = settings.cache_directory
            cache_dirs[i] = cache_dir
            print("cachedir", cache_dir)
            mod.fluid_type = "NONE"

        for i in range(len(self.fluid_collection)):
            emp = self.fluid_collection[i]
            for obj in emp.children:
                if "Fluid" not in obj.modifiers:
                    continue
                if obj.modifiers["Fluid"].fluid_type == "NONE":
                    dom = obj
                elif obj.modifiers["Fluid"].fluid_type == "FLOW":
                    obj.hide_render = True
            assert dom != None

            cache_dir = cache_dirs[i]
            mod = dom.modifiers["Fluid"]
            mod.fluid_type = "DOMAIN"
            print(dom, mod)
            settings = mod.domain_settings

            # have to change this part
            if self.fluid_type in ["water", "lava"]:
                settings.resolution_max = 200
                settings.domain_type = "LIQUID"
                settings.use_mesh = True
                settings.cache_type = "ALL"
                settings.use_diffusion = True
                settings.cache_frame_end = 100
                if self.fluid_type == "lava":
                    settings.viscosity_exponent = 1
                    settings.surface_tension = 0.250

            elif self.fluid_type in ["fire_and_smoke", "fire", "smoke"]:
                settings.resolution_max = 100
                settings.domain_type = "GAS"
                settings.cache_type = "ALL"
                settings.use_noise = True
                settings.vorticity = 0.1
                settings.use_adaptive_domain = True
                settings.cache_frame_end = 100
                settings.flame_vorticity = 0.1
                settings.burning_rate = 0.5

            settings.cache_directory = cache_dir

        return self.fluid_collection

    def create_asset(self, **params) -> bpy.types.Object:
        if self.called_once:
            emp = butil.spawn_empty("fluid")
            return emp

        obj = None
        dom = None
        if self.fluid_type in ["fire_and_smoke", "fire", "smoke"]:
            print("creating gas flow")
            turbulence = add_field((0, 0, 3))
            obj = create_gas_flow(
                location=(0, 0, 1), fluid_type=self.fluid_type, size=0.5
            )
            dom = create_gas_domain(
                location=(0, 0, 1), fluid_type=self.fluid_type, size=8, resolution=100
            )
        elif self.fluid_type in ["water", "lava"]:
            obj = create_liquid_flow(
                location=(0, 0, 0.2),
                fluid_type=self.fluid_type,
                size=0.2,
                flow_behavior="INFLOW",
            )

            dom = create_liquid_domain(
                location=(0, 0, 0), fluid_type=self.fluid_type, size=22, resolution=800
            )

        emp = butil.spawn_empty("fluid")
        obj.parent = emp
        dom.parent = emp
        if self.fluid_type in ["fire_and_smoke", "fire", "smoke"]:
            turbulence.parent = emp

        self.fluid_collection.append(emp)

        self.called_once = True

        return emp


class FlipFluidFactory(AssetFactory):
    # should fix the fluid type problem shouldnt specify here
    def __init__(self, factory_seed, terrain=None):
        super(FlipFluidFactory, self).__init__(factory_seed)
        self.factory_seed = factory_seed

        self.terrain = terrain
        self.max_distance = 40
        self.max_expected_radius = 1
        self.fluid_collection = []
        self.called_once = False
        self.dom = None

    def cull(self, distance: float, vis_distance: float):
        return (
            distance > self.max_distance or vis_distance > self.max_expected_radius * 2
        )

    def finalize_assets(self, assets):
        return self.fluid_collection

    def create_asset(self, **params) -> bpy.types.Object:
        emp = butil.spawn_empty("fluid")

        self.fluid_collection.append(emp)
        return emp

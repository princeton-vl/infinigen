# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.materials import metal
from infinigen.assets.objects.elements.warehouses.pallet import PalletFactory
from infinigen.assets.utils.decorate import (
    read_co,
    remove_faces,
    solidify,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import (
    join_objects,
    new_base_cylinder,
    new_bbox,
    new_cube,
    new_line,
    new_plane,
)
from infinigen.core import surface
from infinigen.core import tags as t
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import write_attr_data
from infinigen.core.tagging import PREFIX
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed


class RackFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(RackFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.depth = uniform(1, 1.2)
            self.width = uniform(4.0, 5.0)
            self.height = uniform(1.6, 1.8)
            self.steps = np.random.randint(3, 6)
            self.thickness = uniform(0.06, 0.08)
            self.hole_radius = self.thickness / 2 * uniform(0.5, 0.6)
            self.support_angle = uniform(np.pi / 6, np.pi / 4)
            self.is_support_round = uniform() < 0.5
            self.frame_height = self.thickness * uniform(3, 4)
            self.frame_count = np.random.randint(20, 30)

            self.stand_surface = self.support_surface = self.frame_surface = metal
            self.pallet_factory = PalletFactory(self.factory_seed)
            self.margin_range = 0.3, 0.5

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        bbox = new_bbox(
            -self.depth - self.thickness / 2,
            self.thickness / 2,
            -self.thickness / 2,
            self.width + self.thickness / 2,
            0,
            self.height * self.steps,
        )
        objs = [bbox]
        for i in range(self.steps):
            obj = new_plane()
            obj.scale = self.depth / 2, self.width / 2 - self.thickness, 1
            obj.location = -self.depth / 2, self.width / 2, self.height * i
            butil.apply_transform(obj, True)
            write_attr_data(
                obj,
                f"{PREFIX}{t.Subpart.SupportSurface.value}",
                np.ones(1).astype(bool),
                "INT",
                "FACE",
            )
            objs.append(obj)
        obj = join_objects(objs)
        return obj

    def create_asset(self, **params) -> bpy.types.Object:
        stands = self.make_stands()
        supports = self.make_supports()
        frames = self.make_frames()
        obj = join_objects(stands + supports + frames)
        co = read_co(obj)
        co[:, -1] = np.clip(co[:, -1], 0, self.height * self.steps)
        write_co(obj, co)
        pallets = [self.pallet_factory(i) for i in range(self.steps * 2)]
        for i, p in enumerate(pallets):
            p.parent = obj
            margin = uniform(*self.margin_range)
            p.location = (
                margin if i % 2 else self.width - margin - p.dimensions[0],
                (self.depth - p.dimensions[1]) / 2,
                i // 2 * self.height,
            )
        self.pallet_factory.finalize_assets(pallets)
        for p in pallets:
            p.parent = obj
        # obj = join_objects([obj] + pallets)
        obj.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(obj)
        return obj

    def make_stands(self):
        obj = new_cube()
        obj.scale = [self.thickness / 2] * 3
        butil.apply_transform(obj, True)
        cylinder = new_base_cylinder()
        cylinder.scale = self.hole_radius, self.hole_radius, self.thickness * 2
        cylinder.rotation_euler[1] = np.pi / 2
        butil.apply_transform(cylinder)
        butil.modify_mesh(obj, "BOOLEAN", object=cylinder, operation="DIFFERENCE")
        cylinder.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(cylinder)
        butil.modify_mesh(obj, "BOOLEAN", object=cylinder, operation="DIFFERENCE")
        butil.delete(cylinder)
        remove_faces(
            obj,
            lambda x, y, z: (np.abs(x) < self.thickness * 0.49)
            & (np.abs(y) < self.thickness * 0.49)
            & (np.abs(z) < self.thickness * 0.49),
        )
        remove_faces(obj, lambda x, y, z: np.abs(x) + np.abs(y) < self.thickness * 0.1)
        obj.location[-1] = self.thickness / 2
        butil.apply_transform(obj, True)
        butil.modify_mesh(
            obj,
            "ARRAY",
            count=int(np.ceil(self.height / self.thickness * self.steps)),
            relative_offset_displace=(0, 0, 1),
            use_merge_vertices=True,
        )
        write_attribute(obj, 1, "stand", "FACE")
        stands = [obj]
        for locs in [(0, 1), (1, 1), (1, 0)]:
            o = deep_clone_obj(obj)
            o.location = locs[0] * self.width, locs[1] * self.depth, 0
            butil.apply_transform(o, True)
            stands.append(o)
        return stands

    def make_supports(self):
        n = int(
            np.floor(self.height * self.steps / self.depth / np.tan(self.support_angle))
        )
        obj = new_line(n, self.height * self.steps)
        obj.rotation_euler[1] = -np.pi / 2
        butil.apply_transform(obj, True)
        co = read_co(obj)
        co[1::2, 1] = self.depth
        write_co(obj, co)
        if self.is_support_round:
            surface.add_geomod(
                obj, geo_radius, apply=True, input_args=[self.thickness / 2, 16]
            )
        else:
            solidify(obj, 1, self.thickness)
        write_attribute(obj, 1, "support", "FACE")
        o = deep_clone_obj(obj)
        o.location[0] = self.width
        return [obj, o]

    def make_frames(self):
        x_bar = new_cube()
        x_bar.scale = self.width / 2, self.thickness / 2, self.frame_height / 2
        x_bar.location = self.width / 2, 0, self.height - self.frame_height / 2
        butil.apply_transform(x_bar, True)
        x_bar_ = deep_clone_obj(x_bar)
        x_bar_.location[1] = self.depth
        butil.apply_transform(x_bar_, True)
        y_bar = new_cube()
        y_bar.scale = self.thickness / 2, self.depth / 2, self.thickness / 2
        margin = self.width / self.frame_count
        y_bar.location = margin, self.depth / 2, self.height - self.thickness / 2
        butil.apply_transform(y_bar, True)
        butil.modify_mesh(
            y_bar,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            count=self.frame_count - 1,
            constant_offset_displace=(margin, 0, 0),
        )
        frames = [x_bar, x_bar_, y_bar]
        for i in range(1, self.steps - 1):
            for obj in [x_bar, x_bar_, y_bar]:
                o = deep_clone_obj(obj)
                o.location[-1] += self.height * i
                butil.apply_transform(o, True)
                frames.append(o)

        for o in frames:
            write_attribute(o, 1, "frame", "FACE")
        return frames

    def finalize_assets(self, assets):
        self.stand_surface.apply(assets, "stand", metal_color="bw")
        self.support_surface.apply(assets, "support", metal_color="bw")
        self.frame_surface.apply(assets, "frame", metal_color="bw")

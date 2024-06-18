# Copyright (c) Princeton University.

import bpy
import bmesh
import numpy as np
from numpy.random import uniform

from infinigen.assets.utils.decorate import read_co, write_attribute, write_co, read_area, mirror, read_normal
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import data2mesh, join_objects, mesh2obj, new_bbox, new_cube, new_plane
from infinigen.assets.utils.uv import compute_uv_direction, face_corner2faces, unwrap_faces
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import read_attr_data, write_attr_data
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform
from infinigen.core.util import blender as butil


class TVFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(TVFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.aspect_ratio = np.random.choice([9 / 16, 3 / 4])
            self.screen_bevel_width = uniform(0, .01)
            self.side_margin = log_uniform(.005, .01)
            self.bottom_margin = uniform(.005, .03)
            self.depth = uniform(.02, .04)
            self.has_depth_extrude = uniform() < .4
            if self.has_depth_extrude:
                self.depth_extrude = self.depth * uniform(2, 5)
            else:
                self.depth_extrude = self.depth * 1.5
            self.leg_type = np.random.choice(['two-legged', 'single-legged'])  # 'none',
            self.leg_length = uniform(.1, .2)
            self.leg_length_y = uniform(.1, .15)
            self.leg_radius = uniform(.008, .015)
            self.leg_width = uniform(.5, .8)
            self.leg_bevel_width = uniform(.01, .02)

            materials = self.get_material_params()
            self.surface = materials['surface']
            self.scratch = materials['scratch']
            self.edge_wear = materials['edge_wear']
            self.screen_surface = materials['screen_surface']
            self.support_surface = materials['support']


        return {
            'surface': surface, 'scratch': scratch, 'edge_wear': edge_wear, 'screen_surface': screen_surface,
            'support': support
        }

    @property
    def height(self):
        return self.aspect_ratio * self.width

    @property
    def total_width(self):
        return self.width + 2 * self.side_margin

    @property
    def total_height(self):
        return self.height + self.side_margin + self.bottom_margin

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        match self.leg_type:
            case 'two-legged':
                max_x = self.leg_length_y / 2 - (1 - self.leg_width) * self.depth_extrude
            case _:
                max_x = self.leg_length_y / 2 - self.depth_extrude / 2
        return new_bbox(
            - self.depth_extrude - self.depth, max_x, -self.total_width / 2,
            self.total_width / 2, -self.leg_length - self.leg_radius / 2, self.total_height
        )

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.make_base()
        self.make_screen(obj)
        parts = [obj]
        match self.leg_type:
            case 'two-legged':
                legs = self.add_two_legs()
            case _:
                legs = self.add_single_leg()
        for l in legs:
            write_attribute(l, 1, 'leg', 'FACE', 'INT')
        parts.extend(legs)
        obj = join_objects(parts)
        return obj

    def make_screen(self, obj):
        cutter = new_cube()
        cutter.location = 0, -1, 1
        butil.apply_transform(cutter, True)
        cutter.scale = self.width / 2, 1, self.height / 2
        cutter.location = 0, 1e-3, self.bottom_margin
        butil.apply_transform(cutter, True)
        butil.modify_mesh(obj, 'BOOLEAN', object=cutter, operation='DIFFERENCE')
        butil.delete(cutter)
        areas = read_area(obj)
        screen = np.zeros(len(areas), int)
        y = read_normal(obj)[:, 1] < 0
        screen[np.argmax(areas + 1e5 * y)] = 1
        fc2f = face_corner2faces(obj)
        unwrap_faces(obj, screen)
        bbox = compute_uv_direction(obj, 'x', 'z', screen[fc2f])
        write_attr_data(obj, 'screen', screen, domain='FACE', type='INT')
        self.screen_surface.apply(obj, 'screen', bbox)

    def make_base(self):
        obj = new_cube()
        obj.location = 0, 1, 1
        butil.apply_transform(obj, True)
        obj.scale = self.total_width / 2, self.depth / 2, self.total_height / 2
        butil.apply_transform(obj)
        butil.modify_mesh(obj, 'BEVEL', width=self.screen_bevel_width, segments=8)
        if not self.has_depth_extrude:
            return obj
        with butil.ViewportMode(obj, 'EDIT'):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if f.normal[1] > .5]
            bmesh.ops.delete(bm, geom=geom, context='FACES_KEEP_BOUNDARY')
            bmesh.update_edit_mesh(obj.data)
            bpy.ops.mesh.select_mode(type='EDGE')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.region_to_loop()
        height_min, height_max = self.total_height * uniform(.1, .3), self.total_height * uniform(.5, .7)
        width = self.total_width * uniform(.3, .6)
        extra = new_plane()
        extra.scale = width / 2, (height_max - height_min) / 2, 1
        extra.rotation_euler[0] = -np.pi / 2
        extra.location = 0, self.depth_extrude + self.depth, self.total_height / 2
        obj = join_objects([obj, extra])
        with butil.ViewportMode(obj, 'EDIT'):
            bpy.ops.mesh.select_mode(type='EDGE')
            bpy.ops.mesh.bridge_edge_loops(number_cuts=32, profile_shape_factor=-uniform(.0, .4))
        x, y, z = read_co(obj).T
        z += (height_max + height_min - self.total_height) / 2 * np.clip(
            y - self.depth, 0,
            None
        ) / self.depth_extrude
        write_co(obj, np.stack([x, y, z], -1))
        return obj

    def add_two_legs(self):
        vertices = (-self.total_width / 2 * self.leg_width * uniform(0, .6), 0, self.total_height * uniform(.3, .5)), (
            0, 0, -self.leg_length), (
            0, self.leg_length_y / 2, -self.leg_length), (0, -self.leg_length_y / 2, -self.leg_length)
        edges = (0, 1), (1, 2), (1, 3)
        leg = mesh2obj(data2mesh(vertices, edges))
        surface.add_geomod(leg, geo_radius, apply=True, input_args=[self.leg_radius, 16])
        x, y, z = read_co(leg).T
        write_co(leg, np.stack([x, y, np.maximum(z, -self.leg_length - self.leg_radius * uniform(.0, .6))], -1))
        leg_ = deep_clone_obj(leg)
        butil.select_none()
        leg.location = self.total_width / 2 * self.leg_width, (1 - self.leg_width) * self.depth_extrude, 0
        butil.apply_transform(leg, True)
        mirror(leg_)
        leg_.location = -self.total_width / 2 * self.leg_width, (1 - self.leg_width) * self.depth_extrude, 0
        butil.apply_transform(leg_, True)
        return [leg, leg_]

    def add_single_leg(self):
        leg = new_cube()
        leg.location = 0, 1, 1
        butil.apply_transform(leg, True)
        leg.location = 0, self.depth_extrude / 2, -self.leg_length
        leg.scale = [self.total_width * uniform(.05, .1), self.leg_radius,
            (self.leg_length + self.total_height * uniform(.3, .5)) / 2]
        butil.apply_transform(leg, True)
        butil.modify_mesh(leg, 'BEVEL', width=self.leg_bevel_width, segments=8)
        base = new_cube()
        base.location = 0, self.depth_extrude / 2, -self.leg_length
        base.scale = [self.total_width * uniform(.15, .3), self.leg_length_y / 2, self.leg_radius]
        butil.apply_transform(base, True)
        butil.modify_mesh(base, 'BEVEL', width=self.leg_bevel_width, segments=8)
        return [leg, base]

    def finalize_assets(self, assets):
        self.surface.apply(assets, selection='!screen', rough=True, metal_color='bw')
        self.support_surface.apply(assets, selection='leg', rough=True, metal_color='bw')


class MonitorFactory(TVFactory):
    def __init__(self, factory_seed, coarse=False):
        super(MonitorFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.width = log_uniform(.4, .8)
            self.leg_type = 'single-legged'

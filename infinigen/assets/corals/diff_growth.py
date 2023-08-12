# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform

from infinigen.assets.corals.base import BaseCoralFactory
from infinigen.assets.corals.tentacles import make_upward_points_fn, make_radius_points_fn
from infinigen.infinigen_gpl.extras.diff_growth import build_diff_growth
from infinigen.assets.utils.object import mesh2obj, data2mesh
from infinigen.assets.utils.decorate import geo_extension, read_co
from infinigen.assets.utils.mesh import polygon_angles
import infinigen.core.util.blender as butil
from infinigen.core import surface
from infinigen.core.util.math import FixedSeed
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

class DiffGrowthBaseCoralFactory(BaseCoralFactory):
    default_scale = [1] * 3

    def __init__(self, factory_seed, coarse=False):
        super(DiffGrowthBaseCoralFactory, self).__init__(factory_seed, coarse)
        self.makers = [self.leather_make, self.flat_make]
        self.weights = [.7, .3]
        with FixedSeed(self.factory_seed):
            self.maker = np.random.choice(self.makers, p=self.weights)
            if self.maker == self.flat_make:
                self.tentacle_prob = .8
                self.points_fn = make_upward_points_fn(.05, np.pi / 3)
            else:
                self.tentacle_prob = .5
                self.points_fn = make_radius_points_fn(.05, .5)

    @staticmethod
    def diff_growth_make(name, n_colonies=1, **kwargs):
        n_base = 4
        stride = 2
        if n_colonies > 1:
            angles = polygon_angles(np.random.randint(2, 6))
            colony_offsets = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)]).T * stride
        else:
            colony_offsets = np.zeros((1, 3))

        vertices_all, faces_all = [], []
        for i, offset in enumerate(colony_offsets):
            angles = polygon_angles(n_base)
            vertices = np.block(
                [[np.cos(angles), 0], [np.sin(angles), 0], [np.zeros(n_base + 1)]]).T + np.expand_dims(offset,
                                                                                                       0)
            faces = np.stack([np.arange(n_base), np.roll(np.arange(n_base), 1), np.full(n_base, n_base)]).T + (
                    n_base + 1) * i
            vertices_all.append(vertices)
            faces_all.append(faces)
        vertices = np.concatenate(vertices_all)
        faces = np.concatenate(faces_all)
        obj = mesh2obj(data2mesh(vertices, [], faces, 'polygon'))

        boundary = obj.vertex_groups.new(name='Boundary')
        boundary_vertices = set(range(len(vertices)))
        boundary_vertices.difference(range(n_base, len(vertices), n_base + 1))
        boundary.add(list(boundary_vertices), 1.0, 'REPLACE')
        build_diff_growth(obj, boundary.index, **kwargs)
        obj.name = name
        return obj

    @staticmethod
    def leather_make():
        prob_multiple_colonies = .5
        n_colonies = np.random.randint(2, 3) if uniform() < prob_multiple_colonies else 1
        growth_vec = 0, 0, uniform(.8, 1.2)
        growth_scale = 1, 1, uniform(.5, .7)
        obj = DiffGrowthBaseCoralFactory.diff_growth_make('leather_coral', n_colonies,
                                                          max_polygons=1e3 * n_colonies, fac_noise=2., dt=.25,
                                                          growth_scale=growth_scale, growth_vec=growth_vec)
        return obj

    @staticmethod
    def flat_make():
        n_colonies = 1
        obj = DiffGrowthBaseCoralFactory.diff_growth_make('flat_coral', n_colonies,
                                                          max_polygons=4e2 * n_colonies, repulsion_radius=2,
                                                          inhibit_shell=1)
        obj.scale = 1, 1, uniform(1, 2)
        butil.apply_transform(obj)
        return obj

    def create_asset(self, face_size=0.01, **params):
        obj = self.maker()
        butil.modify_mesh(obj, 'SMOOTH', iterations=2)
        levels = 2
        butil.modify_mesh(obj, 'SUBSURF', render_levels=levels, levels=levels)
        obj.scale = 2 * np.array(self.default_scale) / max(obj.dimensions[:2])
        butil.apply_transform(obj)
        surface.add_geomod(obj, geo_extension, apply=True)
        butil.modify_mesh(obj, 'SOLIDIFY', thickness=.01)

        obj.location = 0, 0, -np.amin(read_co(obj).T[:, -1]) * 0.8
        butil.apply_transform(obj, loc=True)
        tag_object(obj, 'diffgrowth_coral')
        return obj


class LeatherBaseCoralFactory(DiffGrowthBaseCoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super(LeatherBaseCoralFactory, self).__init__(factory_seed, coarse)
        self.maker = self.leather_make


class TableBaseCoralFactory(DiffGrowthBaseCoralFactory):
    def __init__(self, factory_seed, coarse=False):
        super(TableBaseCoralFactory, self).__init__(factory_seed, coarse)
        self.maker = self.flat_make

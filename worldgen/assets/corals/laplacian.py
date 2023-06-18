# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
# Date Signed: April 13 2023 

from assets.corals.base import BaseCoralFactory
from assets.corals.tentacles import make_radius_points_fn
from assets.utils.laplacian import build_laplacian_3d
from assets.utils.object import mesh2obj
from assets.utils.decorate import geo_extension
import util.blender as butil
from surfaces import surface
from assets.utils.tag import tag_object, tag_nodegroup

class CauliflowerBaseCoralFactory(BaseCoralFactory):
    tentacle_prob = 0.4
    noise_strength = .015

    def __init__(self, factory_seed, coarse=False):
        super(CauliflowerBaseCoralFactory, self).__init__(factory_seed, coarse)
        self.points_fn = make_radius_points_fn(.05, .6)

    def create_asset(self, face_size=0.01, **params):
        mesh = build_laplacian_3d()
        obj = mesh2obj(mesh)
        surface.add_geomod(obj, geo_extension, apply=True)
        levels = 1
        butil.modify_mesh(obj, 'SUBSURF', levels=levels, render_levels=levels)
        tag_object(obj, 'cauliflower_coral')
        return obj

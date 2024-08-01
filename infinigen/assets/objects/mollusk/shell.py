# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import uniform

import infinigen.core.util.blender as butil
from infinigen.assets.objects.mollusk.base import BaseMolluskFactory
from infinigen.assets.utils.decorate import displace_vertices
from infinigen.assets.utils.draw import shape_by_angles
from infinigen.assets.utils.object import data2mesh, join_objects, mesh2obj, new_circle
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.tagging import tag_object
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class ShellBaseFactory(BaseMolluskFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.makers = [self.scallop_make, self.clam_make, self.mussel_make]
            self.maker = np.random.choice(self.makers)
            self.z_scale = log_uniform(2, 10)

    def build_ellipse(self, viewpoint=(0.0, 0, 1.0), softness=0.3):
        viewpoint = np.array(viewpoint)
        obj = new_circle(vertices=1024)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.fill_grid()
        surface.add_geomod(
            obj, self.geo_shader_vector, apply=True, attributes=["vector"]
        )
        butil.apply_transform(obj, loc=True)

        def displace(x, y, z):
            r = np.sqrt((x - 1) ** 2 + y**2 + z**2)
            t = 1 - softness + softness * r**4
            return (
                (1 - t)[:, np.newaxis]
                * (viewpoint[np.newaxis, :] - np.stack([x, y, z], -1))
            ).T

        displace_vertices(obj, displace)
        return obj

    @staticmethod
    def geo_shader_vector(nw: NodeWrangler):
        geometry = nw.new_node(
            Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
        )
        pos = nw.new_node(Nodes.InputPosition)
        x, y, z = nw.separate(pos)
        vector = nw.combine(x, y, nw.vector_math("DISTANCE", pos, [1, 0, 0]))
        nw.new_node(
            Nodes.GroupOutput, input_kwargs={"Geometry": geometry, "Vector": vector}
        )
        return geometry

    def scallop_make(self):
        obj = self.build_ellipse()
        obj.scale = 1, 1.2, 1
        butil.apply_transform(obj)
        boundary = 0.42
        outer = uniform(0.28, 0.32)
        inner = uniform(0.18, 0.22)
        s = uniform(0.6, 0.7)
        angles = [-boundary, -outer, -inner, inner, outer, boundary]
        scales = [0, s, 1, 1, s, 0]
        shape_by_angles(obj, np.array(angles) * np.pi, scales)
        self.add_radial_groove(obj)
        obj = self.add_hinge(obj)
        tag_object(obj, "scallop")
        return obj

    def clam_make(self):
        obj = self.build_ellipse(softness=0.5)
        obj.scale = 1, 1.2, 1
        butil.apply_transform(obj)
        s = uniform(0.6, 0.7)
        angles = [
            -uniform(0.4, 0.5),
            -uniform(0.3, 0.35),
            uniform(-0.25, 0.25),
            uniform(0.3, 0.35),
            uniform(0.4, 0.5),
        ]
        scales = [0, s, 1, s, 0]
        shape_by_angles(obj, np.array(angles) * np.pi, scales)
        tag_object(obj, "clam")
        return obj

    def mussel_make(self):
        obj = self.build_ellipse(softness=0.5)
        obj.scale = 1, 3, 1
        butil.apply_transform(obj)
        s = uniform(0.6, 0.8)
        angles = [-0.5, -uniform(0.1, 0.15), uniform(0.0, 0.25), 0.5]
        scales = [0, s, 1, uniform(0.6, 0.8)]
        shape_by_angles(obj, np.array(angles) * np.pi, scales)
        tag_object(obj, "mussel")
        return obj

    @staticmethod
    def add_radial_groove(obj):
        frequency = 45
        scale = 0.02

        def displace(x, y, z):
            a = np.arctan(y / (x + 1e-6 * (x >= 0)))
            r = np.sqrt(x * x + y * y + z * z)
            return scale * np.cos(a * frequency) * np.clip(r - 0.25, 0, None)

        displace_vertices(obj, displace)
        return obj

    @staticmethod
    def add_hinge(obj):
        length = 0.4
        width = 0.1
        x = uniform(0.8, 1.0)
        vertices = [
            [0, -length, 0],
            [width, -length * x, 0],
            [width, length * x, 0],
            [0, length, 0],
        ]
        o = mesh2obj(data2mesh(vertices, [], [[0, 1, 2, 3]], "trap"))
        butil.modify_mesh(
            o, "SUBSURF", render_levels=2, levels=2, subdivision_type="SIMPLE"
        )
        butil.modify_mesh(
            o,
            "DISPLACE",
            strength=0.2,
            texture=bpy.data.textures.new(name="hinge", type="STUCCI"),
        )
        obj = join_objects([obj, o])
        return obj

    def create_asset(self, **params):
        upper = self.maker()
        dim = np.sqrt(upper.dimensions[0] * upper.dimensions[1] + 0.01)
        upper.scale = [1 / dim] * 3
        upper.location[-1] += 0.005
        butil.apply_transform(upper, loc=True)
        lower = butil.deep_clone_obj(upper)
        lower.scale[-1] = -1
        butil.apply_transform(lower)

        base = uniform(0, np.pi / 4)
        lower.rotation_euler[1] = -base
        upper.rotation_euler[1] = -base - uniform(np.pi / 6, np.pi / 3)
        obj = join_objects([lower, upper])
        return obj


class ScallopBaseFactory(ShellBaseFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.maker = self.scallop_make


class ClamBaseFactory(ShellBaseFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.maker = self.clam_make


class MusselBaseFactory(ShellBaseFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.maker = self.mussel_make

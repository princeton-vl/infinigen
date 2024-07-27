# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.deformed_trees.base import BaseDeformedTreeFactory
from infinigen.assets.utils.decorate import (
    read_material_index,
    remove_vertices,
    write_material_index,
)
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import join_objects, new_icosphere, separate_loose
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.random import log_uniform


class RottenTreeFactory(BaseDeformedTreeFactory):
    @staticmethod
    def geo_cutter(nw: NodeWrangler, strength, scale, metric_fn):
        geometry = nw.new_node(
            Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
        )
        x, y, z = nw.separate(nw.new_node(Nodes.InputPosition))
        selection = nw.compare(
            "LESS_THAN", nw.scalar_add(nw.power(x, 2), nw.power(y, 2)), 1
        )
        offset = nw.scalar_multiply(
            nw.new_node(
                Nodes.Clamp,
                [
                    nw.new_node(
                        Nodes.NoiseTexture,
                        input_kwargs={
                            "Vector": nw.new_node(Nodes.InputPosition),
                            "Scale": scale,
                        },
                        attrs={"noise_dimensions": "2D"},
                    ),
                    0.3,
                    0.7,
                ],
            ),
            strength,
        )
        anchors = (0, 1), (1.02, 1), (1.05, 0), (2, 0)
        metric = surface.eval_argument(nw, metric_fn)
        offset = nw.scalar_multiply(offset, nw.build_float_curve(metric, anchors))
        offset = nw.scalar_multiply(
            offset,
            nw.switch(
                nw.compare(
                    "GREATER_THAN", nw.separate(nw.new_node(Nodes.InputNormal))[-1], 0
                ),
                1,
                -1,
            ),
        )
        geometry = nw.new_node(
            Nodes.SetPosition, [geometry, selection, None, nw.combine(0, 0, offset)]
        )
        nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})

    def build_cutter(self, radius, height):
        cutter = new_icosphere(subdivisions=6)
        angle = uniform(-np.pi, 0)
        depth = radius * uniform(0.4, 0.9)
        cutter_scale = np.array(
            [
                radius * uniform(0.8, 1.2),
                radius * uniform(0.8, 1.2),
                log_uniform(1.0, 1.2),
            ]
        )
        cutter_location = np.array(
            [depth * np.cos(angle), depth * np.sin(angle), height]
        )
        cutter.scale = cutter_scale
        cutter.location = cutter_location
        assign_material(cutter, self.material)

        def metric(x, y, z):
            return np.linalg.norm(
                (np.stack([x, y, z], -1) - cutter_location[np.newaxis, :])
                / cutter_scale[np.newaxis, :],
                axis=-1,
            )

        def fn(x, y, z):
            return metric(x, y, z) < 1 + 0.0001

        def inverse_fn(x, y, z):
            return metric(x, y, z) > 1 + 0.0001

        def metric_fn(nw):
            return nw.vector_math(
                "LENGTH",
                nw.divide(
                    nw.sub(nw.new_node(Nodes.InputPosition), cutter_location),
                    cutter_scale,
                ),
            )

        return cutter, fn, inverse_fn, metric_fn

    def create_asset(self, i, distance=0, **params):
        outer = self.build_tree(i, distance, **params)
        radius = max(
            [
                np.sqrt(v.co[0] ** 2 + v.co[1] ** 2)
                for v in outer.data.vertices
                if v.co[-1] < 0.1
            ]
        )
        height = uniform(0.8, 1.6)
        cutter, fn, inverse_fn, metric_fn = self.build_cutter(radius, height)
        butil.modify_mesh(outer, "BOOLEAN", object=cutter, operation="DIFFERENCE")
        outer = separate_loose(outer)
        inner = deep_clone_obj(outer)
        remove_vertices(outer, fn)
        remove_vertices(inner, inverse_fn)
        self.trunk_surface.apply(outer)
        butil.apply_modifiers(outer)

        obj = join_objects([outer, inner])
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.bridge_edge_loops(number_cuts=10, interpolation="LINEAR")

        ring_material_index = list(obj.data.materials).index(
            obj.data.materials["shader_rings"]
        )
        material_indices = read_material_index(obj)
        null_indices = np.array(
            [i for i, m in enumerate(obj.data.materials) if not hasattr(m, "name")]
        )
        material_indices[
            np.any(material_indices[:, np.newaxis] == null_indices[np.newaxis, :], -1)
        ] = ring_material_index
        write_material_index(obj, material_indices)

        noise_strength = cutter.scale[-1] * uniform(0.5, 0.8)
        noise_scale = uniform(10, 15)
        surface.add_geomod(
            obj,
            self.geo_cutter,
            apply=True,
            input_args=[noise_strength, noise_scale, metric_fn],
        )
        surface.add_geomod(obj, self.geo_xyz, apply=True)
        butil.delete(cutter)
        tag_object(outer, "rotten_tree")
        return outer

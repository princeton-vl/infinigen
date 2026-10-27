# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen mushroom (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/mushroom/cap.py; https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/mushroom/generate.py; https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/mushroom/growth.py; https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/mushroom/stem.py)
# - David Yan, Jia Deng: optimized implementation

"""Mushroom factories with per-mushroom collision proxies and direct profile meshing.

Importing raises ImportError when the native kernels are missing, and the gates then
keep the default factories.
"""

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.mushroom import cap as _cap
from infinigen.assets.objects.mushroom import generate as _generate
from infinigen.assets.objects.mushroom import growth as _growth
from infinigen.assets.objects.mushroom import stem as _stem
from infinigen.assets.utils.decorate import subsurface2face_size
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import join_objects
from infinigen.core import surface
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform
from infinigen.perf.assets.objects.mushroom.adaptive import find_locations, mesh_buffers
from infinigen.perf.assets.objects.mushroom.meshing import (
    cap_attributes,
    displace_vertices,
    extension,
    origin2lowest,
    remesh_with_attrs,
    spin,
    texture_displace,
    web_mask,
)
from infinigen.perf.kernels import optional_import

# Every kernel these factories can reach, the GPL ones included, so that a
# missing one fails the import, which the gates handle, rather than a scene.
_KERNELS = (
    "infinigen.perf.assets.objects.mushroom._meshing",
    "infinigen.perf.assets.objects.mushroom._adaptive",
    "infinigen.infinigen_gpl.perf.voronoi._voronoi",
    "infinigen.infinigen_gpl.perf.spin._spin",
)


def _kernels_present():
    """True when every native kernel these factories can reach is importable."""
    hint = "Build with: INFINIGEN_INSTALL_PERF=True pip install -e ."
    return all(optional_import(mod, hint=hint) is not None for mod in _KERNELS)


if not _kernels_present():
    raise ImportError("native mushroom kernels are not built")


class _SurfaceBuffers:
    """A mushroom's (vertices, triangles), read when first unpacked."""

    def __init__(self, obj):
        self.obj = obj
        self._buffers = None

    def __iter__(self):
        if self._buffers is None:
            self._buffers = mesh_buffers(self.obj)
        return iter(self._buffers)


class MushroomFactory(_generate.MushroomFactory):
    """Optimized MushroomFactory: only the methods that differ from the default."""

    def build_mushrooms(self, i, face_size=0.01):
        n = np.random.randint(1, 6)
        mushrooms, keypoints = [], []
        for j in range(n):
            obj = self.factory.create_asset(
                i=j + i * self.max_cluster, face_size=face_size / 2
            )
            mushrooms.append(obj)
            # a lone mushroom sits at the origin, so read its surface only if
            # something else places it (the mushroom scatter reuses keypoints)
            keypoints.append(_SurfaceBuffers(obj) if n == 1 else mesh_buffers(obj))
        return mushrooms, keypoints

    def find_closest(self, keypoints, rotations, start_locs, directions):
        return find_locations(
            keypoints,
            rotations,
            start_locs,
            directions,
            self.radius,
            self.tolerant_length,
        )


class MushroomGrowthFactory(_growth.MushroomGrowthFactory):
    """Optimized MushroomGrowthFactory: moves the origin to the lowest point
    by writing the mesh's positions directly."""

    def create_asset(self, **params):
        cap = self.cap_factory(**params)
        stem = self.stem_factory(**params)
        obj = join_objects([cap, stem])
        origin2lowest(obj)
        return cap


class MushroomCapFactory(_cap.MushroomCapFactory):
    """Optimized MushroomCapFactory: only the methods that differ from the default."""

    def create_asset(self, face_size, **params) -> bpy.types.Object:
        cap_config = self.cap_config
        anchors = cap_config["x_anchors"], 0, cap_config["z_anchors"]
        obj = spin(
            anchors,
            cap_config["vector_locations"],
            face_size=(face_size if max(cap_config["x_anchors"]) > 0.1 else None),
        )
        self.apply_cut(obj)
        remesh_with_attrs(obj, face_size)
        cap_attributes(obj, self)
        assign_material(obj, self.material_cap)

        if self.is_morel:
            with butil.SelectObjects(obj):
                surface.set_active(obj, "morel")
                bpy.ops.geometry.attribute_convert(mode="VERTEX_GROUP")
            butil.modify_mesh(
                obj, "DISPLACE", vertex_group="morel", strength=0.04, mid_level=0.7
            )

        if self.gill_config is not None:
            gill_config = self.gill_config
            anchors = gill_config["x_anchors"], 0, gill_config["z_anchors"]
            gill = spin(
                anchors,
                gill_config["vector_locations"],
                dupli=True,
                loop=True,
                rotation_resolution=np.random.randint(8, 20),
            )
            subsurface2face_size(gill, face_size)
            assign_material(gill, self.material)
            obj = join_objects([obj, gill])

        texture = bpy.data.textures.new(
            name="cap", type=np.random.choice(["STUCCI", "MARBLE"])
        )
        texture.noise_scale = log_uniform(0.01, 0.05)
        texture_displace(obj, texture)

        extension(obj, 0.1)
        butil.modify_mesh(
            obj,
            "SIMPLE_DEFORM",
            deform_method="TWIST",
            angle=uniform(-np.pi / 4, np.pi / 4),
            deform_axis="X",
        )
        r1, r2, r3, r4 = uniform(-0.25, 0.25, 4)
        displace_vertices(
            obj,
            lambda x, y, z: (
                np.where(x > 0, r1, r2) * x,
                np.where(y > 0, r3, r4) * y,
                0,
            ),
        )
        tag_object(obj, "cap")
        return obj


class MushroomStemFactory(_stem.MushroomStemFactory):
    """Optimized MushroomStemFactory: only the methods that differ from the default."""

    def build_solid_web(self, inner_radius):
        """The default solid web spun natively: Blender's spin operator orders faces by
        memory address, so the default's face order changes between runs."""
        outer_radius = inner_radius * uniform(1.5, 3.5)
        z = uniform(0.0, 0.05)
        length = uniform(0.15, 0.2)
        x_anchors = inner_radius, (outer_radius + inner_radius) / 2, outer_radius
        z_anchors = -z, -z - uniform(0.3, 0.4) * length, -z - length
        anchors = x_anchors, 0, z_anchors
        obj = spin(anchors)
        surface.add_geomod(
            obj, self.geo_inverse_band, apply=True, input_args=[-uniform(0.008, 0.01)]
        )
        tag_object(obj, "web")
        return obj

    def build_hollow_web(self, inner_radius):
        outer_radius = inner_radius * uniform(2, 3.5)
        z = uniform(0.0, 0.05)
        length = log_uniform(0.2, 0.4)
        x_anchors = inner_radius, (outer_radius + inner_radius) / 2, outer_radius
        z_anchors = -z, -z - uniform(0.3, 0.4) * length, -z - length
        anchors = x_anchors, 0, z_anchors
        obj = spin(anchors)
        levels = 3
        butil.modify_mesh(obj, "SUBSURF", True, render_levels=levels, levels=levels)
        web_mask(obj, self)
        butil.modify_mesh(obj, "SMOOTH", iterations=2)
        tag_object(obj, "web")
        return obj

    def create_asset(self, face_size, **params) -> bpy.types.Object:
        length = log_uniform(0.4, 0.8)
        x_anchors = (
            0,
            self.inner_radius,
            log_uniform(1, 2) * self.inner_radius,
            self.inner_radius * uniform(1, 1.2),
            0,
        )
        z_anchors = 0, 0, -length * uniform(0.3, 0.7), -length, -length
        anchors = x_anchors, 0, z_anchors
        obj = spin(anchors, [1, 4], face_size=face_size)
        remesh_with_attrs(obj, face_size)
        if self.has_band:
            surface.add_geomod(
                obj,
                self.geo_band,
                apply=True,
                input_args=[length, uniform(0.008, 0.01)],
            )
        assign_material(obj, self.material)

        if self.web_builder is not None:
            web = self.web_builder(self.inner_radius)
            extension(web)
            subsurface2face_size(web, face_size / 2)
            assign_material(obj, self.material_web)
            obj = join_objects([web, obj])

        texture = bpy.data.textures.new(name="cap", type="STUCCI")
        texture.noise_scale = uniform(0.005, 0.01)
        texture_displace(obj, texture)

        butil.modify_mesh(
            obj,
            "SIMPLE_DEFORM",
            deform_method="BEND",
            angle=-uniform(0, np.pi / 2),
            deform_axis="Y",
        )
        tag_object(obj, "stem")
        return obj

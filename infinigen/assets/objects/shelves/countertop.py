# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
import shapely
from numpy.random import uniform

from infinigen.assets.materials import ceramic, marble
from infinigen.assets.materials.woods import wood_tile
from infinigen.assets.utils.decorate import read_center, read_normal, select_faces
from infinigen.assets.utils.mesh import separate_selected, snap_mesh
from infinigen.assets.utils.object import join_objects
from infinigen.assets.utils.shapes import (
    buffer,
    dissolve_limited,
    obj2polygon,
    safe_polygon2obj,
)
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.random import random_general as rg


class CountertopFactory(AssetFactory):
    surfaces = "weighted_choice", (5, marble), (2, ceramic), (2, wood_tile)

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.surface = rg(self.surfaces)
        self.thickness = uniform(0.02, 0.06)
        self.extrusion = 0 if uniform() < 0.4 else uniform(0.02, 0.03)
        self.h_snap = 0.5
        self.v_snap = 0.5
        self.v_merge = 0.1
        self.z_range = 0.5, 1.5
        self.surface = rg(self.surfaces)

    @staticmethod
    def generate_shelves():
        from .kitchen_cabinet import KitchenCabinetFactory
        from .simple_desk import SimpleDeskFactory

        shelves = make_asset_collection(
            [
                KitchenCabinetFactory(np.random.randint(1e7)),
                SimpleDeskFactory(np.random.randint(1e7)),
            ],
            10,
        )
        for s in shelves.objects:
            s.location = *uniform(-1, 1, 2), uniform(0, 0.5)
            s.rotation_euler[-1] = np.pi / 2 * np.random.randint(4)
        return shelves

    def create_asset(self, shelves=None, **params) -> bpy.types.Object:
        if shelves is None:
            shelves_generated = True
            shelves = self.generate_shelves()
        else:
            shelves_generated = False
        geoms, zs = [], []
        for s in shelves.objects:
            t = deep_clone_obj(s)
            z = read_center(t)[:, -1]
            max_z = np.max(z[(self.z_range[0] < z) & (z < self.z_range[1])])
            selection = (
                (read_normal(t)[:, -1] > 0.5) & (z - 1e-2 < max_z) & (max_z < z + 1e-2)
            )
            select_faces(t, selection)
            r = separate_selected(t, True)
            r.location = s.location
            r.rotation_euler = s.rotation_euler
            butil.apply_transform(r, True)
            p = self.rebuffer(obj2polygon(r), self.h_snap)
            q = buffer(p, self.extrusion)
            geoms.append(q)
            zs.append(max_z + s.location[-1])
            butil.delete([r, t])
        indices = np.argsort(zs)
        geoms_ = [geoms[i] for i in indices]
        zs_ = [zs[i] for i in indices]
        geoms, zs = [], []
        for i in range(len(indices)):
            if i == 0:
                geoms.append(geoms_[i])
                zs.append(zs_[i])
            elif zs_[i] < zs[-1] + self.v_merge:
                geoms[-1] = self.rebuffer(geoms[-1].union(geoms_[i]), self.h_snap)
            else:
                geoms.append(geoms_[i])
                zs.append(zs_[i])
        groups = []
        for i in range(len(geoms)):
            for j in range(i):
                if (
                    geoms[i].distance(geoms[j]) <= self.h_snap
                    and zs[i] - zs[j] < self.v_snap
                ):
                    group = next(g for g in groups if j in g)
                    group.add(i)
                    break
            else:
                groups.append({i})
        objs = []
        for group in groups:
            n = len(group)
            geoms_ = [geoms[i] for i in group]
            zs_ = [zs[i] for i in group]
            geom_unions = [
                self.rebuffer(shapely.union_all(geoms_[i:]), self.h_snap / 2)
                for i in range(n)
            ]
            geom_unions.append(shapely.Point())
            shapes = [
                self.rebuffer(geom_unions[i].difference(geom_unions[i + 1]), -1e-4)
                for i in range(n)
            ]
            for s, z in zip(shapes, zs_):
                if s.area > 0:
                    o = safe_polygon2obj(self.rebuffer(s, -1e-4).buffer(0))
                    if o is not None:
                        o.location[-1] = z
                        butil.apply_transform(o, True)
                        objs.append(o)
            ss = []
            for i in range(n - 1, -1, -1):
                for j in range(i - 1, -1, -1):
                    s = buffer(shapes[i], 1e-4).intersection(buffer(shapes[j], 1e-4))
                    ss.append(s)
                    for c in ss[:-1]:
                        s = s.difference(buffer(c, 1e-4))
                    if s.area == 0:
                        continue
                    o = safe_polygon2obj(s)
                    if o is None:
                        continue
                    butil.modify_mesh(o, "WELD", merge_threshold=5e-4)
                    o.location[-1] = zs_[i]
                    with butil.ViewportMode(o, "EDIT"):
                        bpy.ops.mesh.select_mode(type="EDGE")
                        bpy.ops.mesh.select_all(action="SELECT")
                        bpy.ops.mesh.extrude_edges_move(
                            TRANSFORM_OT_translate={"value": (0, 0, zs_[j] - zs_[i])}
                        )
                    objs.append(o)
        obj = join_objects(objs)
        snap_mesh(obj, 2e-2)
        dissolve_limited(obj)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
        butil.modify_mesh(
            obj,
            "SOLIDIFY",
            thickness=self.thickness,
            use_even_offset=True,
            offset=1,
            use_quality_normals=True,
        )

        if shelves_generated:
            for s in shelves.objects:
                s.parent = obj
        return objs[0]

    @staticmethod
    def rebuffer(shape, distance):
        return shape.buffer(distance, join_style="mitre", cap_style="flat").buffer(
            -distance, join_style="mitre", cap_style="flat"
        )

    def finalize_assets(self, assets):
        self.surface.apply(assets)

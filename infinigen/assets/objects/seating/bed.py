# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
from functools import cached_property

import bpy
import numpy as np
import trimesh
from numpy.random import uniform

from infinigen.assets.objects.seating import bedframe, mattress, pillow
from infinigen.assets.scatters import clothes
from infinigen.assets.utils.decorate import decimate, read_co, subsurf
from infinigen.assets.utils.object import obj2trimesh
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.random import log_uniform
from infinigen.core.util.random import random_general as rg


class BedFactory(bedframe.BedFrameFactory):
    mattress_types = "weighted_choice", (1, "coiled"), (3, "wrapped")
    sheet_types = (
        "weighted_choice",
        (4, "quilt"),
        (4, "comforter"),
        (4, "box_comforter"),
        (1, "none"),
    )

    def __init__(self, factory_seed, coarse=False):
        super(BedFactory, self).__init__(factory_seed, coarse)
        self.sheet_type = rg(self.sheet_types)
        self.sheet_folded = uniform() < 0.5
        self.has_cover = uniform() < 0.5

    @cached_property
    def mattress_factory(self):
        factory = mattress.MattressFactory(self.factory_seed, self.coarse)
        factory.type = rg(self.mattress_types)
        factory.width = self.width * uniform(0.88, 0.96)
        factory.size = self.size * uniform(0.88, 0.96)
        return factory

    @cached_property
    def quilt_factory(self):
        from infinigen.assets.objects.clothes.blanket import BlanketFactory

        factory = BlanketFactory(self.factory_seed, self.coarse)
        factory.width = self.mattress_factory.width * uniform(1.4, 1.6)
        factory.size = self.mattress_factory.size * uniform(0.9, 1.1)
        return factory

    @cached_property
    def comforter_factory(self):
        from infinigen.assets.objects.clothes.blanket import ComforterFactory

        factory = ComforterFactory(self.factory_seed, self.coarse)
        factory.width = self.mattress_factory.width * uniform(1.4, 1.8)
        factory.size = self.mattress_factory.size * uniform(0.9, 1.2)
        return factory

    @cached_property
    def box_comforter_factory(self):
        from infinigen.assets.objects.clothes.blanket import BoxComforterFactory

        factory = BoxComforterFactory(self.factory_seed, self.coarse)
        factory.width = self.mattress_factory.width * uniform(1.4, 1.8)
        factory.size = self.mattress_factory.size * uniform(0.9, 1.2)
        return factory

    @cached_property
    def cover_factory(self):
        from infinigen.assets.objects.clothes.blanket import BlanketFactory

        factory = BlanketFactory(self.factory_seed, self.coarse)
        factory.width = self.mattress_factory.width * uniform(1.6, 1.8)
        factory.size = self.mattress_factory.size * uniform(0.3, 0.4)
        return factory

    @cached_property
    def towel_factory(self):
        from infinigen.assets.objects.clothes import TowelFactory

        return TowelFactory(self.factory_seed)

    @cached_property
    def pillow_factory(self):
        return pillow.PillowFactory(self.factory_seed, self.coarse)

    def create_asset(self, i, **params) -> bpy.types.Object:
        frame = super().create_asset(i=i, **params)

        mattress = self.make_mattress(i)
        sheet = self.make_sheet(i, mattress, frame)
        cover = self.make_cover(i, sheet, mattress)

        n_pillows = np.random.randint(2, 4)
        if n_pillows > 0:
            pillow = self.pillow_factory(i)
            pillows = [pillow] + [deep_clone_obj(pillow) for _ in range(n_pillows - 1)]
        else:
            pillows = []
        self.pillow_factory.finalize_assets(pillows)
        points = np.stack(
            [
                uniform(0.1, 0.4, 10) * self.size,
                uniform(-0.3, 0.3, 10) * self.width,
                np.full(10, 1),
            ],
            -1,
        )
        self.scatter(pillows, points, [sheet, mattress])

        n_towels = np.random.randint(1, 2)
        if n_towels > 0:
            towel = self.towel_factory(i)
            towels = [towel] + [deep_clone_obj(towel) for _ in range(n_towels - 1)]
        else:
            towels = []
        self.towel_factory.finalize_assets(towels)
        points = np.stack(
            [
                uniform(0.5, 0.8, 10) * self.size,
                uniform(-0.3, 0.3, 10) * self.width,
                np.full(10, 1),
            ],
            -1,
        )
        self.scatter(towels, points, [sheet, mattress])

        for _ in [mattress, sheet, cover] + pillows + towels:
            _.parent = frame
        butil.select_none()
        return frame

    def make_mattress(self, i):
        mattress = self.mattress_factory(i=i)
        mattress.location = self.size / 2, 0, self.mattress_factory.thickness / 2
        mattress.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(mattress, True)
        self.mattress_factory.finalize_assets(mattress)
        return mattress

    def make_sheet(self, i, mattress, obj):
        match self.sheet_type:
            case "quilt":
                factory = self.quilt_factory
                pressure = 0
            case "comforter":
                factory = self.comforter_factory
                pressure = uniform(1.0, 1.5)
            case _:
                factory = self.box_comforter_factory
                pressure = log_uniform(8, 15)
        sheet = factory(i)
        if self.sheet_folded:
            factory.fold(sheet)
        factory.finalize_assets(sheet)
        z_sheet = mattress.location[-1] + np.max(read_co(mattress)[:, -1])
        sheet.location = factory.size / 2 + uniform(0, 0.15), 0, z_sheet
        sheet.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(sheet, True)
        clothes.cloth_sim(
            sheet,
            [mattress, obj],
            mass=0.05,
            tension_stiffness=2,
            distance_min=5e-3,
            use_pressure=True,
            uniform_pressure_force=pressure,
            use_self_collision=self.sheet_folded,
        )
        subsurf(sheet, 2)
        return sheet

    def make_cover(self, i, sheet, mattress):
        cover = self.cover_factory(i)
        self.cover_factory.finalize_assets(cover)
        z_sheet = sheet.location[-1] + np.max(read_co(sheet)[:, -1])
        cover.location = self.size / 2 + uniform(0, 0.3), 0, z_sheet
        cover.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(cover, True)
        clothes.cloth_sim(
            cover,
            [sheet, mattress],
            80,
            mass=0.05,
            tension_stiffness=2,
            distance_min=5e-3,
        )
        subsurf(cover, 2)
        return cover

    def scatter(self, pillows, points, bases):
        dir = np.array([[0, 0, -1]])
        lengths = np.full(len(points), np.inf)
        for b in bases:
            lengths = np.minimum(
                lengths,
                trimesh.proximity.longest_ray(
                    obj2trimesh(b), points, np.repeat(dir, len(points), 0)
                ),
            )
        points += dir * lengths[:, np.newaxis]
        for a, loc in zip(pillows, decimate(points, len(pillows))):
            a.location = loc
            a.location[-1] += 0.02 - np.min(read_co(a)[:, -1])
            a.rotation_euler[-1] = uniform(0, np.pi)

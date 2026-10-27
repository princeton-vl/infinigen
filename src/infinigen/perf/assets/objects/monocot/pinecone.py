# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen pinecone (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/monocot/pinecone.py)
# - David Yan, Jia Deng: optimized implementation

import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.monocot.pinecone import (
    PineconeFactory as _DefaultPineconeFactory,
)
from infinigen.core.surface import shaderfunc_to_material
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform
from infinigen.perf import options

from ._pinecone.libraries import kernels_available


class PineconeFactory(_DefaultPineconeFactory):
    def __init__(self, factory_seed, coarse=False):
        # without the compiled kernels, use the default implementation rather
        # than fail partway through generation
        mode = "fast" if options.native_pinecone() and kernels_available() else "legacy"
        self.pinecone_backend = mode
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.angle = 2 * np.pi / (np.random.randint(4, 8) + 0.5)
            self.max_y_angle = uniform(0.7, 0.8) * np.pi / 2
            self.leaf_prob = uniform(0.9, 0.95)
            self.count = int(log_uniform(64, 96))
            self.stem_offset = uniform(0.2, 0.4)
            self.perturb = 0
            self.scale_curve = [
                (0, 0.5),
                (0.5, uniform(0.6, 1.0)),
                (1, uniform(0.1, 0.2)),
            ]
            self.bright_color = hsv2rgba(uniform(0.02, 0.06), uniform(0.8, 1.0), 0.01)
            self.dark_color = hsv2rgba(uniform(0.02, 0.06), uniform(0.8, 1.0), 0.005)
            self.material = shaderfunc_to_material(
                self.shader_monocot,
                self.dark_color,
                self.bright_color,
                self.use_distance,
            )

        if mode == "fast":
            # load the kernels now, so a broken build fails before generation
            from ._pinecone import core  # noqa: F401

    def create_asset(self, *args, **kwargs):
        if self.pinecone_backend == "legacy":
            return super().create_asset(*args, **kwargs)
        from ._pinecone import core

        return core.create_asset(self, *args, **kwargs)

    def create_raw(self, *args, **kwargs):
        if self.pinecone_backend == "legacy":
            return super().create_raw(*args, **kwargs)
        from ._pinecone import core

        return core.create_raw(self, *args, **kwargs)

    def build_stem(self, *args, **kwargs):
        if self.pinecone_backend == "legacy":
            return super().build_stem(*args, **kwargs)
        from ._pinecone import core

        return core.build_stem(self, *args, **kwargs)

    def decorate_monocot(self, *args, **kwargs):
        if self.pinecone_backend == "legacy":
            return super().decorate_monocot(*args, **kwargs)
        from ._pinecone import core

        return core.decorate(self, *args, **kwargs)

    def build_instance(self, *args, **kwargs):
        if self.pinecone_backend == "legacy":
            return super().build_instance(*args, **kwargs)
        from ._pinecone import instances

        return instances.build_instance_native(self, *args, **kwargs)

    def make_collection(self, *args, **kwargs):
        if self.pinecone_backend == "legacy":
            return super().make_collection(*args, **kwargs)
        from ._pinecone import instances

        return instances.make_collection(self, *args, **kwargs)

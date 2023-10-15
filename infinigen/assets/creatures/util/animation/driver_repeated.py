# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import numpy as np
from numpy.random import uniform as U, uniform

from infinigen.core.util.math import FixedSeed, int_hash


def repeated_driver(start, end, freq, off=None, seed=None):
    if off is None:
        off = uniform(0, 1)
    if seed is None:
        seed = np.random.randint(1e5)
    with FixedSeed(seed):
        phase = uniform(.2, .8)
        u = phase * uniform(.8, 1.)
        v = (1 - phase) * uniform(.8, 1.)
        t = f"{freq: .4f} * frame+{off:.4f}"
        t = f"{t}-floor({t})"
        return f"{start}+{end - start}*(smoothstep(0,{u},{t})-smoothstep({phase},{phase + v},{t}))"


def bend_bones_lerp(arma, bones, total, freq, rot=None, symmetric=True):
    bone_lengths = []
    for bone in bones:
        length = bone.bone['length'] if isinstance(bone.bone['length'], (int, float)) else 0
        if length >= 0:
            bone_lengths.append((bone, length))
    bone_lengths = list(sorted(bone_lengths, key=lambda _: _[1]))
    bones = [b for (b, _) in bone_lengths] if bone_lengths else bones
    bone = bones[0].bone
    hashable = arma.parent.name, bone['factory_class'], bone['index'] + (bone['side'] > 0) * symmetric
    with FixedSeed(int_hash(hashable)):
        ratio = uniform(1, 2, len(bones))
        ratio /= ratio.sum()
        total = [(t if uniform(0, 1) < .5 else (t[1], t[0])) if isinstance(t, tuple) else (t, t) for t in total]
        o0, o1, o2 = uniform(0, 1, 3)
        seed = np.random.randint(1e5)
        for i, (bone, r) in enumerate(zip(bones, ratio)):
            s = bone.bone['side']
            bone.rotation_mode = 'XYZ'
            (x0, x1), (y0, y1), (z0, z1) = total
            if rot is not None and i == 0:
                x0 += rot[0]
                x1 += rot[0]
                y0 += rot[1]
                y1 += rot[1]
                z0 += rot[2]
                z1 += rot[2]
            driver_x, driver_y, driver_z = [_.driver for _ in bone.driver_add('rotation_euler')]
            driver_x.expression = f"({repeated_driver(x0, x1, freq, o0, seed)})*{r}"
            driver_y.expression = f"({repeated_driver(y0, y1, freq, o1, seed)})*{r}"
            driver_z.expression = f"({repeated_driver(z0, z1, freq, o2, seed)})*{s * r}"

# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf
from procfunc.nodes import types as t

__all__ = [
    "lamp_bulb_nonemissive",
]


@pf.nodes.node_function
def lamp_bulb_nonemissive(
    base_color: t.SocketOrVal[pf.Color] = pf.Color((0.5, 0.44, 0.37)),
) -> pf.Material:
    """Material that is transparent to light rays but translucent to camera.

    Used for lamp bulbs so they don't block the point light inside.
    """
    light_path = pf.nodes.shader.light_path()
    transparent = pf.nodes.shader.transparent_bsdf(color=base_color)
    translucent = pf.nodes.shader.translucent_bsdf(color=base_color)
    surface = pf.nodes.shader.mix_shader(
        factor=light_path.is_camera_ray,
        a=transparent,
        b=translucent,
    )
    return pf.Material(
        surface=surface, displacement=pf.nodes.math.constant((0.0, 0.0, 0.0))
    )

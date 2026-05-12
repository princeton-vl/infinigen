import procfunc as pf
from procfunc.nodes import types as t


@pf.nodes.node_function
def lamp_bulb_nonemissive(
    color: t.SocketOrVal[pf.Color] = pf.Color((0.5, 0.44, 0.37)),
) -> pf.Material:
    """Material that is transparent to light rays but translucent to camera.

    Used for lamp bulbs so they don't block the point light inside.
    """
    light_path = pf.nodes.shader.light_path()
    transparent = pf.nodes.shader.transparent_bsdf(color=color)
    translucent = pf.nodes.shader.translucent_bsdf(color=color)
    surface = pf.nodes.shader.mix_shader(
        factor=light_path.is_camera_ray,
        a=transparent,
        b=translucent,
    )
    return pf.Material(surface=surface)

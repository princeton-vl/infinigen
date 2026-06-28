import procfunc as pf
from procfunc.nodes import types as t


@pf.nodes.node_function
def glass_colored(
    color: t.SocketOrVal[pf.Color],
    roughness: t.SocketOrVal[float] = 0.0,
) -> pf.Material:
    surface = pf.nodes.shader.principled_bsdf(
        base_color=color,
        roughness=roughness,
        ior=1.5,
        transmission_weight=1.0,
    )
    return pf.Material(
        surface=surface,
        displacement=pf.nodes.math.constant((0.0, 0.0, 0.0)),
    )


def glass_colored_color_distribution(rng: pf.RNG) -> pf.Color:
    hue = pf.random.uniform(rng, 0.0, 1.0)
    sat = pf.random.uniform(rng, 0.5, 0.9)
    val = pf.random.uniform(rng, 0.6, 0.9)
    return pf.color.hsv_color(hue=hue, saturation=sat, value=val)


def glass_colored_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector] | None = None,
    color: t.SocketOrVal[pf.Color] | None = None,
    roughness: t.SocketOrVal[float] | None = None,
) -> pf.Material:
    del vector
    if color is None:
        color = pf.control.choice(
            rng,
            [
                (glass_colored_color_distribution(rng), 0.7),
                (pf.color.hsv_color(hue=0.0, saturation=0.0, value=1.0), 0.3),
            ],
        )
    if roughness is None:
        roughness = pf.random.clip_gaussian(rng, 0.0, 0.02, 0.0, 0.05)

    return glass_colored(color=color, roughness=roughness)

import procfunc as pf
from procfunc.nodes import types as t


@pf.nodes.node_function
def glass_colored(
    color: t.SocketOrVal[pf.Color],
    density: t.SocketOrVal[float] = 50.0,
    roughness: t.SocketOrVal[float] = 0.0,
) -> pf.Material:
    surface = pf.nodes.shader.principled_bsdf(
        roughness=roughness,
        ior=1.5,
        transmission_weight=1.0,
    )
    volume = pf.nodes.shader.volume_absorption(color=color, density=density)
    return pf.Material(surface=surface, volume=volume)


def glass_colored_color_distribution(rng: pf.RNG) -> pf.Color:
    hue = pf.random.uniform(rng, 0.0, 1.0)
    sat = pf.random.uniform(rng, 0.5, 0.9)
    val = pf.random.uniform(rng, 0.6, 0.9)
    return pf.color.hsv_color(hue=hue, saturation=sat, value=val)


def glass_colored_distribution(
    rng: pf.RNG,
    vector: pf.ProcNode[pf.Vector] | None = None,
    color: t.SocketOrVal[pf.Color] | None = None,
    density: t.SocketOrVal[float] | None = None,
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
    if density is None:
        density = pf.random.uniform(rng, 30.0, 150.0)
    if roughness is None:
        roughness = pf.random.clip_gaussian(rng, 0.0, 0.02, 0.0, 0.05)

    return glass_colored(color=color, density=density, roughness=roughness)

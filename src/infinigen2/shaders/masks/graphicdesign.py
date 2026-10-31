# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen graphic design mask (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/materials/art.py)
# - Alexander Raistrick: refactor for Infinigen2

import colorsys
from typing import Any

import bpy
import numpy as np
import procfunc as pf
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import (
    Arrow,
    Circle,
    Ellipse,
    FancyBboxPatch,
    Rectangle,
    RegularPolygon,
    Wedge,
)
from procfunc.nodes import types as t

__all__ = [
    "art",
    "art_image_rand",
    "art_rand",
]

_PATCH_TYPES = (
    "circle",
    "circle",
    "rectangle",
    "rectangle",
    "rectangle",
    "rectangle",
    "wedge",
    "polygon",
    "ellipse",
    "arrow",
    "arrow",
    "box",
    "box",
)
_HATCHES = ("/", "\\", "|", "-", "+", "x", "o", "O", ".", "*")
_BOX_STYLES = ("round", "round4", "sawtooth")


def _palette(rng: pf.RNG) -> np.ndarray:
    count = 20
    hue = (rng.uniform() + np.linspace(0.0, 0.5, count)) % 1.0
    saturation = np.concatenate(
        [np.linspace(1.0, 0.5, count // 2), np.linspace(0.5, 1.0, count // 2)]
    )
    if rng.uniform() < 0.5:
        lightness = np.linspace(rng.uniform(), rng.uniform(0.6, 0.8), count)
    else:
        midpoint = rng.uniform(0.6, 0.8)
        rising = np.linspace(rng.uniform(0.1, 0.3), midpoint, count // 2)
        falling = np.linspace(midpoint, rng.uniform(0.1, 0.3), count // 2)
        lightness = np.concatenate([rising, falling])
    saturation *= rng.uniform()
    lightness *= rng.uniform(0.5, 1.0)
    colors = [
        colorsys.hls_to_rgb(h, light, sat)
        for h, light, sat in zip(hue, lightness, saturation, strict=True)
    ]
    return np.asarray(colors)


def _random_color(rng: pf.RNG, palette: np.ndarray) -> np.ndarray:
    roll = rng.uniform()
    if roll < 0.03:
        return np.ones(3)
    if roll < 0.08:
        return np.zeros(3)
    return palette[rng.integers(0, len(palette))]


def _contrasting_colors(
    rng: pf.RNG, palette: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(64):
        first = _random_color(rng, palette)
        second = _random_color(rng, palette)
        if np.abs(first - second).sum() > 0.2:
            return first, second
    return palette[0], palette[len(palette) // 2]


def _locations(rng: pf.RNG, count: int) -> np.ndarray:
    candidates = rng.uniform(0.1, 0.9, size=(count * 8, 2))
    chosen = [rng.integers(0, len(candidates))]
    nearest = np.full(len(candidates), np.inf)
    for _ in range(1, count):
        delta = candidates - candidates[chosen[-1]]
        nearest = np.minimum(nearest, np.square(delta).sum(axis=1))
        nearest[chosen] = -1.0
        chosen.append(int(np.argmax(nearest)))
    return candidates[chosen]


def _scale(rng: pf.RNG) -> float:
    return float(np.exp(rng.uniform(np.log(0.1), np.log(0.5))))


def _add_divider(axis: Axes, rng: pf.RNG, palette: np.ndarray) -> None:
    if rng.uniform() < 0.6:
        return
    angle = 0.0 if rng.uniform() < 0.7 else rng.uniform(5.0, 10.0)
    x = rng.uniform(0.1, 0.9)
    y = rng.uniform(0.1, 0.9)
    options = (
        ((0.0, y), 2.0, 2.0, angle),
        ((0.0, y), 2.0, -2.0, -angle),
        ((1.0, y), -2.0, -2.0, angle),
        ((1.0, y), -2.0, 2.0, -angle),
    )
    origin, width, height, rotation = options[rng.integers(0, len(options))]
    axis.add_patch(
        Rectangle(
            origin,
            width,
            height,
            angle=rotation,
            color=_random_color(rng, palette),
        )
    )


def _add_patch(
    axis: Axes, rng: pf.RNG, palette: np.ndarray, location: np.ndarray
) -> None:
    x, y = location
    width = _scale(rng)
    height = _scale(rng)
    radius = min(width, height) * 0.5
    kind = rng.choice(_PATCH_TYPES)
    alpha = rng.uniform(0.5, 0.8) if rng.uniform() < 0.2 else 1.0
    fill = rng.uniform() < 0.2
    angle = 0.0 if rng.uniform() < 0.8 else rng.uniform(-30.0, 30.0)
    orientation = rng.uniform(0.0, np.pi * 2.0)
    face_color, edge_color = _contrasting_colors(rng, palette)
    hatch = rng.choice(_HATCHES) if rng.uniform() < 0.3 else None
    style: dict[str, Any] = {
        "alpha": alpha,
        "edgecolor": edge_color,
        "facecolor": face_color,
        "fill": fill,
        "hatch": hatch,
        "linewidth": rng.uniform(2.0, 5.0),
    }
    if kind == "circle":
        patch = Circle((x, y), radius, **style)
    elif kind == "rectangle":
        patch = Rectangle(
            (x - width * 0.5, y - height * 0.5),
            width,
            height,
            angle=angle,
            **style,
        )
    elif kind == "wedge":
        start = rng.uniform(0.0, 360.0)
        patch = Wedge(
            (x, y),
            radius,
            start,
            start + rng.uniform(0.0, 360.0),
            width=rng.uniform(0.2, 0.8) * radius,
            **style,
        )
    elif kind == "polygon":
        patch = RegularPolygon(
            (x, y),
            rng.integers(3, 9),
            radius=radius,
            orientation=orientation,
            **style,
        )
    elif kind == "ellipse":
        patch = Ellipse((x, y), width, height, angle=angle, **style)
    elif kind == "arrow":
        direction_x = width if rng.uniform() < 0.5 else -width
        direction_y = height if rng.uniform() < 0.5 else -height
        patch = Arrow(
            x - direction_x * 0.5,
            y - direction_y * 0.5,
            direction_x,
            direction_y,
            width=np.exp(rng.uniform(np.log(0.6), np.log(1.5))),
            **style,
        )
    else:
        pad = rng.uniform(0.2, 0.4) * min(width, height)
        box_style = rng.choice(_BOX_STYLES)
        patch = FancyBboxPatch(
            (x - width * 0.5, y - height * 0.5),
            width - pad,
            height - pad,
            boxstyle=f"{box_style},pad={pad}",
            mutation_scale=np.exp(rng.uniform(np.log(0.6), np.log(1.5))),
            mutation_aspect=np.exp(rng.uniform(np.log(0.6), np.log(1.5))),
            **style,
        )
    axis.add_patch(patch)


@pf.tracer.primitive
def art_image_rand(rng: pf.RNG) -> pf.Image:
    palette = _palette(rng)
    figure = Figure(figsize=(4.0, 4.0), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    background = _random_color(rng, palette)
    axis.set_facecolor(
        (float(background[0]), float(background[1]), float(background[2]))
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    count = int(rng.integers(10, 15))
    _add_divider(axis, rng, palette)
    for location in _locations(rng, count):
        _add_patch(axis, rng, palette, location)
    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba(), dtype=np.float32)[::-1].copy() / 255.0
    image = bpy.data.images.new("Art", width=400, height=400, alpha=True)
    image.pixels.foreach_set(pixels.ravel())
    image.pack()
    figure.clear()
    return pf.Image(image)


@pf.tracer.generator
def art(
    vector: t.SocketOrVal[pf.Vector],
    image: pf.Image,
    noise_mix: t.SocketOrVal[float] = 0.5,
    uv_mix: t.SocketOrVal[float] = 1.0,
) -> pf.ProcNode[pf.Color]:
    voronoi = pf.nodes.texture.voronoi(vector=vector, scale=60.0)
    coarse_noise = pf.nodes.texture.noise(
        vector=vector,
        scale=5.0,
        detail=4.6,
        roughness=0.3789291416,
        normalize=False,
    )
    detail_noise = pf.nodes.texture.noise(
        vector=vector,
        scale=35.4,
        detail=3.3,
        roughness=1.0,
    )
    noise_vector = pf.nodes.math.mix(
        factor=noise_mix,
        a=coarse_noise.fac.astype(dtype=pf.Vector),
        b=detail_noise.color.astype(dtype=pf.Vector),
    )
    warped_vector = pf.nodes.math.mix(
        factor=0.0417,
        a=voronoi.position,
        b=noise_vector,
    )
    image_vector = pf.nodes.math.mix(
        factor=uv_mix,
        a=warped_vector,
        b=vector,
    )
    image_texture = pf.nodes.texture.image(vector=image_vector, image=image)
    return image_texture.color


def _mix_factors_rand(
    rng: pf.RNG,
) -> tuple[pf.ProcNode[float], pf.ProcNode[float]]:
    r_noise_mix, r_uv_choice, r_uv_mix = rng.spawn(3)
    noise_mix = pf.random.uniform(r_noise_mix, 0.2, 1.0)
    distorted_uv_mix = pf.random.uniform(r_uv_mix, 0.0, 0.4)
    uv_mix = pf.control.choice(
        r_uv_choice,
        [
            (distorted_uv_mix, 1.0),
            (1.0, 1.0),
        ],
    )
    return noise_mix, uv_mix


def art_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    image: pf.Image | None = None,
) -> pf.ProcNode[pf.Color]:
    r_image, r_mix = rng.spawn(2)
    if image is None:
        image = art_image_rand(r_image)
    noise_mix, uv_mix = _mix_factors_rand(r_mix)
    return art(vector, image, noise_mix, uv_mix)

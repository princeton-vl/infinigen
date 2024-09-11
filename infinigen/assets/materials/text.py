# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: text & art generators
# - Stamatis Alexandropoulos: image postprocessing effects
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=hpamCaVrbTk by Joey Carlino

import colorsys
import inspect
import io
import logging
import os

import bpy
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Arrow,
    BoxStyle,
    Circle,
    Ellipse,
    FancyBboxPatch,
    Rectangle,
    RegularPolygon,
    Wedge,
)
from numpy.random import rand, uniform
from PIL import Image

from infinigen import repo_root
from infinigen.assets.materials import common
from infinigen.assets.utils.decorate import decimate
from infinigen.assets.utils.misc import generate_text
from infinigen.assets.utils.object import new_plane
from infinigen.assets.utils.uv import compute_uv_direction
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, clip_gaussian
from infinigen.core.util.random import log_uniform
from infinigen.core.util.random import random_general as rg

logger = logging.getLogger(__name__)

font_dir = repo_root() / "infinigen/assets/fonts"
for f in matplotlib.font_manager.findSystemFonts([font_dir]):
    matplotlib.font_manager.fontManager.addfont(f)
font_names = [_.replace("_", " ") for _ in os.listdir(font_dir)]
all_fonts = matplotlib.font_manager.get_font_names()
assert [f in all_fonts for f in font_names]


class Text:
    default_font_name = "DejaVu Sans"
    patch_fns = (
        "weighted_choice",
        (2, Circle),
        (4, Rectangle),
        (1, Wedge),
        (1, RegularPolygon),
        (1, Ellipse),
        (2, Arrow),
        (2, FancyBboxPatch),
    )
    hatches = {"/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"}
    font_weights = ["normal", "bold", "heavy"]
    font_styles = ["normal", "italic", "oblique"]

    def __init__(self, factory_seed, has_barcode=True, emission=0):
        self.factory_seed = factory_seed
        with FixedSeed(self.factory_seed):
            self.size = 4
            self.dpi = 100
            self.colormap = (
                self.build_sequential_colormap()
                if uniform() < 0.5
                else self.build_diverging_colormap()
            )
            self.white_chance = 0.03
            self.black_chance = 0.05

            self.n_patches = np.random.randint(5, 8)
            self.force_horizontal = uniform() < 0.75

            self.n_texts = np.random.randint(2, 4)

            self.n_barcodes = 1 if has_barcode and uniform() < 0.5 else 0
            self.barcode_scale = uniform(0.3, 0.6)
            self.barcode_length = np.random.randint(25, 40)
            self.barcode_aspect = log_uniform(1.5, 3)

            self.emission = emission

    @staticmethod
    def build_diverging_colormap():
        count = 20
        hue = (uniform() + np.linspace(0, 0.5, count)) % 1
        mid = uniform(0.6, 0.8)
        lightness = np.concatenate(
            [
                np.linspace(uniform(0.1, 0.3), mid, count // 2),
                np.linspace(mid, uniform(0.1, 0.3), count // 2),
            ]
        )
        saturation = np.concatenate(
            [np.linspace(1, 0.5, count // 2), np.linspace(0.5, 1, count // 2)]
        )

        # TODO hack
        saturation *= uniform(0, 1)
        lightness *= uniform(0.5, 1)

        return np.array(
            [
                colorsys.hls_to_rgb(h, l, s)
                for h, l, s in zip(hue, lightness, saturation)
            ]
        )

    @staticmethod
    def build_sequential_colormap():
        count = 20
        hue = (uniform() + np.linspace(0, 0.5, count)) % 1
        lightness = np.linspace(uniform(0.0), uniform(0.6, 0.8), count)
        saturation = np.concatenate(
            [np.linspace(1, 0.5, count // 2), np.linspace(0.5, 1, count // 2)]
        )

        # TODO hack
        saturation *= uniform(0, 1)
        lightness *= uniform(0.5, 1)

        return np.array(
            [
                colorsys.hls_to_rgb(h, l, s)
                for h, l, s in zip(hue, lightness, saturation)
            ]
        )

    @property
    def random_color(self):
        r = uniform()
        if r < self.white_chance:
            return np.array([1, 1, 1])
        elif r < self.white_chance + self.black_chance:
            return np.array([0, 0, 0])
        else:
            return self.colormap[np.random.randint(len(self.colormap))]

    @property
    def random_colors(self):
        while True:
            c, d = self.random_color, self.random_color
            if np.abs(c - d).sum() > 0.2:
                return c, d

    def build_image(self, bbox):
        fig = plt.figure(figsize=(self.size, self.size), dpi=self.dpi)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_facecolor(self.random_color)
        locs = self.get_locs(bbox, self.n_patches + self.n_texts + self.n_barcodes)
        self.add_divider(bbox)
        self.add_patches(locs[: self.n_patches], bbox)
        self.add_texts(locs[self.n_patches : self.n_patches + self.n_texts])
        self.add_barcodes(locs[self.n_patches + self.n_texts :])
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        size = self.size * self.dpi
        image = bpy.data.images.new("text_texture", width=size, height=size, alpha=True)
        data = np.asarray(Image.open(buffer), dtype=np.float32)[::-1, :] / 255.0
        image.pixels.foreach_set(data.ravel())
        image.pack()
        plt.close("all")
        plt.clf()
        return image

    @staticmethod
    def loc_uniform(min_, max_, size=None):
        ratio = 0.1
        return uniform(
            min_ + ratio * (max_ - min_), min_ + (1 - ratio) * (max_ - min_), size
        )

    @staticmethod
    def scale_uniform(min_, max_):
        return (max_ - min_) * log_uniform(0.2, 0.8)

    def get_locs(self, bbox, n):
        m = 8 * n
        x, y = (
            self.loc_uniform(bbox[0], bbox[1], m),
            self.loc_uniform(bbox[2], bbox[3], m),
        )
        return decimate(np.stack([x, y], -1), n)

    def add_divider(self, rs):
        if uniform() < 0.6:
            return
        a = 0 if uniform() < 0.7 else uniform(5, 10)
        x, y = self.loc_uniform(rs[0], rs[1]), self.loc_uniform(rs[2], rs[3])
        if rs[0] == 0 or self.force_horizontal:
            args_list = [
                [(0, y), 2, 2, a],
                [(0, y), 2, -2, -a],
                [(1, y), -2, -2, a],
                [(1, y), -2, 2, -a],
            ]
        else:
            args_list = [
                [(x, 0), -2, 2, a],
                [(x, 0), 2, 2, -a],
                [(x, 1), 2, -2, a],
                [(x, 1), -2, -2, -a],
            ]
        args = args_list[np.random.randint(len(args_list))]
        plt.gca().add_patch(
            Rectangle(*args[:-1], angle=args[-1], color=self.random_color)
        )

    def add_patches(self, locs, bbox):
        for x, y in locs:
            w, h = (
                self.scale_uniform(bbox[0], bbox[1]),
                self.scale_uniform(bbox[2], bbox[3]),
            )
            x_, y_ = x - w / 2, y - h / 2
            r = min(w, h) / 2
            fn = rg(self.patch_fns)
            kwargs = {
                "alpha": uniform(0.5, 0.8) if uniform() < 0.2 else 1,
                "fill": uniform() < 0.2,
                "angle": 0 if uniform() < 0.8 else uniform(-30, 30),
                "orientation": uniform(0, np.pi * 2),
            }
            kwargs = {
                k: kwargs[k]
                for k, v in inspect.signature(fn).parameters.items()
                if k in kwargs
            }
            face_color, edge_color = self.random_colors
            kwargs.update(
                {
                    "facecolor": face_color,
                    "edgecolor": edge_color,
                    "hatch": np.random.choice(list(self.hatches))
                    if uniform() < 0.3
                    else "none",
                    "linewidth": uniform(2, 5),
                }
            )
            match fn.__name__:
                case Circle.__name__:
                    patch = Circle((x, y), r, **kwargs)
                case Rectangle.__name__:
                    patch = Rectangle((x_, y_), w, h, **kwargs)
                case Wedge.__name__:
                    start = uniform(0, 360)
                    patch = Wedge(
                        (x, y),
                        r,
                        start,
                        start + uniform(0, 360),
                        width=uniform(0.2, 0.8) * r,
                        **kwargs,
                    )
                case RegularPolygon.__name__:
                    patch = RegularPolygon(
                        (x, y), np.random.randint(3, 9), radius=r, **kwargs
                    )
                case Ellipse.__name__:
                    patch = Ellipse((x, y), w, h, **kwargs)
                case Arrow.__name__:
                    w_, h_ = (
                        (w if uniform() < 0.5 else -w),
                        (h if uniform() < 0.5 else -h),
                    )
                    patch = Arrow(
                        x - w_ / 2,
                        y - h_ / 2,
                        w,
                        h,
                        width=log_uniform(0.6, 1.5),
                        **kwargs,
                    )
                case FancyBboxPatch.__name__:
                    pad = uniform(0.2, 0.4) * min(w, h)
                    box_style = np.random.choice(list(BoxStyle.get_styles().values()))(
                        pad=pad
                    )
                    patch = FancyBboxPatch(
                        (x_, y_),
                        w - pad,
                        h - pad,
                        box_style,
                        mutation_scale=log_uniform(0.6, 1.5),
                        mutation_aspect=log_uniform(0.6, 1.5),
                        **kwargs,
                    )
                case _:
                    raise NotImplementedError
            try:
                plt.gca().add_patch(patch)
            except MemoryError:
                logger.warning(
                    f"Failed to add patch {fn.__name__} at {x, y} with {w, h} due to MemoryError"
                )

    def add_texts(self, locs):
        for x, y in locs:
            x = 0.5 + (x - 0.5) * 0.6
            text = generate_text()
            family = np.random.choice(font_names)
            color, background_color = self.random_colors
            plt.figtext(
                x,
                y,
                text,
                family=family,
                size=log_uniform(0.75, 1)
                * self.dpi
                * clip_gaussian(0.3, 0.2, 0.2, 0.65),
                ha="center",
                va="center",
                c=color,
                rotation=uniform(-10, 10),
                wrap=True,
                fontweight=np.random.choice(self.font_weights),
                fontstyle=np.random.choice(self.font_styles),
                backgroundcolor=background_color,
            )

    def add_barcodes(self, locs):
        fig = plt.gcf()
        for x, y in locs:
            code = np.random.randint(0, 2, self.barcode_length)
            h = self.barcode_scale / self.size
            w = h * self.barcode_aspect
            ax = fig.add_axes((x - w / 2, y - h / 2, w, h))
            ax.set_axis_off()
            ax.imshow(
                code.reshape(1, -1),
                cmap="binary",
                aspect="auto",
                interpolation="nearest",
            )

    def make_shader_func(self, bbox):
        assert bbox[1] - bbox[0] > 0.001 and bbox[3] - bbox[2] > 0.001
        image = self.build_image(bbox)

        def shader_text(nw: NodeWrangler, **kwargs):
            uv_map = nw.new_node(Nodes.UVMap)

            reroute = nw.new_node(Nodes.Reroute, input_kwargs={"Input": uv_map})

            voronoi_texture = nw.new_node(
                Nodes.VoronoiTexture, input_kwargs={"Vector": reroute, "Scale": 60.0000}
            )

            voronoi_texture_1 = nw.new_node(
                Nodes.VoronoiTexture, input_kwargs={"Vector": reroute, "Scale": 60.0000}
            )

            mix = nw.new_node(
                Nodes.Mix,
                input_kwargs={
                    6: voronoi_texture.outputs["Position"],
                    7: voronoi_texture_1.outputs["Position"],
                },
                attrs={"data_type": "RGBA"},
            )

            musgrave_texture = nw.new_node(
                Nodes.MusgraveTexture,
                input_kwargs={"Vector": reroute, "Detail": 5.6000, "Dimension": 1.4000},
            )

            noise_texture_1 = nw.new_node(
                Nodes.NoiseTexture,
                input_kwargs={
                    "Vector": reroute,
                    "Scale": 35.4000,
                    "Detail": 3.3000,
                    "Roughness": 1.0000,
                },
            )

            mix_3 = nw.new_node(
                Nodes.Mix,
                input_kwargs={
                    0: uniform(0.2, 1.0),
                    6: musgrave_texture,
                    7: noise_texture_1.outputs["Color"],
                },
                attrs={"data_type": "RGBA"},
            )

            mix_1 = nw.new_node(
                Nodes.Mix,
                input_kwargs={0: 0.0417, 6: mix.outputs[2], 7: mix_3.outputs[2]},
                attrs={"data_type": "RGBA"},
            )

            if rand() < 0.5:
                mix_2 = nw.new_node(
                    Nodes.Mix,
                    input_kwargs={0: uniform(0, 0.4), 6: mix_1.outputs[2], 7: uv_map},
                    attrs={"data_type": "RGBA"},
                )
            else:
                mix_2 = nw.new_node(
                    Nodes.Mix,
                    input_kwargs={0: 1.0, 6: mix_1.outputs[2], 7: uv_map},
                    attrs={"data_type": "RGBA"},
                )
            # mix_2 = nw.new_node(Nodes.Mix, input_kwargs={0: 0.7375, 6: uv, 7: mix_1.outputs[2]}, attrs={'data_type': 'RGBA'})
            color = nw.new_node(
                Nodes.ShaderImageTexture, [mix_2], attrs={"image": image}
            ).outputs[0]
            roughness = nw.new_node(Nodes.NoiseTexture)
            if self.emission > 0:
                emission = color
                color = (0.05, 0.05, 0.05, 1)
                roughness = 0.05
            else:
                emission = None
            principled_bsdf = nw.new_node(
                Nodes.PrincipledBSDF,
                input_kwargs={
                    "Base Color": color,
                    "Roughness": roughness,
                    "Metallic": uniform(0, 0.5),
                    "Specular IOR Level": uniform(0, 0.2),
                    "Emission Color": emission,
                    "Emission Strength": self.emission,
                },
            )
            nw.new_node(Nodes.MaterialOutput, input_kwargs={"Surface": principled_bsdf})

        return shader_text

    def apply(self, obj, selection=None, bbox=(0, 1, 0, 1), **kwargs):
        common.apply(obj, self.make_shader_func(bbox), selection, **kwargs)


def apply(
    obj, selection=None, bbox=(0, 1, 0, 1), has_barcode=True, emission=0, **kwargs
):
    Text(np.random.randint(1e5), has_barcode, emission).apply(
        obj, selection, bbox, **kwargs
    )


def make_sphere():
    obj = new_plane()
    obj.rotation_euler[0] = np.pi / 2
    butil.apply_transform(obj)
    compute_uv_direction(obj, "x", "z")
    return obj

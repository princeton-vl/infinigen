# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from PIL import Image

from infinigen.core.constraints.example_solver.room import GraphMaker

# noinspection PyUnresolvedReferences
from infinigen.core.util.math import FixedSeed
from infinigen_examples.generate_individual_assets import make_args


def build_scene(idx, path):
    with FixedSeed(idx):
        factory = GraphMaker(idx)
        graph = factory.make_graph(idx)
        factory.draw(graph)
        (path / "images").mkdir(exist_ok=True)
        imgpath = path / f"images/image_{idx:03d}.png"
        plt.savefig(imgpath)
        plt.clf()


def make_grid(args, path, n):
    files = []
    for filename in sorted(os.listdir(f"{path}/images")):
        if filename.endswith(".png"):
            files.append(f"{path}/images/{filename}")
    files = files[:n]
    if len(files) == 0:
        print("No images found")
        return
    with Image.open(files[0]) as i:
        x, y = i.size
    sz_x = list(
        sorted(
            range(1, n + 1), key=lambda x: abs(math.ceil(n / x) / x - args.best_ratio)
        )
    )[0]
    sz_y = math.ceil(n / sz_x)
    img = Image.new("RGBA", (sz_x * x, sz_y * y))
    for idx, file in enumerate(files):
        with Image.open(file) as i:
            img.paste(i, (idx % sz_x * x, idx // sz_x * y))
    img.save(f"{path}/grid.png")


def main(args):
    path = Path(os.getcwd()) / "outputs"
    path.mkdir(exist_ok=True)
    fac_path = path / GraphMaker.__name__
    if fac_path.exists() and args.skip_existing:
        return
    fac_path.mkdir(exist_ok=True)
    n_images = args.n_images
    for idx in range(args.n_images):
        fac_path.mkdir(exist_ok=True)
        build_scene(idx, fac_path)

    make_grid(args, fac_path, n_images)


if __name__ == "__main__":
    args = make_args()
    args.no_mod = args.no_mod or args.fire
    with FixedSeed(1):
        main(args)

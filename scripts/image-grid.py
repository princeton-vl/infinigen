#!/usr/bin/env -S uv run --no-sync python
# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import argparse
import subprocess
import sys

import imageio.v3 as iio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "tile", help="Grid dimensions, e.g. '4x' for 4 columns or 'x3' for 3 rows"
)
parser.add_argument("output", help="Output image path")
parser.add_argument("paths", nargs="*", help="Image paths (also accepts piped stdin)")
parser.add_argument("-o", "--open", action="store_true", help="Open result")
args = parser.parse_args()

paths = list(args.paths)
if not sys.stdin.isatty():
    paths.extend(line.strip() for line in sys.stdin if line.strip())
paths = sorted(paths)
if not paths:
    sys.exit("No input paths. Provide paths as arguments or pipe them via stdin.")

images = [iio.imread(p) for p in paths]

if args.tile.startswith("x"):
    rows = int(args.tile[1:])
    cols = (len(images) + rows - 1) // rows
elif args.tile.endswith("x"):
    cols = int(args.tile[:-1])
    rows = (len(images) + cols - 1) // cols
else:
    sys.exit("Tile must be like '4x' or 'x3'")

while len(images) < rows * cols:
    images.append(np.zeros_like(images[0]))

grid_rows = []
for r in range(rows):
    row_imgs = images[r * cols : (r + 1) * cols]
    grid_rows.append(np.concatenate(row_imgs, axis=1))
grid = np.concatenate(grid_rows, axis=0)

iio.imwrite(args.output, grid)

if args.open:
    subprocess.run(["open", args.output])

# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

import argparse
import json
from pathlib import Path

import numpy as np
from einops import einsum
from imageio.v3 import imread, imwrite
from numpy.linalg import inv


def transform(T, p):
    assert T.shape == (4,4)
    p = T[:3,:3] @ p
    return p + T[:3, [3]]

def from_homog(x):
    return x[:-1] / x[[-1]]

def unproject(depth, K):
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    img_coords = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float64)
    return einsum(depth, img_coords, inv(K), 'H W, H W j, i j -> H W i')

def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('frame', type=int)
    parser.add_argument('--output', type=Path, default=Path("testbed"))
    args = parser.parse_args()

    folder_data = json.loads((args.folder / "summary.json").read_text())

    depth_paths = folder_data["Depth"]['npy']["00"]["00"]
    image_paths = folder_data["Image"]['png']["00"]["00"]
    Ks = folder_data["Camera Intrinsics"]['npy']["00"]["00"]

    frame = f"{args.frame:04d}"

    image = imread(args.folder / image_paths[frame])
    depth = np.load(args.folder / depth_paths[frame])
    print(depth_paths)
    K1 = np.load(args.folder / Ks[frame])
    cam_coords = unproject(depth, K1)

    cam_coords = cam_coords * np.array([1., -1., -1])

    mask = ~np.isinf(depth)
    depth[~mask] = -1

    vy = normalize(cam_coords[1:,1:] - cam_coords[:-1,1:])
    vx = normalize(cam_coords[1:,1:] - cam_coords[1:,:-1])
    cross_prod = np.cross(vy, vx)
    normals = normalize(cross_prod)
    print(cross_prod.shape, mask.shape)
    normals[~mask[1:,1:]] = 0

    normals_color = np.round((normals + 1) * (255/2)).astype(np.uint8)

    imwrite(args.output / "A.png", image)
    print(f'Wrote {args.output / "A.png"}')
    imwrite(args.output / "B.png", normals_color)
    print(f'Wrote {args.output / "B.png"}')
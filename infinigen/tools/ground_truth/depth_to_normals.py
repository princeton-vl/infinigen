# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

import argparse
import shutil
from pathlib import Path

import cv2
import imagesize
import numpy as np
from einops import einsum
from imageio.v3 import imread, imwrite
from numpy.linalg import inv

from ..dataset_loader import get_frame_path

"""
Usage: python -m tools.ground_truth.depth_to_normals <scene-folder> <frame-index>
Output:
- testbed
    - A.png # Original image
    - B.png # Surface normals from depth + finite-difference
    - C.png # Surface normals from geometry
"""

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

    depth_path = get_frame_path(args.folder, 0, args.frame, 'Depth_npy')
    normal_path = get_frame_path(args.folder, 0, args.frame, 'SurfaceNormal_png')
    image_path = get_frame_path(args.folder, 0, args.frame, 'Image_png')
    camview_path = get_frame_path(args.folder, 0, args.frame, 'camview_npz')
    assert depth_path.exists()
    assert image_path.exists()
    assert camview_path.exists()
    assert normal_path.exists()

    image = imread(image_path)
    depth = np.load(depth_path)
    K = np.load(camview_path)['K']
    cam_coords = unproject(depth, K)

    cam_coords = cam_coords * np.array([1., -1., -1])

    mask = ~np.isinf(depth)
    depth[~mask] = -1

    vy = normalize(cam_coords[1:,1:] - cam_coords[:-1,1:])
    vx = normalize(cam_coords[1:,1:] - cam_coords[1:,:-1])
    cross_prod = np.cross(vy, vx)
    normals = normalize(cross_prod)
    normals[~mask[1:,1:]] = 0

    normals_color = np.round((normals + 1) * (255/2)).astype(np.uint8)
    normals_color = cv2.resize(normals_color, imagesize.get(normal_path))

    imwrite(args.output / "A.png", image)
    print(f'Wrote {args.output / "A.png"}')
    imwrite(args.output / "B.png", normals_color)
    print(f'Wrote {args.output / "B.png"}')
    shutil.copyfile(normal_path, args.output / "C.png")
    print(f'Wrote {args.output / "C.png"}')
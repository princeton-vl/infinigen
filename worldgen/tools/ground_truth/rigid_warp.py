# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

import argparse
from pathlib import Path

import cv2
import numpy as np
from einops import einsum
from imageio.v3 import imread, imwrite
from numpy.linalg import inv

from ..dataset_loader import get_frame_path

"""
Usage: python -m tools.ground_truth.rigid_warp <scene-folder> <frame-index-i> <frame-index-j>
Output:
- testbed
    - A.png # Image at frame i
    - B.png # Image at frame j, warped to i
    - C.png # Image at frame j
"""

def transform(T, p):
    assert T.shape == (4,4)
    return einsum(p, T[:3,:3], 'H W j, i j -> H W i') + T[:3, 3]

def from_homog(x):
    return x[...,:-1] / x[...,[-1]]

def reproject(depth1, pose1, pose2, K1, K2):
    H, W = depth1.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    img_1_coords = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float64)
    cam1_coords = einsum(depth1, img_1_coords, inv(K1), 'H W, H W j, i j -> H W i')
    rel_pose = inv(pose2) @ pose1
    cam2_coords = transform(rel_pose, cam1_coords)
    return from_homog(einsum(cam2_coords, K2, 'H W j, i j -> H W i'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('frame_1', type=int)
    parser.add_argument('frame_2', type=int)
    parser.add_argument('--output', type=Path, default=Path("testbed"))
    args = parser.parse_args()

    depth_path = get_frame_path(args.folder, 0, args.frame_1, 'Depth_npy')
    image1_path = get_frame_path(args.folder, 0, args.frame_1, 'Image_png')
    image2_path = get_frame_path(args.folder, 0, args.frame_2, 'Image_png')
    camview1_path = get_frame_path(args.folder, 0, args.frame_1, 'camview_npz')
    camview2_path = get_frame_path(args.folder, 0, args.frame_2, 'camview_npz')

    image2 = imread(image2_path)
    image1 = imread(image1_path)
    depth1 = np.load(depth_path)
    pose1 = np.load(camview1_path)['T']
    pose2 = np.load(camview2_path)['T']
    K1 = np.load(camview1_path)['K']
    K2 = np.load(camview2_path)['K']

    H, W, _ = image1.shape
    depth1 = cv2.resize(np.load(depth_path), dsize=(W, H), interpolation=cv2.INTER_LINEAR)

    img2_coords = reproject(depth1, pose1, pose2, K1, K2)

    warped_image = cv2.remap(image2, img2_coords.astype(np.float32), None, interpolation=cv2.INTER_LINEAR)

    args.output.mkdir(exist_ok=True)
    imwrite(args.output / "A.png", image1)
    print(f'Wrote {args.output / "A.png"}')
    imwrite(args.output / "C.png", image2)
    print(f'Wrote {args.output / "C.png"}')
    imwrite(args.output / "B.png", warped_image)
    print(f'Wrote {args.output / "B.png"}')
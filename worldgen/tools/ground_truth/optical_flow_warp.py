# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

import argparse
from pathlib import Path

import cv2
import numpy as np
from imageio.v3 import imread, imwrite

"""
Usage: python -m tools.ground_truth.rigid_warp <scene-folder> <frame-index-i>
Output:
- testbed
    - A.png # Image at frame i
    - B.png # Image at frame i+1, warped to i
    - C.png # Image at frame i+1
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('frame', type=int)
    parser.add_argument('--output', type=Path, default=Path("testbed"))
    args = parser.parse_args()
    frame1 = f"{args.frame:04d}"
    frame2 = f"{int(frame1)+1:04d}"
    flow3d_path = args.folder / "frames" / "Flow3D" / "camera_0" / f"Flow3D_0_0_{frame1}_0.npy"
    image1_path = args.folder / "frames" / "Image" / "camera_0" / f"Image_0_0_{frame1}_0.png"
    image2_path = args.folder / "frames" / "Image" / "camera_0" / f"Image_0_0_{frame2}_0.png"
    assert flow3d_path.exists()
    assert image1_path.exists()
    assert image2_path.exists()

    def warp_image_with_flow(image2, flow3d):
        H, W, _ = image2.shape
        flow2d = cv2.resize(flow3d, dsize=(W, H), interpolation=cv2.INTER_LINEAR)[...,:2]
        new_coords = flow2d + np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing='xy'), axis=-1)
        warmped_image2 = cv2.remap(image2, new_coords.astype(np.float32), None, interpolation=cv2.INTER_LINEAR)
        return warmped_image2

    image2 = imread(image2_path)
    image1 = imread(image1_path)
    H, W, _ = image1.shape
    shape = (W, H)
    img_to_gt_ratio = 3840 / W

    flow2d_resized = cv2.resize(np.load(flow3d_path), dsize=shape, interpolation=cv2.INTER_LINEAR)[...,:2] / img_to_gt_ratio

    warped_image = warp_image_with_flow(image2, flow2d_resized)

    args.output.mkdir(exist_ok=True)
    imwrite(args.output / "A.png", image1)
    print(f'Wrote {args.output / "A.png"}')
    imwrite(args.output / "C.png", image2)
    print(f'Wrote {args.output / "C.png"}')
    imwrite(args.output / "B.png", warped_image)
    print(f'Wrote {args.output / "B.png"}')
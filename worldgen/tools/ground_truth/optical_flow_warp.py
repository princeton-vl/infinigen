import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from imageio.v3 import imread, imwrite

if __name__ == "__main__":

    import os,sys
    sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('frame', type=int)
    parser.add_argument('--output', type=Path, default=Path("testbed"))
    args = parser.parse_args()

    folder_data = json.loads((args.folder / "summary.json").read_text())
    flow3d_paths = folder_data["Flow3D"]['npy']["00"]["00"]
    image_paths = folder_data["Image"]['png']["00"]["00"]

    def warp_image_with_flow(image2, flow3d):
        H, W, _ = image2.shape
        flow2d = cv2.resize(flow3d, dsize=(W, H), interpolation=cv2.INTER_LINEAR)[...,:2]
        new_coords = flow2d + np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing='xy'), axis=-1)
        warmped_image2 = cv2.remap(image2, new_coords.astype(np.float32), None, interpolation=cv2.INTER_LINEAR)
        return warmped_image2

    frame1 = f"{args.frame:04d}"
    frame2 = f"{int(frame1)+1:04d}"

    image2 = imread(args.folder / image_paths[frame2])
    image1 = imread(args.folder / image_paths[frame1])
    H, W, _ = image1.shape
    shape = (W, H)
    img_to_gt_ratio = 3840 / W

    flow2d_resized = cv2.resize(np.load(args.folder / flow3d_paths[frame1]), dsize=shape, interpolation=cv2.INTER_LINEAR)[...,:2] / img_to_gt_ratio

    warped_image = warp_image_with_flow(image2, flow2d_resized)

    args.output.mkdir(exist_ok=True)
    imwrite(args.output / "A.png", image1)
    print(f'Wrote {args.output / "A.png"}')
    imwrite(args.output / "C.png", image2)
    print(f'Wrote {args.output / "C.png"}')
    imwrite(args.output / "B.png", warped_image)
    print(f'Wrote {args.output / "B.png"}')
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from imageio.v3 import imread, imwrite
from numpy.linalg import inv

def transform(T, p):
    assert T.shape == (4,4)
    p = T[:3,:3] @ p
    return p + T[:3, [3]]

def from_homog(x):
    return x[:-1] / x[[-1]]

def reproject(depth1, pose1, pose2, K1, K2):
    H, W = depth1.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    img_1_coords = np.stack((x, y, np.ones_like(x)), axis=-1).reshape((H*W, 3)).T.astype(np.float64)
    rel_pose = inv(pose2) @ pose1
    cam1_coords = depth1.reshape((1, H*W)) * (inv(K1) @ img_1_coords)
    cam2_coords = transform(rel_pose, cam1_coords)
    img2_coords = from_homog(K2 @ cam2_coords)
    return img2_coords.T.reshape((H, W, 2))


if __name__ == "__main__":

    import os,sys
    sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('frame_1', type=int)
    parser.add_argument('frame_2', type=int)
    parser.add_argument('--output', type=Path, default=Path("testbed"))
    args = parser.parse_args()

    folder_data = json.loads((args.folder / "summary.json").read_text())

    depth_paths = folder_data["Depth"]['npy']["00"]["00"]
    image_paths = folder_data["Image"]['png']["00"]["00"]
    Ks = folder_data["Camera Intrinsics"]['npy']["00"]["00"]
    Ts = folder_data["Camera Pose"]['npy']["00"]["00"]

    frame1 = f"{args.frame_1:04d}"
    frame2 = f"{args.frame_2:04d}"

    image2 = imread(args.folder / image_paths[frame2])
    image1 = imread(args.folder / image_paths[frame1])
    H, W, _ = image1.shape
    depth1 = np.load(args.folder / depth_paths[frame1])
    pose1 = np.load(args.folder / Ts[frame1])
    pose2 = np.load(args.folder / Ts[frame2])
    K1 = np.load(args.folder / Ks[frame1])
    K2 = np.load(args.folder / Ks[frame2])

    H, W, _ = image1.shape
    shape = (W, H)
    depth1 = cv2.resize(np.load(args.folder / depth_paths[frame1]), dsize=shape, interpolation=cv2.INTER_LINEAR)

    img2_coords = reproject(depth1, pose1, pose2, K1, K2)

    warped_image = cv2.remap(image2, img2_coords.astype(np.float32), None, interpolation=cv2.INTER_LINEAR)

    args.output.mkdir(exist_ok=True)
    imwrite(args.output / "A.png", image1)
    print(f'Wrote {args.output / "A.png"}')
    imwrite(args.output / "C.png", image2)
    print(f'Wrote {args.output / "C.png"}')
    imwrite(args.output / "B.png", warped_image)
    print(f'Wrote {args.output / "B.png"}')
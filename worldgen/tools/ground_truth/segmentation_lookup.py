import argparse
import colorsys
import json
from itertools import chain, product
from pathlib import Path

import cv2
import numpy as np
import torch
import torch_scatter
from einops import asnumpy, repeat
from imageio.v3 import imread, imwrite
from tqdm import tqdm

import os,sys
sys.path.append(os.getcwd())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('frame', type=int)
    parser.add_argument('--query', type=str)
    parser.add_argument('--boxes', action='store_true')
    parser.add_argument('--output', type=Path, default=Path("testbed"))
    args = parser.parse_args()

    folder_data = json.loads((args.folder / "summary.json").read_text())

    image = imread(args.folder / folder_data["Image"]['png']["00"]["00"][f"{args.frame:04d}"])
    H, W, _ = image.shape

    tag_mask = np.load(args.folder / folder_data["TagSegmentation"]['npy']["00"]["00"][f"{args.frame:04d}"])
    tag_mask = cv2.resize(tag_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

    instance_segmentation_mask = np.load(args.folder / folder_data["InstanceSegmentation"]['npy']["00"]["00"][f"{args.frame:04d}"])
    instance_segmentation_mask = cv2.resize(instance_segmentation_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

    object_segmentation_mask = np.load(args.folder / folder_data["ObjectSegmentation"]['npy']["00"]["00"][f"{args.frame:04d}"])
    object_segmentation_mask = cv2.resize(object_segmentation_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

    combined_mask = np.stack((object_segmentation_mask, instance_segmentation_mask), axis=-1).reshape((-1, 2))

    uniq, indices = np.unique(combined_mask, return_inverse=True, axis=0)
    random_states = [np.random.RandomState(e+(2**31)+6) for e in uniq]
    unique_colors = (np.asarray([colorsys.hsv_to_rgb(s.uniform(0, 1), s.uniform(0.1, 1), 1) for s in random_states]) * 255).astype(np.uint8)

    if args.boxes:
        xy = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'), dim=-1).view(W*H, 2)
        mins, _ = map(asnumpy, torch_scatter.scatter_min(xy, torch.as_tensor(indices), dim=0))
        maxs, _ = map(asnumpy, torch_scatter.scatter_max(xy, torch.as_tensor(indices), dim=0))
        hilighted = np.copy(image)
        boxes = []
        for (x_min, y_min), (x_max, y_max), color in zip(mins, maxs, unique_colors):
            boxes.append((x_min, x_max, y_min, y_max, color.tolist(), set()))
        for (i, j), idx, tag in tqdm(zip(product(range(H), range(W)), indices, tag_mask.flatten())):
            boxes[idx][-1].add(tag)
    else:
        colors_for_instances = unique_colors[indices].reshape((H, W, 3))
        hilighted = np.zeros((H, W, 3), dtype=np.uint8)

    tag_lookup = json.loads((args.folder / folder_data["Mask Tags"][f"{args.frame:04d}"]).read_text())
    tag_lookup_rev = {v:k for k,v in tag_lookup.items()}
    tags_in_this_image = set(chain.from_iterable(tag_lookup_rev[e].split('.') for e in np.unique(tag_mask) if e > 0))

    if args.query is None:
        print('`--query` not specified. Choices are:')
        for q in tags_in_this_image:
            print(f"- {q}")
    elif args.query not in tags_in_this_image:
        print(f'"{args.query}" doesn\'t match any tag in this image. Choices are:')
        for q in tags_in_this_image:
            print(f"- {q}")
    else:
        for k, v in tag_lookup.items():
            if args.query in k.split('.'):
                if args.boxes:
                    for x_min, x_max, y_min, y_max, color, tags in boxes:
                        if v in tags:
                            points = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
                            for i in range(4):
                                hilighted = cv2.line(hilighted, points[i], points[(i+1)%4], color=color, thickness=2)
                else:
                    m = repeat(tag_mask == v, 'H W -> H W 3')
                    hilighted[m] = colors_for_instances[m]



        args.output.mkdir(exist_ok=True)
        imwrite(args.output / "A.png", image)
        print(f'Wrote {args.output / "A.png"}')
        imwrite(args.output / "B.png", hilighted)
        print(f'Wrote {args.output / "B.png"}')
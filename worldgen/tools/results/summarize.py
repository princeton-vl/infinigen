# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

import argparse
import json
import multiprocessing as mp
import os
import re
from collections import defaultdict
from itertools import chain
from pathlib import Path

import cv2
import flow_vis  # run pip install flow_vis
import numpy as np
import skimage.measure
from einops import repeat
from imageio.v3 import imread
from matplotlib import pyplot as plt
from tqdm import tqdm


def make_defaultdict(inner):
    return (lambda: defaultdict(inner))

def parse_mask_tag_jsons(base_folder):
    for file_path in base_folder.rglob('MaskTag.json'):
        if match := re.fullmatch("fine_([0-9])_([0-9])_([0-9]{4})_([0-9])", file_path.parent.name):
            _, _, frame_str, _ = match.groups()
            yield (int(frame_str), file_path)
    for file_path in base_folder.rglob('MaskTag.json'):
        if match := re.fullmatch("fine.*", file_path.parent.name):
            yield (0, file_path)

def summarize_folder(base_folder):
    base_folder = Path(base_folder)
    output = defaultdict(make_defaultdict(make_defaultdict(make_defaultdict(dict))))
    max_frame = -1
    for file_path in base_folder.rglob('*'):
        if (not file_path.is_file) or ("saved_mesh" in file_path.parts):
            continue

        if match := re.fullmatch("(.*)_([0-9]{4})_([0-9]{2})_([0-9]{2})\.([a-z]+)", file_path.name):
            data_type, frame_str, rig, subcam, suffix = match.groups()
            output[data_type][suffix][rig][subcam][frame_str] = str(file_path.relative_to(base_folder))
            max_frame = max(max_frame, int(frame_str))

    # Rename keys
    output["Camera Pose"] = output.pop("T")
    output["Camera Intrinsics"] = output.pop("K")

    mask_tag_jsons = sorted(parse_mask_tag_jsons(base_folder))
    for frame in range(1, max_frame+1):
        _, closest = max((f,p) for f,p in mask_tag_jsons if (int(f) <= frame))
        output["Mask Tags"][f"{frame:04d}"] = str(closest.relative_to(base_folder))

    output["stats"] = {"Max Frame": max_frame}

    (base_folder / "summary.json").write_text(json.dumps(output, indent=4))
    return base_folder / "summary.json"

def what_is_missing(summary):
    max_frame = summary["stats"]["Max Frame"]
    all_rigs = set(chain((summary["SurfaceNormal"]["png"].keys()), (summary["SurfaceNormal"]["png"].keys())))
    all_subcams = set(chain((summary["SurfaceNormal"]["png"]["00"].keys()), (summary["SurfaceNormal"]["png"]["00"].keys())))
    logs = []
    for rig in all_rigs:
        for subcam in all_subcams:
            gt_frame_keys = set(summary["SurfaceNormal"]["png"][rig][subcam].keys())
            image_frame_keys = set(summary["SurfaceNormal"]["png"][rig][subcam].keys())
            for frame in range(1, max_frame):
                if f"{frame:04d}" not in gt_frame_keys:
                    logs.append(f"Ground truth is missing for frame {frame}")
                if f"{frame:04d}" not in image_frame_keys:
                    logs.append(f"Image is missing for frame {frame}")
    return logs

def process_flow_frame(path, shape):
    flow3d = np.load(path)
    flow2d_resized = cv2.resize(flow3d, dsize=shape, interpolation=cv2.INTER_LINEAR)[...,:2]
    flow2d_resized[(np.abs(flow2d_resized) > 1e3) | np.isnan(flow2d_resized)] = -1
    flow_color = flow_vis.flow_to_color(flow2d_resized, convert_to_bgr=False)
    return flow_color

def process_depth_frame(path, shape):
    depth = np.load(path)
    return cv2.resize(depth, dsize=shape, interpolation=cv2.INTER_LINEAR)

def process_mask(path, shape):
    mask = imread(path)
    H, W = mask.shape
    scale = (W // shape[0], H // shape[1])
    out = skimage.measure.block_reduce(mask, scale, np.max)
    return repeat(out, 'H W -> H W 3')

def frames_to_video(file_path, frames: list, fps=24):
    assert Path(file_path).suffix == '.avi'
    H, W, _ = frames[0].shape
    video = cv2.VideoWriter(str(file_path), cv2.VideoWriter_fourcc(*'DIVX'),frameSize=(W, H), fps=fps)
    for img in frames:
        video.write(img)
    video.release()
    assert os.path.exists(file_path)
    print(f"Wrote {file_path}")

def depth_to_jet(depth, scale_vmin=1.0):
    valid = (depth > 1e-3) & (depth < 1e4)
    vmin = depth[valid].min() * scale_vmin
    vmax = depth[valid].max()
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    depth = cmap(norm(depth))
    depth[~valid] = 1
    return np.ascontiguousarray(depth[...,:3] * 255, dtype=np.uint8)

def process_scene_folder(folder, preview):
    summary_json = summarize_folder(folder)
    folder_data = json.loads(summary_json.read_text())

    missing = what_is_missing(folder_data)
    print("\n".join(missing))

    if not preview:
        return

    depth_paths = folder_data["Depth"]['npy']["00"]["00"]
    flow3d_paths = folder_data["Flow3D"]['npy']["00"]["00"]
    image_paths = folder_data["Image"]['png']["00"]["00"]
    occlusion_boundary_paths = folder_data["OcclusionBoundaries"]['png']["00"]["00"]
    flow_mask_paths = folder_data["Flow3D_Mask"]['png']["00"]["00"]
    all_flow_frames = sorted(image_paths.keys())

    shape = (1280, 720)
    with mp.Pool() as pool:
        all_flow_frames = pool.starmap(process_flow_frame, tqdm([(folder / path, shape) for _, path in sorted(flow3d_paths.items())]))
        all_depth_frames = pool.starmap(process_depth_frame, tqdm([(folder / path, shape) for _, path in sorted(depth_paths.items())]))
        all_occlusion_frames = pool.starmap(process_mask, tqdm([(folder / path, shape) for _, path in sorted(occlusion_boundary_paths.items())]))
        all_flow_mask_frames = pool.starmap(process_mask, tqdm([(folder / path, shape) for _, path in sorted(flow_mask_paths.items())]))

    previews: Path = folder / "previews"
    previews.mkdir(exist_ok=True)
    frames_to_video(previews / 'occlusion_boundaries.avi', all_occlusion_frames)
    frames_to_video(previews / 'flow_mask.avi', all_flow_mask_frames)
    depth_visualization = depth_to_jet(np.asarray(all_depth_frames))
    frames_to_video(previews / 'video_depth.avi', depth_visualization)
    frames_to_video(previews / 'flow_video.avi', all_flow_frames)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()

    process_scene_folder(args.folder, preview=args.preview)

    

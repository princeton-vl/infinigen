# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import argparse
import json
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_frames_dir", type=Path)
    parser.add_argument("point_trajectory_src_frame", type=int)
    args = parser.parse_args()
    assert args.target_frames_dir.exists()
    assert args.target_frames_dir.name.startswith("savemesh_")

    static_mesh_folder = (
        args.target_frames_dir
        / f"frame_{args.point_trajectory_src_frame:04d}"
        / "static_mesh"
    )
    with open(static_mesh_folder / "saved_mesh.json", "r") as f:
        static_json = json.load(f)
    for item in static_json:
        if "filename" in item:
            item["filename"] = "static_" + item["filename"]

    for frame_folder in os.listdir(args.target_frames_dir):
        if not frame_folder.startswith("frame_"):
            continue
        with open(
            args.target_frames_dir / frame_folder / "mesh/saved_mesh.json", "r"
        ) as f:
            current_json = json.load(f)
        current_json = [*current_json, *static_json]
        with open(
            args.target_frames_dir / frame_folder / "mesh/saved_mesh.json", "w"
        ) as f:
            json.dump(current_json, f)
        for npz_path in os.listdir(static_mesh_folder):
            if npz_path.endswith(".npz"):
                os.symlink(
                    static_mesh_folder / npz_path,
                    args.target_frames_dir
                    / frame_folder
                    / "mesh"
                    / ("static_" + npz_path),
                )

# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

import argparse
import os
from pathlib import Path
import platform
import tarfile
import json

import time
from datetime import datetime
import gin
from tqdm import tqdm
import subprocess
from shutil import which, copyfile

from . import smb_client, cleanup

GDRIVE_NAME = None

def rclone_upload_file(src_file, dst_folder):

    if GDRIVE_NAME is None:
        raise ValueError(f'Please specify GDRIVE_NAME')

    assert os.path.exists(src_file), src_file
    cmd = f"{which('rclone')} copy -P {src_file} {GDRIVE_NAME}:{dst_folder}"
    subprocess.check_output(cmd.split())
    print(f"Uploaded {src_file}")

def get_commit_hash():
    git = which('git')
    if git is None:
        return None
    cmd = f"{git} rev-parse HEAD"
    return subprocess.check_output(cmd.split()).decode().strip()

# DO NOT make gin.configurable
# this function gets submitted via pickle in some settings, and gin args are not preserved
def reorganize_before_upload(parent_folder):

    seed = parent_folder.name
    tmpdir = (parent_folder / "tmp" / seed)
    log_dir = tmpdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    frames_folders = list(sorted(parent_folder.glob("frames*")))
    for idx, frames_folder in enumerate(frames_folders):

        subfolder_name = f"resample_{idx}" if (idx > 0) else "original"
        subfolder = tmpdir / subfolder_name
        info_dir = subfolder / "info"
        info_dir.mkdir(parents=True, exist_ok=True)
        ground_truth_dir = subfolder / "ground_truth"
        ground_truth_dir.mkdir(parents=True, exist_ok=True)
        color_passes_dir = subfolder / "color_passes"
        color_passes_dir.mkdir(parents=True, exist_ok=True)
        copyfile(frames_folder / "object_tree.json", info_dir / "object_tree.json")
        for file in tqdm(sorted(frames_folder.rglob("*.png")), desc="Upload .png", disable=True):
            copyfile(file, color_passes_dir / file.name)
        for file in tqdm(sorted(frames_folder.rglob("*.exr")), desc="Upload .exr", disable=True):
            copyfile(file, ground_truth_dir / file.name)
        for file in tqdm(sorted(frames_folder.rglob("*.npy")), desc="Upload .npy", disable=True):
            copyfile(file, info_dir / file.name)
    for ext in ["out", "err"]:
        for file in tqdm(sorted((parent_folder / "logs").rglob(f"*.{ext}")), desc=f"Upload .{ext}", disable=True):
            if not file.is_symlink():
                copyfile(file, log_dir / (file.name + ".txt"))
    for file in tqdm(sorted((parent_folder / "logs").rglob(f"operative_gin_*")), desc=f"operative_gins", disable=True):
        copyfile(file, log_dir / file.name)
    
    copyfile(parent_folder / "run_pipeline.sh", log_dir / "run_pipeline.sh")  

def write_metadata(parent_folder, seed, all_images):
    
    try:
        version = (parent_folder / "coarse" / "version.txt").read_text().splitlines()[0]
    except FileNotFoundError:
        version = None

    metadata = {
        'original_directory': str(parent_folder.resolve()),
        'user': os.environ['USER'],
        'node': platform.node().split('.')[0],
        'timestamp': time.time(),
        'datetime': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        'version': version,
        'commit': get_commit_hash(),
        'n_frames': len(all_images)
    }

    metadata_path = parent_folder/f'{seed}_metadata.json'
    with metadata_path.open('w') as f:
        json.dump(metadata, f, indent=4)

    return metadata_path

def write_thumbnail(parent_folder, seed, all_images):
    if len(all_images) > 0:
        thumb_path = parent_folder/f'{seed}_thumbnail.png'
        copyfile(all_images[0], thumb_path)
    else:
        thumb_path = None    

    return thumb_path

def create_tarball(parent_folder):
    tar_path = parent_folder.with_suffix('.tar.gz')
    print(f"Performing cleanup and tar to {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(parent_folder, os.path.sep)
    assert tar_path.exists()
    return tar_path

def get_upload_func(method='smbclient'):
    if method == 'rclone':
        return rclone_upload_file
    elif method == 'smbclient':
        return smb_client.upload
    else:
        raise ValueError(f'Unrecognized {method=}')  

def get_upload_destfolder(job_folder):
    return Path('infinigen')/'renders'/job_folder.name

# DO NOT make gin.configurable
# this function gets submitted via pickle in some settings, and gin args are not preserved
def upload_job_folder(
    parent_folder, 
    task_uniqname, 
    dir_prefix_len=0, 
    method='smbclient'
):

    parent_folder = Path(parent_folder)
    seed = parent_folder.name

    upload_func = get_upload_func(method)
    
    upload_dest_folder = get_upload_destfolder(parent_folder.parent)
    if dir_prefix_len > 0:
        upload_dest_folder = upload_dest_folder/parent_folder.name[:dir_prefix_len]

    cleanup.cleanup(parent_folder)
    all_images = sorted(list(parent_folder.rglob("frames*/Image*.png")))

    upload_paths = [
        write_thumbnail(parent_folder, seed, all_images),
        write_metadata(parent_folder, seed, all_images),
        create_tarball(parent_folder)
    ]
    
    for f in upload_paths:
        if f is None:
            continue
        upload_func(f, upload_dest_folder)
        f.unlink()

    (parent_folder / "logs" / f"FINISH_{task_uniqname}").touch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('parent_folder', type=Path)
    parser.add_argument('task_uniqname', type=str)
    args = parser.parse_args()

    upload_job_folder(args.parent_folder, args.task_uniqname)
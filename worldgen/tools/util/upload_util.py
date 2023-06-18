# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson
# Date Signed: May 2 2023

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

from . import smb_client

GDRIVE_NAME = None

def rclone_upload_file(src_file, dst_folder):

    if GDRIVE_NAME is None:
        raise ValueError(f'Please specify GDRIVE_NAME')

    assert os.path.exists(src_file), src_file
    cmd = f"{which('rclone')} copy -P {src_file} {GDRIVE_NAME}:{dst_folder}"
    subprocess.check_output(cmd.split())
    print(f"Uploaded {src_file}")

# DO NOT make gin.configurable
# this function gets submitted via pickle in some settings, and gin args are not preserved
def upload_folder(folder, upload_dest_folder, method, metadata=None, **kwargs):

    upload_info_path = folder / f"{folder.name}.json"
    upload_info = {
        'user': os.environ['USER'],
        'node': platform.node().split('.')[0],
        'timestamp': time.time(),
        'datetime': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        **(metadata if metadata is not None  else {})
    }
    with upload_info_path.open('w') as f:
        json.dump(upload_info, f, indent=4)

    with tarfile.open(folder.with_suffix('.tar.gz'), "w:gz") as tar:
        tar.add(folder, os.path.sep)
    assert folder.with_suffix('.tar.gz').exists()

    if method == 'rclone':
        upload_func = rclone_upload_file
    elif method == 'smbclient':
        upload_func = smb_client.upload
    else:
        raise ValueError(f'Unrecognized {method=}')

    upload_func(folder.with_suffix('.tar.gz'), upload_dest_folder, **kwargs)  
    upload_func(upload_info_path, upload_dest_folder, **kwargs)

def upload_job_folder(parent_folder, task_uniqname, dir_prefix_len=3, method='smbclient', **kwargs):

    parent_folder = Path(parent_folder)

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

    version = (parent_folder / "fine" / "version.txt").read_text().splitlines()[0]
    upload_dest_folder = Path('infinigen')/'renders'/version
    if dir_prefix_len != 0:
        upload_dest_folder = upload_dest_folder/seed[:dir_prefix_len]

    metadata = {
        'n_frames_folders': len(frames_folders),
        'original_directory': str(parent_folder.resolve())
    }

    upload_folder(tmpdir, upload_dest_folder, method=method, metadata=metadata, **kwargs)

    (parent_folder / "logs" / f"FINISH_{task_uniqname}").touch()

def test():
    import manage_datagen_jobs
    find_gin = lambda n: os.path.join("tools", "pipeline_configs", f"{n}.gin")
    configs = [find_gin(n) for n in ['andromeda', 'smb_login']]
    gin.parse_config_files_and_bindings(configs, bindings=[])
    upload_folder(Path('outputs/23_01_25_allcs/a4b66f1'), 'upload', dir_prefix_len=3, method='smbclient')

if __name__ == "__main__":
    test()

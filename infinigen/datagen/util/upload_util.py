# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

import argparse
import os
from pathlib import Path
import platform
import tarfile
import json
import itertools

import time
from datetime import datetime
import gin
from tqdm import tqdm
import subprocess
import shutil

from infinigen.core.util.logging import Suppress

from . import smb_client, cleanup

RCLONE_PREFIX_ENVVAR = "INFINIGEN_RCLONE_PREFIX"

UPLOAD_MANIFEST = [
    ('frames*/*', 'KEEP'),
    ('logs/*', 'KEEP'),
    ('fine/scene.blend', 'KEEP'),
    ('run_pipeline.sh', 'KEEP'),

    ('coarse/*.txt', 'KEEP'),
    ('coarse/*.csv', 'KEEP'),
    ('coarse/*.json', 'KEEP'),
    ('fine*/*.txt', 'KEEP'),
    ('fine*/*.csv', 'KEEP'),
    ('fine*/*.json', 'KEEP'),

    ('savemesh*', 'DELETE'),
    ('coarse/assets', 'DELETE'),
    ('coarse/scene.blend*', 'DELETE'),
    ('fine*/assets', 'DELETE'),
    ('tmp', 'DELETE'),
    ('*/*.b_displacement.npy', 'DELETE'),

    # These two only show up during/after upload, we just specify them to prevent an error
    ('*_thumbnail.png', 'KEEP'),
    ('*_metadata.json', 'KEEP'),
]

def check_files_covered(scene_folder, manifest):

    covered = set()

    for glob, _ in UPLOAD_MANIFEST:
        covered |= set(scene_folder.glob(glob))

    extant = set(scene_folder.glob('*'))

    not_covered = extant - covered

    not_covered = {p for p in not_covered if not p.is_dir()}

    if len(not_covered) == 0:
        return
    
    raise ValueError(
        f'{scene_folder=} had {not_covered=}. Please modify {__file__}.UPLOAD_MANIFEST'
        ' to explicitly say whether you want these files to be deleted or included in the final tarball'
    )

def apply_manifest_cleanup(scene_folder, manifest):
    
    check_files_covered(scene_folder, manifest)

    keep = set()
    delete = set()

    for glob, action in manifest:

        affected = set()
        for p in scene_folder.glob(glob):
            affected.add(p)
            if p.is_dir():
                affected |= set(p.rglob("*"))

        print(f'{glob=} {action=} matched {len(affected)=}')

        if action == 'KEEP':
            keep |= affected
        elif action == 'KEEP_MANDATORY':
            if len(affected) == 0:
                raise ValueError(f'In {apply_manifest_cleanup.__name__} {glob=} had {action=} but failed to match any files')
            keep |= affected
        elif action == 'DELETE':
            delete |= set(affected) - keep
        else:
            raise ValueError(f'Unrecognized {action=}')

    assert delete.isdisjoint(keep)

    for f in delete:
        if not f.exists():
            continue
        if f.is_symlink() or not f.is_dir():
            f.unlink()
    for f in delete:
        if not f.exists() or not f.is_dir():
            continue
        if len([f1 for f1 in f.rglob('*') if not f.is_dir()]) == 0:
            shutil.rmtree(f)

def rclone_upload_file(src_file, dst_folder):

    prefix = os.environ.get(RCLONE_PREFIX_ENVVAR)
    if prefix is None:
        raise ValueError(f'Please specify envvar {RCLONE_PREFIX_ENVVAR}')
    if ':' not in prefix:
        raise ValueError(f'Rclone prefix must contain ":" to separate remote from path prefix')

    assert os.path.exists(src_file), src_file
    cmd = f"{shutil.which('rclone')} copy -P {src_file} {prefix}{dst_folder}"
    subprocess.check_output(cmd.split())
    print(f"Uploaded {src_file}")

def get_commit_hash():
    git = shutil.which('git')
    if git is None:
        return None
    try:
        with Suppress():
            cmd = f"{git} rev-parse HEAD"
            return subprocess.check_output(cmd.split()).decode().strip()
    except subprocess.CalledProcessError:
        return None

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
        shutil.copyfile(all_images[0], thumb_path)
    else:
        thumb_path = None    

    return thumb_path

def create_tarball(parent_folder):
    tar_path = parent_folder.with_suffix('.tar.gz')
    print(f"Tarring {parent_folder} to {tar_path}")
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

    print(f'Performing cleanup on {parent_folder}')
    apply_manifest_cleanup(parent_folder, UPLOAD_MANIFEST)

    upload_func = get_upload_func(method)
    
    upload_dest_folder = get_upload_destfolder(parent_folder.parent)
    if dir_prefix_len > 0:
        upload_dest_folder = upload_dest_folder/parent_folder.name[:dir_prefix_len]

    all_images = sorted(list(parent_folder.rglob("**/Image*.png")))

    upload_paths = [
        write_thumbnail(parent_folder, seed, all_images),
        write_metadata(parent_folder, seed, all_images),
        create_tarball(parent_folder)
    ]

    orig_fine_path = parent_folder/'fine'/'scene.blend'
    if orig_fine_path.exists():
        dest_fine_path = parent_folder.parent / f'{seed}_fine.blend'
        shutil.move(orig_fine_path, dest_fine_path)
        upload_paths.append(dest_fine_path)
    
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
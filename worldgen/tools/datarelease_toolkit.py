import os
from pathlib import Path
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import subprocess
import shutil
import json
from itertools import product
from copy import copy, deepcopy
import pdb

import imageio
import numpy as np
import cv2

from .util import smb_client
from .states import parse_suffix, get_suffix

from . import torch_dataset

TOOLKIT_VERSION = '0.0.2'

def mapfunc(f, its, n_workers):
    if n_workers == 1:
        return [f(i) for i in its]
    else:
        with Pool(n_workers) as p:
            return list(tqdm(p.imap(f, its), total=len(its)))

def download_except_already_present(smb_path, local_path, min_date_str, n_workers=1, verbose=False):
    
    remote_paths = list(smb_client.listdir(smb_path))
    local_names = set(f.name for f in local_path.iterdir())

    to_download = [f for f in remote_paths if f.name not in local_names]
        
    download_func = partial(
        smb_client.download, 
        dest_folder=local_path,
        verbose=verbose
    )

    mapfunc(download_func, to_download, n_workers=n_workers)

def untar(p):
    command = f'tar -xf {str(p)} -C {str(p.parent)} --one-top-level'
    dest_path = p.parent/(p.name.split('.')[0])
    assert not dest_path.exists()
    print('Untarring', p, ' --> ', dest_path)
    subprocess.run(command, check=True, shell=True)
    return dest_path

def fix_scene_structure(p, n_subcams):

    def parse_pre_1_1_suffix(suffix):
        stem, ext = suffix.split('.')
        parts = stem.split('_')
        assert len(parts) == 3
        return dict(
            frame=int(parts[0]),
            cam_rig=int(parts[1]),
            subcam=int(parts[2]),
            resample=0
        )
    
    def move(a, b):
        assert b.parent.exists()
        print(a, ' --> ', b)
        shutil.move(a, b)

    def postprocess_image(img_path, frame_folder, subcam):
        if img_path.name.endswith('_displacement.npy'):
            img_path.unlink()
            return
        dtype, *parts = img_path.name.split('_')
        rest = '_'.join(parts)

        if len(rest.split('_')) == 3:
            keys = parse_pre_1_1_suffix(rest)
            new_img_name = dtype + get_suffix(keys)
            new_img_path = img_path.parent/(new_img_name + img_path.suffix)
            move(img_path, new_img_path)
            img_path = new_img_path

        img_keys = parse_suffix(img_path)

        if img_keys['subcam'] != subcam:
            keys = parse_suffix(frame_folder)
            keys['subcam'] = img_keys['subcam']
            new_folder_name = 'frames' + get_suffix(keys)
            new_frames_folder = frame_folder.parent/new_folder_name
            new_img_path = new_frames_folder/img_path.name
            move(img_path, new_img_path)
            img_path = new_img_path

    for subcam in range(n_subcams):
        cam_frames_folders = sorted(list(p.glob(f'frames*_{subcam}')))
        for frame_folder in cam_frames_folders:
            for img_path in frame_folder.iterdir():
                postprocess_image(img_path, frame_folder, subcam)
                
def fix_metadata(p, override_version='1.0.4b'):

    metadata_path = p.parent/(p.name+'_metadata.json')
    if not metadata_path.exists():
        print(f'{p} is missing {metadata_path=}')
        return False
    
    with metadata_path.open('r') as f:
        metadata = json.load(f)
    
    if override_version is not None:
        metadata['version'] = override_version

    with metadata_path.open('w') as f:
        json.dump(metadata, f)

def fix_image_resolution(img_path, base_shape):

    dtype, *_ = img_path.name.split('_')

    if dtype == 'Image':
        return

    npy_equivelant = img_path.parent/(img_path.stem + '.npy')
    is_visualization = (img_path.suffix == '.png' and npy_equivelant.exists())
    is_renderpass = (
        img_path.suffix == '.png' and 
        not npy_equivelant.exists() and
        not 'Occlusion' in img_path.name
    )

    single_res_gt = {'Flow', 'SurfaceNormal'}
    if any(x in dtype for x in single_res_gt):
        mult = 1
    elif is_renderpass:
        mult = 1
    else:
        mult = 2
    target_shape = mult * base_shape

    if img_path.name.endswith('png'):
        img = imageio.imread(img_path)
    elif img_path.name.endswith('npy'):
        img = np.load(img_path)
    else:
        return
    curr_shape = np.array(img.shape[:2])
    
    if np.any(target_shape > curr_shape):
        raise ValueError(img_path, curr_shape, target_shape)

    if np.all(target_shape == curr_shape):
        return

    interp_type = 'INTER_LINEAR' if is_renderpass else 'INTER_NEAREST'
    print(img_path.name, curr_shape, '-->', target_shape, 'via', interp_type)
    interp = cv2.resize(img, target_shape[::-1], interpolation=getattr(cv2, interp_type))
    if dtype == 'Flow3DMask':
        interp = interp.max() * (interp > 0.1)
    
    if img_path.suffix == '.png':
        imageio.imwrite(img_path, interp)
    elif img_path.suffix == '.npy':
        np.save(img_path, interp)
    else:
        assert False, "Impossible"

def optimize_json(json_path):
    with json_path.open('r') as f:
        data = json.load(f, parse_float=lambda x: round(float(x), 6))
    json_path.unlink()
    with json_path.open('w') as f:
        data = json.dump(data, f, indent=0)

def optimize_groundtruth_filesize(scene_folder):

    for frames_folder in sorted(list(scene_folder.glob('frames*'))):

        first_image = next(frames_folder.glob('Image*.png'))
        base_shape = np.array(imageio.imread(first_image).shape[:2])

        for img_path in sorted(list(frames_folder.iterdir())):
            print(img_path)
            if img_path.suffix == '.json':
                optimize_json(img_path)
            else:
                fix_image_resolution(img_path, base_shape)

def parse_jobscene_path(args):

    root = args.smb_root

    if args.jobscene_path is None:
        return [
            f.relative_to(root)
            for d in smb_client.listdir(root)
            for f in smb_client.globdir(d/'*.tar.gz')
        ]
    elif len(args.jobscene_path.parts) == 1: 
        return [
            f.relative_to(root)
            for f in smb_client.listdir(root/args.jobscene_path)
        ]
    elif len(args.jobscene_path.parts) == 2:
        return [args.jobscene_path]
    else:
        raise ValueError(f'Unrecognized {args.jobscene_path=}')

def cleanup_smb(smb_parent_folder):

    for f1, ftype, *_ in smb_client.listdir(smb_parent_folder, extras=True):
        if ftype != 'D':
            print(f'Ignoring {f1}, not a directory')
            continue

        files = list(smb_client.listdir(f1))
        n_files = len(files)

        if n_files <= 1:
            print(f'Removing {f1} {n_files}')
            smb_client.remove(f1)
        else:
            print(f'Keeping {f1} {n_files}')

def fix_missing_camviewdata(local_folder, dummy):
    
    camdata = deepcopy(dummy)
    del camdata['T']
    camdata['baseline'] = 0.075

    for frames_folder in local_folder.glob('frames*'):
        for image_path in frames_folder.glob('Image*.png'):
            idxs = parse_suffix(image_path.name)
            outpath = image_path.parent/('camview' + get_suffix(idxs) + '.npz')

            with outpath.open('wb') as f:
                np.savez(f, camdata)

def fix_frames_folderstructure(p):

    p = args.local_path / p

    frames_dest = p/'frames'
    for frames_old in p.glob('frames_*'):
        print(frames_old)
        for img_path in frames_old.iterdir():
            dtype, *_ = img_path.name.split('_')
            idxs = parse_suffix(img_path.name)
            new_path = frames_dest/dtype/f"camera_{idxs['subcam']}"/img_path.name
            new_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.move(img_path, new_path)
        frames_old.rmdir()

    for savemesh in p.glob('savemesh_*'):
        shutil.rmtree(savemesh)

    for folder in p.iterdir():
        if not folder.is_dir():
            continue
        for f in folder.glob('*b_displacement.npy'):
            f.unlink()


def process_one_scene(p):
    
    print(f'Processing {p}')
    stem = p.name.split('.')[0]
    scene_files = [
        p,
        p.parent/f'{stem}_metadata.json',
        p.parent/f'{stem}_thumbnail.png',
    ]

    local_tar = args.local_path / p
    local_folder = local_tar.parent/(local_tar.name.split('.')[0])
    assert local_tar.name.endswith('.tar.gz'), local_tar

    for s in scene_files:
        local_s = args.local_path/s
        local_s.parent.mkdir(exist_ok=True, parents=True)
        if local_s == local_tar and local_folder.exists():
            continue
        if not local_s.exists():
            smb_client.download(args.smb_root/s, dest_folder=local_s.parent)
        assert local_s.exists()

    if not local_folder.exists():
        assert local_tar.exists()
        untar_folder = untar(local_tar)
        assert untar_folder == local_folder
    assert local_folder.exists()

    print(f'Postprocessing {local_folder=}')
    fix_scene_structure(local_folder, n_subcams=2)
    fix_metadata(local_folder)
    optimize_groundtruth_filesize(local_folder)
    fix_missing_camviewdata(local_folder, dummy=dict(np.load('camview_dummy.npz')))
    fix_frames_folderstructure(local_folder)

    print(f'Validating {local_folder=}')
    dset = torch_dataset.InfinigenSceneDataset(local_folder)
    dset.validate()

    with (local_folder/'PREPROCESSED.txt').open('w') as f:
        f.write(f'{TOOLKIT_VERSION=}')

    if local_tar.exists():
        local_tar.unlink()

def try_process(p):
    try:
        process_one_scene(p)
    except Exception as e:
        with (Path()/'failures.txt').open('a') as f:
            f.write(f'{p} | {e}\n')
        folder_name = p.parent/(p.name.split('.')[0])
        if folder_name.exists():
            shutil.rmtree(folder_name)
        

def main(args):

    job_scene_paths = parse_jobscene_path(args)
    print(job_scene_paths)

    def sort_key(p):
        folder_name = p.parent/(p.name.split('.')[0])
        if folder_name.exists():
            return 0
        if p.exists():
            return 1
        return 2

    job_scene_paths = sorted(
        job_scene_paths,
        key=sort_key,
    )

    ps = list((args.local_path/args.jobscene_path).iterdir())
    mapfunc(try_process, ps, n_workers=args.n_workers)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('local_path', type=Path)
    parser.add_argument('smb_root', type=Path)
    parser.add_argument('--jobscene_path', type=Path, default=None)
    parser.add_argument(
        '--step', 
        type=str, 
        default='stream_all',
        choices=[
            'cleanup_smb',
            'download', 
            'untar', 
            'fix_pre_v1_1', 
            'demo_video', 
            'retar', 
            'stream_all'
        ]
    )
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    #cleanup_smb(args.smb_root)

    main(args)
# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

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
import tarfile

import imageio
import numpy as np
import cv2

from infinigen.datagen.util import smb_client
from infinigen.datagen.states import parse_suffix, get_suffix

from . import dataset_loader
from . import compress_masks

TOOLKIT_VERSION = '0.2.0'

IMAGE_RESIZE_ACTIONS = {

    ('Image', '.png', 'SINGLE_RES', 'INTER_LINEAR'),

    ('camview', '.npz', 'NO_ACTION', 'NO_INTERP'), 

    ('Depth', '.npy', 'DOUBLE_RES', 'INTER_NEAREST'),
    ('Depth', '.png', 'DOUBLE_RES', 'INTER_LINEAR'),
    ('SurfaceNormal', '.npy', 'SINGLE_RES', 'INTER_NEAREST'),
    ('SurfaceNormal', '.png', 'SINGLE_RES', 'INTER_LINEAR'),

    ('InstanceSegmentation', '.npz', 'SINGLE_RES', 'NPZ_INTER_NEAREST'),
    ('InstanceSegmentation', '.png', 'SINGLE_RES', 'INTER_NEAREST'),
    ('ObjectSegmentation', '.npz', 'SINGLE_RES', 'NPZ_INTER_NEAREST'),
    ('ObjectSegmentation', '.png', 'SINGLE_RES', 'INTER_NEAREST'),
    ('TagSegmentation', '.npz', 'SINGLE_RES', 'NPZ_INTER_NEAREST'),
    ('TagSegmentation', '.png', 'SINGLE_RES', 'INTER_NEAREST'),

    ('Objects', '.json', 'COMPRESS_JSON', 'NO_INTERP'),

    ('Flow3D_', '.npy', 'SINGLE_RES', 'INTER_NEAREST'),
    ('Flow3D_', '.png', 'SINGLE_RES', 'INTER_LINEAR'),
    ('Flow3DMask', '.png', 'SINGLE_RES', 'MASK_POOL'),

    ('OcclusionBoundaries', '.png', 'ORIG_RES', 'NO_INTERP'),
    
    ('AO', '.png', 'SINGLE_RES', 'NO_INTERP'), 
    ('DiffCol', '.png', 'SINGLE_RES', 'NO_INTERP'), 
    ('DiffDir', '.png', 'SINGLE_RES', 'NO_INTERP'), 
    ('DiffInd', '.png', 'SINGLE_RES', 'NO_INTERP'), 
    ('Emit', '.png', 'SINGLE_RES', 'NO_INTERP'), 
    ('Env', '.png', 'SINGLE_RES', 'NO_INTERP'), 
    ('GlossCol', '.png', 'SINGLE_RES', 'NO_INTERP'), 
    ('GlossDir', '.png', 'SINGLE_RES', 'NO_INTERP'), 
    ('GlossInd', '.png', 'SINGLE_RES', 'NO_INTERP'), 
    ('TransCol', '.png', 'SINGLE_RES', 'NO_INTERP'), 
    ('TransDir', '.png', 'SINGLE_RES', 'NO_INTERP'), 
    ('TransInd', '.png', 'SINGLE_RES', 'NO_INTERP'), 
    ('VolumeDir', '.png', 'SINGLE_RES', 'NO_INTERP'), 
}

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

def cleanup(p):

    for folder in p.glob('frames_*'):
        for path in folder.iterdir():
            if path.name.endswith('_displacement.npy'):
                path.unlink()
            if path.name == 'assets':
                path.unlink()
            if path.suffix == '.glb':
                path.unlink()

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
        #print(a, ' --> ', b)
        shutil.move(a, b)

    def postprocess_image(img_path, frame_folder, subcam):
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

    cleanup(p)
        
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

def optimize_json_inplace(json_path):
    with json_path.open('r') as f:
        data = json.load(f, parse_float=lambda x: round(float(x), 6))
    json_path.unlink()
    with json_path.open('w') as f:
        data = json.dump(data, f, indent=0)

def resize_inplace(img_path, target_shape, interp_method, npz_prefix='NPZ_'):

    match img_path.suffix:
        case '.png':
            img = imageio.imread(img_path)
        case '.npy':
            img = np.load(img_path)
        case '.npz':
            assert interp_method.startswith(npz_prefix)
            img = np.load(img_path)
        case suffix:
            raise ValueError(f'Unrecognized {suffix=}')

    using_npz_compression = interp_method.startswith(npz_prefix)
    if using_npz_compression:
        interp_method = interp_method[len(npz_prefix):]
        img = compress_masks.recover(img)

    curr_shape = np.array(img.shape[:2])
    
    if np.any(target_shape > curr_shape):
        raise ValueError(img_path, curr_shape, target_shape)

    if np.all(target_shape == curr_shape):
        return

    match interp_method:
        case 'INTER_LINEAR' | 'INTER_NEAREST':
            img = cv2.resize(img, target_shape[::-1], interpolation=getattr(cv2, interp_method))
        case 'MASK_POOL':
            interp = cv2.resize((img.astype('float') / 255), target_shape[::-1], cv2.INTER_LINEAR)
            img = (255 * (interp > 0.01)).astype(img.dtype)
        case _:
            raise ValueError(f'Unrecognized {interp_method=}')
    
    if using_npz_compression:
        img = compress_masks.compress(img)

    match img_path.suffix:
        case '.png':
            imageio.imwrite(img_path, img)
        case '.npy':
            np.save(img_path, img)
        case '.npz':
            np.savez(img_path, **dict(img))
        case _:
            raise ValueError(f'{img_path.suffix=}')

def optimize_groundtruth_filesize(scene_folder):

    frames_folders = sorted(list(scene_folder.glob('frames_*')))

    if len(frames_folders) == 0:
        raise ValueError(f'Couldnt find frames_* in {scene_folder}')

    first_folder_image_paths = list(frames_folders[0].glob('Image*.png'))
    base_img_res = np.array(imageio.imread(first_folder_image_paths[0]).shape[:2])

    for frames_folder in frames_folders:
        for dtype, ext, action, interp_method in IMAGE_RESIZE_ACTIONS:
            targets = sorted(list(frames_folder.glob(f'{dtype}*{ext}')))
            
            if len(targets) != len(first_folder_image_paths):
                raise ValueError(f'Found incorrect {len(targets)=} for {dtype=} {ext=} in {frames_folder=}, expected {len(first_folder_image_paths)}')

            for target_path in targets:
                print(target_path.relative_to(scene_folder.parent), action, interp_method)
                match action:
                    case 'SINGLE_RES':
                        resize_inplace(target_path, base_img_res, interp_method)
                    case 'DOUBLE_RES':
                        resize_inplace(target_path, base_img_res * 2, interp_method)
                    case 'ORIG_RES':
                        pass
                    case 'COMPRESS_JSON':
                        assert interp_method == 'NO_INTERP'
                        optimize_json_inplace(target_path)
                    case 'NO_ACTION':
                        assert interp_method == 'NO_INTERP'
                    case _:
                        raise ValueError(f'Unrecognized {action=}')

def parse_jobscene_path(args):

    root = args.smb_root

    if args.jobscene_path is None:
        for d in smb_client.listdir(root):
            for f in smb_client.globdir(d/'*.tar.gz'):
                yield f.relative_to(root)
    elif len(args.jobscene_path.parts) == 1: 
        d = root/args.jobscene_path
        for f in smb_client.globdir(d/'*.tar.gz'):
            yield f.relative_to(root)
    elif len(args.jobscene_path.parts) == 2:
        yield args.jobscene_path
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

            if outpath.exists():
                continue

            with outpath.open('wb') as f:
                np.savez(f, camdata)

def reorganize_old_framesfolder(frames_old):
    
    frames_old = Path(frames_old)

    for p in frames_old.iterdir():
        if p.is_symlink():
            p.unlink()

    frames_dest = frames_old.parent/"frames"

    for img_path in frames_old.iterdir():
        if img_path.is_dir():
            continue
        dtype, *_ = img_path.name.split('_')
        idxs = parse_suffix(img_path.name)
        new_path = frames_dest/dtype/f"camera_{idxs['subcam']}"/img_path.name
        new_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.move(img_path, new_path)

    if frames_dest != frames_old:
        frames_old.rmdir()

def fix_frames_folderstructure(p):

    p = args.local_path / p

    for frames_old in p.glob('frames_*'):
        reorganize_old_framesfolder(frames_old)        

    for savemesh in p.glob('savemesh_*'):
        shutil.rmtree(savemesh)

    for folder in p.iterdir():
        if not folder.is_dir():
            continue
        for f in folder.glob('*b_displacement.npy'):
            f.unlink()

def retar_for_distribution(local_folder, distrib_path):

    seed = local_folder.name
    frames_folder = (local_folder/'frames')

    for dtype_folder in frames_folder.iterdir():
        for camera_folder in dtype_folder.iterdir():
            exts = set(p.suffix for p in camera_folder.iterdir())
            for ext in exts:
                tar_path = distrib_path/seed/f"{seed}_{dtype_folder.name}_{ext.strip('.')}_{camera_folder.name}.tar.gz"
                if tar_path.exists():
                    continue
                tar_path.parent.mkdir(exist_ok=True, parents=True)
                print(f'Creating {tar_path}')
                with tarfile.open(tar_path, 'w:gz') as f:
                    for img in camera_folder.glob(f'*{ext}'):
                        f.add(img, arcname=img.relative_to(local_folder.parent))

def process_one_scene(p, args):

    stem = p.name.split('.')[0]
    scene_files = [
        p,
        p.parent/f'{stem}_metadata.json',
        p.parent/f'{stem}_thumbnail.png',
    ]

    local_tar = args.local_path / p
    local_folder = local_tar.parent/(local_tar.name.split('.')[0])
    assert local_tar.name.endswith('.tar.gz'), local_tar

    if False:
        for s in scene_files:
            local_s = args.local_path/s
            local_s.parent.mkdir(exist_ok=True, parents=True)
            if local_s == local_tar and local_folder.exists():
                print(f'Skipping {s} as {local_folder} exists')
                continue
            if not local_s.exists():
                print(f'Downloading {s}')
                smb_client.download(args.smb_root/s, dest_folder=local_s.parent)
            assert local_s.exists()

        if not local_folder.exists():
            print(f'Untarring {local_folder}')
            assert local_tar.exists()
            untar_folder = untar(local_tar)
            assert untar_folder == local_folder
        else:
            print(f'Skipping untar {local_tar}')
        assert local_folder.exists()

        if not (local_folder/'PREPROCESSED.txt').exists():
            print(f'Postprocessing {local_folder=}')
            fix_scene_structure(local_folder, n_subcams=2)
            fix_metadata(local_folder)
            optimize_groundtruth_filesize(local_folder)
            fix_frames_folderstructure(local_folder)

            with (local_folder/'PREPROCESSED.txt').open('w') as f:
                f.write(f'{TOOLKIT_VERSION=}')

    if not local_folder.exists():
        return

    print(f'Validating {local_folder=}')
    dset = dataset_loader.InfinigenSceneDataset(local_folder, data_types=dataset_loader.ALLOWED_IMAGE_TYPES)
    dset.validate()

    if local_tar.exists():
        local_tar.unlink()

    if args.distrib_path is not None:
        retar_for_distribution(local_folder, args.distrib_path)

def try_process(p, args):
    process_one_scene(p, args)
    try:
        pass
    except Exception as e:
        print('FAILED', p, e)
        with (Path()/'failures.txt').open('a') as f:
            f.write(f'{p} | {e}\n')
        folder_name = p.parent/(p.name.split('.')[0])
        if folder_name.exists():
            shutil.rmtree(folder_name)
        

def main(args):

    job_scene_paths = list(parse_jobscene_path(args))

    #cleanup_smb(args.smb_root)

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

    print(f'Found {len(job_scene_paths)=}')

    mapfunc(partial(try_process, args=args), job_scene_paths, n_workers=args.n_workers)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('local_path', type=Path)
    parser.add_argument('smb_root', type=Path)
    parser.add_argument('--jobscene_path', type=Path, default=None)
    parser.add_argument('--distrib_path', type=Path, default=None)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    #cleanup_smb(args.smb_root)

    main(args)
# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import argparse
from pathlib import Path
import urllib.request
import re
import subprocess
import json
from multiprocessing import Pool
from functools import partial

SUPPORTED_DATARELEASE_FORMATS = {"0.2.0"}

def wget_path(args, path):
    url = args.repo_url + str(path)
    cmd = f'wget -q -N --show-progress {url} -P {str(args.output_folder)}'
    subprocess.check_call(cmd.split())

def untar_path(args, tarfile):
    assert tarfile.exists()
    cmd = f'tar -xzf {tarfile} -C {args.output_folder}'
    print(cmd)
    subprocess.check_call(cmd.split())
    tarfile.unlink()

def url_to_text(url):
    with urllib.request.urlopen(url) as f:
        return f.read().decode('utf-8')

def check_and_preprocess_args(args, metadata):

    datarelease_format_version = metadata['datarelease_format_version']
    if datarelease_format_version not in SUPPORTED_DATARELEASE_FORMATS:
        raise ValueError(
            f'{args.release_name} uses {datarelease_format_version=} which is not '
            ' supported by this download script. Please download a newer version of the code.'
        )
    
    if args.seeds is None:
        args.seeds = metadata['seeds']
        print(f'User did not specify --seeds, using all available ({len(args.seeds)} total):\n\t{args.seeds}')
    else:
        missing = set(args.seeds) - set(metadata['seeds'])
        if len(missing):
            raise ValueError(f"In user-provided --seeds, {missing} could not be found in {args.release_name} metadata.json")

    if args.cameras is None:
        args.cameras = metadata['cameras']
        print(f'User did not specify --cameras, using all available ({len(args.cameras)} total):\n\t{args.cameras}')
    else:
        missing = set(args.cameras) - set(metadata['cameras'])
        if len(missing):
            raise ValueError(f"In user-provided --cameras, {missing} are not supported acording {args.release_name} metadata.json")

    if args.data_types is None:
        args.data_types = metadata['data_types']
        print(f'User did not specify --data_types, using all available ({len(args.data_types)} total): \n\t{args.data_types}')
    else:
        missing = set(args.data_types) - set(metadata['data_types'])
        if len(missing):
            raise ValueError(f"In user-provided --seeds, {missing} could not be found in {args.release_name} metadata.json")

def process_path(args, path):
    wget_path(args, path)
    untar_path(args, tarfile=args.output_folder/path.name)

def main(args):

    metadata_url = f'{args.repo_url}/{args.release_name}/metadata.json'
    metadata = json.loads(url_to_text(metadata_url))

    print("=" * 10)
    print(f"Description for release {repr(args.release_name)}:")
    print("-" * 10)
    print(metadata['description'])
    print("=" * 10)

    check_and_preprocess_args(args, metadata)

    toplevel = Path(args.release_name)/'renders'

    paths = []
    for seed in args.seeds:
        for camera in args.cameras:
            for imgtype in args.data_types:
                name = f'{seed}_{imgtype}_{camera}.tar.gz'
                paths.append(toplevel/seed/name)

    print(f'User requested {len(args.seeds)} seeds x {len(args.cameras)} cameras x {len(args.data_types)} data types')
    print(f'This script will download and untar {len(paths)} tarballs from {args.repo_url}')
    if input('Do you wish to proceed? [y/n]: ') != 'y':
        exit()

    with Pool(args.n_workers) as pool:
        pool.map(partial(process_path, args), paths)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=Path)
    parser.add_argument(
        "--repo_url",
        type=str,
        help="Fileserver URL to download from",
    )
    parser.add_argument(
        "--release_name",
        type=str,
        help="What named datarelease do you want to download? (pick any toplevel folder name from the URL)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        nargs="+",
        default=None,
        help="What scenes should we download? Omit to download all available in this release",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs='+',
        default=None,
        help="What cameras should we download data for? Omit to download all available in this release",
    )
    parser.add_argument(
        "--data_types",
        type=str,
        nargs='+',
        default=None,
        help="What data types (e.g Image, Depth, etc) should we download data for? Omit to download all available in this release",
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=1
    )

    args = parser.parse_args()
    main(args)


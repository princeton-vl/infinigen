# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import argparse
from collections import OrderedDict
from pathlib import Path
import urllib.request
import re
import subprocess
import json
from multiprocessing import Pool
from functools import partial

SUPPORTED_DATARELEASE_FORMATS = {"0.2.0"}

TEXT_SEPARATOR_LINE = "=" * 60

ANNOT_DESCRIPTIONS = OrderedDict([
    
    ("Image_png", "RGB image as a .png."),
    ("Image_exr", "RGB image as an .exr"),

    ("camview_npz", "Camera intrinsic & extrinsic matricies, IE, camera calibration and poses."),
    
    ("Depth_npy", "Depth"),
    ("Depth_png", "Color-mapped PNG of Depth_npy. FOR VISUALIZATION ONLY"),

    ("SurfaceNormal_npy", "Surface Normals"),
    ("SurfaceNormal_png", "Color-mapped PNG of SurfaceNormal_npy. FOR VISUALIZATION ONLY"),

    ("Flow3D_npy", "Optical Flow and Depth change."),
    ("Flow3D_png", "Color-wheel visualization of the 2D part of Flow3D_npy. FOR VISUALIZATION ONLY"),

    ("Flow3DMask_png", "Flow Occlusion mask."),
    ("OcclusionBoundaries_png", "Occlusion Boundaries."),

    ("ObjectSegmentation_npz", "Semantic Segmentation mask. Compressed using a lookup table - see docs for more info."),
    ("ObjectSegmentation_png", "Color-mapped PNG of ObjectSegmentation.npz. FOR VISUALIZATION ONLY"),

    ("InstanceSegmentation_npz", "Instance Segmentation mask. Compressed using a lookup table - see docs for more info."),
    ("InstanceSegmentation_png", "Color-mapped PNG of InstanceSegmentation.npz. FOR VISUALIZATION ONLY"),

    ("TagSegmentation_npz", "Segmentation mask to help distinguish different parts of the same object. Compressed using a lookup table - see docs for more info."),
    ("TagSegmentation_png", "Color-mapped PNG of TagSegmentation_npz. FOR VISUALIZATION ONLY"),

    ("Objects_json", "LARGE json object specifying names, poses and bounding boxes of objects in the scene. Required for 2D/3D BBox."),

    ("AO_png", "Ambient Occlusion."),
    ("DiffCol_png", "Diffuse Color, a.k.a Albedo."),
    ("DiffDir_png", "Diffuse Direct pass."),
    ("DiffInd_png", "Diffuse Indirect pass."),
    ("Emit_png", "Emission pass."),
    ("Env_png", "Environment pass."),
    ("GlossCol_png", "Glossy color."),
    ("GlossDir_png", "Glossy direct pass."),
    ("GlossInd_png", "Glossy indirect pass."),
    ("TransCol_png", "Transmission color."),
    ("TransDir_png", "Transmission direct pass."),
    ("TransInd_png", "Transmission indirect pass."),
    ("VolumeDir_png", "Volume direct pass."),
])

CAMERA_DESCRIPTIONS = OrderedDict({
    "camera_0": "The default camera; select only this if you just want monocular data.",
    "camera_1": "Select both this camera and the above if you want stereo data."
})

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

def user_select_string_list(values, descriptions_dict=None, extra_msg=None):

    if descriptions_dict is not None:
        def sort_by_description_order(vinp):
            try:
                return next(i for i, k in enumerate(descriptions_dict.keys()) if k == vinp)
            except StopIteration:
                return len(values)
        values = sorted(values, key=sort_by_description_order)

    print(TEXT_SEPARATOR_LINE)
    for i, v in enumerate(values):
        prompt = f'({i}) {v:<25}'
        if descriptions_dict is not None and v in descriptions_dict:
            desc = descriptions_dict[v]
            prompt += f' - {desc}'
        print(prompt)
    
    if extra_msg is not None:
        print(extra_msg)

    print(TEXT_SEPARATOR_LINE)

    print('Please enter your choices from above, as a space-separated list of integers or strings, or type \"ALL\"')
    selections = input('Enter your selection: ')

    print('\n')

    if selections == 'ALL':
        return values

    def postprocess(x):

        try:
            x = int(x)
        except ValueError:
            pass

        if (
            (isinstance(x, str) and x not in values) or
            (isinstance(x, int) and x not in range(len(values)))
        ):
            raise ValueError(f'User provided input \"{x}\" was not recognized, expected integer 0 to {len(values)-1} or a shorthand string')

        if isinstance(x, int):
            x = values[x]

        return x
    
    selections = [x.strip().strip(',') for x in selections.split()]
    selections = [postprocess(x) for x in selections]

    print('Selected: ', selections)
    return selections

def check_and_preprocess_args(args, metadata):

    datarelease_format_version = metadata['datarelease_format_version']
    if datarelease_format_version not in SUPPORTED_DATARELEASE_FORMATS:
        raise ValueError(
            f'{args.release_name} uses {datarelease_format_version=} which is not '
            ' supported by this download script. Please download a newer version of the code.'
        )
    
    if args.data_types is None:
        args.data_types = user_select_string_list(
            metadata['data_types'], 
            ANNOT_DESCRIPTIONS, 
            extra_msg="\nNote: See https://docs.blender.org/manual/en/latest/render/layers/passes.html for a description of Blender-Cycles' render passes"
        )
        if not any("Image" in x for x in args.data_types):
            print('WARNING: User did not request Image_png or Image_exr, this is unusual. Please restart if this was not intended.')
    else:
        missing = set(args.data_types) - set(metadata['data_types'])
        if len(missing):
            raise ValueError(f"In user-provided --seeds, {missing} could not be found in {args.release_name} metadata.json")

    if args.seeds is None:
        n = len(metadata['seeds'])
        print(
            f"How many videos do you wish to download? "
            f"Enter a quantity from 1 to {n}, or type SELECT to pick specific seeds"
        )
        selection = input("Enter your selection: ")
        if selection == 'SELECT':
            args.seeds = user_select_string_list(metadata['seeds'])
        else:
            num_select = int(selection)
            args.seeds = metadata['seeds'][:num_select]
            
            
    
    missing = set(args.seeds) - set(metadata['seeds'])
    if len(missing):
        raise ValueError(f"In user-provided --seeds, {missing} could not be found in {args.release_name} metadata.json")

    if args.cameras is None:
        args.cameras = user_select_string_list(metadata['cameras'], CAMERA_DESCRIPTIONS)
    else:
        missing = set(args.cameras) - set(metadata['cameras'])
        if len(missing):
            raise ValueError(f"In user-provided --cameras, {missing} are not supported acording {args.release_name} metadata.json")

def process_path(args, path):
    wget_path(args, path)
    untar_path(args, tarfile=args.output_folder/path.name)

def main(args):

    if args.release_name is None:
        print(f'Please specify a --release_name. Go to {args.repo_url} in your browser to see what folders are available.')
        exit()

    metadata_url = f'{args.repo_url}/{args.release_name}/metadata.json'
    metadata = json.loads(url_to_text(metadata_url))

    print(TEXT_SEPARATOR_LINE)
    print(f"Description for release {repr(args.release_name)}:")
    print(metadata['description'])
    print(TEXT_SEPARATOR_LINE)
    input("Press Enter to continue...")
    print('\n')

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
    choice = input('Do you wish to proceed? [y/n]: ')
    if not (choice == "" or choice in " yY1"):
        exit()

    with Pool(args.n_workers) as pool:
        pool.map(partial(process_path, args), paths)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_folder', type=Path)
    parser.add_argument(
        "--repo_url",
        type=str,
        default="https://infinigen-data.cs.princeton.edu/",
        help="Fileserver URL to download from",
    )
    parser.add_argument(
        "--release_name",
        type=str,
        default=None,
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


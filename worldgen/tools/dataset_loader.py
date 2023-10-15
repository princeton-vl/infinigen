# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 

from pathlib import Path
import json
import imageio
import numpy as np
import logging

#import torch.utils.data IMPORTED ONLY IF USING get_infinigen_dataset

from .suffixes import parse_suffix, get_suffix

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {

    # Read docs/GroundTruthAnnotations.md for more explanations

    'Image_png',
    #('Image', '.exr'), # NOT IMPLEMENTED

    'camview_npz', # intrinisic, extrinsic, etc

    # names available via EITHER blender_gt.gin and opengl_gt.gin
    'Depth_npy',
    'Depth_png',
    'InstanceSegmentation_npz',
    'InstanceSegmentation_png',
    'ObjectSegmentation_npz',
    'ObjectSegmentation_png',
    'SurfaceNormal_npy',
    'SurfaceNormal_png',
    'Objects_json',

    # blender_gt.gin only provides 2D flow. opengl_gt.gin produces Flow3D instead
    'Flow3D_npy',
    'Flow3D_png',

    # names available ONLY from opengl_gt.gin
    'OcclusionBoundaries_png',
    'TagSegmentation_npz',
    'TagSegmentation_png',
    'Flow3DMask_png',
 
    # info from blender image rendering passes, usually enabled regardless of GT method
    'AO_png',
    'DiffCol_png',
    'DiffDir_png',
    'DiffInd_png',
    'Emit_png',
    'Env_png',
    'GlossCol_png',
    'GlossDir_png',
    'GlossInd_png',
    'TransCol_png',
    'TransDir_png',
    'TransInd_png',
    'VolumeDir_png',
}

def get_blocksize(scene_folder):
    first, second, *_ = sorted(scene_folder.glob('frames*_0'))
    return parse_suffix(second)['frame'] - parse_suffix(first)['frame']

def get_framebounds_inclusive(scene_folder):
    rgb = scene_folder/'frames'/'Image'/'camera_0'
    first, *_, last = sorted(rgb.glob('*.png'))
    return (
        parse_suffix(first)['frame'],
        parse_suffix(last)['frame']
    ) 

def get_cameras_available(scene_folder):
    return [int(p.name.split('_')[-1]) for p in (scene_folder/'frames'/'Image').iterdir()]

def get_imagetypes_available(scene_folder):
    dtypes = []
    for dtype_folder in (scene_folder/'frames').iterdir():
        frames = dtype_folder/'camera_0'
        uniq = set(p.suffix for p in frames.iterdir())
        dtypes += [f'{dtype_folder.name}_{u.strip(".")}' for u in uniq]
    return dtypes

def get_frame_path(scene_folder, cam: int, frame_idx, data_type) -> Path:
    data_type_name, data_type_ext = data_type.split('_')
    imgname = f'{data_type_name}_0_0_{frame_idx:04d}_{cam}.{data_type_ext}'
    return Path(scene_folder)/'frames'/data_type_name/f'camera_{cam}'/imgname

class InfinigenSceneDataset:

    def __init__(
        self, 
        scene_folder: Path,
        data_types: list[str] = None, # see ALLOWED_IMAGE_KEYS above. Use 'None' to retrieve all available PNG datatypes
        cameras=None,
        gt_for_first_camera_only=True,
    ):

        self.scene_folder = Path(scene_folder)
        self.gt_for_first_camera_only = gt_for_first_camera_only

        if data_types is None:
            data_types = get_imagetypes_available(self.scene_folder)
            logging.info(f'{self.__class__.__name__} recieved data_types=None, using whats available in {scene_folder}: {data_types}')
        for t in data_types:
            if t not in ALLOWED_IMAGE_TYPES:
                raise ValueError(f'Recieved data_types containing {t} which is not in ALLOWED_IMAGE_TYPES')
        self.data_types = data_types

        if cameras is None:
            cameras = get_cameras_available(self.scene_folder)
        self.cameras = cameras

        self.framebounds_inclusive = get_framebounds_inclusive(self.scene_folder)

    def __len__(self):
        first, last = self.framebounds_inclusive
        return last - first
    
    @staticmethod
    def load_any_filetype(path):

        match path.suffix:
            case '.png':
                return imageio.imread(path)
            case '.exr':
                raise NotImplementedError
            case '.npy':
                return np.load(path)
            case '.npz':
                return dict(np.load(path))
            case 'json':
                with path.open('r') as f:
                    return json.load(f)
            case _:
                raise ValueError(f'Unhandled {path.suffix=} for {path=}')

    def _imagetypes_to_load(self, cam: int):
        for data_type in self.data_types:
            dtypename = data_type[0]
            if (
                self.gt_for_first_camera_only and
                cam != 0 and
                dtypename != 'Image' and
                dtypename != 'camview'
            ):
                continue
            yield data_type

    def validate(self):
        for i in range(len(self)):
            for cam in self.cameras:
                for dtype in self._imagetypes_to_load(cam):
                    frame = self.framebounds_inclusive[0]
                    p = self.frame_path(frame + i, cam, dtype)
                    if not p.exists():
                        raise ValueError(f'validate() failed for {self.scene_folder}, could not find {p}')

    def frame_path(self, i: int, cam: int, dtype: str):
        frame_num = self.framebounds_inclusive[0] + i
        return get_frame_path(self.scene_folder, cam, frame_num, dtype)

    def __getitem__(self, i):

        def get_camera_images(cam: int):
            imgs = {}
            for dtype in self._imagetypes_to_load(cam):
                path = self.frame_path(i, cam, dtype)
                imgs[dtype] = self.load_any_filetype(path)
            return imgs
        
        per_camera_data = [get_camera_images(i) for i in self.cameras]

        if len(self.cameras) == 1:
            return per_camera_data[0]
        else:
            return per_camera_data

def get_infinigen_dataset(data_folder: Path, mode='concat', validate=False, **kwargs):
    
    import torch.utils.data

    data_folder = Path(data_folder)

    scene_datasets = [
        InfinigenSceneDataset(f, **kwargs)
        for f in data_folder.iterdir()
        if f.is_dir()
    ]

    if validate:
        for d in scene_datasets:
            d.validate()

    match mode:
        case 'concat':
            return torch.utils.data.ConcatDataset(scene_datasets)
        case 'chain':
            return torch.utils.data.ChainDataset(scene_datasets)
        case _:
            raise ValueError(mode)

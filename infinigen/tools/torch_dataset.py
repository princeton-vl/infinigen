from pathlib import Path
import json
import imageio
import numpy as np
import logging

import torch.utils.data

from .data_suffixes import get_suffix, parse_suffix

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {

    # Read docs/GroundTruthAnnotations.md for more explanations

    ('Image', '.png'),
    #('Image', '.exr'), # NOT IMPLEMENTED

    ('camview', '.npz'), # contains intrinsic extrinsic etc

    # names available via EITHER blender_gt.gin and opengl_gt.gin
    ('Depth', '.npy'),
    ('Depth', '.png'),
    ('InstanceSegmentation', '.npy'),
    ('InstanceSegmentation', '.png'),
    ('ObjectSegmentation', '.npy'),
    ('ObjectSegmentation', '.png'),
    ('SurfaceNormal', '.npy'),
    ('SurfaceNormal', '.png'),
    ('Objects', 'json'),

    # blender_gt.gin only provides 2D flow. opengl_gt.gin produces Flow3D instead
    ('Flow3D', '.npy'),
    ('Flow3D', '.png'),

    # names available ONLY from opengl_gt.gin
    ('OcclusionBoundaries', '.npy'),
    ('OcclusionBoundaries', '.png'),
    ('TagSegmentation', '.npy'),
    ('TagSegmentation', '.png'),
    ('Flow3D', '.npy'),
    ('Flow3D', '.png'),
    ('Flow3DMask', '.npy'),
    ('Flow3DMask', '.png'),
 
    # info from blender image rendering passes, usually enabled regardless of GT method
    ('AO', '.png'), 
    ('DiffCol', '.png'), 
    ('DiffDir', '.png'), 
    ('DiffInd', '.png'), 
    ('Emit', '.png'), 
    ('Env', '.png'), 
    ('GlossCol', '.png'), 
    ('GlossDir', '.png'), 
    ('GlossInd', '.png'), 
    ('TransCol', '.png'), 
    ('TransDir', '.png'), 
    ('TransInd', '.png'), 
    ('VolumeDir', '.png'), 
}

def get_blocksize(scene_folder):
    first, second, *_ = sorted(scene_folder.glob('frames*_0'))
    return parse_suffix(second)['frame'] - parse_suffix(first)['frame']

def get_framebounds_inclusive(scene_folder):

    first, *_, last = sorted(scene_folder.glob('frames*_0/Image*'))
    return (
        parse_suffix(first)['frame'],
        parse_suffix(last)['frame']
    ) 

def get_subcams_available(scene_folder):
    return {parse_suffix(f)['subcam'] for f in scene_folder.glob('frames*')}

def get_imagetypes_available(scene_folder, ext):
    def imagetype(frame: Path):
        return (
            frame.name.split('_')[0],
            frame.suffix
        )
    return {imagetype(f) for f in scene_folder.glob(f'frames*/*{ext}')}

class InfinigenSceneDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        scene_folder: Path,
        image_types: list[str] = None, # see ALLOWED_IMAGE_KEYS above. Use 'None' to retrieve all available PNG datatypes
        
        # [0] for monocular, [0, 1] for stereo, 'None' to use whatever is present in the dataset
        subcam_keys=None,
        gt_for_first_camera_only=True,
    ):

        self.scene_folder = Path(scene_folder)
        self.gt_for_first_camera_only = gt_for_first_camera_only

        if image_types is None:
            image_types = get_imagetypes_available(self.scene_folder, ext='.png')
            logging.info(f'{self.__class__.__name__} recieved image_types=None, using whats available in {scene_folder}: {image_types}')
        for t in image_types:
            if t not in ALLOWED_IMAGE_TYPES:
                raise ValueError(f'Recieved image_types containing {t} which is not in ALLOWED_IMAGE_TYPES')
        self.image_types = image_types

        if subcam_keys is None:
            subcam_keys = get_subcams_available(self.scene_folder)
        self.subcam_keys = subcam_keys

        self.block_size = get_blocksize(self.scene_folder)
        self.framebounds_inclusive = get_framebounds_inclusive(self.scene_folder)

    def __len__(self):
        first, last = self.framebounds_inclusive
        return last - first + 1
    
    def frame_path(self, frame_num, subcam, image_type):

        first, last = self.framebounds_inclusive
        assert frame_num  >= first and frame_num <= last

        framefolder_idx = (frame_num - first) // self.block_size
        framefolder_framenum = framefolder_idx * self.block_size + first
        framefolder_keys = dict(
            cam_rig=0, 
            frame=framefolder_framenum, 
            resample=0, 
            subcam=subcam
        )
        framefolder_path = self.scene_folder/f'frames{get_suffix(framefolder_keys)}'

        dtypename, extension = image_type
        image_keys=dict(cam_rig=0, frame=frame_num, resample=0, subcam=subcam)
        image_name = f'{dtypename}{get_suffix(image_keys)}{extension}'

        path = framefolder_path/image_name
        return path
    
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

    def _imagetypes_to_load(self, subcam):
        for image_type in self.image_types:
            dtypename = image_type[0]
            if (
                self.gt_for_first_camera_only and
                subcam != 0 and
                dtypename != 'Image' and
                dtypename != 'camview'
            ):
                continue
            yield image_type

    def validate(self):
        for i in range(len(self)):
            for j in self.subcam_keys:
                for k in self._imagetypes_to_load(j):
                    frame = self.framebounds_inclusive[0]
                    p = self.frame_path(frame + i, j, k)
                    if not p.exists():
                        raise ValueError(f'validate() failed for {self.scene_folder}, could not find {p}')

    def __getitem__(self, i):

        frame_num = self.framebounds_inclusive[0] + i

        def get_camera_images(subcam):
            imgs = {}
            for image_type in self._imagetypes_to_load(subcam):
                path = self.frame_path(frame_num, subcam, image_type)
                assert path.exists(), path
                imgs[image_type[0]] = self.load_any_filetype(path)
            return imgs
        
        per_camera_data = [get_camera_images(i) for i in self.subcam_keys]

        if len(per_camera_data) == 1:
            per_camera_data = per_camera_data[0]

        return per_camera_data

def get_infinigen_dataset(data_folder: Path, mode='concat', validate=False, **kwargs):
    
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

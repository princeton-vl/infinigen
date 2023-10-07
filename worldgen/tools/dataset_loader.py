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

    ('Image', '.png'),
    #('Image', '.exr'), # NOT IMPLEMENTED

    ('camview', '.npz'), # contains intrinsic extrinsic etc

    # names available via EITHER blender_gt.gin and opengl_gt.gin
    ('Depth', '.npy'),
    ('Depth', '.png'),
    ('InstanceSegmentation', '.npz'),
    ('InstanceSegmentation', '.png'),
    ('ObjectSegmentation', '.npz'),
    ('ObjectSegmentation', '.png'),
    ('SurfaceNormal', '.npy'),
    ('SurfaceNormal', '.png'),
    ('Objects', '.json'),

    # blender_gt.gin only provides 2D flow. opengl_gt.gin produces Flow3D instead
    ('Flow3D', '.npy'),
    ('Flow3D', '.png'),

    # names available ONLY from opengl_gt.gin
    ('OcclusionBoundaries', '.png'),
    ('TagSegmentation', '.npz'),
    ('TagSegmentation', '.png'),
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
    rgb = scene_folder/'frames'/'Image'/'camera_0'
    first, *_, last = sorted(rgb.glob('*.png'))
    return (
        parse_suffix(first)['frame'],
        parse_suffix(last)['frame']
    ) 

def get_subcams_available(scene_folder):
    rgb = scene_folder/'frames'/'Image'
    return [int(p.name.split('_')[-1]) for p in rgb.iterdir()]

def get_imagetypes_available(scene_folder):
    dtypes = []
    for dtype_folder in (scene_folder/'frames').iterdir():
        frames = dtype_folder/'camera_0'
        uniq = set(p.suffix for p in frames.iterdir())
        dtypes += [(dtype_folder.name, u) for u in uniq]
    return dtypes

def get_frame_path(scene_folder, cam_idx, frame_idx, data_type_name, data_type_ext):
    imgname = f'{data_type_name}_0_0_{frame_idx:04d}_{cam_idx}{data_type_ext}'
    return scene_folder/'frames'/data_type_name/f'camera_{cam_idx}'/imgname

class InfinigenSceneDataset:

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
            image_types = get_imagetypes_available(self.scene_folder)
            image_types = [v for v in image_types if v[1] != '.exr'] # loading not implemented yet
            logging.info(f'{self.__class__.__name__} recieved image_types=None, using whats available in {scene_folder}: {image_types}')
        for t in image_types:
            if t not in ALLOWED_IMAGE_TYPES:
                raise ValueError(f'Recieved image_types containing {t} which is not in ALLOWED_IMAGE_TYPES')
        self.image_types = image_types

        if subcam_keys is None:
            subcam_keys = get_subcams_available(self.scene_folder)
        self.subcam_keys = subcam_keys

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

    def frame_path(self, i, subcam, dtype):
        frame_num = self.framebounds_inclusive[0] + i
        return get_frame_path(self.scene_folder, subcam, frame_num, dtype[0], dtype[1])

    def __getitem__(self, i):

        def get_camera_images(subcam):
            imgs = {}
            for dtype in self._imagetypes_to_load(subcam):
                path = self.frame_path(subcam, i, subcam, dtype)
                imgs[dtype[0]] = self.load_any_filetype(path)
            return imgs
        
        per_camera_data = [get_camera_images(i) for i in self.subcam_keys]

        if len(per_camera_data) == 1:
            per_camera_data = per_camera_data[0]

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

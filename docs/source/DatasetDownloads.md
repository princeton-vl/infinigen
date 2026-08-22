# Dataset downloads

We provide a tool [cvdpack](https://github.com/princeton-vl/cvdpack) which compresses datareleases into .mp4 or .mkv format using ffmpeg. This saves significant storage (~80% reduced) and download cost for large amounts of video data with depth/normals/flow etc compared to distributing .npy files. Cvdpack can either download from huggingface, unpack into compressed pngs, or unpack into floating-point .npy, or with default `unpack` it will do all of these things in one go.

You can save download time and storage cost by using `--subset` e.g. `--subset gt_type=rgb,depth` `cam=CameraLeft` for a monocular depth project.

You can also run `uvx cvdpack --help` to see options such as multiprocessing or parallizing over slurm. 

## flying-indoor

EMBED YOUTUBE VIDEO BIG AND LARGE

Huggingface Page: https://huggingface.co/datasets/infinigen/2026-08-flying-indoor
Release Date: 2026-08-12

This datarelease contains 1000 videos of 24 frames each spanning 250 unique 3D scenes. Each scene is a rectangular room with detailed wall materials, furniture, and several random objects moving along randomwalks, and 4 stereo camera rigs moving along random walks. Each grouping of 4 videos uses the same underlying 3D scene, and therefore provides wide-baseline camera views of the same scene, so long as the randomly chosen camera views overlap the same parts of the scene. 

This datarelease generated was pre-release versions of infinigen==2.0.0a2. You can approximately reproduce it using our [`flying_indoor` code example](./ExampleProjects.md#flying_indoor). The random seeds we used are shown in filenames and metadata.json, they came from SLURM job IDs and are completely arbitrary. The dataset is curated / filtered only to remove the 1% of scenes which had incomplete data or were somehow interrupted during our run.

### Limitations: furniture surface detail & no material names

This version did not densely subdivide some furniture objects, such as windows and shelves. This means that material displacement e.g. wood/stone texture affects RGB via [bump mapping](https://en.wikipedia.org/wiki/Bump_mapping) but does NOT properly affect geometry groundtruth. The room's wall and floor geometry are not affected by this. Furniture is affected, but typically has small displacement, so is less severe but still present.

Surface normals are severely affected by this; we do not recommend this dataset for surface normal training unless improperly subdivided objects are masked out of the loss. 
Monocular/Stereo Depth, Optical/Scene Flow are affected due to small error in depth, but this is generally only for the bumpiness of wood grain or other features on furniture, so we believe the data could be useful.

Materials are labelled only by object name then by slot id within that object. e.g. the materials of one window appear as `window.04_0`, `window.04_1`, `window.04_2`, so the glass, frame and curtain of that window are not distinguishable by name. Many materials are not named at all and appear only as e.g. `material_37`.
Worse, the `material-index-table.json` which maps the ids in `material-index_*.npy` onto those names is not included in this release, so material ids can be grouped but not identified at all.
This makes masking out materials inconvinient or impossible unless the object and slot naeme are known.
Future datareleases will correct this deficiency aswell.

Future datareleases will correct these deficiencies. We have released this preview dataset incase it is useful for other tasks.

### Download

You can download a few specific scenes and ground truth types using a command similar to below. 
```bash
uvx --from 'cvdpack[hf]' cvdpack unpack \
  --input https://huggingface.co/datasets/infinigen/2026-08-flying-indoor \
  --output flying_indoor \
  --tmp_folder flying_indoor_tmp \
  --subset scene=30377061_0 traj=0 gt_type=rgb,depth cam=CameraLeft
```
These --subset keys align to our youtube video text annotations which may help choose what to download.
tmp_folder will be used to store .png versions of depth before they are converted to the final .npy files.

`scene` selects the 3D scene and `traj` selects which of its 4 camera trajectories you want. Omit `traj` to get all 4 trajectories of that scene, or omit `scene` to get e.g. `traj=0` of every scene. Each key also accepts a comma separated list, e.g. `traj=0,2`.

Or, download and unpack the entire dataset:
```bash
uvx --from 'cvdpack[hf]' cvdpack unpack \
  --input https://huggingface.co/datasets/infinigen/2026-08-flying-indoor \
  --output flying_indoor \
  --tmp_folder flying_indoor_tmp \
  --n_workers 4 --parallel_mode multiprocess
```
Note: You can use ``uvx --from 'cvdpack[hf,slurm]' ..... --parallel_mode slurm --slurm_args slurm_account=myaccount` to unpack on a slurm cluster. 

Or, you can use the traditional huggingface download, then run cvdpack to get compressed .pngs, then again to get .npys for depth etc:
```bash
uvx --from huggingface_hub hf download infinigen/2026-08-flying-indoor \
  --repo-type dataset --local-dir flying_indoor_packed

uvx cvdpack unpack --input flying_indoor_packed --output flying_indoor_png \
  --tmp_folder flying_indoor_tmp --steps unpack_video --n_workers 4

uvx cvdpack unpack --input flying_indoor_png --output flying_indoor \
  --tmp_folder flying_indoor_tmp --steps unquantize --n_workers 4
```

### Video Frames Schema

Each trajectory holds one video per pass and camera
```
30377061_0/30377061_0_traj0/
├── rgb-CameraLeft.mkv
├── rgb-CameraRight.mkv
├── depth-CameraLeft.mkv
├── depth-CameraRight.mkv
├── diffuse-color-CameraLeft.mkv
├── environment-CameraLeft.mkv
├── optical-flow-CameraLeft.mkv
├── surface-normal-CameraLeft.mkv
├── semantic-segmentation-CameraLeft.mkv
├── material-segmentation-CameraLeft.mkv
├── camera-CameraLeft.npz
├── camera-CameraRight.npz
├── object-data.npz
└── metadata.json
```

We only include RGB and Depth for the right stereo camera. We do not include the other GT passes for the right camera, primarily to save storage. You can partially recover these by reprojecting using left camera depth and the known stereo baseline. 

`cvdpack unpack` restores the frames from compressed .mkv video files to raw pngs / npys:
```
30377061_0/30377061_0_traj0/
├── CameraLeft/
│   ├── 0000.png .. 0023.png                      rgb
│   ├── diffuse-color_0000.png .. _0023.png
│   ├── environment_0000.png .. _0023.png
│   ├── depth_0000.npy .. _0023.npy
│   ├── optical-flow_0000.npy .. _0023.npy
│   ├── surface-normal_0000.npy .. _0023.npy
│   ├── object_0000.npy .. _0023.npy              semantic segmentation
│   ├── material-index_0000.npy .. _0023.npy      material segmentation
│   └── camera.npz
├── CameraRight/
│   ├── 0000.png .. 0023.png
│   ├── depth_0000.npy .. _0023.npy
│   └── camera.npz
├── object-data.npz
└── metadata.json
```

### Camera Data

`camera-*.npz` holds the intrinsics and per-frame poses, `object-data.npz` the name, pose and bounding box of every object in the room, and `metadata.json` the seed and per-pass render times. cvdpack copies all three unchanged, so they are byte-identical in the packed and unpacked layouts.

Below, `T` is the number of frames (always 24 in this release), `O` the number of objects in the scene (varies per scene; 128-200 is typical), and `H, W` the image height and width (always 720, 1280).

**Coordinate conventions.** World space is Blender's: right handed, +Z up, meters, with the floor near z=0. Camera space is OpenCV's: +X right, +Y down, +Z forward along the view axis. `depth_*.npy` is distance in meters along camera +Z (planar z-depth), not ray length.

**`camera-<cam>.npz`**

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `K` | `T x 3 x 3` | float64 | Pinhole intrinsics in pixels, `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`. Zero skew, no distortion. Every scene in this release uses `fx = fy = 600`, `cx = W/2 = 640`, `cy = H/2 = 360`, unchanging across frames, but read it per frame rather than hardcoding it. The principal point is the exact image center, so `(0, 0)` is the image's top left corner and pixel `(u, v)` has its center at `(u + 0.5, v + 0.5)`. |
| `T` | `T x 4 x 4` | float64 | Camera-to-world rigid transform, one per frame. `T[i, :3, 3]` is the camera position in world space and `T[i, :3, :3]` its rotation, using the OpenCV axes above. Invert it to get world-to-camera. |
| `HW` | `T x 2` | int64 | `(height, width)` in pixels for that frame. |

```python
cam = np.load("CameraLeft/camera.npz")
K, cam_to_world = cam["K"][i], cam["T"][i]

p_cam = (np.linalg.inv(cam_to_world) @ np.append(p_world, 1))[:3]
u, v = (K @ p_cam)[:2] / p_cam[2]                        # project

z = np.load(f"CameraLeft/depth_{i:04d}.npy")[v, u]       # unproject
p_cam = z * (np.linalg.inv(K) @ [u + 0.5, v + 0.5, 1])
p_world = cam_to_world[:3, :3] @ p_cam + cam_to_world[:3, 3]
```

The stereo pair is already rectified: `CameraRight` is `CameraLeft` translated along the camera's +X axis, with identical rotation. The baseline is randomized per scene but constant along a trajectory, and ranges roughly 0.06m to 0.36m. Recover it for a scene with `np.linalg.inv(T_left[i]) @ T_right[i]`.

### Object Data

**`object-data.npz`**

Per-object 3D ground truth for every mesh object in the scene, over the same frames. Axis 0 of every array is one row per object, in arbitrary order, so identify a row by `object_index` or `object_name` rather than by its position.

Rows are mesh objects only. The cameras and the scene's light sources are assigned segmentation ids alongside the meshes but get no row, which is why `object_index` has gaps; those ids never appear in the segmentation pass either, since neither renders as surface geometry. Lamp and ceiling-light *fixtures* are ordinary mesh rows. For camera pose use `T` in `camera-<cam>.npz` above.

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `location_meters` | `O x 3 x T` | float32 | World-space object origin. |
| `rotation_euler_rad` | `O x 3 x T` | float32 | World-space XYZ Euler angles in radians, i.e. `R = Rz @ Ry @ Rx`. Kept continuous across frames rather than wrapped into `[-pi, pi]`, so consecutive frames can be differenced or interpolated directly. |
| `scale` | `O x 3 x T` | float32 | Per-axis scale, applied to the local bbox below. |
| `local_bbox_min` | `O x 3 x T` | float32 | Axis-aligned bounding box in the object's local frame, before scale. |
| `local_bbox_max` | `O x 3 x T` | float32 | |
| `object_index` | `O` | int32 | The value this object takes in the semantic segmentation pass, `object_*.npy`. Unique per object, but neither contiguous nor related to row order. `0` is reserved for background, so no row uses it. |
| `object_name` | `O` | \|S63 | ASCII byte strings, e.g. `b'room_floor.00'` or `b'chair_rand.001'`; Use `.decode()`. |
| `object_type` | `O` | \|S63 | Blender object type, always `b'MESH'` in this release. |
| `data_name` | `O` | \|S63 | Name of the object's mesh datablock. |
| `data_id` | `O` | int32 | Rows sharing one mesh datablock, i.e. instances of the same asset, share a `data_id`. |
| `frame_start`, `frame_end` | scalar | int32 | Inclusive frame range, `0` and `23` here. Column `i` of the pose arrays is frame `frame_start + i`, which is frame file `{frame_start + i:04d}`. |

Pose is location/rotation/scale rather than a 4x4, and the bounding box is stored in the object's local frame, so a world-space box corner is:
```python
from mathutils import Euler   # or build Rz @ Ry @ Rx yourself

R = np.array(Euler(rotation_euler_rad[o, :, i], "XYZ").to_matrix())
corner_world = R @ (scale[o, :, i] * local_corner) + location_meters[o, :, i]
```
where `local_corner` is one of the 8 combinations of `local_bbox_min` and `local_bbox_max`. Projecting those 8 corners with the camera above gives the object's 3D box in the image, which encloses every pixel of that object's segmentation mask.

Most objects are static, typically about a dozen rows per scene move over the 24 frames, but the arrays are per-frame throughout so no row needs special casing. An object with no pose on a given frame is NaN there; this does not occur in the released scenes, but check rather than assume.

This release ships no `object-index-table.json`, so `object-data.npz` is also the only mapping from semantic segmentation ids back to object names. It covers every object in the scene, including ones the trajectory's cameras never see.

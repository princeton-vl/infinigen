# Ground-Truth Annotations

### Changelog

- 07/03/23: Add specification for Blender's built-in annotations. Save built-in annotations as numpy arrays. Add more information to Objects_XXXX_XX_XX.json. Significant changes to built-in segmentation masks (fixes & filenames). Improve visualizations for built-in annotations. Always save camera parameters in frames/. Update docs.

### Agenda

- Save forward and backward flow for both built-in and advanced annotations.
- Compute flow occlusion using forward-backward consistency.
- Export scene geometry in .ply format.

**Want annotations that we don't currently support? [Fill out a request!](https://github.com/princeton-vl/infinigen/issues/new?assignees=&labels=&projects=&template=request.md&title=%5BREQUEST%5D)**

## Default Annotations from Blender

Infinigen can produce some dense annotations using Blender's built-in render passes. Users may prefer to use these annotations over our extended annotation system's since it requires only the bare-minimum installation. It is also able to run without a GPU.

These annotations are produced when using the `--pipeline_configs blender_gt` ground truth extraction config in [manage_datagen_jobs.py](/README.md#generate-images-in-one-command), or can be done manually as shown in the final step of the [Hello-World](/README.md#generate-a-scene-step-by-step) example.

## Advanced Annotation Pipeline :large_blue_diamond:

We also provide a separate pipeline for extracting the full set of annotations from each image or scene. Features only supported using this annotation method will be denoted with :large_blue_diamond:.

This will allow you to use our own `--pipeline_configs opengl_gt` ground truth extraction config, which provides additional labels such as occlusion boundaries, sub-object segmentation, 3D flow and easy 3D bounding boxes. If you do not need these features, we recommend using the [default annotations](#default-annotations-from-blender). This section is intended for computer vision researchers and power-users. 

### Installation

To ensure all submodule dependencies have been properly cloned, run:
```
git submodule init
git submodule update
```

On Ubuntu, run
```
sudo apt-get install libglm-dev libglew-dev libglfw3-dev libgles2-mesa-dev zlib1g-dev
```

If compiling on WSL, additionally run 
```
sudo apt-get install doxygen
sudo apt-get install libxinerama-dev
sudo apt-get install libxcursor-dev
sudo apt-get install libxi-dev
```

On MacOS, run
```
brew install glfw3
brew install glew
```

If you do not have sudo access, you may attempt the following:
- Install dependencies manually and set your $CPATH variables appropriately. 
- Ask your administrator to install them on your behalf (YMMV).

Finally, run
```
bash install.sh opengl
```
Or, if you have already run `install.sh` earlier, you can just run
```
bash worldgen/tools/compile_opengl.sh
```

### Extended Hello-World

Continuing the [Hello-World](/README.md#generate-a-scene-step-by-step) example, we can produce the full set of annotations that Infinigen supports. Following step 3:

4. Export the geometry from blender to disk
```
$BLENDER -noaudio --background --python generate.py -- --seed 0 --task mesh_save -g desert simple --input_folder outputs/helloworld/fine --output_folder outputs/helloworld/saved_mesh
```
5. Generate dense annotations
```
../process_mesh/build/process_mesh --frame 1 -in outputs/helloworld/saved_mesh -out outputs/helloworld/frames
```
6. Summarize the file structure into a single JSON
```
python tools/summarize.py outputs/helloworld # creating outputs/helloworld/summary.json
```
7. (Optional) Select for a segmentation mask of certain semantic tags, e.g. cactus
```
python tools/ground_truth/segmentation_lookup.py outputs/helloworld 1 --query cactus
```

## Specification

**File structure:**

We provide a python script `summarize.py` which will aggregate all relevant output file paths into a JSON:
```
python tools/summarize.py <output-folder>
```
The resulting `<output-folder>/summary.json` will contains all file paths in the form:
```
{
    <type>: {
        <file-ext>: {
            <rig>: {
                <sub-cam>: {
                    <frame>: <file-path> 
                }
            }
        }
    }
}
```

`<rig>` and `<sub-cam>` are typically both "00" in the monocular setting; `<file-ext>` is typically "npy" or "png" for the the actual data and the visualization, respectively; `<frame>` is a 0-padded 4-digit number, e.g. "0013". `<type>` can be "SurfaceNormal", "Depth", etc. For example
`summary_json["SurfaceNormal"]["npy"]["00"]["00"]["0001"]` -> `'frames/SurfaceNormal_0001_00_00.npy'`

*Note: Currently our advanced ground-truth has only been tested for the aspect-ratio 16-9.*

**Depth**

Depth is stored as a 2160 x 3840 32-bit floating point numpy array.

*Path:* `summary_json["Depth"]["npy"]["00"]["00"]["0001"]` -> `frames/Depth_0001_00_00.npy`

*Visualization:* `summary_json["Depth"]["png"]["00"]["00"]["0001"]` -> `frames/Depth_0001_00_00.png`

<p align="center">
<img src="docs/images/gt_annotations/Depth_0001_00_00.png" width="400" />
</p>

The depth and camera parameters can be used to warp one image to another frame by running:
```
python tools/ground_truth/rigid_warp.py <folder> <first-frame> <second-frame>
```

**Surface Normals**

Surface Normals are stored as a 1080 x 1920 x 3 32-bit floating point numpy array.

The coordinate system for the surface normals is +X -> Right, +Y -> Up, +Z Backward.

*Path:* `summary_json["SurfaceNormal"]["npy"]["00"]["00"]["0001"]` -> `frames/SurfaceNormal_0001_00_00.npy`

*Visualization:* `summary_json["SurfaceNormal"]["png"]["00"]["00"]["0001"]` -> `frames/SurfaceNormal_0001_00_00.png`

<p align="center">
<img src="docs/images/gt_annotations/SurfaceNormal_0001_00_00.png" width="400" />
</p>

### Occlusion Boundaries :large_blue_diamond:

Occlusion Boundaries are stored as a 2160 x 3840 png, with 255 indicating a boundary and 0 otherwise.

*Path/Visualization:* `summary_json["OcclusionBoundaries"]["png"]["00"]["00"]["0001"]` -> `frames/OcclusionBoundaries_0001_00_00.png`

<p align="center">
<img src="docs/images/gt_annotations/OcclusionBoundaries_0001_00_00.png" width="400" />
</p>

### Optical Flow

Optical Flow / Scene Flow is stored as a 2160 x 3840 x 3 32-bit floating point numpy array.

*Note: The values won't be meaningful if this is the final frame in a series, or in the single-view setting.*

Channels 1 & 2 are standard optical flow. Note that the units of optical flow are in pixels measured in the resolution of the *original image*. So if the rendered image is 1080 x 1920, you would want to average-pool this array by 2x.

**3D Motion Vectors** :large_blue_diamond:

Channel 3 is the depth change between this frame and the next.

To see an example of how optical flow can be used to warp one frame to the next, run

```
python tools/ground_truth/optical_flow_warp.py <folder> <frame-number>
```

*Path:* `summary_json["Flow3D"]["npy"]["00"]["00"]["0001"]` -> `frames/Flow3D_0001_00_00.npy`

*Visualization:* `summary_json["Flow3D"]["png"]["00"]["00"]["0001"]` -> `frames/Flow3D_0001_00_00.png`

For the built-in versions, replace `Flow3D` with `Flow`.

### Optical Flow Occlusion :large_blue_diamond:

The mask of occluded pixels for the aforementioned optical flow is stored as a 2160 x 3840 png, with 255 indicating a co-visible pixel and 0 otherwise.

*Note: This mask is computed by comparing the face-ids on the triangle meshes at either end of each flow vector. Infinigen meshes often contain multiple faces per-pixel, resulting in frequent false-negatives (negative=occluded). These false-negatives are generally distributed uniformly over the image (like salt-and-pepper noise), and can be reduced by max-pooling the occlusion mask down to the image resolution.*

*Path/Visualization:* `summary_json["Flow3DMask"]["png"]["00"]["00"]["0001"]` -> `frames/Flow3DMask_0001_00_00.png`

### Camera Intrinsics

Infinigen renders images using a pinhole camera model. The resulting camera intrinsics for each frame are stored as a 3 x 3 numpy matrix.

*Path:* `summary_json["Camera Intrinsics"]["npy"]["00"]["00"]["0001"]` -> `frames/K_0001_00_00.npy`

### Camera Extrinsics

The camera pose is stored as a 4 x 4 numpy matrix mapping from camera coordinates to world coordinates.

As is standard in computer vision, the assumed world coordinate system in the saved camera poses is +X -> Right, +Y -> Down, +Z Forward. This is opposed to how Blender internally represents geometry, with flipped Y and Z axes.

*Path:* `summary_json["Camera Pose"]["npy"]["00"]["00"]["0001"]` -> `frames/T_0001_00_00.npy`

### Panoptic Segmentation

Infinigen saves three types of semantic segmentation masks: 1) Object Segmentation 2) Tag Segmentation 3) Instance Segmentation

*Object Segmentation* distinguishes individual blender objects, and is stored as a 2160 x 3840 32-bit integer numpy array. Each integer in the mask maps to an object in Objects_XXXX_XX_XX.json with the same value for the `"object_index"` field. The definition of "object" is imposed by Blender; generally large or complex assets such as the terrain, trees, or animals are considered one singular object, while a large number of smaller assets (e.g. grass, coral) may be grouped together if they are using instanced-geometry for their implementation.

*Instance Segmentation* distinguishes individual instances of a single object from one another (e.g. separate blades of grass, separate ferns, etc.), and is stored as a 2160 x 3840 32-bit integer numpy array. Each integer in this mask is the *instance-id* for a particular instance, which is unique for that object as defined in the Object Segmentation mask and Objects_XXXX_XX_XX.json.


*Paths:*

`summary_json["ObjectSegmentation"]["npy"]["00"]["00"]["0001"]` -> `frames/ObjectSegmentation_0001_00_00.npy`

`summary_json["InstanceSegmentation"]["npy"]["00"]["00"]["0001"]` -> `frames/InstanceSegmentation_0001_00_00.npy`

`summary_json["Objects"]["json"]["00"]["00"]["0001"]` -> `frames/Objects_0001_00_00.json`

*Visualizations:*

`summary_json["ObjectSegmentation"]["png"]["00"]["00"]["0001"]` -> `frames/ObjectSegmentation_0001_00_00.png`

`summary_json["InstanceSegmentation"]["png"]["00"]["00"]["0001"]` -> `frames/InstanceSegmentation_0001_00_00.png`

#### **Tag Segmentation** :large_blue_diamond:

*Tag Segmentation* distinguishes vertices based on their semantic tags, and is stored as a 2160 x 3840 64-bit integer numpy array. Infinigen tags all vertices with an integer which can be associated to a list of semantic labels in `MaskTag.json`. Compared to Object Segmentation, Infinigen's tagging system is less automatic but much more flexible. Missing features in the tagging system are usually possible and straightforward to implement, wheras in the automaically generated Object Segmentation they are not. 

*Paths:*

`summary_json["TagSegmentation"]["npy"]["00"]["00"]["0001"]` -> `frames/TagSegmentation_0001_00_00.npy`

`summary_json["Mask Tags"][<frame>]` -> `fine/MaskTag.json`

*Visualization:*

`summary_json["TagSegmentation"]["png"]["00"]["00"]["0001"]` -> `frames/TagSegmentation_0001_00_00.png`

Generally, most useful panoptic segmentation masks can be constructed by combining the aforementioned three arrays in some way. As an example, to visualize the 2D and [3D bounding boxes](#object-metadata-and-3d-bounding-boxes) for objects with the *blender_rock* semantic tag in the hello world scene, run 
```
python tools/ground_truth/segmentation_lookup.py outputs/helloworld 1 --query blender_rock --boxes
python tools/ground_truth/bounding_boxes_3d.py outputs/helloworld 1 --query blender_rock
```
which will output

<p align="center">
<img src="docs/images/gt_annotations/blender_rock_2d.png" width="400" /> <img src="docs/images/gt_annotations/blender_rock_3d.png" width="400" />
</p>

By ommitting the --query flag, a list of available tags will be printed.

One could also produce a mask for only *flower petals*:

```
python tools/ground_truth/segmentation_lookup.py outputs/helloworld 1 --query petal
```
<p align="center">
<img src="docs/images/gt_annotations/petal.png" width="400" />
</p>

A benefit of our tagging system is that one can produce a segmentation mask for things which are not a distinct object, such as terrain attributes. For instance, we can highlight only *caves* or *warped rocks*

```
python tools/ground_truth/segmentation_lookup.py outputs/helloworld 1 --query cave
python tools/ground_truth/segmentation_lookup.py outputs/helloworld 1 --query warped_rocks
```
<p align="center">
<img src="docs/images/gt_annotations/caves.png" width="400" /> <img src="docs/images/gt_annotations/warped_rocks.png" width="400" />
</p>

### Object Metadata and 3D bounding boxes

Each item in `Objects_0001_00_00.json` also contains other metadata about each object:
```
# Load object meta data
object_data = json.loads((Path("output/helloworld") / summary_json["Objects"]["json"]["00"]["00"]["0001"]).read_text())

# select nth object
obj = object_data[n]

obj["children"] # list of object indices for children
obj["object_index"] # object index, for lookup in the Object Segmentation mask
obj["num_verts"] # number of vertices
obj["num_faces"] # number of faces (n-gons, not triangles)
obj["name"] # obvious
obj["unapplied_modifiers"] # names of unapplied blender modifiers
obj["materials"] # materials used
```

More fields :large_blue_diamond:
```
obj["tags"] # list of tags which appear on at least one vertex 
obj["min"] # min-corner of bounding box, in object coordinates
obj["max"] # max-corner of bounding box, in object coordinates
obj["model_matrices"] # mapping from instance-ids to 4x4 obj->world transformation matrices
```

The **3D bounding box** for each instance can be computed using `obj["min"]`, `obj["max"]`, `obj["model_matrices"]`. For an example, refer to [the bounding_boxes_3d.py example above](#tag-segmentation-large_blue_diamond).

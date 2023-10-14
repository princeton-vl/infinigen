# Downloading and using Pre-Generated Data

## Downloading pre-generated data

All pre-generated data released by the Princeton Vision and Learning lab is hosted here:
[https://infinigen-data.cs.princeton.edu/](https://infinigen-data.cs.princeton.edu/)

You can download these yourself, or use our rudimentary download/untar script as shown in the examples below. To minimize traffic, please use the --cameras, --seeds and --data_types arguments to download only the data you are interested in.

```
# See all available options (recommended):
python -m tools.download_pregenerated_data --help

# Download a few images with geometry ground truth visualization pngs to inspect locally:
python -m tools.download_pregenerated_data --output_folder outputs/my_download --repo_url https://infinigen-data.cs.princeton.edu/ --release_name 2023_10_13_preview --seeds 4bbdd3e0 2d2c1104 --cameras camera_0 --data_types Image_png Depth_png Flow3D_png SurfaceNormal_png OcclusionBoundaries_png 

# Download only the data needed monocular depth (modify as needed for Flow3D, ObjectSegmentation etc):
python -m tools.download_pregenerated_data --output_folder outputs/my_download --repo_url https://infinigen-data.cs.princeton.edu/ --release_name 2023_10_13_preview --cameras camera_0 --data_types Image_png Depth_npy

# Download everything available in a particular datarelease
python -m tools.download_pregenerated_data --output_folder outputs/my_download --repo_url https://infinigen-data.cs.princeton.edu/ --release_name 2023_10_13_preview
```

## Using Infinigen data with a Pytorch-style dataset class

We provide an example pytorch-style dataset class ([dataset_loader.py](../worldgen/tools)) to help load data in our format. 

Assuming you ran the "Download only the data needed monocular depth" example command above, you should be able to use the following example by running `python` from the `worldgen/` folder:

```python
from tools.dataset_loader import get_infinigen_dataset
dataset = get_infinigen_dataset("outputs/my_download", data_types=["Image_png", "Depth_npy"])
print(len(dataset))
print(dataset[0].keys())
```

Note: dataset_loader.py is designed to be separable from the main infinigen codebase; you can copy/move this file into your own codebase, but you must also copy it's dependency `suffixes.py`, or copy `suffixes.py`'s contents into `dataset_loader.py`.

## Ground Truth
Please see [GroundTruthAnnotations.md](./GroundTruthAnnotations.md) for documentation on the various available ground truth, and examples of how they can be used once loaded.
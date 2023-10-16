# Downloading and using Pre-Generated Data

## Downloading pre-generated data

All pre-generated data released by the Princeton Vision and Learning lab is hosted here:
[https://infinigen-data.cs.princeton.edu/](https://infinigen-data.cs.princeton.edu/)

Please run the script below, which will prompt you to determine what ground truth channels, cameras, how many scenes you wish to download.

```bash
cd worldgen
python tools/download_pregenerated_data.py outputs/my_download --release_name 2023_10_13_preview
```

:warning: Downloading all available annotations will require ~30GB per scene. Selecting only the data you need will minimize your bandwidth and disk usage.

## Using Infinigen data with a Pytorch-style dataset class

We provide an example pytorch-style dataset class ([dataset_loader.py](../infinigen/tools)) to help load data in our format. 

If you create a python script in the root of the repo, you can use the following snippet to load your downloaded data. You may need to replace `outputs/my_download` and `data_types` with the values used during download. 
python
from infinigen.tools.dataset_loader import get_infinigen_dataset
dataset = get_infinigen_dataset("outputs/my_download", data_types=["Image_png", "Depth_npy"])
print(len(dataset))
print(dataset[0].keys())
```

Note: dataset_loader.py is designed to be separable from the main infinigen codebase; you can copy/move this file into your own codebase, but you must also copy it's dependency `suffixes.py`, or copy `suffixes.py`'s contents into `dataset_loader.py`.

## Ground Truth
Please see [GroundTruthAnnotations.md](./GroundTruthAnnotations.md) for documentation on the various available ground truth, and examples of how they can be used once loaded.
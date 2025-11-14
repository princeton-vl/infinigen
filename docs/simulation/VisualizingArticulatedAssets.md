# Visualizing Articulated Assets

This document explains how to visualize articulated assets generated across different simulation environments and seeds.
Currently, MuJoCo is the only visualization supported, with plans to add more simulators in the near future.

## MuJoCo Visualization

The MuJoCo visualizer creates videos showing how joints move, highlighting parent and child bodies.

### Basic Usage

To visualize an asset in MuJoCo run from root of repo:

```bash
python infinigen/tools/sim/visualizer.py --asset_name door --nr 3
```

A list of available assets can be seen in `infinigen/assets/sim_objects/mapping.py`.

This command will in one go:

1. For the seeds 0, 1, 2, generate the assets and export to xml.
2. Render animations (mp4) for each joint in the asset
3. Generate an HTML file to view all animations in one file.

### Command Line Options

| Option              | Description                                                                                                                      |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `--asset_name`      | Name of the asset to render (e.g., "door", "lamp"). Should match keys found in `infinigen/assets/sim_objects/mapping.py`.        |
| `--nr`              | Number of seeds. Will export from `seed=0` to `seed=nr-1`.                                                                       |
| `--rand_seeds`      | Flag. Stores true. Use random seeds instead of sequential numbers. Use together with `--nr`.                                     |
| `--seeds`           | Specific seed values to use (e.g., `--seeds 42 101 253`). Can not use both `--seeds` and `--nr`.                                 |
| `--output_dir`      | Directory to save outputs (default: `infinigen/tools/sim/tmp`)                                                                   |
| `--parent_alpha`    | Transparency level for parent geometries (0-1)                                                                                   |
| `--collision_mesh`  | Use collision meshes instead of visual meshes. Defaults to using only visual, no collision meshes.                               |
| `--remove_existing` | Remove existing files before generating new ones.                                                                                |
| `--use_cached_xml`  | Use cached XML files if available. By default exports asset again. If this flag is set, will use existing xml file if available. |

### Viewing Results

After rendering, open the generated HTML file in brower:

```bash
# Example path: infinigen/tools/sim/tmp/vis_door_0_1_2.html
```

The interface provides:

-   Collapsible sections for each seed
-   Joint animations with different camera views
-   Success/failure indicators for rendering issues

## Vis 2

_Coming soon_

## Vis 3

_Coming soon_

# Configuring Infinigen

This document explains how to configure various features of Infinigen. It assumes you have already completed our Installation instructions and [Hello World](HelloWorld.md) example.

:exclamation: By experimenting with Infinigen's code and config files, you will find scenes which crash or cannot be handled on your hardware. Infinigen scenes are randomized, with a long tail of possible scene complexity and thus compute requirements. If you encounter a scene that does not fit your computing hardware, you should try other seeds, use other config files, or follow up for help.

## Overview

Generating scenes with Infinigen typically involves two main python scripts:
1. [infinigen_examples/generate_nature.py](../infinigen_examples/generate_nature.py) - our example scene composition script, which invokes and places assets to create a realistic nature scene.
1. [manage_jobs.py](../infinigen/datagen/manage_jobs.py) - a script which invokes the above scene composition script many times to generate a useful dataset.

`manage_jobs.py` controls how many and what jobs will be run, and `infinigen_examples/generate_nature.py` determines what will happen during those jobs. Ultimately both programs must be configured correctly in order to make useful data.

### Scene tasks

To complete one Infinigen scene, `manage_jobs.py` will run several "tasks", each composed of a single execution of `infinigen_examples/generate_nature.py`(You ran similar processes yourself in the step by step part of "Hello World"). 

Typically, these tasks are as follows:
   1. `coarse`, which generates coarse terrain shape, puts placeholders for creatures and large trees / obstacles, and generates all small assets (twigs, particles, etc) that will be "instanced", IE scattered across the terrain with repetition.
   1. `populate`, which replaces the placeholders from the previous step with unique, detailed assets (trees, creatures, etc) provided they are necessary in the final camera trajectory.
   1. `rendershort` - render the scene using Blender's CYCLES renderer. If working on a video, this step renders a short block (usually 16 frames) of adjacent video.
      1. `renderbackup` - retry the above step with more RAM etc, to make sure high-cost, detailed scenes (which can crash in the previous step) still complete.
   1. (if enabled via `--pipeline_configs`)`blender_gt` or `opengl_gt` - extract ground truth labels for AI training etc.

`coarse` and `populate` typically run once per scene, whereas `fine_terrain` as well as any render or ground truth tasks may run many times for different cameras or video timesteps.

The outputs for all tasks belonging to a particular scene will be stored in the `outputs/MYJOB/SEED`, where MYJOB was the folder name chosen in the `--output_folder` argument, and `SEED` is the random seed used to create the scene. Each task creates its own detailed log files in `outputs/MYJOB/SEED/logs/TASKNAME.log` and `outputs/MYJOB/SEED/logs/TASKNAME.err` - <b> please check these logs to see detailed error messages if your jobs crash </b>.

Infinigen is designed to run many independent scenes in paralell. This means that even if running all the steps for each scene can take some time, we can still achieve good throughput and utilization (provided it has enough resources - most laptops only have enough RAM for one scene at a time). 

#### Overrides and Config Files

Both `manage_jobs.py` and `infinigen_examples/generate_nature.py` can be configured via the commandline or config files, using [Google's "Gin Config"](https://github.com/google/gin-config). Gin allows you to insert new default keyword arguments ("kwargs") for any function decorated with `@gin.configurable`; many such functions exist in our codebase, and via gin overrides you can create datsets suiting many diverse applications, as is explained in the coming sections.

To use gin, simply add commandline arguments such as `-p compose_scene.rain_particles_chance = 1.0` to override the chance of rain, or `--pipeline_overrides iterate_scene_tasks.frame_range=[1,25]` to set a video's length to 24 frames. You can chain many statements together, separated by spaces, to configure many parts of the system at once. These statements depend on knowing the python names of the function and keyword argument you wish to override. To find parameters you wish to override, you should browse `infinigen_examples/configs/base.gin` and other configs, or `infinigen_examples/generate_nature.py` and the definitions of any functions it calls. Better documentation and organization of the available parameters will come in future versions.

If you find a useful and related combination of these commandline overrides, you can write them into a `.gin` file in `infinigen_examples/configs`. Then, to load that config, just include the name of the file into the `--configs`. If your overrides target `manage_jobs` rather than `infinigen_examples/generate_nature.py` you should place the config file in `datagen/configs` and use `--pipeline_configs` rather than `--configs`. 

Our `infinigen_examples/generate_nature.py` driver always loads [`infinigen_examples/configs/base.gin`][../infinigen_examples/configs/base.gin], and you can inspect / modify this file to see many common and useful gin override options.

`infinigen_examples/generate_nature.py` also expects that one file from (configs/scene_types/)[infinigen_examples/configs/scene_types] will be loaded. These scene_type configs contain gin overrides designed to encode the semantic constraints of real natural habitats (e.g. `infinigen_examples/configs/scene_types/desert.gin` causes sand to appear and cacti to be more likely).

### Moving beyond "Hello World"

Now that you understand the two major python programs and how to configure them, you may notice and wonder about the many configs/overrides provided in our original one-command "Hello World" example:

```
# Original hello world command
python -m infinigen.datagen.manage_jobs --output_folder outputs/hello_world --num_scenes 1 --specific_seed 0 \
--configs desert.gin simple.gin --pipeline_configs local_16GB.gin monocular.gin blender_gt.gin \ 
--pipeline_overrides LocalScheduleHandler.use_gpu=False
```

Here is a breakdown of what every commandline argument does, and ideas for how you could change them / swap them out:
   - `--output_folder outputs/hello_world` - change this to change where the files end up
   - `--specific_seed 0` forces the system to use a random seed of your choice, rather than choosing one at random. Change this seed to get a different random variation, or remove it to have the program choose a seed at random
   - `--num_scenes` decides how many unique scenes the program will attempt to generate before terminating. Once you have removed `--specific_seed`, you can increase this to generate many scenes in sequence or in paralell. 
   - `--configs desert.gin simple.gin` forces the command to generate a desert scene, and to do so with relatively low mesh detail, low render resolution, low render samples, and some asset types disabled.
      - Do `--configs snowy_mountain.gin simple.gin` to try out a different scene type (`snowy_mountain.gin` can instead be any scene_type option from `infinigen_examples/configs/scene_types/`)
      - Remove the `desert.gin` and just specify `--configs simple.gin` to use random scene types according to the weighted list in `datagen/configs/base.gin`.
      - You have the option of removing `simple.gin` and specify neither of the original configs. This turns off the many detail-reduction options included in `simple.gin`, and will create scenes closer to those in our intro video, albeit at significant compute costs. Removing `simple.gin` will likely cause crashes unless using a workstation/server with large amounts of RAM and VRAM. You can find more details on optimizing scene content for performance [here](#config-overrides-for-mesh-detail-and-performance).
   - `--pipeline_configs local_16GB.gin monocular.gin blender_gt.gin`
      - `local_16GB.gin` specifies to run only a single scene at a time, and to run each task as a local python process. See [here](#configuring-available-computing-resources) for more options
      - `monocular.gin` specifies that we want a single image per scene, not stereo or video. See [here](#rendering-video-stereo-and-other-data-formats) for more options.
      - `blender_gt.gin` specifies to extract ground truth labels (depth, surface normals, etc) using Blender's in-built render. If you do not need these, remove this config to save on runtime.
   - `--pipeline_overrides LocalScheduleHandler.use_gpu=False` tells the system not to look for available GPUs, and to _not_ make them available to any jobs. This is intended only to make the Hello World easier to run, and work on non-NVIDIA systems. Please [click here](#using-gpu-acceleration) for full instructions on using a GPU.
            
## Commandline Options in Detail

This section highlights commandline options to help you deploy Infinigen on your compute, enable GPU usage, render various datasets, tune resolution/compute costs, and more.

[Click here to skip to Example Commands](#example-commands)

### Configuring available computing resources

`datagen/configs/compute_platform` provides several preset configs to help you run the appropriate number and type of Infinigen tasks suitable to your hardware. <br>These configs are mutually exclusive, but you must include at least one.</br>

`local_16GB.gin` through `local_256GB.gin` are intended for laptops, desktop workstations, or a single headless node. These configs run each task as a child process on the same machine that ran `manage_jobs.py`. Each config runs the right number of concurrent jobs for the specified amount of RAM. If you have a different amount of RAM than the options provided, you should override `manage_jobs.num_concurrent`to your amount of system RAM divided by 20GB. Ultimately, the amount of concurrent process you can run will depend on the format (single-image vs. video) and complexity (detail, clutter, etc) of the scenes you wish to generate - see [here](#config-overrides-for-mesh-detail-and-performance) to customize.

`slurm.gin` is a special config designed to deploy Infinigen onto SLURM computing clusters. When using this config, each task will be executed as a remote slurm job (using [submitit](https://github.com/facebookincubator/submitit)), with time-limit/CPU/RAM requests set as specified in `slurm.gin`. If your group has a SLURM partition name, please set `PARTITION = "mygroupname"` in `slurm.gin`. If you are able to run very long SLURM jobs (>= ~3 days) you may also consider submitting `manage_jobs` with `local_256GB.gin` as a single long slurm job. 

Please submit a Github Issue for help with deploying Infinigen on your compute.

### Using GPU acceleration

Infinigen currently only supports NVIDIA GPUs. Infinigen can use a GPU in accelerate the following tasks:
   1. In the `rendershort`,  `renderbackup`, `blender_gt` tasks, Blender's CYCLES rendering engine can use a GPU to massively accelerate ray-tracing. This all-but-essential for video rendering, which require a large amount of render jobs. AMD/etc support for this step is possible but not currently supported/implemented.
   1. In the `fine_terrain` step, our custom marching cubes terrain generator has been specialized to run with CUDA, to massively improve runtime and allow higher detail terrain.
   1. In the `opengl_gt` step (if enabled) our custom ground truth code uses OpenGL and thus requires access to a GPU. 

To enable these GPU accelerated steps:
   - First, if you are using a `local_*.gin` pipeline config, you must first remove `--pipeline_overrides LocalScheduleHandler.use_gpu=False` from our Hello World command, or otherwise ensure this value is set to true via configs/overrides. This will make the GPU _visible_ to each child process, and will cause _rendering_ to automatically detect and use the GPU. `slurm.gin` assumes GPUs will be available by default, set it's GPU request amounts to 0 if this is not the case for your cluster.
   - To enable GPU-acceleration for `fine_terrain`, you must ensure that installation was run on a machine with CUDA, then add `cuda_terrain.gin` to your `--pipeline_configs`.
   - OpenGL GT can be enabled described in [Extended ground-truth](GroundTruthAnnotations.md)

Even if GPU access is enabled, *you will likely not see the GPU at 100% usage*. Both blender and our code require some CPU setup time to build acceleration data-structures before GPU usage can start, and even then, a single render job may not saturate your GPU's FLOPS. 
   - If you are using a GPU with >=32 GB VRAM, you may consider `--pipeline_override LocalScheduleHandler.jobs_per_gpu=2` to put 2 render jobs on each GPU and increase utilization. 

If you have more than one GPU and are using a `local_*.gin` compute config, each GPU-enabled task will be assigned to a single GPU, and you will need to run many concurrent jobs to keep your GPUs fed. Blender natively supports multi-GPU rendering, but we do not use this by default - create a Github Issue \[REQUEST\] if your application requires this.

### Rendering Video, Stereo and other data formats

Generating a video, stereo or other dataset typically requires more render jobs, so we must instruct `manage_jobs.py` to run those jobs. `datagen/configs/data_schema/` provides many options for you to use in your `--pipeline_configs`, including `monocular_video.gin` and `stereo.gin`. <br> These configs are typically mutually exclusive, and you must include at least one </br>

:exclamation: Our terrain system resolves its signed distance function (SDF) to view-specific meshes, which must be updated as the camera moves. For video rendering, we strongly recommend using the `high_quality_terrain` config to avoid perceptible flickering and temporal aliasing. This config meshes the SDF at very high detail, to create seamless video. However, it has high compute costs, so we recommend also using `--pipeline_config cuda_terrain` on a machine with an NVIDIA GPU. For applications with fast moving cameras, you may need to update the terrain mesh more frequently by decreasing `iterate_scene_tasks.view_block_size = 16`.

To create longer videos, modify `iterate_scene_tasks.frame_range` in `monocular_video.gin` (note: we use 24fps video by default). `iterate_scene_tasks.view_block_size` controls how many frames will be grouped into each `fine_terrain` and render / ground-truth task.

If you need more than two cameras, or want to customize their placement, see `infinigen_examples/configs/base.gin`'s `camera.spawn_camera_rigs.camera_rig_config` for advice on existing options, or write your own code to instantiate a custom camera setup.

### Config Overrides to Customize Scene Content

:bulb: If you only care about few specific assets, or want to export Infinigen assets to another project, instead see [Generating individual assets](GeneratingIndividualAssets.md).

You can achieve a great deal of customization by browsing and editing `infinigen_examples/configs/base.gin` - e.g. modifying cameras, lighting, asset placement, etc.
   - `base.gin` only provides the default values of these configs, and may be overridden by scene_type configs. To apply a setting globally across all scene types, you should put them in a new config placed at the end of your `--configs` argument (so that it's overrides are applied last), or use commandline overrides.

However, many options exist which are not present in base.gin. At present, you must browse `infinigen_examples/generate_nature.py` to find the part of the code you wish to customize, and look through the relevant code for what more advanced @gin.configurable functions are available. You can also add @gin.configurable to most functions to allow additional configuration. More documentation on available parameters is coming soon.

For most steps of `infinigen_examples/generate_nature.py`'s `compose_scene` function, we use our `RandomStageExecutor` wrapper to decide whether the stage is run, and handle other bookkeeping. This means that if you want to decide the probability with which some asset is included in a scene, you can use the gin override `compose_scene.trees_chance=1.0` or something similar depending on the name string provided as the first argument of the relevant  run_stage calls in this way, e.g. `compose_scene.rain_particles_chance=0.9`to make most scenes rainy, or `compose_scene.flowers_chance=0.1` to make flowers rarer.

A common request is to just turn off things you don't want to see, which can be achieved by adding `compose_scene.trees_chance=0.0` or similar to your `-p` argument or a loaded config file. To conveniently turn off lots of things at the same time, we provide configs in `infinigen_examples/configs/disable_assets` to disable things like all creatures, or all particles.

You will also encounter configs using what we term a "registry pattern", e.g. `infinigen_examples/configs/base_surface_registry.gin`'s `ground_collection`. "Registries", in this project, are a list of discrete generators, with weights indicating how relatively likely they are to be chosen each time the registry is sampled. 
   - For example, in `base_surface_registry.gin`, `surface.registry.beach` specifies `("sand", 10)` to indicate that sand has high weight to be chosen to be assigned for the beach category. 
   - Weights are normalized by their overall sum to obtain a probability distribution. 
   - Name strings undergo lookup in the relevant source code folders, e.g. the name "sand" in a surface registry maps to `infinigen/assets/materials/sand.py`.

### Config Overrides for mesh detail and performance

The quantity, diversity and detail of assets in a scene drastically affects RAM/VRAM requirements and runtime. This section will highlight configurable parameters that may help tune Infinigen to run better on limited hardware, or that could be increased to create larger more detailed scenes.

Infinigen can generate meshes at a wide range of resolutions. Many gin-configurable options exist to customize how detailed our meshes will be. <br>These options, as well as the choice of scene type configs, are the most effective ways to decrease compute costs.</br>
- All mesh resolutions are defined in terms of pixels-per-face. Increasing/decreasing the `compose_scene.generate_resolution` will have corresponding effects on geometry detail. If you wish to render the same mesh at a different resolution, override `render_image.render_resolution_override`
- `OpaqueSphericalMesher.pixels_per_cube` and `TransparentSphericalMesher.pixels_per_cube`. You can increase them from their default, 2 pixels per marching cube, to 4 (as seen in `dev.gin`) or higher, in order to reduce terrain resolution and cost. Low resolution terrain will cause noticeable artifacts in videos, you must use `high_quality_terrain.gin` if rendering videos with moving cameras.
- `target_face_size.global_multiplier` controls the resolution of all other assets. Increase it to 4 or higher to reduce the compute cost of non-terrain assets like plants and creatures.
- All of these options have diminishing returns - a minimal amount of geometry or data is always generated to help define the shape of each asset, which may be larger than the final geometry if you set resolution very low.

Infinigen curbs memory costs by only populating assets up to a certain distance away from the camera (except for terrain, which is essentially unbounded). `base.gin` contains many options to customize how far away, and thus how many, assets will be placed:
   - `placement.populate_all.dist_cull` controls the maximum distance to populate placeholders (for assets such as trees, cacti, creatures, etc). Reducing this will curb the number of assets and especially the number of trees, which can be quite expensive.
   - `compose_scene.inview_distance`control the maximum distance to scatter plants/rocks/medium-sized-objects. Reducing this will reduce the number of poses for these assets, but will not reduce the number of unique underlying meshes, so may have diminishing returns.
   - Similarly to the above, `compose_scene.near_distance` controls the maximum distance to scatter tiny particles like pine needles.
   - Infinigen does not populate assets which are far outside the camera frustrum. You may attempt to reduce camera FOV to minimize how many assets are in view, but be warned there will be minimal or significantly diminishing returns on performance, due to the need to keep out-of-view assets loaded to retain accurate lighting/shadows.

We also provide `infinigen_examples/configs/performance/dev.gin`, a config which sets many of the above performance parameters to achieve lower scenes. We often use this config to obtain previews for development purposes, but it may also be suitable for generating lower resolution images/scenes for some tasks.

Our current system determines asset mesh resolution based on the _closest distance_ it comes to the camera during an entire trajectory. Therefore, longer videos are more expensive, as more assets will be closer to the camera at some point in the trajectory. Options exist to re-generate assets at new resolutions over the course of a video to curb these costs - please make a Github Issue for advice. 

If you find yourself bottlenecked by GPU time, you should consider the following options:
   - Render single images or stereo images, rather than video, such that less render jobs are required for each CPU-bound `coarse`/`populate` job
   - Reduce `base.gin`'s `full/render_image.num_samples = 8192` or `compose_scene.generate_resolution = (1920, 1080)`. This proportionally reduces rendering FLOPS, with some diminishing returns due to BVH setup time.
   - If your GPU(s) are _underutilized_, try the reverse of these tips.

Some scene type configs are also generally more expensive than others. `forest.gin` and `coral.gin` are very expensive due to dense detailed fauna, wheras `artic` and `snowy_mountain` are very cheap. Low-resource compute settings (<64GB) of RAM may only be able to handle a subset of our `infinigen_examples/configs/scene_type/` options, and you may wish to tune the ratios of scene_types by editing `datagen/configs/base.gin` or otherwise overriding `sample_scene_spec.config_distribution`. 

### Other `manage_jobs.py` commandline options

Please run `pythom -m infinigen.datagen.manage_jobs --help` for an up-to-date description of other commandline arguments. We always use `--cleanup big_files --warmup_sec 30000` for large render jobs. Optionally, you can also log render progress to Weights & Biases.

## Example Commands

Below we provide many example commands we have found useful to generate scenes/images/videos.

If you have any issues with these commands, or wish to customize them to your needs, <br>we encourage you to read the above documentation</br>, then follow up on our Github Issues for help.

All commands below are shown with using `local_256GB` config, but you can attempt to swap this for any compute config as discussed in [Configuring available computing resources](#configuring-available-computing-resources).

#### Creating videos similar to the intro video

Most videos in the "Introducing Infinigen" launch video were made using commands similar to the following:

````
python -m infinigen.datagen.manage_jobs --output_folder outputs/my_videos --num_scenes 500 \
    --pipeline_config slurm monocular_video cuda_terrain opengl_gt \
    --cleanup big_files --warmup_sec 60000 --config high_quality_terrain
````

#### Creating large-scale stereo datasets

````
python -m infinigen.datagen.manage_jobs --output_folder outputs/stereo_data --num_scenes 10000 \
    --pipeline_config slurm stereo cuda_terrain opengl_gt \
    --cleanup big_files --warmup_sec 60000 --config high_quality_terrain
````

#### Creating a few low-resolution images to your test changes

```
screen python -m infinigen.datagen.manage_jobs --output_folder outputs/dev --num_scenes 50 \
    --pipeline_config slurm monocular cuda_terrain \
    --cleanup big_files --warmup_sec 1200 --configs dev
```

#### Creating datasets with specific properties or constraints:

These commands are intended as inspiration - please read docs above for more advice on customizing all aspects of Infinigen.

<b> Create images that always have rain: </b>
```
python -m infinigen.datagen.manage_jobs --output_folder outputs/my_videos --num_scenes 500 \
    --pipeline_config slurm monocular cuda_terrain opengl_gt \
    --cleanup big_files --warmup_sec 30000  \
    --overrides compose_scene.rain_particles_chance=1.0
```

:bulb: You can substitute the `rain_particles` in `rain_particles_chance` for any `run_stage` name argument string in `infinigen_examples/generate_nature.py`, such as `trees` or `ground_creatures`.

<b> Create images that only have terrain: </b>
```
python -m infinigen.datagen.manage_jobs --output_folder outputs/my_videos --num_scenes 500 \
    --pipeline_config slurm monocular cuda_terrain opengl_gt \
    --cleanup big_files --warmup_sec 30000 --config no_assets
```
:bulb: You can substitute "no_assets" for `no_creatures` or `no_particles`, or the name of any file under `infinigen_examples/configs`. The command shown uses `infinigen_examples/configs/disable_assets/no_assets.gin`.

<b> Create videos at birds-eye-view camera altitudes: </b>

```
python -m infinigen.datagen.manage_jobs --output_folder outputs/my_videos --num_scenes 500 \
    --pipeline_config slurm monocular_video cuda_terrain opengl_gt \
    --cleanup big_files --warmup_sec 30000 --config high_quality_terrain \
    --overrides camera.camera_pose_proposal.altitude=["uniform", 20, 30]
```

:bulb: The command shown is overriding `infinigen_examples/configs/base.gin`'s default setting of `camera.camera_pose_proposal.altitude`. You can use a similar syntax to override any number of .gin config entries. Separate multiple entries with spaces. 

<b> Create 1 second video clips: </b>
```
python -m infinigen.datagen.manage_jobs --output_folder outputs/my_videos --num_scenes 500 \
    --pipeline_config slurm monocular_video cuda_terrain opengl_gt \
    --cleanup big_files --warmup_sec 30000 --config high_quality_terrain \
    --pipeline_overrides iterate_scene_tasks.frame_range=[1,25]
```

:bulb: This command uses `--pipeline_overrides` rather than `--overrides` since it is providing a gin override to the `manage_jobs.py` process, not some part of the main `infinigen_examples/generate_nature.py` driver.

# Generating Fluid Simulations

This documentation details how to generate fire and water simulations like those shown in the Infinigen launch video. It assumes you have already completed [Installation.md] and [HelloWorld.md]. Fluid simulations require *significant* computational resources.

## Setup

To generate fluids, you run install.sh with the optional FLIP-Fluids setup step (`bash install.sh flip_fluids`), or, please run `bash worldgen/tools/install/compile_flip_fluids.sh` to install flip fluids now.

## Example Commands

#### Generate videos of random scene types, with simulated fire generated on the fly when needed
```
python -m tools.manage_datagen_jobs --output_folder outputs/onthefly  --num_scenes 10 \
    --pipeline_config slurm_high_memory.gin monocular_video.gin \
    --config fast_terrain_assets.gin use_on_the_fly_fire.gin \
    --wandb_mode online --cleanup none --warmup_sec 12000 
```

#### Generate videos of valley scenes with simulated rivers
```
python -m tools.manage_datagen_jobs --output_folder /n/fs/pvl-renders/kkayan/river --num_scenes 10 \
    --pipeline_config slurm_high_memory.gin monocular_video.gin opengl_gt.gin cuda_terrain.gin \
    --pipeline_overrides iterate_scene_tasks.frame_range=[100,244] \ 
    --config simulated_river.gin no_assets.gin no_creatures.gin fast_terrain_assets.gin \
    --wandb_mode online --cleanup none --warmup_sec 12000
```


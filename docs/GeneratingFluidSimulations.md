# Generating Fluid Simulations

This documentation details how to generate fire and water simulations like those shown in the Infinigen launch video. It assumes you have already completed [Installation.md], [HelloWorld.md]. Fluid simulations require *significant* computational resources.

## Setup

Before you can generate fluids, you must run an additional installation step: `bash scripts/install/compile_flip_fluids.sh`, which will compile and install the flip fluids addon. This step has to be completed after having already installed the rest of infinigen.

## Example Commands

#### Generate a video of a single scene with simulated fire generated on the fly
```
python -m infinigen.datagen.manage_jobs --specific_seed 3930249d --output_folder outputs/fire --num_scenes 1 --pipeline_config local_256GB.gin monocular_video.gin --cleanup none --config plain.gin fast_terrain_assets.gin use_on_the_fly_fire.gin
```
Because fluid simulation takes a long time, the fire resolution can be reduced in use_on_the_fly_fire.gin, by setting `set_obj_on_fire.resolution = {resolution}`. This will reduce the fire quality but speed up the simulation.

#### Generate a video of a single valley scene with simulated river
```
python -m infinigen.datagen.manage_jobs --specific_seed 61fc881a --output_folder outputs/river --num_scenes 1 --pipeline_config local_256GB.gin monocular_video.gin opengl_gt.gin cuda_terrain.gin --pipeline_overrides iterate_scene_tasks.frame_range=[100,244] --config river.gin simulated_river.gin no_assets.gin no_creatures.gin fast_terrain_assets.gin --cleanup none 
```
Similar to fire, the simulation can be sped up by reducing the resolution. In simulated_river.gin, the resolution can be modified by setting `make_river.resolution = {resolution}`. The simulation can also be sped up by reducing the simulation duration in simulated_river.gin by setting `make_river.simulation_duration = {duration}`. For instance, before running the command above, the duration can be reduced to a number greater than 200 since that is the last frame of the video.

Also, note that this command will produce a scene without assets to speed up the process. However, the liquids generally interact with the objects by splashing on them, and a scene like this can be produced by removing the `no_assets.gin` option in the above command. 

#### Generate videos of random scene types, with simulated fire generated on the fly when needed
```
python -m infinigen.datagen.manage_jobs --output_folder outputs/onthefly  --num_scenes 10 \
    --pipeline_config slurm_high_memory.gin monocular_video.gin \
    --config fast_terrain_assets.gin use_on_the_fly_fire.gin \
    --cleanup none --warmup_sec 12000 
```

#### Generate videos of valley scenes with simulated rivers
```
python -m infinigen.datagen.manage_jobs --output_folder /n/fs/pvl-renders/kkayan/river --num_scenes 10 \
    --pipeline_config slurm_high_memory.gin monocular_video.gin opengl_gt.gin cuda_terrain.gin \
    --pipeline_overrides iterate_scene_tasks.frame_range=[100,244] \ 
    --config simulated_river.gin no_assets.gin no_creatures.gin fast_terrain_assets.gin \
    --cleanup none --warmup_sec 12000
```



## Using Pre-Generated Fire

High-resolution fluid simulations take a long time, so assets on fire can pre-generated and imported in the scene instead of being baked on-the-fly. This allows for fire to be simulated once, instead of every time a scene is generated. To pre-generate fire assets of all types (bush, tree, creature, cactus, boulder), run:
```
python -m infinigen.tools.submit_asset_cache -f {fire_asset_folder} -n 1 -s -40 -d 184
```
where `fire_asset_folder` is where you want to save the fire. The number of assets per type, start frame and duration can be adjusted.  

If you only want to pre-generate one type of asset once, run:
```
python -m infinigen.assets.fluid.run_asset_cache -f {fire_asset_folder} -a {asset} -s {start_frame} -d {simulation_duration}
```
where `fire_asset_folder` is where you want to save the fire. `asset` can be one of `CachedBushFactory`, `CachedTreeFactory`, `CachedCactusFactory`, `CachedCreatureFactory`, `CachedBoulderFactory`. 

### Import pre-generated fire when generating a scene
After fire is pre-generated with one of the previous commands, edit config/use_cached_fire.gin and set the `FireCachingSystem.asset_folder` variable to `fire_asset_folder` you used when pre-generating fire. After this `use_cached_fire.gin` can be used instead of `use_on_the_fly_fire.gin` when generating a scene. This will import the fire from the folder it is saved instead of simulating it on-the-fly. 
#### Example Command
```
python -m infinigen.datagen.manage_jobs --specific_seed 3930249d --output_folder outputs/fire --num_scenes 1 --pipeline_config local_256GB.gin monocular_video.gin --cleanup none --config plain.gin fast_terrain_assets.gin use_cached_fire.gin
```

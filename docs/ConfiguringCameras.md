# Configuring Cameras

This document gives examples of how to configure cameras in Infinigen for various computer vision tasks.

### Example Commands

##### Stereo Matching

Generate many nature, scenes each with 1 stereo camera:
```bash
python -m infinigen.datagen.manage_jobs --output_folder outputs/stereo_nature --num_scenes 30 \
--pipeline_config stereo.gin local_256GB.gin cuda_terrain.gin blender_gt.gin --configs high_quality_terrain
```

Generate many indoor rooms, each with 20 stereo cameras:
```bash
python -m infinigen.datagen.manage_jobs --output_folder outputs/stereo_indoors --num_scenes 30 \
--pipeline_configs local_256GB.gin stereo.gin blender_gt.gin indoor_background_configs.gin --configs singleroom \
--pipeline_overrides get_cmd.driver_script='infinigen_examples.generate_indoors' \
--overrides camera.spawn_camera_rigs.n_camera_rigs=20 compute_base_views.min_candidates_ratio=2 compose_indoors.terrain_enabled=False compose_indoors.restrict_single_supported_roomtype=True
```

We recommend 20+ cameras per indoor room since room generation is not view-dependent and can be rendered from many angles. This helps overall GPU utilization since many frames are rendered per scene generated. In nature scenes, the current camera code would place cameras very far apart, meaning visible content does not overlap and there is minimal benefit to simply increasing `n_camera_rigs` in nature scenes without also customizing their arrangement. Thus, if you wish to extract more stereo frames per nature scene, we recommend instead rendering a low fps video using the "Random Walk Videos" commands below. 

##### Random Walk Videos

Nature video, slow & smooth random walk camera motion:
```bash
python -m infinigen.datagen.manage_jobs --output_folder outputs/video_smooth_nature --num_scenes 30 \
--pipeline_config monocular_video.gin local_256GB.gin cuda_terrain.gin blender_gt.gin --configs high_quality_terrain \
--pipeline_overrides iterate_scene_tasks.cam_block_size=24
```

Nature video, fast & noisy random walk camera motion:
```bash
python -m infinigen.datagen.manage_jobs --output_folder outputs/video_smooth_nature --num_scenes 30 \
--pipeline_config monocular_video.gin local_256GB.gin cuda_terrain.gin blender_gt.gin --configs high_quality_terrain noisy_video \
--pipeline_overrides iterate_scene_tasks.cam_block_size=24 --overrides configure_render_cycles.adaptive_threshold=0.05
```

Indoor video, slow moving camera motion:
```bash
python -m infinigen.datagen.manage_jobs --output_folder outputs/video_slow_indoor --num_scenes 30 \
--pipeline_configs local_256GB.gin monocular_video.gin blender_gt.gin indoor_background_configs.gin --configs singleroom \
--pipeline_overrides get_cmd.driver_script='infinigen_examples.generate_indoors' \
--overrides compose_indoors.terrain_enabled=False compose_indoors.restrict_single_supported_roomtype=True AnimPolicyRandomWalkLookaround.speed=0.5 AnimPolicyRandomWalkLookaround.step_range=0.5 compose_indoors.animate_cameras_enabled=True
```

:warning: Random walk camera generation is very unlikely to find paths between indoor rooms, and therefore will fail to generate long or fast moving videos for indoor scenes. We will followup soon with a pathfinding-based camera trajectory generator to handle these cases. 

##### Multi-view Camera Arrangement (for Multiview Stereo, NeRF, etc.)

Many tasks require cameras placed in a roughly circular arrangement. Below with some noise added to their angle, roll, pitch, and yaw with respect to the object.

<p align="center">
  <img src="images/multiview_stereo/mvs_indoors.png"/>
  <img src="images/multiview_stereo/mvs_indoors_2.png">
  <img src="images/multiview_stereo/mvs_nature.png"/>
  <img src="images/multiview_stereo/mvs_ocean.png"/>
</p>

Generate a quick test scene (indoor room with no furniture etc) with 5 multiview cameras:
```bash
python -m infinigen.datagen.manage_jobs --output_folder outputs/mvs_test --num_scenes 1 --configs multiview_stereo.gin fast_solve.gin no_objects.gin --pipeline_configs local_256GB.gin monocular.gin blender_gt.gin cuda_terrain.gin indoor_background_configs.gin --overrides camera.spawn_camera_rigs.n_camera_rigs=5 compose_nature.animate_cameras_enabled=False compose_indoors.restrict_single_supported_roomtype=True --pipeline_overrides get_cmd.driver_script='infinigen_examples.generate_indoors' iterate_scene_tasks.n_camera_rigs=5
```

Generate a dataset of indoor rooms with 30 multiview cameras:
```bash
python -m infinigen.datagen.manage_jobs --output_folder outputs/mvs_indoors --num_scenes 30 --pipeline_configs local_256GB.gin monocular.gin blender_gt.gin indoor_background_configs.gin --configs singleroom.gin multiview_stereo.gin --pipeline_overrides get_cmd.driver_script='infinigen_examples.generate_indoors' iterate_scene_tasks.n_camera_rigs=30 --overrides compose_indoors.restrict_single_supported_roomtype=True camera.spawn_camera_rigs.n_camera_rigs=30 
```

Generate a dataset of nature scenes with 30 multiview cameras:
```bash
python -m infinigen.datagen.manage_jobs --output_folder outputs/mvs_nature --num_scenes 30 --configs multiview_stereo.gin --pipeline_configs local_256GB.gin monocular.gin blender_gt.gin cuda_terrain.gin --overrides camera.spawn_camera_rigs.n_camera_rigs=30 compose_nature.animate_cameras_enabled=False --pipeline_overrides iterate_scene_tasks.n_camera_rigs=30
```

##### Custom camera arrangement

Camera poses can be easily manipulated using the Blender API to create any camera arrangement you wish

For example, you could replace our `pose_cameras` step in `generature_nature.py` or `generate_indoors.py` with code as follows:

```python
for i, rig in enumerate(camera_rigs):
  rig.location = (i, 0, np.random.uniform(0, 10))
  rig.rotation_euler = np.deg2rad(np.array([90, 0, 180 * i / len(camera_rigs)]))
```

If you wish to animate the camera rigs to move over the course of a video, you would use code similar to the following:

```python

for i, rig in enumerate(camera_rigs):

  for t in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
    
    rig.location = (t, i, 0) 
    rig.keyframe_insert(data_path="location", frame=t)

    rig.rotation_euler = np.deg2rad(np.array((90, 0, np.random.uniform(-10, 10))))
    rig.keyframe_insert(data_path="rotation_euler", frame=t)
```
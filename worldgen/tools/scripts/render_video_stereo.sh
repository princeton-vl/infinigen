HOSTFIRST=$(hostname | tr "." "\n" | head -n 1)
JOBNAME=$(date '+%m_%d_%H_%M').$HOSTFIRST.$1

python -m tools.manage_datagen_jobs --blender_path ../blender/blender --output_folder outputs/$JOBNAME \
    --num_scenes 1000 --pipeline_config $@ stereo cuda_terrain opengl_gt \
    --wandb_mode online --cleanup big_files --upload --warmup_sec 40000 \
    --override compose_scene.generate_resolution=[1280,720] \

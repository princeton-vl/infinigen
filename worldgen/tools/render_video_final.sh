HOSTFIRST=$(hostname | tr "." "\n" | head -n 1)
JOBNAME=$(date '+%m_%d_%H_%M').$HOSTFIRST.$1
python -m tools.manage_datagen_jobs --blender_path ../blender/blender --output_folder outputs/$JOBNAME --num_scenes 500 \
    --pipeline_config slurm stereo_video cuda_terrain opengl_gt --wandb_mode online --cleanup big_files --warmup_sec 60000 --config trailer high_quality_terrain --upload

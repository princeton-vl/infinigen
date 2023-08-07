# TEMP. REMOVE BEFORE MERGE. 

HOSTFIRST=$(hostname | tr "." "\n" | head -n 1)
JOBNAME=$(date '+%m_%d_%H_%M').$HOSTFIRST.$1
if [ "$2" = "dev" ]; then
    python -m tools.manage_datagen_jobs --blender_path ../blender/blender --output_folder $3/$JOBNAME --num_scenes 20 \
        --pipeline_config $1 monocular_video_river enable_gpu opengl_gt --wandb_mode online --cleanup none --warmup_sec 12000 --config trailer_river dev reuse_terrain_assets simulated_river
else
    python -m tools.manage_datagen_jobs --blender_path ../blender/blender --output_folder $3/$JOBNAME --num_scenes 50 \
        --pipeline_config $1 monocular_video_river enable_gpu opengl_gt --wandb_mode online --cleanup big_files --warmup_sec 12000 --config trailer_river reuse_terrain_assets simulated_river high_quality_terrain
fi

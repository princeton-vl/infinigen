# TEMP. REMOVE BEFORE MERGE. 

HOSTFIRST=$(hostname | tr "." "\n" | head -n 1)
JOBNAME=$(date '+%m_%d_%H_%M').$HOSTFIRST.$1
python -m infinigen.datagen.manage_jobs --output_folder $3/$JOBNAME --num_scenes $2 \
    --pipeline_config $1 monocular_video enable_gpu opengl_gt --wandb_mode online --cleanup none --warmup_sec 10000 --config trailer high_quality_terrain reuse_terrain_assets use_on_the_fly_fire 

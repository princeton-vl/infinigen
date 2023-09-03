HOSTFIRST=$(hostname | tr "." "\n" | head -n 1)
JOBNAME=$(date '+%m_%d_%H_%M').$HOSTFIRST.$1

python -m tools.manage_datagen_jobs --output_folder outputs/$JOBNAME \
    --num_scenes 1000 --pipeline_config $@ stereo_video cuda_terrain opengl_gt upload \
    --wandb_mode online --cleanup except_crashed --warmup_sec 35000 \
    --config high_quality_terrain

HOSTFIRST=$(hostname | tr "." "\n" | head -n 1)
JOBNAME=$(date '+%m_%d_%H_%M').$HOSTFIRST.$1

python -m infinigen.datagen.manage_jobs --output_folder outputs/$JOBNAME \
    --num_scenes 1000 --pipeline_config $@ stereo_video cuda_terrain opengl_gt upload \
    --wandb_mode online --cleanup except_crashed --warmup_sec 25000 \
    --config high_quality_terrain \
    --overrides compose_scene.generate_resolution=[1280,720]

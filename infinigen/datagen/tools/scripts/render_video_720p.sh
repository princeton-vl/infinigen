HOSTFIRST=$(hostname | tr "." "\n" | head -n 1)
JOBNAME=$(date '+%m_%d_%H_%M').$HOSTFIRST.$1

python -m infinigen.datagen.manage_jobs --output_folder outputs/$JOBNAME \
    --num_scenes 100 --pipeline_config $@ stereo_video cuda_terrain opengl_gt \
    --wandb_mode online --cleanup big_files --upload --warmup_sec 40000 \
    --config high_quality_terrain \
    --overrides compose_scene.generate_resolution=[1280,720] \
    --pipeline_overrides sample_scene_spec.config_sample_mode=\'roundrobin\'

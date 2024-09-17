python -m infinigen.datagen.manage_jobs --output_folder /n/fs/scratch/dy2617/system_test/dining2 --num_scenes 1 --specific_seed 0 --configs singleroom fast_solve \
--pipeline_configs slurm monocular blender_gt.gin indoor_background_configs.gin \
--pipeline_overrides get_cmd.driver_script=infinigen_examples.generate_indoors \
slurm_submit_cmd.slurm_account=pvl coarse/slurm_submit_cmd.slurm_account=None \
--overrides compose_indoors.terrain_enabled=True restrict_solving.restrict_parent_rooms=\[\"DiningRoom\"\] &

# screen python -m infinigen.datagen.manage_jobs --output_folder /n/fs/pvl-renders/system_test --num_scenes 1 --specific_seed 0 --configs singleroom \
# --pipeline_configs slurm monocular blender_gt.gin indoor_background_configs.gin \ 
# --pipeline_overrides get_cmd.driver_script=infinigen_examples.generate_indoors \
# slurm_submit_cmd.slurm_account=pvl coarse/slurm_submit_cmd.slurm_account=None \
# --overrides compose_indoors.terrain_enabled=True restrict_solving.restrict_parent_rooms=\[\"Bathroom\"\]

# screen python -m infinigen.datagen.manage_jobs --output_folder /n/fs/pvl-renders/system_test --num_scenes 1 --specific_seed 0 --configs singleroom \
# --pipeline_configs slurm monocular blender_gt.gin indoor_background_configs.gin \ 
# --pipeline_overrides get_cmd.driver_script=infinigen_examples.generate_indoors \
# slurm_submit_cmd.slurm_account=pvl coarse/slurm_submit_cmd.slurm_account=None \
# --overrides compose_indoors.terrain_enabled=True restrict_solving.restrict_parent_rooms=\[\"Bedroom\"\]

# screen python -m infinigen.datagen.manage_jobs --output_folder /n/fs/pvl-renders/system_test --num_scenes 1 --specific_seed 0 --configs singleroom \
# --pipeline_configs slurm monocular blender_gt.gin indoor_background_configs.gin \ 
# --pipeline_overrides get_cmd.driver_script=infinigen_examples.generate_indoors \
# slurm_submit_cmd.slurm_account=pvl coarse/slurm_submit_cmd.slurm_account=None \
# --overrides compose_indoors.terrain_enabled=True restrict_solving.restrict_parent_rooms=\[\"Kitchen\"\]

# screen python -m infinigen.datagen.manage_jobs --output_folder /n/fs/pvl-renders/system_test --num_scenes 1 --specific_seed 0 --configs singleroom \
# --pipeline_configs slurm monocular blender_gt.gin indoor_background_configs.gin \ 
# --pipeline_overrides get_cmd.driver_script=infinigen_examples.generate_indoors \
# slurm_submit_cmd.slurm_account=pvl coarse/slurm_submit_cmd.slurm_account=None \
# --overrides compose_indoors.terrain_enabled=True restrict_solving.restrict_parent_rooms=\[\"LivingRoom\"\]

# screen python -m infinigen.datagen.manage_jobs --output_folder /n/fs/pvl-renders/system_test/desert --num_scenes 1 --specific_seed 0 \
# --configs desert.gin high_quality_terrain  -pipeline_configs slurm.gin monocular.gin blender_gt.gin cuda_terrain \
# --pipeline_overrides slurm_submit_cmd.slurm_account=pvl coarse/slurm_submit_cmd.slurm_account=None 

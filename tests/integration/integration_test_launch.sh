JOBTAG="ifg-int"
INFINIGEN_VERSION=$(python -c "import infinigen; print(infinigen.__version__)")
COMMIT_HASH=$(git rev-parse HEAD | cut -c 1-6) 
VERSION_STRING="v${INFINIGEN_VERSION}_${COMMIT_HASH}"
OUTPUT_PATH=/n/fs/scratch/$USER/integration_test/$VERSION_STRING

for indoor_type in DiningRoom Bathroom Bedroom Kitchen LivingRoom; do
    python -m infinigen.datagen.manage_jobs --output_folder $OUTPUT_PATH/${JOBTAG}_indoor_$indoor_type \
    --num_scenes 1 --specific_seed 0 --configs singleroom fast_solve.gin --pipeline_configs slurm monocular blender_gt.gin indoor_background_configs.gin --cleanup big_files \
    --pipeline_overrides get_cmd.driver_script=infinigen_examples.generate_indoors --overrides compose_indoors.terrain_enabled=True \
    restrict_solving.restrict_parent_rooms=\[\"$indoor_type\"\] &
done

for nature_type in arctic canyon cave coast coral_reef desert forest kelp_forest mountain plain river snowy_mountain under_water; do
    python -m infinigen.datagen.manage_jobs --output_folder $OUTPUT_PATH/${JOBTAG}_nature_$nature_type \
    --num_scenes 1 --specific_seed 0 --pipeline_configs slurm monocular blender_gt.gin --cleanup big_files \
     --configs $nature_type.gin dev.gin &
done

ASSET_ARGS="--slurm --n_workers 30 -n 3"
for asset_type in indoor_objects nature_objects indoor_materials nature_materials; do
    python -m infinigen_examples.generate_individual_assets $ASSET_ARGS --output_folder $OUTPUT_PATH/${JOBTAG}_$asset_type -f tests/assets/list_$asset_type.txt &
done

wait